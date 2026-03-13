# ============================================================
# WHY2 β REPARAMETERIZATION CHECK
#
# THE QUESTION:
#   Is β just a reparameterization of a known spectral quantity?
#   Or does it capture something that effective rank, spectral
#   entropy, and singular value concentration miss?
#
# METHOD — frozen network test (ChatGPT suggestion, adapted):
#   1. Train N models per task to full convergence (post-grok)
#   2. Freeze — no gradients, single β measurement per model
#   3. Also compute: effective_rank, spectral_entropy,
#      top_sv_fraction, participation_ratio
#   4. Check pairwise correlations: β vs each spectral probe
#   5. Plot distribution of β across models — cluster or scatter?
#
# KEY DIFFERENCE FROM CHATGPT'S SUGGESTION:
#   We only measure frozen β on POST-GROK models (acc_te ≥ 0.99).
#   Measuring β on memorized models is meaningless — β is only
#   defined at the fixed point, not during the transient.
#
# USES EXISTING β* NUMBERS where available, then trains fresh
#   models per task × hyperparameter grid for the frozen test.
#
# TASKS: multiply, add, subtract (known β* values)
#   subtract has two basins — expect two clusters (Outcome B)
#   multiply and add should show tight clusters (Outcome A)
#
# HYP GRID per task (varied to test β invariance):
#   seeds:  [42, 123, 7, 17, 71, 31, 99, 13]  (8 seeds)
#   WD:     [0.5, 1.0, 2.0]  (3 values)
#   → up to 24 runs per task, 72 total
#   → only post-grok models enter the analysis
#
# Expected runtime: ~45-55 GPU compute units (T4)
#   (many will grok early and stop — actual time may be less)
# ============================================================

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, warnings, json, matplotlib
from datetime import datetime
from itertools import product
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
PREFIX = f"why2_beta_corr_{TS}"

print(f"WHY2 β REPARAMETERIZATION CHECK  |  device: {DEVICE}  |  {datetime.now():%Y-%m-%d %H:%M}")
print(f"Output prefix: {PREFIX}")
print(f"Question: Is β a reparameterization of effective_rank / spectral_entropy / sv_concentration?")
print("="*65)

# ── Config ────────────────────────────────────────────────────
PRIME        = 97
D_MODEL      = 128
N_HEADS      = 4
D_FF         = 512
LR           = 1e-3
TRAIN_FRAC   = 0.3
PROBE_EVERY  = 50          # less frequent — we care about endpoint, not trajectory
GROK_THRESH  = 0.99
MAX_EPOCHS   = 25000       # enough for all three tasks to grok across WD values
POST_GROK_SETTLE = 3000    # epochs to run after grokking before freezing
                            # gives β time to approach β* before measurement
TASKS        = ["multiply", "add", "subtract"]
SEEDS        = [42, 123, 7, 17, 71, 31, 99, 13]
WD_VALUES    = [0.5, 1.0, 2.0]

# Known β* from prior experiments (calibration reference)
KNOWN_BETA_STAR = {
    "multiply": {"mean": 0.718, "std": 0.046},
    "add":      {"mean": 3.44,  "std": 0.59},
    "subtract_A": {"mean": 4.78, "std": 0.22},
    "subtract_B": {"mean": 15.24, "std": 1.46},
}

print(f"\nHyperparameter grid: {len(SEEDS)} seeds × {len(WD_VALUES)} WD values = "
      f"{len(SEEDS)*len(WD_VALUES)} runs/task × {len(TASKS)} tasks = "
      f"{len(SEEDS)*len(WD_VALUES)*len(TASKS)} total runs")
print(f"Only post-grok models enter analysis (acc_te ≥ {GROK_THRESH})")
print(f"Settle time post-grok before freeze: {POST_GROK_SETTLE} epochs\n")

# ── Data ──────────────────────────────────────────────────────
def make_data(task, seed):
    rng   = torch.Generator(); rng.manual_seed(seed)
    all_a = torch.arange(PRIME).repeat(PRIME)
    all_b = torch.arange(PRIME).repeat_interleave(PRIME)
    if   task == "add":      all_y = (all_a + all_b) % PRIME
    elif task == "subtract": all_y = (all_a - all_b) % PRIME
    elif task == "multiply": all_y = (all_a * all_b) % PRIME
    else: raise ValueError(f"Unknown task: {task}")
    idx  = torch.randperm(len(all_a), generator=rng)
    n_tr = int(len(all_a) * TRAIN_FRAC)
    tr, te = idx[:n_tr], idx[n_tr:]
    return (all_a[tr].to(DEVICE), all_b[tr].to(DEVICE), all_y[tr].to(DEVICE),
            all_a[te].to(DEVICE), all_b[te].to(DEVICE), all_y[te].to(DEVICE))

# ── Model ─────────────────────────────────────────────────────
class GrokNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(PRIME + 1, D_MODEL)
        self.pos   = nn.Embedding(3, D_MODEL)
        self.attn  = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.ReLU(),
                                   nn.Linear(D_FF, D_MODEL))
        self.ln1   = nn.LayerNorm(D_MODEL)
        self.ln2   = nn.LayerNorm(D_MODEL)
        self.unemb = nn.Linear(D_MODEL, PRIME, bias=False)

    def forward(self, a, b):
        eq  = torch.full((a.shape[0],), PRIME, dtype=torch.long, device=a.device)
        seq = torch.stack([a, b, eq], dim=1)
        pos = torch.arange(3, device=a.device)
        h   = self.embed(seq) + self.pos(pos)
        h2, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        h   = self.ln1(h + h2)
        h   = self.ln2(h + self.ff(h))
        return self.unemb(h[:, 2, :]), w

    def logits(self, a, b): return self.forward(a, b)[0]

# ── Fourier basis ──────────────────────────────────────────────
_fc = {}
def get_fourier():
    if PRIME not in _fc:
        F = torch.zeros(PRIME, PRIME)
        F[0] = 1.0 / PRIME**0.5
        for k in range(1, PRIME // 2 + 1):
            j = torch.arange(PRIME, dtype=torch.float32)
            F[2*k-1] = torch.cos(2*math.pi*k*j / PRIME) * (2/PRIME)**0.5
            if 2*k < PRIME:
                F[2*k] = torch.sin(2*math.pi*k*j / PRIME) * (2/PRIME)**0.5
        _fc[PRIME] = F.to(DEVICE)
    return _fc[PRIME]

# ── β probe (standard WHY2) ───────────────────────────────────
def probe_beta(model, ap, bp):
    W  = model.embed.weight[:PRIME].detach().float()
    Wf = get_fourier() @ W
    pw = (Wf**2).mean(dim=1)
    pw = pw / (pw.sum() + 1e-12)
    pw_np = pw.cpu().numpy()
    H_n   = float(np.clip(-np.sum(pw_np * np.log(pw_np + 1e-12)) / math.log(PRIME), 0, 1))
    K_out = float(np.sort(pw_np)[-(max(1, PRIME // 10)):].sum())

    model.eval()
    with torch.no_grad():
        _, w = model.forward(ap, bp)
    w_to_a = float(w[:, :, 2, 0].mean().clip(0, 1).item())
    w_to_b = float(w[:, :, 2, 1].mean().clip(0, 1).item())
    K_in   = float(min(w_to_a + w_to_b, 1.0))
    K_in_asym = abs(w_to_a - w_to_b) / (w_to_a + w_to_b + 1e-12)

    beta = None
    if K_out > 0.01 and K_in > 0.01:
        log_Ko = math.log(K_out)
        if abs(log_Ko) > 1e-6:
            beta = (math.log(K_in) - H_n) / log_Ko

    return beta, K_in, K_out, H_n, K_in_asym

# ── Spectral probes (the reparameterization check probes) ─────
def probe_spectral(model):
    """
    Compute spectral quantities from the embedding matrix W ∈ R^{p × d}.
    These are the 'known' quantities ChatGPT suggested checking against β.

    effective_rank:      exp(H_sv) where H_sv = -Σ p_i log p_i on normalized singular values
    spectral_entropy:    Shannon entropy of normalized squared singular values
    top_sv_fraction:     fraction of total variance in top-10% of singular values
    participation_ratio: (Σ σ_i²)² / Σ σ_i⁴  — measures how many modes carry equal weight
    nuclear_norm_ratio:  nuclear norm / (sqrt(min(p,d)) * frobenius norm) — normalized rank proxy
    fourier_k_out:       our K_out — top-10% Fourier mode power fraction (already in β formula)
    fourier_h_n:         our H_n — Fourier spectral entropy (already in β formula)
    """
    W = model.embed.weight[:PRIME].detach().float().cpu()  # shape: (p, d_model)

    # Singular value decomposition
    try:
        sv = torch.linalg.svdvals(W).numpy()  # shape: (min(p, d_model),)
    except Exception:
        sv = np.linalg.svd(W.numpy(), compute_uv=False)

    sv = sv[sv > 1e-10]  # drop near-zero singular values

    # Normalized singular values (L1)
    sv_norm = sv / (sv.sum() + 1e-12)

    # Spectral entropy of singular values (not squared)
    h_sv = -np.sum(sv_norm * np.log(sv_norm + 1e-12))
    effective_rank = float(np.exp(h_sv))

    # Spectral entropy of squared singular values (power spectrum)
    sv2      = sv**2
    sv2_norm = sv2 / (sv2.sum() + 1e-12)
    spectral_entropy = float(-np.sum(sv2_norm * np.log(sv2_norm + 1e-12)))
    spectral_entropy_norm = spectral_entropy / math.log(len(sv2))  # normalized to [0,1]

    # Top-10% singular value fraction (analogous to K_out but on weight matrix)
    n_top   = max(1, len(sv) // 10)
    top_sv_fraction = float(sv[:n_top].sum() / (sv.sum() + 1e-12))

    # Top-10% power fraction
    top_sv2_fraction = float(sv2[:n_top].sum() / (sv2.sum() + 1e-12))

    # Participation ratio: (Σσ²)² / Σσ⁴
    participation_ratio = float((sv2.sum())**2 / (np.sum(sv**4) + 1e-12))

    # Nuclear norm ratio (normalized rank proxy)
    frob = float(np.sqrt((sv**2).sum()))
    nuclear = float(sv.sum())
    nuclear_norm_ratio = nuclear / (math.sqrt(min(PRIME, D_MODEL)) * frob + 1e-12)

    # Fourier-basis probes (same as β formula components — included for correlation check)
    Wf   = get_fourier().cpu() @ W
    pw   = (Wf**2).mean(dim=1).numpy()
    pw   = pw / (pw.sum() + 1e-12)
    h_n  = float(np.clip(-np.sum(pw * np.log(pw + 1e-12)) / math.log(PRIME), 0, 1))
    n_top_f = max(1, PRIME // 10)
    k_out = float(np.sort(pw)[-n_top_f:].sum())

    return {
        "effective_rank":        effective_rank,
        "spectral_entropy":      spectral_entropy,
        "spectral_entropy_norm": spectral_entropy_norm,
        "top_sv_fraction":       top_sv_fraction,
        "top_sv2_fraction":      top_sv2_fraction,
        "participation_ratio":   participation_ratio,
        "nuclear_norm_ratio":    nuclear_norm_ratio,
        "fourier_k_out":         k_out,    # same as K_out in β formula
        "fourier_h_n":           h_n,      # same as H_n in β formula
        "n_singular_values":     len(sv),
    }

# ── Checkpoint ────────────────────────────────────────────────
def save_checkpoint(records, suffix=""):
    fname = f"{PREFIX}{suffix}.json"
    def clean(obj):
        if isinstance(obj, dict):  return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [clean(x) for x in obj]
        if isinstance(obj, float): return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 6)
        return obj
    with open(fname, "w") as f:
        json.dump(clean(records), f, indent=2)
    print(f"  [ckpt → {fname}]")
    return fname

# ── Train to grok + settle, then freeze and measure ──────────
def train_and_freeze(task, seed, wd):
    torch.manual_seed(seed); np.random.seed(seed)
    atr, btr, ytr, ate, bte, yte = make_data(task, seed)
    pidx = torch.randperm(len(atr))[:256]
    ap, bp = atr[pidx], btr[pidx]

    model = GrokNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)

    grok_ep       = None
    settle_ep     = None
    label         = f"{task} s={seed} WD={wd}"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        F.cross_entropy(model.logits(atr, btr), ytr).backward()
        opt.step(); opt.zero_grad()

        if epoch % PROBE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                acc_te = (model.logits(ate, bte).argmax(-1) == yte).float().mean().item()

            if grok_ep is None and acc_te >= GROK_THRESH:
                grok_ep  = epoch
                settle_ep = epoch + POST_GROK_SETTLE
                print(f"  [{label}] GROK ep={epoch}  settle until ep={settle_ep}")

            # Once settled — freeze and measure everything
            if grok_ep is not None and epoch >= settle_ep:
                model.eval()
                with torch.no_grad():
                    acc_final = (model.logits(ate, bte).argmax(-1) == yte).float().mean().item()

                beta, K_in, K_out, H_n, K_in_asym = probe_beta(model, ap, bp)
                spectral = probe_spectral(model)

                result = {
                    "task": task, "seed": seed, "wd": wd,
                    "grok_ep": grok_ep, "freeze_ep": epoch,
                    "acc_final": round(acc_final, 4),
                    "beta": beta, "K_in": K_in, "K_out": K_out,
                    "H_n": H_n, "K_in_asym": K_in_asym,
                    **spectral,
                }
                b_str = f"{beta:.4f}" if beta is not None else "N/A"
                print(f"  [{label}] FROZEN ep={epoch} | β={b_str}  K_out={K_out:.3f}  "
                      f"eff_rank={spectral['effective_rank']:.1f}  "
                      f"top_sv={spectral['top_sv_fraction']:.3f}")
                return result

    # Did not grok
    print(f"  [{label}] DID NOT GROK in {MAX_EPOCHS} epochs")
    return {"task": task, "seed": seed, "wd": wd, "grok_ep": None,
            "beta": None, "status": "no_grok"}

# ── Main ──────────────────────────────────────────────────────
all_records = []
grokked     = []

try:
    for task in TASKS:
        print(f"\n{'─'*65}\nTASK: {task.upper()}\n{'─'*65}")
        task_count = 0
        for wd, seed in product(WD_VALUES, SEEDS):
            result = train_and_freeze(task, seed, wd)
            all_records.append(result)
            if result.get("beta") is not None:
                grokked.append(result)
                task_count += 1
            if task_count % 6 == 0:
                save_checkpoint(all_records, f"_partial_{task}")

except KeyboardInterrupt:
    print("\n[Interrupted — saving partial results]")
    save_checkpoint(all_records, "_interrupted")

save_checkpoint(all_records, "_all")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"ANALYSIS  —  {len(grokked)} grokked models out of {len(all_records)} total runs")
print(f"{'='*65}")

if len(grokked) < 3:
    print("Not enough grokked models for correlation analysis. Exiting.")
    exit()

# Organize by task
by_task = {t: [r for r in grokked if r["task"] == t] for t in TASKS}
for task in TASKS:
    n = len(by_task[task])
    betas = [r["beta"] for r in by_task[task] if r["beta"] is not None]
    if betas:
        print(f"\n  {task}: {n} models  β={np.mean(betas):.3f}±{np.std(betas):.3f}  "
              f"range=[{min(betas):.2f}, {max(betas):.2f}]")
        # Sub-group by WD
        for wd in WD_VALUES:
            sub = [r["beta"] for r in by_task[task] if r["wd"] == wd and r["beta"] is not None]
            if sub:
                print(f"    WD={wd}: n={len(sub)}  β={np.mean(sub):.3f}±{np.std(sub):.3f}")

# ── Cluster analysis: is β tight within task? ─────────────────
print(f"\n{'─'*65}")
print("CLUSTER ANALYSIS — does β cluster within task?")
print(f"{'─'*65}")
print("  Prediction:")
print("    multiply → tight cluster near 0.72")
print("    add      → tight cluster near 3.44")
print("    subtract → TWO clusters: ~4.78 (Basin A) and ~15 (Basin B)")
print()

for task in TASKS:
    betas = [r["beta"] for r in by_task[task] if r["beta"] is not None]
    if not betas: continue
    mean, std = np.mean(betas), np.std(betas)
    cv = std / mean  # coefficient of variation — tight = CV < 0.15
    known = KNOWN_BETA_STAR.get(task, {})

    if task == "subtract":
        # Expect bimodal — split at 10
        basin_a = [b for b in betas if b < 10]
        basin_b = [b for b in betas if b >= 10]
        print(f"  subtract: {len(betas)} models")
        if basin_a:
            print(f"    Basin A (β<10):  n={len(basin_a)}  "
                  f"β={np.mean(basin_a):.3f}±{np.std(basin_a):.3f}  "
                  f"(known: 4.78±0.22)")
        if basin_b:
            print(f"    Basin B (β≥10):  n={len(basin_b)}  "
                  f"β={np.mean(basin_b):.3f}±{np.std(basin_b):.3f}  "
                  f"(known: 15.24±1.46)")
        if basin_a and basin_b:
            sep = abs(np.mean(basin_b) - np.mean(basin_a))
            print(f"    Separation: {sep:.2f}  "
                  f"({'✓ bimodal — circuit fingerprint confirmed' if sep > 5 else '✗ not bimodal'})")
    else:
        known_mean = known.get("mean")
        match = abs(mean - known_mean) / known_mean < 0.15 if known_mean else None
        flag  = "✓" if match else "✗" if match is not None else "?"
        print(f"  {task}: n={len(betas)}  β={mean:.3f}±{std:.3f}  CV={cv:.3f}  "
              f"known={known_mean:.3f}  {flag} {'TIGHT CLUSTER' if cv < 0.20 else 'SPREAD'}")

# ── Correlation analysis — the core reparameterization check ──
print(f"\n{'─'*65}")
print("CORRELATION CHECK — β vs spectral probes")
print(f"{'─'*65}")
print("  If |r| > 0.95: β is likely a reparameterization of that probe")
print("  If |r| < 0.70: β captures something that probe misses")
print()

# Collect arrays across all grokked models
betas_all = np.array([r["beta"] for r in grokked if r["beta"] is not None], dtype=float)

probes = [
    ("effective_rank",        "Effective rank of embedding W"),
    ("spectral_entropy",      "Spectral entropy of singular values"),
    ("spectral_entropy_norm", "Normalized spectral entropy"),
    ("top_sv_fraction",       "Top-10% singular value fraction"),
    ("top_sv2_fraction",      "Top-10% sv² fraction (power)"),
    ("participation_ratio",   "Participation ratio (σ² distribn)"),
    ("nuclear_norm_ratio",    "Nuclear norm ratio (rank proxy)"),
    ("fourier_k_out",         "Fourier K_out  ← in β formula"),
    ("fourier_h_n",           "Fourier H_n    ← in β formula"),
    ("K_in",                  "K_in (attention routing)"),
    ("K_out",                 "K_out (Fourier concentration)"),
    ("H_n",                   "H_n (Fourier entropy)"),
]

correlations = {}
for probe_key, probe_desc in probes:
    vals = np.array([r.get(probe_key) for r in grokked if r["beta"] is not None], dtype=float)
    mask = np.isfinite(vals) & np.isfinite(betas_all)
    if mask.sum() < 3:
        continue
    r = float(np.corrcoef(betas_all[mask], vals[mask])[0, 1])
    correlations[probe_key] = r
    flag = ("⚠ REPARAMETERIZATION" if abs(r) > 0.95
            else "✓ INDEPENDENT" if abs(r) < 0.70
            else "~ PARTIAL")
    print(f"  r={r:+.3f}  {flag}   {probe_desc}")

# Summary verdict
max_corr_key = max(correlations, key=lambda k: abs(correlations[k]))
max_corr_val = correlations[max_corr_key]
print(f"\n  Highest correlation: {max_corr_key} → r={max_corr_val:.3f}")
if abs(max_corr_val) > 0.95:
    print(f"  VERDICT: β is largely a reparameterization of {max_corr_key}")
    print(f"  → This must be addressed in the paper before submission")
elif abs(max_corr_val) > 0.80:
    print(f"  VERDICT: β is partially related to {max_corr_key} but not equivalent")
    print(f"  → Worth a sentence in Section 5 acknowledging the relationship")
else:
    print(f"  VERDICT: β is NOT a reparameterization of any standard spectral probe")
    print(f"  → β captures something genuinely different from effective rank / entropy")
    print(f"  → This is positive evidence for the framework's novelty")

# ── WD invariance ─────────────────────────────────────────────
print(f"\n{'─'*65}")
print("WD INVARIANCE — does β change with weight decay?")
print(f"{'─'*65}")
print("  If β is a structural invariant, it should be WD-independent")
print()

for task in TASKS:
    wd_means = {}
    for wd in WD_VALUES:
        sub = [r["beta"] for r in by_task[task] if r["wd"] == wd and r["beta"] is not None]
        if sub: wd_means[wd] = (np.mean(sub), np.std(sub), len(sub))
    if len(wd_means) < 2: continue

    vals = list(wd_means.values())
    spread = max(v[0] for v in vals) - min(v[0] for v in vals)
    grand_mean = np.mean([v[0] for v in vals])
    rel_spread = spread / grand_mean * 100
    invar = "✓ WD-invariant" if rel_spread < 20 else "✗ WD-sensitive"

    print(f"  {task}: {invar} (spread={rel_spread:.1f}%)")
    for wd, (m, s, n) in wd_means.items():
        print(f"    WD={wd}: β={m:.3f}±{s:.3f}  (n={n})")

# ── Save correlation summary ───────────────────────────────────
summary = {
    "timestamp": TS,
    "n_total_runs": len(all_records),
    "n_grokked": len(grokked),
    "correlations": correlations,
    "max_correlation": {"probe": max_corr_key, "r": max_corr_val},
    "by_task": {
        task: {
            "n": len(by_task[task]),
            "beta_mean": float(np.mean([r["beta"] for r in by_task[task] if r["beta"] is not None])) if by_task[task] else None,
            "beta_std": float(np.std([r["beta"] for r in by_task[task] if r["beta"] is not None])) if by_task[task] else None,
        }
        for task in TASKS
    }
}
with open(f"{PREFIX}_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  [summary → {PREFIX}_summary.json]")

# ═══════════════════════════════════════════════════════════════
# PLOT  (6 panels)
# ═══════════════════════════════════════════════════════════════

TCOLS = {
    "multiply": "#69ff47",
    "add":      "#00e5ff",
    "subtract": "#ff6b9d",
}

fig = plt.figure(figsize=(26, 18))
fig.patch.set_facecolor("#0a0e1a")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

def sty(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor("#0f1525")
    ax.tick_params(colors="#aabbcc", labelsize=7)
    ax.grid(alpha=0.12, color="#aabbcc")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    ax.set_title(title, color="#e8f4f8", fontsize=8, pad=4)
    if xlabel: ax.set_xlabel(xlabel, color="#aabbcc", fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color="#aabbcc", fontsize=7)

# ── Panel 1: β histogram per task (THE KEY PANEL) ─────────────
ax0 = fig.add_subplot(gs[0, :2])
sty(ax0, "β distribution — frozen post-grok models (key panel)", "β value", "count")
bins = np.linspace(0, 20, 60)
for task in TASKS:
    betas = [r["beta"] for r in by_task[task] if r["beta"] is not None]
    if betas:
        ax0.hist(betas, bins=bins, color=TCOLS[task], alpha=0.6, label=task, edgecolor="none")
# Known β* reference lines
for task, ref in [("multiply", 0.718), ("add", 3.44), ("subtract", 4.78), ("subtract", 15.24)]:
    ax0.axvline(ref, color=TCOLS.get(task, "#fff"), ls="--", lw=1.2, alpha=0.6)
ax0.legend(fontsize=8, facecolor="#0f1525", labelcolor="#aabbcc")
ax0.text(0.98, 0.95, "dashed = known β*", transform=ax0.transAxes,
         color="#aabbcc", fontsize=7, ha="right", va="top")

# ── Panel 2: β vs WD scatter ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 2])
sty(ax1, "β vs WD — invariance check", "weight decay", "β")
for task in TASKS:
    for r in by_task[task]:
        if r["beta"] is None: continue
        jitter = np.random.uniform(-0.03, 0.03)
        ax1.scatter(r["wd"] + jitter, r["beta"], color=TCOLS[task],
                    alpha=0.7, s=20, zorder=3)
ax1.set_xticks(WD_VALUES)
for ref in [0.718, 3.44, 4.78, 15.24]:
    ax1.axhline(ref, color="#aabbcc", ls=":", lw=0.6, alpha=0.4)

# ── Panel 3: β vs effective_rank ──────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
sty(ax2, f"β vs effective_rank  (r={correlations.get('effective_rank', 0):+.3f})",
    "effective rank", "β")
for task in TASKS:
    xs = [r.get("effective_rank") for r in by_task[task] if r["beta"] is not None and r.get("effective_rank")]
    ys = [r["beta"] for r in by_task[task] if r["beta"] is not None and r.get("effective_rank")]
    if xs: ax2.scatter(xs, ys, color=TCOLS[task], alpha=0.7, s=20, label=task)
ax2.legend(fontsize=6, facecolor="#0f1525", labelcolor="#aabbcc")

# ── Panel 4: β vs top_sv_fraction ─────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
sty(ax3, f"β vs top_sv_fraction  (r={correlations.get('top_sv_fraction', 0):+.3f})",
    "top-10% sv fraction", "β")
for task in TASKS:
    xs = [r.get("top_sv_fraction") for r in by_task[task] if r["beta"] is not None and r.get("top_sv_fraction")]
    ys = [r["beta"] for r in by_task[task] if r["beta"] is not None and r.get("top_sv_fraction")]
    if xs: ax3.scatter(xs, ys, color=TCOLS[task], alpha=0.7, s=20, label=task)
ax3.legend(fontsize=6, facecolor="#0f1525", labelcolor="#aabbcc")

# ── Panel 5: β vs K_out (already in formula — expect high r) ──
ax4 = fig.add_subplot(gs[1, 2])
sty(ax4, f"β vs K_out  (r={correlations.get('K_out', 0):+.3f})  [in formula]",
    "K_out", "β")
for task in TASKS:
    xs = [r.get("K_out") for r in by_task[task] if r["beta"] is not None and r.get("K_out")]
    ys = [r["beta"] for r in by_task[task] if r["beta"] is not None and r.get("K_out")]
    if xs: ax4.scatter(xs, ys, color=TCOLS[task], alpha=0.7, s=20, label=task)
ax4.legend(fontsize=6, facecolor="#0f1525", labelcolor="#aabbcc")

# ── Panel 6: correlation bar chart ────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
sty(ax5, "Pearson r: β vs spectral probes — reparameterization check", "probe", "|r|")
probe_labels = [k for k in correlations]
probe_vals   = [abs(correlations[k]) for k in probe_labels]
colors_bar   = ["#ff6b6b" if v > 0.95 else "#ffd93d" if v > 0.80 else "#69ff47"
                for v in probe_vals]
bars = ax5.bar(range(len(probe_labels)), probe_vals, color=colors_bar, alpha=0.85)
ax5.set_xticks(range(len(probe_labels)))
ax5.set_xticklabels([k.replace("_", "\n") for k in probe_labels],
                    color="#aabbcc", fontsize=6, rotation=0)
ax5.axhline(0.95, color="#ff6b6b", ls="--", lw=1.2, alpha=0.7)
ax5.axhline(0.70, color="#ffd93d", ls="--", lw=1.0, alpha=0.7)
ax5.text(len(probe_labels) - 0.5, 0.96, "reparameterization threshold",
         color="#ff6b6b", fontsize=6, ha="right")
ax5.text(len(probe_labels) - 0.5, 0.71, "independence threshold",
         color="#ffd93d", fontsize=6, ha="right")
ax5.set_ylim(0, 1.05)

# ── Panel 7: subtract basin separation ────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
sty(ax6, "subtract β clusters — algorithm fingerprint?", "β", "count")
sub_betas = [r["beta"] for r in by_task["subtract"] if r["beta"] is not None]
if sub_betas:
    bins_s = np.linspace(0, 22, 40)
    basin_a = [b for b in sub_betas if b < 10]
    basin_b = [b for b in sub_betas if b >= 10]
    if basin_a: ax6.hist(basin_a, bins=bins_s, color="#ff6b9d", alpha=0.8,
                          label=f"Basin A (n={len(basin_a)})")
    if basin_b: ax6.hist(basin_b, bins=bins_s, color="#a29bfe", alpha=0.8,
                          label=f"Basin B (n={len(basin_b)})")
    ax6.axvline(4.78,  color="#ff6b9d", ls="--", lw=1.2, alpha=0.7)
    ax6.axvline(15.24, color="#a29bfe", ls="--", lw=1.2, alpha=0.7)
    ax6.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")

verdict_str = ("β IS NOT a reparameterization" if abs(max_corr_val) < 0.80
               else "β PARTIALLY overlaps" if abs(max_corr_val) < 0.95
               else "β MAY be a reparameterization — check carefully")

fig.suptitle(
    f"WHY2 β Reparameterization Check  —  {verdict_str}\n"
    f"green=multiply  cyan=add  pink=subtract  "
    f"red bars=reparameterization zone  yellow=partial  green=independent",
    color="#e8f4f8", fontsize=10, fontweight="bold"
)

plot_fname = f"{PREFIX}.png"
plt.savefig(plot_fname, dpi=130, bbox_inches="tight", facecolor="#0a0e1a")
print(f"\nSaved: {plot_fname}")

print(f"\n{'='*65}")
print(f"DONE  {datetime.now():%H:%M}")
print(f"{'='*65}")
print(f"\nCopy to Drive:")
print(f"  !cp {PREFIX}*.* /content/drive/MyDrive/")
print(f"\nKey results to report:")
print(f"  1. Does subtract show TWO clusters? (basin fingerprint)")
print(f"  2. Highest |r| with spectral probe: {max_corr_key} → {max_corr_val:.3f}")
print(f"  3. Are cluster means consistent with known β* values?")
print(f"     multiply: {KNOWN_BETA_STAR['multiply']['mean']:.3f}")
print(f"     add:      {KNOWN_BETA_STAR['add']['mean']:.3f}")
print(f"     subtract A: {KNOWN_BETA_STAR['subtract_A']['mean']:.3f}")
print(f"     subtract B: {KNOWN_BETA_STAR['subtract_B']['mean']:.3f}")