# ============================================================
# WHY2 EXPERIMENT 3: β* vs TRUE FOURIER COMPLEXITY
#
# THE QUESTION:
#   Does β* scale with the number of Fourier modes the exact
#   ground-truth algorithm for each task requires?
#
#   For modular arithmetic on Z_p, the exact solutions are known:
#   - add(a,b) = (a+b) mod p:  solved by 1 Fourier frequency per residue
#     → spectral complexity = p (linear, p distinct frequencies)
#   - multiply(a,b) = (a*b) mod p:  needs multiplicative characters
#     → spectral complexity = p-1 (characters of the cyclic group)
#   - subtract: same as add but antisymmetric
#     → spectral complexity ≈ p (same as add, but directed)
#
#   BUT: different primes p give different numbers of modes.
#   PREDICTION: β*(task, p) should be INVARIANT across p for the
#   same task (already established in v6 for add/multiply at p=97,113).
#
#   MORE INTERESTING PREDICTION (from text_3.txt suggestion):
#   β* ∝ (number of active Fourier modes at β* convergence)
#   We can MEASURE this directly: count modes with power > threshold.
#   If β* = f(mode_count), that would turn β* into a predictable quantity.
#
# EXTENDED TEST: new tasks with known Fourier complexity:
#   - division(a,b) = a * b^{-1} mod p (isomorphic to multiply, AMF=0)
#     PREDICTION: β*(division) ≈ β*(multiply) ≈ 0.69
#   - affine(a,b) = 2a + b mod p (linear combination, intermediate)
#     PREDICTION: β*(affine) between multiply and add
#   - triple(a,b) = (a+b+a*b) mod p (from v6 tier 1a data)
#     PREDICTION: β*(triple) > β*(subtract) (most complex)
#
# METHOD:
#   1. Train each task to convergence (same settings as prior studies)
#   2. At β* convergence: count active Fourier modes (power > 1/p threshold)
#   3. Plot β* vs mode_count — test for linear relationship
#   4. Compare division β* to multiply β* (isomorphism test)
#
# SEEDS: [42, 123, 7, 17, 71, 31]  (6 seeds per task)
# WD: 1.0
# TASKS: multiply, add, subtract, division, affine, triple
#
# Expected runtime: ~30-40 min on free T4
# ============================================================

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, json, warnings, matplotlib
from datetime import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
PREFIX = f"why2_exp3_fourier_{TS}"

print(f"WHY2 EXP 3: β* vs TRUE FOURIER COMPLEXITY  |  device: {DEVICE}")
print(f"Prediction: β* scales with active Fourier mode count at convergence")
print("="*65)

PRIME          = 97
D_MODEL        = 128
N_HEADS        = 4
D_FF           = 512
LR             = 1e-3
TRAIN_FRAC     = 0.3
PROBE_EVERY    = 50
GROK_THRESH    = 0.99
MAX_EPOCHS     = 22000
POST_GROK_SETTLE = 3000
MODE_THRESHOLD = 1.0 / PRIME   # mode "active" if power > 1/p (above uniform baseline)
WD             = 1.0
SEEDS          = [42, 123, 7, 17, 71, 31]

# Task definitions
TASKS = {
    "multiply": lambda a, b, p: (a * b) % p,
    "add":      lambda a, b, p: (a + b) % p,
    "subtract": lambda a, b, p: (a - b) % p,
    "division": lambda a, b, p: (a * pow(int(b), p-2, p)) % p if b != 0 else 0,
    "affine":   lambda a, b, p: (2*a + b) % p,
    "triple":   lambda a, b, p: (a + b + a*b) % p,
}

# Theoretical Fourier complexity (number of nontrivial modes the exact solution needs)
# These are analytical predictions we're testing against β*
THEORETICAL_COMPLEXITY = {
    "multiply": "~(p-1)/2 multiplicative char modes",
    "add":      "~p linear modes",
    "subtract": "~p linear modes (directed)",
    "division": "~(p-1)/2 (isomorphic to multiply)",
    "affine":   "~p (linear combination)",
    "triple":   "~p + cross-freq modes (nonlinear)",
}

# Known β* priors (from prior experiments where available)
PRIOR_BETA = {
    "multiply": (0.685, 0.057),
    "add":      (2.802, 0.803),
    "subtract": (4.213, 1.613),   # Basin A only
    "division": (0.69,  0.05),    # PREDICTION: same as multiply
    "affine":   (None,  None),    # Unknown — this is new
    "triple":   (None,  None),    # Unknown — this is new
}

# ── Data ──────────────────────────────────────────────────────
def make_data(task_fn, prime, train_frac, seed):
    torch.manual_seed(seed)
    pairs  = [(a, b) for a in range(prime) for b in range(prime)]
    labels = [task_fn(a, b, prime) for a, b in pairs]
    idx    = torch.randperm(len(pairs))
    n_tr   = int(len(pairs) * train_frac)
    tr, te = idx[:n_tr], idx[n_tr:]
    def mk(s):
        X = torch.tensor([[pairs[i][0], pairs[i][1], prime] for i in s])
        Y = torch.tensor([labels[i] for i in s])
        return X, Y
    return mk(tr), mk(te)

# ── Model ─────────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(self, prime, d_model, n_heads, d_ff):
        super().__init__()
        self.embed = nn.Embedding(prime + 1, d_model)
        self.pos   = nn.Embedding(3, d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.out   = nn.Linear(d_model, prime)

    def forward(self, x):
        pos = torch.arange(3, device=x.device).unsqueeze(0)
        h   = self.embed(x) + self.pos(pos)
        h2, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        h = self.norm1(h + h2)
        h = self.norm2(h + self.ff2(F.relu(self.ff1(h))))
        return self.out(h[:, 2, :]), w

# ── WHY2 + mode count probe ───────────────────────────────────
def probe_full(model, prime, mode_threshold):
    model.eval()
    pairs = torch.tensor([[a, b, prime] for a in range(prime) for b in range(prime)],
                          device=DEVICE)
    with torch.no_grad():
        _, attn_w = model(pairs)

    w_ops = attn_w[:, :, 2, 0] + attn_w[:, :, 2, 1]
    K_in  = w_ops.mean().item()
    asym  = (attn_w[:, :, 2, 0] - attn_w[:, :, 2, 1]).abs().mean().item()

    E = model.embed.weight[:prime].detach().float()
    F_basis = torch.zeros(prime, prime, device=DEVICE)
    for k in range(prime):
        for n in range(prime):
            F_basis[k, n] = math.cos(2 * math.pi * k * n / prime)
    F_basis /= math.sqrt(prime)

    power = torch.stack([(F_basis[k] @ E).pow(2).sum() for k in range(prime)])
    power = power / power.sum()
    power_np = power.cpu().numpy()

    top_k = max(1, prime // 10)
    K_out  = float(power.topk(top_k).values.sum())
    H_n    = float(-(power * (power + 1e-10).log()).sum() / math.log(prime))

    # Count active modes
    active_modes = int((power_np > mode_threshold).sum())
    # Mode entropy (another complexity measure)
    mode_entropy = float(-(power * (power + 1e-10).log()).sum())

    beta = float('nan')
    if 1e-6 < K_out < 1-1e-6 and K_in > 1e-6:
        beta = (math.log(K_in) - H_n) / math.log(K_out)

    return {
        "K_in": K_in, "K_out": K_out, "H_n": H_n, "K_in_asym": asym,
        "beta": beta, "active_modes": active_modes,
        "mode_entropy": mode_entropy,
        "power_spectrum": power_np.tolist()
    }

# ── Training loop ─────────────────────────────────────────────
def train_task(task_name, seed, wd=WD):
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_fn = TASKS[task_name]
    (X_tr, Y_tr), (X_te, Y_te) = make_data(task_fn, PRIME, TRAIN_FRAC, seed)
    X_tr, Y_tr = X_tr.to(DEVICE), Y_tr.to(DEVICE)
    X_te, Y_te = X_te.to(DEVICE), Y_te.to(DEVICE)

    model = Transformer(PRIME, D_MODEL, N_HEADS, D_FF).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)

    grok_ep = None

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        logits, _ = model(X_tr)
        loss = F.cross_entropy(logits, Y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % PROBE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_te)[0].argmax(-1) == Y_te).float().mean().item()

            if grok_ep is None and acc >= GROK_THRESH:
                grok_ep = ep
                print(f"    {task_name} seed={seed} grokked ep={ep}")

            if grok_ep is not None and ep >= grok_ep + POST_GROK_SETTLE:
                p = probe_full(model, PRIME, MODE_THRESHOLD)
                p.update({"task": task_name, "seed": seed, "wd": wd,
                           "grok_ep": grok_ep, "freeze_ep": ep, "acc_final": acc})
                return p

        if ep % 5000 == 0 and grok_ep is None:
            model.eval()
            with torch.no_grad():
                acc = (model(X_te)[0].argmax(-1) == Y_te).float().mean().item()
            print(f"    {task_name} seed={seed} ep={ep} acc={acc:.3f} (no grok)")

    return {"task": task_name, "seed": seed, "wd": wd,
            "grok_ep": None, "status": "no_grok"}

# ── Main ──────────────────────────────────────────────────────
results = []
task_list = list(TASKS.keys())
total = len(task_list) * len(SEEDS)
done = 0

for task_name in task_list:
    print(f"\n{'='*55}")
    print(f"TASK: {task_name.upper()}  |  Theory: {THEORETICAL_COMPLEXITY[task_name]}")
    if PRIOR_BETA[task_name][0]:
        print(f"  Prior β*: {PRIOR_BETA[task_name][0]:.3f} ± {PRIOR_BETA[task_name][1]:.3f}")
    else:
        print(f"  Prior β*: UNKNOWN (new task)")
    print(f"{'='*55}")

    for seed in SEEDS:
        done += 1
        print(f"\n[{done}/{total}] {task_name} seed={seed}")
        r = train_task(task_name, seed)
        results.append(r)
        if r.get("beta"):
            print(f"  β*={r['beta']:.4f}  K_out={r['K_out']:.4f}"
                  f"  active_modes={r['active_modes']}")

# Save
outfile = f"{PREFIX}_results.json"
# Strip power spectra from saved JSON (large)
save_results = [{k:v for k,v in r.items() if k != "power_spectrum"} for r in results]
with open(outfile, "w") as f:
    json.dump(save_results, f, indent=2)
print(f"\nSaved: {outfile}")

# ── Analysis ──────────────────────────────────────────────────
print("\n" + "="*65)
print("RESULTS: β* AND ACTIVE FOURIER MODES BY TASK")
print("="*65)

grokked = [r for r in results if r.get("grok_ep") and r.get("beta")
           and not math.isnan(r["beta"])]

task_summary = {}
for task_name in task_list:
    vals  = [r for r in grokked if r["task"] == task_name]
    betas = [r["beta"] for r in vals]
    modes = [r["active_modes"] for r in vals]
    if betas:
        task_summary[task_name] = {
            "beta_mean": np.mean(betas), "beta_std": np.std(betas),
            "modes_mean": np.mean(modes), "modes_std": np.std(modes),
            "n": len(betas)
        }
        prior_m, prior_s = PRIOR_BETA[task_name]
        prior_str = f"prior={prior_m:.3f}±{prior_s:.3f}" if prior_m else "prior=NEW"
        print(f"\n{task_name:10s}: n={len(betas)}")
        print(f"  β*    = {np.mean(betas):.3f} ± {np.std(betas):.3f}  ({prior_str})")
        print(f"  modes = {np.mean(modes):.1f} ± {np.std(modes):.1f}")

# Test β* vs mode count correlation
print("\n" + "-"*55)
print("CORRELATION: β* vs active Fourier modes")
task_beta  = [task_summary[t]["beta_mean"] for t in task_list if t in task_summary]
task_modes = [task_summary[t]["modes_mean"] for t in task_list if t in task_summary]
task_names_used = [t for t in task_list if t in task_summary]

if len(task_beta) >= 3:
    r = np.corrcoef(task_beta, task_modes)[0, 1]
    print(f"  r(β*, mode_count) across tasks: {r:.4f}")
    if r > 0.8:
        print("  STRONG positive correlation — β* scales with Fourier complexity")
    elif r > 0.5:
        print("  MODERATE correlation")
    else:
        print("  WEAK correlation — β* does not scale simply with mode count")

# Division isomorphism test
if "division" in task_summary and "multiply" in task_summary:
    div_b = task_summary["division"]["beta_mean"]
    mul_b = task_summary["multiply"]["beta_mean"]
    mul_s = task_summary["multiply"]["beta_std"]
    diff  = abs(div_b - mul_b)
    print(f"\nDIVISION ISOMORPHISM TEST:")
    print(f"  β*(division) = {div_b:.3f}  β*(multiply) = {mul_b:.3f}")
    print(f"  |diff| = {diff:.3f}  within-task σ(multiply) = {mul_s:.3f}")
    if diff < 2 * mul_s:
        print("  CONFIRMED: division ≈ multiply (isomorphism holds)")
    else:
        print("  NOT CONFIRMED: division ≠ multiply")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: β* by task (ordered)
ax = axes[0]
sorted_tasks = sorted(task_summary.keys(), key=lambda t: task_summary[t]["beta_mean"])
y_pos = range(len(sorted_tasks))
for y, t in zip(y_pos, sorted_tasks):
    m = task_summary[t]["beta_mean"]
    s = task_summary[t]["beta_std"]
    color = "#2196F3" if t in ["multiply", "add", "subtract"] else "#FF9800"
    ax.barh(y, m, xerr=s, color=color, alpha=0.8, height=0.6)
ax.set_yticks(list(y_pos)); ax.set_yticklabels(sorted_tasks)
ax.set_xlabel("β*"); ax.set_title("β* by Task", fontweight="bold")
ax.axvline(0, color="black", lw=0.5)

# Panel 2: β* vs active modes scatter
ax = axes[1]
colors_scatter = {"multiply":"#2196F3","add":"#4CAF50","subtract":"#F44336",
                  "division":"#9C27B0","affine":"#FF9800","triple":"#795548"}
for t, s in task_summary.items():
    ax.scatter(s["modes_mean"], s["beta_mean"],
               s=120, color=colors_scatter.get(t, "gray"),
               label=t, zorder=5)
    ax.errorbar(s["modes_mean"], s["beta_mean"],
                yerr=s["beta_std"], xerr=s["modes_std"],
                fmt='none', color=colors_scatter.get(t, "gray"), alpha=0.5)

if len(task_beta) >= 2:
    z = np.polyfit(task_modes, task_beta, 1)
    x_line = np.linspace(min(task_modes)-2, max(task_modes)+2, 100)
    ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, lw=1.5,
            label=f"fit r={r:.3f}")
ax.set_xlabel("Active Fourier modes (power > 1/p)")
ax.set_ylabel("β*")
ax.set_title("β* vs Fourier Mode Count", fontweight="bold")
ax.legend(fontsize=8)

# Panel 3: Power spectra comparison (one rep per task)
ax = axes[2]
task_colors = {"multiply":"#2196F3","add":"#4CAF50","subtract":"#F44336",
               "division":"#9C27B0","affine":"#FF9800","triple":"#795548"}
for task_name in task_list:
    reps = [r for r in grokked if r["task"] == task_name and "power_spectrum" in r]
    if reps:
        ps = np.array(reps[0]["power_spectrum"])
        sorted_ps = np.sort(ps)[::-1]
        ax.plot(sorted_ps[:30], color=task_colors.get(task_name, "gray"),
                label=task_name, lw=1.5)
ax.axhline(1.0/PRIME, color="black", ls="--", lw=1, alpha=0.5, label="uniform (1/p)")
ax.set_xlabel("Mode rank (sorted by power)")
ax.set_ylabel("Normalized power")
ax.set_title("Top-30 Fourier Mode Power", fontweight="bold")
ax.legend(fontsize=8)

fig.suptitle("WHY2 Exp 3: β* vs True Fourier Complexity", fontsize=13, fontweight="bold")
plt.tight_layout()
plotfile = f"{PREFIX}_plot.png"
plt.savefig(plotfile, dpi=150, bbox_inches="tight")
print(f"\nSaved plot: {plotfile}")
print("\nDONE — Experiment 3 complete.")