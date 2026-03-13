# ============================================================
# WHY2 EXPERIMENT 1: BITNET ARCHITECTURE-INVARIANCE
#
# THE QUESTION:
#   Is β* a property of the TASK or the ARCHITECTURE?
#   If WHY2 is right — β* measures the task's group-theoretic
#   symmetry-breaking cost — then ternary-weight (BitNet-style)
#   models should converge to the same β* as full-precision models.
#
# METHOD:
#   Train matched pairs: full-precision float32 vs ternary {-1,0,+1}
#   Same task, same seed, same WD. Compare β* at convergence.
#   Prediction: |β*_ternary - β*_float| < within-task σ from prior study
#
# TASKS: multiply, add (subtract deferred — too slow for paired test)
# SEEDS: [42, 123, 7, 17, 31, 99]  (6 seeds × 2 tasks × 2 precisions = 24 runs)
# WD: 1.0 (clean regime, not near phase boundary)
#
# BitNet implementation: straight-through estimator for ternary weights
#   Forward:  w_ternary = sign(w) * (|w| > threshold)
#   Backward: pass gradients through as if weights were continuous
#   This is the minimal BitNet-style quantization, no scale factors.
#
# Expected runtime: ~25-35 min on free T4
#   (float runs ~same speed as β_corr study; ternary slightly slower
#    due to quantization overhead but same epoch count)
#
# WHAT CONFIRMS THE PREDICTION:
#   β*_ternary ≈ β*_float (within 1 prior σ) for both tasks
#   → β* is architecture-invariant: it measures the task, not the weights
#
# WHAT WOULD FALSIFY IT:
#   β*_ternary systematically higher or lower than β*_float
#   → β* depends on weight precision, not just task structure
# ============================================================

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, json, warnings, matplotlib
from datetime import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
PREFIX = f"why2_exp1_bitnet_{TS}"

print(f"WHY2 EXP 1: BITNET ARCHITECTURE-INVARIANCE  |  device: {DEVICE}")
print(f"Prediction: β*(ternary) ≈ β*(float) within prior σ")
print("="*65)

# ── Config ────────────────────────────────────────────────────
PRIME          = 97
D_MODEL        = 128
N_HEADS        = 4
D_FF           = 512
LR             = 1e-3
TRAIN_FRAC     = 0.3
PROBE_EVERY    = 50
GROK_THRESH    = 0.99
MAX_EPOCHS     = 20000
POST_GROK_SETTLE = 3000
TASKS          = ["multiply", "add"]
SEEDS          = [42, 123, 7, 17, 31, 99]
WD             = 1.0

# Prior β* from reparameterization study (n=71)
PRIOR = {
    "multiply": {"mean": 0.685, "std": 0.057},
    "add":      {"mean": 2.802, "std": 0.803},
}

# ── Data ──────────────────────────────────────────────────────
def make_data(task, prime, train_frac, seed):
    torch.manual_seed(seed)
    pairs = [(a, b) for a in range(prime) for b in range(prime)]
    if task == "multiply": labels = [(a*b) % prime for a,b in pairs]
    elif task == "add":    labels = [(a+b) % prime for a,b in pairs]
    else:                  labels = [(a-b) % prime for a,b in pairs]
    idx = torch.randperm(len(pairs))
    n_train = int(len(pairs) * train_frac)
    tr, te = idx[:n_train], idx[n_train:]
    def mk(subset):
        X = torch.tensor([[pairs[i][0], pairs[i][1], prime] for i in subset])
        Y = torch.tensor([labels[i] for i in subset])
        return X, Y
    return mk(tr), mk(te)

# ── Ternary quantization (straight-through) ──────────────────
class TernaryLinear(nn.Linear):
    """Linear layer with ternary {-1, 0, +1} weights (STE backward)."""
    def ternarize(self, w):
        threshold = 0.7 * w.abs().mean()
        return torch.where(w.abs() > threshold, w.sign(), torch.zeros_like(w))

    def forward(self, x):
        if self.training or True:  # always quantize in forward
            w_t = self.ternarize(self.weight)
            # STE: gradient flows through as if w_t = weight
            w_t = self.weight + (w_t - self.weight).detach()
        else:
            w_t = self.weight
        return F.linear(x, w_t, self.bias)

# ── Models ────────────────────────────────────────────────────
class FloatTransformer(nn.Module):
    def __init__(self, prime, d_model, n_heads, d_ff):
        super().__init__()
        self.embed   = nn.Embedding(prime + 1, d_model)
        self.pos     = nn.Embedding(3, d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff1     = nn.Linear(d_model, d_ff)
        self.ff2     = nn.Linear(d_ff, d_model)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.out     = nn.Linear(d_model, prime)
        self.prime   = prime
        self.d_model = d_model

    def forward(self, x):
        pos = torch.arange(3, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        h2, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        h = self.norm1(h + h2)
        h = self.norm2(h + self.ff2(F.relu(self.ff1(h))))
        return self.out(h[:, 2, :]), w  # [B, prime], [B, heads, 3, 3]

class TernaryTransformer(nn.Module):
    """Same as FloatTransformer but Linear layers replaced with TernaryLinear."""
    def __init__(self, prime, d_model, n_heads, d_ff):
        super().__init__()
        self.embed   = nn.Embedding(prime + 1, d_model)
        self.pos     = nn.Embedding(3, d_model)
        # Attention projections — use standard MHA but patch weight matrix
        self.attn    = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Replace FF layers with ternary
        self.ff1     = TernaryLinear(d_model, d_ff)
        self.ff2     = TernaryLinear(d_ff, d_model)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.out     = TernaryLinear(d_model, prime)
        self.prime   = prime
        self.d_model = d_model

    def forward(self, x):
        pos = torch.arange(3, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        h2, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        h = self.norm1(h + h2)
        h = self.norm2(h + self.ff2(F.relu(self.ff1(h))))
        return self.out(h[:, 2, :]), w

# ── WHY2 Probes ───────────────────────────────────────────────
def compute_why2(model, prime):
    """Compute K_in, K_out, H_n, K_in_asym, beta from frozen model."""
    model.eval()
    # Build full dataset
    pairs = torch.tensor([[a, b, prime] for a in range(prime) for b in range(prime)],
                          device=DEVICE)
    with torch.no_grad():
        _, attn_w = model(pairs)  # attn_w: [B, heads, 3, 3]

    # K_in: mean attention from '=' (pos 2) to operands (pos 0,1)
    w_to_ops = attn_w[:, :, 2, 0] + attn_w[:, :, 2, 1]  # [B, heads]
    K_in = w_to_ops.mean().item()

    # K_in_asym: |w_{2->0} - w_{2->1}|
    asym = (attn_w[:, :, 2, 0] - attn_w[:, :, 2, 1]).abs().mean().item()

    # K_out: Fourier spectral concentration of embedding matrix
    E = model.embed.weight[:prime].detach().float()  # [prime, d_model]
    F_basis = torch.zeros(prime, prime, device=DEVICE)
    for k in range(prime):
        for n in range(prime):
            F_basis[k, n] = math.cos(2 * math.pi * k * n / prime)
    F_basis = F_basis / math.sqrt(prime)

    power = torch.zeros(prime, device=DEVICE)
    for k in range(prime):
        proj = F_basis[k] @ E  # [d_model]
        power[k] = proj.pow(2).sum()
    power = power / power.sum()

    top_k = max(1, prime // 10)
    K_out = power.topk(top_k).values.sum().item()

    # H_n: normalized spectral entropy
    eps = 1e-10
    H_n = -(power * (power + eps).log()).sum().item() / math.log(prime)

    # beta
    if K_out > 1e-6 and K_out < 1.0 - 1e-6 and K_in > 1e-6:
        beta = (math.log(K_in) - H_n) / math.log(K_out)
    else:
        beta = float('nan')

    return {"K_in": K_in, "K_out": K_out, "H_n": H_n,
            "K_in_asym": asym, "beta": beta}

# ── Training loop ─────────────────────────────────────────────
def train_model(task, seed, precision, wd=WD):
    torch.manual_seed(seed)
    np.random.seed(seed)

    (X_tr, Y_tr), (X_te, Y_te) = make_data(task, PRIME, TRAIN_FRAC, seed)
    X_tr, Y_tr = X_tr.to(DEVICE), Y_tr.to(DEVICE)
    X_te, Y_te = X_te.to(DEVICE), Y_te.to(DEVICE)

    if precision == "float":
        model = FloatTransformer(PRIME, D_MODEL, N_HEADS, D_FF).to(DEVICE)
    else:
        model = TernaryTransformer(PRIME, D_MODEL, N_HEADS, D_FF).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)

    grok_ep = None
    freeze_ep = None
    beta_history = []

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        logits, _ = model(X_tr)
        loss = F.cross_entropy(logits, Y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % PROBE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                acc_te = (model(X_te)[0].argmax(-1) == Y_te).float().mean().item()

            if grok_ep is None and acc_te >= GROK_THRESH:
                grok_ep = ep
                print(f"    [{precision:7s}] seed={seed} grokked ep={ep} acc={acc_te:.4f}")

            if grok_ep is not None and ep >= grok_ep + POST_GROK_SETTLE:
                freeze_ep = ep
                probes = compute_why2(model, PRIME)
                probes.update({"task": task, "seed": seed, "wd": wd,
                                "precision": precision, "grok_ep": grok_ep,
                                "freeze_ep": freeze_ep, "acc_final": acc_te})
                return probes

        if ep % 5000 == 0:
            model.eval()
            with torch.no_grad():
                acc_te = (model(X_te)[0].argmax(-1) == Y_te).float().mean().item()
            print(f"    [{precision:7s}] seed={seed} ep={ep} acc={acc_te:.3f}"
                  + (f" (grokked ep={grok_ep})" if grok_ep else " (not grokked)"))

    # Did not reach settle — return what we have
    probes = compute_why2(model, PRIME)
    probes.update({"task": task, "seed": seed, "wd": wd,
                    "precision": precision, "grok_ep": grok_ep,
                    "freeze_ep": None, "acc_final": 0.0, "status": "no_settle"})
    return probes

# ── Main ──────────────────────────────────────────────────────
results = []
total = len(TASKS) * len(SEEDS) * 2
done = 0

for task in TASKS:
    print(f"\n{'='*55}")
    print(f"TASK: {task.upper()}")
    print(f"{'='*55}")
    for seed in SEEDS:
        for precision in ["float", "ternary"]:
            done += 1
            print(f"\n[{done}/{total}] task={task} seed={seed} precision={precision}")
            r = train_model(task, seed, precision)
            results.append(r)
            print(f"  β={r.get('beta', 'nan'):.4f}  K_out={r.get('K_out', 0):.4f}"
                  f"  grok={r.get('grok_ep', 'none')}")

# ── Save raw results ──────────────────────────────────────────
outfile = f"{PREFIX}_results.json"
with open(outfile, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {outfile}")

# ── Analysis ──────────────────────────────────────────────────
print("\n" + "="*65)
print("RESULTS: β* FLOAT vs TERNARY")
print("="*65)

grokked = [r for r in results if r.get("grok_ep") and r.get("beta") and not math.isnan(r["beta"])]

for task in TASKS:
    print(f"\n{task.upper()}:")
    prior_mean = PRIOR[task]["mean"]
    prior_std  = PRIOR[task]["std"]

    for precision in ["float", "ternary"]:
        vals = [r["beta"] for r in grokked
                if r["task"] == task and r["precision"] == precision]
        if vals:
            m, s = np.mean(vals), np.std(vals)
            z = abs(m - prior_mean) / prior_std
            verdict = "MATCH" if z < 2.0 else "MISMATCH"
            print(f"  {precision:7s}: n={len(vals)}  β*={m:.3f} ± {s:.3f}"
                  f"  prior={prior_mean:.3f}±{prior_std:.3f}"
                  f"  z={z:.2f}  [{verdict}]")

    # Direct float vs ternary comparison
    f_vals = [r["beta"] for r in grokked if r["task"] == task and r["precision"] == "float"]
    t_vals = [r["beta"] for r in grokked if r["task"] == task and r["precision"] == "ternary"]
    # Match by seed
    paired = [(r["beta"] for r in grokked if r["task"]==task and r["precision"]=="float" and r["seed"]==s),
              (r["beta"] for r in grokked if r["task"]==task and r["precision"]=="ternary" and r["seed"]==s)]
    diffs = []
    for seed in SEEDS:
        f = [r["beta"] for r in grokked if r["task"]==task and r["precision"]=="float" and r["seed"]==seed]
        t = [r["beta"] for r in grokked if r["task"]==task and r["precision"]=="ternary" and r["seed"]==seed]
        if f and t:
            diffs.append(t[0] - f[0])
    if diffs:
        print(f"  Paired diff (ternary - float): mean={np.mean(diffs):+.3f}  std={np.std(diffs):.3f}")
        verdict = "ARCHITECTURE-INVARIANT" if abs(np.mean(diffs)) < prior_std else "ARCHITECTURE-DEPENDENT"
        print(f"  VERDICT: {verdict}")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(TASKS), figsize=(5*len(TASKS), 5))
if len(TASKS) == 1: axes = [axes]
colors = {"float": "#2196F3", "ternary": "#FF5722"}

for ax, task in zip(axes, TASKS):
    prior_mean = PRIOR[task]["mean"]
    prior_std  = PRIOR[task]["std"]

    for prec in ["float", "ternary"]:
        vals = [r["beta"] for r in grokked if r["task"]==task and r["precision"]==prec]
        seeds_used = [r["seed"] for r in grokked if r["task"]==task and r["precision"]==prec]
        if vals:
            x = [0.8 if prec == "float" else 1.2] * len(vals)
            ax.scatter(x, vals, color=colors[prec], s=80, zorder=5, label=prec, alpha=0.8)
            ax.plot([x[0]-0.1, x[0]+0.1], [np.mean(vals)]*2, color=colors[prec], lw=3)

    # Prior band
    ax.axhline(prior_mean, color="gray", lw=1.5, ls="--", alpha=0.7, label="prior mean")
    ax.axhspan(prior_mean - prior_std, prior_mean + prior_std,
               color="gray", alpha=0.15, label="prior ±1σ")

    ax.set_title(f"{task}", fontsize=13, fontweight="bold")
    ax.set_ylabel("β*", fontsize=11)
    ax.set_xticks([0.8, 1.2]); ax.set_xticklabels(["float32", "ternary"])
    ax.legend(fontsize=9)
    ax.set_xlim(0.4, 1.6)

fig.suptitle("WHY2 Exp 1: β* Architecture-Invariance\n(float32 vs ternary weights)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plotfile = f"{PREFIX}_plot.png"
plt.savefig(plotfile, dpi=150, bbox_inches="tight")
print(f"\nSaved plot: {plotfile}")
print("\nDONE — Experiment 1 complete.")