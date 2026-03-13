# ============================================================
# WHY2 EXPERIMENT 4 v2: β₂ PROBE — K_in_asym DYNAMICS
#
# CHANGES FROM v1 (based on pre-check results):
#   1. WD_VALUES: removed 0.5 — pre-check showed oscillatory regime
#      at WD=0.5, β* never stabilizes, asym signal is noise
#   2. SUBTRACT_SEEDS: reduced from 24 to 16 — 32 runs at WD=[1,2]
#      fits in ~45 min on free T4 (was 72 runs, ~70 min, session risk)
#   3. Basin B framing relaxed: Basin B does NOT appear at WD≥1.0
#      Primary test is now CONTINUOUS β* predictor within Basin A:
#      does β₂ at K_out=0.83 predict the final β* value (4.8–7.6)?
#      r(β₂@threshold, β*) is the key output.
#   4. Early predictor reframed: instead of binary A/B classification,
#      test whether β₂ 500 epochs before threshold predicts β* value
#
# THE QUESTIONS (revised):
#   PRIMARY: Does β₂* (K_in_asym/K_in at convergence) exist as a
#     second fixed point for subtract, separate from β*?
#     - multiply/add: expect β₂* → 0 (attention symmetrizes)
#     - subtract: expect β₂* > 0 (persistent asymmetry)
#     - Is β₂* WD-invariant like β*?
#
#   SECONDARY: Does β₂ at K_out=0.83 crossing predict final β*?
#     r(β₂@threshold, β*_final) — continuous predictor test
#     If r > 0.7: β₂ trajectory carries predictive information
#     about crystallization depth before it's complete
#
#   TERTIARY: What is the β₂ trajectory shape for subtract?
#     Does β₂ rise then fall (collapse to symmetry) or
#     does it rise and plateau (persistent asymmetry fixed point)?
#
# SEEDS:
#   Subtract: 16 seeds × WD=[1.0, 2.0] = 32 runs
#   Controls: multiply, add × 6 seeds × WD=1.0 = 12 runs
#   Total: 44 runs
#
# Expected runtime: ~45-50 min on free T4
# ============================================================

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, json, warnings, matplotlib
from datetime import datetime
from scipy.ndimage import uniform_filter1d
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
PREFIX = f"why2_exp4_beta2_{TS}"

print(f"WHY2 EXP 4 v2: β₂ PROBE — K_in_asym DYNAMICS  |  device: {DEVICE}")
print(f"Primary: Does β₂* exist as second fixed point for subtract?")
print(f"Secondary: Does β₂@K_out=0.83 predict final β* value?")
print("="*65)

PRIME            = 97
D_MODEL          = 128
N_HEADS          = 4
D_FF             = 512
LR               = 1e-3
TRAIN_FRAC       = 0.3
PROBE_EVERY      = 25
MAX_EPOCHS       = 20000
GROK_THRESH      = 0.99
POST_GROK_EPOCHS = 5000   # 5000 sufficient at WD≥1.0 (settled by then)
KOUT_THRESHOLD   = 0.83
BASIN_B_THRESH   = 8.0    # kept for labeling but not expected to trigger

# FIX 1: WD=0.5 removed — oscillatory regime confirmed by pre-check
WD_VALUES     = [1.0, 2.0]

# FIX 2: 16 seeds instead of 24
SUBTRACT_SEEDS = [42, 123, 7, 17, 71, 31, 99, 13,
                  2,  5,  8, 11, 14, 19, 23, 27]
CONTROL_SEEDS  = [42, 123, 7, 17, 31, 99]

# ── Data ──────────────────────────────────────────────────────
def make_data(task, prime, train_frac, seed):
    torch.manual_seed(seed)
    pairs  = [(a, b) for a in range(prime) for b in range(prime)]
    if task == "subtract": labels = [(a-b)%prime for a,b in pairs]
    elif task == "add":    labels = [(a+b)%prime for a,b in pairs]
    else:                  labels = [(a*b)%prime for a,b in pairs]
    idx  = torch.randperm(len(pairs))
    n_tr = int(len(pairs) * train_frac)
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

# ── Probe ─────────────────────────────────────────────────────
def probe(model, prime):
    model.eval()
    pairs = torch.tensor([[a, b, prime] for a in range(prime) for b in range(prime)],
                          device=DEVICE)
    with torch.no_grad():
        _, attn_w = model(pairs)

    w0    = attn_w[:, :, 2, 0]
    w1    = attn_w[:, :, 2, 1]
    K_in  = (w0 + w1).mean().item()
    asym  = (w0 - w1).abs().mean().item()
    beta2 = asym / K_in if K_in > 1e-6 else 0.0

    E = model.embed.weight[:prime].detach().float()
    F_basis = torch.zeros(prime, prime, device=DEVICE)
    for k in range(prime):
        for n in range(prime):
            F_basis[k, n] = math.cos(2 * math.pi * k * n / prime)
    F_basis /= math.sqrt(prime)
    power = torch.stack([(F_basis[k] @ E).pow(2).sum() for k in range(prime)])
    power = power / power.sum()

    top_k = max(1, prime // 10)
    K_out = float(power.topk(top_k).values.sum())
    H_n   = float(-(power * (power + 1e-10).log()).sum() / math.log(prime))

    beta = float('nan')
    if 1e-6 < K_out < 1-1e-6 and K_in > 1e-6:
        beta = (math.log(K_in) - H_n) / math.log(K_out)

    return {"K_in": K_in, "K_out": K_out, "H_n": H_n,
            "K_in_asym": asym, "beta2": beta2, "beta": beta}

# ── Training ──────────────────────────────────────────────────
def run(task, seed, wd):
    torch.manual_seed(seed)
    np.random.seed(seed)

    (X_tr, Y_tr), (X_te, Y_te) = make_data(task, PRIME, TRAIN_FRAC, seed)
    X_tr, Y_tr = X_tr.to(DEVICE), Y_tr.to(DEVICE)
    X_te, Y_te = X_te.to(DEVICE), Y_te.to(DEVICE)

    model = Transformer(PRIME, D_MODEL, N_HEADS, D_FF).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)

    traj    = []
    grok_ep = None

    max_ep = MAX_EPOCHS if task == "subtract" else 12000

    for ep in range(1, max_ep + 1):
        model.train()
        logits, _ = model(X_tr)
        loss = F.cross_entropy(logits, Y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % PROBE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_te)[0].argmax(-1) == Y_te).float().mean().item()
            p = probe(model, PRIME)
            p["ep"] = ep; p["acc"] = acc
            traj.append(p)

            if grok_ep is None and acc >= GROK_THRESH:
                grok_ep = ep
                print(f"    {task} seed={seed} wd={wd} grokked ep={ep}"
                      f"  β={p['beta']:.3f}  β₂={p['beta2']:.4f}")

            if grok_ep is not None and ep >= grok_ep + POST_GROK_EPOCHS:
                break

    # β* and β₂* from last 40 valid probes
    late       = [t for t in traj[-40:] if not math.isnan(t["beta"])]
    beta_star  = np.mean([t["beta"]  for t in late]) if late else float('nan')
    beta2_star = np.mean([t["beta2"] for t in late]) if late else float('nan')
    # FIX 3: basin label kept but not the primary analysis axis
    basin = "B" if beta_star > BASIN_B_THRESH else ("A" if not math.isnan(beta_star) else "?")

    return {
        "task": task, "seed": seed, "wd": wd,
        "grok_ep": grok_ep, "beta_star": beta_star,
        "beta2_star": beta2_star, "basin": basin,
        "trajectory": traj
    }

# ── β₂ dynamics analysis ──────────────────────────────────────
def analyze_beta2(run_data):
    traj  = run_data["trajectory"]
    valid = [t for t in traj if not math.isnan(t["beta"])]
    if len(valid) < 10:
        return None

    eps    = np.array([t["ep"]    for t in valid])
    beta2  = np.array([t["beta2"] for t in valid])
    kout   = np.array([t["K_out"] for t in valid])
    beta   = np.array([t["beta"]  for t in valid])
    kasym  = np.array([t["K_in_asym"] for t in valid])

    beta2_sm = uniform_filter1d(beta2, size=5)
    db2_dt   = np.gradient(beta2_sm, eps)

    # Find K_out=0.83 threshold crossing
    thresh_idx = None
    for i in range(1, len(kout)):
        if kout[i-1] < KOUT_THRESHOLD <= kout[i]:
            thresh_idx = i
            break

    # FIX 4: continuous predictor — β₂ 500 epochs before threshold
    early_pred_beta2 = None
    early_pred_db2   = None
    if thresh_idx is not None:
        pred_ep = eps[thresh_idx] - 500
        candidates = [i for i in range(thresh_idx)
                      if abs(eps[i] - pred_ep) < 100]
        if candidates:
            i_pred = candidates[-1]
            early_pred_beta2 = float(beta2_sm[i_pred])
            early_pred_db2   = float(db2_dt[i_pred])

    # β₂ trajectory shape: does it rise-then-fall or rise-and-plateau?
    # Measure: max β₂ value and when it occurs relative to grok
    max_beta2_idx = int(np.argmax(beta2_sm))
    max_beta2_val = float(beta2_sm[max_beta2_idx])
    max_beta2_ep  = int(eps[max_beta2_idx])
    final_beta2   = float(beta2_sm[-1])
    # collapse ratio: if final << max, asymmetry collapsed
    collapse_ratio = final_beta2 / max_beta2_val if max_beta2_val > 1e-6 else 1.0

    return {
        "task": run_data["task"], "seed": run_data["seed"],
        "wd": run_data["wd"], "basin": run_data["basin"],
        "beta_star": run_data["beta_star"],
        "beta2_star": run_data["beta2_star"],
        "grok_ep": run_data["grok_ep"],
        # Full trajectories for plotting
        "epochs":       eps.tolist(),
        "beta":         beta.tolist(),
        "beta2":        beta2.tolist(),
        "beta2_smooth": beta2_sm.tolist(),
        "kout":         kout.tolist(),
        "kasym":        kasym.tolist(),
        "db2_dt":       db2_dt.tolist(),
        # Threshold crossing
        "thresh_idx":          thresh_idx,
        "db2_at_threshold":    float(db2_dt[thresh_idx])    if thresh_idx is not None else None,
        "beta2_at_threshold":  float(beta2_sm[thresh_idx])  if thresh_idx is not None else None,
        "kout_at_threshold":   float(kout[thresh_idx])      if thresh_idx is not None else None,
        # Early predictor (500 ep before threshold)
        "early_pred_beta2":    early_pred_beta2,
        "early_pred_db2":      early_pred_db2,
        # Trajectory shape
        "max_beta2_val":   max_beta2_val,
        "max_beta2_ep":    max_beta2_ep,
        "final_beta2":     final_beta2,
        "collapse_ratio":  collapse_ratio,
    }

# ── Main ──────────────────────────────────────────────────────
all_results  = []
all_analyses = []

# Controls: multiply and add at WD=1.0
for task in ["multiply", "add"]:
    print(f"\n{'='*55}")
    print(f"CONTROL: {task.upper()}  (expect β₂* → 0)")
    print(f"{'='*55}")
    for seed in CONTROL_SEEDS:
        r = run(task, seed, 1.0)
        all_results.append(r)
        a = analyze_beta2(r)
        if a: all_analyses.append(a)
        if not math.isnan(r.get("beta_star", float('nan'))):
            print(f"  seed={seed}: β*={r['beta_star']:.4f}  β₂*={r['beta2_star']:.5f}"
                  f"  grok={r['grok_ep']}")

# Main: subtract at WD=1.0 and WD=2.0
print(f"\n{'='*55}")
print(f"MAIN: SUBTRACT")
print(f"{len(SUBTRACT_SEEDS)} seeds × {len(WD_VALUES)} WD = {len(SUBTRACT_SEEDS)*len(WD_VALUES)} runs")
print(f"{'='*55}")
done  = 0
total = len(SUBTRACT_SEEDS) * len(WD_VALUES)

for wd in WD_VALUES:
    print(f"\n── WD = {wd} ──────────────────────────────────────")
    for seed in SUBTRACT_SEEDS:
        done += 1
        r = run("subtract", seed, wd)
        all_results.append(r)
        a = analyze_beta2(r)
        if a: all_analyses.append(a)
        bs  = r.get("beta_star", float('nan'))
        b2s = r.get("beta2_star", float('nan'))
        thr = a["beta2_at_threshold"] if a and a["beta2_at_threshold"] else float('nan')
        print(f"[{done:2d}/{total}] seed={seed:3d} wd={wd}: "
              f"β*={bs:.4f}  β₂*={b2s:.5f}  β₂@thr={thr:.4f}"
              + (f"  grok={r['grok_ep']}" if r.get("grok_ep") else "  NO GROK"))

# ── Save ──────────────────────────────────────────────────────
outfile = f"{PREFIX}_results.json"
# Strip large trajectory arrays from JSON save (keep for analysis, not needed in file)
strip_keys = {"epochs","beta","beta2","kout","kasym","db2_dt","beta2_smooth"}
save_data = {
    "results": [{"task":r["task"],"seed":r["seed"],"wd":r["wd"],
                  "grok_ep":r["grok_ep"],"beta_star":r["beta_star"],
                  "beta2_star":r["beta2_star"],"basin":r["basin"]} for r in all_results],
    "analyses": [{k:v for k,v in a.items() if k not in strip_keys}
                 for a in all_analyses]
}
with open(outfile, "w") as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {outfile}")

# ── Analysis ──────────────────────────────────────────────────
grokked = [r for r in all_results if r.get("grok_ep")
           and not math.isnan(r.get("beta_star", float('nan')))]

print("\n" + "="*65)
print("β₂* FIXED POINT ANALYSIS")
print("="*65)

for task in ["multiply", "add"]:
    vals = [r["beta2_star"] for r in grokked if r["task"]==task]
    print(f"\n{task} (control, n={len(vals)}):")
    if vals:
        print(f"  β₂* = {np.mean(vals):.5f} ± {np.std(vals):.5f}")
        verdict = "→ TRIVIAL ZERO (symmetry restored)" if np.mean(vals) < 0.05 else "→ NONZERO (unexpected)"
        print(f"  {verdict}")

print(f"\nsubtract:")
for wd in WD_VALUES:
    vals = [r["beta2_star"] for r in grokked if r["task"]=="subtract" and r["wd"]==wd]
    if vals:
        print(f"  WD={wd}: n={len(vals)}  β₂* = {np.mean(vals):.5f} ± {np.std(vals):.5f}")

# WD-invariance of β₂*
sub_b2 = [r for r in grokked if r["task"]=="subtract"]
b2_wd1 = [r["beta2_star"] for r in sub_b2 if r["wd"]==1.0]
b2_wd2 = [r["beta2_star"] for r in sub_b2 if r["wd"]==2.0]
if b2_wd1 and b2_wd2:
    diff = abs(np.mean(b2_wd1) - np.mean(b2_wd2))
    pooled_std = np.std(b2_wd1 + b2_wd2)
    print(f"\n  WD-invariance test:")
    print(f"  WD=1.0 β₂* = {np.mean(b2_wd1):.5f} ± {np.std(b2_wd1):.5f}")
    print(f"  WD=2.0 β₂* = {np.mean(b2_wd2):.5f} ± {np.std(b2_wd2):.5f}")
    print(f"  |diff| = {diff:.5f}  pooled σ = {pooled_std:.5f}")
    if diff < pooled_std:
        print(f"  VERDICT: β₂* is WD-INVARIANT → genuine second fixed point")
    else:
        print(f"  VERDICT: β₂* depends on WD → NOT a clean fixed point")

print("\n" + "="*65)
print("FIX 4: CONTINUOUS β* PREDICTOR TEST")
print("Does β₂ at K_out=0.83 predict final β*?")
print("="*65)

sub_analyses = [a for a in all_analyses
                if a["task"]=="subtract" and a.get("beta2_at_threshold") is not None]

if sub_analyses:
    beta2_at_thr = [a["beta2_at_threshold"] for a in sub_analyses]
    beta_star_vals = [a["beta_star"] for a in sub_analyses]

    r_val = np.corrcoef(beta2_at_thr, beta_star_vals)[0,1]
    print(f"\nn={len(sub_analyses)} subtract runs with K_out=0.83 crossing")
    print(f"r(β₂@threshold, β*_final) = {r_val:.4f}")
    if abs(r_val) > 0.7:
        print(f"STRONG: β₂ at crossing carries predictive info about final β*")
    elif abs(r_val) > 0.4:
        print(f"MODERATE: β₂ at crossing weakly predictive of β*")
    else:
        print(f"WEAK: β₂ at threshold does not predict β*")

    # Also test early predictor (500 ep before threshold)
    early = [a for a in sub_analyses if a.get("early_pred_beta2") is not None]
    if early:
        ep_b2   = [a["early_pred_beta2"] for a in early]
        ep_bstar= [a["beta_star"] for a in early]
        r_early = np.corrcoef(ep_b2, ep_bstar)[0,1]
        print(f"\nEarly predictor (500ep before threshold): n={len(early)}")
        print(f"r(β₂@500ep_before, β*_final) = {r_early:.4f}")
        if abs(r_early) > abs(r_val):
            print(f"EARLY IS BETTER: pre-threshold β₂ more predictive than at-threshold")
        else:
            print(f"AT-THRESHOLD IS BETTER: crossing value more predictive than pre-crossing")

    # Per-WD breakdown
    for wd in WD_VALUES:
        sub_wd = [a for a in sub_analyses if a["wd"]==wd]
        if len(sub_wd) >= 3:
            b2t = [a["beta2_at_threshold"] for a in sub_wd]
            bst = [a["beta_star"] for a in sub_wd]
            r_wd = np.corrcoef(b2t, bst)[0,1]
            print(f"\n  WD={wd}: n={len(sub_wd)}  r={r_wd:.4f}")
            for a in sorted(sub_wd, key=lambda x: x["beta_star"]):
                print(f"    seed={a['seed']:3d}  β*={a['beta_star']:.4f}"
                      f"  β₂@thr={a['beta2_at_threshold']:.4f}"
                      f"  β₂*={a['beta2_star']:.5f}")

print("\n" + "="*65)
print("TRAJECTORY SHAPE: rise-then-fall vs rise-then-plateau")
print("="*65)

for task in ["multiply", "add", "subtract"]:
    sub = [a for a in all_analyses if a["task"]==task]
    if not sub: continue
    collapses = [a["collapse_ratio"] for a in sub]
    max_b2s   = [a["max_beta2_val"] for a in sub]
    print(f"\n{task}: n={len(sub)}")
    print(f"  max β₂ (peak asym): {np.mean(max_b2s):.4f} ± {np.std(max_b2s):.4f}")
    print(f"  collapse ratio (final/max): {np.mean(collapses):.4f} ± {np.std(collapses):.4f}")
    if np.mean(collapses) < 0.5:
        print(f"  SHAPE: rise-then-COLLAPSE (symmetry mostly restored)")
    elif np.mean(collapses) > 0.8:
        print(f"  SHAPE: rise-then-PLATEAU (persistent asymmetry = β₂* > 0)")
    else:
        print(f"  SHAPE: partial collapse (intermediate)")

# ── Plot ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: β₂ trajectories for subtract (colored by WD)
ax1 = fig.add_subplot(gs[0, :2])
colors_wd = {1.0: "#2196F3", 2.0: "#F44336"}
sub_traj = [a for a in all_analyses if a["task"]=="subtract" and "beta2" in a]
for a in sub_traj:
    col = colors_wd.get(a["wd"], "gray")
    ax1.plot(a["epochs"], a["beta2"], color=col, alpha=0.4, lw=1.2)
# Overlay mean trajectory per WD
for wd in WD_VALUES:
    wd_trajs = [a for a in sub_traj if a["wd"]==wd]
    if wd_trajs:
        # interpolate to common epoch grid
        max_len = max(len(a["epochs"]) for a in wd_trajs)
        col = colors_wd[wd]
        ax1.axvline(0, color="gray", lw=0.5)
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=colors_wd[wd], label=f"WD={wd}") for wd in WD_VALUES],
           fontsize=9)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("β₂ = K_in_asym / K_in")
ax1.set_title("β₂ Trajectories: subtract by WD", fontweight="bold")

# Panel 2: β₂* distribution by task
ax2 = fig.add_subplot(gs[0, 2])
task_groups = [
    ("multiply", [r for r in grokked if r["task"]=="multiply"], "#9E9E9E"),
    ("add",      [r for r in grokked if r["task"]=="add"],      "#4CAF50"),
    ("sub WD=1", [r for r in grokked if r["task"]=="subtract" and r["wd"]==1.0], "#2196F3"),
    ("sub WD=2", [r for r in grokked if r["task"]=="subtract" and r["wd"]==2.0], "#F44336"),
]
for i, (label, runs, col) in enumerate(task_groups):
    vals = [r["beta2_star"] for r in runs]
    if vals:
        ax2.scatter([i]*len(vals), vals, color=col, s=50, alpha=0.7, zorder=5)
        ax2.plot([i-0.2, i+0.2], [np.mean(vals)]*2, color=col, lw=3)
ax2.set_xticks(range(len(task_groups)))
ax2.set_xticklabels([g[0] for g in task_groups], rotation=15, fontsize=8)
ax2.set_ylabel("β₂*")
ax2.set_title("β₂* by Task / WD", fontweight="bold")
ax2.axhline(0, color="black", lw=1, ls="--", alpha=0.4)

# Panel 3: β₂@threshold vs β*_final scatter (continuous predictor)
ax3 = fig.add_subplot(gs[1, :2])
for wd in WD_VALUES:
    sub_wd = [a for a in sub_analyses if a["wd"]==wd]
    if sub_wd:
        x = [a["beta2_at_threshold"] for a in sub_wd]
        y = [a["beta_star"] for a in sub_wd]
        col = colors_wd[wd]
        ax3.scatter(x, y, color=col, s=80, alpha=0.8, label=f"WD={wd}", zorder=5)
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            xl = np.linspace(min(x)-0.02, max(x)+0.02, 50)
            ax3.plot(xl, np.polyval(z, xl), color=col, lw=1.5, ls="--", alpha=0.6)
ax3.set_xlabel("β₂ at K_out=0.83 threshold")
ax3.set_ylabel("Final β*")
ax3.set_title(f"Continuous Predictor: β₂@threshold → β*\n"
              f"r={r_val:.4f}" if sub_analyses else "β₂@threshold → β*",
              fontweight="bold")
ax3.legend(fontsize=9)

# Panel 4: Trajectory shape — collapse ratio
ax4 = fig.add_subplot(gs[1, 2])
shape_groups = [
    ("multiply", [a for a in all_analyses if a["task"]=="multiply"], "#9E9E9E"),
    ("add",      [a for a in all_analyses if a["task"]=="add"],      "#4CAF50"),
    ("subtract", [a for a in all_analyses if a["task"]=="subtract"], "#2196F3"),
]
for i, (label, runs, col) in enumerate(shape_groups):
    vals = [a["collapse_ratio"] for a in runs]
    if vals:
        ax4.scatter([i]*len(vals), vals, color=col, s=50, alpha=0.7)
        ax4.plot([i-0.2, i+0.2], [np.mean(vals)]*2, color=col, lw=3)
ax4.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.5, label="collapse threshold")
ax4.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.3)
ax4.set_xticks([0,1,2]); ax4.set_xticklabels(["multiply","add","subtract"])
ax4.set_ylabel("Collapse ratio (final β₂ / peak β₂)")
ax4.set_title("β₂ Trajectory Shape\n(1=plateau, 0=full collapse)", fontweight="bold")

# Panel 5: β and β₂ co-evolution, two representative subtract seeds
ax5 = fig.add_subplot(gs[2, :2])
reps = [a for a in all_analyses if a["task"]=="subtract"
        and a.get("thresh_idx") and a["wd"]==1.0
        and not math.isnan(a["beta_star"])]
reps_sorted = sorted(reps, key=lambda a: a["beta_star"])
# Pick lowest and highest β* seeds
to_plot = []
if reps_sorted: to_plot.append(reps_sorted[0])
if len(reps_sorted) > 1: to_plot.append(reps_sorted[-1])

palette = ["#2196F3", "#F44336"]
for a, col in zip(to_plot, palette):
    eps  = np.array(a["epochs"])
    b2   = np.array(a["beta2_smooth"])
    b    = np.array(a["beta"])
    # Normalize β to same scale as β₂ for overlay
    b_sc = b / b.max() * b2.max() if b.max() > 0 else b
    ax5.plot(eps, b2, color=col, lw=2,
             label=f"β₂ seed={a['seed']} β*={a['beta_star']:.2f}")
    ax5.plot(eps, b_sc, color=col, lw=1.5, ls="--", alpha=0.5,
             label=f"β (scaled) seed={a['seed']}")
    if a["thresh_idx"]:
        ax5.axvline(a["epochs"][a["thresh_idx"]], color=col, lw=1, ls=":", alpha=0.7)
ax5.set_xlabel("Epoch"); ax5.set_ylabel("Value")
ax5.set_title("β and β₂ Co-evolution (low vs high β* seeds, WD=1.0)", fontweight="bold")
ax5.legend(fontsize=7)

# Panel 6: β₂* WD-invariance scatter
ax6 = fig.add_subplot(gs[2, 2])
if b2_wd1 and b2_wd2:
    # Match by seed
    seeds_both = set(r["seed"] for r in sub_b2 if r["wd"]==1.0) & \
                 set(r["seed"] for r in sub_b2 if r["wd"]==2.0)
    x_vals, y_vals = [], []
    for s in sorted(seeds_both):
        v1 = next((r["beta2_star"] for r in sub_b2 if r["seed"]==s and r["wd"]==1.0), None)
        v2 = next((r["beta2_star"] for r in sub_b2 if r["seed"]==s and r["wd"]==2.0), None)
        if v1 and v2:
            x_vals.append(v1); y_vals.append(v2)
    if x_vals:
        ax6.scatter(x_vals, y_vals, color="#673AB7", s=70, alpha=0.8, zorder=5)
        lim = [min(x_vals+y_vals)-0.01, max(x_vals+y_vals)+0.01]
        ax6.plot(lim, lim, "k--", lw=1.5, alpha=0.5, label="y=x (perfect invariance)")
        r_wd = np.corrcoef(x_vals, y_vals)[0,1]
        ax6.set_xlabel("β₂* at WD=1.0"); ax6.set_ylabel("β₂* at WD=2.0")
        ax6.set_title(f"β₂* WD-Invariance\nr={r_wd:.4f}", fontweight="bold")
        ax6.legend(fontsize=8)

fig.suptitle("WHY2 Exp 4 v2: β₂ Probe — K_in_asym as Second Fixed Point",
             fontsize=13, fontweight="bold")
plotfile = f"{PREFIX}_plot.png"
plt.savefig(plotfile, dpi=150, bbox_inches="tight")
print(f"\nSaved plot: {plotfile}")
print("\nDONE — Experiment 4 v2 complete.")