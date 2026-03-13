# ============================================================
# WHY2 β VELOCITY — THREE-TASK ORDERING EXPERIMENT
#
# Goal: confirm the predicted β* ordering:
#   multiply β* ≈ 0.82  (lowest, "fluid")
#   add      β* ≈ 4.9   (middle)
#   subtract β* > add   (highest, "most crystallized")
#
# Prediction from relaxation rates at grokking:
#   multiply: 2β²≈0.67,  add: 2β²≈2.08,  subtract: 2β²≈4.15
#   → subtract flows fastest → highest β* fixed point
#
# Also tracks K_in directional asymmetry for subtract:
#   K_in_asym = |K_in_a - K_in_b| / (K_in_a + K_in_b)
#   Hypothesis: subtract asym stays elevated post-grok,
#               add/multiply collapse toward 0
#
# Crash-safe: checkpoints every 1000 epochs
# Dense probing: every 25 epochs
# Tasks: add, subtract, multiply — 3 seeds each = 9 runs
# Max epochs: 18000 (subtract groks later ~9000, needs longer tail)
#
# Expected runtime: ~60-75 min on T4 GPU
# ============================================================

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, warnings, json, matplotlib
from datetime import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"WHY2 β* Ordering  |  device: {DEVICE}  |  {datetime.now():%Y-%m-%d %H:%M}")
print(f"Checkpoint prefix: why2_bvel3_{TS}")
print(f"Prediction: β*(multiply) < β*(add) < β*(subtract)")
print("="*65)

# ── Config ────────────────────────────────────────────────────
PRIME       = 97
D_MODEL     = 128
N_HEADS     = 4
D_FF        = 512
LR          = 1e-3
WD          = 1.0
TRAIN_FRAC  = 0.3
PROBE_EVERY = 25
GROK_THRESH = 0.99
MAX_EPOCHS  = 18000   # subtract needs ~9000 to grok + 9000 post
CKPT_EVERY  = 1000
TASKS       = ["add", "subtract", "multiply"]
SEEDS       = [42, 123, 7]
BETA_WINDOW = 10

# ── Data ──────────────────────────────────────────────────────
def make_data(task, seed):
    rng = torch.Generator(); rng.manual_seed(seed)
    all_a = torch.arange(PRIME).repeat(PRIME)
    all_b = torch.arange(PRIME).repeat_interleave(PRIME)
    if task == "add":
        all_y = (all_a + all_b) % PRIME
    elif task == "subtract":
        all_y = (all_a - all_b) % PRIME
    else:
        all_y = (all_a * all_b) % PRIME
    idx  = torch.randperm(len(all_a), generator=rng)
    n_tr = int(len(all_a) * TRAIN_FRAC)
    tr, te = idx[:n_tr], idx[n_tr:]
    return (all_a[tr].to(DEVICE), all_b[tr].to(DEVICE), all_y[tr].to(DEVICE),
            all_a[te].to(DEVICE), all_b[te].to(DEVICE), all_y[te].to(DEVICE))

# ── Model ─────────────────────────────────────────────────────
class GrokNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(PRIME+1, D_MODEL)
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
        h = self.ln1(h + h2)
        h = self.ln2(h + self.ff(h))
        return self.unemb(h[:, 2, :]), w

    def logits(self, a, b):
        return self.forward(a, b)[0]

# ── Probes ────────────────────────────────────────────────────
_fc = {}
def get_fourier():
    if PRIME not in _fc:
        F = torch.zeros(PRIME, PRIME); F[0] = 1.0/PRIME**0.5
        for k in range(1, PRIME//2+1):
            j = torch.arange(PRIME, dtype=torch.float32)
            F[2*k-1] = torch.cos(2*math.pi*k*j/PRIME) * (2/PRIME)**0.5
            if 2*k < PRIME:
                F[2*k] = torch.sin(2*math.pi*k*j/PRIME) * (2/PRIME)**0.5
        _fc[PRIME] = F.to(DEVICE)
    return _fc[PRIME]

def probe(model, ap, bp):
    # K_out, H_n from embedding Fourier spectrum
    W  = model.embed.weight[:PRIME].detach().float()
    Wf = get_fourier() @ W
    pw = (Wf**2).mean(dim=1)
    pw = pw / (pw.sum() + 1e-12)
    pw_np = pw.cpu().numpy()
    H_n   = float(np.clip(-np.sum(pw_np * np.log(pw_np + 1e-12)) / math.log(PRIME), 0, 1))
    K_out = float(np.sort(pw_np)[-(max(1, PRIME//10)):].sum())

    # K_in total + directional split
    model.eval()
    with torch.no_grad():
        _, w = model.forward(ap, bp)
    # w shape: (batch, heads, seq, seq) — attention from pos 2 to pos 0,1
    w_to_a   = float(w[:, :, 2, 0].mean().clip(0, 1).item())
    w_to_b   = float(w[:, :, 2, 1].mean().clip(0, 1).item())
    K_in     = float(min(w_to_a + w_to_b, 1.0))
    K_in_a   = w_to_a
    K_in_b   = w_to_b
    K_in_asym = abs(K_in_a - K_in_b) / (K_in_a + K_in_b + 1e-12)

    # beta
    beta = None
    if K_out > 0.01 and K_in > 0.01:
        log_Ko = math.log(K_out)
        if abs(log_Ko) > 1e-6:
            beta = (math.log(K_in) - H_n) / log_Ko

    # manifold distance V
    V = None
    if beta is not None:
        try:
            V = (K_out**beta - K_in * math.exp(-H_n))**2
        except:
            pass

    return K_in, K_out, H_n, beta, V, K_in_asym, K_in_a, K_in_b

# ── Rolling dβ/dt ─────────────────────────────────────────────
def compute_dbdt(history, window=BETA_WINDOW):
    betas  = [r["beta"]  for r in history[-window:] if r["beta"] is not None]
    epochs = [r["epoch"] for r in history[-window:] if r["beta"] is not None]
    if len(betas) < 3: return None
    x = np.array(epochs, dtype=float); y = np.array(betas, dtype=float)
    x -= x.mean()
    return float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0

# ── Checkpoint ────────────────────────────────────────────────
def save_checkpoint(all_results, suffix=""):
    def clean(obj):
        if isinstance(obj, dict):  return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [clean(x) for x in obj]
        if isinstance(obj, float): return round(obj, 6)
        return obj
    fname = f"why2_bvel3_{TS}{suffix}.json"
    with open(fname, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"  [ckpt: {fname}]")
    return fname

# ── Train ─────────────────────────────────────────────────────
def train_run(task, seed, all_results):
    torch.manual_seed(seed); np.random.seed(seed)
    atr, btr, ytr, ate, bte, yte = make_data(task, seed)
    pidx = torch.randperm(len(atr))[:256]
    ap, bp = atr[pidx], btr[pidx]

    model = GrokNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    history = []; grok_ep = None; last_ckpt = 0

    print(f"\n  [{task} s={seed}] training to ep={MAX_EPOCHS}...")

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        loss = F.cross_entropy(model.logits(atr, btr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % PROBE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                acc_te = (model.logits(ate, bte).argmax(-1) == yte).float().mean().item()

            K_in, K_out, H_n, beta, V, K_in_asym, K_in_a, K_in_b = probe(model, ap, bp)
            ratio = K_out / (K_in + 1e-8)
            dbdt  = compute_dbdt(history)

            rec = {"epoch": epoch, "acc_te": acc_te,
                   "K_in": K_in, "K_out": K_out, "H_n": H_n,
                   "beta": beta, "V": V, "ratio": ratio, "dbdt": dbdt,
                   "K_in_asym": K_in_asym, "K_in_a": K_in_a, "K_in_b": K_in_b,
                   "post_grok": (epoch - grok_ep) if grok_ep else None}
            history.append(rec)

            if grok_ep is None and acc_te >= GROK_THRESH:
                grok_ep = epoch
                b_str = f"{beta:.4f}" if beta else "N/A"
                print(f"  [{task} s={seed}] *** GROK ep={epoch} | "
                      f"β={b_str}  K_out={K_out:.3f}  K_in={K_in:.3f}  asym={K_in_asym:.3f}")

            if grok_ep and epoch % 500 == 0:
                b_str = f"{beta:.4f}" if beta else "N/A"
                d_str = f"{dbdt:.6f}" if dbdt else "N/A"
                print(f"  [{task} s={seed}] ep={epoch:5d} | "
                      f"β={b_str}  dβ/dt={d_str}  K_out={K_out:.3f}  asym={K_in_asym:.3f}")

            if epoch - last_ckpt >= CKPT_EVERY:
                all_results[f"{task}_s{seed}"] = {
                    "task": task, "seed": seed,
                    "grok_ep": grok_ep, "history": history}
                save_checkpoint(all_results, f"_ckpt{epoch}")
                last_ckpt = epoch

    if history:
        r = history[-1]
        b_str = f"{r['beta']:.4f}" if r['beta'] else "N/A"
        d_str = f"{r['dbdt']:.6f}" if r['dbdt'] else "N/A"
        print(f"  [{task} s={seed}] FINAL ep={r['epoch']} | "
              f"β={b_str}  dβ/dt={d_str}  K_out={r['K_out']:.3f}")

    return grok_ep, history

# ── Main ──────────────────────────────────────────────────────
all_results = {}
try:
    for task in TASKS:
        print(f"\n{'─'*55}\nTASK: {task.upper()}\n{'─'*55}")
        for seed in SEEDS:
            grok_ep, history = train_run(task, seed, all_results)
            all_results[f"{task}_s{seed}"] = {
                "task": task, "seed": seed,
                "grok_ep": grok_ep, "history": history}
            save_checkpoint(all_results)

except KeyboardInterrupt:
    print("\n[Interrupted — saving partial results...]")
    save_checkpoint(all_results, "_interrupted")

# ── Velocity Analysis ─────────────────────────────────────────
print(f"\n{'='*65}")
print("β VELOCITY ANALYSIS — three-task ordering")
print(f"{'='*65}")
print(f"Prediction: β*(multiply) < β*(add) < β*(subtract)\n")

task_summaries = {}
for task in TASKS:
    beta_groks, beta_finals, decel_ratios, beta_stars = [], [], [], []
    for seed in SEEDS:
        key = f"{task}_s{seed}"
        if key not in all_results: continue
        run = all_results[key]
        grok_ep = run["grok_ep"]
        hist    = run["history"]
        if not grok_ep or not hist: continue

        post = [r for r in hist if r.get("post_grok") and r["post_grok"] > 0
                and r["dbdt"] is not None]
        if len(post) < 6: continue

        n = len(post)
        early_v = np.mean([r["dbdt"] for r in post[:n//3]])
        late_v  = np.mean([r["dbdt"] for r in post[2*n//3:]])

        b_grok  = next((r["beta"] for r in hist
                        if r["epoch"] >= grok_ep and r["beta"]), None)
        b_final = hist[-1]["beta"]
        if b_grok: beta_groks.append(b_grok)
        if b_final: beta_finals.append(b_final)

        dr = late_v / early_v if abs(early_v) > 1e-9 else None
        if dr: decel_ratios.append(dr)

        # β* estimate from velocity ratio method
        if early_v > 1e-9 and late_v > 0 and b_grok and b_final and abs(early_v - late_v) > 1e-9:
            bs = (late_v * b_grok - early_v * b_final) / (late_v - early_v)
            if 0 < bs < 30:
                beta_stars.append(bs)

        print(f"  {task} s={seed}:")
        print(f"    dβ/dt early={early_v:.6f}  late={late_v:.6f}  "
              f"decel_ratio={dr:.2f}" if dr else "    dβ/dt decel ratio: N/A")
        b_g = f"{b_grok:.3f}" if b_grok else "N/A"
        b_f = f"{b_final:.3f}" if b_final else "N/A"
        print(f"    β@grok={b_g}  β@final={b_f}")
        if late_v < early_v * 0.5:
            print(f"    → DECELERATING ✓")
        elif late_v > early_v * 1.1:
            print(f"    → ACCELERATING")
        else:
            print(f"    → CONSTANT VELOCITY")

    summary = {
        "beta_grok_mean":  float(np.mean(beta_groks))  if beta_groks  else None,
        "beta_grok_std":   float(np.std(beta_groks))   if beta_groks  else None,
        "beta_final_mean": float(np.mean(beta_finals)) if beta_finals else None,
        "beta_final_std":  float(np.std(beta_finals))  if beta_finals else None,
        "beta_star_mean":  float(np.mean(beta_stars))  if beta_stars  else None,
        "beta_star_std":   float(np.std(beta_stars))   if beta_stars  else None,
        "decel_ratio_mean":float(np.mean(decel_ratios))if decel_ratios else None,
    }
    task_summaries[task] = summary

print(f"\n{'─'*55}")
print("TASK SUMMARY — β* ordering")
print(f"{'─'*55}")
for task in TASKS:
    s = task_summaries.get(task, {})
    bg  = f"{s['beta_grok_mean']:.3f}±{s['beta_grok_std']:.3f}" if s.get('beta_grok_mean') else "N/A"
    bf  = f"{s['beta_final_mean']:.3f}±{s['beta_final_std']:.3f}" if s.get('beta_final_mean') else "N/A"
    bs  = f"{s['beta_star_mean']:.2f}±{s['beta_star_std']:.2f}" if s.get('beta_star_mean') else "N/A"
    dr  = f"{s['decel_ratio_mean']:.2f}" if s.get('decel_ratio_mean') else "N/A"
    print(f"  {task:10s}  β@grok={bg}  β@final={bf}  β*≈{bs}  decel={dr}")

# Check ordering
bs_vals = {t: task_summaries[t].get("beta_star_mean") for t in TASKS
           if task_summaries.get(t, {}).get("beta_star_mean")}
if len(bs_vals) == 3:
    ordering_correct = bs_vals.get("multiply",0) < bs_vals.get("add",0) < bs_vals.get("subtract",0)
    print(f"\n  Predicted ordering confirmed: {ordering_correct}")
    if ordering_correct:
        print(f"  ✓ multiply β* < add β* < subtract β*")
        print(f"  → β* encodes algebraic task complexity")
    else:
        actual = sorted(bs_vals.items(), key=lambda x: x[1])
        print(f"  Actual ordering: {' < '.join(t for t,_ in actual)}")

# ── Asymmetry Analysis ────────────────────────────────────────
print(f"\n{'─'*55}")
print("K_in DIRECTIONAL ASYMMETRY — subtract vs add/multiply")
print(f"{'─'*55}")
print("Prediction: subtract asym stays elevated post-grok; others collapse\n")

for task in TASKS:
    for seed in SEEDS:
        key = f"{task}_s{seed}"
        if key not in all_results: continue
        run = all_results[key]
        grok_ep = run["grok_ep"]
        hist    = run["history"]
        if not grok_ep: continue

        asym_at_grok = next((r["K_in_asym"] for r in hist
                             if r["epoch"] >= grok_ep and r.get("K_in_asym") is not None), None)
        post_asym = [r["K_in_asym"] for r in hist
                     if r.get("post_grok") and r["post_grok"] > 2000
                     and r.get("K_in_asym") is not None]
        asym_late = np.mean(post_asym) if post_asym else None

        ag = f"{asym_at_grok:.4f}" if asym_at_grok else "N/A"
        al = f"{asym_late:.4f}" if asym_late else "N/A"
        print(f"  {task} s={seed}: asym@grok={ag}  asym@late={al}")

# ── Plot ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor("#0a0e1a")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)
TCOLS = {"add": "#00e5ff", "subtract": "#ff6b9d", "multiply": "#69ff47"}

def sty(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor("#0f1525"); ax.tick_params(colors="#aabbcc", labelsize=7)
    ax.grid(alpha=0.12, color="#aabbcc")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    ax.set_title(title, color="#e8f4f8", fontsize=8, pad=4)
    if xlabel: ax.set_xlabel(xlabel, color="#aabbcc", fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color="#aabbcc", fontsize=7)

# Panel 1: β full trajectories — all 3 tasks
ax0 = fig.add_subplot(gs[0, :2])
sty(ax0, "β(t) — all three tasks (dashed = grok)", "epoch", "β")
for key, run in all_results.items():
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not hist: continue
    ep = [r["epoch"] for r in hist if r["beta"] is not None]
    bt = [r["beta"]  for r in hist if r["beta"] is not None]
    if not bt: continue
    ax0.plot(ep, bt, color=TCOLS[task], lw=1.2, alpha=0.7,
             label=task if run["seed"] == 42 else "")
    if grok_ep:
        ax0.axvline(grok_ep, color=TCOLS[task], ls="--", lw=0.8, alpha=0.35)
ax0.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")
ax0.set_ylim(0, 8)

# Panel 2: dβ/dt post-grok
ax1 = fig.add_subplot(gs[0, 2])
sty(ax1, "dβ/dt post-grok", "epochs after grok", "dβ/dt")
for key, run in all_results.items():
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not grok_ep: continue
    pep = [r["epoch"] - grok_ep for r in hist
           if r.get("post_grok") and r["post_grok"] > 0 and r["dbdt"] is not None]
    pdb = [r["dbdt"] for r in hist
           if r.get("post_grok") and r["post_grok"] > 0 and r["dbdt"] is not None]
    if pep:
        ax1.plot(pep, pdb, color=TCOLS[task], lw=1.2, alpha=0.7,
                 label=task if run["seed"] == 42 else "")
ax1.axhline(0, color="#ffffff", ls="--", lw=0.8, alpha=0.4)
ax1.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")

# Panel 3: K_out trajectories
ax2 = fig.add_subplot(gs[1, :2])
sty(ax2, "K_out(t) — Fourier consolidation", "epoch", "K_out")
for key, run in all_results.items():
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not hist: continue
    ep = [r["epoch"] for r in hist]
    ko = [r["K_out"] for r in hist]
    ax2.plot(ep, ko, color=TCOLS[task], lw=1.2, alpha=0.7,
             label=task if run["seed"] == 42 else "")
    if grok_ep:
        ax2.axvline(grok_ep, color=TCOLS[task], ls="--", lw=0.8, alpha=0.35)
ax2.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")
ax2.set_ylim(0, 1.05)

# Panel 4: K_in asymmetry trajectories
ax3 = fig.add_subplot(gs[1, 2])
sty(ax3, "K_in directional asymmetry", "epoch", "|K_in_a - K_in_b| / K_in")
for key, run in all_results.items():
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not hist: continue
    ep   = [r["epoch"]    for r in hist if r.get("K_in_asym") is not None]
    asym = [r["K_in_asym"] for r in hist if r.get("K_in_asym") is not None]
    if ep:
        ax3.plot(ep, asym, color=TCOLS[task], lw=1.2, alpha=0.7,
                 label=task if run["seed"] == 42 else "")
        if grok_ep:
            ax3.axvline(grok_ep, color=TCOLS[task], ls="--", lw=0.8, alpha=0.35)
ax3.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")
ax3.set_ylim(0, 0.25)

# Panel 5: β* bar chart summary
ax4 = fig.add_subplot(gs[2, 0])
sty(ax4, "β* estimates — task ordering", "task", "β* (estimated fixed point)")
tasks_plot = [t for t in TASKS if task_summaries.get(t, {}).get("beta_star_mean")]
means = [task_summaries[t]["beta_star_mean"] for t in tasks_plot]
stds  = [task_summaries[t]["beta_star_std"]  for t in tasks_plot]
bars  = ax4.bar(tasks_plot, means, yerr=stds, color=[TCOLS[t] for t in tasks_plot],
                alpha=0.8, capsize=5, error_kw={"color": "#aabbcc", "lw": 1.5})
ax4.tick_params(axis="x", colors="#aabbcc")

# Panel 6: σ² prediction (T/2β*²) — relative to multiply
ax5 = fig.add_subplot(gs[2, 1])
sty(ax5, "Predicted equilibrium variance (rel. to multiply)", "task", "σ² / σ²_multiply")
if means:
    ref = means[0] if tasks_plot[0] == "multiply" else means[tasks_plot.index("multiply")] if "multiply" in tasks_plot else 1
    rel_vars = [(ref / m)**2 for m in means]  # σ² ∝ 1/β*², so ratio = (β*_ref/β*_task)²
    ax5.bar(tasks_plot, rel_vars, color=[TCOLS[t] for t in tasks_plot], alpha=0.8)
    ax5.tick_params(axis="x", colors="#aabbcc")

# Panel 7: manifold distance V
ax6 = fig.add_subplot(gs[2, 2])
sty(ax6, "Manifold distance V (log scale)", "epoch", "V")
for key, run in all_results.items():
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not hist: continue
    ep = [r["epoch"] for r in hist if r.get("V") is not None]
    Vs = [max(r["V"], 1e-10) for r in hist if r.get("V") is not None]
    if Vs:
        ax6.semilogy(ep, Vs, color=TCOLS[task], lw=1.2, alpha=0.7,
                     label=task if run["seed"] == 42 else "")
        if grok_ep:
            ax6.axvline(grok_ep, color=TCOLS[task], ls="--", lw=0.8, alpha=0.35)
ax6.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")

fig.suptitle(f"WHY2 β* Ordering — multiply < add < subtract predicted\n"
             f"add=cyan  subtract=pink  multiply=green  dashed=grok epoch",
             color="#e8f4f8", fontsize=11, fontweight="bold")

fname = f"why2_bvel3_{TS}.png"
plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor="#0a0e1a")
plt.show()
print(f"Saved: {fname}")

# Final save
final_fname = save_checkpoint(all_results, "_final")
print(f"\nDone at {datetime.now():%H:%M}")
print(f"Upload {final_fname} when finished")
print(f"\nKey question: is β*(subtract) > β*(add) > β*(multiply)?")