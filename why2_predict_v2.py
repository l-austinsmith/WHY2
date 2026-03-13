# ============================================================
# WHY2 STRUCTURAL PREDICTION TEST — v2
#
# WHY v1 FAILED TO CONVERGE:
#   All three new tasks showed K_out ≈ 0.1 flat at ep=18000.
#   Networks memorized but did not grok — β* was never reached.
#   The predictions could not be evaluated, not falsified.
#
# v2 CHANGES (three targeted fixes):
#
#   1. MAX_EPOCHS: 18000 → 36000
#      K_out curves were still rising at ep=18000 for affine.
#      Division and multiply_add never crystallized at all.
#      These tasks are harder than subtract (grok ~6000-9000).
#      Need double the time budget to reach β* fixed point.
#
#   2. DUAL WEIGHT DECAY: WD=1.0 AND WD=0.5 per task
#      WD=1.0 was calibrated for subtract. These tasks may need
#      less aggressive regularization to escape memorization.
#      Run both per task; report whichever groks. If both grok,
#      report both and check if β* is WD-invariant (it should be
#      if it's a true structural invariant).
#
#   3. TRAIN_FRAC: division uses 0.4 instead of 0.3
#      Division excludes b=0 → only 9312 samples vs 9409.
#      At TRAIN_FRAC=0.3: ~2793 training pairs — borderline.
#      At TRAIN_FRAC=0.4: ~3725 training pairs — safer margin.
#      Add/subtract grok reliably at 0.3; division gets 0.4.
#
# CONVERGENCE MONITOR — new:
#   Added K_out plateau detector. Prints warning if K_out hasn't
#   moved >0.01 in last 5000 epochs post-grok. Tells you early
#   if a run is stuck and you should interrupt + check WD.
#
# PREDICTIONS (unchanged from v1 — made before any training):
#   division      a·b⁻¹ mod p  NMC=1.0  AMF=1.00  β*_pred=1.39  [1.2, 1.6]
#   affine        (2a+b) mod p  NMC=2.0  AMF=0.50  β*_pred=4.11  [3.7, 4.5]
#   multiply_add  a·b+a mod p   NMC=1.5  AMF=0.30  β*_pred=2.38  [2.0, 2.8]
#
# Expected runtime: ~40-48 GPU compute units (T4) for full run
# ============================================================

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, warnings, json, matplotlib
from datetime import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
PREFIX = f"why2_predict_v2_{TS}"

print(f"WHY2 STRUCTURAL PREDICTION TEST v2  |  device: {DEVICE}  |  {datetime.now():%Y-%m-%d %H:%M}")
print(f"Output prefix: {PREFIX}")
print(f"v2 changes: 36k epochs | dual WD (1.0 + 0.5) | division TRAIN_FRAC=0.4")
print("="*65)

# ── Structural predictions (unchanged from v1) ─────────────────
PREDICTIONS = {
    "multiply":     {"NMC": 1.0, "AMF": 0.00, "pred": 0.72, "lo": None, "hi": None, "calib": True},
    "add":          {"NMC": 2.0, "AMF": 0.00, "pred": 3.44, "lo": None, "hi": None, "calib": True},
    "subtract":     {"NMC": 2.0, "AMF": 1.00, "pred": 4.78, "lo": None, "hi": None, "calib": True},
    "division":     {"NMC": 1.0, "AMF": 1.00, "pred": 1.39, "lo": 1.2,  "hi": 1.6,  "calib": False},
    "affine":       {"NMC": 2.0, "AMF": 0.50, "pred": 4.11, "lo": 3.7,  "hi": 4.5,  "calib": False},
    "multiply_add": {"NMC": 1.5, "AMF": 0.30, "pred": 2.38, "lo": 2.0,  "hi": 2.8,  "calib": False},
}

print("\nSTRUCTURAL PREDICTIONS (pre-training, identical to v1):")
print(f"  {'Task':<14} {'NMC':>5} {'AMF':>5} {'β*_pred':>8} {'Range':>14}")
print(f"  {'─'*14} {'─'*5} {'─'*5} {'─'*8} {'─'*14}")
for task, p in PREDICTIONS.items():
    rng = f"[{p['lo']:.1f}, {p['hi']:.1f}]" if not p["calib"] else "(calibration)"
    print(f"  {task:<14} {p['NMC']:>5.1f} {p['AMF']:>5.2f} {p['pred']:>8.2f} {rng:>14}")
print(f"\n  Predictor: β* = 2.72·NMC + 0.67·AMF·NMC − 2.00\n")

# ── Config ────────────────────────────────────────────────────
PRIME        = 97
D_MODEL      = 128
N_HEADS      = 4
D_FF         = 512
LR           = 1e-3
PROBE_EVERY  = 25
GROK_THRESH  = 0.99
MAX_EPOCHS   = 36000          # v2: doubled from 18000
CKPT_EVERY   = 2000           # less noise in output
TASKS        = ["division", "affine", "multiply_add"]
SEEDS        = [42, 123, 7]
WD_VALUES    = [1.0, 0.5]     # v2: dual WD — run both per task
BETA_WINDOW  = 10
BSTAR_WINDOW = 150            # more post-grok probes for stable estimate

# Per-task train fractions — v2 fix for division
TRAIN_FRACS = {
    "division":    0.4,        # v2: 0.3→0.4 (b=0 exclusion reduces dataset)
    "affine":      0.3,
    "multiply_add":0.3,
}

# ── Modular inverse ───────────────────────────────────────────
def mod_inverse(b_tensor, p):
    result = torch.zeros_like(b_tensor)
    for i in range(len(b_tensor)):
        b = b_tensor[i].item()
        if b != 0:
            result[i] = pow(int(b), p - 2, p)
    return result

# ── Data ──────────────────────────────────────────────────────
def make_data(task, seed, train_frac):
    rng   = torch.Generator(); rng.manual_seed(seed)
    all_a = torch.arange(PRIME).repeat(PRIME)
    all_b = torch.arange(PRIME).repeat_interleave(PRIME)

    if task == "division":
        mask  = all_b != 0
        all_a = all_a[mask]; all_b = all_b[mask]
        b_inv = mod_inverse(all_b, PRIME)
        all_y = (all_a * b_inv) % PRIME
    elif task == "affine":
        all_y = (2 * all_a + all_b) % PRIME
    elif task == "multiply_add":
        all_y = (all_a * all_b + all_a) % PRIME
    else:
        raise ValueError(f"Unknown task: {task}")

    assert all_y.min() >= 0 and all_y.max() < PRIME
    idx  = torch.randperm(len(all_a), generator=rng)
    n_tr = int(len(all_a) * train_frac)
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

    def logits(self, a, b):
        return self.forward(a, b)[0]

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

# ── Probes ────────────────────────────────────────────────────
def probe(model, ap, bp):
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
    w_to_a    = float(w[:, :, 2, 0].mean().clip(0, 1).item())
    w_to_b    = float(w[:, :, 2, 1].mean().clip(0, 1).item())
    K_in      = float(min(w_to_a + w_to_b, 1.0))
    K_in_asym = abs(w_to_a - w_to_b) / (w_to_a + w_to_b + 1e-12)

    beta = None
    if K_out > 0.01 and K_in > 0.01:
        log_Ko = math.log(K_out)
        if abs(log_Ko) > 1e-6:
            beta = (math.log(K_in) - H_n) / log_Ko

    V = None
    if beta is not None:
        try:
            V = (K_out**beta - K_in * math.exp(-H_n))**2
        except Exception:
            pass

    return K_in, K_out, H_n, beta, V, K_in_asym, w_to_a, w_to_b

# ── Rolling dβ/dt ─────────────────────────────────────────────
def compute_dbdt(history, window=BETA_WINDOW):
    betas  = [r["beta"]  for r in history[-window:] if r["beta"] is not None]
    epochs = [r["epoch"] for r in history[-window:] if r["beta"] is not None]
    if len(betas) < 3: return None
    x = np.array(epochs, dtype=float); y = np.array(betas, dtype=float)
    x -= x.mean()
    denom = np.dot(x, x)
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0

# ── K_out plateau detector ────────────────────────────────────
def check_plateau(history, grok_ep, window=200):
    """Returns True if K_out hasn't moved >0.01 in last `window` probes post-grok."""
    if not grok_ep: return False
    post = [r["K_out"] for r in history if r.get("post_grok") and r["post_grok"] > 0]
    if len(post) < window: return False
    recent = post[-window:]
    return (max(recent) - min(recent)) < 0.01

# ── Checkpoint ────────────────────────────────────────────────
def save_checkpoint(all_results, suffix=""):
    def clean(obj):
        if isinstance(obj, dict):  return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [clean(x) for x in obj]
        if isinstance(obj, float): return round(obj, 6)
        return obj
    fname = f"{PREFIX}{suffix}.json"
    with open(fname, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"  [ckpt → {fname}]")
    return fname

# ── Train one run ─────────────────────────────────────────────
def train_run(task, seed, wd, all_results):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_frac = TRAIN_FRACS[task]
    atr, btr, ytr, ate, bte, yte = make_data(task, seed, train_frac)

    pidx = torch.randperm(len(atr))[:256]
    ap, bp = atr[pidx], btr[pidx]

    model = GrokNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)

    history   = []
    grok_ep   = None
    last_ckpt = 0
    plateau_warned = False

    label = f"{task} s={seed} WD={wd}"
    print(f"\n  [{label}] training to ep={MAX_EPOCHS} (train_frac={train_frac})...")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss = F.cross_entropy(model.logits(atr, btr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % PROBE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                acc_te = (model.logits(ate, bte).argmax(-1) == yte).float().mean().item()

            K_in, K_out, H_n, beta, V, K_in_asym, w_a, w_b = probe(model, ap, bp)
            ratio = K_out / (K_in + 1e-8)
            dbdt  = compute_dbdt(history)

            rec = {
                "epoch": epoch, "acc_te": round(acc_te, 4),
                "K_in": K_in, "K_out": K_out, "H_n": H_n,
                "beta": beta, "V": V, "ratio": ratio, "dbdt": dbdt,
                "K_in_asym": K_in_asym, "w_to_a": w_a, "w_to_b": w_b,
                "post_grok": (epoch - grok_ep) if grok_ep else None,
            }
            history.append(rec)

            # Detect grokking
            if grok_ep is None and acc_te >= GROK_THRESH:
                grok_ep = epoch
                b_str = f"{beta:.4f}" if beta is not None else "N/A"
                print(f"  [{label}] *** GROK ep={epoch} | "
                      f"β={b_str}  K_out={K_out:.3f}  asym={K_in_asym:.3f}")

            # Progress every 1000 epochs post-grok
            if grok_ep and epoch % 1000 == 0:
                b_str = f"{beta:.4f}" if beta is not None else "N/A"
                d_str = f"{dbdt:.6f}" if dbdt is not None else "N/A"
                print(f"  [{label}] ep={epoch:5d} | "
                      f"β={b_str}  dβ/dt={d_str}  K_out={K_out:.3f}  asym={K_in_asym:.3f}")

            # Plateau warning — print once if K_out stuck post-grok
            if grok_ep and not plateau_warned and check_plateau(history, grok_ep):
                print(f"  [{label}] ⚠ K_out PLATEAU at ep={epoch} — "
                      f"K_out={K_out:.3f} not rising. May need more epochs or lower WD.")
                plateau_warned = True

            # Checkpoint
            if epoch - last_ckpt >= CKPT_EVERY:
                run_key = f"{task}_s{seed}_WD{wd}"
                all_results[run_key] = {
                    "task": task, "seed": seed, "wd": wd,
                    "grok_ep": grok_ep, "history": history}
                save_checkpoint(all_results, f"_ckpt{epoch}")
                last_ckpt = epoch

    if history:
        r = history[-1]
        b_str = f"{r['beta']:.4f}" if r["beta"] is not None else "N/A"
        d_str = f"{r['dbdt']:.6f}" if r["dbdt"] is not None else "N/A"
        ko_trend = "rising" if len(history) > 10 and history[-1]["K_out"] > history[-10]["K_out"] else "flat"
        print(f"  [{label}] FINAL ep={r['epoch']} | "
              f"β={b_str}  dβ/dt={d_str}  K_out={r['K_out']:.3f} ({ko_trend})")

    return grok_ep, history

# ── Main loop ─────────────────────────────────────────────────
all_results = {}
try:
    for task in TASKS:
        pred = PREDICTIONS[task]
        for wd in WD_VALUES:
            print(f"\n{'─'*65}")
            print(f"TASK: {task.upper()}  |  WD={wd}")
            print(f"  Prediction: β* ≈ {pred['pred']:.2f}  "
                  f"(NMC={pred['NMC']:.1f}, AMF={pred['AMF']:.2f})  "
                  f"Range: [{pred['lo']:.1f}, {pred['hi']:.1f}]")
            print(f"  train_frac={TRAIN_FRACS[task]}  max_epochs={MAX_EPOCHS}")
            print(f"{'─'*65}")

            for seed in SEEDS:
                grok_ep, history = train_run(task, seed, wd, all_results)
                run_key = f"{task}_s{seed}_WD{wd}"
                all_results[run_key] = {
                    "task": task, "seed": seed, "wd": wd,
                    "grok_ep": grok_ep, "history": history}
                save_checkpoint(all_results)

except KeyboardInterrupt:
    print("\n[Interrupted — saving partial results]")
    save_checkpoint(all_results, "_interrupted")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("β* ANALYSIS — per task × WD combination")
print(f"{'='*65}")

# Collect β* per (task, wd)
task_wd_summaries = {}

for task in TASKS:
    for wd in WD_VALUES:
        beta_stars  = []
        beta_groks  = []
        beta_finals = []
        grok_eps    = []

        for seed in SEEDS:
            key = f"{task}_s{seed}_WD{wd}"
            if key not in all_results: continue
            run     = all_results[key]
            grok_ep = run["grok_ep"]
            hist    = run["history"]
            if not grok_ep or not hist: continue

            post = [r for r in hist
                    if r.get("post_grok") and r["post_grok"] > 0
                    and r["dbdt"] is not None]
            if len(post) < 6: continue

            # β* = mean of last BSTAR_WINDOW post-grok probes
            late_betas = [r["beta"] for r in post[-BSTAR_WINDOW:]
                          if r["beta"] is not None and r["beta"] > 0]
            if late_betas:
                beta_stars.append(float(np.mean(late_betas)))

            b_grok = next((r["beta"] for r in hist
                           if r["epoch"] >= grok_ep and r["beta"] is not None), None)
            if b_grok:  beta_groks.append(b_grok)
            if hist[-1]["beta"]: beta_finals.append(hist[-1]["beta"])
            grok_eps.append(grok_ep)

        combo_key = f"{task}_WD{wd}"
        task_wd_summaries[combo_key] = {
            "task": task, "wd": wd,
            "n_grokked":       len(grok_eps),
            "grok_eps":        grok_eps,
            "beta_star_mean":  float(np.mean(beta_stars))  if beta_stars  else None,
            "beta_star_std":   float(np.std(beta_stars))   if beta_stars  else None,
            "beta_grok_mean":  float(np.mean(beta_groks))  if beta_groks  else None,
            "beta_final_mean": float(np.mean(beta_finals)) if beta_finals else None,
        }

        n = len(grok_eps)
        if n > 0:
            g_mean = float(np.mean(grok_eps))
            bs_str = (f"{task_wd_summaries[combo_key]['beta_star_mean']:.3f}"
                      f"±{task_wd_summaries[combo_key]['beta_star_std']:.3f}"
                      if task_wd_summaries[combo_key]['beta_star_mean'] else "N/A")
            print(f"\n  {task} WD={wd}: {n}/3 grokked | "
                  f"grok_ep_mean={g_mean:.0f} | β*={bs_str}")
        else:
            print(f"\n  {task} WD={wd}: 0/3 grokked — DID NOT GROK in {MAX_EPOCHS} epochs")

# ── Pick best β* per task (prefer WD that grokked more; tie → lower WD) ──
print(f"\n{'─'*65}")
print("BEST β* PER TASK (most grokked seeds, then lowest WD)")
print(f"{'─'*65}")

task_summaries = {}   # task → best summary (for prediction table)

for task in TASKS:
    best = None
    for wd in WD_VALUES:
        combo = task_wd_summaries.get(f"{task}_WD{wd}")
        if combo is None or combo["n_grokked"] == 0: continue
        if combo["beta_star_mean"] is None: continue
        if best is None or combo["n_grokked"] > best["n_grokked"]:
            best = combo
        elif combo["n_grokked"] == best["n_grokked"]:
            # Tie: prefer lower WD (less likely to suppress grokking)
            if wd < best["wd"]:
                best = combo

    task_summaries[task] = best
    if best:
        print(f"  {task}: β*={best['beta_star_mean']:.3f}±{best['beta_star_std']:.3f}  "
              f"(WD={best['wd']}, n={best['n_grokked']}/3 grokked)")
    else:
        print(f"  {task}: NO GROKKING — predictions cannot be evaluated")

# ═══════════════════════════════════════════════════════════════
# PREDICTION TABLE — the whole point
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("PREDICTION vs RESULT TABLE")
print(f"{'='*65}")
print(f"  Predictor: β* = 2.72·NMC + 0.67·AMF·NMC − 2.00\n")
print(f"  {'Task':<14} {'Pred':>6} {'Range':>12} {'Measured':>12} {'In range?':>10}  Verdict")
print(f"  {'─'*14} {'─'*6} {'─'*12} {'─'*12} {'─'*10}  {'─'*22}")

all_pass  = True
any_grok  = False
results_for_json = {}

for task in TASKS:
    p    = PREDICTIONS[task]
    best = task_summaries.get(task)
    bs   = best["beta_star_mean"] if best else None
    bs_s = best["beta_star_std"]  if best else None

    pred_str = f"{p['pred']:.2f}"
    rng_str  = f"[{p['lo']:.1f},{p['hi']:.1f}]"

    if bs is not None:
        any_grok  = True
        meas_str  = f"{bs:.2f}±{bs_s:.2f}"
        in_range  = p["lo"] <= bs <= p["hi"]
        verdict   = "✓ CONFIRMED" if in_range else "✗ OUTSIDE RANGE"
        if not in_range: all_pass = False
    else:
        meas_str = "N/A (no grok)"
        in_range = None
        verdict  = "— DID NOT GROK (need more epochs)"
        all_pass = False

    print(f"  ★{task:<13} {pred_str:>6} {rng_str:>12} {meas_str:>12} "
          f"{'YES' if in_range else 'NO' if in_range is False else 'N/A':>10}  {verdict}")

    results_for_json[task] = {
        "predicted": p["pred"], "lo": p["lo"], "hi": p["hi"],
        "measured": bs, "measured_std": bs_s,
        "in_range": in_range, "NMC": p["NMC"], "AMF": p["AMF"],
    }

print(f"\n  ALL PREDICTIONS CONFIRMED: {all_pass}")

if all_pass and any_grok:
    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  FRAMEWORK VERDICT: β* IS A STRUCTURAL INVARIANT        ║
  ║                                                          ║
  ║  The algebraic symmetry-breaking predictor works.        ║
  ║  β* can be computed from task structure BEFORE training. ║
  ║  WHY2 has transitioned from measurement → prediction.    ║
  ╚══════════════════════════════════════════════════════════╝""")
elif any_grok:
    print(f"\n  PARTIAL — residual analysis:")
    for task in TASKS:
        r  = results_for_json[task]
        bs = r["measured"]
        if bs is None or r["in_range"]: continue
        residual = bs - r["predicted"]
        pct      = residual / r["predicted"] * 100
        print(f"\n  {task}: measured={bs:.2f}  predicted={r['predicted']:.2f}  "
              f"residual={residual:+.2f} ({pct:+.1f}%)")
        if residual > 0.3:
            print(f"    → β* HIGHER than predicted")
            print(f"    → Possible: extra symmetry-breaking cost not in NMC/AMF")
        elif residual < -0.3:
            print(f"    → β* LOWER than predicted")
            print(f"    → Possible: hidden symmetry reduces effective AMF")
            print(f"    → Action: recheck AMF assignment for {task}")
else:
    print(f"\n  NO TASKS GROKKED — predictions cannot be evaluated.")
    print(f"  Consider: more epochs, lower WD, higher train_frac, or smaller p.")

# ── WD invariance check ───────────────────────────────────────
print(f"\n{'─'*65}")
print("WD INVARIANCE CHECK")
print(f"  If β* is a structural invariant, it should be WD-independent.")
print(f"{'─'*65}")

for task in TASKS:
    vals = {}
    for wd in WD_VALUES:
        combo = task_wd_summaries.get(f"{task}_WD{wd}")
        if combo and combo["beta_star_mean"] is not None:
            vals[wd] = combo["beta_star_mean"]
    if len(vals) == 2:
        diff = abs(vals[WD_VALUES[0]] - vals[WD_VALUES[1]])
        rel  = diff / np.mean(list(vals.values())) * 100
        invar = "✓ WD-invariant" if rel < 15 else "✗ WD-sensitive"
        print(f"  {task}: WD=1.0 → β*={vals.get(1.0, 'N/A'):.3f}  "
              f"WD=0.5 → β*={vals.get(0.5, 'N/A'):.3f}  "
              f"diff={diff:.3f} ({rel:.1f}%)  {invar}")
    elif len(vals) == 1:
        wd_only = list(vals.keys())[0]
        print(f"  {task}: only WD={wd_only} grokked — cannot check invariance")
    else:
        print(f"  {task}: no grokking — cannot check invariance")

# ── Full ordering check ───────────────────────────────────────
print(f"\n{'─'*65}")
print("FULL ORDERING (all 6 tasks)")
print(f"{'─'*65}")

known    = {"multiply": 0.72, "add": 3.44, "subtract": 4.78}
measured = {t: task_summaries[t]["beta_star_mean"]
            for t in TASKS if task_summaries.get(t) and task_summaries[t]}
all_vals = {**known, **{k: v for k, v in measured.items() if v is not None}}

for i, (t, v) in enumerate(sorted(all_vals.items(), key=lambda x: x[1])):
    src  = "(calibration)" if t in known else "(measured now)"
    pred = PREDICTIONS[t]["pred"]
    print(f"  {i+1}. {t:<14}  β*={v:.2f}  (pred={pred:.2f})  {src}")

# Check ordering among new tasks
new_vals = {t: measured.get(t) for t in TASKS if measured.get(t) is not None}
if len(new_vals) == 3:
    m_add = new_vals.get("multiply_add")
    div   = new_vals.get("division")
    aff   = new_vals.get("affine")
    order_ok = m_add < div < aff if all(v is not None for v in [m_add, div, aff]) else None
    print(f"\n  Predicted ordering: multiply_add < division < affine")
    if order_ok is not None:
        print(f"  multiply_add={m_add:.2f}  division={div:.2f}  affine={aff:.2f}")
        print(f"  Ordering confirmed: {order_ok}")

# ── Save JSON ─────────────────────────────────────────────────
final_data = {
    "version": "v2",
    "timestamp": TS,
    "config": {
        "MAX_EPOCHS": MAX_EPOCHS, "WD_VALUES": WD_VALUES,
        "TRAIN_FRACS": TRAIN_FRACS, "PRIME": PRIME,
    },
    "predictions": PREDICTIONS,
    "task_wd_summaries": task_wd_summaries,
    "task_summaries": {t: s for t, s in task_summaries.items() if s},
    "prediction_results": results_for_json,
    "all_predictions_confirmed": all_pass,
}
final_json = save_checkpoint(all_results, "_final")
pred_json  = f"{PREFIX}_predictions.json"
with open(pred_json, "w") as f:
    json.dump(final_data, f, indent=2)
print(f"  [predictions → {pred_json}]")

# ═══════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════

TCOLS = {
    "division":    "#ff9f43",
    "affine":      "#a29bfe",
    "multiply_add":"#55efc4",
    "multiply":    "#69ff47",
    "add":         "#00e5ff",
    "subtract":    "#ff6b9d",
}
WD_STYLES = {1.0: "-", 0.5: "--"}

fig = plt.figure(figsize=(28, 20))
fig.patch.set_facecolor("#0a0e1a")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.33)

def sty(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor("#0f1525")
    ax.tick_params(colors="#aabbcc", labelsize=7)
    ax.grid(alpha=0.12, color="#aabbcc")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    ax.set_title(title, color="#e8f4f8", fontsize=8, pad=4)
    if xlabel: ax.set_xlabel(xlabel, color="#aabbcc", fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color="#aabbcc", fontsize=7)

# Panel 1: β(t) — WD=1.0 runs
ax0 = fig.add_subplot(gs[0, :2])
sty(ax0, "β(t) — WD=1.0 runs (dashed = grok epoch)", "epoch", "β")
for key, run in all_results.items():
    if run.get("wd") != 1.0: continue
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not hist: continue
    ep = [r["epoch"] for r in hist if r["beta"] is not None]
    bt = [r["beta"]  for r in hist if r["beta"] is not None]
    if not bt: continue
    col = TCOLS.get(task, "#fff")
    ax0.plot(ep, bt, color=col, lw=1.2, alpha=0.7,
             label=task if run["seed"] == 42 else "")
    if grok_ep: ax0.axvline(grok_ep, color=col, ls="--", lw=0.7, alpha=0.35)
ax0.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")
ax0.set_ylim(0, 8)

# Panel 2: β(t) — WD=0.5 runs
ax0b = fig.add_subplot(gs[0, 2])
sty(ax0b, "β(t) — WD=0.5 runs", "epoch", "β")
for key, run in all_results.items():
    if run.get("wd") != 0.5: continue
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not hist: continue
    ep = [r["epoch"] for r in hist if r["beta"] is not None]
    bt = [r["beta"]  for r in hist if r["beta"] is not None]
    if not bt: continue
    col = TCOLS.get(task, "#fff")
    ax0b.plot(ep, bt, color=col, lw=1.2, alpha=0.7,
              label=task if run["seed"] == 42 else "")
    if grok_ep: ax0b.axvline(grok_ep, color=col, ls="--", lw=0.7, alpha=0.35)
ax0b.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")
ax0b.set_ylim(0, 8)

# Panel 3: K_out(t) — WD=1.0
ax1 = fig.add_subplot(gs[1, :2])
sty(ax1, "K_out(t) — WD=1.0  (watch for crystallization)", "epoch", "K_out")
for key, run in all_results.items():
    if run.get("wd") != 1.0: continue
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not hist: continue
    ep = [r["epoch"] for r in hist]
    ko = [r["K_out"] for r in hist]
    ax1.plot(ep, ko, color=TCOLS.get(task, "#fff"), lw=1.2, alpha=0.7,
             label=task if run["seed"] == 42 else "")
    if grok_ep: ax1.axvline(grok_ep, color=TCOLS.get(task, "#fff"), ls="--", lw=0.7, alpha=0.35)
ax1.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")
ax1.set_ylim(0, 1.05)

# Panel 4: K_out(t) — WD=0.5
ax1b = fig.add_subplot(gs[1, 2])
sty(ax1b, "K_out(t) — WD=0.5", "epoch", "K_out")
for key, run in all_results.items():
    if run.get("wd") != 0.5: continue
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not hist: continue
    ep = [r["epoch"] for r in hist]
    ko = [r["K_out"] for r in hist]
    ax1b.plot(ep, ko, color=TCOLS.get(task, "#fff"), lw=1.2, alpha=0.7,
              label=task if run["seed"] == 42 else "")
    if grok_ep: ax1b.axvline(grok_ep, color=TCOLS.get(task, "#fff"), ls="--", lw=0.7, alpha=0.35)
ax1b.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")
ax1b.set_ylim(0, 1.05)

# Panel 5: dβ/dt post-grok — best WD per task
ax2 = fig.add_subplot(gs[2, :2])
sty(ax2, "dβ/dt post-grok — best WD per task (decel → β* found)", "epochs after grok", "dβ/dt")
for task in TASKS:
    best = task_summaries.get(task)
    if not best: continue
    wd = best["wd"]
    for seed in SEEDS:
        key = f"{task}_s{seed}_WD{wd}"
        if key not in all_results: continue
        run     = all_results[key]
        grok_ep = run["grok_ep"]
        if not grok_ep: continue
        hist = run["history"]
        pep = [r["epoch"] - grok_ep for r in hist
               if r.get("post_grok") and r["post_grok"] > 0 and r["dbdt"] is not None]
        pdb = [r["dbdt"] for r in hist
               if r.get("post_grok") and r["post_grok"] > 0 and r["dbdt"] is not None]
        if pep:
            ax2.plot(pep, pdb, color=TCOLS.get(task, "#fff"), lw=1.0, alpha=0.6,
                     label=f"{task} (WD={wd})" if seed == 42 else "")
ax2.axhline(0, color="#ffffff", ls="--", lw=0.8, alpha=0.4)
ax2.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")

# Panel 6: K_in asymmetry
ax2b = fig.add_subplot(gs[2, 2])
sty(ax2b, "K_in asymmetry post-grok", "epoch", "|w_a−w_b|/K_in")
for key, run in all_results.items():
    if run.get("wd") != 1.0: continue
    hist = run["history"]; grok_ep = run["grok_ep"]; task = run["task"]
    if not hist: continue
    ep   = [r["epoch"]     for r in hist if r.get("K_in_asym") is not None]
    asym = [r["K_in_asym"] for r in hist if r.get("K_in_asym") is not None]
    if ep:
        ax2b.plot(ep, asym, color=TCOLS.get(task, "#fff"), lw=1.0, alpha=0.6,
                  label=task if run["seed"] == 42 else "")
        if grok_ep:
            ax2b.axvline(grok_ep, color=TCOLS.get(task, "#fff"), ls="--", lw=0.7, alpha=0.3)
ax2b.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")
ax2b.set_ylim(0, 0.35)

# Panel 7: Prediction vs Result (key panel)
ax3 = fig.add_subplot(gs[3, :2])
sty(ax3, "β* — PREDICTION vs MEASUREMENT (the key panel)", "task", "β*")

bar_tasks  = TASKS
bar_pred   = [PREDICTIONS[t]["pred"] for t in bar_tasks]
bar_lo     = [PREDICTIONS[t]["lo"]   for t in bar_tasks]
bar_hi     = [PREDICTIONS[t]["hi"]   for t in bar_tasks]
bar_meas   = [task_summaries.get(t, {}) and task_summaries[t]["beta_star_mean"]
              if task_summaries.get(t) else None for t in bar_tasks]
bar_meas_e = [task_summaries.get(t, {}) and task_summaries[t]["beta_star_std"]
              if task_summaries.get(t) else None for t in bar_tasks]
bar_meas   = [task_summaries[t]["beta_star_mean"] if task_summaries.get(t) else None for t in bar_tasks]
bar_meas_e = [task_summaries[t]["beta_star_std"]  if task_summaries.get(t) else None for t in bar_tasks]

x = np.arange(len(bar_tasks)); w = 0.35
ax3.bar(x - w/2, bar_pred, w, color=[TCOLS.get(t, "#aaa") for t in bar_tasks],
        alpha=0.35, label="Predicted β*", zorder=2)
for i, (lo, hi, pred) in enumerate(zip(bar_lo, bar_hi, bar_pred)):
    if lo is not None:
        ax3.errorbar(x[i] - w/2, pred, yerr=[[pred - lo], [hi - pred]],
                     fmt="none", color="#ffffff", capsize=6, lw=2, zorder=3)

meas_vals = [v if v is not None else 0 for v in bar_meas]
meas_errs = [v if v is not None else 0 for v in bar_meas_e]
ax3.bar(x + w/2, meas_vals, w, color=[TCOLS.get(t, "#aaa") for t in bar_tasks],
        alpha=0.9, label="Measured β*", zorder=2)
ax3.errorbar(x + w/2, meas_vals, yerr=meas_errs,
             fmt="none", color="#ffffff", capsize=4, lw=1.5, zorder=3)
ax3.set_xticks(x)
ax3.set_xticklabels([t.replace("_", "\n") for t in bar_tasks], color="#aabbcc", fontsize=8)
ax3.legend(fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")
for t_ref, v_ref in known.items():
    ax3.axhline(v_ref, color=TCOLS.get(t_ref, "#aaa"), ls=":", lw=0.8, alpha=0.5)
    ax3.text(len(bar_tasks) - 0.05, v_ref + 0.06, t_ref,
             color=TCOLS.get(t_ref, "#aaa"), fontsize=6, ha="right", alpha=0.7)

# Panel 8: Full ordering
ax4 = fig.add_subplot(gs[3, 2])
sty(ax4, "Full β* ordering (all 6 tasks)", "β*", "")
all_tasks_ordered = sorted(all_vals.items(), key=lambda x: x[1])
y_pos  = np.arange(len(all_tasks_ordered))
vals_p = [v for _, v in all_tasks_ordered]
names  = [t for t, _ in all_tasks_ordered]
is_new = ["★ " if t in TASKS else "  " for t in names]
ax4.barh(y_pos, vals_p, color=[TCOLS.get(t, "#aabbcc") for t in names], alpha=0.8)
ax4.set_yticks(y_pos)
ax4.set_yticklabels([f"{f}{n}" for f, n in zip(is_new, names)], color="#aabbcc", fontsize=8)
ax4.set_xlabel("β*", color="#aabbcc", fontsize=7)
for i, t in enumerate(names):
    ax4.plot(PREDICTIONS[t]["pred"], i, "D", color="#ffffff", ms=5, zorder=5, alpha=0.8)
ax4.legend(handles=[plt.Line2D([0],[0], marker="D", color="#fff", ls="none", ms=5)],
           labels=["predicted β*"], fontsize=7, facecolor="#0f1525", labelcolor="#aabbcc")

verdict = "✓ ALL CONFIRMED" if all_pass and any_grok else "PARTIAL" if any_grok else "NO GROKKING"
fig.suptitle(
    f"WHY2 Structural Prediction Test v2 — {verdict}\n"
    f"orange=division  purple=affine  teal=multiply_add  "
    f"solid=WD1.0  dashed=WD0.5  ★=new tasks  ◆=predicted β*",
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
print(f"\nKey question: do division/affine/multiply_add β* land in predicted ranges?")
print(f"  → YES (all in range): β* is a structural invariant — framework predicts")
print(f"  → PARTIAL: check residuals + WD invariance table for what physics is missing")
print(f"  → NO GROKKING: increase MAX_EPOCHS further or try WD=0.2")