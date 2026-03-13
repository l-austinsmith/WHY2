# ============================================================
# WHY2 T_eff TEST v3 — Basin Confirmation + s31 Spike Investigation
#
# Goals:
#   A) Basin confirmation: add s17 (Basin A) and s71 (Basin B)
#      from stability run to confirm β·T_smooth constants:
#        Basin A target: ~0.88
#        Basin B target: ~23.4
#
#   B) s31 spike investigation: re-run s31 with:
#      - Dense probing every DENSE_EVERY=5 epochs in ep 13500–18000
#        (the region where spikes at ep15500 and ep18000 occurred)
#      - Spike detector: flag any probe where T_smooth > SPIKE_MULT×median
#        and print full diagnostic (grad_var, v_mean, K_in, K_out, loss,
#        β, dβ/dt, raw vs smooth comparison)
#      - Track whether spikes correlate with dβ/dt sign changes
#        (re-exploration events?) or with loss jumps
#
# Seeds:
#   17  → Basin A (confirmed stable, 72-run basin stability study)
#   71  → Basin B (confirmed stable, 72-run study, same execution env as s31/s7)
#   31  → re-run with spike instrumentation (Basin A this run, B in bvel3)
#
# All three predictions re-tested on new seeds.
# Final summary table: all 5 seeds (v2 + v3) side by side.
#
# Runtime: ~2 hours on T4
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
print(f"WHY2 T_eff v3  |  device: {DEVICE}  |  {datetime.now():%Y-%m-%d %H:%M}")
print(f"Output prefix: why2_teff_v3_{TS}")
print(f"Goals: (A) basin constant confirmation  (B) s31 spike investigation")
print("="*65)

# ── Config ────────────────────────────────────────────────────
PRIME        = 97
D_MODEL      = 128
N_HEADS      = 4
D_FF         = 512
LR           = 1e-3
WD           = 1.0
TRAIN_FRAC   = 0.3
PROBE_EVERY  = 25       # standard probe interval
DENSE_EVERY  = 5        # dense probe for s31 late-epoch window
DENSE_START  = 13500    # start dense probing here (before first v2 spike at ep15500)
GROK_THRESH  = 0.99
MAX_EPOCHS   = 18000
CKPT_EVERY   = 500
EMA_TAU      = 500      # same as v2 for direct comparison
BETA_WINDOW  = 10
SPIKE_MULT   = 5.0      # flag T_smooth > SPIKE_MULT × running median post-grok
BT_WINDOW    = 50       # use last 50 probes for β·T_smooth constant (vs 20 in v2)

# Seeds:
#   17, 71  → new seeds for basin confirmation
#   31      → re-run with spike instrumentation
NEW_SEEDS    = [17, 71]
REINSPECT    = [31]
ALL_SEEDS    = NEW_SEEDS + REINSPECT

# Known v2 constants for comparison (printed in summary)
V2_CONSTANTS = {
    42: {"basin": "A", "bt_mean": 0.8783, "bt_cv": 0.27, "beta_final": 4.7837},
    31: {"basin": "A", "bt_mean": 0.8916, "bt_cv": 2.05, "beta_final": 3.8487},
     7: {"basin": "B", "bt_mean": 23.433, "bt_cv": 0.21, "beta_final": 38.607},
}

def ema_alpha():
    return PROBE_EVERY / EMA_TAU

# ── Data ──────────────────────────────────────────────────────
def make_data(seed):
    rng   = torch.Generator(); rng.manual_seed(seed)
    all_a = torch.arange(PRIME).repeat(PRIME)
    all_b = torch.arange(PRIME).repeat_interleave(PRIME)
    all_y = (all_a - all_b) % PRIME
    idx   = torch.randperm(len(all_a), generator=rng)
    n_tr  = int(len(all_a) * TRAIN_FRAC)
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

# ── Fourier probe ──────────────────────────────────────────────
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

def probe_beta(model, ap, bp):
    W  = model.embed.weight[:PRIME].detach().float()
    Wf = get_fourier() @ W
    pw = (Wf**2).mean(dim=1)
    pw = pw / (pw.sum() + 1e-12)
    pw_np = pw.cpu().numpy()
    H_n   = float(np.clip(-np.sum(pw_np * np.log(pw_np + 1e-12)) / math.log(PRIME), 0, 1))
    K_out = float(np.sort(pw_np)[-(max(1, PRIME//10)):].sum())
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
    return K_in, K_out, H_n, beta, K_in_asym

# ── T_eff probe ───────────────────────────────────────────────
def probe_teff(model, opt):
    grad_norms = []
    v_vals     = []
    for group in opt.param_groups:
        lr = group['lr']
        for p in group['params']:
            if p.grad is None: continue
            state = opt.state[p]
            if 'exp_avg_sq' not in state: continue
            grad_norms.append(p.grad.detach().float().norm().item())
            v_vals.append(state['exp_avg_sq'].detach().float().mean().item())
    if len(grad_norms) < 2: return None, None, None, None
    grad_norms = np.array(grad_norms)
    v_vals     = np.array(v_vals)
    grad_var   = float(np.var(grad_norms))
    v_mean     = float(np.mean(v_vals))
    grad_norm  = float(np.sqrt(np.mean(grad_norms**2)))
    if v_mean < 1e-30: return None, None, None, grad_norm
    return lr * grad_var / v_mean, grad_var, v_mean, grad_norm

# ── Rolling β/dt ──────────────────────────────────────────────
def compute_dbdt(history, window=BETA_WINDOW):
    betas  = [r["beta"]  for r in history[-window:] if r["beta"]  is not None]
    epochs = [r["epoch"] for r in history[-window:] if r["beta"]  is not None]
    if len(betas) < 3: return None
    x = np.array(epochs, dtype=float); y = np.array(betas, dtype=float)
    x -= x.mean()
    return float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0

# ── Running median helper ─────────────────────────────────────
def running_median(vals, new_val):
    """Return median of vals + new_val (last 50 values)."""
    combined = (vals + [new_val])[-50:]
    return float(np.median(combined)), combined

# ── Checkpoint ────────────────────────────────────────────────
def save_checkpoint(all_results, suffix=""):
    def clean(obj):
        if isinstance(obj, dict):  return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [clean(x) for x in obj]
        if isinstance(obj, float): return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 8)
        return obj
    fname = f"why2_teff_v3_{TS}{suffix}.json"
    with open(fname, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"  [ckpt → {fname}]")
    return fname

# ── Train ─────────────────────────────────────────────────────
def train_run(seed, all_results, dense_probe=False):
    """
    dense_probe=True: use DENSE_EVERY probe interval from DENSE_START onward,
    and activate spike detection in that window.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    atr, btr, ytr, ate, bte, yte = make_data(seed)
    pidx = torch.randperm(len(atr))[:256]
    ap, bp = atr[pidx], btr[pidx]

    model = GrokNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    history      = []
    grok_ep      = None
    last_ckpt    = 0
    alpha        = ema_alpha()
    T_eff_ema    = None

    # Spike detection state (post-grok only)
    postgrok_tsmooth_vals = []   # running window for median computation
    n_spikes              = 0
    spike_log             = []   # list of spike event dicts

    mode_str = f"dense ep{DENSE_START}+" if dense_probe else "standard"
    print(f"\n  [subtract s={seed}] training to ep={MAX_EPOCHS}  "
          f"(EMA τ={EMA_TAU}, α={alpha:.4f}, probe={mode_str})...")

    for epoch in range(1, MAX_EPOCHS+1):
        # Decide probe interval for this epoch
        is_dense_epoch = dense_probe and epoch >= DENSE_START
        interval       = DENSE_EVERY if is_dense_epoch else PROBE_EVERY
        do_probe       = (epoch % interval == 0)

        model.train()
        loss = F.cross_entropy(model.logits(atr, btr), ytr)
        opt.zero_grad(); loss.backward()
        T_raw, grad_var, v_mean, grad_norm = probe_teff(model, opt)
        opt.step()

        # EMA update — always at standard PROBE_EVERY cadence to keep τ consistent
        # (dense probes update EMA too, at higher frequency in DENSE window —
        #  this is intentional: we want T_smooth to respond faster there)
        if do_probe and T_raw is not None and T_raw > 0:
            if T_eff_ema is None:
                T_eff_ema = T_raw
            else:
                T_eff_ema = alpha * T_raw + (1.0 - alpha) * T_eff_ema

        if do_probe:
            model.eval()
            with torch.no_grad():
                acc_te   = (model.logits(ate, bte).argmax(-1) == yte).float().mean().item()
                loss_val = F.cross_entropy(model.logits(atr, btr), ytr).item()

            K_in, K_out, H_n, beta, K_in_asym = probe_beta(model, ap, bp)
            dbdt = compute_dbdt(history)

            beta_teff_raw    = (beta * T_raw)     if (beta is not None and T_raw    is not None) else None
            beta_teff_smooth = (beta * T_eff_ema) if (beta is not None and T_eff_ema is not None) else None

            # ── Spike detection (post-grok only) ─────────────────
            is_spike = False
            spike_ratio = None
            if grok_ep and T_eff_ema is not None and T_eff_ema > 0:
                med, postgrok_tsmooth_vals = running_median(
                    postgrok_tsmooth_vals, T_eff_ema)
                if len(postgrok_tsmooth_vals) >= 5 and T_eff_ema > SPIKE_MULT * med:
                    is_spike   = True
                    spike_ratio = T_eff_ema / med
                    n_spikes   += 1
                    spike_ev = {
                        "epoch":       epoch,
                        "T_smooth":    T_eff_ema,
                        "T_raw":       T_raw,
                        "median":      med,
                        "spike_ratio": spike_ratio,
                        "beta":        beta,
                        "dbdt":        dbdt,
                        "K_in":        K_in,
                        "K_out":       K_out,
                        "H_n":         H_n,
                        "loss":        loss_val,
                        "grad_var":    grad_var,
                        "v_mean":      v_mean,
                        "grad_norm":   grad_norm,
                        "acc_te":      acc_te,
                    }
                    spike_log.append(spike_ev)
                    print(f"  [subtract s={seed}] *** SPIKE ep={epoch:5d} | "
                          f"T_s={T_eff_ema:.3e}  T_raw={T_raw:.3e}  "
                          f"ratio={spike_ratio:.1f}×median  "
                          f"β={beta:.4f}  dβ/dt={dbdt:.6f}  "
                          f"K_out={K_out:.3f}  loss={loss_val:.4f}  "
                          f"grad_var={grad_var:.3e}  v_mean={v_mean:.3e}")
            elif grok_ep and T_eff_ema is not None:
                _, postgrok_tsmooth_vals = running_median(
                    postgrok_tsmooth_vals, T_eff_ema)

            rec = {
                "epoch":            epoch,
                "acc_te":           acc_te,
                "loss":             loss_val,
                "K_in":             K_in,
                "K_out":            K_out,
                "H_n":              H_n,
                "beta":             beta,
                "dbdt":             dbdt,
                "K_in_asym":        K_in_asym,
                "T_eff_raw":        T_raw,
                "T_eff_smooth":     T_eff_ema,
                "grad_var":         grad_var,
                "v_mean":           v_mean,
                "grad_norm":        grad_norm,
                "beta_teff_raw":    beta_teff_raw,
                "beta_teff_smooth": beta_teff_smooth,
                "is_spike":         is_spike,
                "spike_ratio":      spike_ratio,
                "post_grok":        (epoch - grok_ep) if grok_ep else None,
                "dense":            is_dense_epoch,
            }
            history.append(rec)

            if grok_ep is None and acc_te >= GROK_THRESH:
                grok_ep = epoch
                b_str = f"{beta:.4f}"      if beta      is not None else "N/A"
                r_str = f"{T_raw:.4e}"     if T_raw     is not None else "N/A"
                s_str = f"{T_eff_ema:.4e}" if T_eff_ema is not None else "N/A"
                print(f"  [subtract s={seed}] *** GROK ep={epoch} | "
                      f"β={b_str}  T_raw={r_str}  T_smooth={s_str}  K_out={K_out:.3f}")

            if epoch % 500 == 0:
                b_str  = f"{beta:.4f}"           if beta           is not None else "N/A"
                r_str  = f"{T_raw:.3e}"          if T_raw          is not None else "N/A"
                s_str  = f"{T_eff_ema:.3e}"      if T_eff_ema      is not None else "N/A"
                bs_str = f"{beta_teff_smooth:.4f}" if beta_teff_smooth is not None else "N/A"
                d_str  = f"{dbdt:.6f}"           if dbdt           is not None else "N/A"
                sp_str = f"  [spikes={n_spikes}]" if dense_probe else ""
                print(f"  [subtract s={seed}] ep={epoch:5d} | "
                      f"β={b_str}  T_raw={r_str}  T_smooth={s_str}  "
                      f"β·T_s={bs_str}  dβ/dt={d_str}{sp_str}")

            if epoch - last_ckpt >= CKPT_EVERY:
                all_results[f"subtract_s{seed}"] = {
                    "seed": seed, "grok_ep": grok_ep,
                    "ema_tau": EMA_TAU, "ema_alpha": alpha,
                    "dense_probe": dense_probe,
                    "spike_log": spike_log,
                    "n_spikes": n_spikes,
                    "history": history}
                save_checkpoint(all_results, f"_ckpt{epoch}")
                last_ckpt = epoch

    # Final
    if history:
        r = history[-1]
        b_str  = f"{r['beta']:.4f}"             if r['beta']          is not None else "N/A"
        s_str  = f"{r['T_eff_smooth']:.3e}"     if r['T_eff_smooth']  is not None else "N/A"
        bs_str = f"{r['beta_teff_smooth']:.4f}" if r['beta_teff_smooth'] is not None else "N/A"
        print(f"  [subtract s={seed}] FINAL | β={b_str}  T_smooth={s_str}  "
              f"β·T_smooth={bs_str}  total_spikes={n_spikes}")

    if dense_probe and spike_log:
        print(f"\n  [subtract s={seed}] SPIKE SUMMARY ({n_spikes} spikes detected):")
        for ev in spike_log:
            sign_dbdt = "↑" if (ev['dbdt'] or 0) > 0 else "↓"
            print(f"    ep={ev['epoch']:5d}  T_s={ev['T_smooth']:.3e}  "
                  f"ratio={ev['spike_ratio']:.1f}×  "
                  f"β={ev['beta']:.3f} ({sign_dbdt}dβ/dt={ev['dbdt']:.5f})  "
                  f"K_out={ev['K_out']:.3f}  loss={ev['loss']:.4f}  "
                  f"grad_var={ev['grad_var']:.3e}  v_mean={ev['v_mean']:.3e}")
        # Check: do spikes correlate with dβ/dt sign changes?
        dbdt_at_spikes = [ev['dbdt'] for ev in spike_log if ev['dbdt'] is not None]
        if dbdt_at_spikes:
            print(f"    dβ/dt at spikes: mean={np.mean(dbdt_at_spikes):.5f}  "
                  f"std={np.std(dbdt_at_spikes):.5f}  "
                  f"pos={sum(1 for x in dbdt_at_spikes if x>0)}/{len(dbdt_at_spikes)}")
        # Check: loss at spikes vs overall post-grok mean
        loss_at_spikes = [ev['loss'] for ev in spike_log]
        postgrok_losses = [r['loss'] for r in history
                           if r.get('post_grok') and r['loss'] is not None]
        if loss_at_spikes and postgrok_losses:
            print(f"    loss at spikes: {np.mean(loss_at_spikes):.4f} ± {np.std(loss_at_spikes):.4f}")
            print(f"    loss post-grok overall: {np.mean(postgrok_losses):.4f} ± {np.std(postgrok_losses):.4f}")
        # grad_var at spikes
        gv_at_spikes = [ev['grad_var'] for ev in spike_log if ev['grad_var'] is not None]
        postgrok_gv  = [r['grad_var'] for r in history
                        if r.get('post_grok') and r['grad_var'] is not None]
        if gv_at_spikes and postgrok_gv:
            print(f"    grad_var at spikes: {np.mean(gv_at_spikes):.3e} ± {np.std(gv_at_spikes):.3e}")
            print(f"    grad_var post-grok overall: {np.mean(postgrok_gv):.3e} ± {np.std(postgrok_gv):.3e}")

    return grok_ep, history, spike_log

# ── Main ──────────────────────────────────────────────────────
all_results = {}
try:
    # New seeds — standard probing
    for seed in NEW_SEEDS:
        grok_ep, history, _ = train_run(seed, all_results, dense_probe=False)
        all_results[f"subtract_s{seed}"] = {
            "seed": seed, "grok_ep": grok_ep,
            "ema_tau": EMA_TAU, "ema_alpha": ema_alpha(),
            "dense_probe": False, "spike_log": [], "n_spikes": 0,
            "history": history}
        save_checkpoint(all_results)

    # s31 — dense + spike instrumentation
    for seed in REINSPECT:
        grok_ep, history, spike_log = train_run(seed, all_results, dense_probe=True)
        all_results[f"subtract_s{seed}"] = {
            "seed": seed, "grok_ep": grok_ep,
            "ema_tau": EMA_TAU, "ema_alpha": ema_alpha(),
            "dense_probe": True, "spike_log": spike_log,
            "n_spikes": len(spike_log),
            "history": history}
        save_checkpoint(all_results)

except KeyboardInterrupt:
    print("\n[Interrupted — saving partial results...]")
    save_checkpoint(all_results, "_interrupted")

# ── Analysis ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print("T_eff v3 ANALYSIS")
print(f"{'='*65}")
print(f"EMA τ={EMA_TAU} epochs  α={ema_alpha():.4f}")
print(f"β·T_smooth window: last {BT_WINDOW} post-grok probes\n")

results_v3 = {}

for seed in ALL_SEEDS:
    key = f"subtract_s{seed}"
    if key not in all_results:
        print(f"  s={seed}: NO DATA"); continue

    run     = all_results[key]
    hist    = run["history"]
    grok_ep = run["grok_ep"]
    if not hist: continue

    valid_s = [r for r in hist
               if r.get("T_eff_smooth") is not None
               and r["T_eff_smooth"] > 0
               and r.get("beta") is not None]
    if len(valid_s) < 5:
        print(f"  s={seed}: insufficient data"); continue

    pre_s  = [r for r in valid_s if grok_ep and r["epoch"] < grok_ep]
    post_s = [r for r in valid_s if grok_ep and r["epoch"] > grok_ep]

    T_pre  = np.mean([r["T_eff_smooth"] for r in pre_s])  if pre_s  else float('nan')
    T_post = np.mean([r["T_eff_smooth"] for r in post_s]) if post_s else float('nan')
    T_grok = next((r["T_eff_smooth"] for r in valid_s
                   if grok_ep and r["epoch"] >= grok_ep), None)

    # β·T_smooth constant — last BT_WINDOW post-grok probes, excluding spikes
    post_nospike = [r for r in post_s if not r.get("is_spike", False)]
    bt_clean = [r["beta_teff_smooth"] for r in post_nospike[-BT_WINDOW:]
                if r.get("beta_teff_smooth") is not None]
    bt_all   = [r["beta_teff_smooth"] for r in post_s[-BT_WINDOW:]
                if r.get("beta_teff_smooth") is not None]

    bt_clean_mean = np.mean(bt_clean) if bt_clean else float('nan')
    bt_clean_std  = np.std(bt_clean)  if bt_clean else float('nan')
    bt_clean_cv   = bt_clean_std / abs(bt_clean_mean) if (bt_clean and abs(bt_clean_mean) > 1e-12) else float('nan')
    bt_all_mean   = np.mean(bt_all)   if bt_all   else float('nan')
    bt_all_cv     = np.std(bt_all) / abs(bt_all_mean) if (bt_all and abs(bt_all_mean) > 1e-12) else float('nan')

    # corr(β, 1/T_smooth) post-grok
    if len(post_s) >= 5:
        betas_p = [r["beta"]         for r in post_s if r["beta"] is not None]
        tsmooth = [r["T_eff_smooth"]  for r in post_s
                   if r["T_eff_smooth"] is not None and r["T_eff_smooth"] > 0]
        n_c = min(len(betas_p), len(tsmooth))
        corr_s = float(np.corrcoef(betas_p[:n_c], [1/t for t in tsmooth[:n_c]])[0,1]) \
                 if n_c >= 5 else float('nan')
    else:
        corr_s = float('nan')

    b_final     = hist[-1]["beta"]
    basin_label = "A" if (b_final is not None and b_final < 10) else "B"
    n_spikes    = run.get("n_spikes", 0)

    print(f"  s={seed} (Basin {basin_label}):  β_final={b_final:.4f}  grok_ep={grok_ep}"
          if b_final else f"  s={seed}: β N/A")
    print(f"    T_smooth  pre={T_pre:.3e}  post={T_post:.3e}  @ grok={T_grok:.3e}"
          if T_grok else f"    T_smooth  pre={T_pre:.3e}  post={T_post:.3e}")
    print(f"    β·T_smooth (last {BT_WINDOW}, excl spikes): {bt_clean_mean:.4f} ± {bt_clean_std:.4f}  CV={bt_clean_cv:.2f}"
          if not math.isnan(bt_clean_mean) else "    β·T_smooth (clean): N/A")
    print(f"    β·T_smooth (last {BT_WINDOW}, all):         {bt_all_mean:.4f}  CV={bt_all_cv:.2f}"
          if not math.isnan(bt_all_mean) else "    β·T_smooth (all): N/A")
    print(f"    corr(β, 1/T_smooth) = {corr_s:.4f}"
          if not math.isnan(corr_s) else "    corr: N/A")
    if n_spikes > 0:
        print(f"    T_smooth spikes detected: {n_spikes}")
    print()

    results_v3[f"s{seed}"] = {
        "seed": seed, "basin": basin_label, "beta_final": b_final,
        "grok_ep": grok_ep, "T_pre": T_pre, "T_post": T_post, "T_grok": T_grok,
        "bt_clean_mean": bt_clean_mean, "bt_clean_std": bt_clean_std, "bt_clean_cv": bt_clean_cv,
        "bt_all_mean": bt_all_mean, "bt_all_cv": bt_all_cv,
        "corr_smooth": corr_s, "n_spikes": n_spikes,
    }

# ── Combined summary table: v2 + v3 ──────────────────────────
print(f"\n{'='*65}")
print("COMBINED SUMMARY — ALL 5 SEEDS (v2 + v3)")
print(f"{'='*65}")
print(f"{'Seed':>6}  {'Basin':>6}  {'β_final':>10}  {'β·T_s mean':>12}  {'CV':>6}  {'Source':>6}")
print("-"*55)

# v2 seeds
for seed, info in V2_CONSTANTS.items():
    print(f"  s{seed:2d}    {info['basin']:>6}  {info['beta_final']:>10.4f}  "
          f"{info['bt_mean']:>12.4f}  {info['bt_cv']:>6.2f}    v2")

# v3 seeds
for skey, info in results_v3.items():
    seed = info['seed']
    b    = info['bt_clean_mean']
    cv   = info['bt_clean_cv']
    bf   = info['beta_final']
    tag  = "(dense)" if seed in REINSPECT else ""
    b_str  = f"{b:.4f}"  if not math.isnan(b)  else "N/A"
    cv_str = f"{cv:.2f}" if not math.isnan(cv) else "N/A"
    bf_str = f"{bf:.4f}" if bf is not None else "N/A"
    print(f"  s{seed:2d}    {info['basin']:>6}  {bf_str:>10}  "
          f"{b_str:>12}  {cv_str:>6}    v3 {tag}")

print()

# Basin constant estimates
all_A = ([V2_CONSTANTS[s]["bt_mean"] for s in V2_CONSTANTS if V2_CONSTANTS[s]["basin"]=="A"] +
         [results_v3[k]["bt_clean_mean"] for k in results_v3
          if results_v3[k]["basin"]=="A" and not math.isnan(results_v3[k]["bt_clean_mean"])])
all_B = ([V2_CONSTANTS[s]["bt_mean"] for s in V2_CONSTANTS if V2_CONSTANTS[s]["basin"]=="B"] +
         [results_v3[k]["bt_clean_mean"] for k in results_v3
          if results_v3[k]["basin"]=="B" and not math.isnan(results_v3[k]["bt_clean_mean"])])

print("BASIN CONSTANT ESTIMATES (β·T_smooth, all seeds combined):")
if all_A:
    print(f"  Basin A: {np.mean(all_A):.4f} ± {np.std(all_A):.4f}  (n={len(all_A)})")
if all_B:
    print(f"  Basin B: {np.mean(all_B):.4f} ± {np.std(all_B):.4f}  (n={len(all_B)})")
if all_A and all_B:
    ratio = np.mean(all_B) / np.mean(all_A) if np.mean(all_A) > 0 else float('nan')
    print(f"  Ratio B/A: {ratio:.1f}×")

# Prediction 2 check
basin_A_v3 = [results_v3[k] for k in results_v3 if results_v3[k]["basin"]=="A"
              and not math.isnan(results_v3[k].get("T_post", float('nan')))]
basin_B_v3 = [results_v3[k] for k in results_v3 if results_v3[k]["basin"]=="B"
              and not math.isnan(results_v3[k].get("T_post", float('nan')))]
if basin_A_v3 and basin_B_v3:
    T_A = np.mean([v["T_post"] for v in basin_A_v3])
    T_B = np.mean([v["T_post"] for v in basin_B_v3])
    print(f"\nPREDICTION 2 (v3 seeds): Basin A T_post={T_A:.3e}  Basin B T_post={T_B:.3e}  "
          f"{'CONFIRMED' if T_B > T_A else 'NOT CONFIRMED'}")

# ── Plots ─────────────────────────────────────────────────────
COLORS = {17: "#2196F3", 71: "#F44336", 31: "#9C27B0"}
LABELS = {17: "s17 (Basin A)", 71: "s71 (Basin B)", 31: "s31 (A/dense)"}

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)
fig.suptitle(
    f"WHY2 — T_eff v3: Basin Confirmation + s31 Spike Investigation\n"
    f"Seeds: {ALL_SEEDS}  |  {TS}",
    fontsize=12, y=0.98)
axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

def ep_series(hist, field, require_pos=False):
    eps, vals = [], []
    for r in hist:
        v = r.get(field)
        if v is None: continue
        if require_pos and v <= 0: continue
        eps.append(r["epoch"]); vals.append(v)
    return eps, vals

for seed in ALL_SEEDS:
    key = f"subtract_s{seed}"
    if key not in all_results: continue
    hist    = all_results[key]["history"]
    grok_ep = all_results[key]["grok_ep"]
    spike_log = all_results[key].get("spike_log", [])
    c = COLORS[seed]; lbl = LABELS[seed]

    # [0,0] β over time
    ep, bv = ep_series(hist, "beta")
    if ep: axes[0][0].plot(ep, bv, color=c, label=lbl, lw=1.5)
    if grok_ep: axes[0][0].axvline(grok_ep, color=c, lw=0.8, ls="--", alpha=0.5)

    # [0,1] T_smooth over time — mark spikes
    ep_r, tv_r = ep_series(hist, "T_eff_raw", require_pos=True)
    ep_s, tv_s = ep_series(hist, "T_eff_smooth", require_pos=True)
    if ep_r: axes[0][1].semilogy(ep_r, tv_r, color=c, lw=0.5, alpha=0.3)
    if ep_s: axes[0][1].semilogy(ep_s, tv_s, color=c, lw=2.0, alpha=0.9, label=lbl)
    if grok_ep: axes[0][1].axvline(grok_ep, color=c, lw=0.8, ls="--", alpha=0.5)
    # Mark spike epochs
    for sp in spike_log:
        axes[0][1].axvline(sp["epoch"], color=c, lw=1.0, ls=":", alpha=0.7)
        axes[0][1].scatter([sp["epoch"]], [sp["T_smooth"]], color=c, s=40,
                           marker="^", zorder=5)

    # [0,2] β · T_smooth — mark spikes
    ep, bt = ep_series(hist, "beta_teff_smooth")
    if ep: axes[0][2].plot(ep, bt, color=c, label=lbl, lw=1.5)
    if grok_ep: axes[0][2].axvline(grok_ep, color=c, lw=0.8, ls="--", alpha=0.5)
    for sp in spike_log:
        if sp.get("beta") and sp.get("T_smooth"):
            axes[0][2].scatter([sp["epoch"]], [sp["beta"] * sp["T_smooth"]],
                               color=c, s=40, marker="^", zorder=5)

    # [1,0] β vs 1/T_smooth scatter (post-grok, excl spikes)
    post_clean = [r for r in hist
                  if r.get("T_eff_smooth") and r.get("beta")
                  and r["T_eff_smooth"] > 0 and grok_ep and r["epoch"] > grok_ep
                  and not r.get("is_spike", False)]
    if len(post_clean) >= 5:
        inv_ts = [1/r["T_eff_smooth"] for r in post_clean]
        betas  = [r["beta"]           for r in post_clean]
        axes[1][0].scatter(inv_ts, betas, color=c, s=10, alpha=0.45, label=lbl)
        if len(inv_ts) >= 10:
            x_arr = np.array(inv_ts); y_arr = np.array(betas)
            x_lim = np.percentile(x_arr, [2, 98])
            mask  = (x_arr >= x_lim[0]) & (x_arr <= x_lim[1])
            if mask.sum() >= 5:
                m, b_ = np.polyfit(x_arr[mask], y_arr[mask], 1)
                xs    = np.linspace(x_lim[0], x_lim[1], 50)
                axes[1][0].plot(xs, m*xs + b_, color=c, lw=1.5, ls="--", alpha=0.7)

    # [1,1] K_out over time
    ep, kv = ep_series(hist, "K_out")
    if ep: axes[1][1].plot(ep, kv, color=c, label=lbl, lw=1.5)
    if grok_ep: axes[1][1].axvline(grok_ep, color=c, lw=0.8, ls="--", alpha=0.5)

    # [1,2] dense-probe zoom for s31: T_smooth from ep13500 onward
    if seed == 31:
        dense_ep = [r["epoch"] for r in hist
                    if r["epoch"] >= DENSE_START and r.get("T_eff_smooth") and r["T_eff_smooth"] > 0]
        dense_ts = [r["T_eff_smooth"] for r in hist
                    if r["epoch"] >= DENSE_START and r.get("T_eff_smooth") and r["T_eff_smooth"] > 0]
        dense_bt = [r["beta_teff_smooth"] for r in hist
                    if r["epoch"] >= DENSE_START and r.get("beta_teff_smooth") is not None]
        dense_ep_bt = [r["epoch"] for r in hist
                       if r["epoch"] >= DENSE_START and r.get("beta_teff_smooth") is not None]
        if dense_ep:
            axes[1][2].semilogy(dense_ep, dense_ts, color=c, lw=1.5, label="T_smooth (dense)")
            for sp in spike_log:
                if sp["epoch"] >= DENSE_START:
                    axes[1][2].scatter([sp["epoch"]], [sp["T_smooth"]],
                                       color="red", s=60, marker="^", zorder=5,
                                       label="spike" if sp == spike_log[0] else "")
    else:
        # For non-s31 seeds: gradient variance
        ep, gv = ep_series(hist, "grad_var", require_pos=True)
        if ep: axes[1][2].semilogy(ep, gv, color=c, label=lbl, lw=1.5)
        if grok_ep: axes[1][2].axvline(grok_ep, color=c, lw=0.8, ls="--", alpha=0.5)

    # [2,0] T_smooth post-grok zoom (grok-1000 onward)
    if ep_s and grok_ep:
        post_ep = [e for e in ep_s if e >= max(0, grok_ep - 500)]
        post_tv = [v for e, v in zip(ep_s, tv_s) if e >= max(0, grok_ep - 500)]
        if post_ep:
            axes[2][0].semilogy(post_ep, post_tv, color=c, label=lbl, lw=1.8)
            axes[2][0].axvline(grok_ep, color=c, lw=1.0, ls="--", alpha=0.6)

    # [2,1] Grokking curve
    ep, av = ep_series(hist, "acc_te")
    if ep: axes[2][1].plot(ep, av, color=c, label=lbl, lw=1.5)
    if grok_ep: axes[2][1].axvline(grok_ep, color=c, lw=0.8, ls="--", alpha=0.5)

    # [2,2] β·T_smooth post-grok only (log scale) — clean convergence view
    post_bt_ep = [r["epoch"]           for r in hist
                  if grok_ep and r["epoch"] > grok_ep
                  and r.get("beta_teff_smooth") is not None
                  and r["beta_teff_smooth"] > 0]
    post_bt_v  = [r["beta_teff_smooth"] for r in hist
                  if grok_ep and r["epoch"] > grok_ep
                  and r.get("beta_teff_smooth") is not None
                  and r["beta_teff_smooth"] > 0]
    if post_bt_ep:
        axes[2][2].semilogy(post_bt_ep, post_bt_v, color=c, label=lbl, lw=1.5, alpha=0.8)
        # Draw target line for this seed's basin constant
        if seed in results_v3 and not math.isnan(results_v3[f"s{seed}"]["bt_clean_mean"]):
            tgt = results_v3[f"s{seed}"]["bt_clean_mean"]
            axes[2][2].axhline(tgt, color=c, lw=1.0, ls=":", alpha=0.5)
    if grok_ep: axes[2][2].axvline(grok_ep, color=c, lw=0.8, ls="--", alpha=0.5)

# Labels
axes[0][0].set(xlabel="Epoch", ylabel="β", title="β over Time")
axes[0][1].set(xlabel="Epoch", ylabel="T_smooth (log)",
               title="T_smooth over Time\n[▲ = spike events, dotted = spike epoch]")
axes[0][2].set(xlabel="Epoch", ylabel="β · T_smooth",
               title="β · T_smooth Product")
axes[1][0].set(xlabel="1 / T_smooth", ylabel="β",
               title="β vs 1/T_smooth (post-grok, excl spikes)")
axes[1][1].set(xlabel="Epoch", ylabel="K_out", title="K_out over Time")
axes[1][2].set(xlabel="Epoch", ylabel="",
               title="s31 Dense Zoom: T_smooth ep13500+\n[▲ red = spike]")
axes[2][0].set(xlabel="Epoch", ylabel="T_smooth (log)",
               title="T_smooth — Post-Grok Zoom (grok-500+)")
axes[2][1].set(xlabel="Epoch", ylabel="Test Accuracy", title="Grokking Curve")
axes[2][2].set(xlabel="Epoch", ylabel="β·T_smooth (log)",
               title="β·T_smooth Post-Grok (log)\n[dotted = basin constant]")

for row in axes:
    for ax in row:
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

fig_fname = f"why2_teff_v3_{TS}.png"
fig.savefig(fig_fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n[Figure → {fig_fname}]")

save_checkpoint(all_results)
summary_fname = f"why2_teff_v3_{TS}_summary.json"
def clean_summary(obj):
    if isinstance(obj, dict):  return {str(k): clean_summary(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [clean_summary(x) for x in obj]
    if isinstance(obj, float): return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 8)
    return obj
with open(summary_fname, "w") as f:
    json.dump(clean_summary({"v3_results": results_v3, "v2_constants": V2_CONSTANTS}), f, indent=2)
print(f"[Summary → {summary_fname}]")

print(f"\n{'='*65}")
print("DONE")
print(f"{'='*65}")
print("Key outputs:")
print(f"  {fig_fname}")
print(f"  {summary_fname}")
print(f"  why2_teff_v3_{TS}.json")
print()
print("COPY THESE TO DRIVE:")
print(f"  !cp why2_teff_v3_{TS}*.* /content/drive/MyDrive/")