# ============================================================
# WHY2 BASIN B PRE-CHECK
#
# Quick diagnostic: does Basin B (β* > 15) actually appear
# at WD=0.5 for subtract? And at what epoch range?
#
# If yes → proceed with full Exp 2 (phi^6 curvature)
# If no  → Exp 2 design needs rethinking
#
# 6 seeds × WD=0.5 × subtract only
# Dense probing every 25 epochs — full β trajectory printed
# Expected runtime: ~8-12 min on free T4
# ============================================================

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"WHY2 BASIN B PRE-CHECK  |  device: {DEVICE}  |  {datetime.now():%H:%M:%S}")
print(f"Task: subtract  WD: 0.5  Seeds: 6")
print(f"Watching for Basin B (β* > 8) vs Basin A (β* < 8)")
print("="*65)

PRIME          = 97
D_MODEL        = 128
N_HEADS        = 4
D_FF           = 512
LR             = 1e-3
TRAIN_FRAC     = 0.3
PROBE_EVERY    = 25
MAX_EPOCHS     = 25000
POST_GROK      = 8000
GROK_THRESH    = 0.99
WD             = 0.5
SEEDS          = [42, 123, 7, 17, 71, 31]
BASIN_B_THRESH = 8.0
KOUT_THRESHOLDS = [0.75, 0.80, 0.83, 0.87]  # check curvature at all of these

# ── Data ──────────────────────────────────────────────────────
def make_data(prime, train_frac, seed):
    torch.manual_seed(seed)
    pairs  = [(a, b) for a in range(prime) for b in range(prime)]
    labels = [(a - b) % prime for a, b in pairs]
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

# ── Probe ─────────────────────────────────────────────────────
def probe(model, prime):
    model.eval()
    pairs = torch.tensor([[a, b, prime] for a in range(prime) for b in range(prime)],
                          device=DEVICE)
    with torch.no_grad():
        _, attn_w = model(pairs)

    w0   = attn_w[:, :, 2, 0]
    w1   = attn_w[:, :, 2, 1]
    K_in  = (w0 + w1).mean().item()
    asym  = (w0 - w1).abs().mean().item()

    E = model.embed.weight[:prime].detach().float()
    F_basis = torch.zeros(prime, prime, device=DEVICE)
    for k in range(prime):
        for n in range(prime):
            F_basis[k, n] = math.cos(2 * math.pi * k * n / prime)
    F_basis /= math.sqrt(prime)
    power = torch.stack([(F_basis[k] @ E).pow(2).sum() for k in range(prime)])
    power = power / power.sum()

    top_k = max(1, prime // 10)
    K_out  = float(power.topk(top_k).values.sum())
    H_n    = float(-(power * (power + 1e-10).log()).sum() / math.log(prime))

    beta = float('nan')
    if 1e-6 < K_out < 1-1e-6 and K_in > 1e-6:
        beta = (math.log(K_in) - H_n) / math.log(K_out)

    return {"K_in": K_in, "K_out": K_out, "H_n": H_n,
            "K_in_asym": asym, "beta": beta}

# ── Run ───────────────────────────────────────────────────────
results = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    (X_tr, Y_tr), (X_te, Y_te) = make_data(PRIME, TRAIN_FRAC, seed)
    X_tr, Y_tr = X_tr.to(DEVICE), Y_tr.to(DEVICE)
    X_te, Y_te = X_te.to(DEVICE), Y_te.to(DEVICE)

    model = Transformer(PRIME, D_MODEL, N_HEADS, D_FF).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    traj    = []
    grok_ep = None
    # Track K_out threshold crossings
    crossed = {t: False for t in KOUT_THRESHOLDS}
    crossed_beta = {t: None for t in KOUT_THRESHOLDS}
    crossed_asym = {t: None for t in KOUT_THRESHOLDS}

    print(f"\n{'─'*55}")
    print(f"SEED {seed}  (WD={WD})")
    print(f"{'─'*55}")

    for ep in range(1, MAX_EPOCHS + 1):
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

            # Check threshold crossings
            for thresh in KOUT_THRESHOLDS:
                if not crossed[thresh] and p["K_out"] >= thresh:
                    crossed[thresh] = True
                    crossed_beta[thresh] = p["beta"]
                    crossed_asym[thresh] = p["K_in_asym"]
                    print(f"  >>> K_out crossed {thresh:.2f} at ep={ep}"
                          f"  β={p['beta']:.4f}  asym={p['K_in_asym']:.4f}")

            if grok_ep is None and acc >= GROK_THRESH:
                grok_ep = ep
                print(f"  *** GROKKED ep={ep}  β={p['beta']:.4f}"
                      f"  K_out={p['K_out']:.4f}  asym={p['K_in_asym']:.4f}")

            # Progress print every 2000 epochs
            if ep % 2000 == 0:
                b_str = f"{p['beta']:.4f}" if not math.isnan(p['beta']) else "nan"
                print(f"  ep={ep:6d}  acc={acc:.4f}  β={b_str}"
                      f"  K_out={p['K_out']:.4f}  asym={p['K_in_asym']:.4f}"
                      + (" [grokked]" if grok_ep else ""))

            if grok_ep and ep >= grok_ep + POST_GROK:
                break

    # Final β* from last 40 probes
    late = [t for t in traj[-40:] if not math.isnan(t["beta"])]
    beta_star  = np.mean([t["beta"] for t in late]) if late else float('nan')
    asym_star  = np.mean([t["K_in_asym"] for t in late]) if late else float('nan')
    kout_final = np.mean([t["K_out"] for t in late]) if late else float('nan')
    basin = "B" if beta_star > BASIN_B_THRESH else ("A" if not math.isnan(beta_star) else "?")

    # β trajectory summary: show every 1000 epochs post-grok
    if grok_ep:
        post_grok_traj = [t for t in traj if t["ep"] >= grok_ep]
        print(f"\n  POST-GROK β TRAJECTORY (every ~1000 ep):")
        step = max(1, len(post_grok_traj) // 8)
        for t in post_grok_traj[::step]:
            b_str = f"{t['beta']:.4f}" if not math.isnan(t['beta']) else "nan"
            print(f"    ep={t['ep']:6d}  β={b_str}  K_out={t['K_out']:.4f}"
                  f"  asym={t['K_in_asym']:.4f}  acc={t['acc']:.4f}")

    print(f"\n  RESULT: β*={beta_star:.4f}  K_out={kout_final:.4f}"
          f"  asym*={asym_star:.4f}  basin={basin}"
          f"  grok_ep={grok_ep}")

    # Threshold crossing summary
    print(f"  K_out crossings:")
    for thresh in KOUT_THRESHOLDS:
        if crossed_beta[thresh] is not None:
            print(f"    K_out={thresh:.2f}: β={crossed_beta[thresh]:.4f}"
                  f"  asym={crossed_asym[thresh]:.4f}")
        else:
            print(f"    K_out={thresh:.2f}: NOT REACHED")

    results.append({
        "seed": seed, "wd": WD, "grok_ep": grok_ep,
        "beta_star": beta_star, "kout_final": kout_final,
        "asym_star": asym_star, "basin": basin,
        "crossed_beta": crossed_beta, "crossed_asym": crossed_asym
    })

# ── Final summary ─────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"PRE-CHECK SUMMARY  (subtract, WD={WD})")
print(f"{'='*65}")
print(f"{'Seed':>6}  {'β*':>8}  {'K_out':>6}  {'Basin':>6}  {'Grok ep':>8}  {'asym*':>7}")
for r in results:
    b = f"{r['beta_star']:.4f}" if not math.isnan(r['beta_star']) else "no_grok"
    k = f"{r['kout_final']:.4f}" if not math.isnan(r['kout_final']) else "—"
    print(f"{r['seed']:6d}  {b:>8}  {k:>6}  {r['basin']:>6}  "
          f"{str(r['grok_ep']):>8}  {r['asym_star']:.4f}")

grokked = [r for r in results if r["grok_ep"]]
basin_A = [r for r in grokked if r["basin"] == "A"]
basin_B = [r for r in grokked if r["basin"] == "B"]
no_grok = [r for r in results if not r["grok_ep"]]

print(f"\nBasin A (β* < 8): {len(basin_A)} seeds")
print(f"Basin B (β* > 8): {len(basin_B)} seeds")
print(f"No grok:          {len(no_grok)} seeds")

if basin_B:
    print(f"\nBasin B β* values: {[f'{r['beta_star']:.2f}' for r in basin_B]}")
    print(f"Basin B K_out:     {[f'{r['kout_final']:.4f}' for r in basin_B]}")

print(f"\n{'='*65}")
if basin_B:
    print(f"VERDICT: Basin B EXISTS at WD={WD} ✓")
    print(f"→ Proceed with full Exp 2 (phi^6 curvature analysis)")
    print(f"→ Confirmed threshold range: check K_out crossings above")
elif len(no_grok) >= 3:
    print(f"VERDICT: Most seeds DID NOT GROK at WD={WD}")
    print(f"→ WD=0.5 may be too weak — try WD=0.3 or extend MAX_EPOCHS")
else:
    print(f"VERDICT: All grokked seeds went Basin A at WD={WD}")
    print(f"→ Basin B may need different WD or seeds")
    print(f"→ Check bvel3 seeds that produced Basin B previously")
    print(f"→ Consider: Basin B may only appear at WD < 0.5")
print(f"{'='*65}")