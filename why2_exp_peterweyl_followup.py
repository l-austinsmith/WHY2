# ============================================================
# WHY² EXPERIMENT: PETER-WEYL FOLLOW-UP
#       D_4 vs Q_8 — Can they grok with sufficient data?
#
# MOTIVATION:
#   The Peter-Weyl probe (why2_exp_peterweyl.py) showed that
#   D_4 and Q_8 produce elevated β^PW ≈ 1.2–4.1 during training
#   vs abelian baseline β^PW ≈ 0.5–0.9 — the 2D irrep subspace
#   is activated and carrying real spectral weight. However, no
#   group grokked because TRAIN_FRAC=0.5 leaves only 32 training
#   pairs for order-8 groups — data starvation, not probe failure.
#
# THIS EXPERIMENT:
#   TRAIN_FRAC = 0.8  → 51 training pairs for order-8 groups
#   MAX_EPOCHS = 50000  → longer to allow delayed grokking
#   Groups: D_4, Q_8, Z_8 (abelian control, same order)
#
# PREDICTIONS:
#   1. D_4 and/or Q_8 grok → β^PW* exists and is finite
#   2. β^PW*(D_4) > β^PW*(Q_8) — D_4 costs more (non-normal subgroups)
#   3. β^PW*(non-abelian) > β^PW*(Z_8) — abelian baseline
#   4. β^PW* ≠ β_scalar* for non-abelian groups
#      (scalar probe underestimates the true spectral cost)
#
# ALSO INCLUDES Z_16 as the reliable abelian reference
# (Z_16 grokked 3/6 seeds in both previous experiments)
#
# Expected runtime: ~60-90 min on free Colab T4
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
PREFIX = f"why2_pw_followup_{TS}"

print(f"WHY² PETER-WEYL FOLLOW-UP  |  device: {DEVICE}")
print(f"Question: Do D_4 and Q_8 grok with sufficient data?")
print(f"         Does β^PW*(D_4) ≠ β^PW*(Q_8)?")
print("=" * 65)

# ── Hyperparameters ────────────────────────────────────────────
D_MODEL          = 128
N_HEADS          = 4
D_FF             = 512
LR               = 1e-3
TRAIN_FRAC       = 0.8      # KEY CHANGE: was 0.5, now 0.8
WD               = 1.0
MAX_EPOCHS       = 50000    # KEY CHANGE: was 25000, now 50000
PROBE_EVERY      = 100
GROK_THRESH      = 0.99
POST_GROK_SETTLE = 3000
SEEDS            = [42, 123, 7, 17, 71, 31]


# ── Group definitions ──────────────────────────────────────────

def make_cyclic_table(n):
    return [[(a + b) % n for b in range(n)] for a in range(n)]

def make_dihedral_table(n):
    order = 2 * n
    table = [[0] * order for _ in range(order)]
    for a in range(order):
        for b in range(order):
            if a < n and b < n:
                table[a][b] = (a + b) % n
            elif a < n and b >= n:
                table[a][b] = n + (b - n - a) % n
            elif a >= n and b < n:
                table[a][b] = n + (a - n + b) % n
            else:
                table[a][b] = (b - a) % n
    return table

def make_q8_table():
    return [
        [0,1,2,3,4,5,6,7],[1,0,3,2,5,4,7,6],[2,3,1,0,6,7,5,4],[3,2,0,1,7,6,4,5],
        [4,5,7,6,1,0,2,3],[5,4,6,7,0,1,3,2],[6,7,4,5,3,2,1,0],[7,6,5,4,2,3,0,1],
    ]

def verify_group(table, name):
    n = len(table)
    for e in range(n):
        if all(table[e][x] == x and table[x][e] == x for x in range(n)):
            abelian = all(table[a][b]==table[b][a] for a in range(n) for b in range(n))
            print(f"  {name}: order={n}, identity={e}, abelian={abelian}")
            return
    print(f"  WARNING: {name} has no identity!")


# ── Irrep tables ───────────────────────────────────────────────

def get_irreps(group_name, order):
    irreps = []

    if group_name.startswith("Z_"):
        n = order
        for k in range(n):
            mats = [np.array([[np.exp(2j * math.pi * k * g / n)]], dtype=complex)
                    for g in range(n)]
            irreps.append({"dim": 1, "matrices": mats, "label": f"chi_{k}"})

    elif group_name.startswith("D_"):
        n = order // 2
        irreps.append({"dim":1, "label":"trivial",
                       "matrices":[np.array([[1+0j]]) for _ in range(order)]})
        mats = [np.array([[1+0j]]) if g < n else np.array([[-1+0j]])
                for g in range(order)]
        irreps.append({"dim":1, "matrices":mats, "label":"det"})
        if n % 2 == 0:
            mats = [np.array([[float((-1)**g)+0j]]) if g<n
                    else np.array([[float((-1)**(g-n))+0j]])
                    for g in range(order)]
            irreps.append({"dim":1, "matrices":mats, "label":"alt"})
            mats = [np.array([[float((-1)**g)+0j]]) if g<n
                    else np.array([[-float((-1)**(g-n))+0j]])
                    for g in range(order)]
            irreps.append({"dim":1, "matrices":mats, "label":"alt_det"})
        for k in range(1, n // 2):
            ang = 2 * math.pi * k / n
            mats = []
            for g in range(order):
                j = g if g < n else g - n
                c, s = math.cos(ang * j), math.sin(ang * j)
                if g < n:
                    mats.append(np.array([[c,-s],[s,c]], dtype=complex))
                else:
                    mats.append(np.array([[c,s],[s,-c]], dtype=complex))
            irreps.append({"dim":2, "matrices":mats, "label":f"rho_{k}_2D"})

    elif group_name == "Q_8":
        for chars, lbl in [
            ([1,1,1,1,1,1,1,1], "trivial"),
            ([1,1,1,1,-1,-1,-1,-1], "chi_j"),
            ([1,1,-1,-1,1,1,-1,-1], "chi_i"),
            ([1,1,-1,-1,-1,-1,1,1], "chi_k"),
        ]:
            irreps.append({"dim":1, "label":lbl,
                           "matrices":[np.array([[complex(c)]]) for c in chars]})
        I  = np.eye(2, dtype=complex)
        sx = np.array([[0,1],[1,0]], dtype=complex)
        sy = np.array([[0,-1j],[1j,0]], dtype=complex)
        sz = np.array([[1,0],[0,-1]], dtype=complex)
        irreps.append({"dim":2, "label":"Pauli_2D",
                       "matrices":[I,-I, 1j*sz,-1j*sz, 1j*sy,-1j*sy, 1j*sx,-1j*sx]})
    return irreps


# ── Peter-Weyl spectral power ──────────────────────────────────

def peter_weyl_power(E_np, irreps, order):
    """
    P_rho = (|G|/d_rho) sum_{ij} ||A_{rho,ij}||^2
    A_{rho,ij} = (d_rho/|G|) sum_g E(g) rho_{ij}(g)*
    Parseval: sum_rho P_rho = ||E||_F^2  (verified)
    """
    powers = []
    for rho in irreps:
        d = rho["dim"]; mats = rho["matrices"]; csq = 0.0
        for i in range(d):
            for j in range(d):
                rc  = np.array([np.conj(mats[g][i,j]) for g in range(order)])
                A   = (d / order) * (E_np.T @ rc)
                csq += float(np.real(np.dot(np.conj(A), A)))
        powers.append((order / d) * csq)
    p = np.array(powers, dtype=float)
    return p, p / (p.sum() + 1e-10)


# ── Peter-Weyl WHY² probe ──────────────────────────────────────

def probe_peterweyl(model, order, irreps):
    model.eval()
    pairs = torch.tensor(
        [[a, b, order] for a in range(order) for b in range(order)],
        device=DEVICE)
    with torch.no_grad():
        _, attn_w = model(pairs)

    # K_in
    w_ops = attn_w[:, :, 2, 0] + attn_w[:, :, 2, 1]
    K_in  = float(w_ops.mean())
    asym  = float((attn_w[:, :, 2, 0] - attn_w[:, :, 2, 1]).abs().mean())
    beta2 = asym / K_in if K_in > 1e-6 else float('nan')

    # Peter-Weyl K_out
    E        = model.embed.weight[:order].detach().cpu().float().numpy()
    n_irreps = len(irreps)
    _, p_norm = peter_weyl_power(E, irreps, order)

    top_k    = max(1, n_irreps // 10)
    K_out_pw = float(np.sort(p_norm)[::-1][:top_k].sum())
    eps      = 1e-10
    H_pw     = float(-(p_norm * np.log(p_norm + eps)).sum()
                     / math.log(n_irreps + 1))
    active   = int((p_norm > 1.0 / n_irreps).sum())

    beta_pw = float('nan')
    if 1e-6 < K_out_pw < 1 - 1e-6 and K_in > 1e-6:
        try:
            beta_pw = (math.log(K_in) - H_pw) / math.log(K_out_pw)
        except Exception:
            pass

    # Scalar DFT beta (for comparison)
    E_t = torch.tensor(E, device=DEVICE)
    n   = order
    F_b = torch.zeros(n, n, device=DEVICE)
    for k in range(n):
        for m in range(n):
            F_b[k, m] = math.cos(2 * math.pi * k * m / n)
    F_b /= math.sqrt(n)
    ps  = torch.stack([(F_b[k] @ E_t).pow(2).sum() for k in range(n)])
    ps  = ps / (ps.sum() + 1e-10)
    Kos = float(ps.topk(max(1, n // 10)).values.sum())
    Hs  = float(-(ps * (ps + 1e-10).log()).sum() / math.log(n))
    beta_s = float('nan')
    if 1e-6 < Kos < 1 - 1e-6 and K_in > 1e-6:
        try:
            beta_s = (math.log(K_in) - Hs) / math.log(Kos)
        except Exception:
            pass

    # Per-irrep spectrum for analysis
    return {
        "K_in": K_in, "K_out_pw": K_out_pw, "H_pw": H_pw,
        "beta_pw": beta_pw, "beta2": beta2,
        "K_out_scalar": Kos, "H_scalar": Hs, "beta_scalar": beta_s,
        "active_irreps": active, "n_irreps": n_irreps,
        "pw_spectrum": p_norm.tolist(),
    }


# ── Group registry ──────────────────────────────────────────────

GROUPS = {
    "Z_8":  {"table": make_cyclic_table(8),   "order": 8,  "abelian": True},
    "Z_16": {"table": make_cyclic_table(16),  "order": 16, "abelian": True},
    "D_4":  {"table": make_dihedral_table(4), "order": 8,  "abelian": False},
    "Q_8":  {"table": make_q8_table(),        "order": 8,  "abelian": False},
}

print("\nGroup verification:")
for name, info in GROUPS.items():
    verify_group(info["table"], name)

IRREPS = {}
print("\nIrrep tables:")
for name, info in GROUPS.items():
    irr    = get_irreps(name, info["order"])
    IRREPS[name] = irr
    dims   = [r["dim"] for r in irr]
    sum_d2 = sum(d**2 for d in dims)
    print(f"  {name}: dims={dims}, sum_d^2={sum_d2} "
          f"[{'OK' if sum_d2==info['order'] else 'FAIL'}]")

print(f"\nKey change: TRAIN_FRAC={TRAIN_FRAC}, MAX_EPOCHS={MAX_EPOCHS}")
print(f"  Z_8/D_4/Q_8 training pairs: {int(8*8*TRAIN_FRAC)} (was {int(8*8*0.5)})")
print(f"  Z_16 training pairs: {int(16*16*TRAIN_FRAC)} (was {int(16*16*0.5)})")


# ── Data generation ────────────────────────────────────────────

def make_group_data(table, order, train_frac, seed):
    torch.manual_seed(seed)
    pairs  = [(a, b) for a in range(order) for b in range(order)]
    labels = [table[a][b] for a, b in pairs]
    idx    = torch.randperm(len(pairs))
    n_tr   = int(len(pairs) * train_frac)
    tr, te = idx[:n_tr], idx[n_tr:]
    def mk(s):
        X = torch.tensor([[pairs[i][0], pairs[i][1], order] for i in s], device=DEVICE)
        Y = torch.tensor([labels[i] for i in s], device=DEVICE)
        return X, Y
    return mk(tr), mk(te)


# ── Transformer ────────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(self, vocab_size, output_size, d_model, n_heads, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos   = nn.Embedding(3, d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff1   = nn.Linear(d_model, d_ff)
        self.ff2   = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.out   = nn.Linear(d_model, output_size)

    def forward(self, x):
        pos = torch.arange(3, device=x.device).unsqueeze(0)
        h   = self.embed(x) + self.pos(pos)
        h2, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        h = self.norm1(h + h2)
        h = self.norm2(h + self.ff2(F.relu(self.ff1(h))))
        return self.out(h[:, 2, :]), w


# ── Training loop ──────────────────────────────────────────────

def train_group(group_name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    info   = GROUPS[group_name]
    order  = info["order"]; table = info["table"]; irreps = IRREPS[group_name]
    (X_tr, Y_tr), (X_te, Y_te) = make_group_data(table, order, TRAIN_FRAC, seed)

    model = Transformer(order+1, order, D_MODEL, N_HEADS, D_FF).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    history = []; grokked = False; grok_ep = None; post_grok_count = 0
    beta_pw_post = []; beta_s_post = []; beta2_post = []

    for ep in range(MAX_EPOCHS):
        model.train()
        logits, _ = model(X_tr)
        loss = F.cross_entropy(logits, Y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % PROBE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = (model(X_tr)[0].argmax(1) == Y_tr).float().mean().item()
                te_acc = (model(X_te)[0].argmax(1) == Y_te).float().mean().item()

            p = probe_peterweyl(model, order, irreps)
            p.update({"epoch": ep, "loss": float(loss.item()),
                      "train_acc": tr_acc, "test_acc": te_acc})
            history.append(p)

            if not grokked and te_acc >= GROK_THRESH:
                grokked = True; grok_ep = ep
                print(f"    GROKKED ep={ep}  "
                      f"β^PW={p['beta_pw']:.3f}  "
                      f"β_s={p['beta_scalar']:.3f}  "
                      f"β₂={p['beta2']:.3f}")

            if grokked:
                post_grok_count += PROBE_EVERY
                if not math.isnan(p['beta_pw']):    beta_pw_post.append(p['beta_pw'])
                if not math.isnan(p['beta_scalar']): beta_s_post.append(p['beta_scalar'])
                if not math.isnan(p['beta2']):       beta2_post.append(p['beta2'])

            if ep % 5000 == 0:
                print(f"    ep {ep:5d}  loss={loss.item():.4f}  "
                      f"tr={tr_acc:.3f}  te={te_acc:.3f}  "
                      f"β^PW={p['beta_pw']:.3f}  "
                      f"β_s={p['beta_scalar']:.3f}  "
                      f"β₂={p['beta2']:.3f}")

            if grokked and post_grok_count >= POST_GROK_SETTLE:
                print("    Settled."); break

    def star(lst):
        if len(lst) >= 4:
            t = lst[len(lst)//2:]
            return float(np.mean(t)), float(np.std(t))
        return float('nan'), float('nan')

    bpw_s, bpw_std = star(beta_pw_post)
    bs_s,  bs_std  = star(beta_s_post)
    b2_s,  _       = star(beta2_post)

    return {
        "group": group_name, "order": order, "abelian": info["abelian"], "seed": seed,
        "grokked": grokked, "grok_epoch": grok_ep,
        "beta_pw_star": bpw_s, "beta_pw_std": bpw_std,
        "beta_scalar_star": bs_s, "beta_scalar_std": bs_std,
        "beta2_star": b2_s,
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc":  history[-1]["test_acc"],
        "history_summary": [
            {k: h[k] for k in ["epoch","beta_pw","beta_scalar","beta2",
                                "K_in","K_out_pw","K_out_scalar","H_pw","H_scalar",
                                "train_acc","test_acc","active_irreps","n_irreps"]}
            for h in history
        ]
    }


# ── Main ──────────────────────────────────────────────────────

all_results = []
for group_name in GROUPS:
    info = GROUPS[group_name]
    print(f"\n{'='*60}")
    print(f"GROUP: {group_name}  (order={info['order']}, abelian={info['abelian']})")
    print(f"  Irreps: {[r['label'] for r in IRREPS[group_name]]}")
    print(f"{'='*60}")
    for seed in SEEDS:
        print(f"  Seed {seed}:")
        r = train_group(group_name, seed)
        all_results.append(r)
        print(f"    β^PW*={r['beta_pw_star']:.4f}  "
              f"β_s*={r['beta_scalar_star']:.4f}  "
              f"β₂*={r['beta2_star']:.4f}  "
              f"grokked={r['grokked']}")


# ── Summary ────────────────────────────────────────────────────

print("\n" + "="*75)
print("WHY² PETER-WEYL FOLLOW-UP — SUMMARY")
print("="*75)
print(f"{'Group':<8}{'Order':<7}{'Abelian':<9}"
      f"{'β^PW*':<12}{'β_scalar*':<12}{'β₂*':<10}{'Grokked'}")
print("-"*75)

summary = {}
for group_name in GROUPS:
    gk = [r for r in all_results
          if r["group"]==group_name and r["grokked"]
          and not math.isnan(r["beta_pw_star"])]
    if gk:
        bpw  = float(np.mean([r["beta_pw_star"] for r in gk]))
        bpws = float(np.std( [r["beta_pw_star"] for r in gk]))
        bs   = float(np.nanmean([r["beta_scalar_star"] for r in gk]))
        b2   = float(np.nanmean([r["beta2_star"] for r in gk]))
        ng   = len(gk)
    else:
        bpw=bpws=bs=b2=float('nan'); ng=0

    summary[group_name] = {
        "beta_pw_mean": bpw, "beta_pw_std": bpws,
        "beta_scalar_mean": bs, "beta2_mean": b2,
        "n_grokked": ng,
        "order": GROUPS[group_name]["order"],
        "abelian": GROUPS[group_name]["abelian"],
    }
    def f(x): return f"{x:.4f}" if not math.isnan(x) else "nan"
    print(f"{group_name:<8}{GROUPS[group_name]['order']:<7}"
          f"{str(GROUPS[group_name]['abelian']):<9}"
          f"{f(bpw):<12}{f(bs):<12}{f(b2):<10}{ng}/{len(SEEDS)}")


# ── Prediction checks ──────────────────────────────────────────

print("\n--- PREDICTION CHECKS ---")

# 1. Do non-abelian groups grok?
for g in ["D_4","Q_8"]:
    ng = summary[g]["n_grokked"]
    print(f"  {g} grokked: {ng}/{len(SEEDS)} seeds "
          f"({'YES' if ng>0 else 'NO — data still limiting?'})")

# 2. β^PW*(D_4) vs β^PW*(Q_8)
d4  = summary["D_4"]["beta_pw_mean"]
q8  = summary["Q_8"]["beta_pw_mean"]
z8  = summary["Z_8"]["beta_pw_mean"]
z16 = summary["Z_16"]["beta_pw_mean"]

if not (math.isnan(d4) or math.isnan(q8)):
    diff = d4 - q8
    print(f"\n  β^PW*(D_4)={d4:.4f}, β^PW*(Q_8)={q8:.4f}")
    print(f"  D_4 - Q_8 = {diff:+.4f}")
    if diff > 0.1:
        print("  PREDICTION 2 CONFIRMED: D_4 costs more than Q_8")
        print("  Interpretation: non-normal subgroup structure increases")
        print("  spectral compression cost beyond irrep dimensions alone")
    elif abs(diff) <= 0.1:
        print("  D_4 ≈ Q_8 — irrep structure dominates over lattice")
    else:
        print("  Q_8 > D_4 — unexpected, requires investigation")

# 3. Non-abelian vs abelian
na_bpw = [v for v in [d4, q8] if not math.isnan(v)]
ab_bpw = [v for v in [z8, z16] if not math.isnan(v)]
if na_bpw and ab_bpw:
    print(f"\n  Mean β^PW*(abelian):     {np.mean(ab_bpw):.4f}")
    print(f"  Mean β^PW*(non-abelian): {np.mean(na_bpw):.4f}")
    if np.mean(na_bpw) > np.mean(ab_bpw):
        print("  PREDICTION 3 CONFIRMED: β^PW*(non-abelian) > β^PW*(abelian)")

# 4. β^PW* vs β_scalar* discrepancy for non-abelian
print("\n  Peter-Weyl vs scalar probe comparison:")
for g in GROUPS:
    bpw = summary[g]["beta_pw_mean"]
    bs  = summary[g]["beta_scalar_mean"]
    if not (math.isnan(bpw) or math.isnan(bs)):
        diff = bpw - bs
        print(f"    {g}: β^PW*={bpw:.4f}, β_s*={bs:.4f}, "
              f"Δ={diff:+.4f} "
              f"({'PW>scalar — 2D irrep activated' if diff>0.2 else 'consistent'})")


# ── Save ──────────────────────────────────────────────────────

fname_json = f"{PREFIX}_results.json"
with open(fname_json, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {fname_json}")


# ── Plot ──────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "WHY² Peter-Weyl Follow-up: D₄ vs Q₈ — β^PW Phase Diagram",
    fontsize=13, fontweight='bold')

groups_ordered = ["Z_8", "Z_16", "D_4", "Q_8"]
colors = {'Z_8':'steelblue','Z_16':'steelblue','D_4':'tomato','Q_8':'darkorange'}
markers = {'Z_8':'o','Z_16':'o','D_4':'s','Q_8':'^'}

# Panel 1: β^PW* vs β_scalar* (abelian on diagonal, non-abelian should be off)
ax1 = axes[0]
for g in groups_ordered:
    s=summary[g]; bpw=s["beta_pw_mean"]; bs=s["beta_scalar_mean"]
    if not (math.isnan(bpw) or math.isnan(bs)):
        ax1.scatter(bs, bpw, c=colors[g], marker=markers[g], s=180, zorder=3,
                    label=g)
        ax1.annotate(g, (bs, bpw),
                     textcoords="offset points", xytext=(6,4), fontsize=10)
vals = [summary[g]["beta_pw_mean"] for g in groups_ordered
        if not math.isnan(summary[g]["beta_pw_mean"])]
lim = max(10, max(vals)+0.5) if vals else 10
ax1.plot([0,lim],[0,lim],'k--',alpha=0.3,label='β^PW=β_scalar')
ax1.set_xlabel('β_scalar*', fontsize=11)
ax1.set_ylabel('β^PW*', fontsize=11)
ax1.set_title('Peter-Weyl vs Scalar β*\n(above diagonal = 2D irrep activated)',
              fontsize=10)
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Panel 2: β^PW* by group with error bars
ax2 = axes[1]
names,means,stds,clrs=[],[],[],[]
for g in groups_ordered:
    s=summary[g]
    if not math.isnan(s["beta_pw_mean"]):
        names.append(g); means.append(s["beta_pw_mean"])
        stds.append(s["beta_pw_std"] if not math.isnan(s["beta_pw_std"]) else 0)
        clrs.append(colors[g])
if names:
    bars = ax2.bar(names, means, yerr=stds, color=clrs, capsize=6,
                   edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('β^PW*', fontsize=11)
    ax2.set_title('β^PW* by Group\n(blue=abelian, red/orange=non-abelian)',
                  fontsize=10)
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, mean + 0.05,
                 f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# Panel 3: (β^PW*, β₂*) phase diagram
ax3 = axes[2]
for g in groups_ordered:
    s=summary[g]; bpw=s["beta_pw_mean"]; b2=s["beta2_mean"]
    if not (math.isnan(bpw) or math.isnan(b2)):
        ax3.scatter(bpw, b2, c=colors[g], marker=markers[g], s=180, zorder=3,
                    label=g)
        ax3.annotate(g, (bpw, b2),
                     textcoords="offset points", xytext=(6,4), fontsize=10)
ax3.set_xlabel('β^PW*', fontsize=11)
ax3.set_ylabel('β₂*', fontsize=11)
ax3.set_title('(β^PW*, β₂*) Phase Diagram\n(WHY² representation-theoretic extension)',
              fontsize=10)
ax3.axhline(0.10, color='gray', ls='--', lw=1, alpha=0.5, label='abelian threshold')
ax3.grid(True, alpha=0.3); ax3.legend(fontsize=9)

plt.tight_layout()
fname_png = f"{PREFIX}_plot.png"
plt.savefig(fname_png, dpi=130, bbox_inches='tight')
plt.close()
print(f"Saved: {fname_png}")

print("\nDone.")
print("=" * 65)