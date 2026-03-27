# ============================================================
# WHY2 EXPERIMENT: NON-ABELIAN GROUPS
#
# THE QUESTION:
#   Does β* extend to non-abelian groups?
#   For abelian tasks (multiply, add, subtract) β* encodes
#   algebraic complexity and β₂* detects commutativity.
#   For non-abelian groups (a∘b ≠ b∘a by definition):
#
#   PREDICTION 1 (β*): Non-abelian groups should have β* > subtract
#   because they require more complex representations — the
#   transformer cannot exploit commutativity shortcuts.
#
#   PREDICTION 2 (β₂*): β₂* should be PERSISTENTLY HIGH for all
#   non-abelian tasks, since a∘b ≠ b∘a is baked into the structure.
#   The β₂* vs. β* picture should show both high simultaneously —
#   a new region of the (β*, β₂*) phase diagram not populated by
#   abelian tasks.
#
#   PREDICTION 3 (isomorphism): Groups of the same abstract
#   structure should give matching β*, even if their surface
#   representations differ. D_6 ≅ S_3 — both are the dihedral
#   group of order 6 / symmetric group on 3 elements. If β*
#   detects group structure: β*(D_6) ≈ β*(S_3).
#
# GROUPS TESTED:
#   1. Z_n (cyclic, abelian) — baseline, known β* range
#      Z_8: add mod 8 (abelian, simple)
#      Z_16: add mod 16 (abelian, more complex)
#
#   2. D_n (dihedral, non-abelian) — symmetries of regular n-gon
#      D_4: order 8 (smallest interesting non-abelian dihedral)
#      D_6: order 12 ≅ S_3 (dihedral of hexagon)
#      D_8: order 16 (larger non-abelian)
#
#   3. Q_8 (quaternion group, order 8, non-abelian)
#      The 8 quaternion units: {±1, ±i, ±j, ±k}
#      More "tightly" non-abelian than dihedral — every subgroup
#      is normal, unlike dihedral groups.
#
# GROUP ENCODING:
#   All groups encoded as integers 0...|G|-1.
#   Multiplication defined by precomputed Cayley table.
#   Transformer input: [a, b, |G|] → output: a∘b ∈ {0...|G|-1}
#   This is the exact same architecture as modular arithmetic —
#   just with a different output label space.
#
#   For dihedral D_n (order 2n):
#   Elements: r^k (rotations, k=0..n-1) and s·r^k (reflections)
#   Encoding: 0..n-1 = rotations, n..2n-1 = reflections
#   Multiplication: r^a · r^b = r^{(a+b) mod n}
#                   r^a · s·r^b = s·r^{(b-a) mod n}
#                   s·r^a · r^b = s·r^{(a+b) mod n}
#                   s·r^a · s·r^b = r^{(b-a) mod n}
#
#   For Q_8 (order 8):
#   Elements: {1, -1, i, -i, j, -j, k, -k} → {0,1,2,3,4,5,6,7}
#   Full Cayley table hardcoded.
#
# ARCHITECTURE: Identical to WHY2 exp3 (same as modular arith).
#   Vocab size = |G| + 1 (group elements + separator token)
#   Output size = |G|
#
# SEEDS: [42, 123, 7, 17, 71, 31]  (6 seeds per group)
# WD: 1.0  (same as all WHY2 experiments)
# Expected runtime: ~90-120 min on free Colab T4
#
# KEY DIAGNOSTIC:
#   Look at (β*, β₂*) jointly.
#   Abelian prior: β*(add) ≈ 3.25, β₂*(add) ≈ 0
#   Abelian prior: β*(subtract) ≈ 6.33, β₂*(subtract) ≈ 0.40
#   Non-abelian prediction: β* > 6.33 AND β₂* > 0.40
#   If confirmed → β* and β₂* jointly characterize both the
#   cost AND the symmetry-breaking of any finite group operation.
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
PREFIX = f"why2_nonabelian_{TS}"

print(f"WHY2 NON-ABELIAN GROUPS  |  device: {DEVICE}")
print(f"Question: Does β* extend beyond abelian groups?")
print("=" * 65)

# ── Hyperparameters ───────────────────────────────────────────
D_MODEL        = 128
N_HEADS        = 4
D_FF           = 512
LR             = 1e-3
TRAIN_FRAC     = 0.5      # 50% train — harder generalization for small groups
WD             = 1.0
MAX_EPOCHS     = 25000
PROBE_EVERY    = 100
GROK_THRESH    = 0.99
POST_GROK_SETTLE = 3000
SEEDS          = [42, 123, 7, 17, 71, 31]


# ── Group definitions ─────────────────────────────────────────

def make_cyclic_table(n):
    """Z_n: addition mod n. Abelian baseline."""
    return [[(a + b) % n for b in range(n)] for a in range(n)]


def make_dihedral_table(n):
    """
    D_n: dihedral group of order 2n (symmetries of regular n-gon).
    Elements 0..n-1: rotations r^0..r^{n-1}
    Elements n..2n-1: reflections s·r^0..s·r^{n-1}

    Multiplication rules:
      r^a * r^b     = r^{(a+b) mod n}
      r^a * s·r^b   = s·r^{(b-a) mod n}
      s·r^a * r^b   = s·r^{(a+b) mod n}
      s·r^a * s·r^b = r^{(b-a) mod n}
    """
    order = 2 * n
    table = [[0] * order for _ in range(order)]
    for a in range(order):
        for b in range(order):
            if a < n and b < n:
                # r^a * r^b = r^{(a+b) mod n}
                table[a][b] = (a + b) % n
            elif a < n and b >= n:
                # r^a * s·r^{b-n} = s·r^{(b-n-a) mod n}
                table[a][b] = n + (b - n - a) % n
            elif a >= n and b < n:
                # s·r^{a-n} * r^b = s·r^{(a-n+b) mod n}
                table[a][b] = n + (a - n + b) % n
            else:
                # s·r^{a-n} * s·r^{b-n} = r^{(b-n-(a-n)) mod n} = r^{(b-a) mod n}
                table[a][b] = (b - a) % n
    return table


def make_q8_table():
    """
    Q_8: quaternion group of order 8.
    Elements: {1, -1, i, -i, j, -j, k, -k}
    Encoded as: {0=1, 1=-1, 2=i, 3=-i, 4=j, 5=-j, 6=k, 7=-k}

    Multiplication table (row * col):
    i*j=k, j*k=i, k*i=j (cyclic)
    i*i = j*j = k*k = -1
    (-1)*x = x*(-1) = -x for all x
    """
    # Full Q_8 Cayley table
    # 0=1, 1=-1, 2=i, 3=-i, 4=j, 5=-j, 6=k, 7=-k
    table = [
        # *  1   -1   i   -i   j   -j   k   -k
        [0,  1,  2,   3,  4,   5,  6,   7],   # 1 *
        [1,  0,  3,   2,  5,   4,  7,   6],   # -1 *
        [2,  3,  1,   0,  6,   7,  5,   4],   # i *
        [3,  2,  0,   1,  7,   6,  4,   5],   # -i *
        [4,  5,  7,   6,  1,   0,  2,   3],   # j *
        [5,  4,  6,   7,  0,   1,  3,   2],   # -j *
        [6,  7,  4,   5,  3,   2,  1,   0],   # k *
        [7,  6,  5,   4,  2,   3,  0,   1],   # -k *
    ]
    return table


def verify_group(table, name):
    """Basic sanity checks: closure, identity, associativity sample."""
    n = len(table)
    # Check identity element exists
    for e in range(n):
        if all(table[e][x] == x and table[x][e] == x for x in range(n)):
            identity = e
            break
    else:
        print(f"  WARNING: {name} has no identity element!")
        return False
    # Check closure
    for a in range(n):
        for b in range(n):
            if not (0 <= table[a][b] < n):
                print(f"  WARNING: {name} not closed!")
                return False
    # Check commutativity (should be False for non-abelian)
    abelian = all(table[a][b] == table[b][a] for a in range(n) for b in range(n))
    print(f"  {name}: order={n}, identity={identity}, abelian={abelian}")
    return True


# Define all groups to test
GROUPS = {
    # Abelian baselines (small, to calibrate β* scale for small groups)
    "Z_8":   {"table": make_cyclic_table(8),    "order": 8,  "abelian": True},
    "Z_16":  {"table": make_cyclic_table(16),   "order": 16, "abelian": True},
    # Non-abelian dihedral
    "D_4":   {"table": make_dihedral_table(4),  "order": 8,  "abelian": False},
    "D_6":   {"table": make_dihedral_table(6),  "order": 12, "abelian": False},
    "D_8":   {"table": make_dihedral_table(8),  "order": 16, "abelian": False},
    # Non-abelian quaternion
    "Q_8":   {"table": make_q8_table(),         "order": 8,  "abelian": False},
}

print("\nGroup verification:")
for name, info in GROUPS.items():
    verify_group(info["table"], name)


# ── Data generation ───────────────────────────────────────────
def make_group_data(table, order, train_frac, seed):
    """
    Generate all (a, b, result) triples for the group.
    Input: [a, b, order] (same format as modular arithmetic)
    Output: a∘b ∈ {0..order-1}
    """
    torch.manual_seed(seed)
    pairs  = [(a, b) for a in range(order) for b in range(order)]
    labels = [table[a][b] for a, b in pairs]
    idx    = torch.randperm(len(pairs))
    n_tr   = int(len(pairs) * train_frac)
    tr, te = idx[:n_tr], idx[n_tr:]

    def mk(s):
        X = torch.tensor([[pairs[i][0], pairs[i][1], order]
                           for i in s], device=DEVICE)
        Y = torch.tensor([labels[i] for i in s], device=DEVICE)
        return X, Y

    return mk(tr), mk(te)


# ── Transformer (same as WHY2 exp3, generalized vocab) ────────
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
        h2, w = self.attn(h, h, h, need_weights=True,
                          average_attn_weights=False)
        h = self.norm1(h + h2)
        h = self.norm2(h + self.ff2(F.relu(self.ff1(h))))
        return self.out(h[:, 2, :]), w


# ── WHY2 probe (adapted for arbitrary group order) ────────────
def probe_why2(model, order):
    """
    WHY2 observables for group multiplication task.

    K_in:  attention from output position to the two operands
           (same probe as modular arithmetic, unchanged)
    K_out: top-10% Fourier power of output embedding
           (DFT basis over group elements, analogous to Z_p case)
    H_n:   normalized spectral entropy
    β₂:    attention asymmetry (K_in_asym / K_in)
    """
    model.eval()
    pairs = torch.tensor(
        [[a, b, order] for a in range(order) for b in range(order)],
        device=DEVICE)
    with torch.no_grad():
        _, attn_w = model(pairs)

    # K_in: output position attending to operands
    w_ops = attn_w[:, :, 2, 0] + attn_w[:, :, 2, 1]
    K_in  = w_ops.mean().item()
    asym  = (attn_w[:, :, 2, 0] - attn_w[:, :, 2, 1]).abs().mean().item()
    beta2 = asym / K_in if K_in > 1e-6 else float('nan')

    # K_out, H_n: Fourier analysis of output embedding
    # For groups of order n, use DFT basis over n elements
    E = model.embed.weight[:order].detach().float()
    n = order
    F_basis = torch.zeros(n, n, device=DEVICE)
    for k in range(n):
        for m in range(n):
            F_basis[k, m] = math.cos(2 * math.pi * k * m / n)
    F_basis /= math.sqrt(n)

    power = torch.stack([(F_basis[k] @ E).pow(2).sum() for k in range(n)])
    power = power / (power.sum() + 1e-10)

    top_k  = max(1, n // 10)
    K_out  = float(power.topk(top_k).values.sum())
    H_n    = float(-(power * (power + 1e-10).log()).sum() / math.log(n))
    active = int((power.cpu().numpy() > 1.0 / n).sum())

    beta = float('nan')
    if 1e-6 < K_out < 1 - 1e-6 and K_in > 1e-6:
        try:
            beta = (math.log(K_in) - H_n) / math.log(K_out)
        except:
            pass

    return {
        "K_in": K_in, "K_out": K_out, "H_n": H_n,
        "K_in_asym": asym, "beta2": beta2, "beta": beta,
        "active_modes": active
    }


# ── Training loop ─────────────────────────────────────────────
def train_group(group_name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    info   = GROUPS[group_name]
    order  = info["order"]
    table  = info["table"]

    (X_tr, Y_tr), (X_te, Y_te) = make_group_data(
        table, order, TRAIN_FRAC, seed)

    # vocab_size = order + 1 (elements + separator token)
    model = Transformer(
        vocab_size=order + 1,
        output_size=order,
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    history = []
    grokked = False
    grok_ep = None
    post_grok_count = 0
    beta_post  = []
    beta2_post = []

    for ep in range(MAX_EPOCHS):
        model.train()
        logits, _ = model(X_tr)
        loss = F.cross_entropy(logits, Y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % PROBE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = (model(X_tr)[0].argmax(1) == Y_tr).float().mean().item()
                te_acc = (model(X_te)[0].argmax(1) == Y_te).float().mean().item()

            p = probe_why2(model, order)
            p.update({"epoch": ep, "loss": loss.item(),
                      "train_acc": tr_acc, "test_acc": te_acc})
            history.append(p)

            if not grokked and te_acc >= GROK_THRESH:
                grokked = True
                grok_ep = ep
                print(f"    GROKKED ep={ep}  β={p['beta']:.3f}  "
                      f"β₂={p['beta2']:.3f}  K_out={p['K_out']:.3f}")

            if grokked:
                post_grok_count += PROBE_EVERY
                if not math.isnan(p['beta']):
                    beta_post.append(p['beta'])
                if not math.isnan(p['beta2']):
                    beta2_post.append(p['beta2'])

            if ep % 5000 == 0:
                print(f"    ep {ep:5d}  loss={loss.item():.4f}  "
                      f"tr={tr_acc:.3f}  te={te_acc:.3f}  "
                      f"β={p['beta']:.3f}  β₂={p['beta2']:.3f}")

            if grokked and post_grok_count >= POST_GROK_SETTLE:
                print(f"    Settled.")
                break

    beta_star  = float('nan')
    beta2_star = float('nan')
    converged  = False
    if len(beta_post) >= 4:
        tail       = beta_post[len(beta_post) // 2:]
        beta_star  = float(np.mean(tail))
        beta_std   = float(np.std(tail))
        converged  = beta_std < 0.1 * abs(beta_star)
        tail2      = beta2_post[len(beta2_post) // 2:]
        beta2_star = float(np.mean(tail2))
    else:
        beta_std = float('nan')

    return {
        "group": group_name,
        "order": order,
        "abelian": info["abelian"],
        "seed": seed,
        "grokked": grokked,
        "grok_epoch": grok_ep,
        "beta_star": beta_star,
        "beta_std": beta_std if len(beta_post) >= 4 else float('nan'),
        "beta2_star": beta2_star,
        "converged": converged,
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "history": history
    }


# ── Main ──────────────────────────────────────────────────────
all_results = []

for group_name in GROUPS:
    info = GROUPS[group_name]
    print(f"\n{'='*60}")
    print(f"GROUP: {group_name}  "
          f"(order={info['order']}, abelian={info['abelian']})")
    print(f"{'='*60}")
    group_results = []
    for seed in SEEDS:
        print(f"  Seed {seed}:")
        r = train_group(group_name, seed)
        group_results.append(r)
        print(f"    β*={r['beta_star']:.4f}  β₂*={r['beta2_star']:.4f}  "
              f"grokked={r['grokked']}")
    all_results.extend(group_results)


# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY BY GROUP")
print("=" * 65)
print(f"{'Group':<8} {'Order':<7} {'Abelian':<9} "
      f"{'β* mean':<10} {'β* std':<10} "
      f"{'β₂* mean':<10} {'Grokked':<10}")
print("-" * 64)

summary = {}
for group_name in GROUPS:
    group_res = [r for r in all_results if r["group"] == group_name]
    grokked   = [r for r in group_res if r["grokked"] and not math.isnan(r["beta_star"])]
    if grokked:
        betas  = [r["beta_star"]  for r in grokked]
        beta2s = [r["beta2_star"] for r in grokked]
        b_mean = np.mean(betas)
        b_std  = np.std(betas)
        b2_mean = np.mean(beta2s)
    else:
        b_mean = b_std = b2_mean = float('nan')

    summary[group_name] = {
        "beta_mean": b_mean, "beta_std": b_std,
        "beta2_mean": b2_mean,
        "n_grokked": len(grokked),
        "order": GROUPS[group_name]["order"],
        "abelian": GROUPS[group_name]["abelian"]
    }

    print(f"{group_name:<8} {GROUPS[group_name]['order']:<7} "
          f"{str(GROUPS[group_name]['abelian']):<9} "
          f"{b_mean:<10.4f} {b_std:<10.4f} "
          f"{b2_mean:<10.4f} {len(grokked)}/{len(SEEDS)}")

# Check predictions
print("\n--- PREDICTION CHECKS ---")
abelian_betas   = [summary[g]["beta_mean"] for g in ["Z_8", "Z_16"]
                   if not math.isnan(summary[g]["beta_mean"])]
nonabelian_betas = [summary[g]["beta_mean"] for g in ["D_4", "D_6", "D_8", "Q_8"]
                    if not math.isnan(summary[g]["beta_mean"])]
if abelian_betas and nonabelian_betas:
    print(f"  Mean β*(abelian baselines):    {np.mean(abelian_betas):.4f}")
    print(f"  Mean β*(non-abelian groups):   {np.mean(nonabelian_betas):.4f}")
    if np.mean(nonabelian_betas) > np.mean(abelian_betas):
        print("  PREDICTION 1 CONFIRMED: β*(non-abelian) > β*(abelian)")
    else:
        print("  PREDICTION 1 NOT CONFIRMED")

d6_beta = summary["D_6"]["beta_mean"]
q8_beta = summary["Q_8"]["beta_mean"]
d4_beta = summary["D_4"]["beta_mean"]
if not (math.isnan(d4_beta) or math.isnan(q8_beta)):
    diff = abs(d4_beta - q8_beta)
    print(f"  D_4 β*={d4_beta:.4f}, Q_8 β*={q8_beta:.4f} "
          f"(same order=8, diff={diff:.4f})")
    if diff < 0.5:
        print("  Same-order groups converge to similar β* — structure detected")
    else:
        print("  Same-order groups diverge — β* sensitive to group structure")


# ── Save ──────────────────────────────────────────────────────
save_results = []
for r in all_results:
    entry = {k: v for k, v in r.items() if k != "history"}
    entry["history_summary"] = [
        {k: h[k] for k in ["epoch", "beta", "beta2", "K_in", "K_out",
                             "H_n", "train_acc", "test_acc"]}
        for h in r["history"]
    ]
    save_results.append(entry)

fname_json = f"{PREFIX}_results.json"
with open(fname_json, "w") as f:
    json.dump(save_results, f, indent=2)
print(f"\nSaved: {fname_json}")


# ── Plot ──────────────────────────────────────────────────────
groups_ordered = ["Z_8", "Z_16", "D_4", "Q_8", "D_6", "D_8"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "WHY2 Non-Abelian Groups: β* and β₂* Phase Diagram",
    fontsize=13, fontweight='bold')

# Panel 1: β* by group (bar chart)
ax1 = axes[0]
names, means, stds, colors = [], [], [], []
for g in groups_ordered:
    s = summary[g]
    if not math.isnan(s["beta_mean"]):
        names.append(g)
        means.append(s["beta_mean"])
        stds.append(s["beta_std"])
        colors.append('steelblue' if s["abelian"] else 'tomato')
if names:
    ax1.bar(names, means, yerr=stds, color=colors, capsize=5)
    ax1.set_ylabel('β*')
    ax1.set_title('β* by Group\n(blue=abelian, red=non-abelian)')
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: β₂* by group
ax2 = axes[1]
names2, b2means, b2colors = [], [], []
for g in groups_ordered:
    s = summary[g]
    if not math.isnan(s["beta2_mean"]):
        names2.append(g)
        b2means.append(s["beta2_mean"])
        b2colors.append('steelblue' if s["abelian"] else 'tomato')
if names2:
    ax2.bar(names2, b2means, color=b2colors, capsize=5)
    ax2.set_ylabel('β₂*')
    ax2.set_title('β₂* by Group\n(commutativity order parameter)')
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: (β*, β₂*) scatter — the phase diagram
ax3 = axes[2]
for g in groups_ordered:
    s = summary[g]
    if not (math.isnan(s["beta_mean"]) or math.isnan(s["beta2_mean"])):
        color = 'steelblue' if s["abelian"] else 'tomato'
        marker = 'o' if s["abelian"] else 's'
        ax3.scatter(s["beta_mean"], s["beta2_mean"],
                    c=color, marker=marker, s=120, zorder=3)
        ax3.annotate(g, (s["beta_mean"], s["beta2_mean"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)

ax3.set_xlabel('β*')
ax3.set_ylabel('β₂*')
ax3.set_title('(β*, β₂*) Phase Diagram\n(circle=abelian, square=non-abelian)')
ax3.grid(True, alpha=0.3)
ax3.axhline(0.10, color='gray', ls='--', lw=1, alpha=0.5,
            label='abelian threshold')
ax3.legend(fontsize=8)

plt.tight_layout()
fname_png = f"{PREFIX}_plot.png"
plt.savefig(fname_png, dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {fname_png}")

print("\nDone.")
print("=" * 65)