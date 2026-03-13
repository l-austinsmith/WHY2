# WHY2: A Running Coupling Framework for Grokking

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/l-austinsmith/WHY2/blob/main/experiments/why2_bvel3.py)
[![Results](https://img.shields.io/badge/results-17%20across%207%20experiments-brightgreen)](https://github.com/l-austinsmith/WHY2/tree/main/results)

Grokking transformers trained on modular arithmetic converge to task-specific fixed points in an emergent scalar β, computed from three internal observables: attention routing concentration (K_in), Fourier spectral concentration (K_out), and normalized spectral entropy (H_n). β* orders tasks by algebraic symmetry cost — multiply ≈ 0.63 < add ≈ 3.25 < subtract ≈ 6.33 — spans nearly two orders of magnitude, is inversely correlated with active Fourier mode count (r = −0.85 across six tasks), detects hidden group isomorphisms without circuit inspection, and is invariant to architecture (float32 ≈ ternary) and weight decay.

## Table of Contents
- [Core Results](#the-core-results)
- [Seventeen Results](#seventeen-results)
- [Experiments](#experiments)
- [Quick Start](#quick-start)
- [Repo Structure](#repo-structure)
- [Limitations](#limitations--scope)
- [Paper](#paper)
- [Citation](#citation)
- [Related Work](#related-work)
- [Contributing](#contributing)

The central diagnostic is the emergent equation:

**K_out^β = K_in · exp(−H_n)**

rearranged as:

**β = (log K_in − H_n) / log K_out**

β is not a free parameter, not a fitted exponent, and not equivalent to an inverse thermodynamic temperature. It is computed directly from three simultaneously-measured observables at every probe epoch. The WHY2 equation is formally equivalent to the local potential approximation (LPA) fixed-point condition of the Wetterich functional renormalization group equation — a structural derivation, not an analogy. The interpretation of β* as an empirical inverse c-function is the analogical layer built on top of that derivation.

---

## The Core Results

### Task ordering

β* is ordered by algebraic complexity, spanning nearly two orders of magnitude:

| Task | β* | Active Fourier modes |
|------|----|----------------------|
| multiply | 0.63 ± 0.01 | ~49 |
| division | 0.64 ± 0.05 | ~42 |
| triple (a+b+ab) | 0.69 ± 0.04 | ~48 |
| affine (2a+b) | 1.60 ± 0.19 | ~21 |
| add | 3.25 ± 0.94 | ~18 |
| subtract | 6.33 ± 0.86 | ~11 |

β* is **inversely correlated** with active Fourier mode count (r = −0.85 across six tasks). β* measures spectral compression cost, not mode richness: subtract compresses from ~97 modes to ~11 (89% compression, high cost); multiply retains ~49 (50% compression, low cost). The strongest single correlate is r(β*, K_out) = +0.908.

### Group isomorphism detection

β* detects algebraic isomorphisms **without circuit inspection** — from a single scalar measurement during training.

- **triple(a,b) = a+b+ab mod p** has nonlinear surface form but factors as (1+a)(1+b)−1, making it isomorphic to multiply under index shift x → x+1. WHY2 detects this: β*(triple) ≈ β*(multiply) (|diff| = 0.064, within 1σ).
- **division(a,b) = a·b⁻¹ mod p** is isomorphic to multiply: β*(division) ≈ β*(multiply) (|diff| = 0.013).
- **subtract ≅ add** confirmed across multiple seed sets.

Three independent isomorphism confirmations. The framework is sensitive to algebraic structure, not surface form.

### Architecture invariance

Ternary-weight transformers (weights constrained to {−1, 0, +1} via straight-through estimation, approximating BitNet) converge to the same β* as float32 equivalents:

- Multiply: |β*(ternary) − β*(float32)| = 0.025 ± 0.010 (0.44σ of prior)
- Ternary models grokked **1.3–1.6× faster** — weight quantization accelerates but does not redirect the RG flow

β* is a property of the **task**, not the hardware encoding it.

### Commutativity detection via β₂*

The normalized attention asymmetry β₂ = K_in_asym / K_in converges to a task-specific late-training value β₂*:

- **Commutative tasks** (multiply, add): β₂* → 0 (collapse ratio < 0.10). Attention symmetrizes after grokking because a∘b = b∘a.
- **Non-commutative subtract**: β₂* ≈ 0.32–0.44 (persistent asymmetry, collapse ratio ≈ 0.40). Attention never symmetrizes because a−b ≠ b−a.

The separation is clean: β₂* < 0.10 for all commutative seeds; β₂* > 0.28 for all subtract seeds. The persistent asymmetry in subtract may relate to commutator defect signals in recent geometric accounts of grokking [Xu 2026a], where non-commuting gradients precede generalization and are causally implicated. β₂* is WD-dependent (not a universal fixed point), but its sign is a reliable binary detector of algebraic commutativity. β* remains the sole WD-invariant fixed point.

### Theoretical grounding

The WHY2 equation is formally equivalent to the **local potential approximation (LPA) fixed-point condition of the Wetterich functional renormalization group equation**. Under this identification:

- β* is an empirical inverse c-function (Zamolodchikov 1986) — interpretive layer
- WD-invariance of β* is empirical scheme-independence — interpretive layer
- The subtract two-basin structure (Basin A: β* ≈ 5–8, Basin B: β* ≈ 12–19) signals proximity to a φ⁶ tricritical point
- The training trajectory maps to the radial AdS coordinate in holographic RG (UV = random init, IR = β*)

---

## Seventeen Results

1. K_in and K_out are anti-correlated throughout pre-grokking training
2. The K_out/K_in ratio at grokking is a prime-invariant task complexity fingerprint
3. A sharp phase boundary exists at WD_c ∈ (0.25, 0.30) with critical slowing down
4. K_out decreases monotonically with model size (ρ = −0.815)
5. β converges to finite task-specific fixed points β* with dβ/dt → 0
6. The ordering β*(multiply) < β*(add) < β*(subtract) holds universally
7. Subtract exhibits two-attractor dynamics (Basin A/B) separated by K_out ≈ 0.83
8. Asymmetry persistence at K_out = 0.83 threshold predicts basin assignment
9. No post-initialization intervention shifts basin assignment (144 null results)
10. Epoch-0 geometry is identical across basins; divergence begins within ~25 steps
11. Initial asymmetry asym@0 does not predict β* (r = −0.044, n=30)
12. Basin assignment is fully deterministic (72 runs, zero basin flips)
13. Grokking coincides with local T_eff minimum in all seeds; β is not 1/T_eff
14. β is not a reparameterization of any standard spectral probe (71 models; cross-term partial r = +0.16, below detection threshold)
15. β* is architecture-invariant: ternary and float32 transformers converge to same β* (Exp 5)
16. r(β*, active Fourier modes) = −0.85; r(β*, K_out) = +0.908; three group isomorphisms detected without circuit inspection (Exp 6)
17. β₂* cleanly separates commutative (β₂* ≈ 0) from non-commutative (β₂* > 0.28) tasks; fails WD-invariance, confirming β* as sole universal fixed point (Exp 7)

---

## Experiments

| Script | What it runs | Runtime (free T4) |
|--------|-------------|-------------------|
| `why2_bvel3.py` | Core β velocity: multiply, add, subtract × 3 seeds | ~35 min |
| `why2_beta_corr.py` | Reparameterization check: 71 models | ~40 min |
| `why2_teff_v3.py` | Effective temperature series | ~20 min |
| `why2_predict_v2.py` | Structural prediction test | ~15 min |
| `why2_exp1_bitnet.py` | Architecture invariance: float32 vs ternary | ~45 min |
| `why2_exp3_fourier.py` | Fourier complexity: 6 tasks × 6 seeds | ~60 min |
| `why2_exp4_beta2_v2.py` | β₂ symmetry-breaking order parameter | ~50 min |
| `why2_basin_precheck.py` | Basin B pre-check at WD=0.5 (negative result) | ~30 min |

All scripts are self-contained. Copy any script into a Colab notebook, run it. Output JSON and PNG saved to working directory.

---

## Quick Start

```bash
pip install torch numpy scipy matplotlib
```

```python
# In Colab: Runtime > Change runtime type > T4 GPU
!git clone https://github.com/l-austinsmith/WHY2
%cd WHY2/experiments
exec(open('why2_bvel3.py').read())
```

---

## Repo Structure

```
WHY2/
├── README.md
├── paper/
│   ├── WHY2_Paper_v9.docx
│   └── WHY2_Paper_v9.pdf       ← export from DOCX before pushing
├── experiments/                ← all self-contained scripts
├── results/                    ← JSONs from all runs
└── figures/                    ← PNGs from all runs
```

---

## Limitations & Scope

- **Single-layer transformers only.** Scaling to 2–4 layers is future work; Fourier circuit competition differs in deeper models.
- **Modular arithmetic, p = 97/113.** Larger primes, non-abelian groups, and real-world tasks not yet tested.
- **β is phenomenological.** The equation is data-motivated from Kramers timescale asymmetry. The Wetterich LPA equivalence is a structural derivation; the inverse c-function interpretation is an empirical analogy built on top of it.
- **Subtract basin structure characterized primarily at WD ≥ 1.0.** Basin B appears at lower WD; high-WD runs stay in Basin A. Full Basin B characterization is future work.
- **β₂ threshold predictor is weak (r ≈ 0.33).** Asymmetry dynamics are partially decoupled from spectral compression. Reported honestly.

---

## Paper

**WHY2: A Running Coupling Framework for Grokking**
Lucas Smith, 2026

[`paper/WHY2_Paper_v9.pdf`](paper/WHY2_Paper_v9.pdf) · [`paper/WHY2_Paper_v9.docx`](paper/WHY2_Paper_v9.docx)

17 results, full methodology, seed-level appendix tables, Wetterich functional RG grounding, Zamolodchikov c-theorem and holographic RG connections.

---

## Citation

```bibtex
@article{smith2026why2,
  title   = {WHY2: A Running Coupling Framework for Grokking},
  author  = {Smith, Lucas},
  year    = {2026},
  url     = {https://github.com/l-austinsmith/WHY2}
}
```

---

## Related Work

- Xu, Y. (2026a). Early-Warning Signals of Grokking via Loss-Landscape Geometry. arXiv:2602.16967.
- Xu, Y. (2026b). Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking. arXiv:2602.16746.
- He et al. (2026). Three-stage grokking and Fourier mode competition. arXiv:2602.16849.
- Cullen et al. (2026). Grokking as phase transition between competing basins. arXiv:2603.01192.
- Zhang et al. (2025). Grokking as computational glass relaxation. arXiv:2505.11411.
- Wetterich, C. (1993). Exact renormalization group equation for the effective potential. Physics Letters B, 301(1), 90–94.
- Zamolodchikov, A.B. (1986). Irreversibility of the flux of the renormalization group. JETP Letters, 43(12), 730–732.

---

## Background

Built by a self-taught researcher with no institutional affiliation, no compute budget beyond free Google Colab T4, and no collaborators. Every theoretical connection was found after the experimental result, not before.

The triple isomorphism result was not predicted. It emerged from the measurement and was explained afterward.

---

## What's Next

- Larger prime p (scaling behavior of β*)
- Non-abelian group tasks (cross-term coupling prediction)
- Causal interventions on β₂ trajectory (does perturbing β₂ change β*?)
- Enzyme EC class classification (β* as biological task complexity)
- Ising spin system RG flows

If you have compute and want to collaborate, open an issue.

---

## Contributing

Bug reports, replication attempts, and extension experiments are all welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details on open problems and replication notes.

---

## License

MIT — see [LICENSE](LICENSE).

---

*All experiments reproducible on free hardware. Full seed-level data in the appendix.*
