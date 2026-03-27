# WHY² — Asymmetric Kramers Identity for Grokking

[![Status: Preprint](https://img.shields.io/badge/status-preprint-blue)](https://github.com/l-austinsmith/WHY²)
[![Results: 20](https://img.shields.io/badge/results-20-brightgreen)](https://github.com/l-austinsmith/WHY²)
[![Platform: Colab](https://img.shields.io/badge/platform-Google%20Colab-orange)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**A diagnostic framework for the memorization-to-generalization transition in transformers, grounded in the asymmetric Kramers identity and renormalization group theory.**

---

## The Core Equation

$$\beta = \frac{\log K_{\text{in}} - H_n}{\log K_{\text{out}}}$$

where **K_out** measures spectral compression of the output embedding (fraction of Fourier power in the top 10% of modes), **K_in** measures attention asymmetry toward input operands, and **H_n** is normalized spectral entropy. At grokking, β converges to a stable fixed point β* that encodes the algebraic complexity of the task. This equation is the empirical analogue of the Wetterich LPA fixed-point condition in functional renormalization group theory.

---

## Key Results (20 total)

### Core Framework (Results 1–7)
K_in and K_out are anti-correlated throughout pre-grokking training. The K_out/K_in ratio at grokking is a prime-invariant task complexity fingerprint (multiply ≈ 0.15, add ≈ 0.43). A sharp phase boundary exists at WD_c ∈ (0.25, 0.30). β* is inversely correlated with model size (ρ = −0.815).

### Fixed Point Convergence (Results 8–12)
β converges to a stable post-grokking fixed point β* with CV < 10% for multiply, 19% for add. β* is task-ordered: β*(multiply) ≈ 0.69 < β*(add) ≈ 3.44 < β*(subtract) ≈ 4.78, reflecting algebraic complexity. Subtract exhibits two-basin structure (Basin A: β* ≈ 4.78, Basin B: β* > 15). Basin assignment is deterministic: zero flips across 72 runs.

### Theoretical Connections (Results 13–14)
Grokking coincides with a local T_eff minimum in all tested seeds. β is not a reparameterization of any standard spectral probe (71 models). WD-invariance of β* is empirical scheme-independence in the Wetterich sense.

### Architecture & Spectral Structure (Results 15–17)
Ternary-weight (BitNet) transformers converge to the same β* as float32 (|diff| = 0.025 ± 0.010). β* is architecture-invariant. r(β*, active Fourier modes) = −0.85 across six task types. β*(triple) ≈ β*(multiply) detects hidden group isomorphism without circuit inspection. β₂* encodes algebraic commutativity but fails WD-invariance; β* remains the sole WD-invariant fixed point.

### Domain Boundary (Result 18)
EEG motor imagery classification produces no grokking and no stable β*. K_in collapses to maximum entropy after memorization — WHY²'s probe requires architectures with meaningful asymmetric token routing. Domain currently constrained to discrete-token, operand-structured tasks.

### Non-Abelian Extension (Results 19–20)
Non-abelian group multiplication (D_4, D_6, D_8, Q_8) does not grok under the scalar WHY² probe across 36 seeds. The scalar probe is blind to the non-abelian barrier because the DFT basis is the abelian (d_ρ=1) sector of the full representation-theoretic decomposition.

The **Peter–Weyl generalized probe** replaces scalar DFT K_out with a representation-weighted spectral decomposition using the full irrep basis of G:

$$A_{\rho,ij} = \frac{d_\rho}{|G|} \sum_g E(g) \cdot \rho_{ij}(g)^* \qquad P_\rho = \frac{|G|}{d_\rho} \sum_{ij} \|A_{\rho,ij}\|^2$$

Satisfies Parseval exactly (Σ P_ρ = ‖E‖²_F). Reduces to scalar probe for abelian groups (|β^PW* − β_scalar*| < 0.09). Detects three phenomena invisible to the scalar probe:

1. **Persistent 2D irrep activation**: D_4 and Q_8 produce β^PW = 1.0–3.4 vs. abelian baseline 0.5–1.1
2. **Abelian reduction**: β^PW = β_scalar for abelian groups — probes are nested
3. **Subgroup lattice sensitivity**: β^PW(D_4) > β^PW(Q_8) across all seeds despite identical irrep dimensions [1,1,1,1,2] — D_4 has non-normal subgroups; Q_8 is Hamiltonian

Non-abelian grokking does not occur even at TRAIN_FRAC=0.8, 50,000 epochs. The RG flow for non-abelian multiplication has no accessible IR fixed point under the standard scalar attention mechanism.

---

## β* Empirical Values

| Task | β* | Active Modes | K_out |
|------|----|-------------|-------|
| multiply | 0.69 ± 0.05 | 48.7 | 0.107 |
| division | 0.64 | 41.7 | 0.135 |
| triple (a+b+ab) | 0.69 | 48.0 | 0.133 |
| affine (2a+b) | 1.60 | 21.0 | 0.543 |
| add | 3.44 ± 0.59 | 17.7 | 0.710 |
| subtract (Basin A) | 4.78 ± 0.22 | 10.8 | 0.867 |
| subtract (Basin B) | >15, diverging | — | — |

---

## Repository Structure

```
WHY²/
├── paper/
│   └── WHY²_Paper_v10.docx            # Current paper (20 results)
├── experiments/
│   ├── why2_bvel3.py                  # Core β velocity experiment
│   ├── why2_beta_corr.py              # Spectral reparameterization (71 models)
│   ├── why2_exp1_bitnet.py            # Architecture invariance (BitNet)
│   ├── why2_exp3_fourier.py           # Fourier complexity (6 tasks)
│   ├── why2_exp4_beta2_v2.py          # β₂ commutativity
│   ├── why2_eeg_v1.py                 # EEG domain transfer v1 (null result)
│   ├── why2_eeg_v2.py                 # EEG domain transfer v2 (null result)
│   ├── why2_exp_nonabelian.py         # Non-abelian groups (6 groups × 6 seeds)
│   ├── why2_exp_peterweyl.py          # Peter–Weyl generalized probe
│   └── why2_exp_peterweyl_followup.py # D₄ vs Q₈ follow-up (TRAIN_FRAC=0.8)
├── results/
│   └── why2_results_summary.json
└── README.md
```

---

## Quickstart (Google Colab)

```python
!git clone https://github.com/l-austinsmith/WHY²
%cd WHY²/experiments

# Core result — β velocity (~30 min on T4)
!python why2_bvel3.py

# Peter–Weyl generalized probe (~90 min on T4)
!python why2_exp_peterweyl.py

# D₄ vs Q₈ follow-up (~60 min on T4)
!python why2_exp_peterweyl_followup.py
```

All experiments are self-contained, seed-reproducible, and run on free Colab T4.

---

## Theoretical Connections

| WHY² concept | Theoretical analogue |
|-------------|----------------------|
| β* fixed point | Zamolodchikov c-theorem — empirical inverse c-function |
| WHY² equation | Wetterich (1993) LPA φ⁴ fixed-point condition |
| WD-invariance of β* | Scheme-independence in functional RG |
| Subtract basin structure | φ⁶ tricritical point proximity |
| Execution manifold (Xu 2026b) | Rank-1 submanifold — β* is coordinate along it |
| Peter–Weyl probe | L²(G) decomposition via full irrep basis — DFT generalized to non-abelian G |

---

## Limitations & Scope

- All modular arithmetic experiments use prime p = 97 or small-group Cayley tables; generalization to other primes untested
- Domain currently constrained to discrete-token operand-structured tasks (EEG produces probe mismatch, Result 18)
- Non-abelian groups do not grok at tested scales; Peter–Weyl probe detects spectral structure but no IR fixed point reached
- β₂* fails WD-invariance and is a task-specific commutativity detector only
- All experiments run on free Colab T4; seed counts and epoch budgets constrained accordingly

---

## Citation

```bibtex
@misc{smith2026why2,
  title   = {Timescale Asymmetry and Phase Structure in Grokking Transformers:
             The WHY\textsuperscript{2} Framework},
  author  = {Smith, Lucas},
  year    = {2026},
  note    = {Preprint. github.com/l-austinsmith/WHY²}
}
```

---

## Open Problems

1. Does β^PW* exist for non-abelian groups under a matrix-attention architecture capable of implementing the full Peter–Weyl algorithm?
2. Does β^PW(D_4) > β^PW(Q_8) hold at the fixed point (post-grokking) or only in the pre-grokking transient?
3. Can β* predict grokking timescale from initialization, before training begins?
4. Does the subtract φ⁶ tricritical interpretation survive formal Wetterich beta-function fitting?
5. Does β* generalize to non-modular group-structured tasks (symmetric group S_n, crystallographic groups)?

See [CONTRIBUTING.md](CONTRIBUTING.md) for replication notes.

---

*All experiments reproducible on free Google Colab. No institutional affiliation. No compute budget beyond free tier.*
