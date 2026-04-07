# WHY²: A Spectral Diagnostic Framework for Grokking in Transformers

**Lucas Smith · Independent Researcher**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

WHY² (Weights, Harmonics, Yield) is a framework for probing the internal dynamics of the memorization-to-generalization transition in transformers, commonly known as *grokking*. The framework defines two spectral observables that can be computed during training without access to test accuracy, and derives an emergent diagnostic β whose convergence to a task-specific fixed point β* characterizes the complexity of the learned algorithm.

The core equation is:

```
β = (ln K_in − H_n) / ln K_out
```

where:
- **K_in** — the fraction of attention at the `=` token routed toward input operand positions (attention routing concentration)
- **K_out** — the fraction of power in the top 10% of Fourier modes of the output embedding matrix (spectral crystallization)
- **H_n** — normalized spectral entropy of the Fourier power distribution

β is computed directly from these three simultaneously measured quantities at every probe step. It is not a free parameter and is not fitted to data.

The relationship between these observables follows an asymmetric Kramers barrier-crossing equation:

```
K_out^β = K_in · exp(−H_n)
```

---

## Key Results (23 total)

| # | Result |
|---|--------|
| 1 | K_in and K_out are anti-correlated throughout pre-grokking training — a continuous observable of the memorization-to-generalization transition |
| 2 | The K_out/K_in ratio at grokking is a prime-invariant task fingerprint (multiply: ~0.15, add: ~0.43, no overlap) |
| 3 | A sharp phase boundary exists at weight decay WD_c ∈ (0.25, 0.30) with critical slowing down |
| 4 | K_out at grokking decreases monotonically with model size (ρ = −0.815) |
| 5–6 | β converges to finite task-specific fixed points β*(multiply) ≈ 0.72 < β*(add) ≈ 3.5 < β*(subtract) ≈ 5–15 |
| 7 | Subtract exhibits two-attractor dynamics (Basin A: β* ≈ 5.0, Basin B: β* ≈ 15.2) separated by K_out ≈ 0.83 |
| 8–12 | Basin assignment is fully deterministic, diverges within ~25 gradient steps, and is unaffected by any post-initialization intervention (174 training runs) |
| 13 | Grokking coincides with a local effective-temperature minimum in all tested seeds; β is **not** a literal inverse temperature |
| 14 | β* is not reducible to any combination of 11 standard spectral probes (71 models); K_in explains 0.6% of β variance |
| 15 | Ternary-weight (BitNet-style) transformers converge to the same β* as float32 equivalents |
| 16 | r(β*, active Fourier modes) = −0.85; β* measures spectral compression cost, not mode richness |
| 16b | β*(triple) ≈ β*(multiply) detects the hidden group isomorphism triple(a,b) = a+b+ab ≅ multiply |
| 17 | Attention asymmetry β₂* detects commutativity but fails weight-decay invariance; β* is the sole WD-invariant fixed point |
| 18 | EEG domain transfer: null result; K_in requires asymmetric operand-routing architectures |
| 19 | Scalar probe fails to detect grokking in non-abelian groups (D₄, D₆, D₈, Q₈) |
| 20 | Peter–Weyl generalized probe detects non-abelian spectral structure invisible to the scalar probe; β^PW(D₄) > β^PW(Q₈) despite identical irrep dimensions |
| 21 | S₄ grokked 6/6 seeds; r(β^PW*, 3D irrep power) = +0.978 identifies three distinct grokking algorithms |
| 22 | S₃ fails to grok at any data fraction; the barrier depends on quotient group structure (V₄ ⊴ S₄ provides a factored subalgorithm) |
| 23 | Algorithm selection in S₄ is set by weight initialization geometry in 70% of seeds; β^PW* measures post-grokking spectral concentration, not algorithm selection |

---

## Theoretical Connections

The WHY² equation is formally equivalent to the **local potential approximation (LPA) fixed-point condition** of the Wetterich functional renormalization group equation. This provides:

- A field-theoretic grounding for the observed **weight-decay invariance of β*** (empirical scheme-independence)
- An interpretation of the subtract two-basin structure as proximity to a **φ⁶ tricritical point**
- A connection between β* and **Zamolodchikov's c-theorem**: β* is a monotone scalar characterizing the infrared fixed point of the grokking RG flow

The Peter–Weyl generalization connects β^PW* to the **rank of the Shutman et al. fusion structure tensor** of the implemented algorithm, and to the **quotient group structure** of the group: groups with a suitable normal subgroup factorization (S₄ via V₄) grok; groups without one (S₃) do not.

---

## Repository Structure

```
WHY2/
├── why2_bvel3.py               # Core β velocity experiment (18,000-epoch, 3 tasks)
├── why2_exp1_bitnet.py         # Architecture invariance (BitNet vs float32)
├── why2_exp3_fourier.py        # Fourier complexity across 6 task types
├── why2_exp4_beta2_v2.py       # β₂ commutativity detector
├── why2_eeg_v1.py              # EEG domain transfer (null result, v1)
├── why2_eeg_v2.py              # EEG domain transfer (null result, v2)
├── why2_exp_nonabelian.py      # Non-abelian group experiment
├── why2_exp_peterweyl.py       # Peter–Weyl generalized probe
├── why2_exp_peterweyl_followup.py  # D₄ vs Q₈ follow-up
├── why2_exp_symmetric.py       # S₃ and S₄ experiments
├── why2_s4_spectrum_analysis.py    # S₄ per-irrep spectral analysis
├── why2_s3_diagnostic.py       # S₃ data-saturation sweep
├── why2_s4_early_detection.py  # Early algorithm detection (36 seeds)
├── why2_s4_early_analysis.py   # Post-hoc analysis of early detection
├── why2_finance_probe_v9.py    # Finance domain: WHY²-HAR volatility forecasting
└── WHY2_Paper.docx       # Full manuscript
```

---

## Experimental Programs

| Program | Runs | Key Finding |
|---------|------|-------------|
| Core 63-run experiment | 63 | Phase diagram, task fingerprints, WD boundary |
| β velocity experiment | 9 (×18,000 epochs) | Task-specific fixed points β* |
| Basin selection falsification | 174 | Basin determined within ~25 gradient steps |
| T_eff series | 6 seeds | Grokking = T_eff minimum; β ≠ 1/T |
| Spectral reparameterization | 71 | β* not reducible to standard probes |
| BitNet architecture invariance | 44 | β* is architecture-invariant |
| Six-task Fourier complexity | 44 | r(β*, modes) = −0.85 |
| β₂ commutativity | 44 | β₂* detects commutativity |
| EEG domain transfer | 2 | Null; K_in probe requires operand routing |
| Non-abelian groups | 36 | Scalar probe structural failure |
| Peter–Weyl probe | 2 experiments | Non-abelian structure detection |
| Symmetric groups S₃/S₄ | 6-seed + sweeps | Quotient group determines grokking |
| S₄ early detection | 36 seeds (dense) | Initialization sets irrep; β^PW* measures amplification |

**Total: >700 training runs**

---

## Compute

All experiments were run on free Google Colab T4 GPUs. No paid compute was used.

---

## Quick Start

```python
# Install dependencies
pip install torch numpy

# Compute WHY² observables for a trained model
import torch
import numpy as np

def compute_why2(model, batch, p):
    """
    Compute K_in, K_out, H_n, and beta from a trained transformer.
    
    Args:
        model:  transformer model with accessible attention weights
        batch:  input batch of shape (batch_size, 3) — sequences [a, b, =]
        p:      modular arithmetic prime (vocabulary size)
    
    Returns:
        dict with keys: K_in, K_out, H_n, beta
    """
    # K_in: attention routing concentration at = token (position 2)
    with torch.no_grad():
        _, attn_weights = model.attention(batch, batch, batch, 
                                           need_weights=True)
    # attn_weights shape: (batch, heads, seq_len, seq_len)
    # w_{2→0} and w_{2→1}: from position 2 to positions 0 and 1
    K_in = attn_weights[:, :, 2, :2].sum(dim=-1).mean().item()

    # K_out: Fourier spectral concentration in output embedding
    E = model.embedding.weight.detach().cpu().numpy()  # shape: (p, d_model)
    F = np.array([[np.cos(2*np.pi*k*j/p) for j in range(p)] 
                   for k in range(p)])  # Fourier basis
    power = np.array([np.linalg.norm(F[k] @ E)**2 for k in range(p)])
    power /= power.sum()
    n_top = max(1, p // 10)  # top 10%
    K_out = float(np.sort(power)[::-1][:n_top].sum())

    # H_n: normalized spectral entropy
    eps = 1e-12
    H_n = float(-np.sum(power * np.log(power + eps)) / np.log(p))

    # beta: emergent diagnostic
    import math
    if K_out > 0 and K_out < 1 and K_in > 0:
        beta = (math.log(K_in) - H_n) / math.log(K_out)
    else:
        beta = float('nan')

    return {"K_in": K_in, "K_out": K_out, "H_n": H_n, "beta": beta}
```

---

## Citation

If you use WHY² in your research, please cite:

```bibtex
@misc{smith2026why2,
  title  = {{WHY}$^2$: A Spectral Diagnostic Framework for Grokking in Transformers},
  author = {Smith, Lucas},
  year   = {2026},
  url    = {https://github.com/l-austinsmith/WHY2}
}
```

---

## Related Work

- Power et al. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*
- Nanda et al. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*
- Zhang et al. (2025). Grokking as computational glass relaxation. *arXiv:2505.11411* (NeurIPS 2025 Spotlight)
- He et al. (2026). Three-stage grokking and Fourier mode competition. *arXiv:2602.16849*
- Cullen et al. (2026). Grokking as a phase transition between competing near-zero-loss basins. *arXiv:2603.01192*
- Xu (2026a). Early-warning signals of grokking via loss-landscape geometry. *arXiv:2602.16967*
- Xu (2026b). Low-dimensional and transversely curved optimization dynamics in grokking. *arXiv:2602.16746*
- Wetterich (1993). Exact evolution equation for the effective potential. *Physics Letters B*
- Zamolodchikov (1986). Irreversibility of the flux of the renormalization group in a 2D field theory. *JETP Letters*

---

## License

MIT License. See [LICENSE](LICENSE) for details.
