# Contributing to WHY2

Thanks for your interest. This is an independent research project — contributions, questions, and collaboration ideas are all welcome.

## Ways to Engage

**Open an issue if you:**
- Found a bug in one of the experiment scripts
- Got different results when replicating (seed variance, hardware differences, etc.)
- Have a question about the theory or experimental design
- Want to extend the framework to a new task, architecture, or dataset
- Spotted an error in the paper

**Open a pull request if you:**
- Fixed a reproducibility issue
- Added a new experiment that fits the WHY2 framework
- Improved documentation or code comments

## Replication Notes

All experiments were run on free Google Colab T4 GPU. If you're replicating on different hardware, expect minor numerical differences but the same qualitative results (β* values should match within ~0.05 for multiply/add, within ~0.3 for subtract due to basin sensitivity).

Known instabilities:
- `why2_basin_precheck.py` at WD=0.5 enters an oscillatory regime — this is expected, not a bug
- Subtract Basin B (β* ≈ 12–19) appears stochastically at WD=1.0; not all seeds hit it

## Priorities for Future Work

If you want to extend this and are looking for direction, the most valuable open problems are:

1. **Scaling to 2–4 layer transformers** — does β* remain WD-invariant in deeper models?
2. **Larger primes (p = 257, 509)** — does the β* ordering hold and does the Fourier mode correlation tighten?
3. **Non-abelian group tasks** — does the cross-term coupling w₁₂ appear at sufficient algebraic complexity?
4. **Causal interventions on β₂** — does perturbing the asymmetry trajectory change the final β*?
5. **Wetterich beta function fitting** — fitting dβ/dt trajectories to the LPA ODE to verify the full flow, not just the fixed point

## Code Style

Scripts are intentionally self-contained (no shared utils) for Colab portability. Keep new experiments self-contained for the same reason. Use `torch.manual_seed()`, `numpy.random.seed()`, and `random.seed()` at the top of every script.

## Contact

Open an issue — that's the best way to reach me.