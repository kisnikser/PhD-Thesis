# E4 (chapter 4) — pre-registration, fixed 2026-10-05 before any run
MNIST 14×14 (196 inputs, standardized), GELU MLPs 196–h–10 with h ∈ {16, 32} and 196–16–16–10; runs 0..4; k ∈ {250, …, 50000};
Adam 2000 steps + full-batch float64 L-BFGS, ridge 1e-5. Labels: teacher of the same architecture (exact excess = KL on the test inputs)
and, as a control, real labels (excess vs a 60k reference, 3 seeds). Quantities: see `e4_local_regime.py` docstring.
Clarification, still before the registered run (debug smoke on arch 32, k ≤ 1000, 100 Adam steps): the training-set
sandwich is degenerate under interpolation (train loss ~ 10⁻⁴, tr(H⁻¹C) ~ 10⁻² against N = 6634), because per-sample
training gradients vanish. P2 and P3 therefore use the population plug-in `pop_tr_HinvC`, estimated on the 10k test
inputs with labels from the data-generating distribution. The training-set sandwich is stored and not used for a verdict.
Verdicts (medians over runs; teacher labels):
- P1 population law: 2k·excess/N ∈ [0.67, 1.5] at every k with k/N ≥ 4.
- P2 plug-in: pop_tr_HinvC/(2k) / excess ∈ [0.5, 2] at every k with k/N ≥ 4.
- P3 size dependence: at k = 50000, pop_tr_HinvC grows across architectures with N (rank correlation with N = 1), while pop_lam_max and the bound need not; report ratios.
- Report the smallest k/N from which P1 and P2 hold (local-regime onset), training loss vs L*, and the same quantities for real labels (no verdict).
