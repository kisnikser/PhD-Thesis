# E3 (chapter 3) — pre-registration, fixed 2026-10-05 before any run
MNIST, MLP-L-h (ReLU, biases), L ∈ {2, 3, 4, 6}, h ∈ {64, 128}; runs r = 0..4 (seed-r permutation; J = 2000 fresh objects = last 2000 of the permutation);
k ∈ {256, …, 16384} (×2); Adam 1e-3, batch 128, max(3000, 100 epochs) steps. Quantities: see `e3_constants.py` docstring.
Verdicts (medians over runs):
- Q1 universal order: for k ≥ 1024, C1 and C2_full(σ = 0.01) vary by at most 2× across k for every architecture.
- Q2 prequential identity: gap_fresh / gap_test ∈ [0.9, 1.1] for every k ≥ 1024 and architecture.
- Q3 architecture: C2_full(0.01) at k = 16384 is monotone in L for each h; Spearman(C2_full(0.01), bound) over architectures ≥ 0.5.
- Q4 subspace: kappa_top_10 ≥ 10 · kappa_rand_10 for every architecture at k = 16384; report kappa_top / kappa_oracle (no verdict).
