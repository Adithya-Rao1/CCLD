# Generalizing across coupling strengths.
Every experiment in this repository sets the coupling strength to 0.6. Thus we propose running a sweep across different coupling strengths using a held-out split fixed before any run and comparing across CCLD, DDPM, and SGM. We believe this is the most logical successor protocol as significance within the results can help quantify the effectiveness of introducing coupling within dynamics.

## Experiment info

- **Ablation values**: `coupling_strength ∈ {0.0, 0.3, 0.9}`. `0.0` is a boundary condition where no coupling exists. Thus, CCLD should show no advantage here. Next, `0.3`/`0.9` bracket the established `0.6` point on both sides. We hypothesize a negative correlation between the coupling strength and KL divergence/cross asymmetrical MAE.
- **Ground-truth seed**: pinned at `gt_seed=0` for every strength to match initial experiments.
- **Training seeds**: `0..9` (10 seeds) per (method, coupling_strength) cell. Gives paired-Wilcoxon its
maximum attainable two-sided significance of `p=0.00195` at `n=10`.
- **Sampling budget parameters**: `n_diff_steps=32, `n_train_iters=10000`, `n_samples=40000`
- **Error metrics**: KL divergence to the true stationary Gaussian, Wasserstein-2, pairwise-correlation MAE
- **Failure condition**: at `coupling_strength ∈ {0.3, 0.9}`, a fail occurs if CCLD's mean KL is not
  lower than DDPM's mean KL ot the paired Wilcoxon test does not reach `p≤0.05` at
  `n=10` seeds. Conversely, since no coupling exists at `coupling_strength=0.0`, a fail there occurs if CCLD instead shows a statistically significant advantage over DDPM.
- **Compute**: `3 held-out coupling strengths (0.0, 0.3, 0.9) × 10 seeds × 3 methods (CCLD, DDPM, SGM)` = 90 training runs. The completed antisymmetric-coupling sweep ran 1600 runs at a comparable
per-run budget. Thus, this is well under 1/15th that scale, and hence a small, but useful extension to guide future experiments.

## Commands
```
python -m synthetic.anderson_analytic_n_sweep \
  --seeds 0,1,2,3,4,5,6,7,8,9 --n-train-iters 10000 --n-samples 40000 \
  --n-sweep 5 --n-diff-steps 32 --coupling-strength 0.6 \
  --out-dir results/experiment_3_synthetic/coupling_strength_generalization/design_0.6

# Held-out strengths (repeat for 0.0, 0.3, 0.9)
python -m synthetic.anderson_analytic_n_sweep \
  --seeds 0,1,2,3,4,5,6,7,8,9 --n-train-iters 10000 --n-samples 40000 \
  --n-sweep 5 --n-diff-steps 32 --coupling-strength <VALUE> \
  --out-dir results/experiment_3_synthetic/coupling_strength_generalization/strength_<VALUE>

# DDPM/SDM baselines at the same strengths, matched compute
python -m synthetic.run_experiment \
  --method ddpm --N 5 --coupling-strength <VALUE> \
  --n-diff-steps 32 --n-train-iters 10000 --n-samples 40000 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --out-dir results/experiment_3_synthetic/coupling_strength_generalization/strength_<VALUE>_ddpm
```