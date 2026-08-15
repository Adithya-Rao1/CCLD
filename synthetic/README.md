# Experiment 3 (synthetic): does CSHO-N recover a known coupled-OU joint distribution?

Tests the N-way CSHO drift from `core/` against a fully synthetic ground truth with a
**closed-form** joint distribution: N coupled Ornstein-Uhlenbeck processes. Unlike experiments 1
(vision) and 2 (physics), there is no download step and no real-world label noise -- this is the
repo's theory-grounding experiment, and its results should be read as "does CSHO's coupling
mechanism work at all, under conditions where we know the right answer exactly."

## 1. Ground truth (no download required)

`ground_truth_sde.py::make_ground_truth` builds an N-dimensional coupled OU process

```
dX = -Theta X dt + Sigma dW
```

with `Theta = base_decay * I - coupling_strength * base_decay * W`, where `W` is a symmetric,
row-normalized random coupling-weight matrix. For `coupling_strength` in `[0, 1)`, `Theta` is
guaranteed Hurwitz-stable, so the process has a well-defined stationary Gaussian distribution
whose covariance is the solution to the continuous Lyapunov equation
`Theta @ Cov + Cov @ Theta.T = Sigma @ Sigma.T` (`scipy.linalg.solve_continuous_lyapunov`).
Sampling from the ground truth is exact (a single `MultivariateNormal` draw), and pairwise mutual
information between populations has a closed form in terms of the stationary correlation matrix.

This is the **only experiment needing no downloads and no GPU** -- `run_all.py --mode smoke` and
this repo's fastest correctness check both run through here.

## 2. What's being tested

CSHO-N's coupled drift (`--method csho`), against `csho_independent` (coupling disabled --
degenerates to N decoupled CLD processes), `csho_pairwise` (coupling reused from
`core/coupling.py`'s pairwise mode), and the two baselines `ddpm`/`sdm`, all trained to recover
the same known stationary distribution and evaluated against it directly.

## 3. Metrics

- `kl_divergence`, `wasserstein2` -- closed-form Gaussian KL and 2-Wasserstein distance
  (`metrics.py`) between the generated samples' fitted Gaussian and the true stationary
  distribution.
- `mi_mae` -- mean absolute error between the generated samples' pairwise mutual-information
  matrix and the ground truth's.
- `autocorr_lag1`, `integrated_autocorr_time` -- mixing diagnostics of the reverse SDE chain at a
  fixed diffusion time (`--mixing-t-fixed`, default `n_diff_steps / 2`).
- `hypo_passed`, `hypo_min_eig` -- controllability/Gramian hypoellipticity check
  (`core/hypoellipticity.py`) on the linearized drift.

`autocorr_lag1`/`integrated_autocorr_time`/`hypo_passed`/`hypo_min_eig` are **CSHO-only** -- they
depend on the CSHO drift's structure and are correctly reported as `NaN` for `ddpm`/`sdm`.

## 4. Running an experiment

```
python -m synthetic.run_experiment \
  --config synthetic/config.yaml \
  --N 5 \
  --coupling-strength 0.6 \
  --method csho \
  --damping-regime critically_damped \
  --seeds 0,1,2,3,4 \
  --device cpu \
  --out-dir results/experiment_3_synthetic/csho_N5
```

`--method` is one of `csho` (full mean-field coupling), `csho_independent` (zero coupling),
`csho_pairwise` (pairwise coupling), `ddpm`, `sdm`. Pass `--quick` for a fast CPU smoke run --
it overrides `N` (capped at 3), `n_diff_steps` (5), `batch_size` (32), `n_train_iters` (20),
`n_samples` (200), `prior_probe_samples` (64), `mixing_n_steps` (100), and `seeds` ("0") *after*
argument parsing, so passing your own values for those flags alongside `--quick` is redundant --
`--quick` wins. A `--quick` run takes seconds on CPU.

`--gt-seed` defaults to the run seed if unset, so the ground truth itself varies across seeds in a
multi-seed run; pin `--gt-seed` explicitly to hold the ground truth fixed while only the model's
training seed varies.

## 5. Running the ablation grid

```
python -m synthetic.ablations \
  --seeds 0,1,2,3,4 \
  --axes n_populations,coupling_mode,alpha_beta,damping_regime,diffusion_mode,coupling_strength \
  --out-dir results/experiment_3_synthetic/ablations
```

Six axes, one factor at a time against a fixed default config, using `core/stats.py`'s
`compare_configs` (paired Wilcoxon across seeds):

- `n_populations` -- sweeps `--N` from 2 to `--max-n` (default 5). Cheap here since the ground
  truth is closed-form, so this is the most informative N-sweep in the repo.
- `coupling_mode` -- `csho` / `csho_pairwise` / `csho_independent`.
- `alpha_beta` -- an alpha/beta grid, one factor at a time.
- `damping_regime` -- underdamped / critically_damped / overdamped.
- `diffusion_mode` -- shared vs. independent per-population diffusion noise `G`.
- `coupling_strength` -- **exp3-specific**: sweeps the *ground truth's* true coupling strength
  (`--coupling-strength-grid`, default `0.0,0.3,0.6,0.9`) while the model config (`method=csho`)
  stays fixed. This is the cleanest falsifiable claim available in this experiment: does CSHO's
  advantage over DDPM/VP-SDE grow as the populations become more genuinely dependent?

Needs `--seeds` with >=2 seeds for the significance columns to be populated.

## 6. Notes

- Requires `scipy` (Lyapunov solve) in addition to the repo's usual `torch`/`numpy`/`pyyaml`.
- Runtimes are small even at full (non-`--quick`) settings -- there's no image or PDE data, and
  the score networks are small MLPs operating on `(B, N)` tensors, not spatial fields.
- `coupling_strength` must stay in `[0, 1)`; `make_ground_truth` raises otherwise (this is what
  guarantees `Theta` is Hurwitz-stable and the stationary distribution exists).
