# Generalizing across coupling strengths.
Every experiment in this repository sets the coupling strength to 0.6. Thus we propose running a
sweep across different coupling strengths using a held-out split fixed before any run. The primary
comparison is CCLD against CLD (`--beta 0`, method-label `ccld_independent`). Since these both have velocity augmentation, critical damping, and exact utilize the Van Loan sampler, they are the most natural comparison. Thus, this experiment isolates the correlation between coupling strength and the effects of coupling specifically, rather than confounding it with the architectural differences DDPM/SGM
also carry. However, DDPM and SGM will still be trained at each strength and reported alongside CCLD/CLD, but do not determine the pass/fail criteria below. We believe this is the most logical successor protocol as significance within the results can help quantify the effectiveness of introducing coupling within dynamics.

## Experiment info

- **Ablation values**: `coupling_strength ∈ {0.0, 0.3, 0.9}`. `0.0` is a boundary condition where no coupling exists. Thus, CCLD should show no advantage here. Next, `0.3`/`0.9` bracket the established `0.6` point on both sides. We hypothesize a negative correlation between the coupling strength and KL divergence/cross asymmetrical MAE.
- **Ground-truth seed**: pinned at `gt_seed=0` for every strength to match initial experiments.
- **Training seeds**: `0..9` (10 seeds) per (method, coupling_strength) cell. Gives paired-Wilcoxon its
maximum attainable two-sided significance of `p=0.00195` at `n=10`.
- **Numerical reference**: the closed-form analytic stationary Gaussian of the ground-truth coupled-OU process. Every error metric below will measure the distance to this target.
- **Sampling budget parameters**: `n_diff_steps=32`, `n_train_iters=10000`, `n_samples=40000`
- **Primary error metric**: KL divergence to the true stationary Gaussian. Moreover, Wasserstein-2 and pairwise-correlation MAE are also reported, but do not independently characterize the pass/fail criteria.
- **Primary comparison and failure condition**: CCLD vs. CLD at `coupling_strength ∈ {0.3, 0.9}` — a
  fail occurs if CCLD's mean KL is not lower than CLD's mean KL, or the paired Wilcoxon test does not reach `p≤0.05` at `n=10` seeds. Conversely, since no coupling exists at `coupling_strength=0.0`, a fail there occurs if CCLD instead shows a statistically significant advantage over CLD since there is no real coupling signal for CCLD's extra coupling term to exploit. 
- **Compute**: `3 held-out coupling strengths (0.0, 0.3, 0.9) x 10 seeds x 4 methods (CCLD, CLD, DDPM, SGM)` = 120 training runs. The completed antisymmetric-coupling sweep ran 1600 runs at a comparable per-run budget, so this is well under 1/10th that scale. Thus, this defines a small but useful extension to guide future experiments.
- **Compute cap**: ~7 A100-hours. This figure is extrapolated from 800 runs taking ~15 hours on one A100 (i.e. ~1.1 min/run average). This protocol's 120 runs at that per-run rate extrapolate to ~2.25 hours. To account for any discrepancies, we apply a  ~3x safety margin to get ~7 hours.
- **Duplication check**: Every coupled-OU result currently in `results/` was checked for its `coupling_strength` field and all of them are at `coupling_strength=0.6`. No existing result covers `coupling_strength ∈ {0.0, 0.3, 0.9}` on this system. Thus, this protocol does not duplicate completed work.

## Commands

Repeat the block below for each held-out `<VALUE>` in `{0.0, 0.3, 0.9}`:

```
STRENGTH=<VALUE>
BASELINE_DIR="results/experiment_3_synthetic/coupling_strength_generalization/baselines_strength_${STRENGTH}"
CCLD_OUT_DIR="results/experiment_3_synthetic/coupling_strength_generalization/strength_${STRENGTH}"
CLD_OUT_DIR="${CCLD_OUT_DIR}_independent"

# DDPM + SGM baselines
for METHOD in ddpm sdm; do
  python -m synthetic.run_experiment \
    --method ${METHOD} --N 5 --coupling-strength ${STRENGTH} \
    --time-scale-schedule vp_linear --n-diff-steps 32 \
    --n-train-iters 10000 --n-samples 40000 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --out-dir "${BASELINE_DIR}/N5_${METHOD}"
done

# CCLD 
python -m synthetic.anderson_analytic_n_sweep \
  --seeds 0,1,2,3,4,5,6,7,8,9 --n-train-iters 10000 --n-samples 40000 \
  --n-sweep 5 --n-diff-steps 32 --coupling-strength ${STRENGTH} \
  --sampler exact --baseline-dir "${BASELINE_DIR}" \
  --out-dir "${CCLD_OUT_DIR}"

# CLD 
python -m synthetic.anderson_analytic_n_sweep \
  --seeds 0,1,2,3,4,5,6,7,8,9 --n-train-iters 10000 --n-samples 40000 \
  --n-sweep 5 --n-diff-steps 32 --coupling-strength ${STRENGTH} \
  --beta 0 --method-label ccld_independent --sampler exact --baseline-dir "${BASELINE_DIR}" \
  --out-dir "${CLD_OUT_DIR}"

# CCLD-vs-CLD significance 
python -m synthetic.compare_coupled_vs_uncoupled \
  --seeds 10 --iters 10000 --n-samples 40000 --step-counts 32 \
  --coupled-dir-template "${CCLD_OUT_DIR}" \
  --independent-dir-template "${CLD_OUT_DIR}" \
  --out-dir "results/experiment_3_synthetic/coupling_strength_generalization/ccld_vs_cld_strength_${STRENGTH}"
```

Note: `anderson_analytic_n_sweep.py`'s `--baseline-dir` must point at a directory already
containing `N5_ddpm/ddpm_results.json` and `N5_sdm/sdm_results.json` for the matching
`coupling_strength`. Hence, train DDPM/SGM first, per strength, then run CCLD/CLD.
