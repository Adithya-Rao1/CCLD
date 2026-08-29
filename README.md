# CoupledSHO

A validation suite for an N-way generalization of a hypoelliptic, critically-damped-Langevin-style
("coupled-SHO") diffusion drift. This repo does not ship a production model -- it's a harness for
testing whether a specific coupling mechanism between diffusion tasks actually helps.

The drift is a second-order SDE over N populations: `dX = V dt`, `dV = f(X, V, K, t) dt + G dW`,
where noise only enters through velocity, and each population's restoring force pulls it toward a
(mean-field- or pairwise-) weighted combination of the *other* populations. The production version
of this drift is hardcoded for exactly two populations (e.g. ligand/protein). `core/drift.py`
generalizes it to arbitrary N >= 2, provably reducing to the original 2-population drift at N=2
under default mean-field coupling.

## The claims under test

| Claim | Tested by |
|---|---|
| Coupling helps when populations are genuinely dependent, and is neutral/harmful when they aren't | `coupling_mode` ablation axis (all 3 experiments) vs. `csho_independent`; `synthetic/`'s `coupling_strength` axis; `pde/`'s `problem` axis (bidirectional vs. unidirectional physics) |
| The advantage scales with N | `n_tasks`/`n_populations`/`n_fields` ablation axis (all 3 experiments) |
| Critical damping is the sweet spot, even once cross-task coupling is added | `damping_regime` ablation axis (all 3 experiments), realized `zeta_i(t)` logged via `core/damping.py` |
| Shared vs. per-population diffusion noise `G` matters | `diffusion_mode` ablation axis (all 3 experiments) |
| The linearized drift stays hypoelliptic (Kalman-rank / finite-horizon Gramian controllable) across the full ablation grid | `core/hypoellipticity.py::hypoellipticity_check`, run automatically in `synthetic/run_experiment.py` for every CSHO config, and standalone in `tests/test_hypoellipticity.py` |

Every experiment compares CSHO-N against the same score network with two baselines swapped in for
the drift/schedule: DDPM (discrete cosine-beta ancestral sampling, no coupling) and SDM (continuous
VP-SDE, no coupling) -- see `core/baselines.py`.

## Repo layout

| Path | What it is |
|---|---|
| `core/` | The drift, SDE integrator, baselines, damping calibration, hypoellipticity check, diagnostics, stats, and reporting. No experiment-specific logic. |
| `image/` | Experiment 1 -- multi-task dense vision prediction (depth, normals, segmentation, edges, saliency) on PASCAL/NYUDv2/Cityscapes. |
| `pde/` | Experiment 2 -- multi-physics coupled-PDE surrogate solving on Multiphysics-Bench (6 systems) + PDEBench diffusion-reaction. |
| `synthetic/` | Experiment 3 -- coupled Ornstein-Uhlenbeck processes with a **closed-form** Gaussian ground truth. No downloads, no GPU needed. |
| `tests/` | Unit tests for `core/` plus a smoke test that exercises all three experiments end-to-end on tiny synthetic fixtures. |
| `run_all.py` | Orchestrates all three experiments; `--mode smoke` for a fast no-download pass, `--mode full` for the real ablation grids. |

Each experiment directory has its own README with exact commands, data requirements, and metrics.

**Current status:** `synthetic/` is the only experiment that's runnable end-to-end right now (no
downloads, no GPU required, closed-form ground truth) and is the one this README's Quickstart
covers. `image/` and `pde/` are code-complete and import cleanly but haven't been run against real
data yet -- see their own READMEs and "Known gaps" below.

## `core/` API map

| Module | Provides |
|---|---|
| `core/drift.py` | `drift_fn_n` -- the N-way coupled drift itself. |
| `core/coupling.py` | `build_coupling_matrix` (`mean_field` / `pairwise` / `independent`), `task_similarity_to_coupling`. |
| `core/sde.py` | `build_g_matrix_n`, `em_step_n` (forward), `reverse_step_n` (generative), `ndsm_loss_n` (N-way denoising score matching). |
| `core/damping.py` | Damping-regime calibration (`calibrate_gammas_for_regime`) and realized damping-ratio diagnostics. |
| `core/baselines.py` | DDPM and VP-SDE (SDM) baseline schedules/steps, sharing the same score network as CSHO. |
| `core/hypoellipticity.py` | Controllability-Gramian check on the linearized drift; small-time variance-scaling diagnostic. |
| `core/diagnostics_bridge.py`, `core/diagnose_drift.py` | Gradient/activation hooks, drift/NaN detection during training. |
| `core/stats.py` | Multi-seed aggregation with 95% CIs, paired Wilcoxon significance testing (`compare_configs`). |
| `core/reporting.py` | CSV/JSON/Markdown output, bar and ablation-curve plots (`render_experiment_report`). |

## Install

```
conda create -n coupledsho python=3.10
conda activate coupledsho
pip install -e .
```

This is a pip-installable package (`pyproject.toml`) -- `pip install -e .` installs the repo
editable, plus all dependencies (torch, torchvision, numpy, scipy, pyyaml, matplotlib, pillow,
pyarrow, h5py, huggingface_hub, requests). Install torch for your platform/CUDA build first
(https://pytorch.org/get-started/locally/) if you need a specific one -- otherwise pip resolves
whatever default build PyPI gives it.

`requirements.txt` is kept as an alternative if you'd rather not install the package itself:
`pip install -r requirements.txt`.

Both pin `numpy<2`: torchvision/pyarrow/h5py wheels are commonly built against the NumPy 1.x ABI,
and NumPy 2.x can fail at import with `A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.x`. If you already have NumPy 2.x installed in the target environment, downgrade it
(`pip install "numpy<2"`) rather than overriding this pin.

## Quickstart: synthetic experiments

`synthetic/` is the only experiment that's runnable end-to-end right now -- closed-form Gaussian
ground truth, no downloads, no GPU required. (`image/` and `pde/` are code-complete but need real
datasets downloaded first; see "Per-experiment READMEs" below.) There are two ways to run it:

### Option A -- one config via the CLI

```
python -m synthetic.run_experiment \
  --N 5 \
  --coupling-strength 0.6 \
  --method csho \
  --damping-regime critically_damped \
  --seeds 0,1,2,3,4 \
  --device cpu \
  --out-dir results/experiment_3_synthetic/csho_N5
```

`--method` is one of `csho`, `csho_independent`, `csho_pairwise`, `ddpm`, `sdm`. Pass `--quick` for
a several-second CPU smoke run. Run `python -m synthetic.run_experiment --help` for the full flag
reference (schedule, noise, damping, training, evaluation), or see
[`synthetic/README.md`](synthetic/README.md) for a walkthrough.

### Option B -- the full N=2..5 CSHO-vs-DDPM-vs-SDM sweep

```
bash run_final_synthetic_experiments.sh [N_SEEDS] [N_TRAIN_ITERS] [N_SAMPLES]
```

Defaults: `20 10000 40000` (i.e. `bash run_final_synthetic_experiments.sh` with no args uses
these) -- chosen for statistical rigor: paired Wilcoxon can reach p<0.001 instead of the p=0.0625
floor at 5 seeds, and 40000 eval samples keeps the KL estimator's own noise well below the gap
between methods. It trains DDPM and SDM baselines and the champion CSHO configuration
(Anderson-corrected reverse SDE, mode-decoupled critical damping, Tikhonov-regularized DSM,
per-N-calibrated noise) at N=2,3,4,5, all at the same seed/iteration/sample budget, and writes the
final comparison table to
`results/experiment_3_synthetic/final_champion_seeds<N>_iters<I>/tikhonov_n_sweep_summary.csv`.
Safe to re-run or resume after an interruption -- it skips any baseline already generated. Uses
CUDA automatically if available; expect several hours at the default budget on even a powerful GPU
(e.g. an A100), since the score nets are small MLPs and the bottleneck is Python-loop overhead
across the per-population tensor structure, not GPU compute.

### Sanity checks

Cheap, and catch a different class of problem before committing to either run above:

```
python -m tests.test_hypoellipticity        # controllability-Gramian check across the full N/coupling/damping grid
python -m tests.test_sde_integrator          # em_step_n / reverse_step_n / ndsm_loss_n shape & finiteness
python -m tests.test_drift_properties        # drift_fn_n invariants: permutation equivariance, coupling separability, etc.
python -m tests.test_experiments_smoke       # all 3 experiments, on tiny in-memory fixtures
python run_all.py --mode smoke               # exp1/exp2 skip (no data root configured), exp3 runs for real
```

### `image/` and `pde/`, once you have real data

```
python -m image.run_experiment --config image/config.yaml --data-root /data/nyudv2 --method csho ...
python -m pde.run_experiment --config pde/config.yaml --data-root /data/multiphysics --method csho ...
python run_all.py --mode full --vision-data-root /data/nyudv2 --physics-data-root /data/multiphysics
```

## Per-experiment READMEs

- **[`image/README.md`](image/README.md)** -- vision multi-task. Data: NYUDv2 (HF, no login) /
  PASCAL VOC 2012 (Ultralytics mirror, no login) / Cityscapes (registration required). GPU
  recommended for real runs; runtime scales with `--image-size`/`--n-epochs`/dataset size (no
  general estimate given here -- depends heavily on your hardware and config).
- **[`pde/README.md`](pde/README.md)** -- multi-physics PDE. Data: Multiphysics-Bench (~32GB, HF,
  WebDataset tar.gz) + PDEBench diffusion-reaction (DaRUS, single HDF5 file). Large download; not
  something to do casually.
- **[`synthetic/README.md`](synthetic/README.md)** -- coupled-OU, closed-form ground truth. No
  download, no GPU needed; runs in seconds on CPU even without `--quick`.

## Ablation axes

Every experiment shares 5 axes (`n_tasks`/`n_populations`/`n_fields`, `coupling_mode`,
`alpha_beta`, `damping_regime`, `diffusion_mode`) plus one experiment-specific axis:
`synthetic/`'s `coupling_strength` (sweeps the *ground truth's* true coupling) and `pde/`'s
`problem` (sweeps across all 7 physics problems, labeled by bidirectional/unidirectional coupling).
Each `ablations.py` uses `core/stats.py::compare_configs` (paired Wilcoxon across seeds) against
the first config on each axis as baseline -- pass `--seeds` with >=2 seeds for the significance
columns to populate.

## Output conventions

Every `run_experiment.py` writes, per method, to `--out-dir`: `{method}_summary.csv`,
`{method}_results.json` (full per-seed + aggregated), `{method}_summary.png`, `{method}_report.md`.
Every `ablations.py` writes `ablation_results.csv`, `ablation_significance.csv`,
`ablation_results.json`, `ablation_report.md`. `run_all.py` writes the same triple prefixed
`run_all_*` at the top level, aggregating whichever experiments actually ran.

## Reproducibility

Default 5 seeds per config, mean +/- 95% CI via `core/stats.py::aggregate_over_seeds`, paired
Wilcoxon significance vs. baseline configs via `compare_configs`. `synthetic/`'s `--gt-seed`
defaults to the run seed (so the ground truth itself varies per seed) unless pinned explicitly.

## Known gaps

- No GPU-scale results yet -- everything here has been verified for correctness (imports cleanly,
  runs end-to-end, produces finite/sane metrics) on small synthetic/CPU configurations, not
  validated at the scale/compute the real ablation grids need.
- `pde/`'s Multiphysics-Bench and PDEBench downloads, and `image/`'s three dataset downloads, have
  not been executed as part of building this repo -- the download scripts are code-complete and
  `--help`-clean, but untested against the live data end-to-end.

## License

MIT (see `LICENSE`). The datasets used by `image/` and `pde/` (NYUDv2, PASCAL VOC, Cityscapes,
Multiphysics-Bench, PDEBench) carry their own separate licenses/terms, not covered by this repo's
license -- check each before redistributing anything derived from them.
