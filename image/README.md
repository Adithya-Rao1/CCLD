# Experiment 1 (vision): does CSHO cross-task coupling help multi-task diffusion?

Tests the N-way CSHO drift from `core/` on dense multi-task vision prediction
(depth, surface normals, segmentation, edges, saliency) against DDPM and VP-SDE baselines, and
against CSHO ablations (independent coupling, independent diffusion noise, pairwise task-similarity
coupling), across the damping regimes the drift supports.

## 1. Getting the data

No Kaggle account is needed for any of the three sources below.

**NYUv2** (rgb + depth + semantic segmentation masks, from
[jagennath-hari/nyuv2](https://huggingface.co/datasets/jagennath-hari/nyuv2) on Hugging Face,
MIT licensed, public):

```
python -m image.download_nyudv2 --out-dir /data/nyudv2
```

Only needs `huggingface_hub` (already a repo dependency); set `HF_TOKEN` in the environment if you
ever hit a rate limit or the dataset becomes gated.

**PASCAL VOC 2012** (segmentation, from the same archive mirror Ultralytics' `VOC.yaml` uses --
see [docs.ultralytics.com/datasets/detect/voc](https://docs.ultralytics.com/datasets/detect/voc)):

```
python -m image.download_pascal --out-dir /data/pascal
```

Downloads `VOCtrainval_11-May-2012.zip` by default (segmentation only needs VOC2012); pass
`--parts trainval2007,test2007,trainval2012` for the full Ultralytics detection.

**Cityscapes** (segmentation, login required):

```
export CITYSCAPES_USERNAME=you
export CITYSCAPES_PASSWORD=yourpassword
python -m image.download_cityscapes --out-dir /data/cityscapes
```

Register at [cityscapes-dataset.com/register](https://www.cityscapes-dataset.com/register/).
Credentials are only ever read from the environment, never accepted as CLI args.

## 2. Running an experiment

```
python -m image.run_experiment \
  --config image/config.yaml \
  --data-root /data/nyudv2 \
  --tasks depth,normals \
  --method csho \
  --damping-regime critically_damped \
  --seeds 0,1,2,3,4 \
  --out-dir results/experiment_1_vision/csho_depth_normals
```

`--method` is one of `csho` (full N-way mean-field coupling), `csho_independent` (zero coupling
matrix), `csho_shared_g` / `csho_independent_g` (shared vs. per-task diffusion noise), `csho_pairwise`
(task-similarity-weighted coupling, see `--task-similarity-json`), `ddpm`, `sdm` (VP-SDE). CLI flags
override anything set in `--config`.

## 3. Running the ablation grid

```
python -m image.ablations \
  --data-root /data/nyudv2 --source nyudv2 \
  --seeds 0,1,2,3,4 \
  --axes n_tasks,coupling_mode,alpha_beta,damping_regime,diffusion_mode \
  --out-dir results/experiment_1_vision/ablations
```

Sweeps N=2..5 tasks, coupling mode (mean_field/pairwise/independent), an alpha/beta grid, the three
damping regimes, and shared-vs-independent diffusion noise -- one factor at a time against a fixed
default config, using `core/stats.py`'s `compare_configs` (paired Wilcoxon across seeds) against
each axis's first config as baseline. Needs `--seeds` with >=2 seeds for the significance columns to
be populated (with 1 seed, `compare_configs` correctly skips them rather than fabricating a p-value).

## Important

- The model diffuses in a shared per-task latent space (a 1x1-conv projection of each task's
  initial coarse prediction), not raw pixels -- `core/sde.py`'s cross-population noise coupling
  requires every population's primary tensor to share one shape, which per-task prediction maps
  (different channel counts) don't.
- `core/baselines.py`'s DDPM schedule has `alphas_cumprod[0] == 1.0` exactly (by the cosine
  schedule's own construction) and `betas`/`alphas` one entry shorter than `alphas_cumprod`; calling
  its `ddpm_ancestral_step` at `t=0` divides by zero and at `t=n_diff_steps` is out of bounds.
  `run_experiment.py`'s DDPM eval loop stays strictly inside `[1, n_diff_steps)` and finishes with a
  closed-form x0 estimate instead -- see the comment in `evaluate()`.
- NYUv2's segmentation class count isn't documented anywhere so `dataset.py` infers it from a sample of decoded masks at construction time (override via
  `MultiTaskVisionDataset(..., num_classes=...)`).
- `csho_pairwise`'s default task-similarity matrix is hand-specified (depth<->normals weighted
  highest, saliency weakest) -- pass `--task-similarity-json` to override with a real one.