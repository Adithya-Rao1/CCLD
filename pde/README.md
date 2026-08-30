# Experiment 2 (physics): does CSHO cross-field coupling improve multiphysics surrogate PDE solving?

Tests the N-way CSHO drift on two multi-field coupled-PDE benchmarks:

- [Multiphysics_Bench](https://huggingface.co/datasets/Indulge-Bai/Multiphysics_Bench)
  (arXiv:[2505.17575](https://arxiv.org/abs/2505.17575)), a 6-system FEM-generated coupled-PDE
  benchmark.
- [PDEBench](https://github.com/pdebench/PDEBench)'s 2D diffusion-reaction dataset, a
  2-field (activator/inhibitor) reaction-diffusion system.

Each problem's output field groups (e.g. velocity, temperature, activator/inhibitor
concentration) are modeled as CSHO populations, coupled via the same mean-field/pairwise drift
terms validated in experiment 1, against DDPM and VP-SDE baselines that model the same fields
jointly but without CSHO's coupling.

## 1. Getting the data

Two separate download scripts, one per data source.

**Multiphysics-Bench** (`TE_heat`, `NS_heat`, `E_flow`, `MHD`, `VA`, `Elder`) -- ships on Hugging
Face's `main` branch as two WebDataset tar.gz archives (`training.tar.gz` ~28.9GB,
`testing.tar.gz` ~2.9GB; **~32GB total**). Each tar member's internal path already matches the
`{split}/{problem}/{field}/{sample_idx}.mat` directory layout the dataset loader expects, so
extraction reproduces the target tree directly.

```
python -m pde.download_multiphysics_bench --out-dir /data/multiphysics --splits train,test
```

Use `--splits train` or `--splits test` to fetch only one archive. Set `HF_TOKEN` in the
environment if the repo requires auth. This is a large, multi-hour download -- not something to
run in a smoke-test context.

**PDEBench diffusion-reaction** (`diffusion_reaction`) -- a single HDF5 file
(`2D_diff-react_NA_NA.h5`) hosted on [DaRUS](https://darus.uni-stuttgart.de/), not Hugging Face.
Downloaded via a streaming HTTPS GET and verified against its published md5
(`b8d0b86064193195ddc30c33be5dc949`).

```
python -m pde.download_pdebench --out-dir /data/pdebench
```

## 2. The 7 problems

| short code | physics | input | output field groups | coupling |
|---|---|---|---|---|
| `TE_heat` | Electro-Thermal | `mater` | `Ez` (2, complex) + `T` (1) | bidirectional |
| `NS_heat` | Thermo-Fluid | `Q_heat` | `u` (2) + `T` (1) | bidirectional |
| `E_flow`  | Electro-Fluid | `kappa` | `V` (1) + `u` (2) | unidirectional |
| `MHD`     | Magneto-Hydrodynamic | `Br` | `J` (3) + `u` (2) | bidirectional |
| `VA`      | Acoustic-Structure (Vibro-Acoustic) | `rho_water` | `p` (2, complex) + `S` (6, complex) + `x` (4, complex) | bidirectional |
| `Elder`   | Mass-Transport-Fluid | `S_c` + t0 state (4ch) | `u` (2 fields x 10 rollout steps) + `c` (10 rollout steps) | bidirectional, transient |
| `diffusion_reaction` | Diffusion-Reaction (PDEBench) | initial activator/inhibitor snapshot | up to 4 tasks (`f1`/`f2`/`u1`/`u2`, two snapshots each of the activator/inhibitor fields) | bidirectional |

(Full names and `coupling` labels are sourced from `PROBLEM_SPECS` /
`DIFFUSION_REACTION_METADATA` in `pde/dataset.py`, not hand-transcribed.)

## 3. Task

We aim to predict each problem's output fields from its boundary/source/initial conditions to
compare with the Multiphysics-Bench paper's PINN/FNO/DeepONet/DiffusionPDE benchmark (and, for
`diffusion_reaction`, PDEBench's own baselines). All methods (CSHO variants and the DDPM/VP-SDE
baselines) model every output field of a problem jointly, in order to highlight CSHO's mean-field
coupling term's advantage in facilitating cross-field information exchange.

### N-sweeps: `--n-tasks` / `--task-subset`

Every problem supports overriding how many of its native output fields are modeled jointly:

- `--n-tasks N` selects the first `N` fields, resolved through a small `TASK_SUBSETS` lookup
  table in `dataset.py` for problems/counts that don't have an obvious "first N" (e.g.
  `TE_heat` at N=2, `diffusion_reaction` at N=2). Passing an `N` with no defined subset raises
  a clear error telling you to use `--task-subset` explicitly instead.
- `--task-subset field1,field2,...` bypasses that lookup entirely and selects exactly the named
  fields (which must be among the problem's native task names). This is what `pde/ablations.py`'s
  `n_fields` axis uses to grow MHD's field count (`Jx` -> `Jx,Jy` -> ... -> `Jx,Jy,Jz,u_u,u_v`)
  one field at a time.

## 4. Running experiments

```
python -m pde.run_experiment \
  --config pde/config.yaml \
  --data-root /data/multiphysics \
  --problem NS_heat \
  --method csho \
  --seeds 0,1,2,3,4 \
  --out-dir results/experiment_2_physics/NS_heat_csho
```

For `diffusion_reaction`, point `--data-root` at the PDEBench download directory instead
(`/data/pdebench`).

`--method` is the same roster as experiment 1: `csho`, `csho_independent`, `csho_shared_g`,
`csho_independent_g`, `csho_pairwise`, `ddpm`, `sdm`.

## 5. Metrics

Per task/field, both:

- `{field}_rel_l2` -- relative L2 error, `||pred - target|| / ||target||`.
- `{field}_spectral_l2` -- relative L2 error in the 2D Fourier-magnitude domain (captures
  spectral/frequency-content mismatches that spatial L2 can miss).

For `Elder` specifically (a 10-step rollout), additional aggregate-per-rollout-step metrics:

- `elder_rollout_step{1..10}_rel_l2` -- relative L2 error averaged across `Elder`'s 3 rollout
  fields (`u_u`, `u_v`, `c_flow`) at each of the 10 predicted timesteps, so rollout error growth
  over the horizon is visible independent of the per-field metrics above.

Plus the standard diagnostics: `nan_events`, `explosion_events`, `n_train_steps`.

## 6. Running the ablation grid

```
python -m pde.ablations \
  --data-root /data/multiphysics \
  --seeds 0,1,2,3,4 \
  --axes n_fields,coupling_mode,alpha_beta,damping_regime,diffusion_mode,problem \
  --out-dir results/experiment_2_physics/ablations
```

Six sweep axes:

1. `n_fields` -- field-count sweep (via `--task-subset` growth on MHD's 5 native real-valued
   fields), the exp2 analogue of `image/ablations.py`'s `sweep_n_tasks`.
2. `coupling_mode` -- `csho` / `csho_pairwise` / `csho_independent`.
3. `alpha_beta` -- alpha and beta grids (same pattern as experiment 1).
4. `damping_regime` -- `underdamped` / `critically_damped` / `overdamped`.
5. `diffusion_mode` -- `csho_shared_g` / `csho_independent_g`.
6. `problem` -- **pde-specific**: sweeps across all 7 problems at a fixed method/config, and
   records each problem's `coupling` label (bidirectional vs. unidirectional) alongside its
   metrics. This is a real falsification opportunity for the coupling hypothesis: coupling is
   expected to help under genuine bidirectional physics (`TE_heat`, `NS_heat`, `MHD`, `VA`,
   `Elder`, `diffusion_reaction`) and to be neutral under `E_flow`'s unidirectional physics
   (electric field drives flow, but flow doesn't feed back into the electric field).

`--n-fields-problem` (default `MHD`) and `--default-problem` (default `TE_heat`, used as the
fixed problem for every axis except `problem`) can be overridden if you want the other axes to
run against a different problem.

## 7. SDE machinery and a known open limitation

CSHO's SDE machinery here now matches `synthetic/`'s finalized version: `K_self`/`K_global`
(confinement/coupling stiffness) are fixed constants (`--k-reference`), not learned from data --
this is what makes the exact closed-form transition kernel valid, the same role `K_REFERENCE`
plays in `synthetic/`. Concretely: mode-decoupled critical damping
(`synthetic.drift_coupled_gamma.calibrate_coupled_gammas`, symmetric/antisymmetric
`gamma_self`/`gamma_couple`), the Anderson-corrected reverse SDE
(`synthetic.anderson_sde.anderson_em_step_coupled_gamma`/`anderson_reverse_step_coupled_gamma`),
Tikhonov-regularized DSM (`synthetic.exact_dsm.sample_and_tikhonov_score_target`, `--lam`) in
place of the old `ndsm_loss_n`, and exact matrix-recursion noise calibration
(`synthetic.exact_dsm.calibrate_sigma_for_leak`, `--leak-fraction`) in place of the old flat
`--sigma`.

**Known open limitation:** the calibration's `sigma_x,true^2` and `rho_true` (the empirical
per-field variance and mean pairwise field correlation the leak formula needs) are estimated via
`torch.cov` from one batch of real data encoded through `PhysicsBackbone`/`FieldHead` **at model
initialization, before any training** -- not recomputed as the jointly-trained encoder's latent
distribution shifts over the course of training. This is analogous to a one-time
initialization-scale choice (e.g. Xavier/He init) rather than a moving recalibration, and hasn't
been tested to see whether it actually matters. **Come back and check**: run a real experiment,
then compare the calibrated `sigma` value (and the resulting KL/rel_l2 metrics) against a version
recalibrated from a late-training/converged encoder snapshot instead of the initial one -- if
they're close, the simplification is fine as-is; if the encoder's output scale drifts
significantly during training, this may need periodic recalibration (e.g. once per epoch) rather
than the current once-per-run calibration.

## Important notes

- `Elder`'s 10-timestep rollout is loaded as 30 extra output channels (3 fields x 10 steps)
  rather than adding a time axis to the CSHO integrator itself, keeping it consistent with the
  other problems' single-shot framing at inference (the rollout-step metrics above are computed
  post hoc from those 30 channels, not via iterative rollout during sampling).
- `MultiPhysicsFieldDataset` supports `--max-samples` to cap dataset size for quick runs; real
  Multiphysics-Bench sets are ~10k train / ~1k test per problem, native 128x128 resolution.
- `diffusion_reaction` has no native train/test split in its single PDEBench HDF5 file; the
  dataset loader creates a deterministic 90/10 split by sample index (first 90% -> `training`,
  last 10% -> `testing`/`val`) so `--split` and `--val-split` are guaranteed disjoint.
- `csho_pairwise`'s default field-coupling matrix is uniform since there's no obvious
  "more/less related" prior between, for instance, a problem's velocity and temperature fields.
