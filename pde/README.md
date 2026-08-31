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

## 8. PDE-residual error metric (TE_heat, E_flow, VA)

Alongside `rel_l2`/`spectral_l2` (closeness to the reference solution), `TE_heat`, `E_flow`, and
`VA` evaluations now also report a **physical-validity** metric: plug the predicted output fields
into the problem's actual governing PDE and measure how much the equation is violated. Reported
per equation as `{equation}_pde_residual` (e.g. `flow_continuity_pde_residual`,
`acoustic_real_pde_residual`, `e_field_pde_residual`).

**Metric definition**: DiffusionPDE's own guidance-loss formula, verified against the actual
source (identical across all five `generate_*.py` scripts —
`generate_TE_heat.py`/`generate_E_flow.py`/`generate_NS_heat.py`/`generate_Elder.py`/
`generate_VA.py`): `L_pde = ||residual||_2 / (H*W)` — the PDE-operator residual's L2 norm over the
spatial grid, divided by pixel count, computed per sample then averaged over the batch
(`pde_residual_metric`, `pde/pde_residuals.py`). **Corrected from an earlier version of this file
and of `pde_residual_metric` that used `mean(residual**2)`** — that was never actually verified
against DiffusionPDE's source, just assumed; this is neither MSE nor MAE, and DiffusionPDE itself
has no MAE-style residual metric anywhere in its codebase (confirmed by full-repo grep) — its
evaluation scripts (`evaluate_te_heat.py` etc.) don't call their own PDE-loss functions at all,
reporting only RMSE/nRMSE/MaxError/bRMSE/fRMSE on field *values*, never on PDE residuals.

**Per-term rescaling divisors** (corrected — an earlier version of this file called these a
"PINN-only training-stability hack, deliberately dropped" and this port's residual functions
returned the raw, undivided residual; that framing was wrong). `generate_*.py` — DiffusionPDE's
own guidance-loss computation, not just Multiphysics-Bench's separate `pinns/train_*.py`
scripts — divides the raw residual by a problem/term-specific constant *before* taking the norm.
Because the L2 norm is homogeneous, dividing pre-norm is algebraically identical to dividing the
final metric, so this port applies the same divisors as named, overridable constants (default
arguments, so a future sensitivity sweep can pass different values) in `pde/pde_residuals.py`:

| Constant | Value | Source | Applies to |
|---|---|---|---|
| `TE_HEAT_E_FIELD_RESIDUAL_SCALE` | `1e6` | `generate_TE_heat.py:171` | `te_heat_residual`'s `e_field` term |
| `TE_HEAT_HEAT_RESIDUAL_SCALE` | `1e6` | `generate_TE_heat.py:172` | `te_heat_residual`'s `heat` term |
| `E_FLOW_FLOW_CONTINUITY_SCALE` | `1e3` | `generate_E_flow.py:61` | `e_flow_residual`'s `flow_continuity` term |
| `E_FLOW_CURRENT_CONTINUITY_SCALE` | `1e6` | `generate_E_flow.py:62` | `e_flow_residual`'s `current_continuity` term |
| `VA_ACOUSTIC_SCALE` | `1e6` | `generate_VA.py:127-130` | `va_residual`'s `acoustic_real`/`acoustic_imag` terms |
| `VA_STRUCTURE_SCALE` | `1e3` | `generate_VA.py:131-133` | `va_residual`'s four `structure_{x,y}_{real,imag}` terms |

Discovered because a real (non-smoke) `--score-arch fno` run reported `e_field_pde_residual`≈
9.3e7–9.8e7 and `heat_pde_residual`≈1.58e6–1.59e6 even on the working `attention`-path model —
implausibly large. Dividing those observed numbers by `1e6` lands them at ≈93 and ≈1.6, both in
the sane O(1)-O(100) range expected of a trained model's residual — matching the divisor almost
exactly and confirming this was the missing piece, not a modeling bug. (MHD/NS_heat divisors were
also found while tracing this — `1e2`/`1e2` and `1e6`/`1e3` respectively — recorded here for
completeness even though those two problems are excluded from this metric entirely, see below.)

Also flagged, not actionable but worth knowing: the arXiv paper's Eq. 9 states
`L_pde = mean(residual**2)` (MSE, no sqrt), which does **not** match what the actual vendored code
computes (`norm(residual,2)/(H*W)`, no squaring) — a paper/code discrepancy. This port matches the
*code*, consistent with this repo's established practice of trusting verified source over
descriptions.

**Governing equations, constants, and grid spacing**, extracted from Multiphysics-Bench's own PINN
loss code (`pinns/train_{te_heat,e_flow,VA}.py::get_*_loss`) and cross-validated against the
COMSOL `.m` geometry scripts in `DataProcessing/data_generate/` (in-code comments for grid
spacing are frequently stale/wrong; the numeric literals, confirmed against COMSOL geometry, are
authoritative):

- **E_flow** (`pde/pde_residuals.py::e_flow_residual`, unidirectional -- coupling should be
  neutral here, the falsification case): `div(kappa*grad(V)) = 0` (Poisson potential, `kappa` is
  the input material field) + `du_flow/dx + dv_flow/dy = 0` (flow continuity). Grid 128x128,
  `dx=dy=1.28e-3/128` (1.28mm x 1.28mm domain). Verified byte-for-byte (max abs diff `0.0`)
  against a direct reimplementation of `get_E_flow_loss`'s math on identical synthetic input.
- **VA** (`pde/pde_residuals.py::va_residual`, bidirectional, 6 residual equations across
  complex-valued fields): acoustic Helmholtz `div(grad(p)/rho_water) + omega^2*p/(rho_water*c_ac^2)
  = 0` (real & imag) + structural equilibrium `dSxx/dx + dSxy/dy + x_u = 0`,
  `dSxy/dx + dSyy/dy + x_v = 0` (real & imag each). `omega = pi*1e5` rad/s (50kHz, confirmed
  against `VA.m`'s `f='50[kHz]'`), `c_ac=1.48144e3` m/s (fixed scalar -- COMSOL's actual model
  uses a temperature-dependent `soundspeed(T)`; matched here to the benchmark's own PINN-loss
  simplification for consistency with the reference implementation, not "corrected"). `rho_water`
  is the spatially-varying input field itself, not a scalar. Grid 128x128,
  `dx=dy=(40/128)*1e-3` (40mm x 40mm domain, confirmed against `VA.m`'s `sq1` size).
- **TE_heat** (`pde/pde_residuals.py::te_heat_residual`, bidirectional -- E-field <-> temperature
  via Joule heating): complex Helmholtz `laplace(Ez) + K_E*Ez = 0` where
  `K_E = mu_r*k_0^2*(eps_r - i*sigma/(omega*eps_0))`, `sigma = q*sigma_coef*exp(-Eg/(kB*T))`, plus
  steady-state heat `rho*laplace(T) + 0.5*sigma*|Ez|^2 = 0`.

  `mater` is TE_heat's raw material-property conditioning field -- a 128x128 array per sample
  (`{split}/TE_heat/mater/{idx}.mat`), the same kind of role `kappa` plays in E_flow or
  `rho_water` in VA. Its *physical meaning switches* depending on which region of the domain a
  pixel falls in: inside the elliptical material inclusion it feeds `sigma_coef_map`, the
  coefficient behind the electrical conductivity term `sigma = q*sigma_coef*exp(-Eg/(kB*T))`;
  outside the inclusion it's instead the background's thermal conductivity `rho_map`. So `eps_r`/
  `rho`/`sigma_coef` are piecewise constants (`11.7`/`70`/`mater`-value inside, `1`/`mater`-
  value/`~0` outside) selected via a material-inclusion mask built from `elliptic_params`
  (rotated ellipse -- see the flagged upstream-bug note below for why this is a rotated ellipse
  and not a circle). Getting that mask wrong doesn't just mislabel a region -- it plugs `mater`'s
  numeric value into the wrong physics term for every misclassified pixel. That geometry is
  **not** in the standard `mater`/`Ez`/`T` `.mat` field
  dirs -- Multiphysics-Bench stores it separately, one CSV per sample, in an `ellipticcsv/`
  directory alongside them (confirmed present in the Hugging Face download); `pde/dataset.py`
  now loads it (`_init_standard`'s `_elliptic_dir`, TE_heat only) and `collate_fn` batches it as
  `elliptic_params`. Constants: `f=4e9 Hz`, `k_0=2*pi*f/3e8`, `omega=2*pi*f`, `q=1.602` (**not**
  literal SI elementary charge -- almost certainly a rescaled constant tuned for well-conditioned
  NN training in the original benchmark; matched here exactly rather than "corrected", for the
  same reasoning as VA's fixed `c_ac` above), `mu_r=1`, `eps_0=8.854e-12` F/m,
  `kB=8.6173e-5` eV/K, `Eg=1.12` eV. Grid 128x128, `dx=dy=1e-3` m (128mm x 128mm domain).
  The E-field and heat residual *formulas* (everything except the material-inclusion mask itself)
  are verified byte-for-byte (max abs diff `0.0`) against a direct reimplementation of
  `get_TE_heat_loss`'s math on identical synthetic input, using the same mask on both sides.

  Two implementation notes, both disclosed rather than silently assumed: (1) uses the raw
  `mater` field as loaded from the `.mat` file directly, in the same physical/raw units as every
  other field in this pipeline -- Multiphysics-Bench's own `compute_loss()` rescales a
  *normalized NN input* tensor back to physical units before calling `generate_separa_mater`,
  which doesn't apply here since `mater` is never normalized in this pipeline in the first place;
  sanity-check residual magnitudes against real data before fully trusting this assumption.
  (2) predicted `T` is mapped through a differentiable `softplus(T_raw) + 1.0` (not a hard
  `clamp`) before entering `exp(-Eg/(kB*T))`, so it's always strictly positive with a smooth
  gradient -- an untrained/early-training network's raw output has no constraint keeping it
  physically positive, and `T->0` blows the exponential up to inf/nan; Multiphysics-Bench never
  hits this because their own `T` always passes through a bounded rescaling first.

  **Flagged: upstream bug in Multiphysics-Bench's `identify_mater`, deliberately not matched
  here.** `elliptic_params` (`ellipticcsv/{idx}.csv`) is `[e_a, e_b, angle]` -- an ellipse's
  semi-major axis, semi-minor axis (mm), and rotation angle (degrees) -- confirmed via
  `DataProcessing/data_generate/TE_heat/generate_Elliptic_TE_heat.m` (`parm_e_a=20*rand+10`,
  `parm_e_b=10*rand+10`, `parm_angle=360*rand`, written to CSV in exactly that order) and the
  COMSOL model `TE_heat.m` (`geom1.feature('e1').set('semiaxes',{'e_a' 'e_b'})`,
  `.set('rot','angle')`, with no explicit `.set('pos', ...)` for `e1` -- unlike every other shape
  in the script -- so the ellipse is centered at the domain origin). Multiphysics-Bench's own
  `identify_mater` (`pinns/train_te_heat.py`) instead reads these three values as a **circle's**
  `(center_x, center_y, radius)`: `cx=params[:,0]`, `cy=params[:,1]`, `r=params[:,2]`,
  `(xx-cx)**2+(yy-cy)**2 <= r**2`. Since `angle` ranges up to 360 (used squared, as an effective
  radius up to ~360mm against a 128mm domain), this swallows nearly the whole grid for most
  samples. Confirmed empirically on the real dataset (`pde/compute_te_heat_mater_ranges.py`,
  10000 training samples): **~93% of all pixels classified "inside"**, with fully overlapping
  inside/outside value ranges (`inside=[10.0007, 2.99955e11]`, `outside=[10.0327, 2.98922e11]`) --
  physically nonsensical for a small inclusion in a larger domain. `pde/pde_residuals.py::
  _te_heat_mater_iden` therefore implements the **geometrically correct** rotated-ellipse test
  (rotate the query point by `-angle` into the ellipse's own frame, then
  `(x_local/e_a)^2 + (y_local/e_b)^2 <= 1`) instead of matching `identify_mater`'s formula -- a
  deliberate, documented exception to the "match the benchmark exactly" principle used everywhere
  else in this port, since the original formula doesn't locate the actual material boundary.
  `e_a`/`e_b` are assumed to be in the same mm-based COMSOL length unit as the domain's other
  geometry (128mm/148mm squares use the same bare-number convention); this assumption, and the
  rotation-direction (CW vs CCW) sign convention, were not independently pixel-verified against a
  rendered COMSOL image and should be sanity-checked if TE_heat results look physically off.
  **Action item for the user**: fork Multiphysics-Bench and report this upstream.

  **Verification of the fix**, two independent ways: (1) on the real `ellipticcsv/1.csv` sample
  (`e_a=18.518mm, e_b=13.005mm, angle=320.05deg`), the corrected test classifies 4.59% of pixels
  "inside," matching the analytic ellipse-area prediction (`pi*e_a*e_b / domain_area = 4.62%`)
  almost exactly. (2) `TE_HEAT_MATER_INSIDE_RANGE`/`TE_HEAT_MATER_OUTSIDE_RANGE` (used by
  `te_heat_normalize_mater`) were recomputed against the real dataset with the corrected geometry
  via `pde/compute_te_heat_mater_ranges.py` (10000 training samples,
  `/home/ubuntu/metis-v1-storage/CSHM-data/multiphysics`): `inside=[1.00068e11, 2.99955e11]`,
  `outside=[10.0007, 19.9967]` -- cleanly disjoint, unlike the buggy circle geometry's run (which
  gave fully overlapping ranges, `inside=[10.0007, 2.99955e11]` vs `outside=[10.0327, 2.98922e11]`).
  Note: that run reported `testing: 0 samples found` for TE_heat -- root-caused and handled, see
  "TE_heat eval fix" below.
- **MHD and NS_heat -- excluded from this metric entirely, documented discrepancy.** Verified
  byte-for-byte across three copies of `get_MHD_loss` (`pinns/train_MHD.py`, `evaluate_MHD.py`,
  `DiffusionPDE/scripts/generate_MHD.py`): `Br` and `Jz` are accepted as function parameters but
  **never referenced in the residual body** -- the implementable residual is just two decoupled
  divergence-free constraints (`div(u)=0`, `div(J)=0`), with no Lorentz-force/Ampere's-law
  coupling term, despite the paper's stated equations including one. `get_NS_heat_loss` has the
  same pattern -- no momentum equation anywhere, just continuity + convection-diffusion heat
  transport. Both are real discrepancies between the paper's stated physics and the benchmark's
  own verified reference implementation, not something missed in extraction. MHD stays in the
  existing `rel_l2`/`spectral_l2` comparison (the real data reflects genuine MHD physics even
  though this residual check doesn't); NS_heat was never in scope for this metric.

## 9. Score-network architecture ablation (`--score-arch`)

`--score-arch {attention, fno, songunet}` (default `attention`) selects the network that predicts
scores during CSHO/DDPM/SDM training and sampling, for `TE_heat`, `E_flow`, `VA`. `attention` is
the original, unchanged pipeline (`pde/model.py::PhysicsModel`/`MultiPhysicsScoreNetwork` -- a
pooled per-task latent vector `(B, latent_dim)`, refined via cross-field attention). `fno` and
`songunet` are new, and share the same **native-pixel diffusion** state representation: each
task's diffused state is the full field itself, `(B, 1, H, W)`, with no latent-vector bottleneck.
`fno` denoises with an FNO (`pde/fno_score_net.py`, needs `neuraloperator`:
`pip install neuraloperator`); `songunet` denoises with DiffusionPDE's `SongUNet`
(`pde/songunet_score_net.py`, using `pde/vendored_songunet.py` -- no extra dependency).

`pde/vendored_songunet.py` is a **plain copy** of `SongUNet` and its dependencies
(`UNetBlock`, `AttentionOp`, `PositionalEmbedding`, `FourierEmbedding`, etc.) from
`pde/multiphysics-bench/DiffusionPDE/training/networks.py`, with the `@persistence.persistent_class`
decorators dropped (a `torch_utils`/`dnnlib`-dependent pickle-portability mechanism for
DiffusionPDE's own checkpoint format, never used here). **Not imported from the clone directly**
at runtime, because `pde/multiphysics-bench` is a nested git repo the outer CoupledSHO repo only
tracks as a gitlink (`git ls-tree` shows mode `160000`, no `.gitmodules`) -- a plain
`git clone`/`git pull` of this repo does not bring that nested repo's file content along, which
broke `--score-arch songunet` with `ModuleNotFoundError: No module named 'training'` on a fresh
remote checkout. Vendoring the specific classes actually needed as tracked files sidesteps this
entirely; `pde/multiphysics-bench` is still cloned locally for reference/verification (e.g.
`pde_residuals.py`'s residual formulas were checked against its PINN loss functions) but nothing
in this repo depends on its presence at runtime anymore.

```
python -m pde.run_experiment \
  --config pde/config.yaml --data-root /data/multiphysics --problem TE_heat \
  --method csho --score-arch fno --seeds 0,1,2,3,4 \
  --out-dir results/experiment_2_physics/TE_heat_csho_fno

python -m pde.run_experiment \
  --config pde/config.yaml --data-root /data/multiphysics --problem TE_heat \
  --method csho --score-arch songunet --seeds 0,1,2,3,4 \
  --out-dir results/experiment_2_physics/TE_heat_csho_songunet
```

`--fno-modes` (default `"12,12"`), `--fno-hidden-channels` (default 128), `--fno-init-channels`
(default 32, shared by `fno`/`songunet` -- see below) tune the FNO; `n_modes` must not exceed the
working resolution (`--image-size`). `--songunet-model-channels` (default 32),
`--songunet-channel-mult` (default `"1,2,2"`), `--songunet-num-blocks` (default 2),
`--songunet-attn-resolutions` (default `"16"`, comma-separated, empty string for none) tune
SongUNet -- `channel_mult`'s length is the number of downsampling stages, so it must stay small
enough that `--image-size` doesn't collapse to 0 (e.g. 3 entries needs `--image-size >= 8`).

**Why `fno`/`songunet` needed real architecture changes, not just a network swap.** Read through
`pde/model.py`/`pde/run_experiment.py` before starting this phase and found the `attention` path
doesn't do textbook noise-to-data generative sampling: `PhysicsModel.encode(conditioning)`
produces a *deterministic* per-task starting point (no randomness), and the reverse SDE
(`--n-diff-steps` defaults to 2) is a short, learned refinement of it. More importantly,
`MultiPhysicsScoreNetwork.forward` never sees the position state `X` at all -- only the velocity
`V` (`make_score_fn`'s closure silently drops the `X` argument every caller already supplies). This
was a placeholder limitation, not a design choice to preserve, so both new score networks fix it
by construction: `pde/fno_score_net.py::FNOScoreNetwork` and
`pde/songunet_score_net.py::SongUNetScoreNetwork` condition on **both** `X` and `V` (plus the raw
conditioning field and timestep) at every step, matching proper critically-damped-Langevin-style
scoring (`score(x_t, v_t, t)`, not `score(v_t, t)`). `songunet` conditions on `X`/`V`/conditioning
via channel-concatenation into the network's input, and on the timestep via `SongUNet`'s own native
`noise_labels` embedding pathway (unlike `fno`, which has no native time-conditioning mechanism and
uses a broadcast time channel instead). `SongUNet`'s own EDM preconditioning wrappers
(`VPPrecond`/`VEPrecond`/`EDMPrecond`) are deliberately bypassed -- this repo already has its own
SDE/preconditioning machinery (`synthetic/exact_dsm.py`); the raw network is called directly
(`label_dim=0`, unconditional in `SongUNet`'s own terms) and its output is interpreted as a
velocity-score prediction, exactly like `FNOScoreNetwork`'s output. The `attention` path /
`MultiPhysicsScoreNetwork` is left exactly as-is (superseded for CSHO methods under `fno`/
`songunet`, not patched in place) -- zero risk to already-produced `attention`-path results.

**What changed, concretely, for `fno`/`songunet`:**
- `pde/model.py::SpatialFieldModel` replaces `PhysicsModel` for this path: `PhysicsBackbone` is
  reused unchanged, but `FieldHead`'s vector round-trip (`to_latent`/`readout_proj`/
  `readout_conv`/`readout_out`) is dropped entirely. A lightweight per-task CNN head (structurally
  identical to `FieldHead`'s `y0_head`) produces the deterministic initial state `X0` directly in
  native `(B,1,H,W)` field space -- the only source of the diffused state, no pooling anywhere.
  `decode` is the identity; the reverse SDE's final state *is* the prediction.
- Training loss is `init_loss + lambda_tikhonov * tikhonov` (CSHO) -- `readout_loss` is dropped
  since it would be literally redundant with `init_loss` now that decode is the identity (both
  would compare the same tensor to target).
- `synthetic/`'s SDE machinery (`anderson_sde.py`, `exact_dsm.py`, `drift_coupled_gamma.py`)
  needed **zero changes** -- confirmed via full reads: task coupling lives entirely in Python-list
  nesting and `(N,N)`/`(2N,2N)` matrices (`N` = task count), never in the per-task payload tensor's
  shape; every payload-tensor op is elementwise or `.shape`-derived. The existing
  flatten-into-rows trick (`X[i][0].reshape(-1,1)` -> cat over tasks -> `sample_and_
  tikhonov_score_target` -> reshape back) generalizes unchanged from `(B,latent_dim)` to
  `(B,1,H,W)`; only the *restore* reshape needed to capture `orig_shape = X[0][0].shape` generically
  instead of hardcoding `(B, latent_dim)` (this generalization also applies to the `attention` path
  now, behavior-identical there since `orig_shape` reduces to `(B, latent_dim)`).

**Discovered along the way, fixed in the shared `synthetic/exact_dsm.py`:
`sample_and_tikhonov_score_target`'s default `jitter` was an absolute floor (`1e-8`), not scaled to
`Sigma_t`'s magnitude.** `Sigma_t` at low `t_idx` (few noise-injection steps accumulated so far) is
*mathematically* rank-deficient -- only `N` of `2N` noise dimensions have been injected yet -- so
its smallest eigenvalues sit at floating-point-roundoff level and can go slightly negative once the
calibrated `sigma` scales up, making `torch.linalg.cholesky` fail intermittently. Verified this is
**path-independent** (affects `attention` and `fno` equally -- reproduced by sweeping `sigma`
values directly through `precompute_transition_params`, no encoder involved), was only surfaced by
this phase's new smoke tests, and independently confirmed for real on `synthetic/`'s own N-sweep
(`synthetic/anderson_tikhonov_n_sweep.py`), which hit the identical `linalg.cholesky` failure at
N=5 on the user's A100. `jitter` (`synthetic/exact_dsm.py::sample_and_tikhonov_score_target`,
default now `1e-6`) is interpreted as **relative** to `Sigma_t.diagonal().abs().max()` rather than
a bare additive constant, so it stays negligible for well-conditioned `Sigma_t` but reliably
regularizes the rank-deficient case regardless of how large `sigma` is calibrated to.
`pde/run_experiment.py`'s call site no longer needs (or has) a local override -- it inherits the
fixed default like every other caller (`synthetic/run_experiment.py`,
`synthetic/anderson_tikhonov_n_sweep.py`). Verified robust across 10 random seeds locally
(previously reproducible within a handful).

**A real `fno` run against production data diverged catastrophically after this phase landed --
see Section 12 for the root cause (a compounding score-network output-scale bias), the fix (a
learnable per-task output gain on `FNOScoreNetwork`/`SongUNetScoreNetwork`), and the new
stability-test methodology (`tests/test_pde_reverse_sde_stability.py`) written to catch this class
of bug before spending real GPU time again.**

## 10. TE_heat eval fix: a held-out split carved from training

`testing/TE_heat/mater/` is empty on the real Multiphysics-Bench release -- confirmed by direct
inspection (`os.listdir` on the remote box): `testing/TE_heat/Ez/` and `testing/TE_heat/T/` each
have 1000 populated `.mat` files (`100001.mat`..`101000.mat`), `testing/TE_heat/mater/` has 0.
Instead, testing ships a `polycsv/` directory: one CSV per sample, each holding a **variable
number of `(x, y)` vertex pairs** (11 rows for one sample, 12 for another, confirmed by direct
inspection) -- an arbitrary polygon boundary, not `ellipticcsv`'s fixed `[e_a, e_b, angle]`
three-value ellipse format. This isn't a directory-naming mismatch to patch around: testing's
material inclusions are geometrically more general than training's (polygons vs. ellipses), and
even granting a polygon-membership test, there's no way to recover the material *property
values* inside/outside the inclusion from geometry alone -- confirmed by reading
Multiphysics-Bench's own PINN training scripts (`pinns/train_te_heat.py`), which load their
testing *inputs* from a private pre-cached tensor file
(`/data/bailichen/PDE/PDE/DeepONet/TE_Heat/data/TE_heat_test_128_3w.pt`) never published to
Hugging Face -- there is no documented, reproducible way to regenerate `mater` for testing from
what's publicly available.

Silently falling back to evaluating on the same samples used for training would be data leakage;
crashing (the original failure mode -- `pde/dataset.py`'s sample-index discovery trusted the
first output field's directory listing and assumed every other directory, including the input,
shared the same indices) wastes a full training run before failing at the first evaluation
batch. Instead, `pde/dataset.py::_testing_split_broken` detects this specific situation (a
`testing/{problem}/` directory that **exists** but has zero samples usable across all its field
directories -- deliberately not triggered when the testing directory is simply absent, e.g. every
local smoke-test fixture, which keeps its prior, simpler `except (FileNotFoundError, ValueError):
val_ds = train_ds` fallback in `run_experiment.py` unchanged) and falls back to a deterministic
90/10 split carved from the **training** directory instead.

For reproducibility -- every method/architecture/seed in the grid must be scored against
literally the same held-out samples, not independently-and-possibly-differently re-derived
splits -- this split is computed once and **persisted** to
`pde/held_out_splits/{problem}_held_out_split.pt` on first use, then reused verbatim by every
subsequent run against the same `--data-root` (validated via a `source_problem_root` field in the
cached file; a mismatched root, e.g. a local smoke-test fixture, recomputes fresh in-memory
without touching the persisted canonical file). **Commit this file once generated on the real
data** so the exact held-out split is fixed across machines, not just across runs on one box.
Covered by a dedicated regression test, `tests/test_dataset_held_out_split.py` (verifies the
train/test slices are disjoint, partition the full training set, persist correctly, and that a
wholly-absent testing directory does *not* trigger this fallback).

Only confirmed necessary for `TE_heat` so far; `E_flow`/`VA`'s testing splits have not been
independently verified and may be fine as-is (the mechanism only activates when actually needed).

## 11. Per-task target normalization (network output space)

Every problem's output tasks span wildly different raw physical scales *within the same joint
loss*, confirmed by direct inspection of the real data: `TE_heat`'s `Ez` has |value| mean
~2.7e5 (std ~1.1e5) while its `T` has mean ~302.5 with std ~0.6 (its entire range across 20
samples is 300.9–304.6 K); `E_flow`'s `ec_V` (mean ~25.8) vs `u_flow`/`v_flow` (mean ~±0.005) span
~3 orders of magnitude; `VA`'s `x_u`/`x_v` (mean ~14–22, max ~500–770) vs `Sxx`/`Sxy`/`Syy` (mean
~0.02–0.1) span ~4. None of the *output* target fields were normalized before this fix — only
`TE_heat`'s *input* conditioning (`mater`, via `te_heat_normalize_mater`) was. In an unweighted
per-task L1 sum (`init_loss`/`readout_loss` in `run_experiment.py`), the largest-scale task's loss
term completely dominates the gradient, destabilizing the shared backbone. Confirmed on a real
`TE_heat`/`csho`/`attention` run: `Re{Ez}`/`Im{Ez}` collapsed to `rel_l2`≈0.994 (suspiciously
identical across all 4 seeds — the signature of a degenerate near-constant prediction, not "hard
to learn"), with ~28–31% of training steps triggering the gradient-explosion detector (which is
diagnostic-only and does not skip the optimizer step, so those huge-but-finite gradients were
still being applied). `T`'s deceptively good `rel_l2`≈0.0015 is an artifact of its own near-zero
dynamic range, not evidence of real learning.

**Fix**: `pde/dataset.py::compute_target_norm_stats` streams the full training dataset once
(exact, not a subsample) to compute an exact per-task z-score `(mean, std)`, cached to
`pde/target_norm_stats/{problem}_target_norm_stats.pt` (`_get_or_create_target_norm_stats`, same
caching philosophy as Section 10's held-out split — computed once, reused identically by every
subsequent method/architecture/seed so the whole grid stays validly comparable; a key mismatch,
e.g. a local smoke-test fixture, recomputes fresh in-memory without touching the persisted
canonical file). The network's raw output (`y0`, `model.decode(...)`, and the spatial
(`fno`/`songunet`) path's `X0`) is trained to predict **directly in normalized space** —
`run_experiment.py::_normalize_targets` precomputes `(target - mean)/std` once per batch, and
`init_loss`/`readout_loss` compare the network's raw output to that directly (plain `F.l1_loss`,
no transform on the prediction side). `evaluate()` denormalizes (`_denormalize_preds`,
`pred*std + mean`) only at the very end, right before `rel_l2`/`spectral_l2`/`pde_residual` are
computed, which is the one place raw physical units are actually needed.

**A first attempt at this got the design wrong, worth recording.** The initial version instead
left the network's raw output targeting raw physical scale directly, and only applied `(x-mean)/std`
symmetrically to *both* sides of the L1 comparison (`_normalized_l1_loss(pred, target, mean, std)
= L1((pred-mean)/std, (target-mean)/std)`). That successfully fixed the *instability*
(`explosion_events` dropped from ~28-31% of steps to ~0.3%, confirmed on real `TE_heat` data at
the full 50-epoch budget) but did **not** fix the actual learning problem: algebraically, that
loss still requires `pred → target` exactly, so the network's output head still had to grow huge
internal weights to reach `Ez`'s ~2.7e5 scale, and Adam's step size is bounded in *parameter*
space regardless of the raw distance to travel in *output* space -- confirmed empirically,
`Re{Ez}_rel_l2` stayed at ~0.995 (no better than the original broken run) even at the full 7000
training steps. The corrected design above -- network predicts small, normalized values
throughout; denormalize only for eval -- means the network never needs large internal weights to
represent a large-scale task at all. Verified directly with a synthetic fixture reproducing the
same pathology (one task with a huge raw scale ~2.7e5 like `Ez`, others tiny): the huge-scale
task reached `rel_l2`≈0.096 after 30 epochs on 8 samples, vs. ~0.995 under the first attempt.

**Matches Multiphysics-Bench's structural approach, not its numeric scheme.** Their PINN training
scripts (`pinns/train_{te_heat,e_flow,VA}.py::compute_loss`) follow the exact same *structure* the
corrected design above uses: network predicts in normalized space, the supervised loss compares
normalized-to-normalized, and only *afterward* do they denormalize back to physical units (before
computing their own PDE-residual loss term) -- confirmed by reading `compute_loss()` in all three
files. Where this deliberately differs is the numeric scheme: they use min-max scaling to
`[-0.9, 0.9]` (per-task, from precomputed range files they ship and we don't have) for essentially
everything, with `TE_heat`'s complex `Ez` field specifically using symmetric abs-max scaling shared
across its Re/Im pair to preserve phase (`VA`'s 6 complex fields, by contrast, min-max each
real/imag channel independently, same as everything else). Z-score is used here instead -- since
this normalization only affects our own training loss's internal representation, not anything
directly comparable to their reported numbers, the exact scheme doesn't need to match, only the
structural principle (network output space is normalized; physics/metrics use denormalized raw
units) does.

## 12. FNO reverse-SDE divergence: root cause, output-gain fix, and stability-test methodology

A real (non-smoke) `--score-arch fno` run against `TE_heat` (128x128, 9000 real samples,
`n_diff_steps=20`, `dt=0.5`, `n_epochs=50`, `batch_size=64`, 5 seeds) produced
`Re{Ez}_rel_l2`/`Im{Ez}_rel_l2`≈22-22.5 — the prediction's magnitude ~22x the true field's own
norm — while the identical run under `--score-arch attention` worked correctly (`rel_l2`≈0.3).

**Root cause.** `anderson_reverse_step_coupled_gamma` (`synthetic/anderson_sde.py`) is, for a
linear-Gaussian process with a linear score, a **linear per-step recursion** whose feedback term
is directly proportional to the score network's output. A systematic score-network output-scale
bias `c` therefore compounds **geometrically** over the reverse rollout: `c^n_diff_steps`. Solving
`c^20 ≈ 22.3` gives `c ≈ 1.167`, matching the observed divergence almost exactly and explaining its
tight, nearly seed-independent spread (22.03-22.48 across 5 seeds) — a structural gain bias
reproduces consistently across seeds, stochastic instability would not. `calibrate_coupled_gammas`
(the deterministic drift/damping) depends only on `alpha, beta, k_reference, N, damping_regime` —
identical for both architectures, and demonstrably stable under `attention` with the same numbers
— so the deterministic integrator itself was ruled out. Training-time `explosion_events` stayed
low (1-3/7000 steps) because a smooth-but-mis-scaled function doesn't trip a per-training-step
gradient-norm threshold: the divergence only appears once you *integrate* the bias over 20
`evaluate()`-time reverse steps, and `evaluate()` runs entirely under `@torch.no_grad()` with no
backward pass — training-time diagnostics structurally cannot see it.

Most likely source of the gain bias: `FNOScoreNetwork` (`pde/fno_score_net.py`) had **no
output-scale-constraining mechanism** (no final activation/clamp/normalization — raw
`neuralop.models.FNO` projection output) and, unlike `attention`, **no downstream `decode()` step
to absorb drift** — `MultiPhysicsScoreNetwork` operates on `LayerNorm`-regularized attention
tokens and its co-trained encoder/decoder can jointly compensate for whatever scale the latent
settles at, while `SpatialFieldModel`'s `X0` is rigidly pinned to z-scored physical units (Section
11) with zero decode-side freedom, so any score-network gain bias shows up in the reported metric
completely undiluted. Secondary/contributing candidate: `n_modes=(12,12)` (the FNO's spectral
truncation) is a strong low-pass architectural bias, while the DSM/Tikhonov score target
(`sample_and_tikhonov_score_target`) is built with noise drawn i.i.d. **per pixel** — a target
containing substantial per-pixel-independent (high-frequency) content a 12x12-mode spectral
architecture cannot represent exactly, unlike `attention`'s compact, spatially-unstructured
64-dim latent target.

**Fix: learnable per-task output gain.** `FNOScoreNetwork`/`FlatFNOScoreNetwork`/
`SongUNetScoreNetwork`/`FlatSongUNetScoreNetwork` each now have `self.output_gain =
nn.Parameter(torch.ones(n_tasks))`, multiplied elementwise into the raw network output
(`out * self.output_gain.view(1, -1, 1, 1)`) before it's returned. Initialized to `1.0`, so
behavior at init is unchanged — this cannot make a currently-working configuration worse. Trained
by ordinary backprop through the same Tikhonov/DSM loss already in place, letting the network
self-correct a systematic scale bias via gradient descent, per-task (a single global scalar would
be under-specified, since e.g. `T` was far less affected than `Ez` in the observed run).

**Why the existing smoke-test suite couldn't catch this, and what does.** Every PDE smoke test
(`tests/test_experiments_smoke.py`) runs at `--n-diff-steps 2` and asserted only `np.isfinite` —
never a magnitude bound. `1.17^2≈1.37` at the smoke suite's step count is indistinguishable from
ordinary undertrained-model error, vs. `1.17^20≈23.1` at the real grid's step count — **this bug
was structurally unreachable by the smoke-test methodology, independent of any other gap**
(dataset realism, batch size, epoch count). This repo's `synthetic/` module already has, and uses,
the exact methodology needed: a **zero-score deterministic-drift stability test**
(`synthetic/schedule_diagnostics.py::monte_carlo_probe` — force the score to zero and verify the
reverse trajectory alone doesn't blow up; `writeup/csho_writeup.tex`'s own Verification section
documents this exact technique catching an identical class of bug, unbounded multiplicative growth
in an earlier, uncorrected reverse-SDE sign convention, in the original scalar CSHO design) and
recalibrating at every step count rather than trusting a cached value from a different one
(`synthetic/anderson_tikhonov_n_sweep.py`). None of this existed for `pde/`'s spatial (`fno`/
`songunet`) reverse-sampling pathway before now.

**`tests/test_pde_reverse_sde_stability.py`** ports this methodology to `pde/`, covering all 9
`(score_arch, problem)` combinations (`{attention, fno, songunet} x {TE_heat, E_flow, VA}`) in one
bundled script, each combination printing its own pass/fail line:
1. **Zero-score round-trip** at the real production budget (`n_diff_steps=20, dt=0.5`): force
   `score_outputs` to zero and assert the final/initial state-norm ratio stays under
   `ZERO_SCORE_RATIO_BOUND=3.0`. Isolates "is the integrator itself stable for this
   (gamma, alpha, beta, dt, k_reference, N) configuration" from "is the trained score good."
2. **Step-count sweep** (`n_diff_steps ∈ {2, 4, 8, 16, 20}`): a small, freshly-trained-and-
   recalibrated-per-step-count model (`STEP_SWEEP_N_TRAIN_STEPS=8` gradient steps) rolled out at
   that same step count, asserting the final/initial state-norm ratio stays under
   `STEP_SWEEP_RATIO_BOUND=3.0` at **every** step count, not just the smallest. `1.17^16≈12.3`
   already clearly exceeds this bound well before reaching the real step count — this is the
   check that would have directly caught the FNO divergence.
3. **Realistic-scale synthetic fixture**, used by the step-count sweep (`_realistic_targets`): one
   huge-scale task (~1e5, like `Ez`) and the rest tiny-scale/low-variance (~1e2, like `T`) at
   32x32 resolution — reproduces the actual magnitude disparity (Section 11) that produced the
   observed bug, rather than an easier pure-random-noise regime that happens to pass.
4. **Magnitude assertions added to the existing local smoke suite**: every
   `tests/test_experiments_smoke.py` PDE test now also asserts every `*_rel_l2` metric is `< 5.0`,
   alongside the existing `np.isfinite` check — generous but meaningful (would have failed
   instantly on the observed ~22x), so a regression of this class fails loudly in the fast,
   existing suite too. One test (`test_experiment_2_physics_smoke_va_fno`) needed its
   `--n-epochs` bumped from 1 to 10 to keep this bound meaningful rather than flaky: verified
   locally that VA's 12 real/imag tasks pushed `max_rel_l2` to ~15 purely from being
   undertrained-in-one-epoch (no divergence involved), dropping to ~3.5 by epoch 10.

Meant to run on a GPU box, not a laptop — real FNO/SongUNet forward passes at `n_diff_steps` up to
20 are slow on CPU (it does not need the real Multiphysics-Bench dataset, only compute speed):

```
python -m tests.test_pde_reverse_sde_stability
```

Local (fast, CPU-scale) verification before trusting this: `python -m tests.test_experiments_smoke`
must still pass, including the new magnitude assertions — confirms the divisor fix (Section 8) and
the output-gain fix didn't break anything at smoke scale before the heavier remote check.

**Update: a second, larger root cause found while running this harness for the first time —
`--dt` was violating the `T_max=1.0` invariant `drift_fn_coupled_gamma`'s time_scale schedule
assumes.** Running the zero-score/step-sweep tiers above surfaced that even `attention` (already
known to work in production) diverges on the *deterministic* drift alone (no score, no noise) at
`pde/run_experiment.py`'s real config, `n_diff_steps=20` with `--dt` defaulting to a fixed `0.5`
independent of `n_diff_steps` (`dt*n_diff_steps=10`) — isolated with a direct sweep: stable
(ratio≈0.19-0.87) at every tested `(dt, n_diff_steps)` pair with `dt*n_diff_steps` near `1.0`,
diverging (ratio≈9-40x) as that product grows past it. `synthetic/anderson_tikhonov_n_sweep.py`
already documents and enforces the correct convention (`--dt` defaults to `1/n_diff_steps`,
holding `n_diff_steps*dt` fixed at `1.0` as `n_diff_steps` varies) — `pde/run_experiment.py`'s
`--dt` (default `0.5`, independent of `n_diff_steps`) and `pde/ablations.py`'s own separate `--dt`
forwarding (same default, unconditionally passed through) never adopted it. `run_full_paper_sweep.sh`
passes `--n-diff-steps 20` without overriding `--dt`, silently landing on `dt*n_diff_steps=10` —
confirmed via a direct diagnostic to make the deterministic drift alone diverge ~40x before any
score network is even involved. **Fixed**: both files' `--dt` now default to `None`, resolved to
`1/n_diff_steps` (`pde/run_experiment.py::parse_args`) or left unset so `run_experiment.py`'s own
resolution applies (`pde/ablations.py::base_argv`, only forwards `--dt` when explicitly overridden).
Re-verified with the same isolated deterministic-drift sweep: `dt=0.05, n_diff_steps=20`
(the corrected default) now gives ratio≈0.76, stable. `python -m tests.test_experiments_smoke`
still passes unchanged (every smoke test uses `--n-diff-steps 2`, which already satisfied
`dt*n_diff_steps=1.0` under the old default by coincidence, so this fix is invisible there).

This does not fully explain the original FNO divergence by itself (that investigation's
compounding score-output-gain-bias diagnosis and the Fix 2 output-gain parameter both stand), but
it means the underlying integrator was *also* running outside its valid regime on the real grid,
compounding with any score-network imperfection — likely part of why `attention`'s real result
(`rel_l2`≈0.3) still worked (a fully-converged 50-epoch score learned to compensate) while FNO's
much larger, more architecture-specific bias did not.

**`tests/test_pde_reverse_sde_stability.py` is paused, not finished.** Building it is what
surfaced the `dt` bug above, but the script's own zero-score/step-sweep bounds (`ZERO_SCORE_RATIO_BOUND`,
`STEP_SWEEP_RATIO_BOUND`) were tuned against the broken `dt=0.5` config and have **not** been
re-validated against the corrected `dt=1/n_diff_steps` default, nor against a properly-converged
(not `STEP_SWEEP_N_TRAIN_STEPS=8`) trained score — a locally-trained-for-400-steps diagnostic still
diverged to `inf`/`nan` even after the `dt` fix, suggesting the step-sweep tier's premise (a cheap,
briefly-trained local/remote proxy standing in for a full 50-epoch run) may not be achievable for
this drift formulation at all. Left as-is pending further investigation, not deleted.

**Separately flagged, not yet confirmed on real data**: `synthetic/exact_dsm.py::calibrate_sigma_for_leak`
computes `(k / snr_target) ** 0.5` with no guard against a negative base — if `cov_data`'s mean
pairwise task correlation is non-positive, this silently returns a **complex** Python number, which
downstream code (`torch.tensor(sigma, ...)` feeding `build_g_matrix_n`) silently truncates to its
near-zero real part (the `Casting complex values to real discards the imaginary part` warning some
runs print). Reproduced in 6/6 diagnostic runs using this harness's synthetic (randomly-initialized
encoder) fixtures, but not yet checked against real encoded training data -- `E_flow`'s documented
unidirectional coupling (Section 8's "falsification case") is the most plausible real candidate.
Not fixed yet; investigate against real data before deciding whether to guard it.

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
