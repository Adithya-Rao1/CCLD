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

**Metric definition**: DiffusionPDE's own evaluation formula (Huang et al., NeurIPS 2024,
arXiv:2406.17763) — plain mean-squared PDE-operator residual over the spatial grid,
`L_pde = mean(residual**2)`, no relative/reference normalization. This is deliberately *not*
Multiphysics-Bench's internal PINN-training convention (`pde/multiphysics-bench/pinns/train_*.py`),
which divides by arbitrary problem-specific constants and zeroes/clips boundary and outlier
values purely for gradient-balancing during training — those are training-stability hacks, not a
physically meaningful metric, and are deliberately dropped here.

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
  Note: that run reported `testing: 0 samples found` for TE_heat -- still unresolved, likely a
  different directory layout or naming for the testing split on the remote box; worth checking
  with `ls`/`find` before running a real testing-split evaluation.
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

`--score-arch {attention, fno}` (default `attention`) selects the network that predicts scores
during CSHO/DDPM/SDM training and sampling, for `TE_heat`, `E_flow`, `VA`. `attention` is the
original, unchanged pipeline (`pde/model.py::PhysicsModel`/`MultiPhysicsScoreNetwork` -- a pooled
per-task latent vector `(B, latent_dim)`, refined via cross-field attention). `fno` is new:
**native-pixel diffusion** -- each task's diffused state is the full field itself, `(B, 1, H, W)`,
with no latent-vector bottleneck, denoised by an FNO (`pde/fno_score_net.py`, needs
`neuraloperator`: `pip install neuraloperator`).

```
python -m pde.run_experiment \
  --config pde/config.yaml --data-root /data/multiphysics --problem TE_heat \
  --method csho --score-arch fno --seeds 0,1,2,3,4 \
  --out-dir results/experiment_2_physics/TE_heat_csho_fno
```

`--fno-modes` (default `"12,12"`), `--fno-hidden-channels` (default 128), `--fno-init-channels`
(default 32) tune the FNO; `n_modes` must not exceed the working resolution (`--image-size`).

**Why `fno` needed real architecture changes, not just a network swap.** Read through
`pde/model.py`/`pde/run_experiment.py` before starting this phase and found the `attention` path
doesn't do textbook noise-to-data generative sampling: `PhysicsModel.encode(conditioning)`
produces a *deterministic* per-task starting point (no randomness), and the reverse SDE
(`--n-diff-steps` defaults to 2) is a short, learned refinement of it. More importantly,
`MultiPhysicsScoreNetwork.forward` never sees the position state `X` at all -- only the velocity
`V` (`make_score_fn`'s closure silently drops the `X` argument every caller already supplies). This
was a placeholder limitation, not a design choice to preserve, so `fno`'s score network(s) fix it
by construction: `pde/fno_score_net.py::FNOScoreNetwork` conditions on **both** `X` and `V` (plus
the raw conditioning field and timestep) at every step, matching proper
critically-damped-Langevin-style scoring (`score(x_t, v_t, t)`, not `score(v_t, t)`). The
`attention` path / `MultiPhysicsScoreNetwork` is left exactly as-is (superseded for CSHO methods
under `fno`, not patched in place) -- zero risk to already-produced `attention`-path results.

**What changed, concretely, for `fno`:**
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
