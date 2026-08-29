# Results — Multi-Task Diffusion via Coupled Langevin Dynamics

Working document for turning `results/` output into the paper. Every table below has the exact
CSV/JSON path each cell comes from, so filling this in is a copy-paste-from-file exercise, not a
re-derivation. Replace every `[TBD]` with the real number once the run finishes.

Cityscapes is **excluded** from Experiment 1 (no academic email → no dataset access) — vision
results are NYUDv2 + PASCAL VOC only. Say so explicitly in the paper's limitations/data section;
don't silently drop it.

Assume `$RESULTS_ROOT = /home/ubuntu/metis-v1-storage/results` (per the run commands) for every
path below — adjust if you used a different `--out-dir`.

---

## STATUS NOTE (2026-08-15) — read before pulling synthetic/ numbers for the paper

Debugging session found and fixed 4 real bugs in `synthetic/`'s CSHO training/sampling pipeline
(em_step_n sign convention, single-jump training approximation, missing gradient clipping,
batch-aggregate-instead-of-per-sample coupling) — all confirmed against production reference code,
all committed. Net effect: `csho` KL divergence went **4808 → ~5** (~960x improvement) at
N=3/coupling_strength=0.6/n_diff_steps=20/n_train_iters=2000, stable across seeds.

**However, CSHO still underperforms DDPM/SDM baselines** (KL≈5 vs. DDPM/SDM's ≈0.03-0.1) at these
settings — this is diagnosed as a **noise-schedule calibration issue**, not a remaining bug: the
SHO drift's `time_scale` annealing isn't self-consistent across diffusion-step counts the way
DDPM's purpose-built cosine schedule is (too few steps → not enough time for coupling to build
correlation; too many steps → reverse-process variance explodes instead). No working operating
point has been found yet. Full diagnostic detail in
`~/.claude/plans/i-m-trying-to-finish-synthetic-tiger.md`, "Implementation log" section, items 5-6.

**Do not fill in §1's tables from a pre-recalibration run** — the numbers will look bad (CSHO
losing to baselines) not because coupling doesn't help, but because of this schedule issue. Before
running the real ablation grid for the paper, either (a) find/design a working `n_diff_steps`/
`sigma`/`dt` operating point where the schedule is well-calibrated, or (b) note this as a known
limitation if you're reporting current numbers as-is. This note doesn't apply to §2/§3 (image/pde)
directly, but the same per-sample coupling fix landed in `core/drift.py` (shared by all three
experiments) — worth spot-checking there too once real data is available.

---

## 0. Paper section → evidence map

| Paper section | Pulls from |
|---|---|
| Abstract / headline numbers | §6 (key numbers) |
| Method (the coupled drift) | `core/drift.py`, `README.md`'s drift equation — no results needed |
| Theory grounding / sanity check | §1 (synthetic) |
| Main results — vision | §2 (image) |
| Main results — multi-physics PDE | §3 (pde) |
| Ablations | §4 |
| Limitations | §7 |

Suggested writing order: **§1 → §4 (synthetic axes) → §2 → §3 → §4 (real-data axes) → §0 abstract
last**, since the synthetic experiment is fastest to finish and its numbers tell you whether the
core claim holds at all before you sink a weekend into GPU time on the other two.

---

## 1. Experiment 3 — Synthetic (coupled-OU, closed-form ground truth)

**Claim under test**: CSHO's coupled drift recovers a known joint distribution — not just
marginals, but cross-population dependence — better than DDPM/VP-SDE, and the gap grows with the
ground truth's true coupling strength.

Source: `$RESULTS_ROOT/experiment_3_synthetic/{method}/{method}_summary.csv` (one row per metric,
`mean`/`std`/`ci_lo`/`ci_hi` columns, 5 seeds).

### 1.1 Main comparison (N=5, coupling_strength=0.6)

| Method | KL divergence ↓ | Wasserstein-2 ↓ | MI MAE ↓ | Autocorr(lag 1) | Integrated autocorr time | Hypo. passed |
|---|---|---|---|---|---|---|
| CSHO (mean-field) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CSHO (pairwise) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CSHO (independent) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| DDPM | [TBD] | [TBD] | n/a | n/a | n/a | n/a |
| SDM (VP-SDE) | [TBD] | [TBD] | n/a | n/a | n/a | n/a |

Format as `mean ± std` per cell. `n/a` cells are correct as-is — those diagnostics are CSHO-only by
design (see `synthetic/README.md` §3), don't try to fill them in.

**One-line takeaway** (fill in after the numbers are in): [TBD — e.g. "CSHO-mean-field achieves
Nx lower KL than DDPM/SDM and recovers pairwise MI within Y of ground truth"]

### 1.2 Ablation: does the CSHO advantage grow with true coupling strength?

Source: `$RESULTS_ROOT/experiment_3_synthetic/ablations/ablation_results.csv`, filtered to
`axis == "coupling_strength"`; significance from `ablation_significance.csv`.

| Ground-truth coupling_strength | CSHO KL ↓ | CSHO W2 ↓ | CSHO MI MAE ↓ |
|---|---|---|---|
| 0.0 | [TBD] | [TBD] | [TBD] |
| 0.3 | [TBD] | [TBD] | [TBD] |
| 0.6 | [TBD] | [TBD] | [TBD] |
| 0.9 | [TBD] | [TBD] | [TBD] |

**This is the single most paper-defining plot in the whole repo.** If KL/MI-MAE monotonically
improve (relative to a fixed-coupling baseline) as `coupling_strength` rises, that's your Figure 1
or Figure 2. Plot it: x-axis = coupling_strength, y-axis = KL divergence, one line for CSHO and
one flat/worse line for DDPM/SDM (run DDPM/SDM through the same axis manually if `ablations.py`
didn't already sweep them — check `ablation_results.csv`'s `axis`/`label` columns first).

### 1.3 Ablation: N-sweep, damping regime, alpha/beta, diffusion mode

Source: same `ablation_results.csv`, filtered to `axis in {n_populations, damping_regime,
alpha_beta, diffusion_mode}`.

| Axis | Best config | Worst config | Notes |
|---|---|---|---|
| `n_populations` (N=2..5) | [TBD] | [TBD] | Does the CSHO-vs-baseline gap hold at every N, or only emerge at higher N? |
| `damping_regime` | [TBD] | [TBD] | Is critically-damped still optimal once coupling is added, per the CLD literature's prior finding? |
| `alpha_beta` | [TBD] | [TBD] | |
| `diffusion_mode` (shared vs. independent G) | [TBD] | [TBD] | |

---

## 2. Experiment 1 — Vision multi-task (NYUDv2 + PASCAL; Cityscapes excluded)

**Claim under test**: cross-task coupling improves joint multi-task diffusion on real dense
prediction tasks, and helps the *worst*-performing task most (the "information exchange" claim).

Source: `$RESULTS_ROOT/experiment_1_vision/{source}_{method}/{method}_summary.csv`.

### 2.1 NYUDv2 (5 tasks: depth, normals, segmentation, edges, saliency)

| Method | depth_rmse ↓ | depth_rel_err ↓ | normals_angular_error_deg ↓ | segmentation_miou ↑ | edges_fscore ↑ | saliency_mae ↓ |
|---|---|---|---|---|---|---|
| CSHO (mean-field) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CSHO (pairwise) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CSHO (independent) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CSHO (shared G) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| CSHO (independent G) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| DDPM | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| SDM (VP-SDE) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### 2.2 PASCAL VOC 2012 (3 tasks: segmentation, edges, saliency)

| Method | segmentation_miou ↑ | edges_fscore ↑ | saliency_mae ↓ |
|---|---|---|---|
| CSHO (mean-field) | [TBD] | [TBD] | [TBD] |
| CSHO (pairwise) | [TBD] | [TBD] | [TBD] |
| CSHO (independent) | [TBD] | [TBD] | [TBD] |
| DDPM | [TBD] | [TBD] | [TBD] |
| SDM (VP-SDE) | [TBD] | [TBD] | [TBD] |

### 2.3 Worst-task analysis (the "information exchange" claim)

For each method, identify the single worst-performing task (normalize metrics first — e.g. rank
each task 1-7 across methods rather than comparing raw units) and report it here.

| Method | Worst task (NYUDv2) | Worst-task value | Same task's value under CSHO-independent |
|---|---|---|---|
| CSHO (mean-field) | [TBD] | [TBD] | [TBD] |
| DDPM | [TBD] | [TBD] | — |
| SDM | [TBD] | [TBD] | — |

**One-line takeaway**: [TBD — does CSHO's coupling measurably lift the worst task relative to
independent/baseline, or not?]

### 2.4 Ablation: N-tasks, coupling mode, alpha/beta, damping, diffusion mode

Source: `$RESULTS_ROOT/experiment_1_vision/ablations/ablation_results.csv` (NYUDv2 only, since
that's the 5-task source needed for the N=2..5 sweep).

| Axis | Best config | Notes |
|---|---|---|
| `n_tasks` (N=2..5, growing depth+normals → +segmentation → +edges → +saliency) | [TBD] | Does adding tasks help or hurt the original depth/normals pair? |
| `coupling_mode` | [TBD] | |
| `alpha_beta` | [TBD] | |
| `damping_regime` | [TBD] | |
| `diffusion_mode` | [TBD] | |

---

## 3. Experiment 2 — Multi-physics PDE (Multiphysics-Bench + PDEBench)

**Claim under test**: coupling helps under genuinely bidirectional physics and is neutral under
unidirectional physics (`E_flow` is the built-in negative control).

Source: `$RESULTS_ROOT/experiment_2_physics/{problem}_{method}/{method}_summary.csv`.

### 3.1 Per-problem CSHO vs. baselines (relative L2 error, averaged across each problem's fields)

Pull the mean of all `{field}_rel_l2` columns per problem/method into one number per cell (note
per-field breakdowns separately if a specific field's story is more interesting than the average).

| Problem | Coupling (physics) | CSHO rel-L2 ↓ | DDPM rel-L2 ↓ | SDM rel-L2 ↓ | CSHO advantage |
|---|---|---|---|---|---|
| `TE_heat` | bidirectional | [TBD] | [TBD] | [TBD] | [TBD] |
| `NS_heat` | bidirectional | [TBD] | [TBD] | [TBD] | [TBD] |
| `E_flow` | **unidirectional** | [TBD] | [TBD] | [TBD] | [TBD] |
| `MHD` | bidirectional | [TBD] | [TBD] | [TBD] | [TBD] |
| `VA` | bidirectional | [TBD] | [TBD] | [TBD] | [TBD] |
| `Elder` | bidirectional, transient | [TBD] | [TBD] | [TBD] | [TBD] |
| `diffusion_reaction` (PDEBench) | bidirectional | [TBD] | [TBD] | [TBD] | [TBD] |

**This is the paper's other headline result.** If `E_flow`'s CSHO-advantage column is ~0 while
every bidirectional problem's is positive, that's a genuine falsification-style result worth a
dedicated figure (bar chart, one bar per problem, sorted by coupling type).

### 3.2 Elder rollout stability (does coupling reduce compounding error over 10 steps?)

Source: `$RESULTS_ROOT/experiment_2_physics/Elder_{method}/{method}_summary.csv`,
`elder_rollout_step{1..10}_rel_l2` columns.

| Rollout step | CSHO rel-L2 ↓ | DDPM rel-L2 ↓ | SDM rel-L2 ↓ |
|---|---|---|---|
| 1 | [TBD] | [TBD] | [TBD] |
| 2 | [TBD] | [TBD] | [TBD] |
| ... | ... | ... | ... |
| 10 | [TBD] | [TBD] | [TBD] |

Plot as a line chart, x-axis = rollout step, one line per method — error-accumulation-over-horizon
is a natural figure here.

### 3.3 Spectral error (do predictions get frequency content right, not just amplitude?)

| Problem | CSHO spectral-L2 ↓ | DDPM spectral-L2 ↓ | SDM spectral-L2 ↓ |
|---|---|---|---|
| `NS_heat` | [TBD] | [TBD] | [TBD] |
| `MHD` | [TBD] | [TBD] | [TBD] |

(add more problems if the spatial-L2 story alone isn't convincing enough)

### 3.4 Ablation: n_fields, coupling mode, alpha/beta, damping, diffusion mode, problem

Source: `$RESULTS_ROOT/experiment_2_physics/ablations/ablation_results.csv`.

| Axis | Best config | Notes |
|---|---|---|
| `n_fields` (MHD field-count growth) | [TBD] | |
| `coupling_mode` | [TBD] | |
| `alpha_beta` | [TBD] | |
| `damping_regime` | [TBD] | |
| `diffusion_mode` | [TBD] | |
| `problem` | [TBD] | Cross-reference against §3.1's bidirectional/unidirectional split |

### 3.5 Training stability

| Problem | CSHO nan_events | CSHO explosion_events | DDPM nan_events | SDM nan_events |
|---|---|---|---|---|
| (aggregate across problems, or call out any nonzero row specifically) | [TBD] | [TBD] | [TBD] | [TBD] |

---

## 4. Cross-experiment ablation summary (for a combined ablations figure/table)

If you want one table spanning all three experiments instead of three separate ones (common in
multi-experiment papers' ablation section):

| Axis | Synthetic result | Vision result | PDE result | Consistent across experiments? |
|---|---|---|---|---|
| Coupling mode (mean-field vs. pairwise vs. independent) | [TBD] | [TBD] | [TBD] | [TBD] |
| Damping regime | [TBD] | [TBD] | [TBD] | [TBD] |
| Diffusion mode (shared vs. independent G) | [TBD] | [TBD] | [TBD] | [TBD] |
| N / task-count scaling | [TBD] | [TBD] | [TBD] | [TBD] |

A single "is critical damping optimal everywhere" or "does independent coupling always underperform"
row that holds across all three experiments is a much stronger claim than any one experiment alone
— worth a dedicated sentence in the discussion either way (including if it *doesn't* hold
everywhere; that's a real, reportable finding too).

---

## 5. Statistical rigor checklist (reviewers will ask)

- [ ] Every headline number has `n=5` seeds (or state clearly why not, e.g. compute constraints)
- [ ] Every CSHO-vs-baseline claim has a paired Wilcoxon p-value from `ablation_significance.csv`
      or a manual `core.stats.compare_configs` call — cite it, don't just eyeball the means
- [ ] 95% CIs reported alongside means (already computed by `aggregate_over_seeds` — just don't
      drop the `ci_lo`/`ci_hi` columns when transcribing into the paper's tables)
- [ ] Hypoellipticity check (`hypo_passed`) confirmed `True` for every CSHO config actually used in
      a headline result — if any config fails it, that's a caveat to disclose, not bury

## 6. Key numbers for the abstract / intro

Fill these in last, once §1-3 are done — these are the 3-4 numbers that'll actually appear in your
abstract:

- Synthetic: CSHO achieves **[TBD]x lower KL divergence** than the best baseline at the highest
  tested coupling strength.
- Vision: CSHO improves **[TBD]** (worst-task metric) by **[TBD]%** relative to independent/baseline
  on NYUDv2.
- Physics: CSHO reduces relative-L2 error by **[TBD]%** on bidirectionally-coupled problems, with
  **~0%** change on the unidirectional `E_flow` control — supporting that the gain is specifically
  attributable to genuine physical coupling, not just architecture.
- Theory: the linearized CSHO drift passes the hypoellipticity/controllability check across
  **[TBD]/[TBD]** tested configurations in the full ablation grid.

## 7. Limitations to state explicitly

- Cityscapes excluded from Experiment 1 (no academic email — access requires registration
  verification). Vision results are NYUDv2 + PASCAL only; say this once in the data section rather
  than caveat every table.
- [TBD: fill in after runs] Any config where `hypo_passed == False`, any Elder/diffusion_reaction
  run that used a synthetic-only-tested train/test split (the 90/10 PDEBench split logic was unit
  tested against a synthetic array, not verified against the real `.h5` file until this weekend's
  run — note if anything looked off).
- Single-seed-count caveat if you had to cut `--seeds` down from 5 for compute/time reasons.
