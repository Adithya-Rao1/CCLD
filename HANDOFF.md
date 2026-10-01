# Handoff package

Canonical pointer: `github.com/Adithya-Rao1/CCLD`, commit `52c4eee` (branch `main`). Everything
below was verified against a clean `git clone` of that exact commit, not against the author's
working copy, so it reflects what a new clone actually gives you.

## 1. Environment

```
conda create -n coupledsho python=3.10
conda activate coupledsho
pip install -e .
```

This handoff was verified against Python 3.10.20, `torch==2.13.0`, `numpy==1.26.4`, `scipy==1.15.3`.
If you hit a numerical discrepancy, check your resolved versions against these first.

`neuraloperator` (needed only for `pde/`'s `--score-arch fno`) is not a `pyproject.toml`
dependency, only a `requirements.txt` one. Instal using `pip install -r requirements.txt`.

## 2. Minimal smoke test

```
python -c "import core, pde, synthetic"
python -m synthetic.run_experiment --help > /dev/null
python -m pde.run_experiment --help > /dev/null
python -m synthetic.run_experiment \
  --config synthetic/config.yaml --N 3 --method ccld --quick --device cpu \
  --out-dir /tmp/coupledsho_smoke
```

## 3. Claims → retained results index

Every row below cites where the number lives in this repository snapshot and the exact command to
regenerate it.

| # | Claim | Status | Retained artifact (this commit) | Reproduce |
|---|---|---|---|---|
| 1 | CCLD's coupled drift is hypoelliptic / well-posed as an SDE across coupling modes, damping regimes, and N | Established, live-checked | `hypo_passed`/`hypo_min_eig` fields emitted by every `synthetic.run_experiment` run (see §2's smoke output) | §2's smoke command |
| 2 | CCLD beats DDPM at low step counts (n≤32) across N=2..5, mean-field coupling, single coupling_strength=0.6; advantage not significant at n=64,128 (DDPM catches up) | Established, statistically significant (paired Wilcoxon, 10 seeds, p≤0.006 where claimed) | `README.md` §"Coupled Ornstein-Uhlenbeck processes" (full prose + numbers) and `writeup/figures/stepcount_sweep_grid.png` | `bash run_stepcount_sweep.sh 10 10000 40000` then `python -m synthetic.make_stepcount_figures` |
| 3 | Against SDM/SGM, CCLD recovers pairwise correlation significantly closer to ground truth for n≥32; SGM over-estimates correlation 5-13% independent of step count; SGM has significantly *lower* raw KL than CCLD at n=8 | Established, same sweep as #2 | Same as #2 | Same as #2 |
| 4 | Result in #2/#3 is scoped to coupling_strength=0.6, N≤5 — explicitly flagged as untested outside this regime | Established as a *limitation*, not (yet) generalized | `README.md`, same section | — (this is the gap `PROTOCOL_coupling_strength_generalization.md` targets) |
| 5 | On real TE_heat (Multiphysics-Bench electro-thermal field), CCLD achieves lowest E-field PDE residual of {CCLD, DDPM, SDM}; DDPM more accurate pointwise on Im(Ez)/T; single seed, no significance test possible | Established as a point estimate only — explicitly not a settled comparison | `README.md` §"Coupled PDE field reconstruction" (table + prose) and `writeup/figures/te_heat_field_comparison.png` | Requires downloading Multiphysics-Bench first (`python -m pde.download_multiphysics_bench --out-dir pde/multiphysics-bench`, several GB); then the loop command in `README.md` §"Coupled PDE field reconstruction" |
| 6 | Antisymmetric (skew) coupling injection (`synthetic/skew_coupling.py`, `core/coupling.py`) improves KL over symmetric-only coupling at matched Frobenius-norm magnitude, across N=2..5, both samplers, 5 step counts, 10 seeds | Established, independently re-verified against this snapshot's own data, and now reported in `README.md` §"Non-mean-field (antisymmetric) coupling" | Not present as a file in this snapshot (the raw sweep CSVs live only on the author's machine); numbers were pulled directly from that local data: 19/20 (N,n) cells favor `skew_structured` under the exact sampler, significant (p≤0.01, paired Wilcoxon, n=10 seeds) at N=2,3,4 for n=8; the one exception (N=5,n=16, p=0.002 in the other direction) is reported in `README.md`, not hidden | All code is present and `--help`-clean in this snapshot: `python -m synthetic.directional_recovery_sweep --seeds 0,1,2,3,4,5,6,7,8,9 --gt-seeds 0 --n-orig-sweep 2,3,4,5 --n-diff-steps-sweep 8,16,32,64,128 --samplers euler,exact --n-train-iters-sweep 10000 --n-samples 40000 --target-norm 0.625` (this is a real, multi-hour, GPU-scale sweep — not a smoke test) |
| 7 | Everything else in `RESULTS.md`'s tables (vision/NYUDv2/PASCAL, most PDE ablation axes) | Not established — template only, values are `[TBD]`, never run against real data | n/a | n/a |

**Row 6's one remaining caveat**: the raw evidence (per-seed CSVs) backing the now-published
`README.md` numbers isn't part of this git snapshot, only on the author's machine — worth
preserving deliberately (e.g. an explicit archive) rather than treating `README.md`'s prose as a
substitute for the underlying data, in case anyone needs to re-derive or audit a number later.

## 4. Established vs. current limitations

**Established** (statistically significant or explicitly-scoped point estimates, reproducible from
this snapshot):
- Core drift correctness: hypoellipticity holds across the tested coupling/damping/N grid (row 1).
- Synthetic coupled-OU: CCLD's step-count efficiency advantage over DDPM, and its correlation
  accuracy over SDM, both at a single coupling_strength=0.6 (rows 2-4).
- Antisymmetric/skew coupling improves on symmetric-only coupling at matched magnitude, broadly
  across N and step count under the exact sampler (row 6).

**Current limitations**:
- Every synthetic-OU result is at exactly one coupling_strength (0.6).
- The PDE result (row 5) is a single seed with no baseline comparison across coupling structures.
- An CLD baseline does not exist yet for the synthetic-OU result.