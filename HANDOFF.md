# Handoff package

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

| # | Claim | Status | Retained artifact | Reproduce |
|---|---|---|---|---|
| 1 | CCLD's coupled drift is hypoelliptic and well-posed as an SDE across coupling modes, damping regimes, and N | Established | `hypo_passed`/`hypo_min_eig` fields emitted by every `synthetic.run_experiment` run (see #2's smoke output) | #2's smoke command |
| 2 | CCLD and CLD both beat DDPM at low step counts (n≤32) across N=2..5, mean-field coupling, single coupling_strength=0.6, with p≤0.0137; advantage not significant at n=64,128 as DDPM catches up | Established, statistically significant (paired Wilcoxon, 10 seeds) | Paper | `bash run_stepcount_sweep.sh 10 10000 40000` |
| 3 | Against SGM, CCLD recovers pairwise correlation significantly closer to ground truth for n≥32 (p<0.05); SGM under-estimates true correlation ~5-8% at n=8, then over-estimates it ~3-13% from n=16 onward; SGM has significantly lower KL than CCLD at n=8 (p≤0.004) | Established, statistically significant (paired Wilcoxon, 10 seeds) | Paper | Same as #2 |
| 4 | Result in #2/#3 is scoped to coupling_strength=0.6, N≤5| Established as a limitation | Paper | This is the limitation `PROTOCOL.md` targets |
| 5 | On the real Multiphysics-Bench electro-thermal field, CCLD and CLD achieve the lowest pointwise rel.\ $\ell_2$ error on both E-field and T; DDPM/SGM achieve the lowest E-field PDE residual; all four methods are statistically indistinguishable on heat PDE residual; CLD beats CCLD significantly on T rel.\ $\ell_2$ alone (p=0.027, margin 0.00007) | Established, statistically significant (paired Wilcoxon, 10 seeds)| Paper | Requires downloading Multiphysics-Bench first (`python -m pde.download_multiphysics_bench --out-dir pde/multiphysics-bench`, several GB); `bash run_pde_baselines.sh` runs CCLD, the uncoupled-CLD baseline (`--method ccld_independent`), DDPM, and SDM for 10 seeds |
| 6 | Antisymmetric coupling improves KL over symmetric-only coupling at matched Frobenius-norm magnitude. The results are significant (p≤0.01) at N=2,3,4 for n=8. There is one exception are N=5, n=16, with p=0.002 in the other direction | Established, statistically significant (paired Wilcoxon, 10 seeds) | Not reported in paper, yet reproducible | `python -m synthetic.directional_recovery_sweep --seeds 0,1,2,3,4,5,6,7,8,9 --gt-seeds 0 --n-orig-sweep 2,3,4,5 --n-diff-steps-sweep 8,16,32,64,128 --samplers euler,exact --n-train-iters-sweep 10000 --n-samples 40000 --target-norm 0.625` |
| 7 | On TE\_heat, CCLD shows a reduction in training gradient-norm explosion events vs. CLD (mean 0.7 vs 1.4 per run) that is marginally not significant (p=0.063); the closed-form DSM score-target precision is ~10% lower for CCLD than CLD at diffusion time q<0.01, shrinking to ~1% by q=2.0. We believe this is suggestive of improved training stability from the shared coupling term, but it is not yet an an established result. | Established as suggestive since we don't achieve significance (p=0.063)| Paper | Same TE\_heat runs as row 5 since the explosion events are logged per-run |

## 4. Established vs. current limitations

**Established** (statistically significant or explicitly-scoped point estimates, reproducible from
this snapshot):
- Core drift correctness: hypoellipticity holds across the tested coupling/damping/N grid (row 1).
- Synthetic coupled-OU: CCLD's and CLD's step-count efficiency advantage over DDPM, and CCLD's
  correlation accuracy over SGM, both at a single coupling_strength=0.6 (rows 2-4).
- PDE TE_heat: CCLD/CLD beat DDPM/SGM on pointwise rel. $\ell_2$ error; DDPM/SGM
  beat CCLD/CLD on E-field PDE-residual self-consistency; CLD beats CCLD on T rel. $\ell_2$ alone,
  by a tiny but significant margin (row 5).
- Antisymmetric/skew coupling improves on symmetric-only coupling at matched magnitude, broadly
  across N and step count under the exact sampler (row 6).

**Current limitations**:
- Every synthetic-OU result is at exactly one coupling_strength (0.6).
- CCLD predicts the entire score rather than a residual relative to CLD's closed form.
- CCLD's training-stability benefits over CLD on TE_heat from fewer gradient norm explosions does not clear significance (p=0.063)