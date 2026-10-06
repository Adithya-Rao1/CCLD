#!/usr/bin/env bash
set -euo pipefail

N_SEEDS="${1:-20}"
N_ITERS="${2:-10000}"
N_SAMPLES="${3:-40000}"
STEP_COUNTS=(8 16 32 64 128)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

for STEPS in "${STEP_COUNTS[@]}"; do
  echo "n_diff_steps=${STEPS}"
  bash run_final_synthetic_experiments.sh "${N_SEEDS}" "${N_ITERS}" "${N_SAMPLES}" "${STEPS}"
done

STEP_COUNTS_CSV=$(IFS=,; echo "${STEP_COUNTS[*]}")
SWEEP_OUT_DIR="results/experiment_3_synthetic/stepcount_sweep_seeds${N_SEEDS}_iters${N_ITERS}_samples${N_SAMPLES}"
CLD_SWEEP_OUT_DIR="${SWEEP_OUT_DIR}_independent"
CCLD_VS_CLD_OUT_DIR="results/experiment_3_synthetic/ccld_vs_cld_seeds${N_SEEDS}_iters${N_ITERS}_samples${N_SAMPLES}"

python -m synthetic.aggregate_stepcount_sweep \
  --seeds "${N_SEEDS}" --iters "${N_ITERS}" --n-samples "${N_SAMPLES}" \
  --step-counts "${STEP_COUNTS_CSV}" \
  --out-dir "${SWEEP_OUT_DIR}"

python -m synthetic.aggregate_stepcount_sweep \
  --seeds "${N_SEEDS}" --iters "${N_ITERS}" --n-samples "${N_SAMPLES}" \
  --step-counts "${STEP_COUNTS_CSV}" \
  --dir-template "results/experiment_3_synthetic/final_seeds{seeds}_iters{iters}_samples{samples}_steps{steps}_independent" \
  --out-dir "${CLD_SWEEP_OUT_DIR}"

python -m synthetic.compare_coupled_vs_uncoupled \
  --seeds "${N_SEEDS}" --iters "${N_ITERS}" --n-samples "${N_SAMPLES}" \
  --step-counts "${STEP_COUNTS_CSV}" \
  --out-dir "${CCLD_VS_CLD_OUT_DIR}"

python -m synthetic.merge_stepcount_sweep_runs \
  --summary-csv "${SWEEP_OUT_DIR}/stepcount_sweep_summary.csv" \
  --summary-csv "${CLD_SWEEP_OUT_DIR}/stepcount_sweep_summary.csv" \
  --significance-csv "${SWEEP_OUT_DIR}/stepcount_sweep_significance.csv" \
  --significance-csv "${CLD_SWEEP_OUT_DIR}/stepcount_sweep_significance.csv" \
  --out-dir "${SWEEP_OUT_DIR}"

echo "Combined step-count comparison (CCLD/DDPM/SGM/CLD) written to ${SWEEP_OUT_DIR}/"
echo "CCLD-vs-CLD significance written to ${CCLD_VS_CLD_OUT_DIR}/coupled_vs_independent_significance.csv"