#!/usr/bin/env bash
set -euo pipefail

N_SEEDS="${1:-20}"
N_ITERS="${2:-10000}"
N_SAMPLES="${3:-40000}"
STEP_COUNTS=(8 16 32 64 128)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

for STEPS in "${STEP_COUNTS[@]}"; do
  echo "############################################"
  echo "### n_diff_steps=${STEPS} ###"
  echo "############################################"
  bash run_final_synthetic_experiments.sh "${N_SEEDS}" "${N_ITERS}" "${N_SAMPLES}" "${STEPS}"
done

STEP_COUNTS_CSV=$(IFS=,; echo "${STEP_COUNTS[*]}")
SWEEP_OUT_DIR="results/experiment_3_synthetic/stepcount_sweep_seeds${N_SEEDS}_iters${N_ITERS}"

python -m synthetic.aggregate_stepcount_sweep \
  --seeds "${N_SEEDS}" --iters "${N_ITERS}" \
  --step-counts "${STEP_COUNTS_CSV}" \
  --out-dir "${SWEEP_OUT_DIR}"

echo "=== Combined step-count comparison written to ${SWEEP_OUT_DIR}/ ==="