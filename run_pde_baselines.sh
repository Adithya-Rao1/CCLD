#!/usr/bin/env bash
set -uo pipefail

PDE_DATA_ROOT="${1:-/home/ubuntu/metis-v1-storage/CSHM-data/multiphysics}"
PROBLEM="${2:-TE_heat}"
ARCH="${3:-attention}"
N_EPOCHS="${4:-150}"
N_DIFF_STEPS="${5:-32}"
BATCH_SIZE="${6:-1024}"
LR="${7:-0.01}"
SEEDS="${8:-0,1,2,3,4,5,6,7,8,9}"
NUM_WORKERS="${9:-16}"
CCLD_TAU="${10:-3.5}"
ENCODER_N_EPOCHS="${11:-${N_EPOCHS}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

OUT_DIR="results/experiment_2_physics/${PROBLEM}_${ARCH}"
ENCODER_DIR="${OUT_DIR}/shared_encoders"
mkdir -p "${OUT_DIR}"

echo "${PROBLEM}/${ARCH}: n-epochs=${N_EPOCHS} n-diff-steps=${N_DIFF_STEPS} batch-size=${BATCH_SIZE} lr=${LR} seeds=${SEEDS} ccld-tau=${CCLD_TAU}"

MISSING_ENCODER_SEEDS=""
for SEED in ${SEEDS//,/ }; do
  if [ ! -f "${ENCODER_DIR}/seed${SEED}.pt" ]; then
    MISSING_ENCODER_SEEDS="${MISSING_ENCODER_SEEDS}${MISSING_ENCODER_SEEDS:+,}${SEED}"
  fi
done
if [ -n "${MISSING_ENCODER_SEEDS}" ]; then
  echo "shared encoders: training seeds ${MISSING_ENCODER_SEEDS}"
  python -m pde.run_experiment \
    --config pde/config.yaml \
    --data-root "${PDE_DATA_ROOT}" \
    --problem "${PROBLEM}" \
    --encoder-only \
    --save-encoder-dir "${ENCODER_DIR}" \
    --score-arch "${ARCH}" \
    --n-epochs "${ENCODER_N_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --seeds "${MISSING_ENCODER_SEEDS}" \
    --lr "${LR}" \
    --num-workers "${NUM_WORKERS}" \
    || { echo "FAILED: shared encoder training"; exit 1; }
else
  echo "shared encoders: already present for all seeds in ${ENCODER_DIR}, skipping"
fi

for METHOD in ccld ddpm sdm; do
  RESULT_FILE="${OUT_DIR}/${METHOD}_results.json"
  if [ -f "${RESULT_FILE}" ]; then
    echo "${METHOD}: already present at ${RESULT_FILE}, skipping"
    continue
  fi
  echo "${METHOD}: training"
  python -m pde.run_experiment \
    --config pde/config.yaml \
    --data-root "${PDE_DATA_ROOT}" \
    --problem "${PROBLEM}" \
    --method "${METHOD}" \
    --score-arch "${ARCH}" \
    --n-epochs "${N_EPOCHS}" \
    --n-diff-steps "${N_DIFF_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --seeds "${SEEDS}" \
    --lr "${LR}" \
    --num-workers "${NUM_WORKERS}" \
    --out-dir "${OUT_DIR}" \
    --frozen-encoder-dir "${ENCODER_DIR}" \
    --ccld-tau "${CCLD_TAU}" \
    || echo "FAILED: ${PROBLEM}/${ARCH}/${METHOD}"
done

echo "Done. Results in ${OUT_DIR}/{ccld,ddpm,sdm}_results.json"
