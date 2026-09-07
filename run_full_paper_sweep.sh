#!/usr/bin/env bash
set -uo pipefail

N_SEEDS="${1:-20}"
N_ITERS="${2:-10000}"
N_SAMPLES="${3:-40000}"
PDE_DATA_ROOT="${4:-/home/ubuntu/metis-v1-storage/CSHM-data/multiphysics}"
PDE_N_EPOCHS="${5:-150}"
PDE_N_DIFF_STEPS="${6:-32}"
PDE_SEEDS="${7:-0,1,2,3,4,5,6,7,8,9}"
PDE_NUM_WORKERS="${8:-16}"
PDE_ARCHES="${9:-attention,fno,unet_model}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PROBLEMS=(TE_heat)
METHODS=(csho ddpm sdm)
IFS=',' read -ra ARCHES <<< "${PDE_ARCHES}"
declare -A PDE_LR=(
  [attention]="0.01"
  [fno]="0.003"
  [unet_model]="0.01"
)
declare -A PDE_BATCH_SIZE=(
  [attention]="1024"
  [fno]="256"
  [unet_model]="256"
)

echo "############################################################"
echo "### Pre-flight: checking pde/ score-arch dependencies    ###"
echo "############################################################"
if [[ " ${ARCHES[*]} " == *" fno "* || " ${ARCHES[*]} " == *" unet_model "* ]]; then
  python -c "
import sys
try:
    import neuralop  # noqa: F401
except ImportError:
    print('FATAL: neuraloperator is not installed in this environment (needed for --score-arch fno).')
    print('        pip install neuraloperator')
    sys.exit(1)
try:
    from pde.unet_model_score_net import UNetModelScoreNetwork  # noqa: F401
except Exception as e:
    print(f'FATAL: pde/unet_model_score_net.py failed to import (needed for --score-arch unet_model): {e}')
    sys.exit(1)
print('OK: fno and unet_model dependencies import cleanly.')
" || exit 1
else
  echo "Skipping fno/unet_model dependency check -- not in ARCHES=(${ARCHES[*]})."
fi

echo "############################################################"
echo "### PHASE 1: synthetic/ step-count x N=2..5 sweep         ###"
echo "############################################################"
bash run_stepcount_sweep.sh "${N_SEEDS}" "${N_ITERS}" "${N_SAMPLES}"

echo "############################################################"
echo "### PHASE 2: pde/ multiphysics grid                       ###"
echo "############################################################"
echo "=== data-root: ${PDE_DATA_ROOT}, n-epochs: ${PDE_N_EPOCHS}, n-diff-steps: ${PDE_N_DIFF_STEPS}, seeds: ${PDE_SEEDS}, num-workers: ${PDE_NUM_WORKERS}, arches: ${ARCHES[*]} ==="

for PROBLEM in "${PROBLEMS[@]}"; do
  for ARCH in "${ARCHES[@]}"; do
    OUT_DIR="results/experiment_2_physics/${PROBLEM}_${ARCH}"
    mkdir -p "${OUT_DIR}"
    for METHOD in "${METHODS[@]}"; do
      RESULT_FILE="${OUT_DIR}/${METHOD}_results.json"
      if [ -f "${RESULT_FILE}" ]; then
        echo "--- ${PROBLEM} / ${METHOD} / ${ARCH}: already present at ${RESULT_FILE}, skipping ---"
        continue
      fi
      echo "--- ${PROBLEM} / ${METHOD} / ${ARCH}: training (lr=${PDE_LR[${ARCH}]}, batch-size=${PDE_BATCH_SIZE[${ARCH}]}) ---"
      python -m pde.run_experiment \
        --config pde/config.yaml \
        --data-root "${PDE_DATA_ROOT}" \
        --problem "${PROBLEM}" \
        --method "${METHOD}" \
        --score-arch "${ARCH}" \
        --n-epochs "${PDE_N_EPOCHS}" \
        --n-diff-steps "${PDE_N_DIFF_STEPS}" \
        --batch-size "${PDE_BATCH_SIZE[${ARCH}]}" \
        --seeds "${PDE_SEEDS}" \
        --lr "${PDE_LR[${ARCH}]}" \
        --num-workers "${PDE_NUM_WORKERS}" \
        --out-dir "${OUT_DIR}" \
        || echo "!!! FAILED: ${PROBLEM} / ${METHOD} / ${ARCH} -- see output above, continuing with the rest of the grid !!!"
    done
  done
done

echo "############################################################"
echo "### Done.                                                  ###"
echo "############################################################"
echo "=== Synthetic sweep: results/experiment_3_synthetic/stepcount_sweep_seeds${N_SEEDS}_iters${N_ITERS}/ ==="
echo "=== PDE grid: results/experiment_2_physics/{problem}_{arch}/{method}_results.json ==="