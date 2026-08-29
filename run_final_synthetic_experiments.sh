#!/usr/bin/env bash
set -euo pipefail

N_SEEDS="${1:-20}"
N_ITERS="${2:-10000}"
N_SAMPLES="${3:-40000}"
N_DIFF_STEPS="${4:-500}"
DT=$(python3 -c "print(1.0/${N_DIFF_STEPS})")
SEEDS=$(seq -s, 0 $((N_SEEDS - 1)))

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

BASELINE_DIR="results/experiment_3_synthetic/final_baselines_seeds${N_SEEDS}_iters${N_ITERS}_steps${N_DIFF_STEPS}"
CSHO_OUT_DIR="results/experiment_3_synthetic/final_champion_seeds${N_SEEDS}_iters${N_ITERS}_steps${N_DIFF_STEPS}"

echo "=== Config: seeds=0..$((N_SEEDS-1)) (n=${N_SEEDS}), n_train_iters=${N_ITERS}, n_samples=${N_SAMPLES}, n_diff_steps=${N_DIFF_STEPS}, dt=${DT} ==="
echo "=== Baselines -> ${BASELINE_DIR}, champion -> ${CSHO_OUT_DIR} ==="

for N in 2 3 4 5; do
  for METHOD in ddpm sdm; do
    OUT="${BASELINE_DIR}/N${N}_${METHOD}"
    if [ -f "${OUT}/${METHOD}_results.json" ]; then
      echo "--- N=${N} ${METHOD}: already present at ${OUT}, skipping ---"
      continue
    fi
    echo "--- N=${N} ${METHOD}: training ---"
    python -m synthetic.run_experiment \
      --method "${METHOD}" \
      --N "${N}" \
      --coupling-strength 0.6 \
      --time-scale-schedule vp_linear \
      --n-diff-steps "${N_DIFF_STEPS}" \
      --dt "${DT}" \
      --n-train-iters "${N_ITERS}" \
      --n-samples "${N_SAMPLES}" \
      --seeds "${SEEDS}" \
      --out-dir "${OUT}"
  done
done

echo "--- CSHO-Tikhonov champion, N=2..5 ---"
python -m synthetic.anderson_tikhonov_n_sweep \
  --seeds "${SEEDS}" \
  --n-train-iters "${N_ITERS}" \
  --n-samples "${N_SAMPLES}" \
  --n-diff-steps "${N_DIFF_STEPS}" \
  --dt "${DT}" \
  --n-sweep 2,3,4,5 \
  --out-dir "${CSHO_OUT_DIR}" \
  --baseline-dir "${BASELINE_DIR}"

echo "=== Done. Final comparison written to ${CSHO_OUT_DIR}/tikhonov_n_sweep_summary.csv ==="
echo "=== Per-seed table (for Wilcoxon): ${CSHO_OUT_DIR}/tikhonov_n_sweep_per_seed.csv ==="
echo "=== Significance table: ${CSHO_OUT_DIR}/tikhonov_n_sweep_significance.csv ==="