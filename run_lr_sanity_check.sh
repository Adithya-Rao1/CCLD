#!/usr/bin/env bash
set -uo pipefail

PDE_DATA_ROOT="${1:-${PDE_DATA_ROOT:-pde/multiphysics-bench}}"
N_EPOCHS="${2:-50}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ARCHES=(fno unet_model)
LRS=(0.0001 0.0003 0.001 0.003)

echo "LR sanity check (post closed-form/FDT recalibration)"
echo "TE_heat/ccld/{fno,unet_model}                       "
echo "n_epochs=${N_EPOCHS}, seed=0, lr in {${LRS[*]}}     "

for ARCH in "${ARCHES[@]}"; do
  for LR in "${LRS[@]}"; do
    OUT_DIR="results/lr_sanity_check_v2/TE_heat_ccld_${ARCH}_lr${LR}"
    RESULT_FILE="${OUT_DIR}/ccld_results.json"
    if [ -f "${RESULT_FILE}" ]; then
      echo "${ARCH} lr=${LR}: already present at ${RESULT_FILE}, skipping"
      continue
    fi
    mkdir -p "${OUT_DIR}"
    echo "${ARCH} lr=${LR}: training"
    python -m pde.run_experiment \
      --config pde/config.yaml \
      --data-root "${PDE_DATA_ROOT}" \
      --problem TE_heat --method ccld --score-arch "${ARCH}" \
      --n-diff-steps 20 --n-epochs "${N_EPOCHS}" --batch-size 64 --seeds 0 \
      --lr "${LR}" \
      --out-dir "${OUT_DIR}" \
      || echo "!!! FAILED: ${ARCH} lr=${LR} -- see output above !!!"
  done
done

echo "Summary                                               "
python -c "
import json, os, glob

rows = []
for arch in ['fno', 'unet_model']:
    for lr in ['0.0001', '0.0003', '0.001', '0.003']:
        path = f'results/lr_sanity_check_v2/TE_heat_ccld_{arch}_lr{lr}/ccld_results.json'
        if not os.path.exists(path):
            rows.append((arch, lr, None))
            continue
        with open(path) as f:
            data = json.load(f)
        summary = data['summary']
        rows.append((arch, lr, summary))

print(f\"{'arch':10s} {'lr':8s} {'Re{Ez}_rel_l2':14s} {'Im{Ez}_rel_l2':14s} {'T_rel_l2':12s} {'explosions':10s}\")
for arch, lr, summary in rows:
    if summary is None:
        print(f'{arch:10s} {lr:8s} MISSING (run failed or was skipped)')
        continue
    rez = summary.get('Re{Ez}_rel_l2', {}).get('mean', float('nan'))
    iez = summary.get('Im{Ez}_rel_l2', {}).get('mean', float('nan'))
    t = summary.get('T_rel_l2', {}).get('mean', float('nan'))
    exp = summary.get('explosion_events', {}).get('mean', float('nan'))
    print(f'{arch:10s} {lr:8s} {rez:<14.4f} {iez:<14.4f} {t:<12.6f} {exp:<10.1f}')
"