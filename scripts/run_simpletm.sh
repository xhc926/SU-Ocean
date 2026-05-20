#!/bin/bash
# SimpleTM baseline
#
#   bash scripts/run_simpletm.sh
#   bash scripts/run_simpletm.sh sal_1_12    # only one --data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINTS="${CHECKPOINTS:-/root/autodl-tmp/simpletm/area3/checkpoints}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/simpletm/area3/logs}"
RESULTS_DIR="${RESULTS_DIR:-/root/autodl-tmp/simpletm/area3/results}"

VARS=(sal_1_12 ssh_1_12 uo_1_12 vo_1_12)
SEQ_LENS=(16 24 32 48)

if [ -n "${1:-}" ]; then
  VARS=("$1")
fi

MODEL="simpletm"

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  HALF=$((SEQ_LEN / 2))
  LABEL_LEN=$HALF
  PRED_LEN=$HALF
  for VAR in "${VARS[@]}"; do
    echo "=========================================="
    echo ">>> SimpleTM  data=${VAR}  seq_len=${SEQ_LEN}  label_len=${LABEL_LEN}  pred_len=${PRED_LEN}"
    echo "=========================================="

    python -u run.py \
      --checkpoints "${CHECKPOINTS}" \
      --log_dir "${LOG_DIR}" \
      --results_dir "${RESULTS_DIR}" \
      --model "${MODEL}" \
      --data "${VAR}" \
      --features M \
      --freq w \
      --embed timeF \
      --seq_len "${SEQ_LEN}" \
      --label_len "${LABEL_LEN}" \
      --pred_len "${PRED_LEN}" \
      --factor "${FACTOR:-3}" \
      --attn prob \
      --d_model "${D_MODEL:-128}" \
      --d_ff "${D_FF:-64}" \
      --e_layers "${E_LAYERS:-2}" \
      --d_layers "${D_LAYERS:-1}" \
      --n_heads "${N_HEADS:-4}" \
      --dropout "${DROPOUT:-0.1}" \
      --geomattn_dropout "${GEOMATTN_DROPOUT:-0.5}" \
      --requires_grad "${REQUIRES_GRAD:-1}" \
      --wv "${WV:-db1}" \
      --m "${M_LEVELS:-3}" \
      --simpletm_kernel_size "${SIMPLETM_KERNEL_SIZE:-0}" \
      --alpha "${ALPHA:-1.0}" \
      --simpletm_use_norm "${SIMPLETM_USE_NORM:-1}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --train_epochs "${TRAIN_EPOCHS:-50}" \
      --learning_rate "${LR:-0.0001}" \
      --lradj type3 \
      --patience "${PATIENCE:-5}" \
      --itr "${ITR:-3}"

    echo ""
  done
done

echo ">>> SimpleTM runs finished."
