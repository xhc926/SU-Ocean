#!/bin/bash
# OLinear 单要素 baseline（area3，与 data/generate_corrmat.py 产出的 .npy 配套）
# 需已生成：{AREA_ROOT}/olinear_q/{sal|uo|vo|ssh}_{seq_len|pred_len}_ratio{RATIO_TAG}.npy
# 若 label_len=pred_len=seq_len/2，必须对每个 SEQ_LEN 同时生成 lag=SEQ_LEN 与 lag=SEQ_LEN/2（例如 seq=16 时要 sal_16 与 sal_8）。
#   python data/generate_corrmat.py --area-root "${AREA_ROOT}" --lags 8,12,16,24,32,48
#
# 用法（在仓库根目录 uniocean/ 下执行）:
#   bash scripts/run_olinear.sh
#   bash scripts/run_olinear.sh sal_1_12          # 只跑一个 --data
#
# 可通过环境变量覆盖路径或超参，例如:
#   AREA_ROOT=/data/area3 SEQ_LENS="32 48" bash scripts/run_olinear.sh
#   EMBED_SIZE=16 D_MODEL=512 D_FF=512 LR=0.001 bash scripts/run_olinear.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 与 run.py 里 sal_1_12 等的 root 一致；corrmat 默认写在 AREA_ROOT/olinear_q/
AREA_ROOT="${AREA_ROOT:-/root/autodl-tmp/data/upsampled/1-4/area2}"
Q_SUBDIR="${Q_SUBDIR:-olinear_q}"
# 与 generate_corrmat.py 默认 train_ratio*base_ratio 一致（文件名里的 ratio0.7）
RATIO_TAG="${RATIO_TAG:-0.7}"

CHECKPOINTS="${CHECKPOINTS:-/root/autodl-tmp/olinear/area2/checkpoints}"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/olinear/area2/logs}"
RESULTS_DIR="${RESULTS_DIR:-/root/autodl-tmp/olinear/area2/results}"

VARS=(swh_1_4 u10_1_4 v10_1_4)
SEQ_LENS=(16 24 32 48)

if [ -n "${1:-}" ]; then
  VARS=("$1")
fi

MODEL="olinear"

# Q .npys 的文件名前缀：sal_1_12 -> sal（取第一个 '_' 前字段）
q_prefix_for_data() {
  local d="$1"
  echo "${d%%_*}"
}

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  HALF=$((SEQ_LEN / 2))
  LABEL_LEN=$HALF
  PRED_LEN=$HALF
  for VAR in "${VARS[@]}"; do
    QP="$(q_prefix_for_data "${VAR}")"
    Q_IN="${Q_SUBDIR}/${QP}_${SEQ_LEN}_ratio${RATIO_TAG}.npy"
    Q_OUT="${Q_SUBDIR}/${QP}_${PRED_LEN}_ratio${RATIO_TAG}.npy"

    if [ ! -f "${AREA_ROOT}/${Q_IN}" ] || [ ! -f "${AREA_ROOT}/${Q_OUT}" ]; then
      echo "[WARN] Missing Q matrix for ${VAR} (seq=${SEQ_LEN}, pred=${PRED_LEN}):"
      echo "       expect ${AREA_ROOT}/${Q_IN}"
      echo "       and    ${AREA_ROOT}/${Q_OUT}"
      echo "       Run: python data/generate_corrmat.py --area-root ${AREA_ROOT} --lags ...,${SEQ_LEN},...,${PRED_LEN},..."
      echo ""
    fi

    echo "=========================================="
    echo ">>> OLinear  data=${VAR}  seq_len=${SEQ_LEN}  label_len=${LABEL_LEN}  pred_len=${PRED_LEN}"
    echo ">>> Q_in=${Q_IN}  Q_out=${Q_OUT}"
    echo "=========================================="

    python -u run.py \
      --checkpoints "${CHECKPOINTS}" \
      --log_dir "${LOG_DIR}" \
      --results_dir "${RESULTS_DIR}" \
      --model "${MODEL}" \
      --data "${VAR}" \
      --features M \
      --freq w \
      --seq_len "${SEQ_LEN}" \
      --label_len "${LABEL_LEN}" \
      --pred_len "${PRED_LEN}" \
      --q_mat_file "${Q_IN}" \
      --q_out_mat_file "${Q_OUT}" \
      --embed_size "${EMBED_SIZE:-1}" \
      --d_model "${D_MODEL:-128}" \
      --d_ff "${D_FF:-64}" \
      --e_layers "${E_LAYERS:-2}" \
      --dropout "${DROPOUT:-0.1}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --train_epochs "${TRAIN_EPOCHS:-50}" \
      --learning_rate "${LR:-0.0001}" \
      --lradj type3 \
      --patience "${PATIENCE:-5}" \
      --itr "${ITR:-3}"

    echo ""
  done
done

echo ">>> OLinear runs finished."
