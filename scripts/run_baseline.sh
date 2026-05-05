#!/bin/bash
# Baseline：单要素 EMAformer，依次跑 area2 与 area3（一次性跑完）
#   area2：swh_1_4, u10_1_4, v10_1_4（1-4 / area2，1075 点）
#   area3：sal_1_12, ssh_1_12, uo_1_12, vo_1_12（1-12 / area3，637 点）
# seq_len 默认多组，label_len = pred_len = seq_len/2
#
# 用法（需在仓库根目录）:
#   bash scripts/run_baseline.sh              # 先 area2 全部变量，再 area3 全部变量
#   bash scripts/run_baseline.sh area2        # 仅 area2
#   bash scripts/run_baseline.sh area3        # 仅 area3
#   bash scripts/run_baseline.sh area2 swh_1_4  # 仅 area2 下单个 data（可选）
#
# enc_in/dec_in/c_out 由 run.py 的 data_parser 按 --data 与 --features 自动设置。
# ckpt / log / results：/root/autodl-tmp/${MODEL}/area{2,3}/... 随 MODEL 变化，不写死模型名。

SEQ_LENS=(16 24 32 48)
MODEL=emaformer
OUT_ROOT=/root/autodl-tmp

run_one_region() {
  local REGION_NAME="$1"
  local CHECKPOINTS="$2"
  local LOG_DIR="$3"
  local RESULTS_DIR="$4"
  local LAND_MASK="$5"
  shift 5
  local VARS=("$@")

  for SEQ_LEN in "${SEQ_LENS[@]}"; do
    local HALF=$((SEQ_LEN / 2))
    local LABEL_LEN=$HALF
    local PRED_LEN=$HALF
    for VAR in "${VARS[@]}"; do
      echo "=========================================="
      echo ">>> [$REGION_NAME] model=$MODEL data=$VAR seq_len=$SEQ_LEN label_len=$LABEL_LEN pred_len=$PRED_LEN"
      echo "=========================================="
      python -u run.py \
        --checkpoints "$CHECKPOINTS" \
        --log_dir "$LOG_DIR" \
        --results_dir "$RESULTS_DIR" \
        --model "$MODEL" \
        --data "$VAR" \
        --land_mask_path "$LAND_MASK" \
        --features M \
        --attn prob \
        --freq w \
        --seq_len "$SEQ_LEN" \
        --label_len "$LABEL_LEN" \
        --pred_len "$PRED_LEN" \
        --d_model 128 \
        --n_heads 4 \
        --e_layers 2 \
        --d_layers 1 \
        --d_ff 64 \
        --batch_size 32 \
        --train_epochs 50 \
        --learning_rate 0.0001 \
        --lradj type3 \
        --dropout 0.1 \
        --patience 5 \
        --itr 1
      echo ""
    done
  done
}

# ---------- area2 ----------
AREA2_VARS=(swh_1_4 u10_1_4 v10_1_4)
AREA2_CKPT="${OUT_ROOT}/${MODEL}/area2/checkpoints/"
AREA2_LOG="${OUT_ROOT}/${MODEL}/area2/logs"
AREA2_RES="${OUT_ROOT}/${MODEL}/area2/results"
AREA2_MASK=/root/autodl-tmp/data/upsampled/1-4/area2/land_mask.pkl

# ---------- area3 ----------
AREA3_VARS=(sal_1_12 ssh_1_12 uo_1_12 vo_1_12)
AREA3_CKPT="${OUT_ROOT}/${MODEL}/area3/checkpoints/"
AREA3_LOG="${OUT_ROOT}/${MODEL}/area3/logs"
AREA3_RES="${OUT_ROOT}/${MODEL}/area3/results"
AREA3_MASK=/root/autodl-tmp/data/upsampled/1-12/area3/land_mask.pkl

if [ -n "$1" ] && [ "$1" != "area2" ] && [ "$1" != "area3" ]; then
  echo "Usage: $0 [area2|area3] [optional_single_data_name]"
  exit 1
fi

if [ -z "$1" ] || [ "$1" = "area2" ]; then
  if [ -n "$2" ] && [ "$1" = "area2" ]; then
    run_one_region "area2" "$AREA2_CKPT" "$AREA2_LOG" "$AREA2_RES" "$AREA2_MASK" "$2"
  else
    run_one_region "area2" "$AREA2_CKPT" "$AREA2_LOG" "$AREA2_RES" "$AREA2_MASK" "${AREA2_VARS[@]}"
  fi
fi

if [ -z "$1" ] || [ "$1" = "area3" ]; then
  if [ -n "$2" ] && [ "$1" = "area3" ]; then
    run_one_region "area3" "$AREA3_CKPT" "$AREA3_LOG" "$AREA3_RES" "$AREA3_MASK" "$2"
  else
    run_one_region "area3" "$AREA3_CKPT" "$AREA3_LOG" "$AREA3_RES" "$AREA3_MASK" "${AREA3_VARS[@]}"
  fi
fi

echo ">>> All baselines completed."
