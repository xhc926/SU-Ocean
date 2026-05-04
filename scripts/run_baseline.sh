#!/bin/bash
# Baseline: 单要素 iTransformer，默认 swh/u10/v10（area2，1075 点）
# seq_len ∈ {16,24,32,48}，label_len = pred_len = seq_len/2
# 用法（需在仓库根目录）:
#   bash scripts/run_baseline.sh              # 默认三要素 × 四组窗口
#   bash scripts/run_baseline.sh swh_1_4    # 仅跑指定 data 名
#
# enc_in/dec_in/c_out 由 run.py 的 data_parser 按 --data 与 --features 自动设置，无需在脚本里写。

<<<<<<< HEAD
VARS=(sal_1_12 ssh_1_12 uo_1_12 vo_1_12)
=======
VARS=(swh_1_4 u10_1_4 v10_1_4)
>>>>>>> origin/area2
SEQ_LENS=(32 48)
if [ -n "$1" ]; then
  VARS=("$1")
fi

MODEL=itransformer
for SEQ_LEN in "${SEQ_LENS[@]}"; do
  HALF=$((SEQ_LEN / 2))
  LABEL_LEN=$HALF
  PRED_LEN=$HALF
  for VAR in "${VARS[@]}"; do
    echo "=========================================="
    echo ">>> Running baseline: model=$MODEL, data=$VAR, seq_len=$SEQ_LEN, label_len=$LABEL_LEN, pred_len=$PRED_LEN"
    echo "=========================================="
    python -u run.py \
      --checkpoints /root/autodl-tmp/baseline/area3/checkpoints/ \
      --log_dir /root/autodl-tmp/baseline/area3/logs \
      --results_dir /root/autodl-tmp/baseline/area3/results \
      --model $MODEL \
      --data $VAR \
      --land_mask_path /root/autodl-tmp/data/upsampled/1-12/area3/land_mask.pkl \
      --features M \
      --attn prob \
      --freq w \
      --seq_len $SEQ_LEN \
      --label_len $LABEL_LEN \
      --pred_len $PRED_LEN \
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
      --itr 5
    echo ""
  done
done

echo ">>> All baselines completed."
