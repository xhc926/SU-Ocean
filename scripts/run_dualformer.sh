#!/bin/bash
# Dualformer 训练脚本（area2 单要素，与 run_baseline.sh 风格一致）
# 用法（需在仓库根目录 uniocean/）:
#   bash scripts/run_dualformer.sh
#   bash scripts/run_dualformer.sh swh_1_4
#
# enc_in/dec_in/c_out 由 run.py 的 data_parser 按 --data 与 --features 自动设置。

VARS=(v10_1_4)
SEQ_LENS=(48)
if [ -n "$1" ]; then
  VARS=("$1")
fi

MODEL=dualformer
for SEQ_LEN in "${SEQ_LENS[@]}"; do
  HALF=$((SEQ_LEN / 2))
  LABEL_LEN=$HALF
  PRED_LEN=$HALF
  for VAR in "${VARS[@]}"; do
    echo "=========================================="
    echo ">>> Running Dualformer: data=$VAR, seq_len=$SEQ_LEN, label_len=$LABEL_LEN, pred_len=$PRED_LEN"
    echo "=========================================="
    python -u run.py \
      --checkpoints /root/autodl-tmp/dualformer/checkpoints/ \
      --log_dir /root/autodl-tmp/dualformer/logs \
      --results_dir /root/autodl-tmp/dualformer/results \
      --model $MODEL \
      --data $VAR \
      --land_mask_path /root/autodl-tmp/data/upsampled/1-4/area2/land_mask.pkl \
      --features M \
      --attn prob \
      --freq w \
      --seq_len $SEQ_LEN \
      --label_len $LABEL_LEN \
      --pred_len $PRED_LEN \
      --d_model 2048 \
      --n_heads 4 \
      --e_layers 2 \
      --d_layers 1 \
      --d_ff 2048 \
      --batch_size 32 \
      --train_epochs 50 \
      --learning_rate 0.0001 \
      --lradj type3 \
      --dropout 0.1 \
      --patience 5 \
      --itr 7
    echo ""
  done
done

echo ">>> Dualformer runs completed."
