#!/bin/bash
# Dualformer 
# 
#   bash scripts/run_dualformer.sh
#   bash scripts/run_dualformer.sh swh_1_4

VARS=(sal_1_12 ssh_1_12 uo_1_12 vo_1_12)
SEQ_LENS=(16 24 32 48)
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
      --land_mask_path /root/autodl-tmp/data/upsampled/1-4/area3/land_mask.pkl \
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
      --itr 3
    echo ""
  done
done

echo ">>> Dualformer runs completed."
