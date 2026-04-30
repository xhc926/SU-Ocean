#!/bin/bash
# Cross-factor 融合消融：保留 3 要素输入，禁用 cross-factor 融合（itransformerUniAbl）。
# 与 run_multiscale.sh 同风格：固定 ALL4（area3）、多尺度、同一组超参。
#

DATA=ALL4
SEQ_LENS=(16 24 32 48)
SCALE_SETS=("2 1")
MODEL=itransformerUniAbl

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  HALF=$((SEQ_LEN / 2))
  LABEL_LEN=$HALF
  PRED_LEN=$HALF
  for SCALES in "${SCALE_SETS[@]}"; do
    echo "=========================================="
    echo ">>> Running multiscale: model=$MODEL, data=$DATA, seq_len=$SEQ_LEN, label_len=$LABEL_LEN, pred_len=$PRED_LEN, scales=$SCALES"
    echo "=========================================="
    python -u run.py \
      --checkpoints /root/autodl-tmp/ms/checkpoints/area3 \
      --log_dir /root/autodl-tmp/ms/logs/area3 \
      --results_dir /root/autodl-tmp/ms/results/area3 \
      --model $MODEL \
      --data $DATA \
      --root_path /root/autodl-tmp/data/upsampled/1-12/area3 \
      --features M \
      --attn prob \
      --freq w \
      --seq_len $SEQ_LEN \
      --label_len $LABEL_LEN \
      --pred_len $PRED_LEN \
      --d_model 64 \
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
      --itr 5 \
      --use_multi_scale \
      --scales $SCALES 
      echo ""
  done
done

echo ">>> All cross-factor ablation runs completed."
