#!/bin/bash
# Multiscale: ALL4 四要素（sal / ssh / uo / vo）+ iTransformerUniOcean4。
# 数据目录：/root/autodl-tmp/ms/results/area3，需含 sal.pkl ssh.pkl uo.pkl vo.pkl。
# 对 seq_len ∈ {16,24,32,48} 循环，label_len = pred_len = seq_len/2。
# SCALE_SETS: 每组为传给 run.py --scales 的若干整数（空格分隔）。
#
# 用法:
#   bash scripts/run_multiscale.sh

DATA=ALL4
SEQ_LENS=(16 24 32 48)
SCALE_SETS=("2 1")

MODEL=itransformerUniOcean4

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
      --itr 5 \
      --use_multi_scale \
      --scales $SCALES 
      echo ""
  done
done

echo ">>> All multiscale runs completed."
