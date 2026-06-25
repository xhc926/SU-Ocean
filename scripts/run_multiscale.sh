#!/bin/bash
# Multiscale: ALL2（swh / u10 / v10）+ iTransformerUHSM, ALL4（sal / ssh / uo / vo）+ iTransformerUHSM4。
# or EMAformerUHSM/EMAformerUHSM4
# data：/root/autodl-tmp/ms/results/area3
# seq_len ∈ {16,24,32,48}，label_len = pred_len = seq_len/2。
# SCALE_SETS:  run.py --scales several integers。
#
# 
# bash scripts/run_multiscale.sh

DATA=ALL2
SEQ_LENS=(16 24 32) # for seq_len=48, let d_model=64
SCALE_SETS=("2 1" "4 2 1" "3 1")

MODEL=emaformerUHSM

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  HALF=$((SEQ_LEN / 2))
  LABEL_LEN=$HALF
  PRED_LEN=$HALF
  for SCALES in "${SCALE_SETS[@]}"; do
    echo "=========================================="
    echo ">>> Running multiscale: model=$MODEL, data=$DATA, seq_len=$SEQ_LEN, label_len=$LABEL_LEN, pred_len=$PRED_LEN, scales=$SCALES"
    echo "=========================================="
    python -u run.py \
      --checkpoints /root/autodl-tmp/${MODEL}/ms/checkpoints/area2 \
      --log_dir /root/autodl-tmp/${MODEL}/ms/logs/area2 \
      --results_dir /root/autodl-tmp/${MODEL}/ms/results/area2 \
      --land_mask_path /root/autodl-tmp/data/upsampled/1-4/area2/land_mask.pkl \
      --model $MODEL \
      --data $DATA \
      --root_path /root/autodl-tmp/data/upsampled/1-4/area2 \
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
