#!/bin/bash
# Cross-factor 融合消融：保留 3 要素输入，禁用 cross-factor 融合（itransformerUniAbl）。
# 与 run_multiscale.sh 同风格：固定 ALL2（area2）、多尺度、同一组超参。
#
# 注意：
#   - 不要传 --single_factor_ablation（run.py 里 type=bool 会把字符串 "False" 误判为 True）。
#   - 也不要用 True：那是“单要素输入”消融；本模型是三分支 + 无跨要素 MLP。
#   - 请在 uniocean 目录下执行：  bash scripts/run_ablation.sh
#   - 若要对照完整版，把 MODEL 改为 itransformerUniOcean 即可（其余参数保持一致）。

DATA=ALL2
SEQ_LENS=(16 24 32 48)
SCALE_SETS=("2 1")
MODEL=itransformerUniAbl

for SEQ_LEN in "${SEQ_LENS[@]}"; do
  HALF=$((SEQ_LEN / 2))
  LABEL_LEN=$HALF
  PRED_LEN=$HALF
  for SCALES in "${SCALE_SETS[@]}"; do
    echo "=========================================="
    echo ">>> Running cross-factor ablation: model=$MODEL, data=$DATA, seq_len=$SEQ_LEN, label_len=$LABEL_LEN, pred_len=$PRED_LEN, scales=$SCALES"
    echo "=========================================="
    python -u run.py \
      --checkpoints /root/autodl-tmp/ms/checkpoints_abl/ \
      --log_dir /root/autodl-tmp/ms/logs_abl/ \
      --results_dir /root/autodl-tmp/ms/results_abl/ \
      --model $MODEL \
      --data $DATA \
      --root_path /root/autodl-tmp/data/upsampled/1-4/area2 \
      --land_mask_path /root/autodl-tmp/data/upsampled/1-4/area2/land_mask.pkl \
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
      --d_ff 128 \
      --batch_size 32 \
      --train_epochs 50 \
      --learning_rate 0.0001 \
      --lradj type3 \
      --dropout 0.1 \
      --patience 5 \
      --itr 7 \
      --use_multi_scale \
      --scales $SCALES
    echo ""
  done
done

echo ">>> All cross-factor ablation runs completed."
