#dropout=0.2
#python -u run.py --model itransformerUniOcean --data ALL2 --attn prob --freq w --root_path /root/autodl-tmp/ALL/ --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --batch_size 32 --use_multi_scale --train_epoch 50

#python -u run.py --model itransformerUniOcean --data ALL2 --attn prob --freq w --root_path /root/autodl-tmp/ALL/ --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 20 --use_multi_scale --train_epoch 50

#python -u run.py --model itransformerUniOcean --data ALL2 --attn prob --freq w --root_path /root/autodl-tmp/ALL/ --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 12 --use_multi_scale --train_epoch 50

#python -u run.py --model itransformer --data OHC5 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --batch_size 32 --train_epoch 50

#python -u run.py --model itransformer --data OHC5 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --train_epoch 50

#python -u run.py --model itransformer --data OHC5 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --train_epoch 50

#python -u run.py --model itransformer --data SAL8 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --batch_size 32 --train_epoch 50

#python -u run.py --model itransformer --data SAL8 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --train_epoch 50

#python -u run.py --model itransformer --data SAL8 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --train_epoch 50

#python -u run.py --model itransformer --data OISST4 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --batch_size 32 --train_epoch 50

#python -u run.py --model itransformer --data OISST4 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --train_epoch 50

#python -u run.py --model itransformer --data OISST4 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --train_epoch 50
#dropout=0.05
python -u run.py --model itransformerUniOcean --data ALL2 --attn prob --freq w --root_path /root/autodl-tmp/ALL/ --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --batch_size 32 --use_multi_scale --train_epoch 50 --dropout 0.05

python -u run.py --model itransformerUniOcean --data ALL2 --attn prob --freq w --root_path /root/autodl-tmp/ALL/ --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 20 --use_multi_scale --train_epoch 50 --dropout 0.05

python -u run.py --model itransformerUniOcean --data ALL2 --attn prob --freq w --root_path /root/autodl-tmp/ALL/ --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 12 --use_multi_scale --train_epoch 50 --dropout 0.05

python -u run.py --model itransformer --data OHC5 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --batch_size 32 --train_epoch 50 --dropout 0.05

python -u run.py --model itransformer --data OHC5 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --train_epoch 50 --dropout 0.05

python -u run.py --model itransformer --data OHC5 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --train_epoch 50 --dropout 0.05

python -u run.py --model itransformer --data SAL8 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --batch_size 32 --train_epoch 50 --dropout 0.05

python -u run.py --model itransformer --data SAL8 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --train_epoch 50 --dropout 0.05

python -u run.py --model itransformer --data SAL8 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --train_epoch 50 --dropout 0.05

python -u run.py --model itransformer --data OISST4 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 32 --label_len 16 --pred_len 16 --lradj type3 --batch_size 32 --train_epoch 50 --dropout 0.05

python -u run.py --model itransformer --data OISST4 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 64 --label_len 32 --pred_len 32 --lradj type3 --batch_size 24 --train_epoch 50 --dropout 0.05

python -u run.py --model itransformer --data OISST4 --attn prob --freq w --root_path /root/autodl-tmp/  --seq_len 96 --label_len 48 --pred_len 48 --lradj type3 --batch_size 16 --train_epoch 50 --dropout 0.05