import argparse
import os
import sys
import random
import numpy as np
import torch
from datetime import datetime

from exp.exp_uniocean import Exp_UniOcean

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

parser = argparse.ArgumentParser(description='[UniOcean]')

parser.add_argument('--model', type=str, required=True, default='informer')
parser.add_argument('--data', type=str, required=True, default='ALL2')
parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/')
parser.add_argument('--data_path', type=str, default='ALL2.pkl')    
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='/root/autodl-tmp/checkpoints/')
parser.add_argument('--log_dir', type=str, default='/root/autodl-tmp/logs')
parser.add_argument('--results_dir', type=str, default='/root/autodl-tmp/results')

parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=24)

parser.add_argument('--enc_in', type=int, default=64800)
parser.add_argument('--dec_in', type=int, default=64800)
parser.add_argument('--c_out', type=int, default=64800)
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--s_layers', type=str, default='3,2,1')
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--move_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=3)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--distil', action='store_false', default=True)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--attn', type=str, default='prob')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--mix', action='store_false', default=True)
parser.add_argument('--cols', type=str, nargs='+')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--train_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--lradj', type=str, default='type3')
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--inverse', action='store_true', default=False)
parser.add_argument('--land_mask_path', type=str, default='')
parser.add_argument('--scale_mask_mode', type=str, default='soft', choices=['soft', 'hard', 'off'])

parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--use_multi_gpu', action='store_true', default=True)
parser.add_argument('--devices', type=str, default='0')
parser.add_argument('--get_prediction', action='store_true', default=False)

parser.add_argument('--multi_task', type=bool, default=False)
parser.add_argument('--use_multi_scale', action='store_true')
parser.add_argument('--scales', default=[8,4,2,1], nargs='+', type=int)
parser.add_argument('--scale_factor', type=int, default=2)

parser.add_argument('--conv_dff', type=int, default=32)
parser.add_argument('--patembed', action='store_true', default=False)

parser.add_argument('--version', type=str, default='Wavelets')
parser.add_argument('--mode_select', type=str, default='low')
parser.add_argument('--modes', type=int, default=64)
parser.add_argument('--L', type=int, default=3)
parser.add_argument('--base', type=str, default='legendre')
parser.add_argument('--cross_activation', type=str, default='tanh')

parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--hidden_dim', default=[16,8,1])
parser.add_argument('--kernel_size', type=tuple, default=(3,3))
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--batch_first', type=bool, default=True)
parser.add_argument('--bias', type=bool, default=True)
parser.add_argument('--return_all_layers', type=bool, default=False)
parser.add_argument('--seed', type=int, default=-1, help='-1 means auto-generate a random base seed')
args = parser.parse_args()

log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)

original_stdout = sys.stdout
original_stderr = sys.stderr
log_file = None
log_file_path = None

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    
def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
data_parser = {
    'SST5':{'data':'SST5.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'SST6':{'data':'SST6.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'SST7':{'data':'SST7.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'SAL5':{'data':'SAL5.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'SAL6':{'data':'SAL6.pkl','T':'target','M':[10000,10000,10000],'S':[1,1,1],'MS':[10000,10000,1]},
    'SAL8':{'data':'SAL8.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OHC1':{'data':'OHC1.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OHC5':{'data':'OHC5.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'ICEC1':{'data':'ICEC1.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OC1':{'data':'OC1.pkl','T':'target','M':[64440,64440,64440],'S':[1,1,1],'MS':[64440,64440,1]},
    'OISSS1':{'data':'OISSS1.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OISSS3':{'data':'OISSS3.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OISST1':{'data':'OISST1.pkl','T':'target','M':[1036800,1036800,1036800],'S':[1,1,1],'MS':[1036800,1036800,1]},
    'OISST2':{'data':'OISST2.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OISST3':{'data':'OISST3.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OISST4':{'data':'OISST4.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'OISST5':{'data':'OISST5.pkl','T':'target','M':[64800,64800,64800],'S':[1,1,1],'MS':[64800,64800,1]},
    'ALL1':{'data':'ALL1.pkl','T':'target','M':[153,153,153],'S':[1,1,1],'MS':[153,153,1], 'root':'/root/autodl-tmp/data/upsampled/1-4/area1'},
    'ALL2':{'data':'ALL2.pkl','T':'target','M':[1075,1075,1075],'S':[1,1,1],'MS':[1075,1075,1], 'root':'/root/autodl-tmp/data/upsampled/1-4/area2'},
    'swh_1_4':{'data':'swh.pkl','T':'target','M':[1075,1075,1075],'S':[1,1,1],'MS':[1075,1075,1], 'root':'/root/autodl-tmp/data/upsampled/1-4/area2'},
    'u10_1_4':{'data':'u10.pkl','T':'target','M':[1075,1075,1075],'S':[1,1,1],'MS':[1075,1075,1], 'root':'/root/autodl-tmp/data/upsampled/1-4/area2'},
    'v10_1_4':{'data':'v10.pkl','T':'target','M':[1075,1075,1075],'S':[1,1,1],'MS':[1075,1075,1], 'root':'/root/autodl-tmp/data/upsampled/1-4/area2'},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]
    if 'root' in data_info:
        args.root_path = data_info['root']

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

Exp = Exp_UniOcean

try:
    for ii in range(args.itr):
        if args.seed < 0:
            run_seed = int.from_bytes(os.urandom(4), byteorder='little', signed=False)
        else:
            run_seed = args.seed + ii
        _set_global_seed(run_seed)
        if log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()
        
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dp{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}_cdf{}_sf{}_ms{}_smm{}_pe{}_itr{}'.format(args.model, args.data, args.features, 
                    args.seq_len, args.label_len, args.pred_len, args.dropout,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                    args.embed, args.distil, args.mix, args.des,args.multi_task ,args.conv_dff, args.scale_factor, args.scales, args.scale_mask_mode, args.patembed, ii)
        
        log_file_path = os.path.join(log_dir, f'{setting}.log')
        log_file = open(log_file_path, 'a', encoding='utf-8')
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        
        print(f'Log file: {log_file_path}')
        if args.seed < 0:
            print(f'Random seed (itr={ii}, random): {run_seed}')
        else:
            print(f'Random seed (base={args.seed}, itr={ii}): {run_seed}')
        print('=' * 80)
        print('Args in experiment:')
        print(args)
        print('=' * 80)

        exp = Exp(args)
        
        if args.get_prediction:
            exp.get_prediction(setting)
        else:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

        torch.cuda.empty_cache()
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    if log_file is not None:
        log_file.close()
        if log_file_path is not None:
            print(f'\nLog saved to: {log_file_path}')
