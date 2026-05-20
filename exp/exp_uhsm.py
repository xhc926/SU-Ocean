from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model_base import ConvLSTM,GRU
from models.model_iTransformer import iTransformer, iTransformerUHSM, iTransformerUHSM4
from models.model_iTransformer_ablation import iTransformerUHSMAbl
from models.model_OLinear import OLinear
from models.model_SimpleTM import SimpleTM
from models.model_Dualformer import Dualformer
from models.model_EMAformer import EMAformer, EMAformerUHSM, EMAformerUHSM4
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import inspect

import warnings
warnings.filterwarnings('ignore')

class Exp_UHSM(Exp_Basic):
    def __init__(self, args):
        super(Exp_UHSM, self).__init__(args)
        self.land_mask = None
        if getattr(args, 'land_mask_path', '') and os.path.exists(args.land_mask_path):
            import pandas as pd
            mask_df = pd.read_pickle(args.land_mask_path)
            mask_np = mask_df.values.astype(np.float32).reshape(-1)
            self.land_mask = torch.from_numpy(mask_np).float()
    
    def _build_model(self):
        model_name = self.args.model
        model_dict = {
            'convlstm':ConvLSTM,
            'gru':GRU,
            'itransformer':iTransformer,
            'itransformerUHSM':iTransformerUHSM,
            'itransformerUHSM4':iTransformerUHSM4,
            'itransformerUHSMAbl':iTransformerUHSMAbl,
            'olinear': OLinear,
            'simpletm':SimpleTM,
            'dualformer':Dualformer,
            'emaformer':EMAformer,
            'emaformerUHSM':EMAformerUHSM,
            'emaformerUHSM4':EMAformerUHSM4,
        }

        def _model_extra_init_kwargs(model_cls):
            sig = inspect.signature(model_cls.__init__)
            kw = {}
            if 'land_mask_path' in sig.parameters:
                kw['land_mask_path'] = getattr(self.args, 'land_mask_path', '')
            if 'scale_mask_mode' in sig.parameters:
                kw['scale_mask_mode'] = getattr(self.args, 'scale_mask_mode', 'soft')
            return kw

        model = None
        if model_name == 'convlstm':
            model = model_dict[model_name](
                self.args.input_dim, self.args.hidden_dim, self.args.kernel_size,
                self.args.num_layers, self.args.seq_len, self.args.batch_first,
                self.args.bias, self.args.return_all_layers, self.device
            ).float()
        elif model_name == 'gru':
            model = model_dict[model_name](
                self.args.enc_in, self.args.d_model, self.args.num_layers,
            )
        elif model_name in ['itransformer', 'itransformerUHSM',
                            'itransformerUHSM4', 'itransformerUHSMAbl',
                            'olinear', 'simpletm', 'dualformer',
                            'emaformer', 'emaformerUHSM', 'emaformerUHSM4']:
            model_kwargs = {}
            if model_name == 'simpletm':
                model_kwargs.update({
                    'geomattn_dropout': getattr(self.args, 'geomattn_dropout', 0.5),
                    'requires_grad': getattr(self.args, 'requires_grad', 1),
                    'wv': getattr(self.args, 'wv', 'db1'),
                    'm': getattr(self.args, 'm', 3),
                    'simpletm_kernel_size': getattr(self.args, 'simpletm_kernel_size', None),
                    'alpha': getattr(self.args, 'alpha', 1.0),
                    'simpletm_use_norm': getattr(self.args, 'simpletm_use_norm', 1),
                })
            elif model_name == 'olinear':
                model_kwargs.update({
                    'Q_chan_indep': getattr(self.args, 'Q_chan_indep', False),
                    'q_mat_file': getattr(self.args, 'q_mat_file', ''),
                    'q_out_mat_file': getattr(self.args, 'q_out_mat_file', ''),
                    'Q_MAT_file': getattr(self.args, 'Q_MAT_file', ''),
                    'Q_OUT_MAT_file': getattr(self.args, 'Q_OUT_MAT_file', ''),
                    'temp_patch_len': getattr(self.args, 'temp_patch_len', 1),
                    'temp_stride': getattr(self.args, 'temp_stride', 1),
                    'embed_size': getattr(self.args, 'embed_size', 1),
                    'CKA_flag': getattr(self.args, 'CKA_flag', False),
                    'root_path': getattr(self.args, 'root_path', '.'),
                })
            model_kwargs.update(_model_extra_init_kwargs(model_dict[model_name]))
            model = model_dict[model_name](
                self.args.enc_in, self.args.dec_in, self.args.c_out,
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                self.args.factor, self.args.d_model, self.args.n_heads,
                self.args.e_layers, self.args.d_layers, self.args.d_ff,
                self.args.move_avg, self.args.dropout, self.args.attn,
                self.args.embed, self.args.freq, self.args.activation,
                self.args.output_attention, self.args.distil, self.args.mix,
                self.args.use_multi_scale, self.args.patembed,
                self.args.scales, self.args.scale_factor,
                self.args.version, self.args.mode_select, self.args.modes,
                self.args.L, self.args.base, self.args.cross_activation,
                self.args.conv_dff, self.device,
                **model_kwargs
            ).float()
        else:
            raise ValueError(f'Unknown model: {self.args.model}')

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            model = model.cuda()
            total = sum([param.nelement() for param in model.parameters()])
            print('total param:' ,total)

        return model

    def _unwrap_forecast_out(self, raw):
        if isinstance(raw, tuple):
            return raw[0]
        return raw

    def _is_all_mode(self):
        return self.args.root_path in ['./data/ALL/', '/root/autodl-tmp/ALL/', '/root/autodl-tmp/data/upsampled/1-4/area1',
                                                 '/root/autodl-tmp/data/upsampled/1-4/area2',
                                                 '/root/autodl-tmp/data/upsampled/1-12/area3',
                                                 '/root/autodl-tmp/data/upsampled/1-12/area3/short',
                                                 '/root/autodl-tmp/ms/results/area3'] and \
            self.args.data in ['ALL1', 'ALL2', 'ALL3', 'ALL4', 'ALL5', 'ALL6']

    def _mask_like(self, ref_tensor):
        if self.land_mask is None:
            return None
        mask = self.land_mask.to(ref_tensor.device)
        if mask.dim() != 1:
            mask = mask.reshape(-1)
        if ref_tensor.shape[-1] != mask.numel():
            return None
        view_shape = [1] * (ref_tensor.dim() - 1) + [mask.numel()]
        return mask.view(*view_shape).expand_as(ref_tensor)

    def _masked_or_raw_mse(self, pred, true):
        sq = (pred - true) ** 2
        if self.land_mask is not None:
            mask = self._mask_like(pred)
            if mask is not None:
                return (sq * mask).sum() / (mask.sum() + 1e-8)
        return sq.mean()

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'custom':Dataset_Custom,
            'SST5': Dataset_Custom, 'SST6': Dataset_Custom, 'SST7': Dataset_Custom,
            'SAL5': Dataset_Custom, 'SAL6': Dataset_Custom, 'SAL8': Dataset_Custom,
            'OHC1': Dataset_Custom, 'OHC5': Dataset_Custom,
            'ICEC1': Dataset_Custom, 'OC1': Dataset_Custom,
            'OISSS1': Dataset_Custom, 'OISSS3': Dataset_Custom,
            'OISST1': Dataset_Custom, 'OISST2': Dataset_Custom, 'OISST3': Dataset_Custom, 'OISST4': Dataset_Custom, 'OISST5': Dataset_Custom,
            'ALL1': Dataset_Custom, 'ALL2': Dataset_Custom,
            'ALL3': Dataset_Custom, 'ALL4': Dataset_Custom,
            'msl_1_4': Dataset_Custom, 'swh_1_4': Dataset_Custom,
            'u10_1_4': Dataset_Custom, 'v10_1_4': Dataset_Custom,
            'sal_1_12': Dataset_Custom, 'sst_1_12': Dataset_Custom,
            'uo_1_12': Dataset_Custom, 'vo_1_12': Dataset_Custom,
            'ssh_1_12': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path, data_path=args.data_path,
            flag=flag, size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features, target=args.target, inverse=args.inverse,
            timeenc=timeenc, freq=freq, cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set, batch_size=batch_size, shuffle=shuffle_flag,
            num_workers=args.num_workers, drop_last=True)
        return data_set, data_loader

    def _select_optimizer(self):
        weight_decay = getattr(self.args, 'weight_decay', 0.0)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=weight_decay)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                
                if self._is_all_mode():
                    factor_num = pred.shape[-1]
                    factor_losses = []
                    for k in range(factor_num):
                        loss_k = self._masked_or_raw_mse(pred[:, :, :, k], true[:, :, :, k])
                        factor_losses.append(loss_k.detach().cpu().item())
                    total_loss.append(float(np.mean(factor_losses)))
                else:
                    loss = self._masked_or_raw_mse(pred, true)
                    total_loss.append(loss.detach().cpu().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)

        if self.args.multi_task:
            if not os.path.exists(path):
                os.makedirs(path)
            model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(model_path))

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        patience = self.args.patience
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                if self._is_all_mode():
                    factor_num = pred.shape[-1]
                    factor_losses = [self._masked_or_raw_mse(pred[:, :, :, k], true[:, :, :, k]) for k in range(factor_num)]
                    loss = torch.stack(factor_losses).mean()
                    train_loss.append(loss.item())
                else:
                    loss = self._masked_or_raw_mse(pred, true)
                    train_loss.append(loss.item()) 
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if getattr(self.args, 'grad_clip', 0) > 0:
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if getattr(self.args, 'grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
        
        preds = np.array(preds)
        trues = np.array(trues)
        
        print('test shape:', preds.shape, trues.shape)
        if self._is_all_mode():
            preds = preds.reshape(-1, preds.shape[-3], preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-3], trues.shape[-2], trues.shape[-1])
        else:
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = os.path.join(getattr(self.args, 'results_dir', '/root/autodl-tmp/results'), setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self._is_all_mode():
            metric_pack = []
            for i in range(preds.shape[-1]):
                pred_i = preds[:, :, :, i]
                true_i = trues[:, :, :, i]
                if self.land_mask is not None:
                    mask_np = self.land_mask.numpy()
                    if mask_np.ndim > 1:
                        mask_np = mask_np.reshape(-1)
                    if pred_i.shape[-1] == mask_np.shape[0]:
                        mask_bc = np.broadcast_to(mask_np.reshape(1, 1, -1), pred_i.shape)
                        mse_i = np.sum((pred_i - true_i) ** 2 * mask_bc) / (np.sum(mask_bc) + 1e-8)
                        rmse_i = np.sqrt(mse_i)
                        mae_i = np.sum(np.abs(pred_i - true_i) * mask_bc) / (np.sum(mask_bc) + 1e-8)
                    else:
                        mae_i, mse_i, rmse_i, _, _ = metric(pred_i, true_i)
                else:
                    mae_i, mse_i, rmse_i, _, _ = metric(pred_i, true_i)
                print('factor{} mse:{}, mae:{}, rmse:{}'.format(i + 1, mse_i, mae_i, rmse_i))
                metric_pack.extend([mse_i, mae_i, rmse_i])
            np.save(folder_path+'metrics.npy', np.array(metric_pack))
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
        else:
            if self.land_mask is not None:
                mask_np = self.land_mask.numpy()
                if mask_np.ndim > 1:
                    mask_np = mask_np.reshape(-1)
                if preds.shape[-1] == mask_np.shape[0]:
                    mask_bc = np.broadcast_to(mask_np.reshape(1, 1, -1), preds.shape)
                    mse_masked = np.sum((preds - trues) ** 2 * mask_bc) / (np.sum(mask_bc) + 1e-8)
                    rmse_masked = np.sqrt(mse_masked)
                    mae_masked = np.sum(np.abs(preds - trues) * mask_bc) / (np.sum(mask_bc) + 1e-8)
                    print('mse:{}, mae:{}, rmse:{}'.format(mse_masked, mae_masked, rmse_masked))
                    np.save(folder_path+'metrics.npy', np.array([mae_masked, mse_masked, rmse_masked]))
                else:
                    mae, mse, rmse, mape, mspe = metric(preds, trues)
                    print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
                    np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            else:
                mae, mse, rmse, mape, mspe = metric(preds, trues)
                print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
                np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)

        return
    
    def get_prediction(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        preds = []
        trues = []
        times = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_time) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            times.append(batch_y_time.detach().cpu().numpy())
        preds = np.array(preds)
        trues = np.array(trues)
        times = np.array(times)
        print('shape:', preds.shape, trues.shape, times.shape)
        folder_path = os.path.join(getattr(self.args, 'results_dir', '/root/autodl-tmp/results'), setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path+'pred_.npy', preds)
        np.save(folder_path+'true_.npy', trues)
        np.save(folder_path+'time_.npy', times)
        return 

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        preds = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
        preds = np.array(preds)
        if self._is_all_mode():
            preds = preds.reshape(-1, preds.shape[-3], preds.shape[-2], preds.shape[-1])
        else:
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        folder_path = os.path.join(getattr(self.args, 'results_dir', '/root/autodl-tmp/results'), setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path+'real_prediction.npy', preds)
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if self.args.padding==0:
            if self._is_all_mode():
                factor_num = batch_y.shape[-1]
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-2], factor_num]).float()
            else:
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            if self._is_all_mode():
                factor_num = batch_y.shape[-1]
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-2], factor_num]).float()
            else:
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        
        if self._is_all_mode():
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:,:], dec_inp], dim=1).float().to(self.device)
        else:
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)

        if self._is_all_mode():
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)
            if self.args.get_prediction:
                outputs = dataset_object.inverse_transform(outputs)
                batch_y = dataset_object.inverse_transform(batch_y)

            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:, :].to(self.device)
                
        else:
            if self.args.model=='convlstm' or self.args.model=='gru':
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():outputs = self.model(batch_x)[0]
                else:
                    outputs = self.model(batch_x)[0]
                if self.args.get_prediction:
                    outputs = dataset_object.inverse_transform(outputs)
                    batch_y = dataset_object.inverse_transform(batch_y)
            else:
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._unwrap_forecast_out(
                            self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
                else:
                    outputs = self._unwrap_forecast_out(
                        self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
                if self.args.inverse:
                    outputs = dataset_object.inverse_transform(outputs)
                if self.args.get_prediction:
                    outputs = dataset_object.inverse_transform(outputs)
                    batch_y = dataset_object.inverse_transform(batch_y)

            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
