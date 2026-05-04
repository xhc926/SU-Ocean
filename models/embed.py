import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TimePositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, scale=None):
        return self.pe[:, :x.size(1)]


PositionalEmbedding = TimePositionalEmbedding
PositionalEmbedding_new = TimePositionalEmbedding

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class TokenEmbedding_new(nn.Module):
    def __init__(self, c_in, P, d_model):
        super(TokenEmbedding_new, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.P=P
        self.tokenConv1 = nn.Conv1d(in_channels=P*P, out_channels=8, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        self.tokenConv2 = nn.Conv1d(in_channels=c_in//(P*P)*8, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        P=self.P
        B, C, _= x.shape
        x = x.reshape(B,C,-1,360)
        B, C, H, W = x.shape
        x = x.reshape(B,C,H//P,P,W//P,P).permute(0, 1, 2, 4, 3, 5).reshape(B,C,-1, P*P)
        _, _, N, _ =x.shape
        x = x.permute(0, 2, 1, 3).reshape(B*N, C, P*P)
        x = self.tokenConv1(x.permute(0, 2, 1)).transpose(1,2)
        x = x.reshape(B, N, C,-1).permute(0, 2, 1, 3).reshape(B, C, -1)
        x = self.tokenConv2(x.permute(0, 2, 1)).transpose(1,2)
        return x

class SpaceEmbedding(nn.Module):
    def __init__(self, c_in, d_model, land_mask_path=''):
        super(SpaceEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.land_mask_1d = None
        self.land_mask_2d_9271 = None
        self.land_mask_2d_1075 = None
        self.land_mask_2d_153 = None
        self.land_mask_2d_637 = None
        if land_mask_path and os.path.exists(land_mask_path):
            mask_df = pd.read_pickle(land_mask_path)
            mask_np = mask_df.values.astype(np.float32).reshape(-1)
            self.land_mask_1d = torch.from_numpy(mask_np)
            if mask_np.shape[0] == 9271:
                self.land_mask_2d_9271 = torch.from_numpy(mask_np.reshape(73, 127))
            elif mask_np.shape[0] == 1075:
                self.land_mask_2d_1075 = torch.from_numpy(mask_np.reshape(25, 43))
            elif mask_np.shape[0] == 153:
                self.land_mask_2d_153 = torch.from_numpy(mask_np.reshape(17, 9))
            elif mask_np.shape[0] == 637:
                self.land_mask_2d_637 = torch.from_numpy(mask_np.reshape(49, 13))

        self.SpaceConv = nn.Conv2d(in_channels=c_in, out_channels=c_in, 
                                    kernel_size=3, padding=padding, padding_mode='zeros')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        B, C, D = x.shape
        mask_2d = None
        if D == 64800:
            x = x.reshape(B, C, 180, 360)
            if self.land_mask_1d is not None and self.land_mask_1d.numel() == D:
                mask_2d = self.land_mask_1d.reshape(180, 360).to(x.device)
        elif D == 9271:
            x = x.reshape(B, C, 73, 127)
            if self.land_mask_2d_9271 is not None:
                mask_2d = self.land_mask_2d_9271.to(x.device)
            elif self.land_mask_1d is not None and self.land_mask_1d.numel() == D:
                mask_2d = self.land_mask_1d.reshape(73, 127).to(x.device)
        elif D == 1075:
            x = x.reshape(B, C, 25, 43)
            if self.land_mask_2d_1075 is not None:
                mask_2d = self.land_mask_2d_1075.to(x.device)
            elif self.land_mask_1d is not None and self.land_mask_1d.numel() == D:
                mask_2d = self.land_mask_1d.reshape(25, 43).to(x.device)
        elif D == 153:
            x = x.reshape(B, C, 17, 9)
            if self.land_mask_2d_153 is not None:
                mask_2d = self.land_mask_2d_153.to(x.device)
            elif self.land_mask_1d is not None and self.land_mask_1d.numel() == D:
                mask_2d = self.land_mask_1d.reshape(17, 9).to(x.device)
        elif D == 637:
            x = x.reshape(B, C, 49, 13)
            if self.land_mask_2d_637 is not None:
                mask_2d = self.land_mask_2d_637.to(x.device)
            elif self.land_mask_1d is not None and self.land_mask_1d.numel() == D:
                mask_2d = self.land_mask_1d.reshape(49, 13).to(x.device)
        elif D == 740:
            x = x.reshape(B, C, 20, 37)
        elif D == 204:
            x = x.reshape(B, C, 12, 17)
        else:
            raise ValueError(
                "SpaceEmbedding: unsupported spatial dim {}, supported: 64800/9271/6411/1075/637/740/204".format(D)
            )

        if mask_2d is not None:
            mask_bc = mask_2d.unsqueeze(0).unsqueeze(0)
            x = x * mask_bc
        x = self.SpaceConv(x)
        if mask_2d is not None:
            x = x * mask_bc
        x = x.reshape(B, C, D)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13
        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class TimeFeatureEmbedding_new(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding_new, self).__init__()
        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp+1, d_model, bias=False)
        self.concats = dict()

    def forward(self, x, scale=1):
        if (scale, x.shape[0], x.shape[1]) not in self.concats:
            concat_tensor = torch.tensor([[[1 / scale - 0.5]]], device=x.device).repeat(x.shape[0], x.shape[1], 1)
            self.concats[(scale, x.shape[0], x.shape[1])] = concat_tensor
        else:
            concat_tensor = self.concats[(scale, x.shape[0], x.shape[1])]
        x = torch.cat((x, concat_tensor), 2)
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, channel_or_embed='fixed', embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        if isinstance(channel_or_embed, str):
            embed_type, freq, dropout = channel_or_embed, embed_type, freq
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.time_position_embedding = TimePositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.time_position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_scale(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, is_decoder=False):
        super(DataEmbedding_scale, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.time_position_embedding = TimePositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.is_decoder = is_decoder

    def forward(self, x, x_mark, scale, first_scale, label_len):
        x = self.value_embedding(x) + self.time_position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_Spacialpatch(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, is_decoder=False):
        super(DataEmbedding_Spacialpatch, self).__init__()
        self.value_embedding = TokenEmbedding_new(c_in=c_in, P=6, d_model=d_model)
        self.time_position_embedding = TimePositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.is_decoder = is_decoder

    def forward(self, x, x_mark, scale, first_scale, label_len):
        x = self.value_embedding(x) + self.time_position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_Spacialpatch_(nn.Module):
    def __init__(self, c_in, d_model, channel_or_embed='fixed', embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_Spacialpatch_, self).__init__()
        if isinstance(channel_or_embed, str):
            embed_type, freq, dropout = channel_or_embed, embed_type, freq
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.patch_embedding = TokenEmbedding_new(c_in=c_in, P=6, d_model=d_model)
        self.time_position_embedding = TimePositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.patch_embedding(x) + self.time_position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.time_position_embedding = TimePositionalEmbedding(d_model=1, max_len=max(5000, c_in + 1))
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        time_stub = x.permute(0, 2, 1)[..., :1]
        time_pe = self.time_position_embedding(time_stub).permute(0, 2, 1)
        x = x + time_pe
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x_mark_transposed = x_mark.permute(0, 2, 1)
            x_combined = torch.cat([x, x_mark_transposed], dim=1)
            x = self.value_embedding(x_combined)
        return self.dropout(x)
