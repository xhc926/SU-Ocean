import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd

from models.encoder import Encoder, EncoderLayer
from models.attn import FullAttention, AttentionLayer
from models.embed import SpaceEmbedding, TimePositionalEmbedding


class moving_avg_spatial(nn.Module):
    def __init__(self, spatial_shape=(73, 127)):
        super(moving_avg_spatial, self).__init__()
        self.H, self.W = spatial_shape

    def _pool_one(self, x, scale):
        B, T, D = x.shape
        if D != self.H * self.W:
            raise ValueError(f"moving_avg_spatial expects D={self.H*self.W}, got {D}")
        x2d = x.reshape(B, T, self.H, self.W).reshape(B * T, 1, self.H, self.W)
        x2d = F.avg_pool2d(x2d, kernel_size=scale, stride=scale)
        Hs, Ws = x2d.shape[-2], x2d.shape[-1]
        return x2d.reshape(B, T, Hs * Ws)

    def forward(self, x, scale=1, if_all=False, data_num=3):
        if x is None:
            return None
        if if_all:
            tmp = []
            for i in range(data_num):
                y = x[:, :, :, i]
                y = self._pool_one(y, scale)
                tmp.append(y)
            return torch.stack(tmp, axis=3)
        return self._pool_one(x, scale)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed='fixed', freq='h', dropout=0.1):
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


class iTransformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention=False, distil=True, mix=True, use_multi_scale=False, patembed=False,
                scales=[32,16,4,1], scale_factor=4, 
                version='Wavelets', mode_select='low', modes=64, L=3, base='legendre', cross_activation='tanh',
                conv_dff=32,
                device=torch.device('cuda:0'),
                land_mask_path='',
                scale_mask_mode='soft'):
        super(iTransformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.use_norm = True
        self.enc_in = enc_in
        self.d_model = d_model
        self.dropout = dropout

        self.space_embedding = SpaceEmbedding(seq_len, d_model, land_mask_path=land_mask_path)

        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)

        Attn = FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            conv_layers=None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.projector = nn.Linear(d_model, out_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, N = x_enc.shape
        
        x_enc_spatial = self.space_embedding(x_enc)
        
        if self.use_norm:
            means = x_enc_spatial.mean(1, keepdim=True).detach()
            x_enc_spatial = x_enc_spatial - means
            stdev = torch.sqrt(torch.var(x_enc_spatial, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_spatial = x_enc_spatial / stdev

        x_vars = x_enc_spatial.permute(0, 2, 1)

        enc_out = self.enc_embedding(x_vars, x_mark_enc)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # x_mark 非空时 DataEmbedding_inverted 会在 dim=1 上拼接时间特征，token 数变为 N + d_time；反归一化仅对应空间维 N
        n_spatial = N
        dec_out_vars = self.projector(enc_out[:, :n_spatial, :])
        dec_out = dec_out_vars.permute(0, 2, 1)
        
        if self.use_norm:
            dec_out = dec_out * stdev[:, :1, :].repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, :1, :].repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], None
        else:
            return dec_out[:, -self.pred_len:, :]


class iTransformerUni(nn.Module):
    """
    iTransformer for multi-factor prediction (hardcoded 3 factors).
    Coarse-to-Fine hierarchical residual across spatial scales.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention=False, distil=True, mix=True, use_multi_scale=False, patembed=False,
                scales=[32,16,4,1], scale_factor=4, 
                version='Wavelets', mode_select='low', modes=64, L=3, base='legendre', cross_activation='tanh',
                conv_dff=32,
                device=torch.device('cuda:0'),
                land_mask_path='',
                scale_mask_mode='soft'):
        super(iTransformerUni, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.use_norm = True
        self.dropout = dropout
        self.use_multi_scale = use_multi_scale
        self.scales = scales
        self.d_model = d_model
        self.factor_num = 3
        self.land_mask_flat = None
        self.scale_mask_mode = str(scale_mask_mode).lower()
        if self.scale_mask_mode not in ['soft', 'hard', 'off']:
            raise ValueError("scale_mask_mode must be one of ['soft', 'hard', 'off'], got {}".format(scale_mask_mode))

        self.space_embedding = SpaceEmbedding(seq_len, d_model, land_mask_path=land_mask_path)
        if enc_in == 73 * 127:
            self.spatial_shape = (73, 127)
        elif enc_in == 25 * 43:
            self.spatial_shape = (25, 43)
        elif enc_in == 17 * 9:
            self.spatial_shape = (17, 9)
        elif enc_in == 49 * 13:
            self.spatial_shape = (49, 13)
        else:
            self.spatial_shape = None
        self.spatial_mv = moving_avg_spatial(self.spatial_shape) if self.spatial_shape is not None else None
        
        if self.use_multi_scale and self.spatial_shape is not None:
            self.spatial_scales = [int(s) for s in scales if int(s) >= 1]
        else:
            self.spatial_scales = [1]
        if 1 not in self.spatial_scales:
            self.spatial_scales.append(1)
        self.spatial_scales = sorted(list(set(self.spatial_scales)), reverse=True)
        
        if land_mask_path and os.path.exists(land_mask_path):
            mask_df = pd.read_pickle(land_mask_path)
            mask_np = mask_df.values.astype('float32').reshape(-1)
            if mask_np.shape[0] == enc_in:
                self.land_mask_flat = torch.from_numpy(mask_np)

        # 3-factor embeddings
        self.enc_embedding_one = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        self.enc_embedding_two = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        self.enc_embedding_three = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)

        # Per-scale fusions (3 factors)
        self.scale_fusions_one = nn.ModuleDict()
        self.scale_fusions_two = nn.ModuleDict()
        self.scale_fusions_three = nn.ModuleDict()
        
        for s in self.spatial_scales:
            key = str(s)
            self.scale_fusions_one[key] = nn.Linear(2 * d_model, d_model, bias=True)
            self.scale_fusions_two[key] = nn.Linear(2 * d_model, d_model, bias=True)
            self.scale_fusions_three[key] = nn.Linear(2 * d_model, d_model, bias=True)

        Attn = FullAttention
        self.encoder_one = Encoder([EncoderLayer(AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads, mix=False), d_model, d_ff, dropout=dropout, activation=activation) for l in range(e_layers)], conv_layers=None, norm_layer=torch.nn.LayerNorm(d_model))
        self.encoder_two = Encoder([EncoderLayer(AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads, mix=False), d_model, d_ff, dropout=dropout, activation=activation) for l in range(e_layers)], conv_layers=None, norm_layer=torch.nn.LayerNorm(d_model))
        self.encoder_three = Encoder([EncoderLayer(AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads, mix=False), d_model, d_ff, dropout=dropout, activation=activation) for l in range(e_layers)], conv_layers=None, norm_layer=torch.nn.LayerNorm(d_model))
        
        self.projector_one = nn.Linear(d_model, out_len, bias=True)
        self.projector_two = nn.Linear(d_model, out_len, bias=True)
        self.projector_three = nn.Linear(d_model, out_len, bias=True)
        
        # Cross-factor MLP: 3 * d_model -> d_model
        self.mlp_one = nn.Linear(3 * d_model, d_model, bias=True)
        self.mlp_two = nn.Linear(3 * d_model, d_model, bias=True)
        self.mlp_three = nn.Linear(3 * d_model, d_model, bias=True)

    def _spatial_downsample(self, x, scale):
        if scale == 1 or self.spatial_mv is None:
            return x
        return self.spatial_mv(x, scale=scale, if_all=False)

    def _resize_between_scales(self, x, from_scale, to_scale):
        if self.spatial_shape is None or from_scale == to_scale:
            return x
        B, T, Df = x.shape
        hf = self.spatial_shape[0] // from_scale
        wf = self.spatial_shape[1] // from_scale
        ht = self.spatial_shape[0] // to_scale
        wt = self.spatial_shape[1] // to_scale
        if Df != hf * wf:
            return x
        x2d = x.reshape(B, T, hf, wf).reshape(B * T, 1, hf, wf)
        x2d = F.interpolate(x2d, size=(ht, wt), mode='bilinear', align_corners=False)
        return x2d.reshape(B, T, ht * wt)

    def _scale_mask_flat(self, scale, device, dtype):
        if self.land_mask_flat is None or self.scale_mask_mode == 'off':
            return None
        if self.spatial_shape is None:
            mask_flat = self.land_mask_flat.to(device=device, dtype=dtype)
            if self.scale_mask_mode == 'hard':
                mask_flat = (mask_flat > 0.5).to(dtype=dtype)
            return mask_flat
        mask2d = self.land_mask_flat.reshape(1, 1, self.spatial_shape[0], self.spatial_shape[1]).to(device=device, dtype=dtype)
        if scale > 1:
            mask2d = F.avg_pool2d(mask2d, kernel_size=scale, stride=scale)
        if self.scale_mask_mode == 'hard':
            mask2d = (mask2d > 0.5).to(dtype=dtype)
        return mask2d.reshape(-1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        if x_enc.shape[-1] < 3:
            raise ValueError("iTransformerUni expects 3 factors in x_enc, got {}".format(x_enc.shape[-1]))

        x_factor = [
            self.space_embedding(x_enc[:,:,:,0]),
            self.space_embedding(x_enc[:,:,:,1]),
            self.space_embedding(x_enc[:,:,:,2]),
        ]

        fusions = [self.scale_fusions_one, self.scale_fusions_two, self.scale_fusions_three]
        embeddings = [self.enc_embedding_one, self.enc_embedding_two, self.enc_embedding_three]
        encoders = [self.encoder_one, self.encoder_two, self.encoder_three]
        projectors = [self.projector_one, self.projector_two, self.projector_three]
        mlps = [self.mlp_one, self.mlp_two, self.mlp_three]

        hidden_prev = [None, None, None]
        prev_scale = None
        attns_one = None
        
        final_means = [None, None, None]
        final_stdevs = [None, None, None]

        for scale in self.spatial_scales:
            key = str(scale)
            
            for i in range(3):
                x_scale_enc = self._spatial_downsample(x_factor[i], scale)
                n_spatial_tokens = x_scale_enc.shape[-1]
                
                if self.use_norm:
                    means_i = x_scale_enc.mean(1, keepdim=True).detach()
                    x_scale_enc = x_scale_enc - means_i
                    stdev_i = torch.sqrt(torch.var(x_scale_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
                    x_scale_enc = x_scale_enc / stdev_i
                    if scale == self.spatial_scales[-1]:
                        final_means[i] = means_i
                        final_stdevs[i] = stdev_i

                mask_flat_s = self._scale_mask_flat(scale, x_scale_enc.device, x_scale_enc.dtype)
                if mask_flat_s is not None and mask_flat_s.numel() == x_scale_enc.shape[-1]:
                    x_scale_enc = x_scale_enc * mask_flat_s.view(1, 1, -1)

                x_tokens = x_scale_enc.permute(0, 2, 1)
                enc_embed = embeddings[i](x_tokens, x_mark_enc)

                if hidden_prev[i] is None:
                    x_dec_i = enc_embed
                else:
                    prev_hidden = hidden_prev[i]
                    prev_grid = prev_hidden.permute(0, 2, 1)
                    cur_grid_prior = self._resize_between_scales(prev_grid, from_scale=prev_scale, to_scale=scale)
                    x_dec_spatial = cur_grid_prior.permute(0, 2, 1)
                    if x_dec_spatial.shape[1] != n_spatial_tokens:
                        valid_len = min(x_dec_spatial.shape[1], n_spatial_tokens)
                        x_dec_spatial = x_dec_spatial[:, :valid_len, :]
                        n_spatial_tokens = valid_len
                    if enc_embed.shape[1] > n_spatial_tokens:
                        # Keep time tokens aligned with current scale embedding; only spatial tokens are cross-scale resized.
                        x_dec_i = torch.cat([x_dec_spatial, enc_embed[:, n_spatial_tokens:, :]], dim=1)
                    else:
                        x_dec_i = x_dec_spatial
                    if x_dec_i.shape[1] != enc_embed.shape[1]:
                        valid_len = min(x_dec_i.shape[1], enc_embed.shape[1])
                        x_dec_i = x_dec_i[:, :valid_len, :]
                        enc_embed = enc_embed[:, :valid_len, :]
                        n_spatial_tokens = min(n_spatial_tokens, valid_len)

                fused_input = fusions[i][key](torch.cat([enc_embed, x_dec_i], dim=-1))
                
                if self.training:
                    fused_input = F.dropout(fused_input, p=self.dropout * 0.3, inplace=False)
                
                out_hidden, attns_tmp = encoders[i](fused_input, attn_mask=None)
                if i == 0 and scale == self.spatial_scales[-1]:
                    attns_one = attns_tmp
                
                hidden_prev[i] = out_hidden[:, :n_spatial_tokens, :]
                
            prev_scale = scale

        all_factors = torch.cat(hidden_prev, dim=2)  # [B, D, 3 * d_model]
        fused_list = [mlps[i](all_factors) for i in range(3)]

        dec_out_list = []
        for i in range(3):
            dec_out_scale = projectors[i](fused_list[i]).permute(0, 2, 1)
            
            if self.training:
                dec_out_scale = F.dropout(dec_out_scale, p=self.dropout, inplace=False)
                
            if self.use_norm:
                dec_out_scale = dec_out_scale * final_stdevs[i][:, :1, :].repeat(1, self.pred_len, 1)
                dec_out_scale = dec_out_scale + final_means[i][:, :1, :].repeat(1, self.pred_len, 1)
                
            dec_out_list.append(dec_out_scale)

        dec_out = torch.stack(dec_out_list, dim=3)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :, :], attns_one
        else:
            return dec_out[:, -self.pred_len:, :, :]


class iTransformerUni4(nn.Module):
    """
    iTransformer for multi-factor prediction (hardcoded 4 factors).
    Coarse-to-Fine hierarchical residual across spatial scales.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention=False, distil=True, mix=True, use_multi_scale=False, patembed=False,
                scales=[32, 16, 4, 1], scale_factor=4,
                version='Wavelets', mode_select='low', modes=64, L=3, base='legendre', cross_activation='tanh',
                conv_dff=32,
                device=torch.device('cuda:0'),
                land_mask_path='',
                scale_mask_mode='soft'):
        super(iTransformerUni4, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.use_norm = True
        self.dropout = dropout
        self.use_multi_scale = use_multi_scale
        self.scales = scales
        self.d_model = d_model
        self.factor_num = 4
        self.land_mask_flat = None
        self.scale_mask_mode = str(scale_mask_mode).lower()
        if self.scale_mask_mode not in ['soft', 'hard', 'off']:
            raise ValueError("scale_mask_mode must be one of ['soft', 'hard', 'off'], got {}".format(scale_mask_mode))

        self.space_embedding = SpaceEmbedding(seq_len, d_model, land_mask_path=land_mask_path)
        if enc_in == 73 * 127:
            self.spatial_shape = (73, 127)
        elif enc_in == 25 * 43:
            self.spatial_shape = (25, 43)
        elif enc_in == 17 * 9:
            self.spatial_shape = (17, 9)
        elif enc_in == 49 * 13:
            self.spatial_shape = (49, 13)
        else:
            self.spatial_shape = None
        self.spatial_mv = moving_avg_spatial(self.spatial_shape) if self.spatial_shape is not None else None

        if self.use_multi_scale and self.spatial_shape is not None:
            self.spatial_scales = [int(s) for s in scales if int(s) >= 1]
        else:
            self.spatial_scales = [1]
        if 1 not in self.spatial_scales:
            self.spatial_scales.append(1)
        self.spatial_scales = sorted(list(set(self.spatial_scales)), reverse=True)

        if land_mask_path and os.path.exists(land_mask_path):
            mask_df = pd.read_pickle(land_mask_path)
            mask_np = mask_df.values.astype('float32').reshape(-1)
            if mask_np.shape[0] == enc_in:
                self.land_mask_flat = torch.from_numpy(mask_np)

        self.enc_embeddings = nn.ModuleList(
            [DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout) for _ in range(self.factor_num)]
        )

        self.scale_fusions = nn.ModuleList([nn.ModuleDict() for _ in range(self.factor_num)])
        for s in self.spatial_scales:
            key = str(s)
            for i in range(self.factor_num):
                self.scale_fusions[i][key] = nn.Linear(2 * d_model, d_model, bias=True)

        Attn = FullAttention
        self.encoders = nn.ModuleList([
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                conv_layers=None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for _ in range(self.factor_num)
        ])

        self.projectors = nn.ModuleList(
            [nn.Linear(d_model, out_len, bias=True) for _ in range(self.factor_num)]
        )

        # Cross-factor MLP heads (keep explicit naming style with iTransformerUni).
        self.mlp_one = nn.Linear(self.factor_num * d_model, d_model, bias=True)
        self.mlp_two = nn.Linear(self.factor_num * d_model, d_model, bias=True)
        self.mlp_three = nn.Linear(self.factor_num * d_model, d_model, bias=True)
        self.mlp_four = nn.Linear(self.factor_num * d_model, d_model, bias=True)

    def _spatial_downsample(self, x, scale):
        if scale == 1 or self.spatial_mv is None:
            return x
        return self.spatial_mv(x, scale=scale, if_all=False)

    def _resize_between_scales(self, x, from_scale, to_scale):
        if self.spatial_shape is None or from_scale == to_scale:
            return x
        B, T, Df = x.shape
        hf = self.spatial_shape[0] // from_scale
        wf = self.spatial_shape[1] // from_scale
        ht = self.spatial_shape[0] // to_scale
        wt = self.spatial_shape[1] // to_scale
        if Df != hf * wf:
            return x
        x2d = x.reshape(B, T, hf, wf).reshape(B * T, 1, hf, wf)
        x2d = F.interpolate(x2d, size=(ht, wt), mode='bilinear', align_corners=False)
        return x2d.reshape(B, T, ht * wt)

    def _scale_mask_flat(self, scale, device, dtype):
        if self.land_mask_flat is None or self.scale_mask_mode == 'off':
            return None
        if self.spatial_shape is None:
            mask_flat = self.land_mask_flat.to(device=device, dtype=dtype)
            if self.scale_mask_mode == 'hard':
                mask_flat = (mask_flat > 0.5).to(dtype=dtype)
            return mask_flat
        mask2d = self.land_mask_flat.reshape(1, 1, self.spatial_shape[0], self.spatial_shape[1]).to(device=device, dtype=dtype)
        if scale > 1:
            mask2d = F.avg_pool2d(mask2d, kernel_size=scale, stride=scale)
        if self.scale_mask_mode == 'hard':
            mask2d = (mask2d > 0.5).to(dtype=dtype)
        return mask2d.reshape(-1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        if x_enc.shape[-1] < self.factor_num:
            raise ValueError("iTransformerUni4 expects 4 factors in x_enc, got {}".format(x_enc.shape[-1]))

        x_factor = [self.space_embedding(x_enc[:, :, :, i]) for i in range(self.factor_num)]

        hidden_prev = [None for _ in range(self.factor_num)]
        prev_scale = None
        attns_one = None

        final_means = [None for _ in range(self.factor_num)]
        final_stdevs = [None for _ in range(self.factor_num)]

        for scale in self.spatial_scales:
            key = str(scale)

            for i in range(self.factor_num):
                x_scale_enc = self._spatial_downsample(x_factor[i], scale)
                n_spatial_tokens = x_scale_enc.shape[-1]

                if self.use_norm:
                    means_i = x_scale_enc.mean(1, keepdim=True).detach()
                    x_scale_enc = x_scale_enc - means_i
                    stdev_i = torch.sqrt(torch.var(x_scale_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
                    x_scale_enc = x_scale_enc / stdev_i
                    if scale == self.spatial_scales[-1]:
                        final_means[i] = means_i
                        final_stdevs[i] = stdev_i

                mask_flat_s = self._scale_mask_flat(scale, x_scale_enc.device, x_scale_enc.dtype)
                if mask_flat_s is not None and mask_flat_s.numel() == x_scale_enc.shape[-1]:
                    x_scale_enc = x_scale_enc * mask_flat_s.view(1, 1, -1)

                x_tokens = x_scale_enc.permute(0, 2, 1)
                enc_embed = self.enc_embeddings[i](x_tokens, x_mark_enc)

                if hidden_prev[i] is None:
                    x_dec_i = enc_embed
                else:
                    prev_hidden = hidden_prev[i]
                    prev_grid = prev_hidden.permute(0, 2, 1)
                    cur_grid_prior = self._resize_between_scales(prev_grid, from_scale=prev_scale, to_scale=scale)
                    x_dec_spatial = cur_grid_prior.permute(0, 2, 1)
                    if x_dec_spatial.shape[1] != n_spatial_tokens:
                        valid_len = min(x_dec_spatial.shape[1], n_spatial_tokens)
                        x_dec_spatial = x_dec_spatial[:, :valid_len, :]
                        n_spatial_tokens = valid_len
                    if enc_embed.shape[1] > n_spatial_tokens:
                        x_dec_i = torch.cat([x_dec_spatial, enc_embed[:, n_spatial_tokens:, :]], dim=1)
                    else:
                        x_dec_i = x_dec_spatial
                    if x_dec_i.shape[1] != enc_embed.shape[1]:
                        valid_len = min(x_dec_i.shape[1], enc_embed.shape[1])
                        x_dec_i = x_dec_i[:, :valid_len, :]
                        enc_embed = enc_embed[:, :valid_len, :]
                        n_spatial_tokens = min(n_spatial_tokens, valid_len)

                fused_input = self.scale_fusions[i][key](torch.cat([enc_embed, x_dec_i], dim=-1))
                if self.training:
                    fused_input = F.dropout(fused_input, p=self.dropout * 0.3, inplace=False)

                out_hidden, attns_tmp = self.encoders[i](fused_input, attn_mask=None)
                if i == 0 and scale == self.spatial_scales[-1]:
                    attns_one = attns_tmp

                hidden_prev[i] = out_hidden[:, :n_spatial_tokens, :]
            prev_scale = scale

        all_factors = torch.cat(hidden_prev, dim=2)
        mlps = [self.mlp_one, self.mlp_two, self.mlp_three, self.mlp_four]
        fused_list = [mlps[i](all_factors) for i in range(self.factor_num)]

        dec_out_list = []
        for i in range(self.factor_num):
            dec_out_scale = self.projectors[i](fused_list[i]).permute(0, 2, 1)

            if self.training:
                dec_out_scale = F.dropout(dec_out_scale, p=self.dropout, inplace=False)

            if self.use_norm:
                dec_out_scale = dec_out_scale * final_stdevs[i][:, :1, :].repeat(1, self.pred_len, 1)
                dec_out_scale = dec_out_scale + final_means[i][:, :1, :].repeat(1, self.pred_len, 1)
            dec_out_list.append(dec_out_scale)

        dec_out = torch.stack(dec_out_list, dim=3)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :, :], attns_one
        else:
            return dec_out[:, -self.pred_len:, :, :]
