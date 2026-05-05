import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class moving_avg_spatial(nn.Module):
    def __init__(self, spatial_shape=(73, 127)):
        super(moving_avg_spatial, self).__init__()
        self.H, self.W = spatial_shape

    def _pool_one(self, x, scale):
        bsz, steps, dim = x.shape
        if dim != self.H * self.W:
            raise ValueError(
                "moving_avg_spatial expects D={}, got {}".format(self.H * self.W, dim)
            )
        x2d = x.reshape(bsz, steps, self.H, self.W).reshape(bsz * steps, 1, self.H, self.W)
        x2d = F.avg_pool2d(x2d, kernel_size=scale, stride=scale)
        hs, ws = x2d.shape[-2], x2d.shape[-1]
        return x2d.reshape(bsz, steps, hs * ws)

    def forward(self, x, scale=1):
        if x is None:
            return None
        return self._pool_one(x, scale)


def _infer_spatial_shape(enc_in):
    if enc_in == 73 * 127:
        return (73, 127)
    if enc_in == 25 * 43:
        return (25, 43)
    if enc_in == 17 * 9:
        return (17, 9)
    if enc_in == 49 * 13:
        return (49, 13)
    return None


def _build_output_projector(d_model, pred_len, output_proj_dropout):
    return nn.Sequential(
        nn.Linear(d_model, d_model * 2),
        nn.GELU(),
        nn.Dropout(output_proj_dropout),
        nn.Linear(d_model * 2, d_model * 4),
        nn.GELU(),
        nn.Dropout(output_proj_dropout),
        nn.Linear(d_model * 4, pred_len),
    )

class EMAformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, use_multi_scale=False, patembed=False,
                 scales=[32, 16, 4, 1], scale_factor=4,
                 version='Wavelets', mode_select='low', modes=64, L=3, base='legendre', cross_activation='tanh',
                 conv_dff=32, device=torch.device('cuda:0'),
                 land_mask_path='', scale_mask_mode='soft',
                 cycle=24, output_proj_dropout=0.1, use_norm=True):
        super(EMAformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = out_len
        self.output_attention = output_attention
        self.use_norm = True
        self.d_model = d_model
        self.cycle_len = int(cycle)
        self.enc_in = enc_in

        # Keep compatibility with Exp optional kwargs injection.
        self.land_mask_path = land_mask_path
        self.scale_mask_mode = scale_mask_mode

        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(output_proj_dropout),
            nn.Linear(d_model * 4, out_len),
        )

        self.channel_embedding = nn.Parameter(torch.zeros(enc_in, d_model))
        self.phase_embedding = nn.Embedding(self.cycle_len, d_model)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        self.joint_embedding = nn.Embedding(self.cycle_len, self.enc_in * self.d_model)
        nn.init.xavier_normal_(self.joint_embedding.weight)
        nn.init.xavier_normal_(self.channel_embedding)

    def _resolve_phase(self, x_enc, cycle_index=None):
        bsz = x_enc.shape[0]
        if cycle_index is None:
            return torch.zeros(bsz, dtype=torch.long, device=x_enc.device)
        if not torch.is_tensor(cycle_index):
            cycle_index = torch.as_tensor(cycle_index, dtype=torch.long, device=x_enc.device)
        cycle_index = cycle_index.to(x_enc.device).long().view(-1)
        if cycle_index.numel() == 1 and bsz > 1:
            cycle_index = cycle_index.repeat(bsz)
        elif cycle_index.numel() != bsz:
            cycle_index = cycle_index[:1].repeat(bsz)
        return torch.remainder(cycle_index, self.cycle_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        B, _, N = x_enc.shape
        phase = self._resolve_phase(x_enc, cycle_index)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        channel_emb = self.channel_embedding.expand(B, N, -1)
        phase_emb = self.phase_embedding(phase.view(-1, 1).expand(B, N))
        joint_emb = self.joint_embedding(phase).reshape(B, self.enc_in, self.d_model)
        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_origin = enc_out

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out + enc_origin).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index=None, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, cycle_index)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

class _EMAformerUniBase(nn.Module):
    def __init__(self, factor_num, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, use_multi_scale=False, patembed=False,
                 scales=[32, 16, 4, 1], scale_factor=4,
                 version='Wavelets', mode_select='low', modes=64, L=3, base='legendre', cross_activation='tanh',
                 conv_dff=32, device=torch.device('cuda:0'),
                 land_mask_path='', scale_mask_mode='soft',
                 cycle=24, output_proj_dropout=0.1, use_norm=True):
        super(_EMAformerUniBase, self).__init__()
        self.factor_num = factor_num
        self.pred_len = out_len
        self.output_attention = output_attention
        self.seq_len = seq_len
        self.label_len = label_len
        self.use_norm = True
        self.dropout = dropout
        self.use_multi_scale = use_multi_scale
        self.scales = scales
        self.d_model = d_model
        self.enc_in = enc_in
        self.cycle_len = int(cycle)
        self.land_mask_flat = None
        self.scale_mask_mode = str(scale_mask_mode).lower()
        if self.scale_mask_mode not in ['soft', 'hard', 'off']:
            raise ValueError(
                "scale_mask_mode must be one of ['soft', 'hard', 'off'], got {}".format(
                    scale_mask_mode
                )
            )

        self.spatial_shape = _infer_spatial_shape(enc_in)
        self.spatial_mv = (
            moving_avg_spatial(self.spatial_shape) if self.spatial_shape is not None else None
        )

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

        self.phase_embedding = nn.Embedding(self.cycle_len, d_model)
        nn.init.xavier_normal_(self.phase_embedding.weight)

        self.channel_embeddings = nn.ParameterList(
            [nn.Parameter(torch.zeros(enc_in, d_model)) for _ in range(self.factor_num)]
        )
        for p in self.channel_embeddings:
            nn.init.xavier_normal_(p)

        self.joint_embeddings = nn.ModuleList([nn.ModuleDict() for _ in range(self.factor_num)])
        for i in range(self.factor_num):
            for s in self.spatial_scales:
                key = str(s)
                if self.spatial_shape is not None:
                    token_num = (self.spatial_shape[0] // s) * (self.spatial_shape[1] // s)
                else:
                    token_num = enc_in
                self.joint_embeddings[i][key] = nn.Embedding(
                    self.cycle_len, token_num * d_model
                )
                nn.init.xavier_normal_(self.joint_embeddings[i][key].weight)

        self.enc_embeddings = nn.ModuleList(
            [DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout) for _ in range(self.factor_num)]
        )

        self.scale_fusions = nn.ModuleList([nn.ModuleDict() for _ in range(self.factor_num)])
        for s in self.spatial_scales:
            key = str(s)
            for i in range(self.factor_num):
                self.scale_fusions[i][key] = nn.Linear(2 * d_model, d_model, bias=True)

        attn_layer = FullAttention
        self.encoders = nn.ModuleList(
            [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                attn_layer(
                                    False,
                                    factor,
                                    attention_dropout=dropout,
                                    output_attention=output_attention,
                                ),
                                d_model,
                                n_heads,
                            ),
                            d_model,
                            d_ff,
                            dropout=dropout,
                            activation=activation,
                        )
                        for _ in range(e_layers)
                    ],
                    conv_layers=None,
                    norm_layer=torch.nn.LayerNorm(d_model),
                )
                for _ in range(self.factor_num)
            ]
        )

        self.projectors = nn.ModuleList(
            [_build_output_projector(d_model, out_len, output_proj_dropout) for _ in range(self.factor_num)]
        )
        self.mlps = nn.ModuleList(
            [nn.Linear(self.factor_num * d_model, d_model, bias=True) for _ in range(self.factor_num)]
        )

    def _resolve_phase(self, x_enc, cycle_index=None):
        bsz = x_enc.shape[0]
        if cycle_index is None:
            return torch.zeros(bsz, dtype=torch.long, device=x_enc.device)
        if not torch.is_tensor(cycle_index):
            cycle_index = torch.as_tensor(cycle_index, dtype=torch.long, device=x_enc.device)
        cycle_index = cycle_index.to(x_enc.device).long().view(-1)
        if cycle_index.numel() == 1 and bsz > 1:
            cycle_index = cycle_index.repeat(bsz)
        elif cycle_index.numel() != bsz:
            cycle_index = cycle_index[:1].repeat(bsz)
        return torch.remainder(cycle_index, self.cycle_len)

    def _spatial_downsample(self, x, scale):
        if scale == 1 or self.spatial_mv is None:
            return x
        return self.spatial_mv(x, scale=scale)

    def _resize_between_scales(self, x, from_scale, to_scale):
        if self.spatial_shape is None or from_scale == to_scale:
            return x
        bsz, steps, dim_from = x.shape
        h_from = self.spatial_shape[0] // from_scale
        w_from = self.spatial_shape[1] // from_scale
        h_to = self.spatial_shape[0] // to_scale
        w_to = self.spatial_shape[1] // to_scale
        if dim_from != h_from * w_from:
            return x
        x2d = x.reshape(bsz, steps, h_from, w_from).reshape(bsz * steps, 1, h_from, w_from)
        x2d = F.interpolate(x2d, size=(h_to, w_to), mode='bilinear', align_corners=False)
        return x2d.reshape(bsz, steps, h_to * w_to)

    def _scale_mask_flat(self, scale, device, dtype):
        if self.land_mask_flat is None or self.scale_mask_mode == 'off':
            return None
        if self.spatial_shape is None:
            mask_flat = self.land_mask_flat.to(device=device, dtype=dtype)
            if self.scale_mask_mode == 'hard':
                mask_flat = (mask_flat > 0.5).to(dtype=dtype)
            return mask_flat
        mask2d = self.land_mask_flat.reshape(
            1, 1, self.spatial_shape[0], self.spatial_shape[1]
        ).to(device=device, dtype=dtype)
        if scale > 1:
            mask2d = F.avg_pool2d(mask2d, kernel_size=scale, stride=scale)
        if self.scale_mask_mode == 'hard':
            mask2d = (mask2d > 0.5).to(dtype=dtype)
        return mask2d.reshape(-1)

    def _add_ema_embeddings(self, enc_embed, factor_idx, n_spatial_tokens, phase, scale_key):
        bsz = enc_embed.shape[0]
        channel_src = self.channel_embeddings[factor_idx]
        if n_spatial_tokens <= channel_src.shape[0]:
            channel_base = channel_src[:n_spatial_tokens, :]
        else:
            pad_num = n_spatial_tokens - channel_src.shape[0]
            channel_base = torch.cat(
                [channel_src, channel_src[-1:, :].expand(pad_num, -1)], dim=0
            )
        channel_emb = channel_base.unsqueeze(0).expand(bsz, -1, -1)
        phase_emb = self.phase_embedding(phase.view(-1, 1).expand(bsz, n_spatial_tokens))
        joint_emb = self.joint_embeddings[factor_idx][scale_key](phase).reshape(
            bsz, n_spatial_tokens, self.d_model
        )

        spatial_tokens = enc_embed[:, :n_spatial_tokens, :] + channel_emb + phase_emb + joint_emb
        if enc_embed.shape[1] > n_spatial_tokens:
            return torch.cat([spatial_tokens, enc_embed[:, n_spatial_tokens:, :]], dim=1)
        return spatial_tokens

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                cycle_index=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if x_enc.shape[-1] < self.factor_num:
            raise ValueError(
                "EMAformerUni expects {} factors in x_enc, got {}".format(
                    self.factor_num, x_enc.shape[-1]
                )
            )

        phase = self._resolve_phase(x_enc[:, :, :, 0], cycle_index)
        x_factor = [x_enc[:, :, :, i] for i in range(self.factor_num)]

        hidden_prev = [None for _ in range(self.factor_num)]
        prev_scale = None
        attns_first = None

        final_means = [None for _ in range(self.factor_num)]
        final_stdevs = [None for _ in range(self.factor_num)]

        for scale in self.spatial_scales:
            key = str(scale)
            for i in range(self.factor_num):
                x_scale_enc = self._spatial_downsample(x_factor[i], scale)
                n_spatial_tokens = x_scale_enc.shape[-1]

                means_i = x_scale_enc.mean(1, keepdim=True).detach()
                stdev_i = torch.sqrt(
                    torch.var(x_scale_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
                )
                x_scale_enc = (x_scale_enc - means_i) / stdev_i
                if scale == self.spatial_scales[-1]:
                    final_means[i] = means_i
                    final_stdevs[i] = stdev_i

                mask_flat_s = self._scale_mask_flat(scale, x_scale_enc.device, x_scale_enc.dtype)
                if mask_flat_s is not None and mask_flat_s.numel() == x_scale_enc.shape[-1]:
                    x_scale_enc = x_scale_enc * mask_flat_s.view(1, 1, -1)

                enc_embed = self.enc_embeddings[i](x_scale_enc, x_mark_enc)
                enc_embed = self._add_ema_embeddings(enc_embed, i, n_spatial_tokens, phase, key)

                if hidden_prev[i] is None:
                    x_dec_i = enc_embed
                else:
                    prev_hidden = hidden_prev[i]
                    prev_grid = prev_hidden.permute(0, 2, 1)
                    cur_grid_prior = self._resize_between_scales(
                        prev_grid, from_scale=prev_scale, to_scale=scale
                    )
                    x_dec_spatial = cur_grid_prior.permute(0, 2, 1)

                    if x_dec_spatial.shape[1] != n_spatial_tokens:
                        valid_len = min(x_dec_spatial.shape[1], n_spatial_tokens)
                        x_dec_spatial = x_dec_spatial[:, :valid_len, :]
                        n_spatial_tokens = valid_len

                    if enc_embed.shape[1] > n_spatial_tokens:
                        x_dec_i = torch.cat(
                            [x_dec_spatial, enc_embed[:, n_spatial_tokens:, :]], dim=1
                        )
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
                    attns_first = attns_tmp
                hidden_prev[i] = out_hidden[:, :n_spatial_tokens, :]

            prev_scale = scale

        all_factors = torch.cat(hidden_prev, dim=2)
        fused_list = [self.mlps[i](all_factors) for i in range(self.factor_num)]

        dec_out_list = []
        for i in range(self.factor_num):
            dec_out_scale = self.projectors[i](fused_list[i]).permute(0, 2, 1)
            if self.training:
                dec_out_scale = F.dropout(dec_out_scale, p=self.dropout, inplace=False)
            dec_out_scale = dec_out_scale * final_stdevs[i][:, :1, :].repeat(1, self.pred_len, 1)
            dec_out_scale = dec_out_scale + final_means[i][:, :1, :].repeat(1, self.pred_len, 1)
            dec_out_list.append(dec_out_scale)

        dec_out = torch.stack(dec_out_list, dim=3)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :, :], attns_first
        return dec_out[:, -self.pred_len:, :, :]


class EMAformerUni(_EMAformerUniBase):
    """
    EMAformer for multi-factor prediction (hardcoded 3 factors).
    Coarse-to-Fine hierarchical residual across spatial scales + cross-factor fusion.
    """

    def __init__(self, *args, **kwargs):
        super(EMAformerUni, self).__init__(3, *args, **kwargs)


class EMAformerUni4(_EMAformerUniBase):
    """
    EMAformer for multi-factor prediction (hardcoded 4 factors).
    Coarse-to-Fine hierarchical residual across spatial scales + cross-factor fusion.
    """

    def __init__(self, *args, **kwargs):
        super(EMAformerUni4, self).__init__(4, *args, **kwargs)
