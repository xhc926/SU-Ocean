import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


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