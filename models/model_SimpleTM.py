import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_Encoder import Encoder, EncoderLayer
from layers.SWTAttention_Family import GeomAttentionLayer, GeomAttention
from layers.Embed import DataEmbedding_inverted


class SimpleTM(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, use_multi_scale=False, patembed=False,
                 scales=[32, 16, 4, 1], scale_factor=4,
                 version='Wavelets', mode_select='low', modes=64, L=3, base='legendre', cross_activation='tanh',
                 conv_dff=32, device=torch.device('cuda:0'),
                 land_mask_path='', scale_mask_mode='soft',
                 geomattn_dropout=0.5, requires_grad=1, wv='db1', m=3,
                 simpletm_kernel_size=None, alpha=1.0, simpletm_use_norm=1):
        super(SimpleTM, self).__init__()
        # Keep compatibility with Exp's optional kwargs injection.
        self.land_mask_path = land_mask_path
        self.scale_mask_mode = scale_mask_mode
        self.seq_len = seq_len
        self.pred_len = out_len
        self.output_attention = output_attention
        un = simpletm_use_norm
        self.use_norm = bool(un) if not isinstance(un, bool) else un
        self.geomattn_dropout = geomattn_dropout
        self.alpha = alpha
        self.kernel_size = simpletm_kernel_size
        requires_grad = bool(requires_grad)

        enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        self.enc_embedding = enc_embedding

        encoder = Encoder(
            [  
                EncoderLayer(
                    GeomAttentionLayer(
                        GeomAttention(
                            False, factor, attention_dropout=dropout, 
                            output_attention=output_attention, alpha=self.alpha
                        ),
                        d_model, 
                        requires_grad=requires_grad, 
                        wv=wv, 
                        m=m, 
                        d_channel=dec_in, 
                        kernel_size=self.kernel_size, 
                        geomattn_dropout=self.geomattn_dropout
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(e_layers) 
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder = encoder

        projector = nn.Linear(d_model, self.pred_len, bias=True)
        self.projector = projector


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # x_enc /= stdev
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        enc_embedding = self.enc_embedding
        encoder = self.encoder
        projector = self.projector
        # Linear projection: [B, L, N] -> [B, N(+time_tokens), d_model]
        enc_out = enc_embedding(x_enc, x_mark_enc) 
        # Keep only spatial tokens (N). DataEmbedding_inverted may append time-feature
        # tokens when x_mark_enc is provided; SWT groups must match spatial channel count.
        enc_out = enc_out[:, :N, :]

        # SimpleTM Layer                B L' N -> B L' N 
        enc_out, attns = encoder(enc_out, attn_mask=None)

        # Output Projection             B L' N -> B H (Horizon) N
        dec_out = projector(enc_out).permute(0, 2, 1)[:, :, :N] 

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out, attns
