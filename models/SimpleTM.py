import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_Encoder import Encoder, EncoderLayer
from layers.SWTAttention_Family import GeomAttentionLayer, GeomAttention
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        un = getattr(configs, 'simpletm_use_norm', getattr(configs, 'use_norm', True))
        self.use_norm = bool(un) if not isinstance(un, bool) else un
        self.geomattn_dropout = configs.geomattn_dropout
        self.alpha = configs.alpha
        self.kernel_size = getattr(configs, 'simpletm_kernel_size', None)
        requires_grad = bool(getattr(configs, 'requires_grad', True))

        enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, 
                                               configs.embed, configs.freq, configs.dropout)
        self.enc_embedding = enc_embedding

        encoder = Encoder(
            [  
                EncoderLayer(
                    GeomAttentionLayer(
                        GeomAttention(
                            False, configs.factor, attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention, alpha=self.alpha
                        ),
                        configs.d_model, 
                        requires_grad=requires_grad, 
                        wv=configs.wv, 
                        m=configs.m, 
                        d_channel=configs.dec_in, 
                        kernel_size=self.kernel_size, 
                        geomattn_dropout=self.geomattn_dropout
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers) 
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder = encoder

        projector = nn.Linear(configs.d_model, self.pred_len, bias=True)
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
