import os
import torch
import torch.nn as nn
import numpy as np
from layers.RevIN import RevIN
from layers.OLinear_Transformer_EncDec import Encoder_ori, LinearEncoder


def _resolve_path(root_path: str, p: str) -> str:
    if not p:
        return ''
    if os.path.isfile(p):
        return p
    joined = os.path.join(root_path, p)
    if os.path.isfile(joined):
        return joined
    return p


class OLinear(nn.Module):
    """
    OLinear-style forecaster. Requires offline-generated Q matrices (see data/generate_corrmat.py):
      - q_mat_file: (seq_len, seq_len), temporal orthogonal basis from train-split covariance
      - q_out_mat_file: (pred_len, pred_len)
    Paths may be absolute or relative to configs.root_path.
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, move_avg=25,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, use_multi_scale=False, patembed=False,
                 scales=[32, 16, 4, 1], scale_factor=4,
                 version='Wavelets', mode_select='low', modes=64, L=3, base='legendre', cross_activation='tanh',
                 conv_dff=32, device=torch.device('cuda:0'),
                 land_mask_path='', scale_mask_mode='soft',
                 Q_chan_indep=False, q_mat_file='', q_out_mat_file='',
                 Q_MAT_file='', Q_OUT_MAT_file='',
                 temp_patch_len=1, temp_stride=1, embed_size=1, CKA_flag=False,
                 root_path='.'):
        super(OLinear, self).__init__()
        # Keep compatibility with Exp's optional kwargs injection.
        self.land_mask_path = land_mask_path
        self.scale_mask_mode = scale_mask_mode
        self.pred_len = out_len
        self.enc_in = enc_in  # channels
        self.seq_len = seq_len
        self.hidden_size = self.d_model = d_model  # hidden_size
        self.d_ff = d_ff  # d_ff
        self.Q_chan_indep = bool(Q_chan_indep)

        q_mat_dir = Q_MAT_file if self.Q_chan_indep else q_mat_file
        q_mat_dir = _resolve_path(root_path, q_mat_dir)
        if not os.path.isfile(q_mat_dir):
            raise FileNotFoundError(
                f'OLinear: Q_in not found: {q_mat_dir} (need shape seq_len={self.seq_len}). '
                f'Run: python data/generate_corrmat.py --lags ...,{self.seq_len},... '
                f'or pass --q_mat_file.'
            )

        q_out_mat_dir = Q_OUT_MAT_file if self.Q_chan_indep else q_out_mat_file
        q_out_mat_dir = _resolve_path(root_path, q_out_mat_dir)
        if not os.path.isfile(q_out_mat_dir):
            raise FileNotFoundError(
                f'OLinear: Q_out not found: {q_out_mat_dir} (need shape pred_len={self.pred_len}). '
                f'Regenerate: python data/generate_corrmat.py --lags ...,{self.pred_len},... '
                f'or set --q_out_mat_file to an existing .npy.'
            )

        q_np = np.load(q_mat_dir).astype(np.float32)
        q_out_np = np.load(q_out_mat_dir).astype(np.float32)

        if self.Q_chan_indep:
            assert q_np.ndim == 3, 'Q_in expects (N, T, T) when Q_chan_indep'
            assert q_out_np.ndim == 3, 'Q_out expects (N, P, P) when Q_chan_indep'
            assert q_np.shape[0] == self.enc_in and q_np.shape[1] == self.seq_len
            assert q_out_np.shape[0] == self.enc_in and q_out_np.shape[1] == self.pred_len
        else:
            assert q_np.ndim == 2 and q_np.shape[0] == self.seq_len
            assert q_out_np.ndim == 2 and q_out_np.shape[0] == self.pred_len

        self.register_buffer('Q_mat', torch.from_numpy(q_np))
        self.register_buffer('Q_out_mat', torch.from_numpy(q_out_np))

        self.patch_len = temp_patch_len
        self.stride = temp_stride
        self.embed_size = embed_size

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(dropout)

        # #############  transformer related  #########
        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=d_model, d_ff=d_ff, CovMat=None,
                    dropout=dropout, activation=activation, token_num=self.enc_in,
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            one_output=True,
            CKA_flag=bool(CKA_flag),
        )
        self.ortho_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.pred_len * self.embed_size)
        )

        # learnable delta
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.seq_len))
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))

    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_Trans(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # orthogonal transformation
        # [B, N, D, T]
        if self.Q_chan_indep:
            x_trans = torch.einsum('bndt,ntv->bndv', x, self.Q_mat.transpose(-1, -2))
        else:
            x_trans = torch.einsum('bndt,tv->bndv', x, self.Q_mat.transpose(-1, -2)) + self.delta1
            # added on 25/1/30
            # x_trans = F.gelu(x_trans)
            # [B, N, D, T]
        assert x_trans.shape[-1] == self.seq_len

        # ########## transformer ####
        x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.pred_len)

        # [B, N, D, tau]; orthogonal transformation
        if self.Q_chan_indep:
            x = torch.einsum('bndt,ntv->bndv', x_trans, self.Q_out_mat)
        else:
            x = torch.einsum('bndt,tv->bndv', x_trans, self.Q_out_mat) + self.delta2
            # added on 25/1/30
            # x = F.gelu(x)

        # [B, N, tau, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        x = self.revin_layer(x, mode='norm')
        x_ori = x

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, tau, D]
        x = self.Fre_Trans(x)

        # linear
        # [B, N, tau*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
