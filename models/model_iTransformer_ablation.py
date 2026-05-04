import torch
import torch.nn as nn

from models.model_iTransformer import iTransformerUni4, iTransformerUni


class _SelfFactorSelector(nn.Module):
    """
    Select only one factor slice from concatenated factor features.
    """
    def __init__(self, factor_idx, d_model):
        super(_SelfFactorSelector, self).__init__()
        self.factor_idx = int(factor_idx)
        self.d_model = int(d_model)

    def forward(self, x):
        start = self.factor_idx * self.d_model
        end = start + self.d_model
        if x.shape[-1] < end:
            raise ValueError(
                "Input last dim {} is too small for factor {} slice [{}:{}].".format(
                    x.shape[-1], self.factor_idx, start, end
                )
            )
        return x[..., start:end]


class iTransformerUniAbl(iTransformerUni4):
    """
    Ablation model:
    - Keep 4-factor inputs and per-factor encoders unchanged.
    - Disable cross-factor fusion by replacing MLPs with factor-local selectors.
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
        super(iTransformerUniAbl, self).__init__(
            enc_in, dec_in, c_out, seq_len, label_len, out_len,
            factor=factor, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=d_layers, d_ff=d_ff, move_avg=move_avg,
            dropout=dropout, attn=attn, embed=embed, freq=freq, activation=activation,
            output_attention=output_attention, distil=distil, mix=mix, use_multi_scale=use_multi_scale, patembed=patembed,
            scales=scales, scale_factor=scale_factor,
            version=version, mode_select=mode_select, modes=modes, L=L, base=base, cross_activation=cross_activation,
            conv_dff=conv_dff,
            device=device,
            land_mask_path=land_mask_path,
            scale_mask_mode=scale_mask_mode,
        )

        # Replace cross-factor MLPs with identity-like selectors on each factor slice.
        self.mlp_one = _SelfFactorSelector(0, self.d_model)
        self.mlp_two = _SelfFactorSelector(1, self.d_model)
        self.mlp_three = _SelfFactorSelector(2, self.d_model)
        # For 3-factor ablation, we don't need the fourth MLP.
        self.mlp_four = _SelfFactorSelector(3, self.d_model)
