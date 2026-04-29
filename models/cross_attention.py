"""
交叉注意力模块：用于时空双流融合
实现时间流和空间流之间的交互
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    交叉注意力模块：用于融合时间流和空间流的特征
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 交叉注意力层
        self.cross_attn_temporal = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False
        )
        self.cross_attn_spatial = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 融合层
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, temporal_feat, spatial_feat):
        """
        前向传播
        
        Args:
            temporal_feat: 时间流特征 [B, N, d_model] 或 [B, L, d_model]
            spatial_feat: 空间流特征 [B, N, d_model] 或 [B, L, d_model]
        
        Returns:
            fused_feat: 融合后的特征 [B, N, d_model] 或 [B, L, d_model]
        """
        # 确保输入维度一致
        # 如果维度是 [B, N, d_model]，需要转换为 [N, B, d_model] 用于 MultiheadAttention
        if temporal_feat.dim() == 3:
            B, N, D = temporal_feat.shape
            # 转换为 [N, B, d_model] 格式
            temporal_feat_seq = temporal_feat.permute(1, 0, 2)  # [N, B, d_model]
            spatial_feat_seq = spatial_feat.permute(1, 0, 2)     # [N, B, d_model]
        else:
            temporal_feat_seq = temporal_feat
            spatial_feat_seq = spatial_feat
        
        # 1. 时间流关注空间流
        # temporal_feat 作为 query，spatial_feat 作为 key 和 value
        temporal_enhanced, _ = self.cross_attn_temporal(
            temporal_feat_seq, spatial_feat_seq, spatial_feat_seq
        )
        # temporal_enhanced: [N, B, d_model]
        
        # 残差连接和归一化
        if temporal_feat.dim() == 3:
            temporal_enhanced = temporal_enhanced.permute(1, 0, 2)  # [B, N, d_model]
            temporal_enhanced = self.norm1(temporal_feat + self.dropout(temporal_enhanced))
        else:
            temporal_enhanced = self.norm1(temporal_feat + self.dropout(temporal_enhanced))
        
        # 2. 空间流关注时间流
        # spatial_feat 作为 query，temporal_feat 作为 key 和 value
        if temporal_feat.dim() == 3:
            temporal_enhanced_seq = temporal_enhanced.permute(1, 0, 2)  # [N, B, d_model]
            spatial_feat_seq = spatial_feat.permute(1, 0, 2)             # [N, B, d_model]
        else:
            temporal_enhanced_seq = temporal_enhanced
            spatial_feat_seq = spatial_feat
            
        spatial_enhanced, _ = self.cross_attn_spatial(
            spatial_feat_seq, temporal_enhanced_seq, temporal_enhanced_seq
        )
        # spatial_enhanced: [N, B, d_model]
        
        # 残差连接和归一化
        if temporal_feat.dim() == 3:
            spatial_enhanced = spatial_enhanced.permute(1, 0, 2)  # [B, N, d_model]
            spatial_enhanced = self.norm2(spatial_feat + self.dropout(spatial_enhanced))
        else:
            spatial_enhanced = self.norm2(spatial_feat + self.dropout(spatial_enhanced))
        
        # 3. 融合两个流的特征
        fused = torch.cat([temporal_enhanced, spatial_enhanced], dim=-1)  # [B, N, 2*d_model]
        fused = self.fusion(fused)  # [B, N, d_model]
        fused = self.norm3(fused)
        
        return fused

