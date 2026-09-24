"""
Multi-View Cross-Attention Epipolar Spatial Transformer
Aggregates visual feature representations across multiple camera angles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class CrossViewTransformerFusion(nn.Module):
    """
    Multi-Head Cross-Attention Transformer across auxiliary viewpoints.
    """
    def __init__(self, embed_dim: int = 64, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, query_feat: torch.Tensor, key_feat_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            query_feat: Tensor of shape (B, C, H, W) - Primary view feature map
            key_feat_list: List of Tensors of shape (B, C, H, W) - Auxiliary view feature maps
            
        Returns:
            Fused feature map (B, C, H, W)
        """
        if not key_feat_list:
            return query_feat

        b, c, h, w = query_feat.shape
        # Spatial pooling for efficient cross-attention token sequence
        max_dim = 8
        if h > max_dim or w > max_dim:
            q_pooled = F.adaptive_avg_pool2d(query_feat, (max_dim, max_dim))
            q = q_pooled.flatten(2).permute(0, 2, 1) # (B, 256, C)
            keys_list = [
                F.adaptive_avg_pool2d(k, (max_dim, max_dim)).flatten(2).permute(0, 2, 1)
                for k in key_feat_list
            ]
            kv = torch.cat(keys_list, dim=1) # (B, V * 256, C)
            
            q_norm = self.norm1(q)
            kv_norm = self.norm1(kv)
            attn_out, _ = self.multihead_attn(q_norm, kv_norm, kv_norm)
            x = q + attn_out
            x = x + self.ffn(self.norm2(x))
            
            out_pooled = x.permute(0, 2, 1).view(b, c, max_dim, max_dim)
            out_res = F.interpolate(out_pooled, size=(h, w), mode='bilinear', align_corners=False)
            return query_feat + out_res
        else:
            q = query_feat.flatten(2).permute(0, 2, 1)
            keys_list = [k.flatten(2).permute(0, 2, 1) for k in key_feat_list]
            kv = torch.cat(keys_list, dim=1)

            q_norm = self.norm1(q)
            kv_norm = self.norm1(kv)

            attn_out, _ = self.multihead_attn(q_norm, kv_norm, kv_norm)
            x = q + attn_out
            x = x + self.ffn(self.norm2(x))

            out = x.permute(0, 2, 1).view(b, c, h, w)
            return out


class EpipolarFeatureAggregator(nn.Module):
    """
    Combines multi-scale FPN feature maps across multi-view streams.
    """
    def __init__(self, fpn_channels: int = 128):
        super().__init__()
        self.fusion_p2 = CrossViewTransformerFusion(embed_dim=fpn_channels, num_heads=4)
        self.fusion_p3 = CrossViewTransformerFusion(embed_dim=fpn_channels, num_heads=4)
        self.fusion_p4 = CrossViewTransformerFusion(embed_dim=fpn_channels, num_heads=4)

    def forward(
        self,
        primary_fpn: Dict[str, torch.Tensor],
        aux_fpn_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if not aux_fpn_list:
            return primary_fpn

        p2_aux = [aux["p2"] for aux in aux_fpn_list]
        p3_aux = [aux["p3"] for aux in aux_fpn_list]
        p4_aux = [aux["p4"] for aux in aux_fpn_list]

        return {
            "p2": self.fusion_p2(primary_fpn["p2"], p2_aux),
            "p3": self.fusion_p3(primary_fpn["p3"], p3_aux),
            "p4": self.fusion_p4(primary_fpn["p4"], p4_aux),
            "p5": primary_fpn["p5"]
        }
