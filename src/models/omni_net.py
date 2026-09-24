"""
OmniNet: Unified Multi-Task Semantic Understanding Architecture
Jointly predicts semantic terrain masks, oriented bounding boxes (OBB), and epistemic/aleatoric uncertainty.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BackboneFPN, ConvBlock
from .multi_view_fusion import EpipolarFeatureAggregator

@dataclass
class OmniNetOutput:
    seg_logits: torch.Tensor             # (B, num_classes, H, W)
    seg_probs: torch.Tensor              # (B, num_classes, H, W)
    seg_uncertainty: torch.Tensor        # (B, 1, H, W) - Predictive entropy / MC variance
    bbox_cls_logits: torch.Tensor        # (B, num_anchors/grids, num_classes)
    bbox_reg: torch.Tensor               # (B, num_anchors/grids, 5) -> [cx, cy, w, h, angle]
    calibrated_confidences: torch.Tensor # (B, num_anchors/grids)


class SemanticSegmentationHead(nn.Module):
    """
    Multi-Scale Feature Aggregation Head for Semantic Terrain Segmentation.
    """
    def __init__(self, fpn_channels: int = 128, num_classes: int = 26, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout2d(dropout)

        self.up_p5 = nn.Sequential(
            ConvBlock(fpn_channels, 64, 3, 1, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        )
        self.up_p4 = nn.Sequential(
            ConvBlock(fpn_channels, 64, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        self.up_p3 = nn.Sequential(
            ConvBlock(fpn_channels, 64, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up_p2 = ConvBlock(fpn_channels, 64, 3, 1, 1)

        # Fused classification layer
        self.classifier = nn.Sequential(
            ConvBlock(256, 128, 3, 1, 1),
            self.dropout,
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, fpn_features: Dict[str, torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        p2 = self.up_p2(fpn_features["p2"])
        p3 = self.up_p3(fpn_features["p3"])
        p4 = self.up_p4(fpn_features["p4"])
        p5 = self.up_p5(fpn_features["p5"])

        fused = torch.cat([p2, p3, p4, p5], dim=1) # (B, 256, H/4, W/4)
        logits_low = self.classifier(fused)
        logits = F.interpolate(logits_low, size=target_size, mode='bilinear', align_corners=False)
        return logits


class OrientedDetectionHead(nn.Module):
    """
    Anchor-Free Oriented Bounding Box (OBB) Detection Head.
    """
    def __init__(self, in_channels: int = 128, num_classes: int = 26):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        self.reg_head = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, 5, 1) # cx_offset, cy_offset, log(w), log(h), angle
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_head(feat) # (B, C, H, W)
        reg_preds = self.reg_head(feat)  # (B, 5, H, W)
        return cls_logits, reg_preds


class OmniNet(nn.Module):
    """
    Unified Multi-Task Research Model for OmniSight.
    """
    def __init__(
        self,
        num_classes: int = 26,
        fpn_channels: int = 128,
        temperature: float = 1.0,
        enable_mc_dropout: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        self.enable_mc_dropout = enable_mc_dropout

        self.backbone = BackboneFPN(in_channels=3, fpn_channels=fpn_channels)
        self.multi_view_aggregator = EpipolarFeatureAggregator(fpn_channels=fpn_channels)
        self.seg_head = SemanticSegmentationHead(fpn_channels=fpn_channels, num_classes=num_classes)
        self.det_head = OrientedDetectionHead(in_channels=fpn_channels, num_classes=num_classes)

    def forward(
        self,
        primary_img: torch.Tensor,
        aux_images: Optional[List[torch.Tensor]] = None,
        mc_samples: int = 1
    ) -> OmniNetOutput:
        """
        Forward pass with optional multi-view feature fusion and Monte Carlo epistemic uncertainty.
        """
        b, _, h, w = primary_img.shape
        primary_fpn = self.backbone(primary_img)

        aux_fpn_list = []
        if aux_images is not None and len(aux_images) > 0:
            for aux in aux_images:
                aux_fpn_list.append(self.backbone(aux))

        fused_fpn = self.multi_view_aggregator(primary_fpn, aux_fpn_list)

        # Monte Carlo Dropout for epistemic uncertainty if requested
        if mc_samples > 1 or self.enable_mc_dropout:
            self.seg_head.train() # Enable dropout during evaluation
            logits_list = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    l = self.seg_head(fused_fpn, target_size=(h, w))
                    logits_list.append(F.softmax(l / self.temperature, dim=1))
            stacked_probs = torch.stack(logits_list, dim=0) # (N, B, C, H, W)
            mean_probs = stacked_probs.mean(dim=0)
            uncertainty = stacked_probs.var(dim=0).mean(dim=1, keepdim=True) # Mean class variance
            seg_logits = torch.log(mean_probs + 1e-8)
            seg_probs = mean_probs
        else:
            seg_logits = self.seg_head(fused_fpn, target_size=(h, w))
            seg_probs = F.softmax(seg_logits / self.temperature, dim=1)
            # Predictive entropy as aleatoric uncertainty
            uncertainty = -torch.sum(seg_probs * torch.log(seg_probs + 1e-8), dim=1, keepdim=True)

        # Detection head on P3 feature map
        det_cls_logits, det_reg = self.det_head(fused_fpn["p3"])
        det_cls_flat = det_cls_logits.flatten(2).permute(0, 2, 1) # (B, HW, C)
        det_reg_flat = det_reg.flatten(2).permute(0, 2, 1)        # (B, HW, 5)

        # Calibrated object-level confidences
        calibrated_conf = torch.max(F.softmax(det_cls_flat / self.temperature, dim=-1), dim=-1)[0]

        return OmniNetOutput(
            seg_logits=seg_logits,
            seg_probs=seg_probs,
            seg_uncertainty=uncertainty,
            bbox_cls_logits=det_cls_flat,
            bbox_reg=det_reg_flat,
            calibrated_confidences=calibrated_conf
        )
