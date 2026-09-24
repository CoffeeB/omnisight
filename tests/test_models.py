import pytest
import torch
from src.models.backbone import BackboneFPN
from src.models.multi_view_fusion import EpipolarFeatureAggregator
from src.models.omni_net import OmniNet

def test_backbone_fpn():
    backbone = BackboneFPN(in_channels=3, fpn_channels=64)
    x = torch.randn(2, 3, 128, 128)
    feats = backbone(x)

    assert "p2" in feats
    assert "p3" in feats
    assert "p4" in feats
    assert "p5" in feats
    assert feats["p2"].shape == (2, 64, 32, 32)
    assert feats["p3"].shape == (2, 64, 16, 16)
    assert feats["p4"].shape == (2, 64, 8, 8)
    assert feats["p5"].shape == (2, 64, 4, 4)

def test_multi_view_fusion():
    aggregator = EpipolarFeatureAggregator(fpn_channels=64)
    primary = {
        "p2": torch.randn(2, 64, 16, 16),
        "p3": torch.randn(2, 64, 8, 8),
        "p4": torch.randn(2, 64, 4, 4),
        "p5": torch.randn(2, 64, 2, 2)
    }
    aux1 = {
        "p2": torch.randn(2, 64, 16, 16),
        "p3": torch.randn(2, 64, 8, 8),
        "p4": torch.randn(2, 64, 4, 4),
        "p5": torch.randn(2, 64, 2, 2)
    }

    fused = aggregator(primary, [aux1])
    assert fused["p2"].shape == (2, 64, 16, 16)
    assert fused["p3"].shape == (2, 64, 8, 8)

def test_omni_net_forward():
    model = OmniNet(num_classes=26, fpn_channels=64)
    x_primary = torch.randn(1, 3, 128, 128)
    x_aux = torch.randn(1, 3, 128, 128)

    out = model(x_primary, aux_images=[x_aux])
    assert out.seg_logits.shape == (1, 26, 128, 128)
    assert out.seg_probs.shape == (1, 26, 128, 128)
    assert out.seg_uncertainty.shape == (1, 1, 128, 128)
    assert out.bbox_cls_logits.shape[-1] == 26
    assert out.bbox_reg.shape[-1] == 5
