"""
Multi-Scale Backbone with Feature Pyramid Network (FPN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn2(self.conv2(self.conv1(x))) + x)


class BackboneFPN(nn.Module):
    """
    Hierarchical Feature Pyramid Network extracting P2, P3, P4, P5 multi-scale feature maps.
    """
    def __init__(self, in_channels: int = 3, fpn_channels: int = 64):
        super().__init__()
        self.fpn_channels = fpn_channels

        # Stage 1: Stem (Stride 2)
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 16, 3, stride=2, padding=1),
            ConvBlock(16, 32, 3, stride=1, padding=1)
        )

        # Stage 2: Stride 4 (C2)
        self.stage2 = nn.Sequential(
            ConvBlock(32, 32, 3, stride=2, padding=1),
            ResidualBlock(32)
        )

        # Stage 3: Stride 8 (C3)
        self.stage3 = nn.Sequential(
            ConvBlock(32, 64, 3, stride=2, padding=1),
            ResidualBlock(64)
        )

        # Stage 4: Stride 16 (C4)
        self.stage4 = nn.Sequential(
            ConvBlock(64, 128, 3, stride=2, padding=1),
            ResidualBlock(128)
        )

        # Stage 5: Stride 32 (C5)
        self.stage5 = nn.Sequential(
            ConvBlock(128, 128, 3, stride=2, padding=1),
            ResidualBlock(128)
        )

        # Lateral 1x1 convolutions
        self.lat5 = nn.Conv2d(128, fpn_channels, 1)
        self.lat4 = nn.Conv2d(128, fpn_channels, 1)
        self.lat3 = nn.Conv2d(64, fpn_channels, 1)
        self.lat2 = nn.Conv2d(32, fpn_channels, 1)

        # Output smooth 3x3 convolutions
        self.smooth5 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')

        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
