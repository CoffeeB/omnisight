"""
Unified Multi-Modal Dataset Interface
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class SceneSample:
    primary_image: np.ndarray             # (H, W, 3) uint8 RGB
    auxiliary_views: List[np.ndarray]     # List of (H, W, 3) auxiliary camera views
    semantic_mask: np.ndarray             # (H, W) int32 class IDs
    detection_boxes: np.ndarray           # (N, 5) [cx, cy, w, h, angle] or (N, 4) [xmin, ymin, xmax, ymax]
    detection_classes: np.ndarray         # (N,) int32 class IDs
    metadata: Dict[str, Any]


class BaseDataset(Dataset):
    """
    Abstract Base Dataset for OmniSight multi-view perception tasks.
    """
    def __init__(self, samples: List[SceneSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        primary_img = sample.primary_image
        aux_imgs = sample.auxiliary_views
        mask = sample.semantic_mask
        boxes = sample.detection_boxes
        classes = sample.detection_classes

        if self.transform is not None:
            primary_img, aux_imgs, mask, boxes = self.transform(primary_img, aux_imgs, mask, boxes)

        # Convert to tensors
        p_tensor = torch.from_numpy(primary_img.transpose(2, 0, 1)).float() / 255.0
        aux_tensors = [
            torch.from_numpy(aux.transpose(2, 0, 1)).float() / 255.0 for aux in aux_imgs
        ]
        m_tensor = torch.from_numpy(mask).long()

        return {
            "primary_image": p_tensor,
            "auxiliary_views": aux_tensors,
            "semantic_mask": m_tensor,
            "detection_boxes": boxes,
            "detection_classes": classes,
            "metadata": sample.metadata
        }
