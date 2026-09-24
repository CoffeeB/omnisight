"""
Semantic Segmentation Inference, Mask Generation, and Multi-Class Evaluators
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class SegmentationResult:
    class_mask: np.ndarray       # (H, W) int32 class IDs
    probability_map: np.ndarray  # (H, W) float32 max class probabilities
    uncertainty_map: np.ndarray  # (H, W) float32 predictive entropy or variance
    class_proportions: Dict[str, float]


def compute_semantic_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = 26
) -> Dict[str, float]:
    """
    Computes per-class IoU, Mean IoU (mIoU), Overall Pixel Accuracy, and F1-score via fast vectorized confusion matrix.
    """
    pred_flat = pred_mask.flatten().astype(np.int64)
    gt_flat = gt_mask.flatten().astype(np.int64)

    # Filter invalid labels
    valid = (gt_flat >= 0) & (gt_flat < num_classes) & (pred_flat >= 0) & (pred_flat < num_classes)
    p = pred_flat[valid]
    g = gt_flat[valid]

    if len(p) == 0:
        return {"mIoU": 0.0, "mean_F1": 0.0, "pixel_accuracy": 0.0, "evaluated_classes_count": 0}

    # Confusion matrix: index = g * num_classes + p
    conf = np.bincount(g * num_classes + p, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp

    denominator = tp + fp + fn
    # Skip background class 0 from mIoU calculation
    valid_classes = (denominator > 0) & (np.arange(num_classes) > 0)

    if not np.any(valid_classes):
        mean_iou = 0.0
        mean_f1 = 0.0
    else:
        ious = tp[valid_classes] / denominator[valid_classes]
        prec = tp[valid_classes] / np.maximum(1, tp[valid_classes] + fp[valid_classes])
        rec = tp[valid_classes] / np.maximum(1, tp[valid_classes] + fn[valid_classes])
        f1s = 2 * (prec * rec) / np.maximum(1e-8, prec + rec)

        mean_iou = float(np.mean(ious))
        mean_f1 = float(np.mean(f1s))

    overall_acc = float(tp.sum() / max(1, conf.sum()))

    return {
        "mIoU": mean_iou,
        "mean_F1": mean_f1,
        "pixel_accuracy": overall_acc,
        "evaluated_classes_count": int(np.sum(valid_classes))
    }


class SemanticSegmentationPredictor:
    """
    Inference helper for semantic segmentation models.
    """
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def predict(
        self,
        seg_logits: torch.Tensor,
        uncertainty_tensor: Optional[torch.Tensor] = None
    ) -> SegmentationResult:
        probs = F.softmax(seg_logits[0], dim=0).detach().cpu().numpy() # (C, H, W)
        pred_mask = np.argmax(probs, axis=0).astype(np.int32)
        conf_map = np.max(probs, axis=0).astype(np.float32)

        if uncertainty_tensor is not None:
            unc_map = uncertainty_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
        else:
            unc_map = (1.0 - conf_map).astype(np.float32)

        total_pixels = float(pred_mask.size)
        proportions = {}
        for cid in np.unique(pred_mask):
            c_name = self.class_names[cid] if cid < len(self.class_names) else f"Class_{cid}"
            proportions[c_name] = float((pred_mask == cid).sum()) / total_pixels

        return SegmentationResult(
            class_mask=pred_mask,
            probability_map=conf_map,
            uncertainty_map=unc_map,
            class_proportions=proportions
        )
