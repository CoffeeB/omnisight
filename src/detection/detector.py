"""
Oriented Bounding Box (OBB) & Axis-Aligned Object Detection Engine
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .postprocess import rotated_non_max_suppression, non_max_suppression

@dataclass
class DetectionPrediction:
    boxes: np.ndarray        # (N, 4) [xmin, ymin, xmax, ymax] or (N, 5) [cx, cy, w, h, angle]
    scores: np.ndarray       # (N,) Calibrated confidence scores
    class_ids: np.ndarray    # (N,) Integer class indices
    class_names: List[str]   # (N,) Human-readable category strings
    uncertainties: np.ndarray# (N,) Epistemic/aleatoric uncertainty score
    is_oriented: bool = True

    def to_json_dict(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i in range(len(self.scores)):
            entry: Dict[str, Any] = {
                "object": self.class_names[i],
                "class_id": int(self.class_ids[i]),
                "confidence": float(np.round(self.scores[i], 4)),
                "uncertainty": float(np.round(self.uncertainties[i], 4)),
            }
            if self.is_oriented:
                entry["rotated_bbox"] = {
                    "cx": float(np.round(self.boxes[i, 0], 2)),
                    "cy": float(np.round(self.boxes[i, 1], 2)),
                    "width": float(np.round(self.boxes[i, 2], 2)),
                    "height": float(np.round(self.boxes[i, 3], 2)),
                    "angle_rad": float(np.round(self.boxes[i, 4], 4))
                }
            else:
                entry["bbox"] = [
                    float(np.round(c, 2)) for c in self.boxes[i]
                ]
            results.append(entry)
        return results


class OrientedDetector:
    """
    Detector wrapping OmniNet detection output with decoding and suppression.
    """
    def __init__(
        self,
        class_names: List[str],
        score_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        strides: int = 8
    ):
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.stride = strides

    def decode_predictions(
        self,
        cls_logits: torch.Tensor,
        reg_preds: torch.Tensor,
        img_shape: tuple,
        uncertainty_map: Optional[torch.Tensor] = None
    ) -> DetectionPrediction:
        """
        Decodes grid predictions into rotated bounding boxes with calibrated confidences.
        """
        h, w = img_shape[:2]
        grid_h, grid_w = h // self.stride, w // self.stride

        probs = F.softmax(cls_logits[0], dim=-1).detach().cpu().numpy() # (HW, num_classes)
        regs = reg_preds[0].detach().cpu().numpy()                      # (HW, 5)

        # Build 2D grid coordinates
        ys, xs = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
        xs = (xs.flatten() + 0.5) * self.stride
        ys = (ys.flatten() + 0.5) * self.stride

        scores = np.max(probs[:, 1:], axis=-1) # Exclude background class 0
        pred_classes = np.argmax(probs[:, 1:], axis=-1) + 1

        mask = scores >= self.score_thresh
        if not np.any(mask):
            return DetectionPrediction(
                boxes=np.zeros((0, 5), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                class_ids=np.zeros(0, dtype=np.int32),
                class_names=[],
                uncertainties=np.zeros(0, dtype=np.float32),
                is_oriented=True
            )

        valid_scores = scores[mask]
        valid_classes = pred_classes[mask]
        valid_regs = regs[mask]
        valid_xs = xs[mask]
        valid_ys = ys[mask]

        cx = valid_xs + valid_regs[:, 0] * self.stride
        cy = valid_ys + valid_regs[:, 1] * self.stride
        bw = np.exp(np.clip(valid_regs[:, 2], -2.0, 4.0)) * self.stride * 2.0
        bh = np.exp(np.clip(valid_regs[:, 3], -2.0, 4.0)) * self.stride * 2.0
        angle = np.clip(valid_regs[:, 4], -np.pi / 2, np.pi / 2)

        boxes_5d = np.stack([cx, cy, bw, bh, angle], axis=1)

        # Extract uncertainty at box centers
        if uncertainty_map is not None:
            unc_np = uncertainty_map[0, 0].detach().cpu().numpy()
            cx_int = np.clip(cx.astype(int), 0, w - 1)
            cy_int = np.clip(cy.astype(int), 0, h - 1)
            unc_scores = unc_np[cy_int, cx_int]
        else:
            unc_scores = 1.0 - valid_scores

        keep_indices = rotated_non_max_suppression(
            boxes_5d, valid_scores, iou_thresh=self.iou_thresh
        )

        final_boxes = boxes_5d[keep_indices]
        final_scores = valid_scores[keep_indices]
        final_classes = valid_classes[keep_indices]
        final_unc = unc_scores[keep_indices]
        final_names = [
            self.class_names[cid] if cid < len(self.class_names) else f"Class_{cid}"
            for cid in final_classes
        ]

        return DetectionPrediction(
            boxes=final_boxes,
            scores=final_scores,
            class_ids=final_classes,
            class_names=final_names,
            uncertainties=final_unc,
            is_oriented=True
        )
