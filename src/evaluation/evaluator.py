"""
Comprehensive OmniSight Multi-Task Evaluator
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch

from .metrics import (
    compute_mean_average_precision,
    compute_boundary_iou,
    EvaluationSummary
)
from ..segmentation.semantic import compute_semantic_metrics
from ..confidence.metrics import evaluate_calibration

class OmniSightEvaluator:
    """
    Evaluates both segmentation and detection predictions against ground-truth labels.
    """
    def __init__(self, class_names: List[str], num_classes: int = 26):
        self.class_names = class_names
        self.num_classes = num_classes

    def evaluate_dataset(
        self,
        pred_segmentations: List[np.ndarray],
        gt_segmentations: List[np.ndarray],
        pred_detections: List[Dict[str, np.ndarray]],
        gt_detections: List[Dict[str, np.ndarray]],
        pred_probs: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive benchmark evaluation across dataset.
        """
        # 1. Semantic Segmentation Evaluation
        mious = []
        f1s = []
        b_ious = []
        for p_mask, g_mask in zip(pred_segmentations, gt_segmentations):
            s_met = compute_semantic_metrics(p_mask, g_mask, self.num_classes)
            mious.append(s_met["mIoU"])
            f1s.append(s_met["mean_F1"])
            b_ious.append(compute_boundary_iou(p_mask, g_mask))

        mean_miou = float(np.mean(mious)) if mious else 0.0
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        mean_b_iou = float(np.mean(b_ious)) if b_ious else 0.0

        # 2. Object Detection Evaluation (mAP)
        map_50, map_50_95, per_cls_ap = compute_mean_average_precision(
            pred_detections,
            gt_detections,
            num_classes=self.num_classes
        )

        # 3. Calibration Evaluation (if probability tensors provided)
        calib_stats = {}
        if pred_probs is not None and len(pred_probs) > 0:
            flat_probs = np.concatenate([p.reshape(-1, self.num_classes) for p in pred_probs], axis=0)
            flat_gts = np.concatenate([g.flatten() for g in gt_segmentations], axis=0)
            
            # Subsample for speed if very large
            if len(flat_gts) > 100000:
                idx = np.random.choice(len(flat_gts), 100000, replace=False)
                flat_probs = flat_probs[idx]
                flat_gts = flat_gts[idx]

            calib_res = evaluate_calibration(flat_probs, flat_gts)
            calib_stats = {
                "ECE": calib_res.ece,
                "MCE": calib_res.mce,
                "BrierScore": calib_res.brier_score
            }

        return {
            "mIoU": mean_miou,
            "mean_F1": mean_f1,
            "boundary_IoU": mean_b_iou,
            "mAP_50": map_50,
            "mAP_50_95": map_50_95,
            "calibration": calib_stats
        }
