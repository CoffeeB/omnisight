"""
mAP (COCO/PASCAL format), IoU/mIoU, Boundary-IoU, Precision, Recall, and F1-score
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import cv2

from ..detection.postprocess import compute_rotated_iou_py, compute_iou_axis_aligned_py

@dataclass
class EvaluationSummary:
    mAP_50: float
    mAP_50_95: float
    mIoU: float
    mean_f1: float
    boundary_iou: float
    class_metrics: Dict[str, Dict[str, float]]


def compute_ap_voc_coco(
    recalls: np.ndarray,
    precisions: np.ndarray
) -> float:
    """
    Computes Average Precision (AP) with 101-point interpolation (COCO standard).
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 101 recall points
    recall_pts = np.linspace(0, 1, 101)
    inds = np.searchsorted(mrec, recall_pts, side='left')
    ap = np.mean(mpre[inds])
    return float(ap)


def compute_mean_average_precision(
    predictions: List[Dict], # List of {"boxes": (N, 4/5), "scores": (N,), "classes": (N,)}
    ground_truths: List[Dict], # List of {"boxes": (M, 4/5), "classes": (M,)}
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    num_classes: int = 26
) -> Tuple[float, float, Dict[int, float]]:
    """
    Calculates mAP@50 and mAP@50:95 across rotated or axis-aligned detections.
    """
    aps_per_iou = {iou_th: [] for iou_th in iou_thresholds}
    per_class_ap50 = {}

    for c in range(1, num_classes):
        # Gather all predictions and GTs for class c
        c_scores = []
        c_matches = {iou_th: [] for iou_th in iou_thresholds}
        total_gt = 0

        for pred_dict, gt_dict in zip(predictions, ground_truths):
            p_mask = pred_dict["classes"] == c
            g_mask = gt_dict["classes"] == c

            p_boxes = pred_dict["boxes"][p_mask]
            p_sc = pred_dict["scores"][p_mask]
            g_boxes = gt_dict["boxes"][g_mask]
            total_gt += len(g_boxes)

            if len(p_boxes) == 0:
                continue

            order = np.argsort(-p_sc)
            p_boxes = p_boxes[order]
            p_sc = p_sc[order]
            c_scores.extend(p_sc.tolist())

            for iou_th in iou_thresholds:
                gt_matched = np.zeros(len(g_boxes), dtype=bool)
                for pb in p_boxes:
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g_idx, gb in enumerate(g_boxes):
                        if gt_matched[g_idx]:
                            continue
                        if len(pb) == 5 and len(gb) == 5:
                            iou = compute_rotated_iou_py(pb, gb)
                        else:
                            iou = compute_iou_axis_aligned_py(pb[:4], gb[:4])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx

                    if best_iou >= iou_th and best_gt_idx >= 0:
                        gt_matched[best_gt_idx] = True
                        c_matches[iou_th].append(1) # True positive
                    else:
                        c_matches[iou_th].append(0) # False positive

        if total_gt == 0:
            continue

        c_scores = np.array(c_scores)
        sort_order = np.argsort(-c_scores) if len(c_scores) > 0 else np.array([])

        for iou_th in iou_thresholds:
            if len(c_matches[iou_th]) == 0:
                aps_per_iou[iou_th].append(0.0)
                continue
            matches = np.array(c_matches[iou_th])[sort_order]
            tps = np.cumsum(matches)
            fps = np.cumsum(1 - matches)
            recalls = tps / total_gt
            precisions = tps / (tps + fps)
            ap = compute_ap_voc_coco(recalls, precisions)
            aps_per_iou[iou_th].append(ap)
            if abs(iou_th - 0.5) < 1e-4:
                per_class_ap50[c] = ap

    map_50 = float(np.mean(aps_per_iou[0.5])) if aps_per_iou[0.5] else 0.0
    all_map_list = [np.mean(aps_per_iou[th]) for th in iou_thresholds if aps_per_iou[th]]
    map_50_95 = float(np.mean(all_map_list)) if all_map_list else 0.0

    return map_50, map_50_95, per_class_ap50


def compute_boundary_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dilation_pixels: int = 3
) -> float:
    """
    Computes Boundary IoU (Cheng et al., CVPR 2021) focusing strictly on object edges.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1))
    
    # Extract boundaries via morphological gradient
    pred_boundary = cv2.morphologyEx((pred_mask > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    gt_boundary = cv2.morphologyEx((gt_mask > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel)

    intersection = np.logical_and(pred_boundary > 0, gt_boundary > 0).sum()
    union = np.logical_or(pred_boundary > 0, gt_boundary > 0).sum()

    if union == 0:
        return 1.0 if (pred_boundary.sum() == 0 and gt_boundary.sum() == 0) else 0.0

    return float(intersection / union)
