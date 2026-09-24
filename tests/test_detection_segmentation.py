import pytest
import numpy as np
import torch
from src.detection.detector import OrientedDetector
from src.detection.postprocess import (
    compute_iou_axis_aligned_py,
    compute_rotated_iou_py,
    rotated_non_max_suppression
)
from src.segmentation.semantic import compute_semantic_metrics, SemanticSegmentationPredictor
from src.segmentation.instance import extract_instance_masks
from datasets.synthetic_generator import CLASS_NAMES

def test_iou_calculations():
    # Axis-aligned box IoU
    box1 = np.array([0, 0, 10, 10], dtype=np.float32)
    box2 = np.array([5, 0, 15, 10], dtype=np.float32)
    iou = compute_iou_axis_aligned_py(box1, box2)
    assert abs(iou - 0.3333) < 1e-3

    # Rotated box IoU (identical squares)
    rbox1 = np.array([5, 5, 10, 10, 0.0], dtype=np.float32)
    rbox2 = np.array([5, 5, 10, 10, 0.0], dtype=np.float32)
    r_iou = compute_rotated_iou_py(rbox1, rbox2)
    assert abs(r_iou - 1.0) < 1e-2

def test_rotated_nms():
    boxes = np.array([
        [5, 5, 10, 10, 0.0],
        [5.5, 5.5, 10, 10, 0.0], # Overlapping
        [50, 50, 10, 10, 0.0],  # Far away
    ], dtype=np.float32)
    scores = np.array([0.95, 0.85, 0.90], dtype=np.float32)

    keep = rotated_non_max_suppression(boxes, scores, iou_thresh=0.5)
    assert len(keep) == 2
    assert 0 in keep and 2 in keep

def test_semantic_metrics():
    gt = np.zeros((50, 50), dtype=np.int32)
    gt[10:30, 10:30] = 1
    pred = gt.copy()
    pred[25:35, 25:35] = 1 # Slight perturbation

    metrics = compute_semantic_metrics(pred, gt, num_classes=26)
    assert "mIoU" in metrics
    assert metrics["mIoU"] > 0.0
    assert metrics["pixel_accuracy"] > 0.8

def test_instance_mask_extraction():
    sem_mask = np.zeros((50, 50), dtype=np.int32)
    sem_mask[10:30, 10:30] = 1
    boxes = np.array([[20, 20, 20, 20, 0.0]], dtype=np.float32)
    classes = np.array([1], dtype=np.int32)
    scores = np.array([0.95], dtype=np.float32)

    res = extract_instance_masks(sem_mask, boxes, classes, scores, min_area_pixels=10)
    assert len(res.instance_masks) == 1
    assert res.instance_masks[0].shape == (50, 50)
