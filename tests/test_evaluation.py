import pytest
import numpy as np
from src.evaluation.metrics import compute_mean_average_precision, compute_boundary_iou
from src.evaluation.robust_bench import RobustnessBenchmarkRunner

def test_map_computation():
    predictions = [{
        "boxes": np.array([[10, 10, 20, 20]], dtype=np.float32),
        "scores": np.array([0.95], dtype=np.float32),
        "classes": np.array([1], dtype=np.int32)
    }]
    ground_truths = [{
        "boxes": np.array([[10, 10, 20, 20]], dtype=np.float32),
        "classes": np.array([1], dtype=np.int32)
    }]

    map_50, map_50_95, _ = compute_mean_average_precision(predictions, ground_truths, num_classes=5)
    assert map_50 == 1.0
    assert map_50_95 == 1.0

def test_boundary_iou():
    gt = np.zeros((40, 40), dtype=np.int32)
    gt[10:30, 10:30] = 1
    pred = gt.copy()

    b_iou = compute_boundary_iou(pred, gt)
    assert b_iou == 1.0

def test_robustness_benchmark_runner():
    runner = RobustnessBenchmarkRunner(clean_benchmark_score=0.80)
    rdi = runner.compute_rdi(degraded_score=0.60)
    assert abs(rdi - 0.25) < 1e-4

    summary = runner.summarize_perturbation_suite({"fog": 0.60, "rain": 0.70})
    assert "mean_RDI" in summary
    assert "per_condition_RDI" in summary
