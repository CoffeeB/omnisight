import pytest
import numpy as np
import torch
from src.confidence.metrics import (
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    evaluate_calibration
)
from src.confidence.calibrator import TemperatureScaler

def test_ece_perfect_calibration():
    # Perfectly calibrated case: 100 samples with 0.8 conf and exactly 80% accuracy
    confs = np.full(100, 0.8)
    accs = np.zeros(100)
    accs[:80] = 1.0

    ece = compute_expected_calibration_error(confs, accs, num_bins=10)
    assert abs(ece) < 1e-4

def test_temperature_scaler():
    scaler = TemperatureScaler(init_temperature=2.0)
    logits = torch.tensor([[10.0, 5.0, 1.0]])
    scaled = scaler(logits)

    assert torch.allclose(scaled, logits / 2.0)
    # Scaled softmax should be softer / less peaky
    p_orig = torch.softmax(logits, dim=-1)
    p_scaled = torch.softmax(scaled, dim=-1)
    assert p_scaled.max() < p_orig.max()

def test_evaluate_calibration():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4]
    ])
    labels = np.array([0, 0, 0, 1])
    metrics = evaluate_calibration(probs, labels, num_bins=5)

    assert metrics.ece >= 0.0
    assert metrics.mce >= 0.0
    assert metrics.brier_score >= 0.0
