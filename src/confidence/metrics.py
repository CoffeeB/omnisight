"""
Confidence Calibration Metrics (ECE, MCE, Brier Score, Reliability Diagram Bins)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class CalibrationMetrics:
    ece: float
    mce: float
    brier_score: float
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]


def compute_calibration_bins(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions predictions into equally spaced confidence bins [0, 1].
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accs = np.zeros(num_bins, dtype=np.float64)
    bin_confs = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = np.sum(in_bin)
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = np.mean(accuracies[in_bin])
            bin_confs[i] = np.mean(confidences[in_bin])

    return bin_accs, bin_confs, bin_counts


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15
) -> float:
    """
    Computes ECE: Weighted absolute difference between accuracy and confidence across bins.
    """
    bin_accs, bin_confs, bin_counts = compute_calibration_bins(confidences, accuracies, num_bins)
    total_samples = np.sum(bin_counts)
    if total_samples == 0:
        return 0.0

    ece = np.sum(np.abs(bin_accs - bin_confs) * (bin_counts / total_samples))
    return float(ece)


def compute_maximum_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15
) -> float:
    """
    Computes MCE: Maximum deviation across non-empty confidence bins.
    """
    bin_accs, bin_confs, bin_counts = compute_calibration_bins(confidences, accuracies, num_bins)
    valid_bins = bin_counts > 0
    if not np.any(valid_bins):
        return 0.0
    mce = np.max(np.abs(bin_accs[valid_bins] - bin_confs[valid_bins]))
    return float(mce)


def compute_brier_score(
    probs: np.ndarray,
    ground_truth_one_hot: np.ndarray
) -> float:
    """
    Computes multi-class Brier Score: BS = 1/N * sum((p_ik - y_ik)^2)
    """
    return float(np.mean(np.sum((probs - ground_truth_one_hot) ** 2, axis=-1)))


def evaluate_calibration(
    predicted_probs: np.ndarray,
    ground_truth_labels: np.ndarray,
    num_bins: int = 15
) -> CalibrationMetrics:
    """
    Comprehensive evaluation of predictive confidence calibration.
    """
    confidences = np.max(predicted_probs, axis=-1)
    predicted_classes = np.argmax(predicted_probs, axis=-1)
    accuracies = (predicted_classes == ground_truth_labels).astype(np.float64)

    num_classes = predicted_probs.shape[-1]
    gt_one_hot = np.eye(num_classes)[ground_truth_labels]
    brier = compute_brier_score(predicted_probs, gt_one_hot)

    bin_accs, bin_confs, bin_counts = compute_calibration_bins(confidences, accuracies, num_bins)
    total_samples = np.sum(bin_counts)

    ece = float(np.sum(np.abs(bin_accs - bin_confs) * (bin_counts / max(1, total_samples))))
    valid_bins = bin_counts > 0
    mce = float(np.max(np.abs(bin_accs[valid_bins] - bin_confs[valid_bins]))) if np.any(valid_bins) else 0.0

    return CalibrationMetrics(
        ece=ece,
        mce=mce,
        brier_score=brier,
        bin_accuracies=bin_accs.tolist(),
        bin_confidences=bin_confs.tolist(),
        bin_counts=bin_counts.tolist()
    )
