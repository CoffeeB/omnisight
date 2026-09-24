"""
OmniSight Epistemic & Aleatoric Uncertainty and Confidence Calibration Subsystem
"""

from .calibrator import (
    TemperatureScaler,
    MonteCarloDropoutCalibrator,
    CalibratedInferenceEngine
)
from .metrics import (
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    compute_brier_score,
    compute_calibration_bins,
    CalibrationMetrics
)

__all__ = [
    "TemperatureScaler",
    "MonteCarloDropoutCalibrator",
    "CalibratedInferenceEngine",
    "compute_expected_calibration_error",
    "compute_maximum_calibration_error",
    "compute_brier_score",
    "compute_calibration_bins",
    "CalibrationMetrics",
]
