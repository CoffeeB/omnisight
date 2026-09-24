"""
OmniSight Object & Oriented Bounding Box Detection Subsystem
"""

from .detector import OrientedDetector, DetectionPrediction
from .postprocess import non_max_suppression, rotated_non_max_suppression

__all__ = [
    "OrientedDetector",
    "DetectionPrediction",
    "non_max_suppression",
    "rotated_non_max_suppression",
]
