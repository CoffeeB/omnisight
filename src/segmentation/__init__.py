"""
OmniSight Semantic and Instance Segmentation Module
"""

from .semantic import (
    SemanticSegmentationPredictor,
    compute_semantic_metrics,
    SegmentationResult
)
from .instance import (
    extract_instance_masks,
    InstanceMaskResult
)

__all__ = [
    "SemanticSegmentationPredictor",
    "compute_semantic_metrics",
    "SegmentationResult",
    "extract_instance_masks",
    "InstanceMaskResult",
]
