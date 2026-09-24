"""
OmniSight Physical Preprocessing & Sensor Degradation Module
"""

from .atmospheric import (
    apply_koschmieder_fog,
    apply_rain_streaks,
    apply_solar_shadows,
    apply_low_light_noise,
    apply_sunset_lighting,
    AtmosphericDegradationPipeline
)
from .geometry import (
    apply_viewpoint_transformation,
    compute_homography_matrix,
    ViewpointConfig,
    ViewpointType
)
from .altitude import (
    resample_ground_sample_distance,
    AltitudeScaleEngine
)

__all__ = [
    "apply_koschmieder_fog",
    "apply_rain_streaks",
    "apply_solar_shadows",
    "apply_low_light_noise",
    "apply_sunset_lighting",
    "AtmosphericDegradationPipeline",
    "apply_viewpoint_transformation",
    "compute_homography_matrix",
    "ViewpointConfig",
    "ViewpointType",
    "resample_ground_sample_distance",
    "AltitudeScaleEngine",
]
