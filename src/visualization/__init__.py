"""
OmniSight Visualization & Publication Figure Rendering Subsystem
"""

from .visualizer import (
    render_semantic_overlay,
    render_detection_boxes,
    render_uncertainty_heatmap,
    render_multi_view_grid,
    CLASS_COLORS
)
from .reliability_plots import (
    plot_reliability_diagram,
    plot_calibration_curve_comparison,
    plot_degradation_profile
)
from .dashboard import generate_research_dashboard

__all__ = [
    "render_semantic_overlay",
    "render_detection_boxes",
    "render_uncertainty_heatmap",
    "render_multi_view_grid",
    "CLASS_COLORS",
    "plot_reliability_diagram",
    "plot_calibration_curve_comparison",
    "plot_degradation_profile",
    "generate_research_dashboard",
]
