"""
Reliability Diagrams, Calibration Curves, and Degradation Profile Visualizers
"""

from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_reliability_diagram(
    bin_accuracies: List[float],
    bin_confidences: List[float],
    bin_counts: List[int],
    ece: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders publication-grade Reliability Diagram with ECE indicator.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    bins = np.linspace(0, 1, len(bin_accuracies) + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]

    # Subplot 1: Reliability Diagram
    ax1.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    ax1.bar(
        bin_centers, bin_accuracies, width=width * 0.9,
        alpha=0.6, color='#2b5c8f', edgecolor='black', label='Outputs'
    )
    # Gap highlight
    gaps = np.abs(np.array(bin_accuracies) - np.array(bin_confidences))
    ax1.bar(
        bin_centers, gaps, bottom=np.minimum(bin_accuracies, bin_confidences),
        width=width * 0.9, alpha=0.35, color='#d9534f', hatch='//', label='Calibration Gap'
    )

    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.set_title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_ylim(0, 1.05)

    # Subplot 2: Confidence Histogram
    ax2.bar(bin_centers, bin_counts, width=width * 0.9, color='#5cb85c', edgecolor='black')
    ax2.set_xlabel("Confidence Bin", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_calibration_curve_comparison(
    uncalibrated_ece: float,
    uncalibrated_accs: List[float],
    calibrated_ece: float,
    calibrated_accs: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compares calibration curves before and after temperature scaling.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    bins = np.linspace(0, 1, len(uncalibrated_accs) + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    ax.plot([0, 1], [0, 1], 'k--', label='Ideal Perfect Calibration')
    ax.plot(bin_centers, uncalibrated_accs, 'r-o', label=f'Uncalibrated Softmax (ECE={uncalibrated_ece:.3f})')
    ax.plot(bin_centers, calibrated_accs, 'b-s', label=f'Temperature Scaled (ECE={calibrated_ece:.3f})')

    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_title("Post-Hoc Probability Calibration Comparison", fontsize=12, fontweight='bold')
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_degradation_profile(
    x_labels: List[str],
    single_view_scores: List[float],
    multi_view_scores: List[float],
    metric_name: str = "mIoU",
    title: str = "Robustness Degradation Profile",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots multi-view vs single-view degradation under perturbation levels.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(x_labels))
    width = 0.35

    ax.bar(x - width / 2, single_view_scores, width, label='Single-View Baseline', color='#d9534f', alpha=0.85)
    ax.bar(x + width / 2, multi_view_scores, width, label='OmniSight Multi-View (Ours)', color='#2b5c8f', alpha=0.85)

    ax.set_xlabel("Perturbation / Condition", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha='right')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig
