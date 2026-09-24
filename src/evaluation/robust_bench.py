"""
Perturbation & Robustness Degradation Benchmark Suite
"""

from typing import Dict, List, Any
import numpy as np

class RobustnessBenchmarkRunner:
    """
    Computes Robustness Degradation Index (RDI) and Relative Performance Retention.
    """
    def __init__(self, clean_benchmark_score: float):
        self.clean_score = max(1e-6, clean_benchmark_score)

    def compute_rdi(self, degraded_score: float) -> float:
        """
        RDI = 1.0 - (degraded_score / clean_score)
        Lower is better (0.0 = zero degradation).
        """
        retention = degraded_score / self.clean_score
        return float(max(0.0, 1.0 - retention))

    def summarize_perturbation_suite(
        self,
        condition_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Summarizes degradation across all environmental/viewpoint perturbations.
        """
        rdis = {}
        retentions = {}

        for condition, score in condition_scores.items():
            rdi = self.compute_rdi(score)
            rdis[condition] = rdi
            retentions[condition] = float(score / self.clean_score)

        mean_rdi = float(np.mean(list(rdis.values())))
        mean_retention = float(np.mean(list(retentions.values())))

        return {
            "clean_score": self.clean_score,
            "mean_RDI": mean_rdi,
            "mean_retention": mean_retention,
            "per_condition_RDI": rdis,
            "per_condition_retention": retentions
        }
