"""
Post-Hoc Probability Calibration and Uncertainty Estimators
"""

from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TemperatureScaler(nn.Module):
    """
    Platt Scaling / Single-Parameter Temperature Scaler (Guo et al., ICML 2017).
    Optimizes T* > 0 via Negative Log-Likelihood on a validation set.
    """
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scales logits by 1/T.
        """
        temp = torch.clamp(self.temperature, min=0.01)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100) -> float:
        """
        Optimizes temperature parameter using L-BFGS or Adam.
        """
        self.temperature.requires_grad = True
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature.requires_grad = False
        return float(self.temperature.item())


class MonteCarloDropoutCalibrator:
    """
    Monte Carlo Epistemic Uncertainty Estimator (Gal & Ghahramani, ICML 2016).
    """
    def __init__(self, model: nn.Module, num_samples: int = 20):
        self.model = model
        self.num_samples = num_samples

    def predict_with_uncertainty(
        self,
        image_tensor: torch.Tensor,
        aux_tensors: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes N stochastic forward passes.
        
        Returns:
            mean_probs: (B, C, H, W)
            epistemic_var: (B, 1, H, W)
            predictive_entropy: (B, 1, H, W)
        """
        self.model.eval()
        # Enable dropout layers while keeping batchnorm in eval mode
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        probs_samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                out = self.model(image_tensor, aux_images=aux_tensors)
                probs = F.softmax(out.seg_logits, dim=1)
                probs_samples.append(probs)

        # (N, B, C, H, W)
        stacked = torch.stack(probs_samples, dim=0)
        mean_probs = stacked.mean(dim=0)
        epistemic_var = stacked.var(dim=0).mean(dim=1, keepdim=True)
        # Predictive entropy: -sum(p * log p)
        pred_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1, keepdim=True)

        return mean_probs, epistemic_var, pred_entropy


class CalibratedInferenceEngine:
    """
    Unified calibrated predictor for production deployment and benchmarking.
    """
    def __init__(self, model: nn.Module, temperature: float = 1.35):
        self.model = model
        self.scaler = TemperatureScaler(init_temperature=temperature)

    def calibrate_probabilities(self, logits: np.ndarray) -> np.ndarray:
        logits_t = torch.from_numpy(logits)
        scaled_logits = self.scaler(logits_t)
        probs = F.softmax(scaled_logits, dim=-1).numpy()
        return probs
