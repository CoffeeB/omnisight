"""
Altitude Scale-Space & Ground Sample Distance (GSD) Simulator
Models camera elevation shifts from 10m to 300m via Modulation Transfer Function (MTF) blur and decimation.
"""

from typing import Tuple, Optional, List
import numpy as np
import cv2

ALTITUDE_GSD_MAP = {
    10: 0.005,   # 10m -> 0.5 cm/px
    30: 0.015,   # 30m -> 1.5 cm/px
    60: 0.030,   # 60m -> 3.0 cm/px
    120: 0.060,  # 120m -> 6.0 cm/px
    300: 0.150,  # 300m -> 15.0 cm/px
}

def resample_ground_sample_distance(
    image: np.ndarray,
    target_altitude_m: float,
    base_altitude_m: float = 10.0,
    mask: Optional[np.ndarray] = None,
    boxes: Optional[List[List[float]]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[List[float]]]]:
    """
    Simulates observation from higher altitude.
    Applies anti-aliasing PSF Gaussian blur followed by downsampling and bicubic reconstruction.
    """
    if target_altitude_m <= base_altitude_m:
        return image, mask, boxes

    h, w = image.shape[:2]
    scale_factor = target_altitude_m / base_altitude_m  # e.g., 60m / 10m = 6x

    # PSF sigma calculation according to Nyquist limit
    sigma = 0.5 * np.sqrt(max(0.1, scale_factor**2 - 1.0))
    ksize = int(2 * np.ceil(2 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1

    blurred_img = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # Sub-sampled intermediate low-res resolution
    lr_w = max(4, int(w / scale_factor))
    lr_h = max(4, int(h / scale_factor))

    downsampled = cv2.resize(blurred_img, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    # Reconstructed sensor canvas representation
    reconstructed_img = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)

    return reconstructed_img, mask, boxes


class AltitudeScaleEngine:
    """
    Multi-altitude scale manager.
    """
    def __init__(self, base_altitude: float = 10.0):
        self.base_altitude = base_altitude

    def simulate(self, image: np.ndarray, altitude: float) -> np.ndarray:
        out_img, _, _ = resample_ground_sample_distance(
            image,
            target_altitude_m=altitude,
            base_altitude_m=self.base_altitude
        )
        return out_img
