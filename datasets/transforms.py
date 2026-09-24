"""
Physical Degradation & Multi-View Transforms Pipeline
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from src.preprocessing.atmospheric import AtmosphericDegradationPipeline
from src.preprocessing.geometry import apply_viewpoint_transformation, ViewpointConfig
from src.preprocessing.altitude import resample_ground_sample_distance

class OmniTransformPipeline:
    """
    Applies physical perturbations (viewpoint shift, altitude GSD, occlusion, weather) on dataset samples.
    """
    def __init__(
        self,
        viewpoint_config: Optional[ViewpointConfig] = None,
        altitude_m: Optional[float] = None,
        occlusion_ratio: float = 0.0,
        atmospheric_mode: str = "clean",
        atmospheric_params: Optional[Dict[str, Any]] = None
    ):
        self.viewpoint_config = viewpoint_config
        self.altitude_m = altitude_m
        self.occlusion_ratio = occlusion_ratio
        self.atmos_engine = AtmosphericDegradationPipeline(atmospheric_mode, atmospheric_params)

    def apply_occlusion(self, image: np.ndarray, mask: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        if ratio <= 0.0:
            return image, mask

        h, w = image.shape[:2]
        occ_img = image.copy()
        occ_mask = mask.copy()

        # Occlude rectangular blocks proportional to ratio
        num_blocks = int(np.ceil(ratio * 8))
        block_w = int(w * np.sqrt(ratio) * 0.4)
        block_h = int(h * np.sqrt(ratio) * 0.4)

        for _ in range(num_blocks):
            bx = np.random.randint(0, max(1, w - block_w))
            by = np.random.randint(0, max(1, h - block_h))
            occ_img[by:by+block_h, bx:bx+block_w] = np.random.randint(20, 50, 3)
            # Mask out occluded area to void class 0
            occ_mask[by:by+block_h, bx:bx+block_w] = 0

        return occ_img, occ_mask

    def __call__(
        self,
        primary_img: np.ndarray,
        aux_imgs: List[np.ndarray],
        mask: np.ndarray,
        boxes: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
        # 1. Viewpoint transformation if specified
        if self.viewpoint_config is not None:
            primary_img, mask, _ = apply_viewpoint_transformation(
                primary_img, self.viewpoint_config, mask=mask
            )

        # 2. Altitude GSD downsampling
        if self.altitude_m is not None and self.altitude_m > 10.0:
            primary_img, mask, _ = resample_ground_sample_distance(
                primary_img, target_altitude_m=self.altitude_m, base_altitude_m=10.0, mask=mask
            )

        # 3. Artificial Occlusion
        if self.occlusion_ratio > 0.0:
            primary_img, mask = self.apply_occlusion(primary_img, mask, self.occlusion_ratio)

        # 4. Atmospheric / environmental degradation
        primary_img = self.atmos_engine(primary_img)

        return primary_img, aux_imgs, mask, boxes
