"""
Geometric & Multi-View Perspective Transformations
Simulates viewpoint rotations (Front, Side, Rear, Top-down/Nadir, Oblique, Low-angle) and projective homography.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
import numpy as np
import cv2

class ViewpointType(str, Enum):
    FRONT = "front"
    SIDE = "side"
    REAR = "rear"
    TOP_DOWN = "top_down"
    OBLIQUE = "oblique"
    LOW_ANGLE = "low_angle"

@dataclass
class ViewpointConfig:
    viewpoint: ViewpointType
    pitch_deg: float = 0.0  # Elevation/tilt (-90=nadir, 0=horizontal)
    yaw_deg: float = 0.0    # Azimuth rotation (0=front, 90=side, 180=rear)
    roll_deg: float = 0.0   # Sensor roll
    fov_deg: float = 60.0   # Field of view
    focal_length_px: float = 500.0

    @classmethod
    def from_type(cls, vtype: Union[str, ViewpointType]) -> "ViewpointConfig":
        if isinstance(vtype, str):
            vtype = ViewpointType(vtype.lower())
        
        if vtype == ViewpointType.FRONT:
            return cls(viewpoint=vtype, pitch_deg=0.0, yaw_deg=0.0)
        elif vtype == ViewpointType.SIDE:
            return cls(viewpoint=vtype, pitch_deg=0.0, yaw_deg=90.0)
        elif vtype == ViewpointType.REAR:
            return cls(viewpoint=vtype, pitch_deg=0.0, yaw_deg=180.0)
        elif vtype == ViewpointType.TOP_DOWN:
            return cls(viewpoint=vtype, pitch_deg=-85.0, yaw_deg=0.0)
        elif vtype == ViewpointType.OBLIQUE:
            return cls(viewpoint=vtype, pitch_deg=-45.0, yaw_deg=30.0)
        elif vtype == ViewpointType.LOW_ANGLE:
            return cls(viewpoint=vtype, pitch_deg=20.0, yaw_deg=0.0)
        return cls(viewpoint=ViewpointType.FRONT)


def compute_homography_matrix(
    img_size: Tuple[int, int],
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float = 0.0,
    fov_deg: float = 60.0
) -> np.ndarray:
    """
    Computes 3x3 projective homography matrix representing 3D camera orientation.
    """
    h, w = img_size
    f = (w * 0.5) / np.tan(np.radians(fov_deg * 0.5))

    # Intrinsic matrix K
    K = np.array([
        [f, 0, w * 0.5],
        [0, f, h * 0.5],
        [0, 0, 1.0]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # Convert Euler angles to rotation matrix R
    rx = np.radians(pitch_deg)
    ry = np.radians(yaw_deg)
    rz = np.radians(roll_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    H = K @ R @ K_inv
    H = H / (H[2, 2] + 1e-8)
    return H


def apply_viewpoint_transformation(
    image: np.ndarray,
    config: ViewpointConfig,
    mask: Optional[np.ndarray] = None,
    boxes: Optional[List[List[float]]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[List[float]]]]:
    """
    Warps image, semantic mask, and bounding boxes to the target camera viewpoint.
    """
    h, w = image.shape[:2]
    H = compute_homography_matrix(
        (h, w),
        pitch_deg=config.pitch_deg,
        yaw_deg=config.yaw_deg,
        roll_deg=config.roll_deg,
        fov_deg=config.fov_deg
    )

    warped_img = cv2.warpPerspective(
        image, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    warped_mask = None
    if mask is not None:
        mask_u8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
        warped_mask_u8 = cv2.warpPerspective(
            mask_u8, H, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        warped_mask = warped_mask_u8.astype(np.int32)

    warped_boxes = None
    if boxes is not None:
        warped_boxes = []
        for box in boxes:
            # box: [xmin, ymin, xmax, ymax, score, class_id] or [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = box[:4]
            corners = np.array([
                [xmin, ymin, 1.0],
                [xmax, ymin, 1.0],
                [xmax, ymax, 1.0],
                [xmin, ymax, 1.0]
            ], dtype=np.float64).T

            transformed_corners = H @ corners
            transformed_corners /= (transformed_corners[2:3, :] + 1e-8)

            t_xs = transformed_corners[0, :]
            t_ys = transformed_corners[1, :]

            n_xmin = float(np.clip(np.min(t_xs), 0, w - 1))
            n_ymin = float(np.clip(np.min(t_ys), 0, h - 1))
            n_xmax = float(np.clip(np.max(t_xs), 0, w - 1))
            n_ymax = float(np.clip(np.max(t_ys), 0, h - 1))

            if (n_xmax - n_xmin) > 3 and (n_ymax - n_ymin) > 3:
                new_box = [n_xmin, n_ymin, n_xmax, n_ymax] + list(box[4:])
                warped_boxes.append(new_box)

    return warped_img, warped_mask, warped_boxes
