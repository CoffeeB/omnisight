"""
Non-Maximum Suppression (Axis-Aligned and Rotated Bounding Boxes)
Provides fast Python implementations with seamless fallback to C++20 acceleration kernels.
"""

from typing import List
import numpy as np
import cv2

# Optional C++ acceleration import
try:
    import omnisight_accel  # type: ignore
    HAS_CPP_ACCEL = True
except ImportError:
    HAS_CPP_ACCEL = False


def compute_iou_axis_aligned_py(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ixmin = max(box_a[0], box_b[0])
    iymin = max(box_a[1], box_b[1])
    ixmax = min(box_a[2], box_b[2])
    iymax = min(box_a[3], box_b[3])

    iw = max(0.0, ixmax - ixmin)
    ih = max(0.0, iymax - iymin)
    inter_area = iw * ih

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    if union_area <= 1e-6:
        return 0.0
    return inter_area / union_area


def rotated_box_to_corners(box: np.ndarray) -> np.ndarray:
    """
    Converts [cx, cy, w, h, angle_rad] to 4 corner points (4, 2).
    """
    cx, cy, w, h, angle = box
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    hw, hh = w * 0.5, h * 0.5

    # 4 local corners
    local_pts = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh]
    ], dtype=np.float32)

    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated = local_pts @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


def compute_rotated_iou_py(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Computes Intersection-over-Union between two rotated boxes via OpenCV polygon clipping.
    """
    corners_a = rotated_box_to_corners(box_a)
    corners_b = rotated_box_to_corners(box_b)

    # Use cv2.rotatedRectangleIntersection
    rrect_a = ((float(box_a[0]), float(box_a[1])), (float(box_a[2]), float(box_a[3])), float(np.degrees(box_a[4])))
    rrect_b = ((float(box_b[0]), float(box_b[1])), (float(box_b[2]), float(box_b[3])), float(np.degrees(box_b[4])))

    ret, inter_pts = cv2.rotatedRectangleIntersection(rrect_a, rrect_b)
    if ret == cv2.INTERSECT_NONE or inter_pts is None:
        return 0.0

    inter_area = cv2.contourArea(inter_pts)
    area_a = float(box_a[2] * box_a[3])
    area_b = float(box_b[2] * box_b[3])
    union_area = area_a + area_b - inter_area
    if union_area <= 1e-6:
        return 0.0
    return float(np.clip(inter_area / union_area, 0.0, 1.0))


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
    max_output: int = 300
) -> List[int]:
    """
    Standard Axis-Aligned Non-Maximum Suppression.
    """
    if len(boxes) == 0:
        return []

    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(order)):
        curr_idx = order[i]
        if suppressed[curr_idx]:
            continue
        keep.append(int(curr_idx))
        if len(keep) >= max_output:
            break

        for j in range(i + 1, len(order)):
            next_idx = order[j]
            if suppressed[next_idx]:
                continue
            iou = compute_iou_axis_aligned_py(boxes[curr_idx], boxes[next_idx])
            if iou >= iou_thresh:
                suppressed[next_idx] = True

    return keep


def rotated_non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.45,
    max_output: int = 300
) -> List[int]:
    """
    Rotated Oriented Bounding Box Non-Maximum Suppression.
    """
    if len(boxes) == 0:
        return []

    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(order)):
        curr_idx = order[i]
        if suppressed[curr_idx]:
            continue
        keep.append(int(curr_idx))
        if len(keep) >= max_output:
            break

        for j in range(i + 1, len(order)):
            next_idx = order[j]
            if suppressed[next_idx]:
                continue
            iou = compute_rotated_iou_py(boxes[curr_idx], boxes[next_idx])
            if iou >= iou_thresh:
                suppressed[next_idx] = True

    return keep
