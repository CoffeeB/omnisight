"""
Instance Segmentation & Connected Component Mask Extraction
Extracts instance polygons and individual segment masks from semantic maps and bounding box detections.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2

@dataclass
class InstanceMaskResult:
    instance_masks: List[np.ndarray] # List of binary (H, W) masks
    instance_class_ids: List[int]
    instance_scores: List[float]
    instance_boxes: List[List[float]]


def extract_instance_masks(
    semantic_mask: np.ndarray,
    detection_boxes: np.ndarray,
    detection_classes: np.ndarray,
    detection_scores: np.ndarray,
    min_area_pixels: int = 15
) -> InstanceMaskResult:
    """
    Refines instance boundaries by intersecting bounding box regions with class-specific semantic masks.
    """
    h, w = semantic_mask.shape[:2]
    inst_masks = []
    inst_classes = []
    inst_scores = []
    inst_boxes = []

    for i in range(len(detection_scores)):
        cid = int(detection_classes[i])
        score = float(detection_scores[i])
        box = detection_boxes[i] # [cx, cy, w, h, angle] or [xmin, ymin, xmax, ymax]

        # Construct instance binary mask
        instance_canvas = np.zeros((h, w), dtype=np.uint8)

        if len(box) == 5:
            # Rotated box
            cx, cy, bw, bh, angle = box
            rrect = ((float(cx), float(cy)), (float(bw), float(bh)), float(np.degrees(angle)))
            box_pts = cv2.boxPoints(rrect).astype(np.int32)
            cv2.fillPoly(instance_canvas, [box_pts], color=(1,))
        else:
            xmin, ymin, xmax, ymax = [int(v) for v in box[:4]]
            instance_canvas[ymin:ymax, xmin:xmax] = 1

        # Intersect with semantic mask for this class
        refined_mask = np.logical_and(instance_canvas == 1, semantic_mask == cid).astype(np.uint8)

        if refined_mask.sum() >= min_area_pixels:
            inst_masks.append(refined_mask)
            inst_classes.append(cid)
            inst_scores.append(score)
            inst_boxes.append(list(box))

    return InstanceMaskResult(
        instance_masks=inst_masks,
        instance_class_ids=inst_classes,
        instance_scores=inst_scores,
        instance_boxes=inst_boxes
    )
