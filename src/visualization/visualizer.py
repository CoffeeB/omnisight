"""
Overlay Renderer for Semantic Masks, Oriented Bounding Boxes, and Uncertainty Heatmaps
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

CLASS_COLORS = {
    0: (0, 0, 0),        # Void / Unlabeled
    1: (220, 20, 60),    # Residential Building (Crimson)
    2: (139, 0, 0),      # Warehouse (Dark Red)
    3: (255, 140, 0),    # Tower (Dark Orange)
    4: (255, 215, 0),    # Bridge (Gold)
    5: (184, 134, 11),   # Industrial Facility (Dark Goldenrod)
    6: (128, 128, 128),  # Road (Gray)
    7: (70, 70, 70),     # Highway (Dark Slate)
    8: (192, 192, 192),  # Intersection (Silver)
    9: (107, 142, 35),   # Railway (Olive Drab)
    10: (188, 143, 143), # Footpath (Rosy Brown)
    11: (34, 139, 34),   # Dense Forest (Forest Green)
    12: (50, 205, 50),   # Individual Tree (Lime Green)
    13: (154, 205, 50),  # Grassland (Yellow Green)
    14: (0, 191, 255),   # River (Deep Sky Blue)
    15: (0, 0, 205),     # Water Body (Medium Blue)
    16: (112, 128, 144), # Rock (Slate Gray)
    17: (238, 214, 175), # Sand (Warm Sand)
    18: (218, 165, 32),  # Farmland (Goldenrod)
    19: (0, 255, 255),   # Car (Cyan)
    20: (0, 139, 139),   # Truck (Dark Cyan)
    21: (255, 105, 180), # Motorcycle (Hot Pink)
    22: (75, 0, 130),    # Bus (Indigo)
    23: (255, 69, 0),    # Construction Vehicle (Orange Red)
    24: (160, 82, 45),   # Cattle / Horse (Sienna)
    25: (255, 0, 255),   # Person / Group (Magenta)
}

def render_semantic_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.55
) -> np.ndarray:
    """
    Renders alpha-blended colored semantic segmentation overlay.
    """
    h, w = image.shape[:2]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cid, color in CLASS_COLORS.items():
        if cid == 0:
            continue
        c_mask = mask == cid
        if np.any(c_mask):
            color_mask[c_mask] = color

    has_class = mask > 0
    blended = image.copy()
    blended[has_class] = cv2.addWeighted(
        image[has_class], 1.0 - alpha,
        color_mask[has_class], alpha, 0
    )
    return blended


def render_detection_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    uncertainties: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws oriented or axis-aligned bounding boxes with confidence and uncertainty badges.
    """
    canvas = image.copy()

    for i in range(len(scores)):
        score = scores[i]
        cname = class_names[i]
        box = boxes[i]

        unc_str = f" ±{uncertainties[i]:.2f}" if uncertainties is not None else ""
        label = f"{cname} {score:.2f}{unc_str}"

        if len(box) == 5:
            # Rotated bounding box: [cx, cy, w, h, angle]
            cx, cy, bw, bh, angle = box
            rrect = ((float(cx), float(cy)), (float(bw), float(bh)), float(np.degrees(angle)))
            pts = cv2.boxPoints(rrect).astype(np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            pt_label = (int(cx - bw * 0.4), int(cy))
        else:
            xmin, ymin, xmax, ymax = [int(v) for v in box[:4]]
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            pt_label = (xmin, max(15, ymin - 5))

        # Text label badge
        cv2.putText(
            canvas, label, pt_label,
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, label, pt_label,
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    return canvas


def render_uncertainty_heatmap(
    image: np.ndarray,
    uncertainty_map: np.ndarray,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Renders Jet/Turbo color-mapped uncertainty overlay.
    """
    unc_norm = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min() + 1e-8)
    heatmap = (unc_norm * 255.0).astype(np.uint8)
    colored_heat = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored_heat = cv2.cvtColor(colored_heat, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image, 1.0 - alpha, colored_heat, alpha, 0)
    return blended


def render_multi_view_grid(
    views: List[Tuple[str, np.ndarray]],
    cols: int = 3
) -> np.ndarray:
    """
    Assembles multi-view image streams into a single annotated research grid.
    """
    n = len(views)
    rows = int(np.ceil(n / cols))
    cell_h, cell_w = views[0][1].shape[:2]

    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for idx, (title, img) in enumerate(views):
        r = idx // cols
        c = idx % cols
        cell_img = cv2.resize(img, (cell_w, cell_h)) if img.shape[:2] != (cell_h, cell_w) else img
        cv2.putText(
            cell_img, title, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            cell_img, title, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
        )
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell_img

    return grid
