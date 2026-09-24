"""
High-Fidelity Synthetic Multi-View Scene Generator
Generates realistic multi-view terrain, structures, infrastructure, vehicles, and pedestrians with full ground truth.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from .base_loader import SceneSample
from src.preprocessing.geometry import apply_viewpoint_transformation, ViewpointConfig, ViewpointType
from src.visualization.visualizer import CLASS_COLORS

CLASS_NAMES = [
    "Background",
    "Residential Building",
    "Warehouse",
    "Tower",
    "Bridge",
    "Industrial Facility",
    "Road",
    "Highway",
    "Intersection",
    "Railway",
    "Footpath",
    "Dense Forest",
    "Individual Tree",
    "Grassland",
    "River",
    "Water Body",
    "Rock",
    "Sand",
    "Farmland",
    "Car",
    "Truck",
    "Motorcycle",
    "Bus",
    "Construction Vehicle",
    "Cattle / Horse",
    "Person (Group)",
]

class SyntheticSceneGenerator:
    """
    Synthesizes rich aerial/ground scenes containing diverse terrain textures and structured foreground objects.
    """
    def __init__(self, img_size: Tuple[int, int] = (512, 512), seed: Optional[int] = None):
        self.img_size = img_size
        if seed is not None:
            np.random.seed(seed)

    def generate_scene(self) -> SceneSample:
        h, w = self.img_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.int32)
        boxes_list = []
        classes_list = []

        # 1. Base Terrain Synthesis: Grassland / Farmland / Water / Forest
        terrain_type = np.random.choice(["rural", "urban_edge", "river_crossing", "coastal"])
        
        # Base background color
        if terrain_type == "rural":
            img[:] = [70, 140, 50] # Greenish grass
            mask[:] = 13 # Grassland
            # Add farmland patches
            for _ in range(np.random.randint(1, 3)):
                fw = max(20, int(w * np.random.uniform(0.2, 0.35)))
                fh = max(20, int(h * np.random.uniform(0.2, 0.35)))
                fx = np.random.randint(0, max(1, w - fw))
                fy = np.random.randint(0, max(1, h - fh))
                img[fy:fy+fh, fx:fx+fw] = [180, 150, 60] + np.random.randint(-15, 15, 3)
                mask[fy:fy+fh, fx:fx+fw] = 18 # Farmland
        elif terrain_type == "river_crossing":
            img[:] = [80, 150, 60]
            mask[:] = 13 # Grassland
            # Draw meandering river
            river_pts = np.array([
                [0, int(h * 0.4)], [int(w * 0.3), int(h * 0.45)],
                [int(w * 0.7), int(h * 0.35)], [w, int(h * 0.5)]
            ], dtype=np.int32)
            cv2.polylines(img, [river_pts], False, (180, 100, 20), max(8, int(h * 0.1)))
            cv2.polylines(mask, [river_pts], False, 14, max(8, int(h * 0.1))) # River
        else:
            img[:] = [120, 120, 120]
            mask[:] = 6 # Road / Urban ground

        # 2. Roads & Infrastructure
        road_y = int(h * 0.65)
        road_thick = max(6, int(h * 0.08))
        cv2.line(img, (0, road_y), (w, road_y), (60, 60, 60), road_thick)
        cv2.line(mask, (0, road_y), (w, road_y), 6, road_thick) # Road

        # 3. Structural Objects (Buildings / Warehouses / Bridges)
        num_buildings = np.random.randint(1, 3)
        for _ in range(num_buildings):
            bw = max(16, int(w * np.random.uniform(0.12, 0.25)))
            bh = max(16, int(h * np.random.uniform(0.12, 0.22)))
            bx = np.random.randint(4, max(5, w - bw - 4))
            by = np.random.randint(4, max(5, int(h * 0.5) - bh))
            angle = float(np.random.uniform(-0.4, 0.4))
            cid = int(np.random.choice([1, 2, 5])) # Residential, Warehouse, Industrial

            # Rotated rect
            rrect = ((float(bx + bw/2), float(by + bh/2)), (float(bw), float(bh)), float(np.degrees(angle)))
            pts = cv2.boxPoints(rrect).astype(np.int32)

            color = CLASS_COLORS[cid]
            cv2.fillPoly(img, [pts], color)
            cv2.fillPoly(mask, [pts], int(cid))

            boxes_list.append([bx + bw/2, by + bh/2, bw, bh, angle])
            classes_list.append(cid)

        # 4. Vehicles on Road
        num_vehicles = np.random.randint(1, 4)
        for _ in range(num_vehicles):
            vw = max(8, int(w * np.random.uniform(0.06, 0.12)))
            vh = max(5, int(h * np.random.uniform(0.04, 0.07)))
            vx = np.random.randint(4, max(5, w - vw - 4))
            vy = road_y + np.random.randint(-2, 3)
            v_cid = int(np.random.choice([19, 20, 22])) # Car, Truck, Bus
            v_angle = 0.0

            rrect = ((float(vx), float(vy)), (float(vw), float(vh)), 0.0)
            pts = cv2.boxPoints(rrect).astype(np.int32)
            cv2.fillPoly(img, [pts], CLASS_COLORS[v_cid])
            cv2.fillPoly(mask, [pts], int(v_cid))

            boxes_list.append([vx, vy, vw, vh, v_angle])
            classes_list.append(v_cid)

        # 5. Pedestrians & Livestock
        num_ped = np.random.randint(1, 3)
        for _ in range(num_ped):
            px = np.random.randint(4, max(5, w - 8))
            py = np.random.randint(max(6, int(h * 0.75)), max(7, h - 6))
            pw, ph = 4, 4
            p_cid = int(np.random.choice([24, 25])) # Cattle or Person

            cv2.circle(img, (px, py), 3, CLASS_COLORS[p_cid], -1)
            cv2.circle(mask, (px, py), 3, int(p_cid), -1)

            boxes_list.append([px, py, pw, ph, 0.0])
            classes_list.append(p_cid)

        # 6. Synthesize Auxiliary Views (e.g. Side, Oblique, Top-down)
        aux_views = []
        for vtype in [ViewpointType.SIDE, ViewpointType.OBLIQUE]:
            cfg = ViewpointConfig.from_type(vtype)
            aux_img, _, _ = apply_viewpoint_transformation(img, cfg)
            aux_views.append(aux_img)

        return SceneSample(
            primary_image=img,
            auxiliary_views=aux_views,
            semantic_mask=mask,
            detection_boxes=np.array(boxes_list, dtype=np.float32) if boxes_list else np.zeros((0, 5), dtype=np.float32),
            detection_classes=np.array(classes_list, dtype=np.int32) if classes_list else np.zeros(0, dtype=np.int32),
            metadata={"terrain_type": terrain_type, "resolution": self.img_size}
        )


def generate_benchmark_dataset(num_samples: int = 50, img_size: Tuple[int, int] = (256, 256)) -> List[SceneSample]:
    gen = SyntheticSceneGenerator(img_size=img_size, seed=42)
    return [gen.generate_scene() for _ in range(num_samples)]
