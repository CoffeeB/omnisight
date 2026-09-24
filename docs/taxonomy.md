# OmniSight Semantic Taxonomy Specification

This document defines the 24 fine-grained semantic categories partitioned across 6 macro-level groupings for environmental and infrastructure understanding.

---

## 1. Taxonomic Class Hierarchy

| Class ID | Macro Group | Class Name | Description | Key Spectral / Morphological Features |
| :---: | :--- | :--- | :--- | :--- |
| **0** | Background | `Background / Void` | Unclassified background, sky, sensor void | N/A |
| **1** | Structures | `Residential Building` | Detached homes, apartments, sloped/flat roofs | Regular rectilinear geometries, pitched/shingled surfaces |
| **2** | Structures | `Warehouse` | Large-scale logistical flat-roof facilities | High surface area, metal/membrane roofing, loading docks |
| **3** | Structures | `Tower` | Telecommunications, radio, electrical pylons | High aspect ratio, lattice or steel tubular frame |
| **4** | Structures | `Bridge` | Span structures over water/valleys, viaducts | Linear elevated deck, abutments, shadow over underlying surface |
| **5** | Structures | `Industrial Facility` | Refineries, cooling towers, factories | Complex piping, cylindrical silos, metallic textures |
| **6** | Infrastructure | `Road` | Paved asphalt, municipal roadways | Continuous planar ribbons, lane markings, median dividers |
| **7** | Infrastructure | `Highway` | Multi-lane high-speed corridors, interchanges | Wide multi-lane asphalt/concrete, guardrails |
| **8** | Infrastructure | `Intersection` | Crossroads, roundabouts, junctions | Junction geometries, traffic signal poles, crosswalk stripes |
| **9** | Infrastructure | `Railway` | Ballasted steel rail tracks | Parallel line pairs, periodic sleeper cross-ties, ballast gravel |
| **10** | Infrastructure | `Footpath` | Pedestrian walkways, trails, sidewalks | Narrow winding or parallel concrete/dirt paths |
| **11** | Terrain & Flora| `Dense Forest` | Dense tree canopy cover, woodland | High texture variance, green/infrared reflectance, irregular crown |
| **12** | Terrain & Flora| `Individual Tree` | Isolated trees, urban street trees | Circular/oval canopy, cast radial shadow |
| **13** | Terrain & Flora| `Grassland / Meadow` | Open pasture, savanna, natural grass | Low-roughness planar green/yellow hue |
| **14** | Terrain & Flora| `River / Stream` | Flowing freshwater watercourses | Meandering linear/curved boundaries, specular reflection |
| **15** | Terrain & Flora| `Water Body` | Lakes, reservoirs, ponds, sea | Low reflectance in NIR, uniform absorption, shoreline contour |
| **16** | Terrain & Flora| `Exposed Rock` | Cliffs, rocky outcrops, quarries | High roughness, gray/brown mineral hues, fracture lines |
| **17** | Terrain & Flora| `Sand` | Dunes, riverbanks, arid ground | Uniform fine texture, bright yellow-beige reflectance |
| **18** | Terrain & Flora| `Farmland` | Agricultural plots, row crops | Regular quadrilateral field boundaries, periodic crop furrows |
| **19** | Vehicles | `Car` | Passenger sedans, SUVs, compact vehicles | Rectangular compact footprint, windshield glint |
| **20** | Vehicles | `Truck` | Heavy freight trucks, semi-trailers | Long articulated rectangular footprint, high shadow height |
| **21** | Vehicles | `Motorcycle` | Two-wheeled motor vehicles | Small elongated footprint |
| **22** | Vehicles | `Bus` | Transit coaches, school buses | Long single-chassis box, uniform roof |
| **23** | Vehicles | `Construction Vehicle` | Excavators, bulldozers, cranes, graders | Heavy tracks, boom arms, yellow/orange paint |
| **24** | Fauna & Herd | `Cattle / Horse` | Livestock, grazing animals | Elongated quadruped footprint, grouped in pastures |
| **25** | Pedestrians | `Person (Individual / Group)` | Standing, walking, or grouped pedestrians | Sub-meter footprint, vertical shadow projection |

---

## 2. Color Palette Representation (RGB Visualization Map)

```python
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
```
