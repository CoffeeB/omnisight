# OmniSight Technical Architecture

---

## 1. System Pipeline Architecture

OmniSight is designed modularly around 6 decoupled sub-packages:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Input Imagery Modalities                │
                  │  (Multi-View Aerial/Ground Streams or Single Oblique)  │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │           src/preprocessing/ Physical Engine           │
                  │  - Koschmieder Fog Scattering                          │
                  │  - Rain Streak Directional Filtering                   │
                  │  - GSD & MTF Altitude Downsampling                     │
                  │  - Projective Homography & Pitch/Roll Warping          │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │               src/models/ Neural Engine                │
                  │  - BackboneFPN (ResNet / ConvNeXt / Swin Feature Pyramids)
                  │  - CrossViewFusion (Cross-Attention Epipolar Alignment)│
                  │  - OmniNet Multi-Task Joint Head                       │
                  └─────────────┬───────────────────────────┬──────────────┘
                                │                           │
                 ┌──────────────┴───────────┐ ┌─────────────┴────────────┐
                 ▼                          ▼ ▼                          ▼
    ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
    │   src/detection/          │ │   src/segmentation/       │ │   src/confidence/         │
    │   - Oriented BBox (OBB)   │ │   - Semantic Terrain Mask │ │   - Epistemic MC Dropout  │
    │   - Axis-Aligned Bounding │ │   - Boundary-Aware Loss   │ │   - Temperature Scaling   │
    │   - Rotated Soft-NMS      │ │   - Multi-Scale Seg Head  │ │   - ECE / MCE Estimators  │
    └─────────────┬─────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘
                  │                             │                             │
                  └───────────────────────┬─────┴─────────────────────────────┘
                                          │
                                          ▼
                  ┌────────────────────────────────────────────────────────┐
                  │           src/evaluation/ & visualization/             │
                  │  - mAP@50:95 & mIoU Evaluation Engine                  │
                  │  - Plotly & Matplotlib Interactive Dashboards          │
                  │  - Reliability Diagrams & Degradation Curves           │
                  └────────────────────────────────────────────────────────┘
```

---

## 2. Key Mathematical Components

### 2.1 Multi-View Epipolar Spatial Transformer
Given $V$ views with feature maps $\{\mathbf{F}_v\}_{v=1}^V$, where $\mathbf{F}_v \in \mathbb{R}^{C \times H \times W}$, the query points in reference ground frame $\mathbf{q} \in \mathbb{R}^3$ are projected onto view $v$ via projection matrix $\mathbf{P}_v = \mathbf{K}_v [\mathbf{R}_v | \mathbf{t}_v]$:

$$\mathbf{p}_{v, \mathbf{q}} = \mathbf{P}_v \tilde{\mathbf{q}}$$

The fused feature representation $\mathbf{Z}(\mathbf{q})$ is aggregated via scaled dot-product multi-head attention:

$$\mathbf{Z}(\mathbf{q}) = \sum_{v=1}^V \text{Softmax}\left(\frac{\mathbf{Q}(\mathbf{q}) \mathbf{K}(\mathbf{F}_v(\mathbf{p}_{v, \mathbf{q}}))^\top}{\sqrt{d_k}}\right) \mathbf{V}(\mathbf{F}_v(\mathbf{p}_{v, \mathbf{q}}))$$

### 2.2 Rotated Bounding Box Representation
Oriented Bounding Boxes are parameterized as 5-tuples:

$$\mathbf{b} = (c_x, c_y, w, h, \theta) \quad \text{where } \theta \in [-\pi/2, \pi/2)$$

Smooth L1 loss coupled with IoU loss ensures stable angular convergence.
