# Academic Literature Review: Semantic Earth & Terrestrial Vision Under Degraded Modalities

**OmniSight Research Foundation**

---

## 1. Remote Sensing & Aerial Vision Foundations

### 1.1 Earth Observation Benchmarks & Multi-Task Representations
- **LoveDA (Land-Cover Dataset for Agriculture & Urban)** (Wang et al., *NeurIPS 2021*): Highlighted the domain shift challenge between rural and urban scenes, showing that standard semantic segmentation heads drop up to 18% mIoU across geographic distributions.
- **iSAID & DOTA** (Zamir et al., *CVPRW 2019*; Xia et al., *CVPR 2018*): Standardized Oriented Bounding Box (OBB) detection for overhead imagery where axis-aligned bounding boxes encompass excessive background clutter for high-aspect-ratio objects (bridges, large vehicles, towers).
- **xView & SpaceNet** (Lam et al., *arXiv 2018*; Van Etten et al., *CVPR 2018*): Pioneered fine-grained complex infrastructure classification under diverse satellite Ground Sample Distances (GSD).

---

## 2. Multi-View Geometry & Feature Fusion

### 2.1 Cross-View Attention & Bird's-Eye-View (BEV) Transforms
- **LSS (Lift, Splat, Shoot)** (Philion & Sanja, *ECCV 2020*) and **BEVFormer** (Li et al., *ECCV 2022*): Established spatiotemporal transformer cross-attention to project multi-camera perspective features onto a unified ground plane.
- **Aerial-to-Ground Cross-View Matching** (Zhai et al., *CVPR 2017*; Shi et al., *CVPR 2019*): Explored polar geometric warping to bridge the dramatic perspective mismatch between ground-level cameras and overhead aerial sensors.

---

## 3. Physical Atmospheric Models & Synthetic Degradation

### 3.1 Scattering & Haze Optics
- **Koschmieder's Law** (Koschmieder, 1924; He et al., *CVPR 2009*): The atmospheric attenuation equation $\mathbf{I}(\mathbf{x}) = \mathbf{J}(\mathbf{x}) e^{-\beta d(\mathbf{x})} + \mathbf{A}(1 - e^{-\beta d(\mathbf{x})})$ remains the gold standard in physical fog synthesis.
- **Rain & Precipitation Rendering** (Garg & Nayar, *SIGGRAPH 2006*; Yang et al., *TPAMI 2020*): Models the photometric properties of rain streaks as high-velocity light-refracting particles with directional motion blur.

---

## 4. Uncertainty Estimation & Calibration in Safety-Critical Vision

### 4.1 Calibration & Out-of-Distribution Detection
- **Temperature Scaling** (Guo et al., *ICML 2017*): Demonstrated that modern deep neural networks with batch normalization and deep residual layers are overconfident, and single-parameter Platt/temperature scaling recalibrates maximum softmax outputs without sacrificing top-1 accuracy.
- **Monte Carlo Dropout for Epistemic Uncertainty** (Gal & Ghahramani, *ICML 2016*): Formulated dropout at inference time as an approximation to variational Bayesian inference, enabling pixel-wise and bounding-box variance estimation.
- **Dirichlet Calibration** (Kull et al., *NeurIPS 2019*): Extended calibration across multi-class distributions with Dirichlet priors.

---

## 5. Synthesis & Research Gaps Addressed by OmniSight

While prior work investigates isolated factors (e.g., haze removal or OBB detection in isolation), **OmniSight provides a unified benchmark integrating camera viewpoint pitch/roll shifts, GSD altitude variations, atmospheric physics, and epistemic uncertainty quantification into a single reproducible experimental framework.**
