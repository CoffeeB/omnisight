# Formal Research Experiment Protocols (01 – 05)

**OmniSight Benchmark Suite**

This document establishes the exact experimental setup, mathematical perturbations, control groups, and quantitative metrics for all 5 OmniSight research experiments.

---

## 🔬 Experiment 01: Viewpoint Robustness

### Objective
Measure semantic segmentation (mIoU) and oriented object detection (mAP) degradation when environments and objects are observed across 6 distinct spatial viewing orientations.

### Viewpoint Configurations
1. **Front** ($\theta = 0^\circ, \phi = 0^\circ$): Canonical horizontal ground-level view.
2. **Side** ($\theta = 90^\circ, \phi = 0^\circ$): Lateral ground perspective.
3. **Rear** ($\theta = 180^\circ, \phi = 0^\circ$): Back perspective.
4. **Top-Down (Nadir)** ($\phi = 90^\circ$): Pure overhead aerial sensor.
5. **Oblique** ($\phi = 45^\circ$): $45^\circ$ tilted airborne UAV perspective.
6. **Low-Angle** ($\phi = -20^\circ$ to $-30^\circ$): Upward tilted perspective from ground level.

### Metrics Computed
- $\text{mAP}_{50}$ and $\text{mAP}_{50:95}$ (COCO criteria)
- $\text{mIoU}$ (mean Intersection-over-Union across 24 classes)
- Class-wise precision, recall, and F1-score
- Mean Prediction Confidence per viewpoint

---

## 🔬 Experiment 02: Altitude Variation & Scale-Space Invariance

### Objective
Analyze model degradation as camera altitude $h$ varies over an order of magnitude, altering the Ground Sample Distance ($\text{GSD}$).

### Altitude Levels
- **$10\text{ m}$**: Micro-UAV altitude ($\text{GSD} \approx 0.5\text{ cm/pixel}$) - High structural and fine texture resolution.
- **$30\text{ m}$**: Low-altitude inspection ($\text{GSD} \approx 1.5\text{ cm/pixel}$).
- **$60\text{ m}$**: Standard commercial UAV survey ($\text{GSD} \approx 3.0\text{ cm/pixel}$).
- **$120\text{ m}$**: High-altitude drone mapping ($\text{GSD} \approx 6.0\text{ cm/pixel}$).
- **$300\text{ m}$**: Aerial reconnaissance ($\text{GSD} \approx 15.0\text{ cm/pixel}$) - Nyquist limit challenges for small animals and pedestrians.

### Mathematical Formulation
The effective image resolution is modulated via Modulation Transfer Function (MTF) point spread function blur:

$$\mathbf{I}_{h} = (\mathbf{I}_0 * \mathcal{G}_{\sigma(h)}) \downarrow_{s(h)}$$

where scale factor $s(h) = \frac{h}{h_0}$ and Gaussian blur standard deviation $\sigma(h) = 0.5 \cdot \sqrt{s(h)^2 - 1}$.

---

## 🔬 Experiment 03: Structured & Random Occlusion

### Objective
Quantify model resilience when targets are partially obscured by intervening obstacles (tree canopies, structural beams, smoke clouds).

### Occlusion Levels
- **$0\%$**: Baseline unoccluded reference.
- **$10\%$**: Light occlusion (e.g., foliage fringe).
- **$25\%$**: Moderate occlusion (e.g., parking beneath light canopy).
- **$50\%$**: Severe occlusion (half of the target masked by building shadow or barrier).
- **$75\%$**: Extreme occlusion (only partial corner or roof visible).

### Occlusion Modes
1. **Random Patch Masking**: Uniformly distributed masking blocks.
2. **Structured Cutout**: Contiguous geometric occlusion representing physical walls/structures.

---

## 🔬 Experiment 04: Environmental & Atmospheric Degradation

### Objective
Measure model invariance across varying optical, illumination, and atmospheric weather states.

### Atmospheric Profiles
1. **Bright Daylight**: $6500\text{K}$ solar color temperature, high contrast, clear air ($\beta = 0.001$).
2. **Sunset / Golden-Hour**: $3200\text{K}$ warm solar angle, elongated shadows, high dynamic range.
3. **Koschmieder Fog**: Dense optical scattering with extinction coefficient $\beta = 0.05$, atmospheric airlight $\mathbf{A} = [0.8, 0.8, 0.85]$.
4. **Rain Streaks**: Directional velocity streaks with alpha blending and photometric refraction.
5. **Harsh Solar Shadows**: High-contrast directional shadow polygons obscuring ground radiometric features.
6. **Low Light / Dawn**: Photon starvation, low SNR, Rayleigh noise amplification.

---

## 🔬 Experiment 05: Confidence Calibration & Uncertainty Quantification

### Objective
Verify that predictive confidences match empirical posterior accuracies, preventing hazardous overconfidence under degraded inputs.

### Calibration Framework
1. **Uncalibrated Baseline**: Standard Softmax outputs $\hat{p}_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$.
2. **Temperature-Scaled Model**: Optimized parameter $T^* > 0$ such that $\hat{p}_i(T) = \frac{\exp(z_i / T^*)}{\sum_j \exp(z_j / T^*)}$.
3. **Monte Carlo Epistemic Uncertainty**: $N=20$ forward passes under stochastic dropout, computing predictive entropy $\mathcal{H}(y|\mathbf{x})$ and variance $\sigma^2_{\text{epistemic}}$.

### Metrics
- **Expected Calibration Error (ECE)** (15 equal-width bins)
- **Maximum Calibration Error (MCE)**
- **Brier Score** $\text{BS} = \frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K (\hat{p}_{ik} - y_{ik})^2$
- **Reliability Diagrams** (Accuracy vs. Confidence plots)
