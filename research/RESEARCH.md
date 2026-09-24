# RESEARCH: Multi-View Semantic Terrain and Environmental Understanding Under Degraded Sensory Modalities

**OmniSight Research Group**  
*Laboratory for Robust Spatial Intelligence & Earth Observation*  
*Date: 2026-09-24*

---

## 1. Executive Summary & Vision

Modern computer vision models exhibit high benchmark performance when evaluated under canonical, distributionally aligned conditions (nadir aerial viewpoints, clear daylight atmospheric conditions, unoccluded line-of-sight). However, real-world deployment across disaster recovery, environmental forestry monitoring, infrastructure inspection, and ecological surveying frequently presents non-ideal capture geometries and atmospheric phenomena.

**OmniSight** is a research-grade software platform formulated to systematically measure, evaluate, and mitigate the performance cliff experienced by deep vision architectures when processing:
1. **Extreme Viewpoint Transformations**: Non-canonical oblique (15°–75° tilt), top-down nadir, and low-angle terrestrial perspectives.
2. **Altitude & Scale-Space Invariance**: Ground Sample Distance (GSD) shifts spanning micro-UAV altitudes ($10\text{ m}$) to medium-altitude reconnaissance ($300\text{ m}$).
3. **Structured & Random Occlusion**: Natural canopy obstruction, building shadow cut-offs, and partial structural occlusions ($10\%$ to $75\%$).
4. **Atmospheric & Illumination Attenuation**: Koschmieder optical scattering (haze/fog), dynamic rain streak convolution, solar shadow projection, and low-light photon starvation.
5. **Confidence Calibration & Uncertainty Decomposition**: Quantifying epistemic (model) and aleatoric (data) uncertainty to avoid catastrophic overconfidence under out-of-distribution (OOD) degradations.

---

## 2. Core Research Question

> **How can computer vision systems maintain accurate semantic understanding of environments when objects are viewed from unusual angles, partially occluded, or captured under degraded environmental conditions?**

---

## 3. Formal Problem Formulation & Research Hypotheses

### 3.1 Mathematical Model of Multi-View Degradation

Let an environment state be denoted as $\mathcal{S} \subset \mathbb{R}^3$, representing 3D spatial geometry and radiometric reflectance. An observation image $\mathbf{I}_v$ captured from camera viewpoint $v = (\mathbf{R}_v, \mathbf{t}_v, \mathbf{K}_v)$ at altitude $h_v$ under environmental condition $\mathbf{e} = (\beta, \mathbf{s}, \mathbf{L})$ is modeled by the physical image formation operator $\mathcal{T}$:

$$\mathbf{I}_v = \mathcal{T}(\mathcal{S}; v, h_v, \mathbf{e}) + \mathbf{\eta}$$

where:
- $\mathbf{R}_v \in SO(3), \mathbf{t}_v \in \mathbb{R}^3$ are camera extrinsic rotation and translation matrices.
- $h_v$ determines the Ground Sample Distance $\text{GSD} = \frac{h_v \cdot p}{f}$ (with pixel pitch $p$ and focal length $f$).
- $\beta$ is the atmospheric extinction coefficient governing the Koschmieder transmission map:
  $$T(\mathbf{x}) = \exp(-\beta \cdot d(\mathbf{x}))$$
  yielding the degraded observation $\mathbf{I}(\mathbf{x}) = \mathbf{J}(\mathbf{x}) T(\mathbf{x}) + \mathbf{A}(1 - T(\mathbf{x}))$, where $\mathbf{J}$ is true scene radiance and $\mathbf{A}$ is atmospheric airlight.
- $\mathbf{\eta} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ accounts for sensor noise.

### 3.2 Primary Research Hypothesis

$$\mathcal{H}_1: \quad \mathbb{E}_{v, \mathbf{e}}\left[\text{mIoU}\left(\mathcal{F}_{\text{MV}}(\{\mathbf{I}_{v_i}\}_{i=1}^V), \mathcal{S}_{\text{GT}}\right)\right] > \mathbb{E}_{v, \mathbf{e}}\left[\text{mIoU}\left(\mathcal{F}_{\text{SV}}(\mathbf{I}_{v}), \mathcal{S}_{\text{GT}}\right)\right] + \delta$$

*Hypothesis 1 (Multi-View Epipolar Cross-Attention Superiority)*: A multi-view spatial transformer $\mathcal{F}_{\text{MV}}$ that aggregates cross-view epipolar features yields statistically significant improvements ($\delta > 0.08\text{ mIoU}$) over independent single-view models $\mathcal{F}_{\text{SV}}$ when evaluated across non-canonical viewpoints and severe atmospheric attenuations.

### 3.3 Secondary Research Hypothesis (Confidence Calibration)

$$\mathcal{H}_2: \quad \text{ECE}\left(\mathcal{M}_{\text{calib}}(\mathbf{I}_{\text{degraded}})\right) < 0.05 \quad \text{while} \quad \text{ECE}\left(\mathcal{M}_{\text{uncalib}}(\mathbf{I}_{\text{degraded}})\right) > 0.20$$

*Hypothesis 2 (Calibration Integrity)*: Post-hoc temperature scaling and Monte Carlo epistemic uncertainty estimation bound the Expected Calibration Error ($\text{ECE} \le 0.05$) under out-of-distribution environmental shifts, preventing high-confidence hallucination on occluded and rain/fog obscured targets.

---

## 4. Ethical Mandate & Scope Boundaries

OmniSight strictly operates under scientific, civil, and environmental mandates:
1. **Target Identification Scope**: Identification is strictly categorical at semantic segmentation and oriented bounding box levels (e.g., `Bridge`, `Warehouse`, `Farmland`, `Cattle`, `Pedestrian (Group)`).
2. **Exclusion of Biometrics**: The platform contains **zero** biometric or facial recognition features, zero facial landmark extraction, and zero individual identification models.
3. **Exclusion of Kinetic Targeting**: No ballistic or military weapons-targeting subsystems are present.
4. **Target Applications**: Wildfire boundary mapping, flood inundation assessment, precision agro-forestry, infrastructure deterioration auditing, and ecological biodiversity tracking.

---

## 5. Experimental Protocols Overview

| Experiment ID | Title | Primary Independent Variables | Key Benchmark Metrics |
| :--- | :--- | :--- | :--- |
| **EXP-01** | Viewpoint Robustness | Viewing angles: Front, Side, Rear, Top-down (Nadir), Oblique ($45^\circ$), Low-angle ($15^\circ$) | $\text{mAP}_{50:95}$, $\text{mIoU}$, Precision, Recall |
| **EXP-02** | Altitude Scale-Space | Altitudes: $10\text{ m}, 30\text{ m}, 60\text{ m}, 120\text{ m}, 300\text{ m}$ (GSD $0.5\text{ cm}$ to $15\text{ cm}$) | Small-object $\text{AP}_S$, Boundary-IoU |
| **EXP-03** | Occlusion Degradation | Occlusion ratios: $0\%, 10\%, 25\%, 50\%, 75\%$ (Random & Structural) | Mask mAP, IoU retention curve |
| **EXP-04** | Atmospheric Conditions | Daylight, Sunset/Golden-Hour, Koschmieder Fog, Rain Streaks, Harsh Shadows, Low-light | Robustness Degradation Index (RDI) |
| **EXP-05** | Confidence Calibration | Temperature Scaling ($T$), Monte Carlo Dropout ($N=20$) | ECE, MCE, Brier Score, Reliability Curves |

---

## 6. Mathematical Definition of Evaluation Metrics

### 6.1 Expected Calibration Error (ECE)
Partition predictions into $M$ equally spaced confidence bins $B_m \subset (0, 1]$:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

where:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i), \quad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$$

### 6.2 Mean Intersection-over-Union (mIoU)
$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$

### 6.3 Robustness Degradation Index (RDI)
$$\text{RDI}_{\text{condition}} = 1.0 - \frac{\text{mIoU}(\mathbf{I}_{\text{degraded}})}{\text{mIoU}(\mathbf{I}_{\text{canonical}})}$$
