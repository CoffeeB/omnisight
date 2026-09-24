# OmniSight Research Reproduction Guide

This guide details step-by-step instructions for reproducing all experiments reported in the paper.

---

## 1. Environment Setup

### Option A: Local Python Environment
```bash
# Recommended Python 3.11 or 3.12
python3 -m venv venv
source venv/bin/activate

# Install requirements and OmniSight package
pip install -r requirements.txt
pip install -e .
```

### Option B: Docker Container
```bash
docker-compose up --build
```

---

## 2. Reproducing Benchmark Experiments

### Experiment 01: Viewpoint Invariance
Tests 6 canonical and non-canonical viewing perspectives (Front, Side, Rear, Top-down, Oblique, Low-angle):
```bash
python run_experiment.py --experiment viewpoint --samples 150 --visualize
```

### Experiment 02: Altitude Scale-Space Variation
Evaluates Ground Sample Distance (GSD) degradation across 10m, 30m, 60m, 120m, and 300m altitudes:
```bash
python run_experiment.py --experiment altitude --visualize
```

### Experiment 03: Structured Occlusion
Tests target occlusion across 0%, 10%, 25%, 50%, and 75% occlusion levels:
```bash
python run_experiment.py --experiment occlusion --visualize
```

### Experiment 04: Environmental & Atmospheric Conditions
Evaluates performance under Bright Daylight, Sunset, Koschmieder Fog, Rain Streaks, Harsh Solar Shadows, and Low Light:
```bash
python run_experiment.py --experiment environmental --visualize
```

### Experiment 05: Confidence Calibration & ECE
Validates temperature scaling, Monte Carlo dropout, Expected Calibration Error (ECE), and reliability diagrams:
```bash
python run_experiment.py --experiment calibration --visualize
```

### Complete Benchmark Suite
To execute all 5 experiments and produce publication tables and interactive dashboards:
```bash
python run_experiment.py --experiment all --output-dir ./benchmark_artifacts
```
