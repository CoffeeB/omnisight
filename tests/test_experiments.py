import pytest
import os
import shutil
from experiments.exp01_viewpoint import run_viewpoint_experiment
from experiments.exp02_altitude import run_altitude_experiment
from experiments.exp03_occlusion import run_occlusion_experiment
from experiments.exp04_environmental import run_environmental_experiment
from experiments.exp05_calibration import run_calibration_experiment

@pytest.fixture
def temp_output_dir(tmp_path):
    out = tmp_path / "test_exp_out"
    out.mkdir()
    return str(out)

def test_run_viewpoint_exp(temp_output_dir):
    res = run_viewpoint_experiment(num_samples=1, output_dir=temp_output_dir, visualize=False, viewpoints=["front", "oblique"])
    assert "single_view" in res
    assert "multi_view" in res
    assert "front" in res["single_view"]

def test_run_altitude_exp(temp_output_dir):
    res = run_altitude_experiment(num_samples=1, output_dir=temp_output_dir, visualize=False, altitudes=[10, 60])
    assert "altitude_results" in res
    assert "10" in res["altitude_results"]

def test_run_occlusion_exp(temp_output_dir):
    res = run_occlusion_experiment(num_samples=1, output_dir=temp_output_dir, visualize=False, occlusion_ratios=[0.0, 0.5])
    assert "occlusion_results" in res
    assert "0.0" in res["occlusion_results"]

def test_run_environmental_exp(temp_output_dir):
    res = run_environmental_experiment(num_samples=1, output_dir=temp_output_dir, visualize=False, conditions=["daylight", "fog"])
    assert "environmental_results" in res

def test_run_calibration_exp(temp_output_dir):
    res = run_calibration_experiment(num_samples=1, output_dir=temp_output_dir, visualize=False)
    assert "uncalibrated" in res
    assert "calibrated" in res
    assert res["calibrated"]["ECE"] < res["uncalibrated"]["ECE"]
