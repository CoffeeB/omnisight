import pytest
import numpy as np
from src.preprocessing.atmospheric import (
    apply_koschmieder_fog,
    apply_rain_streaks,
    apply_solar_shadows,
    apply_low_light_noise,
    apply_sunset_lighting
)
from src.preprocessing.geometry import (
    apply_viewpoint_transformation,
    ViewpointConfig,
    ViewpointType
)
from src.preprocessing.altitude import resample_ground_sample_distance

def test_atmospheric_fog():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    fog_img = apply_koschmieder_fog(img, beta=0.05)
    assert fog_img.shape == (100, 100, 3)
    assert fog_img.dtype == np.uint8
    # Fog adds ambient airlight, so image mean luminance should increase
    assert fog_img.mean() > img.mean()

def test_rain_streaks():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 100
    rain_img = apply_rain_streaks(img, intensity=0.8)
    assert rain_img.shape == (100, 100, 3)
    assert rain_img.dtype == np.uint8

def test_solar_shadows():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    shadow_img = apply_solar_shadows(img, num_shadows=3, shadow_intensity=0.4)
    assert shadow_img.shape == (100, 100, 3)
    assert shadow_img.mean() < img.mean()

def test_viewpoint_warp():
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[40:80, 40:80] = 255
    mask = np.zeros((120, 120), dtype=np.int32)
    mask[40:80, 40:80] = 1

    cfg = ViewpointConfig.from_type(ViewpointType.OBLIQUE)
    warped_img, warped_mask, _ = apply_viewpoint_transformation(img, cfg, mask=mask)
    assert warped_img.shape == (120, 120, 3)
    assert warped_mask is not None
    assert warped_mask.shape == (120, 120)

def test_altitude_resample():
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    res_img, _, _ = resample_ground_sample_distance(img, target_altitude_m=60.0, base_altitude_m=10.0)
    assert res_img.shape == (128, 128, 3)
