"""
Physical Atmospheric & Environmental Degradation Models
Implements Koschmieder optical scattering, rain streak dynamics, solar shadow projection, and photon starvation.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

def apply_koschmieder_fog(
    image: np.ndarray,
    beta: float = 0.04,
    airlight: Tuple[float, float, float] = (0.85, 0.88, 0.92),
    depth_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Applies atmospheric scattering according to Koschmieder's Law:
    I(x) = J(x) * T(x) + A * (1 - T(x))
    where T(x) = exp(-beta * d(x))
    
    Args:
        image: RGB image uint8 [0, 255] or float32 [0.0, 1.0]
        beta: Extinction coefficient (fog density, typical 0.01 to 0.08)
        airlight: Atmospheric ambient airlight color vector (RGB in [0, 1])
        depth_map: Optional normalized depth map [0, 1]. If None, synthetic gradient is used.
        
    Returns:
        Fog-degraded image in same dtype and scale as input.
    """
    is_uint8 = image.dtype == np.uint8
    img_float = image.astype(np.float32) / 255.0 if is_uint8 else image.astype(np.float32)
    h, w = img_float.shape[:2]

    if depth_map is None:
        # Synthetic realistic depth gradient (foreground to horizon)
        y_coords = np.linspace(0.1, 1.0, h, dtype=np.float32)[:, None]
        depth = np.repeat(y_coords, w, axis=1)
    else:
        depth = depth_map.astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / depth.max()

    # Transmission map T(x) = exp(-beta * depth * scale)
    transmission = np.exp(-beta * depth * 30.0)
    transmission = np.clip(transmission, 0.05, 1.0)[:, :, None]

    airlight_arr = np.array(airlight, dtype=np.float32).reshape(1, 1, 3)

    degraded = img_float * transmission + airlight_arr * (1.0 - transmission)
    degraded = np.clip(degraded, 0.0, 1.0)

    return (degraded * 255.0).astype(np.uint8) if is_uint8 else degraded


def apply_rain_streaks(
    image: np.ndarray,
    intensity: float = 0.6,
    angle_deg: float = 75.0,
    streak_length: int = 25,
    streak_width: int = 1
) -> np.ndarray:
    """
    Renders dynamic rain streaks with directional motion blur and refractive attenuation.
    
    Args:
        image: RGB image uint8 [0, 255]
        intensity: Rain density factor [0.0, 1.0]
        angle_deg: Rain fall angle relative to horizontal
        streak_length: Pixel length of streaks
        streak_width: Pixel width
    """
    is_uint8 = image.dtype == np.uint8
    img_float = image.astype(np.float32) / 255.0 if is_uint8 else image.astype(np.float32)
    h, w = img_float.shape[:2]

    # Generate sparse particle noise
    density = int(h * w * 0.008 * intensity)
    noise = np.zeros((h, w), dtype=np.float32)
    if density > 0:
        ys = np.random.randint(0, h, density)
        xs = np.random.randint(0, w, density)
        noise[ys, xs] = np.random.uniform(0.6, 1.0, density)

    # Directional motion blur kernel
    kernel_size = max(3, streak_length)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    rad = np.radians(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    for i in range(-center, center + 1):
        x = int(round(center + i * cos_a))
        y = int(round(center + i * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum

    rain_layer = cv2.filter2D(noise, -1, kernel)
    rain_layer = np.clip(rain_layer * 1.8, 0.0, 1.0)[:, :, None]

    # Alpha composite rain streaks with subtle scene desaturation
    rain_color = np.array([0.9, 0.92, 0.95], dtype=np.float32).reshape(1, 1, 3)
    degraded = img_float * (1.0 - rain_layer * 0.7) + rain_color * (rain_layer * 0.7)
    degraded = np.clip(degraded, 0.0, 1.0)

    return (degraded * 255.0).astype(np.uint8) if is_uint8 else degraded


def apply_solar_shadows(
    image: np.ndarray,
    num_shadows: int = 4,
    shadow_intensity: float = 0.55
) -> np.ndarray:
    """
    Projects harsh directional polygonal shadow masks across the terrain.
    
    Args:
        image: RGB image uint8 [0, 255]
        num_shadows: Number of geometric shadow polygons
        shadow_intensity: Darkness of shadow (0.0=black, 1.0=no change)
    """
    is_uint8 = image.dtype == np.uint8
    img_float = image.astype(np.float32) / 255.0 if is_uint8 else image.astype(np.float32)
    h, w = img_float.shape[:2]

    shadow_mask = np.ones((h, w), dtype=np.float32)

    for _ in range(num_shadows):
        num_vertices = np.random.randint(4, 7)
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(w // 8, w // 3)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
        pts = []
        for a in angles:
            r = radius * np.random.uniform(0.6, 1.4)
            px = int(np.clip(cx + r * np.cos(a), 0, w - 1))
            py = int(np.clip(cy + r * np.sin(a), 0, h - 1))
            pts.append([px, py])
        pts_arr = np.array([pts], dtype=np.int32)
        poly_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(poly_mask, [np.array(pts, dtype=np.int32)], color=(1.0,))
        shadow_mask = np.minimum(shadow_mask, 1.0 - poly_mask * (1.0 - shadow_intensity))

    # Soften shadow edges slightly
    shadow_mask = cv2.GaussianBlur(shadow_mask, (11, 11), 3.0)[:, :, None]
    degraded = img_float * shadow_mask
    degraded = np.clip(degraded, 0.0, 1.0)

    return (degraded * 255.0).astype(np.uint8) if is_uint8 else degraded


def apply_sunset_lighting(image: np.ndarray) -> np.ndarray:
    """
    Applies golden hour 3200K warm solar color temperature and elevated contrast.
    """
    is_uint8 = image.dtype == np.uint8
    img_float = image.astype(np.float32) / 255.0 if is_uint8 else image.astype(np.float32)

    # Color shift: Enhance red/orange, moderate green, suppress blue
    tint = np.array([1.18, 0.95, 0.68], dtype=np.float32).reshape(1, 1, 3)
    degraded = img_float * tint
    degraded = np.clip(degraded, 0.0, 1.0)
    # Non-linear gamma curve for rich shadows
    degraded = np.power(degraded, 1.15)
    return (degraded * 255.0).astype(np.uint8) if is_uint8 else degraded


def apply_low_light_noise(
    image: np.ndarray,
    luminance_factor: float = 0.25,
    noise_sigma: float = 0.04
) -> np.ndarray:
    """
    Simulates low-light photon starvation and Poisson-Gaussian sensor noise.
    """
    is_uint8 = image.dtype == np.uint8
    img_float = image.astype(np.float32) / 255.0 if is_uint8 else image.astype(np.float32)

    dimmed = img_float * luminance_factor
    noise = np.random.normal(0.0, noise_sigma, dimmed.shape).astype(np.float32)
    degraded = np.clip(dimmed + noise, 0.0, 1.0)

    return (degraded * 255.0).astype(np.uint8) if is_uint8 else degraded


class AtmosphericDegradationPipeline:
    """
    Configurable atmospheric and environmental degradation engine.
    """
    def __init__(self, mode: str = "clean", params: Optional[Dict[str, Any]] = None):
        self.mode = mode.lower()
        self.params = params or {}

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.mode == "clean" or self.mode == "daylight":
            return image
        elif self.mode == "fog":
            beta = self.params.get("beta", 0.04)
            return apply_koschmieder_fog(image, beta=beta)
        elif self.mode == "rain":
            intensity = self.params.get("intensity", 0.65)
            return apply_rain_streaks(image, intensity=intensity)
        elif self.mode == "shadows":
            num_shadows = self.params.get("num_shadows", 4)
            return apply_solar_shadows(image, num_shadows=num_shadows)
        elif self.mode == "sunset":
            return apply_sunset_lighting(image)
        elif self.mode == "low_light":
            lum = self.params.get("luminance", 0.25)
            return apply_low_light_noise(image, luminance_factor=lum)
        else:
            return image
