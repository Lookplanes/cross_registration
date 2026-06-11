"""
Intensity statistics: mean, std, percentiles, skewness, kurtosis, SNR.
"""

import numpy as np
from scipy import stats


def percentile_clip(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Clip image to [low, high] percentiles to suppress outliers."""
    p_low, p_high = np.percentile(img, [low, high])
    return np.clip(img, p_low, p_high)


def extract_intensity_features(img: np.ndarray, clip_percentile: bool = True) -> dict[str, float]:
    """Extract intensity statistics from a 2D grayscale image.

    Args:
        img: 2D numpy array.
        clip_percentile: if True, clip to [1%, 99%] before computing stats.

    Returns:
        Dict of scalar features (int_mean, int_std, int_skewness, ...).
    """
    if clip_percentile:
        img = percentile_clip(img, 1.0, 99.0)

    feats: dict[str, float] = {}
    feats["int_mean"] = float(np.mean(img))
    feats["int_std"] = float(np.std(img))
    feats["int_min"] = float(np.min(img))
    feats["int_max"] = float(np.max(img))

    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        feats[f"int_p{p:02d}"] = float(np.percentile(img, p))

    feats["int_skewness"] = float(stats.skew(img.ravel()))
    feats["int_kurtosis"] = float(stats.kurtosis(img.ravel()))
    feats["int_dynamic_range"] = feats["int_p99"] - feats["int_p01"]
    feats["int_snr"] = feats["int_mean"] / (feats["int_std"] + 1e-8)
    return feats
