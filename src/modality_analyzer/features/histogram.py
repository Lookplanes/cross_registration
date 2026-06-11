"""
Histogram shape features: entropy, peak height, active bin count.
"""

import numpy as np


def extract_histogram_features(img: np.ndarray, bins: int = 256) -> dict[str, float]:
    """Extract histogram-derived features.

    Args:
        img: 2D numpy array.
        bins: number of histogram bins.

    Returns:
        Dict with hist_entropy, hist_max_peak, hist_active_bins.
    """
    hist, _ = np.histogram(img.ravel(), bins=bins, density=True)
    hist = hist / (hist.sum() + 1e-12)

    feats: dict[str, float] = {}
    feats["hist_entropy"] = float(-np.sum(hist * np.log(hist + 1e-12)))
    feats["hist_max_peak"] = float(np.max(hist))
    feats["hist_active_bins"] = float(np.sum(hist > 0.1 * np.mean(hist)))
    return feats
