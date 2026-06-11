"""
Gradient / edge features via Sobel filter.
"""

import numpy as np
from skimage.filters import sobel


def extract_gradient_features(img: np.ndarray) -> dict[str, float]:
    """Extract gradient-based edge features.

    Args:
        img: 2D numpy array (float32 recommended).

    Returns:
        Dict with grad_mean, grad_std, grad_p90, edge_density.
    """
    img_f = img.astype(np.float32)
    grad = sobel(img_f)

    feats: dict[str, float] = {}
    feats["grad_mean"] = float(np.mean(grad))
    feats["grad_std"] = float(np.std(grad))
    feats["grad_p90"] = float(np.percentile(grad, 90))
    feats["grad_p95"] = float(np.percentile(grad, 95))

    # edge density: fraction of pixels with gradient > mean + 1*std
    edge_mask = grad > (feats["grad_mean"] + feats["grad_std"])
    feats["edge_density"] = float(np.mean(edge_mask))
    return feats
