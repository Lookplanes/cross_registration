"""
GLCM texture features: contrast, dissimilarity, homogeneity, energy, correlation.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_glcm_features(
    img: np.ndarray,
    levels: int = 64,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
) -> dict[str, float]:
    """Extract GLCM (Gray-Level Co-occurrence Matrix) texture features.

    Args:
        img: 2D numpy array.
        levels: quantization levels for GLCM.
        distances: pixel pair distances (default [1, 3]).
        angles: radian angles (default 0, π/4, π/2, 3π/4).

    Returns:
        Dict with glcm_{property}_mean and glcm_{property}_std.
    """
    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Quantize to [0, levels-1]
    from .intensity import percentile_clip
    img_c = percentile_clip(img, 1.0, 99.0)
    vmin, vmax = img_c.min(), img_c.max()
    img_q = np.floor((img_c - vmin) / (vmax - vmin + 1e-8) * (levels - 1)).astype(np.uint8)

    glcm = graycomatrix(img_q, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)

    feats: dict[str, float] = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        vals = graycoprops(glcm, prop).ravel()
        feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"glcm_{prop}_std"] = float(np.std(vals))
    return feats
