"""
Feature extraction aggregator — extracts all feature categories from a 2D image.
"""

from .intensity import extract_intensity_features
from .histogram import extract_histogram_features
from .texture import extract_glcm_features
from .gradient import extract_gradient_features
from .frequency import extract_frequency_features


def extract_all_features(img: "np.ndarray") -> dict[str, float]:  # noqa: F821
    """Extract all feature categories from a 2D grayscale image.

    Args:
        img: 2D numpy array.

    Returns:
        Flat dict of ~40 scalar features across 5 categories:
        intensity, histogram, GLCM texture, gradient, frequency.
    """
    features: dict[str, float] = {}
    features.update(extract_intensity_features(img))
    features.update(extract_histogram_features(img))
    features.update(extract_glcm_features(img))
    features.update(extract_gradient_features(img))
    features.update(extract_frequency_features(img))
    return features
