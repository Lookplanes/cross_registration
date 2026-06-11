"""
Frequency domain features via 2D FFT.
"""

import numpy as np


def extract_frequency_features(img: np.ndarray, num_bands: int = 6) -> dict[str, float]:
    """Extract frequency-domain energy distribution features.

    Args:
        img: 2D numpy array (float32 recommended).
        num_bands: number of radial frequency bands.

    Returns:
        Dict with freq_band_{i}_energy, freq_band_{i}_ratio, freq_total_energy,
        freq_low_high_ratio.
    """
    fft = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
    power = np.abs(fft) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = np.sqrt(cy ** 2 + cx ** 2)

    feats: dict[str, float] = {}
    for i in range(1, num_bands + 1):
        r_low = max_r * (i - 1) / num_bands
        r_high = max_r * i / num_bands
        mask = (r >= r_low) & (r < r_high)
        feats[f"freq_band_{i:02d}_energy"] = float(power[mask].sum())

    total_energy = float(power.sum())
    feats["freq_total_energy"] = total_energy

    for i in range(1, num_bands + 1):
        feats[f"freq_band_{i:02d}_ratio"] = feats[f"freq_band_{i:02d}_energy"] / (total_energy + 1e-12)

    # Low vs high frequency ratio (bands 1-2 vs 5-N)
    low_energy = sum(feats[f"freq_band_{i:02d}_energy"] for i in range(1, 3))
    high_energy = sum(feats[f"freq_band_{i:02d}_energy"] for i in range(5, min(num_bands, 6) + 1))
    feats["freq_low_high_ratio"] = float(low_energy / (high_energy + 1e-12))

    return feats
