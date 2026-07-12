"""
modality_analyzer — Cross-modality feature analysis toolkit.

Purpose: compare image features across different microscopy modalities to
find which modalities are most similar (good translation targets) and which
are most different (hardest to align).

Quick reference
---------------
Handcrafted features (19-dim)::

    from modality_analyzer.features import extract_all_features
    features = extract_all_features(image_2d)  # np.ndarray -> dict

    from modality_analyzer.visualize import plot_domain_gap
    fig = plot_domain_gap(df, CORE_FEATURES, FEAT_LABELS)

ResNet deep features (512-dim)::

    from modality_analyzer.features.resnet import build_resnet_extractor, load_image
    model = build_resnet_extractor("cpu")

    from modality_analyzer.visualize.tsne_highlight import run_tsne_pipeline
    fig = run_tsne_pipeline(df, target_modalities=["Fluorescence", "H&E"], ...)

I/O helpers::

    from modality_analyzer.io import discover_images, load_checkpoint, save_checkpoint
"""

# ---------------------------------------------------------------------------
# Core feature set — consistent across all feature extractions and plots
# ---------------------------------------------------------------------------
CORE_FEATURES: list[str] = [
    "int_mean", "int_std", "int_skewness", "int_kurtosis",
    "int_dynamic_range", "int_snr",
    "hist_entropy", "hist_max_peak", "hist_active_bins",
    "glcm_contrast_mean", "glcm_dissimilarity_mean",
    "glcm_homogeneity_mean", "glcm_energy_mean", "glcm_correlation_mean",
    "grad_mean", "grad_std", "grad_p90", "edge_density",
    "freq_low_high_ratio",
]

# Short display labels for plots
FEAT_LABELS: dict[str, str] = {
    "int_mean": "Mean",
    "int_std": "Std",
    "int_skewness": "Skew",
    "int_kurtosis": "Kurt",
    "int_dynamic_range": "DynRange",
    "int_snr": "SNR",
    "hist_entropy": "Entropy",
    "hist_max_peak": "Peak",
    "hist_active_bins": "ActiveBins",
    "glcm_contrast_mean": "GLCM_Contrast",
    "glcm_dissimilarity_mean": "GLCM_Dissim",
    "glcm_homogeneity_mean": "GLCM_Homog",
    "glcm_energy_mean": "GLCM_Energy",
    "glcm_correlation_mean": "GLCM_Corr",
    "grad_mean": "GradMean",
    "grad_std": "GradStd",
    "grad_p90": "GradP90",
    "edge_density": "EdgeDens",
    "freq_low_high_ratio": "Low/HighFreq",
}
