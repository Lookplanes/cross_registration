"""
Modality distance matrix: pairwise Z-score between modality feature vectors.

  heatmap rows/cols = modality names
  cell value        = Z-score distance (modality_A vs modality_B)
  diagonal          = 0 (self-distance)

Use this to find which modalities cluster together (good translation targets)
and which are far apart (hardest to translate between).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_domain_gap(
    df: pd.DataFrame,
    core_features: list[str],
    feature_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot pairwise Z-score distance matrix between all modalities.

    Each modality's feature vector is the mean across all its images.
    Distance = Z-score of difference normalised by global std per feature.

    Args:
        df: DataFrame with ``modality_name`` and feature columns.
        core_features: feature column names to include.
        feature_labels: optional short labels for feature columns.

    Returns:
        matplotlib Figure.
    """
    modalities = sorted(df["modality_name"].unique())
    if len(modalities) < 2:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Need >= 2 modalities", ha="center", va="center")
        return fig

    # Per-modality mean feature vector
    means = {}
    for mod in modalities:
        sub = df[df["modality_name"] == mod][core_features]
        means[mod] = sub.mean()

    # Global std for Z-score normalisation
    global_std = df[core_features].std().replace(0, 1e-8)

    # Pairwise distance matrix
    n = len(modalities)
    dist = np.zeros((n, n), dtype=np.float32)
    for i, mi in enumerate(modalities):
        for j, mj in enumerate(modalities):
            if i == j:
                dist[i, j] = 0.0
            else:
                z = ((means[mi] - means[mj]) / global_std).abs().mean()
                dist[i, j] = float(z)

    labels = modalities
    fig, ax = plt.subplots(figsize=(max(5, n * 1.0), max(3.5, n * 0.7)))
    sns.heatmap(
        dist, ax=ax, xticklabels=labels, yticklabels=labels,
        annot=True, fmt=".2f", cmap="YlOrRd",
        cbar_kws={"label": "Mean Z-score distance"},
        linewidths=0.5,
    )
    ax.set_title("Modality Distance Matrix (lower = more similar)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    return fig
