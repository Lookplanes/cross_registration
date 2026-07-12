"""
PCA overview: 2D scatter coloured by modality + top feature loadings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def plot_pca_overview(
    df: pd.DataFrame,
    core_features: list[str],
    feature_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """PCA scatter plot coloured by modality, with top feature loadings.

    Args:
        df: DataFrame with ``modality_name`` and feature columns.
        core_features: feature columns to use for PCA.
        feature_labels: optional short labels for loadings plot.

    Returns:
        matplotlib Figure with two subplots (scatter + loadings).
    """
    data = df[core_features].fillna(0).replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Scatter ---
    modalities = sorted(df["modality_name"].unique())
    cmap = plt.cm.tab10
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

    for i, mod in enumerate(modalities):
        mask = df["modality_name"] == mod
        ax1.scatter(
            X_pca[mask.values, 0], X_pca[mask.values, 1],
            c=[cmap(i % 10)], marker=markers[i % len(markers)],
            s=35, alpha=0.7, label=mod,
        )

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("PCA: Modality Feature Space")
    ax1.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))

    # --- Top loadings ---
    loadings = np.abs(pca.components_.T)
    top_n = min(15, len(core_features))
    top_idx = np.argsort(loadings.sum(axis=1))[-top_n:]
    labels_list = [
        feature_labels.get(core_features[i], core_features[i]) if feature_labels else core_features[i]
        for i in top_idx
    ]

    y_pos = range(len(top_idx))
    ax2.barh(y_pos, loadings[top_idx, 0], height=0.35, label="PC1", alpha=0.8)
    ax2.barh(y_pos, loadings[top_idx, 1], height=0.35, label="PC2", alpha=0.8, left=loadings[top_idx, 0])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_list, fontsize=7)
    ax2.set_xlabel("Absolute loading")
    ax2.set_title("Top Feature Contributions to PC1/PC2")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    return fig
