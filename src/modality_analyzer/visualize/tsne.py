"""
t-SNE overview: 2D scatter coloured by modality, with per-modality sample counts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def plot_tsne_overview(
    df: pd.DataFrame,
    core_features: list[str],
    feature_labels: dict[str, str] | None = None,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> plt.Figure:
    """t-SNE scatter plot coloured by modality, with sample-count bar chart.

    Args:
        df: DataFrame with ``modality_name`` and feature columns.
        core_features: feature columns to use for t-SNE.
        feature_labels: unused (kept for API compatibility).
        perplexity: t-SNE perplexity parameter.
        random_state: random seed for reproducibility.

    Returns:
        matplotlib Figure with two subplots (t-SNE scatter + sample counts).
    """
    data = df[core_features].fillna(0).replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    n_samples = len(X)
    # Adjust perplexity if it exceeds n_samples
    perp = min(perplexity, max(5, n_samples / 3 - 1))

    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state, max_iter=1000)
    X_tsne = tsne.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- t-SNE Scatter ---
    modalities = sorted(df["modality_name"].unique())
    cmap = plt.cm.tab10
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

    for i, mod in enumerate(modalities):
        mask = df["modality_name"] == mod
        ax1.scatter(
            X_tsne[mask.values, 0], X_tsne[mask.values, 1],
            c=[cmap(i % 10)], marker=markers[i % len(markers)],
            s=35, alpha=0.7, label=mod,
        )

    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.set_title(f"t-SNE: Modality Feature Space (perplexity={perp:.0f})")
    ax1.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))

    # --- Sample counts per modality ---
    counts = df["modality_name"].value_counts().reindex(modalities)
    colors = [cmap(i % 10) for i in range(len(modalities))]
    ax2.bar(range(len(modalities)), counts.values, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(modalities)))
    ax2.set_xticklabels(modalities, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Sample count")
    ax2.set_title("Samples per Modality")

    fig.tight_layout()
    return fig
