"""
Domain gap heatmap: Z-score distance between Hub and Source channels per study.

  heatmap rows  = (Study, Source channel)
  heatmap cols  = feature names
  cell value    = (Source_mean - Hub_mean) / Hub_std
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
    """Plot a Z-score heatmap of Hub→Source domain gap per study.

    Args:
        df: DataFrame with columns ``study``, ``channel_type``, ``channel_name``,
            and all feature columns in ``core_features``.
        core_features: list of feature column names to include.
        feature_labels: optional mapping feature_name → display label.

    Returns:
        matplotlib Figure.
    """
    studies = sorted(df["study"].unique())
    rows_data: list[list[float]] = []
    row_labels: list[str] = []

    for study in studies:
        sub = df[df["study"] == study]
        hub = sub[sub["channel_type"] == "hub"]
        if hub.empty:
            continue
        hub_mean = hub[core_features].mean()
        hub_std = hub[core_features].std().replace(0, 1e-8)

        for src_name in sorted(sub[sub["channel_type"] == "source"]["channel_name"].unique()):
            src = sub[(sub["channel_type"] == "source") & (sub["channel_name"] == src_name)]
            if src.empty:
                continue
            src_mean = src[core_features].mean()
            z = ((src_mean - hub_mean) / hub_std).tolist()
            rows_data.append(z)
            row_labels.append(f"{study[:12]} | {src_name}")

    if not rows_data:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # Label columns
    labels = [feature_labels.get(f, f) if feature_labels else f for f in core_features]

    z_arr = np.array(rows_data, dtype=np.float32)
    vmax = max(abs(z_arr).max(), 0.5)

    fig, ax = plt.subplots(figsize=(max(12, len(core_features) * 0.7), max(4, len(row_labels) * 0.35)))
    sns.heatmap(
        z_arr, ax=ax, xticklabels=labels, yticklabels=row_labels,
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        cbar_kws={"label": "Z-score (Source vs Hub)"},
        linewidths=0.5,
    )
    ax.set_title("Domain Gap: Hub → Source Feature Distance")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Study | Source Channel")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    return fig
