"""
Per-study radar chart: Hub vs each Source channel across core features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_per_study_radar(
    df: pd.DataFrame,
    core_features: list[str],
    feature_labels: dict[str, str] | None = None,
    max_studies: int = 6,
) -> plt.Figure:
    """Radar chart showing Hub vs Source feature profiles per study.

    Features are normalised to [0, 1] across all rows for comparability.

    Args:
        df: DataFrame with ``study``, ``channel_type``, ``channel_name``.
        core_features: feature columns to include (max 10 recommended).
        feature_labels: optional short display labels.
        max_studies: max studies to plot (one subplot per study).

    Returns:
        matplotlib Figure with one radar subplot per study.
    """
    features = core_features[:10]  # radar gets cluttered beyond ~10
    labels = [feature_labels.get(f, f) if feature_labels else f for f in features]
    studies = sorted(df["study"].unique())[:max_studies]

    n_cols = min(3, len(studies))
    n_rows = int(np.ceil(len(studies) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows),
                             subplot_kw={"projection": "polar"}, squeeze=False)

    # Normalise each feature to [0,1] globally
    norm_df = df.copy()
    for f in features:
        fmin, fmax = df[f].min(), df[f].max()
        if fmax - fmin > 1e-8:
            norm_df[f] = (df[f] - fmin) / (fmax - fmin)
        else:
            norm_df[f] = 0.0

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    for idx, study in enumerate(studies):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = norm_df[norm_df["study"] == study]

        hub = sub[sub["channel_type"] == "hub"]
        if not hub.empty:
            vals = hub[features].mean().tolist()
            vals += vals[:1]
            ax.fill(angles, vals, alpha=0.2, label="Hub")
            ax.plot(angles, vals, linewidth=2, label="Hub")

        for src_name in sorted(sub[sub["channel_type"] == "source"]["channel_name"].unique()):
            src = sub[(sub["channel_type"] == "source") & (sub["channel_name"] == src_name)]
            if src.empty:
                continue
            vals = src[features].mean().tolist()
            vals += vals[:1]
            ax.fill(angles, vals, alpha=0.1)
            ax.plot(angles, vals, linewidth=1, linestyle="--", label=src_name)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(study[:30], fontsize=9)
        ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Hide empty subplots
    for idx in range(len(studies), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Hub vs Source Feature Profiles (normalised)", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig
