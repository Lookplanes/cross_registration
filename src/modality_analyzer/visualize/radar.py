"""
Per-modality radar chart: one profile line per modality across core features.

All modalities are plotted on the same radar for direct visual comparison.
If there are many modalities, they are split across subplots (max 6 per plot).
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
    """Radar chart showing per-modality feature profiles.

    Features are normalised to [0, 1] globally for comparability.

    Args:
        df: DataFrame with ``modality_name`` and feature columns.
        core_features: feature columns (max 10 recommended).
        feature_labels: optional short display labels.
        max_studies: max modalities per subplot (unused; kept for API compat).

    Returns:
        matplotlib Figure with one radar showing all modalities.
    """
    features = core_features[:10]
    labels = [feature_labels.get(f, f) if feature_labels else f for f in features]
    modalities = sorted(df["modality_name"].unique())

    # Normalise each feature to [0,1] globally
    norm_df = df.copy()
    for f in features:
        fmin, fmax = df[f].min(), df[f].max()
        if fmax - fmin > 1e-8:
            norm_df[f] = (df[f] - fmin) / (fmax - fmin)
        else:
            norm_df[f] = 0.0

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for i, mod in enumerate(modalities):
        sub = norm_df[norm_df["modality_name"] == mod]
        if sub.empty:
            continue
        vals = sub[features].mean().tolist()
        vals += vals[:1]
        color = cmap(i % 10)
        ax.fill(angles, vals, alpha=0.05, color=color)
        ax.plot(angles, vals, linewidth=2, color=color, label=mod)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("Modality Feature Profiles (normalised)", fontsize=12, pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    return fig
