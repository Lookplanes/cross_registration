"""
t-SNE visualization with **center-modality highlighting**.

All non-center modalities are drawn as small gray dots in the background;
center modalities are drawn as large colored markers on top with a legend.

Public API
----------
- :func:`compute_tsne` — PCA pre-reduction → t-SNE → (N, 2) array.
- :func:`plot_tsne_highlight` — scatter + legend + optional count bar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OTHER_COLOR = "#b0b0b0"  # light gray for non-center modalities


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute_tsne(
    df: pd.DataFrame,
    feat_cols: list[str] | None = None,
    perplexity: float = 30.0,
    random_state: int = 42,
    pca_dim: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """StandardScaler → optional PCA → t-SNE → 2D embedding.

    Args:
        df: DataFrame containing feature columns.
        feat_cols: Feature column names.  Auto-detected as ``feat_*`` if None.
        perplexity: t-SNE perplexity (auto-clamped if too large).
        random_state: Random seed.
        pca_dim: PCA target dimensionality before t-SNE (0 = skip PCA).
        verbose: Print progress messages.

    Returns:
        ``(X_tsne, perp_used)`` where ``X_tsne`` is ``(N, 2)`` and
        ``perp_used`` is the actual perplexity value.
    """
    if feat_cols is None:
        feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    if not feat_cols:
        raise ValueError("No feature columns found (neither given nor feat_* in df)")

    if verbose:
        print(f"Preparing data: {len(df)} samples × {len(feat_cols)} features")

    data = df[feat_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X = StandardScaler().fit_transform(data)

    # PCA pre-reduction
    if pca_dim > 0:
        pca_dim = min(pca_dim, X.shape[1], X.shape[0] - 1)
        if pca_dim < X.shape[1]:
            if verbose:
                print(f"PCA: {X.shape[1]} → {pca_dim} dims ...")
            pca = PCA(n_components=pca_dim, random_state=random_state)
            X = pca.fit_transform(X)
            if verbose:
                print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    n = len(X)
    perp = min(perplexity, max(5, n / 3 - 1))
    if verbose:
        print(f"t-SNE: {n} samples, perplexity={perp:.0f} ...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state,
                max_iter=1000, n_jobs=1)
    X_tsne = tsne.fit_transform(X)
    return X_tsne, perp


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_tsne_highlight(
    df: pd.DataFrame,
    X_tsne: np.ndarray,
    perp: float,
    target_modalities: list[str],
    color_map: dict[str, str] | None = None,
    other_color: str = DEFAULT_OTHER_COLOR,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 6.5),
    title: str = "Modality Landscape",
) -> plt.Figure:
    """Scatter + count-bar figure with center modalities highlighted.

    **Left panel** — t-SNE scatter:
      - Non-target modalities: gray ``.`` markers (low alpha), single legend entry.
      - Target modalities: large ``o`` markers, distinct colours, black edge.

    **Right panel** — sample-count bar chart:
      - Target modalities sorted first, bold labels, distinct colours.
      - Non-target modalities sorted by count desc, gray.

    Args:
        df: DataFrame with ``modality_name`` column (and optionally ``is_center``).
        X_tsne: ``(N, 2)`` array from :func:`compute_tsne`.
        perp: Perplexity used (shown in title).
        target_modalities: Modality names to highlight.
        color_map: ``{modality_name: hex_color}`` dict for target modalities.
            Unmapped targets get ``"#333333"``.
        other_color: Color for non-target modalities.
        output_path: If given, save figure to this path.
        figsize: ``(width, height)`` in inches.
        title: Figure suptitle prefix.

    Returns:
        matplotlib Figure (not closed — caller should close or save).
    """
    if color_map is None:
        color_map = {}

    all_mods = sorted(df["modality_name"].unique())
    target_in_data = [m for m in target_modalities if m in all_mods]
    other_mods = [m for m in all_mods if m not in target_modalities]
    n_other = len(other_mods)
    n_other_samples = df["modality_name"].isin(other_mods).sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ---- Left: t-SNE scatter ----
    # Background: non-target modalities
    if other_mods:
        mask_other = df["modality_name"].isin(other_mods)
        ax1.scatter(
            X_tsne[mask_other.values, 0], X_tsne[mask_other.values, 1],
            c=other_color, marker=".", s=8, alpha=0.25,
            label=f"Other ({n_other} modalities, {n_other_samples} imgs)",
            rasterized=True,
        )

    # Foreground: target modalities
    for mod in target_in_data:
        mask = df["modality_name"] == mod
        if not mask.any():
            continue
        color = color_map.get(mod, "#333333")
        ax1.scatter(
            X_tsne[mask.values, 0], X_tsne[mask.values, 1],
            c=color, marker="o", s=30, alpha=0.85,
            edgecolors="black", linewidths=0.3,
            label=f"★ {mod}",
            zorder=5,
        )

    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    n_total = len(all_mods)
    ax1.set_title(f"{title}: {n_total} Modalities (t-SNE, perplexity={perp:.0f})")
    ax1.legend(
        fontsize=8, loc="upper left",
        bbox_to_anchor=(1.01, 1), borderaxespad=0,
        title="Center Modalities ★", title_fontsize=9,
    )

    # ---- Right: sample-count bar chart ----
    counts = df["modality_name"].value_counts()
    # Order: targets first, then others by count desc
    order = target_in_data + sorted(other_mods, key=lambda m: counts.get(m, 0), reverse=True)
    counts_ordered = counts.reindex(order)

    x_positions = range(len(order))
    # Draw non-target bars first (low alpha), then target bars on top
    non_target_mask = [m not in target_modalities for m in order]
    target_mask = [m in target_modalities for m in order]

    if any(non_target_mask):
        ax2.bar(
            [x for i, x in enumerate(x_positions) if non_target_mask[i]],
            [counts_ordered.values[i] for i in range(len(order)) if non_target_mask[i]],
            color=other_color, alpha=0.4,
        )
    if any(target_mask):
        ax2.bar(
            [x for i, x in enumerate(x_positions) if target_mask[i]],
            [counts_ordered.values[i] for i in range(len(order)) if target_mask[i]],
            color=[color_map.get(m, "#333333") for i, m in enumerate(order) if target_mask[i]],
            alpha=0.85,
        )

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(order, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("Sample count")
    ax2.set_title(f"Samples per Modality ({n_total} total)")

    # Bold target modality labels
    for i, m in enumerate(order):
        if m in target_modalities:
            ax2.get_xticklabels()[i].set_fontweight("bold")
            ax2.get_xticklabels()[i].set_fontsize(7)

    fig.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")

    return fig


# ---------------------------------------------------------------------------
# Convenience: all-in-one
# ---------------------------------------------------------------------------

def run_tsne_pipeline(
    df: pd.DataFrame,
    target_modalities: list[str],
    color_map: dict[str, str] | None = None,
    output_path: str | Path | None = None,
    perplexity: float = 30.0,
    random_state: int = 42,
    pca_dim: int = 50,
    other_color: str = DEFAULT_OTHER_COLOR,
    title: str = "Modality Landscape",
) -> plt.Figure:
    """Run the full t-SNE pipeline: compute → plot.

    Convenience wrapper that calls :func:`compute_tsne` then
    :func:`plot_tsne_highlight`.

    Returns:
        matplotlib Figure (caller should ``plt.close(fig)`` when done).
    """
    X_tsne, perp = compute_tsne(
        df, perplexity=perplexity, random_state=random_state,
        pca_dim=pca_dim, verbose=True,
    )
    fig = plot_tsne_highlight(
        df=df, X_tsne=X_tsne, perp=perp,
        target_modalities=target_modalities,
        color_map=color_map,
        other_color=other_color,
        output_path=output_path,
        title=title,
    )
    return fig
