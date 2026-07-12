#!/usr/bin/env python3
"""
Generate domain gap analysis figures from a features CSV.

Usage::

    python scripts/analyze_domain_gap.py \\
        --input results/features.csv \\
        --output-dir results/figures/ \\
        --plots domain_gap,pca,radar
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on PYTHONPATH
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
import pandas as pd
from modality_analyzer import CORE_FEATURES, FEAT_LABELS
from modality_analyzer.visualize import plot_domain_gap, plot_pca_overview, plot_per_study_radar, plot_tsne_overview

PLOT_REGISTRY = {
    "distance": ("distance_matrix.png", plot_domain_gap),
    "pca": ("pca_overview.png", plot_pca_overview),
    "radar": ("radar_profiles.png", plot_per_study_radar),
    "tsne": ("tsne_overview.png", plot_tsne_overview),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate domain gap analysis figures")
    p.add_argument("--input", required=True, help="Path to features.csv")
    p.add_argument("--output-dir", default="figures", help="Output directory for PNG files")
    p.add_argument("--plots", default="distance,pca,radar",
                   help="Comma-separated plot names: distance,pca,radar")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    # Auto-detect feature columns: prefer CORE_FEATURES, fall back to feat_*
    if all(c in df.columns for c in CORE_FEATURES[:3]):
        features = list(CORE_FEATURES)
        labels = dict(FEAT_LABELS)
        print(f"Detected handcrafted features: {len(features)} dims")
    else:
        features = sorted([c for c in df.columns if c.startswith("feat_")])
        labels = None
        if not features:
            print("[ERROR] No feature columns found (neither CORE_FEATURES nor feat_*)")
            return
        print(f"Detected ResNet features: {len(features)} dims")

    # Fill NaNs / Infs
    df[features] = df[features].fillna(0).replace([float("inf"), float("-inf")], 0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_names = [s.strip() for s in args.plots.split(",") if s.strip()]

    for name in plot_names:
        if name not in PLOT_REGISTRY:
            print(f"[SKIP] Unknown plot: {name}")
            continue
        filename, plot_fn = PLOT_REGISTRY[name]
        print(f"Generating {filename} ...", end=" ", flush=True)

        fig = plot_fn(df, features, labels)
        path = out_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"-> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
