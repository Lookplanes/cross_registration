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
from modality_analyzer.visualize import plot_domain_gap, plot_pca_overview, plot_per_study_radar

PLOT_REGISTRY = {
    "domain_gap": ("domain_gap.png", plot_domain_gap),
    "pca": ("pca_overview.png", plot_pca_overview),
    "radar": ("per_study_radar.png", plot_per_study_radar),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate domain gap analysis figures")
    p.add_argument("--input", required=True, help="Path to features.csv")
    p.add_argument("--output-dir", default="figures", help="Output directory for PNG files")
    p.add_argument("--plots", default="domain_gap,pca,radar",
                   help="Comma-separated plot names: domain_gap,pca,radar")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    # Fill NaNs / Infs
    df[CORE_FEATURES] = df[CORE_FEATURES].fillna(0).replace([float("inf"), float("-inf")], 0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_names = [s.strip() for s in args.plots.split(",") if s.strip()]

    for name in plot_names:
        if name not in PLOT_REGISTRY:
            print(f"[SKIP] Unknown plot: {name}")
            continue
        filename, plot_fn = PLOT_REGISTRY[name]
        print(f"Generating {filename} ...", end=" ", flush=True)

        fig = plot_fn(df, CORE_FEATURES, FEAT_LABELS)
        path = out_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"-> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
