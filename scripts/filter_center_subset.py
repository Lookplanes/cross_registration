#!/usr/bin/env python3
"""
Filter the full modality landscape to a specific 4-modality center subset
and re-generate t-SNE visualization.

Kept modalities:
  - Fluorescence
  - Confocal → Conf-Tubulin (only channel-0 / tubulin, p%4==0)
  - H&E
  - MSI

Usage::

    python scripts/filter_center_subset.py \
        --input results/figures_full/all_resnet_features.csv \
        --output-dir results/figures_center4
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modality_analyzer.visualize.tsne_highlight import run_tsne_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_MODALITIES = ["Fluorescence", "Conf-Tubulin", "H&E", "MSI"]
CENTER_COLORS: dict[str, str] = {
    "Fluorescence":  "#2ca02c",
    "Conf-Tubulin":  "#ff7f0e",
    "H&E":           "#d62728",
    "MSI":           "#8c564b",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter to 4-modality center subset and generate t-SNE"
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="results/figures_center4")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--pca-dim", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Confocal Tubulin filter
# ---------------------------------------------------------------------------
def _is_tubulin_channel(path: str) -> bool:
    """Check if a confocal image path belongs to Tubulin channel (p%4==0)."""
    m = re.search(r"_p(\d+)\.png$", path)
    if m:
        return int(m.group(1)) % 4 == 0
    return False


def filter_confocal_tubulin(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Tubulin channel rows and rename modality to Conf-Tubulin."""
    conf_mask = df["modality_name"] == "Confocal"
    if not conf_mask.any():
        return df
    conf = df[conf_mask]
    tubulin_mask = conf["source_path"].apply(_is_tubulin_channel)
    print(f"  Confocal total: {len(conf)} → Tubulin: {tubulin_mask.sum()}"
          f" (dropped {len(conf) - tubulin_mask.sum()})")
    df = df[~conf_mask].copy()
    tubulin = conf[tubulin_mask].copy()
    tubulin["modality_name"] = "Conf-Tubulin"
    return pd.concat([df, tubulin], ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}, modalities: {df['modality_name'].nunique()}")

    # Filter Confocal to Tubulin
    print("\n--- Filtering Confocal → Conf-Tubulin ---")
    df = filter_confocal_tubulin(df)

    # Drop 2PM, MACSima
    for drop in ["2PM", "MACSima"]:
        before = len(df)
        df = df[df["modality_name"] != drop].copy()
        if len(df) < before:
            print(f"Dropped {drop}: {before - len(df)} rows")

    # Print summary
    for m in TARGET_MODALITIES:
        n = (df["modality_name"] == m).sum()
        print(f"  {m}: {n} images")
    other = df[~df["modality_name"].isin(TARGET_MODALITIES)]["modality_name"].nunique()
    print(f"  Other modalities: {other}")

    # Save CSV
    csv_path = output_dir / "center4_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved filtered CSV: {csv_path}")

    # t-SNE
    print(f"\n{'='*60}\nGenerating t-SNE visualization ...")
    run_tsne_pipeline(
        df=df,
        target_modalities=TARGET_MODALITIES,
        color_map=CENTER_COLORS,
        output_path=output_dir / "center4_tsne.png",
        perplexity=args.perplexity,
        random_state=args.seed,
        pca_dim=args.pca_dim,
        title="Center Modality Subset (4)",
    )
    print("Done.")


if __name__ == "__main__":
    main()
