#!/usr/bin/env python3
"""
Full modality landscape analysis: ResNet18 features + t-SNE visualization.

Extracts 512-dim ResNet features from ALL modalities (center + collected),
then generates a t-SNE plot where only the center modalities are distinctly
highlighted — all other collected modalities are shown as "Other" in gray.

Usage::

    python scripts/run_full_tsne.py \\
        --config configs/analysis/all_modalities.yaml \\
        --output-dir results/figures_full \\
        --max-images 100 \\
        --batch-size 32
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from modality_analyzer.io import (
    discover_images, load_checkpoint, save_checkpoint, ensure_is_center_column,
)
from modality_analyzer.features.resnet import (
    FEAT_DIM, build_resnet_extractor, load_image,
)
from modality_analyzer.visualize.tsne_highlight import run_tsne_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CENTER_MODALITIES = ["2PM", "Confocal", "Fluorescence", "H&E", "MACSima", "MSI"]
CENTER_COLORS: dict[str, str] = {
    "2PM":          "#1f77b4",
    "Confocal":     "#ff7f0e",
    "Fluorescence": "#2ca02c",
    "H&E":          "#d62728",
    "MACSima":      "#9467bd",
    "MSI":          "#8c564b",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full modality landscape: ResNet features + t-SNE with center highlight"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default="results/figures_full")
    p.add_argument("--features-csv", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--skip-extraction", action="store_true")
    p.add_argument("--pca-dim", type=int, default=50)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_all_features(config: dict, args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        output_dir / "full_checkpoint.json"
    )

    processed = load_checkpoint(checkpoint_path)
    if processed:
        print(f"[Checkpoint] restored {len(processed)} records\n")

    print("Loading ResNet18 ...")
    model = build_resnet_extractor(device)

    modalities = config.get("modalities", config)
    all_rows: list[dict[str, Any]] = []

    for mod_key, mod_cfg in modalities.items():
        if not isinstance(mod_cfg, dict):
            continue
        mod_cfg.setdefault("label", mod_key)
        mod_cfg.setdefault("center", False)

        is_center = mod_cfg["center"]
        tag = "[CENTER]" if is_center else "[other]"
        print(f"\n{'='*60}\n{tag} {mod_key}\n{'='*60}")

        images = discover_images(
            data_root=mod_cfg["data_root"],
            source_dir=mod_cfg["source_dir"],
            glob_pattern=mod_cfg.get("glob_pattern", "**/*"),
            label=mod_cfg["label"],
            is_center=is_center,
        )
        print(f"  Found {len(images)} images")

        if len(images) > args.max_images:
            images = random.sample(images, args.max_images)
            print(f"  Sampled {args.max_images} images")

        images = [img for img in images if f"{mod_key}|{img['path']}" not in processed]
        if not images:
            print("  All already processed")
            continue

        for i in tqdm(range(0, len(images), args.batch_size),
                      desc=f"  {mod_key}", unit="batch"):
            batch = images[i:i + args.batch_size]
            try:
                tensors = [load_image(img["path"]) for img in batch]
                batch_tensor = torch.stack(tensors).to(device)
                with torch.no_grad():
                    features = model(batch_tensor)
                features_np = features.cpu().numpy()
                for j, img in enumerate(batch):
                    unique_key = f"{mod_key}|{img['path']}"
                    if unique_key in processed:
                        continue
                    row: dict[str, Any] = {
                        "modality_name": img["modality_name"],
                        "is_center": img["is_center"],
                        "source_path": img["path"],
                    }
                    for d in range(FEAT_DIM):
                        row[f"feat_{d:03d}"] = float(features_np[j, d])
                    all_rows.append(row)
                    processed.add(unique_key)
            except Exception as e:
                print(f"\n  [ERROR] batch: {e}")

        save_checkpoint(checkpoint_path, processed)

    if not all_rows:
        raise RuntimeError("No features extracted!")
    df = pd.DataFrame(all_rows)
    for col in [f"feat_{d:03d}" for d in range(FEAT_DIM)]:
        if col not in df.columns:
            df[col] = 0.0
    csv_path = args.features_csv or (output_dir / "all_resnet_features.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved features: {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    total_mods = len(config.get("modalities", config))
    center_count = sum(1 for v in config.get("modalities", config).values()
                       if isinstance(v, dict) and v.get("center", False))
    print(f"Config: {total_mods} modalities ({center_count} center)\n")

    if args.skip_extraction:
        csv_path = args.features_csv or (output_dir / "all_resnet_features.csv")
        if not Path(csv_path).exists():
            print(f"[ERROR] Features CSV not found: {csv_path}")
            sys.exit(1)
        print(f"Loading existing features: {csv_path}")
        df = pd.read_csv(csv_path)
        df = ensure_is_center_column(df, CENTER_MODALITIES)
    else:
        df = extract_all_features(config, args)

    print(f"\n{'='*60}\nGenerating t-SNE visualization ...")
    run_tsne_pipeline(
        df=df,
        target_modalities=CENTER_MODALITIES,
        color_map=CENTER_COLORS,
        output_path=output_dir / "full_tsne_center.png",
        perplexity=args.perplexity,
        random_state=args.seed,
        pca_dim=args.pca_dim,
        title="Modality Landscape",
    )
    print("Done.")


if __name__ == "__main__":
    main()
