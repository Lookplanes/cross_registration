#!/usr/bin/env python3
"""
Extract features from multi-modality image datasets into a CSV.

Each image is labelled by its modality name. The resulting features.csv is
consumed by ``analyze_domain_gap.py`` to produce distance matrices and PCA plots.

Usage::

    python scripts/extract_features.py \\
        --data-root /data2/wuyh \\
        --config configs/analysis/modality_sources.yaml \\
        --output results/features.csv \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on PYTHONPATH
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract multi-modality image features to CSV")
    p.add_argument("--data-root", required=True, help="Root directory with per-modality subdirs")
    p.add_argument("--config", required=True, help="YAML config listing modality sources")
    p.add_argument("--output", default="features.csv", help="Output CSV path")
    p.add_argument("--checkpoint", default="checkpoint.json", help="Resume checkpoint JSON path")
    p.add_argument("--max-images", type=int, default=200, help="Max images per modality")
    p.add_argument("--max-size", type=int, default=1024, help="Resize large images to this max dim")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def discover_images(data_root: Path, modality_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Discover all images for a single modality.

    modality_cfg may contain:
      - source_dir: subdirectory under data_root (required)
      - glob_pattern: glob pattern relative to source_dir (default "**/*")
      - label: modality display name (defaults to key name)
    """
    source_dir = data_root / modality_cfg["source_dir"]
    pattern = modality_cfg.get("glob_pattern", "**/*")
    label = modality_cfg.get("label", "")

    if not source_dir.exists():
        print(f"  [WARN] {source_dir} not found, skipping")
        return []

    files: list[Path] = []
    for ext in IMAGE_EXT:
        files.extend(source_dir.glob(f"{pattern}{ext}"))
        files.extend(source_dir.glob(f"{pattern}{ext.upper()}"))

    files = sorted(set(files))

    images: list[dict[str, Any]] = []
    for f in files:
        if f.suffix.lower() not in IMAGE_EXT:
            continue
        images.append({
            "path": str(f),
            "modality_name": label if label else mod_cfg.get("label", "unknown"),
        })
    return images


def load_and_resize(path: str, max_size: int) -> np.ndarray:
    """Load image as grayscale float32, optionally resize to max_size."""
    img = Image.open(path).convert("L")
    w, h = img.size
    if max_size > 0 and max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    output_csv = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    config = load_config(args.config)

    # Resume
    processed: set[str] = set()
    if checkpoint_path.exists():
        processed = set(json.loads(checkpoint_path.read_text()))
        print(f"[Checkpoint] restored {len(processed)} records")

    from modality_analyzer.features import extract_all_features

    modalities = config.get("modalities", config)
    all_rows: list[dict[str, Any]] = []

    for mod_key, mod_cfg in modalities.items():
        if not isinstance(mod_cfg, dict):
            mod_cfg = {"source_dir": mod_cfg}
        mod_cfg.setdefault("label", mod_key)

        print(f"\n{'='*60}\nModality: {mod_key}\n{'='*60}")
        images = discover_images(data_root, mod_cfg)
        print(f"  Found {len(images)} images")

        if len(images) > args.max_images:
            images = random.sample(images, args.max_images)
            print(f"  Sampled {args.max_images} images")

        for img_info in tqdm(images, desc=f"  {mod_key}", unit="img"):
            unique_key = f"{mod_key}|{img_info['path']}"
            if unique_key in processed:
                continue

            try:
                img = load_and_resize(img_info["path"], args.max_size)
                feats = extract_all_features(img)
                row = {
                    "modality_name": img_info["modality_name"],
                    "source_path": img_info["path"],
                }
                row.update(feats)
                all_rows.append(row)
                processed.add(unique_key)
            except Exception as e:
                print(f"\n  [ERROR] {img_info['path']}: {e}")

        checkpoint_path.write_text(json.dumps(list(processed)))
        print(f"  [Checkpoint] saved ({len(processed)} records)")

    if all_rows:
        df = pd.DataFrame(all_rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved: {output_csv}")
        print(f"  Rows: {len(df)}  |  Modalities: {df['modality_name'].nunique()}  |  Features: {len(df.columns) - 2}")
    else:
        print("\n[WARN] No data extracted")


if __name__ == "__main__":
    main()
