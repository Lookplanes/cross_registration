#!/usr/bin/env python3
"""
Extract ResNet deep features from multi-modality images into a CSV.

Uses a pretrained ResNet18 (ImageNet) with the final classification layer
removed, producing 512-dim feature vectors per image.

Usage::

    python scripts/extract_resnet_features.py \\
        --data-root /data2/wuyh \\
        --config configs/analysis/modality_sources.yaml \\
        --output results/resnet_features.csv \\
        --batch-size 32
"""

from __future__ import annotations

import argparse
import json
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
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import yaml
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Image discovery (reuse from extract_features.py)
# ---------------------------------------------------------------------------
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def discover_images(data_root: Path, modality_cfg: dict[str, Any]) -> list[dict[str, Any]]:
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
            "modality_name": label,
        })
    return images


# ---------------------------------------------------------------------------
# ResNet feature extractor
# ---------------------------------------------------------------------------
def build_resnet_extractor(device: torch.device) -> nn.Module:
    """Build a ResNet18 feature extractor (512-dim output)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove final fc layer, keep avgpool
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESNET_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_image_resnet(path: str) -> torch.Tensor:
    """Load image, convert to RGB, apply ResNet preprocessing."""
    img = Image.open(path).convert("RGB")
    return RESNET_TRANSFORM(img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ResNet features to CSV")
    p.add_argument("--data-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", default="results/resnet_features.csv")
    p.add_argument("--checkpoint", default="results/resnet_checkpoint.json")
    p.add_argument("--max-images", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_csv = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resume
    processed: set[str] = set()
    if checkpoint_path.exists():
        processed = set(json.loads(checkpoint_path.read_text()))

    # Build model
    print("Loading ResNet18 ...")
    model = build_resnet_extractor(device)
    print(f"Feature dim: 512")

    modalities = config.get("modalities", config)
    all_rows: list[dict[str, Any]] = []
    feat_dim = 512

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

        # Filter already processed
        images = [img for img in images if f"{mod_key}|{img['path']}" not in processed]
        if not images:
            print("  All already processed")
            continue

        # Process in batches
        for i in tqdm(range(0, len(images), args.batch_size), desc=f"  {mod_key}", unit="batch"):
            batch = images[i:i + args.batch_size]
            try:
                tensors = [load_image_resnet(img["path"]) for img in batch]
                batch_tensor = torch.stack(tensors).to(device)

                with torch.no_grad():
                    features = model(batch_tensor)  # (B, 512)

                features_np = features.cpu().numpy()
                for j, img in enumerate(batch):
                    unique_key = f"{mod_key}|{img['path']}"
                    if unique_key in processed:
                        continue
                    row = {
                        "modality_name": img["modality_name"],
                        "source_path": img["path"],
                    }
                    for d in range(feat_dim):
                        row[f"feat_{d:03d}"] = float(features_np[j, d])
                    all_rows.append(row)
                    processed.add(unique_key)
            except Exception as e:
                print(f"\n  [ERROR] batch: {e}")

        checkpoint_path.write_text(json.dumps(list(processed)))

    if all_rows:
        df = pd.DataFrame(all_rows)
        # Ensure feature columns exist
        feat_cols = [f"feat_{d:03d}" for d in range(feat_dim)]
        for col in feat_cols:
            if col not in df.columns:
                df[col] = 0.0
        df.to_csv(output_csv, index=False)
        print(f"\nSaved: {output_csv}")
        print(f"  Rows: {len(df)}  |  Modalities: {df['modality_name'].nunique()}  |  Features: 512")
    else:
        print("\n[WARN] No data extracted")


if __name__ == "__main__":
    main()
