#!/usr/bin/env python3
"""
Extract features from multi-channel TIFF datasets (e.g. IDR) into a CSV.

Usage::

    python scripts/extract_features.py \\
        --data-root /data2/xujr/idr_data/test_feature \\
        --config configs/analysis/idr_channels.yaml \\
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
import tifffile
import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract multi-channel image features to CSV")
    p.add_argument("--data-root", required=True, help="Root directory containing Study subdirs")
    p.add_argument("--config", required=True, help="YAML channel mapping config")
    p.add_argument("--output", default="features.csv", help="Output CSV path")
    p.add_argument("--checkpoint", default="checkpoint.json", help="Resume checkpoint JSON path")
    p.add_argument("--max-pairs", type=int, default=500, help="Max image pairs per study")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_channel_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return raw.get("datasets", raw)


def get_study_config(study_name: str, all_configs: dict[str, Any]) -> dict[str, Any] | None:
    for key, cfg in all_configs.items():
        if study_name.startswith(key):
            return cfg  # type: ignore[return-value]
    return None


# ---------------------------------------------------------------------------
# Data discovery & pairing (ref main.py logic, extracted)
# ---------------------------------------------------------------------------

def auto_detect_channels(screen_path: Path):
    """Auto-detect hub (nucleus) and source channels from directory names."""
    channel_dirs = sorted([d for d in screen_path.iterdir() if d.is_dir() and d.name.startswith("channel_")])
    hub_ids, src_map = [], {}
    nucleus_keywords = {"hoechst", "nuclei", "dapi", "dna", "h2b"}
    for d in channel_dirs:
        idx = int(d.name.split("_")[1])
        src_map[idx] = f"Ch{idx}"
        # Heuristic: check if channel name hints at nucleus
        name_lower = d.name.lower()
        if any(kw in name_lower for kw in nucleus_keywords):
            hub_ids.append(idx)
    if not hub_ids:
        hub_ids = [max(src_map.keys())]  # fallback: last channel
    src_map = {k: v for k, v in src_map.items() if k not in hub_ids}
    return hub_ids, src_map


def pair_images(study_path: Path, study_config: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    study_name = study_path.name
    hub_label = study_config.get("hub_label", "Hub")
    screen_configs = study_config.get("screen_configs", {})

    for screen_path in sorted(study_path.iterdir()):
        if not screen_path.is_dir():
            continue
        screen_name = screen_path.name

        hub_ids = study_config["hub_channels"]
        src_map = screen_configs.get(screen_name, study_config["source_channels"])

        if hub_ids == "auto_nuclei":
            hub_ids, src_map = auto_detect_channels(screen_path)
            if not hub_ids:
                print(f"  [WARN] {study_name}/{screen_name}: cannot auto-detect, skip")
                continue

        # Collect per-channel file sets
        channel_files: dict[int, set] = {}
        for ch_idx in list(hub_ids) + list(src_map.keys()):
            ch_dir = screen_path / f"channel_{ch_idx}"
            if not ch_dir.is_dir():
                continue
            tiffs = {f.name for f in ch_dir.glob("*.tiff")} | {f.name for f in ch_dir.glob("*.tif")}
            channel_files[ch_idx] = tiffs

        if not channel_files:
            continue

        all_sets = list(channel_files.values())
        common = all_sets[0].intersection(*all_sets[1:]) if len(all_sets) > 1 else all_sets[0]
        if not common:
            print(f"  [WARN] {study_name}/{screen_name}: no common paired files, skip")
            continue

        for fname in sorted(common):
            pairs.append({
                "study": study_name,
                "screen": screen_name,
                "image_id": Path(fname).stem,
                "hub_label": hub_label,
                "hub_paths": {hid: str(screen_path / f"channel_{hid}" / fname) for hid in hub_ids},
                "source_paths": {sid: (str(screen_path / f"channel_{sid}" / fname), sname)
                                 for sid, sname in src_map.items()},
            })
    return pairs


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
    all_configs = load_channel_config(args.config)

    # Resume
    processed: set[str] = set()
    if checkpoint_path.exists():
        processed = set(json.loads(checkpoint_path.read_text()))
        print(f"[Checkpoint] restored {len(processed)} records")

    from modality_analyzer.features import extract_all_features

    studies = sorted([p for p in data_root.iterdir() if p.is_dir()])
    print(f"Found {len(studies)} studies: {[s.name for s in studies]}")

    all_rows: list[dict[str, Any]] = []

    for study_path in studies:
        study_name = study_path.name
        config = get_study_config(study_name, all_configs)
        if config is None:
            print(f"[SKIP] {study_name}: no channel config")
            continue

        print(f"\n{'='*60}\nStudy: {study_name}\n{'='*60}")
        pairs = pair_images(study_path, config)
        print(f"  {len(pairs)} image pairs found")

        if len(pairs) > args.max_pairs:
            pairs = random.sample(pairs, args.max_pairs)
            print(f"  sampled {args.max_pairs} pairs")

        for pair in tqdm(pairs, desc=f"  {study_name}", unit="pair"):
            unique_key = f"{pair['study']}|{pair['screen']}|{pair['image_id']}"
            if unique_key in processed:
                continue

            try:
                # Hub
                for hid, hpath in pair["hub_paths"].items():
                    img = tifffile.imread(hpath)
                    feats = extract_all_features(img)
                    row = {
                        "study": pair["study"], "screen": pair["screen"],
                        "image_id": pair["image_id"], "channel_type": "hub",
                        "channel_index": hid, "channel_name": pair.get("hub_label", f"hub_ch{hid}"),
                    }
                    row.update(feats)
                    all_rows.append(row)

                # Source
                for sid, (spath, sname) in pair["source_paths"].items():
                    img = tifffile.imread(spath)
                    feats = extract_all_features(img)
                    row = {
                        "study": pair["study"], "screen": pair["screen"],
                        "image_id": pair["image_id"], "channel_type": "source",
                        "channel_index": sid, "channel_name": sname,
                    }
                    row.update(feats)
                    all_rows.append(row)

                processed.add(unique_key)
            except Exception as e:
                print(f"\n  [ERROR] {pair['study']}/{pair['screen']}/{pair['image_id']}: {e}")

        checkpoint_path.write_text(json.dumps(list(processed)))
        print(f"  [Checkpoint] saved ({len(processed)} records)")

    if all_rows:
        df = pd.DataFrame(all_rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved: {output_csv}")
        print(f"  Rows: {len(df)}  |  Studies: {df['study'].nunique()}  |  Features: {len(df.columns) - 5}")
    else:
        print("\n[WARN] No data extracted")


if __name__ == "__main__":
    main()
