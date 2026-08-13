#!/usr/bin/env python3
"""Exhaustively verify an offline Stage 2 dataset and write an audit report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


PATH_FIELDS = {
    "moving_path": "moving",
    "fixed_path": "fixed",
    "flow_path": "gt_flow",
    "valid_mask_path": "valid_mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-per-direction", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def inspect_sample(
    item: tuple[dict[str, str], dict[str, Path]], image_size: int,
) -> dict[str, float]:
    row, paths = item
    for key in ("moving_path", "fixed_path"):
        with Image.open(paths[key]) as image:
            if image.size != (image_size, image_size) or image.mode != "RGB":
                raise ValueError(
                    f"{row['sample_id']} {key}: expected RGB {image_size}x{image_size}, "
                    f"got {image.mode} {image.size}"
                )
            image.verify()
    with Image.open(paths["valid_mask_path"]) as image:
        if image.size != (image_size, image_size) or image.mode != "L":
            raise ValueError(
                f"{row['sample_id']} mask: expected L {image_size}x{image_size}, "
                f"got {image.mode} {image.size}"
            )
        mask = np.asarray(image, dtype=np.uint8)
    unique = set(np.unique(mask).tolist())
    if not unique.issubset({0, 255}):
        raise ValueError(f"{row['sample_id']} mask is not binary: {unique}")
    valid_fraction = float((mask > 0).mean())
    if valid_fraction <= 0.0:
        raise ValueError(f"{row['sample_id']} has an empty valid mask")

    flow = np.load(paths["flow_path"], allow_pickle=False)
    if flow.shape != (2, image_size, image_size):
        raise ValueError(f"{row['sample_id']} flow shape is {flow.shape}")
    if flow.dtype != np.float16:
        raise ValueError(f"{row['sample_id']} flow dtype is {flow.dtype}, expected float16")
    if not np.isfinite(flow).all():
        raise ValueError(f"{row['sample_id']} flow contains NaN or Inf")
    magnitude = np.sqrt(
        flow[0].astype(np.float32) ** 2 + flow[1].astype(np.float32) ** 2,
    )
    return {
        "valid_fraction": valid_fraction,
        "flow_mean_magnitude": float(magnitude.mean()),
        "flow_max_magnitude": float(magnitude.max()),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split", "sample_id", "translation_direction", "direction",
            "moving_id", "fixed_id", "moving_modality", "fixed_modality",
            *PATH_FIELDS,
        }
        missing_fields = sorted(required - set(reader.fieldnames or []))
        if missing_fields:
            raise ValueError(f"manifest missing fields: {missing_fields}")
        rows = list(reader)
    if len(rows) != args.expected_samples:
        raise RuntimeError(
            f"expected {args.expected_samples} rows, found {len(rows)}"
        )
    if any(row["split"] != args.split for row in rows):
        raise ValueError("manifest contains an unexpected split")

    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("sample IDs are not unique")
    translation_counts = Counter(row["translation_direction"] for row in rows)
    registration_counts = Counter(row["direction"] for row in rows)
    for label, counts in (
        ("translation", translation_counts), ("registration", registration_counts),
    ):
        if len(counts) != 12 or set(counts.values()) != {args.expected_per_direction}:
            raise RuntimeError(f"{label} directions are unbalanced: {counts}")
    for row in rows:
        expected = f"{row['moving_modality']}->{row['fixed_modality']}"
        if row["direction"] != expected:
            raise ValueError(f"modality label mismatch for {row['sample_id']}")
        if int(row["moving_id"]) == int(row["fixed_id"]):
            raise ValueError(f"same-domain pair in {row['sample_id']}")

    sample_paths: list[tuple[dict[str, str], dict[str, Path]]] = []
    manifest_sets: dict[str, set[Path]] = {value: set() for value in PATH_FIELDS.values()}
    for row in rows:
        paths = {}
        for field, directory in PATH_FIELDS.items():
            path = (root / row[field]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"missing/unsafe path: {row[field]}")
            paths[field] = path
            manifest_sets[directory].add(path)
        sample_paths.append((row, paths))

    directory_counts = {}
    for directory, expected_paths in manifest_sets.items():
        suffix = ".npy" if directory == "gt_flow" else ".png"
        actual_paths = set((root / args.split / directory).glob(f"*{suffix}"))
        if actual_paths != expected_paths:
            raise RuntimeError(
                f"{directory} file set differs from manifest: "
                f"missing={len(expected_paths - actual_paths)}, "
                f"orphan={len(actual_paths - expected_paths)}"
            )
        directory_counts[directory] = len(actual_paths)
    temporary_files = [str(path) for path in (root / args.split).rglob("*.tmp")]
    if temporary_files:
        raise RuntimeError(f"temporary files remain: {temporary_files[:3]}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        diagnostics = list(executor.map(
            lambda item: inspect_sample(item, args.image_size), sample_paths,
        ))
    valid = np.asarray([item["valid_fraction"] for item in diagnostics])
    flow_mean = np.asarray([item["flow_mean_magnitude"] for item in diagnostics])
    flow_max = np.asarray([item["flow_max_magnitude"] for item in diagnostics])
    total_bytes = sum(
        path.stat().st_size for paths in manifest_sets.values() for path in paths
    )
    report = {
        "status": "passed",
        "manifest": str(manifest),
        "split": args.split,
        "samples": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "translation_direction_counts": dict(sorted(translation_counts.items())),
        "registration_direction_counts": dict(sorted(registration_counts.items())),
        "file_counts": directory_counts,
        "temporary_files": 0,
        "data_bytes": total_bytes,
        "data_gib": total_bytes / 1024 ** 3,
        "valid_fraction": {
            "min": float(valid.min()), "mean": float(valid.mean()),
            "max": float(valid.max()),
        },
        "flow_mean_magnitude_px": {
            "min": float(flow_mean.min()), "mean": float(flow_mean.mean()),
            "max": float(flow_mean.max()),
        },
        "flow_max_magnitude_px": {
            "min": float(flow_max.min()), "mean": float(flow_max.mean()),
            "max": float(flow_max.max()),
        },
    }
    output = Path(args.output).expanduser().resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
