#!/usr/bin/env python3
"""Merge disjoint offline Stage 2 shard manifests with completeness checks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-per-direction", type=int, required=True)
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--output-config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows: list[dict[str, str]] = []
    fieldnames = None
    for value in args.inputs:
        path = Path(value).expanduser().resolve()
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"manifest schema mismatch: {path}")
            rows.extend(reader)
    if len(rows) != args.expected_samples:
        raise RuntimeError(
            f"expected {args.expected_samples} rows, found {len(rows)}"
        )
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        duplicates = [name for name, count in Counter(sample_ids).items() if count > 1]
        raise RuntimeError(f"duplicate sample IDs: {duplicates[:3]}")
    translation_counts = Counter(row["translation_direction"] for row in rows)
    invalid_counts = {
        direction: count for direction, count in translation_counts.items()
        if count != args.expected_per_direction
    }
    if invalid_counts or len(translation_counts) != 12:
        raise RuntimeError(f"unbalanced translation directions: {translation_counts}")
    missing = []
    for row in rows:
        for key in ("moving_path", "fixed_path", "flow_path", "valid_mask_path"):
            if not (root / row[key]).is_file():
                missing.append((row["sample_id"], key))
                break
    if missing:
        raise FileNotFoundError(f"incomplete samples: {missing[:3]}")
    rows.sort(key=lambda row: row["sample_id"])
    output = Path(args.output).expanduser().resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    if bool(args.configs) != bool(args.output_config):
        raise ValueError("--configs and --output-config must be provided together")
    if args.configs:
        configs = [
            json.loads(Path(value).expanduser().resolve().read_text(encoding="utf-8"))
            for value in args.configs
        ]
        invariant_keys = (
            "format_version", "checkpoint", "checkpoint_sha256", "source_manifest",
            "manifest_root", "source_split", "output_split", "modalities",
            "samples_per_translation_direction", "pair_direction", "preprocessing",
            "deformation", "storage", "save_diagnostics", "seed",
        )
        reference = configs[0]
        mismatches = [
            key for key in invariant_keys
            if any(config.get(key) != reference.get(key) for config in configs[1:])
        ]
        if mismatches:
            raise RuntimeError(f"shard config mismatch: {mismatches}")
        selected = sorted({
            direction for config in configs
            for direction in config["selected_translation_directions"]
        })
        if len(selected) != 12:
            raise RuntimeError(f"merged configs cover {len(selected)} directions, not 12")
        merged_config = {
            key: reference[key] for key in invariant_keys
        }
        merged_config.update({
            "build_mode": "sharded",
            "selected_translation_directions": selected,
            "total_expected_samples": args.expected_samples,
            "completed_samples": len(rows),
            "generated_this_run": sum(
                int(config.get("generated_this_run", 0)) for config in configs
            ),
            "reused_this_run": sum(
                int(config.get("reused_this_run", 0)) for config in configs
            ),
            "shard_configs": [str(Path(value).expanduser().resolve()) for value in args.configs],
        })
        config_output = Path(args.output_config).expanduser().resolve()
        config_temporary = config_output.with_suffix(config_output.suffix + ".tmp")
        with config_temporary.open("w", encoding="utf-8") as handle:
            json.dump(merged_config, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(config_temporary, config_output)
    print(f"Merged {len(rows)} samples: {output}")
    print("Directions: " + " ".join(
        f"{name}={count}" for name, count in sorted(translation_counts.items())
    ))
    if args.output_config:
        print(f"Merged config: {Path(args.output_config).expanduser().resolve()}")


if __name__ == "__main__":
    main()
