#!/usr/bin/env python3
"""Build non-destructive HEMIT manifests for the informative-marker ablation.

The main training manifest remains unpaired: H&E and DAPI keep every preprocessed training patch,
while panCK and CD3 independently keep patches whose signal fraction reaches
the requested threshold.

An additional paired training manifest contains only rows where all marker
domains reach the threshold. It is an explicit supervised-control input and
is never selected by the unpaired launcher.

Fixed test samples remain paired: a row is eligible only when DAPI, panCK and
CD3 from the same spatial patch all reach the threshold.  No images are copied,
linked, deleted or rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DOMAINS = ("he", "dapi", "panck", "cd3")
MARKERS = ("dapi", "panck", "cd3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--signal-threshold", type=float, default=0.05)
    parser.add_argument(
        "--paired-anchor-fraction", type=float, default=0.05,
        help="Deterministic fraction of eligible paired train patches to label",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _signal(row: dict[str, str], domain: str) -> float:
    return float(row[f"{domain}_signal_fraction"])


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.signal_threshold <= 1.0:
        raise ValueError("--signal-threshold must be in [0, 1]")
    if not 0.0 < args.paired_anchor_fraction <= 1.0:
        raise ValueError("--paired-anchor-fraction must be in (0, 1]")

    input_path = Path(args.input_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        source_fields = list(reader.fieldnames or [])

    required = {
        "split", "patch_id",
        *(f"{domain}_path" for domain in DOMAINS),
        *(f"{domain}_signal_fraction" for domain in MARKERS),
    }
    missing = sorted(required.difference(source_fields))
    if missing:
        raise ValueError(f"input manifest is missing columns: {missing}")

    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]

    train_output = output_dir / "train_unpaired.csv"
    long_fields = (
        "split", "modality", "patch_id", "patch_path", "signal_fraction",
        "selection_rule",
    )
    train_counts: dict[str, int] = {}
    with train_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for domain in DOMAINS:
            selected = (
                train_rows
                if domain in ("he", "dapi")
                else [
                    row for row in train_rows
                    if _signal(row, domain) >= args.signal_threshold
                ]
            )
            train_counts[domain] = len(selected)
            for row in selected:
                writer.writerow({
                    "split": "train",
                    "modality": domain,
                    "patch_id": row["patch_id"],
                    "patch_path": row[f"{domain}_path"],
                    "signal_fraction": (
                        row.get(f"{domain}_signal_fraction", "")
                    ),
                    "selection_rule": (
                        "all_preprocessed_tissue_patches"
                        if domain in ("he", "dapi")
                        else f"{domain}_signal_fraction>={args.signal_threshold:g}"
                    ),
                })

    paired_train_rows = [
        row for row in train_rows
        if all(
            _signal(row, domain) >= args.signal_threshold
            for domain in MARKERS
        )
    ]
    paired_train_output = output_dir / "train_paired.csv"
    with paired_train_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for row in paired_train_rows:
            for domain in DOMAINS:
                writer.writerow({
                    "split": "train",
                    "modality": domain,
                    "patch_id": row["patch_id"],
                    "patch_path": row[f"{domain}_path"],
                    "signal_fraction": row.get(
                        f"{domain}_signal_fraction", "",
                    ),
                    "selection_rule": (
                        "all_four_domains_paired_and_all_markers_"
                        f"signal_fraction>={args.signal_threshold:g}"
                    ),
                })

    anchor_count = max(
        1, round(len(paired_train_rows) * args.paired_anchor_fraction),
    )
    paired_anchor_rows = random.Random(args.seed).sample(
        paired_train_rows, anchor_count,
    )
    paired_anchor_rows.sort(key=lambda row: row["patch_id"])
    percentage = round(args.paired_anchor_fraction * 100)
    paired_anchor_output = output_dir / (
        f"train_paired_anchor_{percentage:02d}.csv"
    )
    with paired_anchor_output.open(
        "w", encoding="utf-8", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for row in paired_anchor_rows:
            for domain in DOMAINS:
                writer.writerow({
                    "split": "train",
                    "modality": domain,
                    "patch_id": row["patch_id"],
                    "patch_path": row[f"{domain}_path"],
                    "signal_fraction": row.get(
                        f"{domain}_signal_fraction", "",
                    ),
                    "selection_rule": (
                        f"deterministic_{args.paired_anchor_fraction:g}_"
                        "subset_of_paired_signal_eligible_train_patches"
                    ),
                })

    paired_test_rows = [
        row for row in test_rows
        if all(
            _signal(row, domain) >= args.signal_threshold
            for domain in MARKERS
        )
    ]
    paired_output = output_dir / "paired_test.csv"
    with paired_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(paired_test_rows)

    summary = {
        "source_manifest": str(input_path),
        "signal_threshold": args.signal_threshold,
        "training_semantics": {
            "pairing": "unpaired",
            "he": "all preprocessed train patches",
            "dapi": "all preprocessed train patches",
            "panck": "independently filtered by signal fraction",
            "cd3": "independently filtered by signal fraction",
        },
        "train_original_rows": len(train_rows),
        "train_selected_per_domain": train_counts,
        "paired_train_semantics": (
            "same patch_id and coordinates for all four domains; "
            "DAPI, panCK and CD3 signal fractions all meet the threshold"
        ),
        "paired_train_rows": len(paired_train_rows),
        "paired_anchor_fraction": args.paired_anchor_fraction,
        "paired_anchor_seed": args.seed,
        "paired_anchor_rows": len(paired_anchor_rows),
        "paired_test_semantics": (
            "same patch_id and coordinates for all four domains; "
            "DAPI, panCK and CD3 signal fractions all meet the threshold"
        ),
        "test_original_rows": len(test_rows),
        "paired_test_selected_rows": len(paired_test_rows),
        "outputs": {
            "train_unpaired": str(train_output),
            "train_paired": str(paired_train_output),
            "train_paired_anchor": str(paired_anchor_output),
            "paired_test": str(paired_output),
        },
    }
    summary_output = output_dir / "summary.json"
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
