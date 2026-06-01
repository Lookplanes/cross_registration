#!/usr/bin/env python3
import csv
import os
import re
from pathlib import Path




OUTPUT_CSV = "results/merged_epoch_loss.csv"


EPOCH_LOSS_PATTERN = re.compile(r"Epoch\s+(\d+)\s+Loss:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
VAL_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+Validation:\s+Mean\s+EPE:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s+"
    r"Mean\s+SSIM:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s+"
    r"Mean\s+ZNCC:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def parse_single_log(log_path: Path):
    rows_by_epoch = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            loss_match = EPOCH_LOSS_PATTERN.search(line)
            if loss_match:
                epoch = int(loss_match.group(1))
                epoch_loss = float(loss_match.group(2))
                rows_by_epoch.setdefault(epoch, {})["epoch_loss"] = epoch_loss

            val_match = VAL_PATTERN.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                rows = rows_by_epoch.setdefault(epoch, {})
                rows["val_epe"] = float(val_match.group(2))
                rows["val_ssim"] = float(val_match.group(3))
                rows["val_zncc"] = float(val_match.group(4))

    parsed_rows = []
    for epoch in sorted(rows_by_epoch):
        row = rows_by_epoch[epoch]
        parsed_rows.append(
            {
                # "source_file": str(log_path),
                "source_name": log_path.name,
                "epoch": epoch,
                "epoch_loss": row.get("epoch_loss"),
                "val_epe": row.get("val_epe"),
                "val_ssim": row.get("val_ssim"),
                "val_zncc": row.get("val_zncc"),
            }
        )

    return parsed_rows


def main():
    base_dir = '/home/xujr/cross_registration/log'
    all_rows = []

    log_files = [
        'nohup_train-full.log',
        'nohup_train-identity.log',
        'nohup_train-no_affine.log',
        'nohup_train-no_appearance.log',
    ]

    log_files_pth = [str(Path(base_dir) / log_file) for log_file in log_files]

    for path_str in log_files_pth:
        print(f"[INFO] processing log file: {path_str}")
        log_path = resolve_path(base_dir, path_str)
        if not log_path.exists():
            print(f"[WARN] file not found, skipped: {log_path}")
            continue

        parsed = parse_single_log(log_path)
        print(f"[INFO] parsed {len(parsed)} epoch rows from: {log_path}")
        all_rows.extend(parsed)

    output_path = resolve_path(base_dir, OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        # "source_file",
        "source_name",
        "epoch",
        "epoch_loss",
        "val_epe",
        "val_ssim",
        "val_zncc",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[DONE] wrote {len(all_rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()
