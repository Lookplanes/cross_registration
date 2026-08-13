#!/usr/bin/env python3
"""Run read-only dataset preflight checks and optionally write JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from crossreg.data.validation import (
    validate_multidomain_translation_dataset,
    validate_registration_dataset,
    validate_translation_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a CrossReg dataset")
    parser.add_argument("--task", required=True, choices=[
        "translation", "multidomain-translation", "registration",
    ])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--pairing-mode", choices=["unpaired", "paired"], default="unpaired")
    parser.add_argument("--require-flow", action="store_true")
    parser.add_argument("--require-mask", action="store_true")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum files checked per domain/task; 0 checks all")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    maximum = None if args.max_samples == 0 else args.max_samples
    if args.task == "translation":
        report = validate_translation_dataset(args.data_dir, args.pairing_mode, maximum)
    elif args.task == "multidomain-translation":
        report = validate_multidomain_translation_dataset(args.data_dir, args.pairing_mode, maximum)
    else:
        report = validate_registration_dataset(
            args.data_dir, args.require_flow, args.require_mask, maximum,
        )
    payload = report.to_dict()
    errors = sum(issue.level == "error" for issue in report.issues)
    warnings = sum(issue.level == "warning" for issue in report.issues)
    print(f"Dataset validation: {payload['status'].upper()}")
    print(f"Task: {report.task}  Root: {report.root}")
    print(f"Errors: {errors}  Warnings: {warnings}")
    print(json.dumps(report.statistics, indent=2, ensure_ascii=False))
    for issue in report.issues:
        location = f" [{issue.path}]" if issue.path else ""
        print(f"{issue.level.upper()} {issue.code}: {issue.message}{location}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
