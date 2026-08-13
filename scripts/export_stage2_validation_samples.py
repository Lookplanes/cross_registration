#!/usr/bin/env python3
"""Export fixed validation samples from a trained offline Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_src = _root / "src"
for path in (_root, _src):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from torch.utils.data import DataLoader

from crossreg.data.stage2_offline import OfflineStage2Dataset
from crossreg.registration.visualization import RegistrationSampleCollector
from scripts.train_stage2_offline import build_model, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--samples-per-direction", type=int, default=1)
    parser.add_argument("--flow-limit", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    stage1_path = Path(model_config["encoder_checkpoint"]).expanduser().resolve()
    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
    model = build_model(stage1, tuple(model_config["img_size"]), device)
    model.load_state_dict(checkpoint["registration_model"], strict=True)

    dataset = OfflineStage2Dataset(args.manifest, args.root)
    options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        options.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **options)
    collector = RegistrationSampleCollector(args.samples_per_direction)
    metrics = validate(model, loader, device, 0, sample_collector=collector)
    output = collector.save(args.output, flow_limit=args.flow_limit)
    metrics_output = Path(
        args.metrics_output or output.with_suffix(".json")
    ).expanduser().resolve()
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(f"Validation samples: {output}")
    print(f"Validation metrics: {metrics_output}")


if __name__ == "__main__":
    main()
