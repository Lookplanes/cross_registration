#!/usr/bin/env python3
"""Probe what a trained conditional TransCUT PatchGAN uses as domain evidence.

The probe is read-only.  It compares the discriminator's correct-vs-wrong
modality margin on real paired images before and after controlled spatial or
appearance perturbations.  It also measures whether the gradient of that
conditional margin is concentrated on positive marker pixels.

This does not evaluate generator quality and does not modify a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from crossreg.translation.transcut.conditional_discriminator import (
    ConditionalPatchDiscriminator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--paired-manifest", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--load-size", type=int, default=286)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--tile-sizes", default="64,32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def load_image(
    path: Path, channels: int, load_size: int, crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        image = image.convert(mode)
        image = TF.resize(
            image, [load_size, load_size], interpolation=InterpolationMode.BICUBIC,
        )
        image = TF.center_crop(image, [crop_size, crop_size])
        raw = TF.to_tensor(image)
    return TF.normalize(raw, [0.5] * channels, [0.5] * channels), raw


def tile_shuffle(x: torch.Tensor, tile_size: int, seed: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    if height % tile_size or width % tile_size:
        raise ValueError(
            f"tile size {tile_size} must divide image size {height}x{width}"
        )
    rows, cols = height // tile_size, width // tile_size
    tiles = x.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(
        x.size(0), rows * cols, x.size(1), tile_size, tile_size,
    )
    generator = torch.Generator(device=x.device).manual_seed(seed)
    shuffled = []
    for sample_tiles in tiles:
        order = torch.randperm(rows * cols, generator=generator, device=x.device)
        sample = sample_tiles[order].reshape(
            rows, cols, x.size(1), tile_size, tile_size,
        )
        sample = sample.permute(2, 0, 3, 1, 4).reshape(
            x.size(1), height, width,
        )
        shuffled.append(sample)
    return torch.stack(shuffled)


def pixel_shuffle(x: torch.Tensor, seed: int) -> torch.Tensor:
    flat = x.flatten(2)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    outputs = []
    for sample in flat:
        order = torch.randperm(flat.size(2), generator=generator, device=x.device)
        outputs.append(sample[:, order].reshape(x.size(1), *x.shape[-2:]))
    return torch.stack(outputs)


def local_contrast_normalize(x: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    padding = kernel_size // 2
    padded = F.pad(x, [padding] * 4, mode="reflect")
    mean = F.avg_pool2d(padded, kernel_size, stride=1)
    squared = F.avg_pool2d(padded.square(), kernel_size, stride=1)
    std = (squared - mean.square()).clamp_min(1e-4).sqrt()
    return ((x - mean) / (3.0 * std)).clamp(-1.0, 1.0)


def conditional_scores(
    discriminator: ConditionalPatchDiscriminator,
    images: torch.Tensor,
    correct_ids: torch.Tensor,
    num_modalities: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    correct = discriminator(images, correct_ids).flatten(1).mean(1)
    wrong_scores = []
    for offset in range(1, num_modalities):
        wrong_ids = (correct_ids + offset) % num_modalities
        wrong_scores.append(
            discriminator(images, wrong_ids).flatten(1).mean(1)
        )
    wrong = torch.stack(wrong_scores).mean(0)
    return correct, wrong, correct - wrong


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
    }


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample and batch counts must be positive")
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    config = checkpoint["config"]
    names = list(checkpoint["modality_names"])
    channels = int(config["input_nc"])
    num_modalities = int(config["num_modalities"])
    discriminator = ConditionalPatchDiscriminator(
        input_nc=channels,
        num_modalities=num_modalities,
        ndf=int(config["ndf"]),
        n_layers=int(config["n_layers_D"]),
    ).to(device)
    discriminator.load_state_dict(checkpoint["netD"], strict=True)
    discriminator.eval()

    manifest_path = Path(args.paired_manifest).expanduser().resolve()
    manifest_root = Path(args.manifest_root).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["patch_id", *(f"{name}_path" for name in names)]
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"paired manifest is missing columns: {missing}")
        rows = [
            row for row in reader
            if not row.get("split") or row["split"] == args.split
        ]
    if not rows:
        raise RuntimeError(f"no rows selected for split {args.split!r}")
    rows = random.Random(args.seed).sample(rows, min(args.max_samples, len(rows)))

    tile_sizes = [int(value) for value in args.tile_sizes.split(",") if value]
    transformations = ["original", *(f"tile_shuffle_{size}" for size in tile_sizes)]
    transformations += ["pixel_shuffle", "gaussian_blur", "local_contrast"]
    scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    saliency: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        for modality_id, name in enumerate(names):
            loaded = [
                load_image(
                    manifest_root / row[f"{name}_path"], channels,
                    args.load_size, args.crop_size,
                )
                for row in batch_rows
            ]
            images = torch.stack([item[0] for item in loaded]).to(device)
            raw = torch.stack([item[1] for item in loaded]).to(device)
            ids = torch.full(
                (images.size(0),), modality_id, dtype=torch.long, device=device,
            )
            variants = {"original": images}
            for tile_size in tile_sizes:
                variants[f"tile_shuffle_{tile_size}"] = tile_shuffle(
                    images, tile_size, args.seed + start + modality_id,
                )
            variants["pixel_shuffle"] = pixel_shuffle(
                images, args.seed + start + modality_id,
            )
            variants["gaussian_blur"] = TF.gaussian_blur(
                images, kernel_size=[9, 9], sigma=[2.0, 2.0],
            )
            variants["local_contrast"] = local_contrast_normalize(images)

            with torch.inference_mode():
                for transform_name, variant in variants.items():
                    correct, wrong, margin = conditional_scores(
                        discriminator, variant, ids, num_modalities,
                    )
                    for metric, tensor in (
                        ("correct_score", correct),
                        ("wrong_score", wrong),
                        ("conditional_margin", margin),
                    ):
                        scores[name][transform_name][metric].extend(
                            tensor.cpu().tolist()
                        )

            if name == "he":
                continue
            saliency_input = images.detach().requires_grad_(True)
            _, _, margin = conditional_scores(
                discriminator, saliency_input, ids, num_modalities,
            )
            gradient = torch.autograd.grad(margin.sum(), saliency_input)[0]
            gradient = gradient.abs().mean(1)
            marker_mask = raw.mean(1) > (5.0 / 255.0)
            for sample_gradient, sample_mask in zip(gradient, marker_mask):
                inside = sample_gradient[sample_mask]
                outside = sample_gradient[~sample_mask]
                inside_mean = inside.mean()
                outside_mean = outside.mean().clamp_min(1e-12)
                total_mass = sample_gradient.sum().clamp_min(1e-12)
                area_fraction = sample_mask.float().mean().clamp_min(1e-12)
                mass_fraction = sample_gradient[sample_mask].sum() / total_mass
                saliency[name]["positive_area_fraction"].append(
                    float(area_fraction)
                )
                saliency[name]["positive_saliency_mass"].append(
                    float(mass_fraction)
                )
                saliency[name]["positive_saliency_enrichment"].append(
                    float(mass_fraction / area_fraction)
                )
                saliency[name]["inside_outside_mean_ratio"].append(
                    float(inside_mean / outside_mean)
                )

    summarized_scores = {
        name: {
            transform: {
                metric: describe(values)
                for metric, values in metrics.items()
            }
            for transform, metrics in transforms.items()
        }
        for name, transforms in scores.items()
    }
    margin_retention = {}
    for name in names:
        baseline = summarized_scores[name]["original"]["conditional_margin"]["mean"]
        margin_retention[name] = {
            transform: (
                values["conditional_margin"]["mean"] / baseline
                if abs(baseline) > 1e-12 else float("nan")
            )
            for transform, values in summarized_scores[name].items()
        }
    result = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_root": str(manifest_root),
        "split": args.split,
        "sample_count": len(rows),
        "seed": args.seed,
        "modalities": names,
        "discriminator": {
            "type": "ConditionalPatchDiscriminator",
            "ndf": int(config["ndf"]),
            "n_layers": int(config["n_layers_D"]),
        },
        "transformations": transformations,
        "scores": summarized_scores,
        "conditional_margin_retention": margin_retention,
        "marker_saliency": {
            name: {metric: describe(values) for metric, values in metrics.items()}
            for name, metrics in saliency.items()
        },
        "interpretation_boundary": (
            "Tile shuffling preserves pixels and within-tile texture but creates "
            "new tile boundaries. Local-contrast images are deliberately out of "
            "the training distribution. Results are a causal probe of the trained "
            "discriminator, not a generator-quality metric."
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved discriminator probe: {output_path}")
    print(text)


if __name__ == "__main__":
    main()
