#!/usr/bin/env python3
"""Evaluate whether TransCUT commutes with known spatial transforms.

This is a read-only diagnostic for

    G(T(x), source_id, target_id) ~= T(G(x, source_id, target_id)).

Both sides are images in the same target domain, so their direct difference
measures translation-induced geometric inconsistency without comparing the
appearance of two real modalities.  It is deliberately an evaluation script,
not a training loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from crossreg.translation.transcut import TransCUT, TransCUTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--paired-manifest", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=2)
    parser.add_argument("--load-size", type=int, default=286)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--shift-pixels", type=float, default=8.0)
    parser.add_argument("--rotation-degrees", type=float, default=3.0)
    parser.add_argument("--elastic-pixels", type=float, default=4.0)
    parser.add_argument("--valid-margin", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_image(path: Path, channels: int, load_size: int, crop_size: int) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        image = image.convert(mode)
        image = TF.resize(
            image, [load_size, load_size], interpolation=InterpolationMode.BICUBIC,
        )
        image = TF.center_crop(image, [crop_size, crop_size])
        tensor = TF.to_tensor(image)
    return TF.normalize(tensor, [0.5] * channels, [0.5] * channels)


def base_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = (torch.arange(height, device=device) + 0.5) * (2.0 / height) - 1.0
    xs = (torch.arange(width, device=device) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def make_grids(
    height: int, width: int, args: argparse.Namespace, device: torch.device,
) -> dict[str, torch.Tensor]:
    identity = base_grid(height, width, device)
    shifted = identity.clone()
    shifted[..., 0] += 2.0 * args.shift_pixels / width

    angle = math.radians(args.rotation_degrees)
    theta = torch.tensor(
        [[[math.cos(angle), -math.sin(angle), 0.0],
          [math.sin(angle), math.cos(angle), 0.0]]],
        dtype=torch.float32, device=device,
    )
    rotated = F.affine_grid(theta, [1, 1, height, width], align_corners=False)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(1, 2, height, width, generator=generator, device=device)
    # Repeated average pooling produces a smooth, low-frequency displacement.
    for _ in range(4):
        noise = F.avg_pool2d(noise, 17, stride=1, padding=8)
    scale = noise.flatten(2).std(dim=2, keepdim=True).clamp_min(1e-6)
    noise = noise / scale.unsqueeze(-1)
    noise = noise.clamp(-2.0, 2.0)
    elastic = identity.clone()
    elastic[..., 0] += noise[:, 0] * (args.elastic_pixels / width)
    elastic[..., 1] += noise[:, 1] * (args.elastic_pixels / height)
    return {
        f"translate_x_{args.shift_pixels:g}px": shifted,
        f"rotate_{args.rotation_degrees:g}deg": rotated,
        f"elastic_rms_{args.elastic_pixels:g}px": elastic,
    }


def warp(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(
        image, grid.expand(image.size(0), -1, -1, -1), mode="bilinear",
        padding_mode="border", align_corners=False,
    )


def valid_mask(grid: torch.Tensor, height: int, width: int, margin: int) -> torch.Tensor:
    valid = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
    if margin:
        interior = torch.zeros_like(valid)
        interior[:, margin:height - margin, margin:width - margin] = True
        valid &= interior
    return valid.unsqueeze(1)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(value)
    return value[expanded].mean()


def display_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


def pixel_mae(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    return float(masked_mean((display_tensor(left) - display_tensor(right)).abs(), mask))


def metrics(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    # Inputs are converted from model range [-1, 1] to display range [0, 1].
    left = display_tensor(left)
    right = display_tensor(right)
    difference = (left - right).abs()
    mae = masked_mean(difference, mask)
    mse = masked_mean((left - right).square(), mask)
    gx_left = left[..., :, 1:] - left[..., :, :-1]
    gx_right = right[..., :, 1:] - right[..., :, :-1]
    gy_left = left[..., 1:, :] - left[..., :-1, :]
    gy_right = right[..., 1:, :] - right[..., :-1, :]
    grad_x = masked_mean((gx_left - gx_right).abs(), mask[..., :, 1:])
    grad_y = masked_mean((gy_left - gy_right).abs(), mask[..., 1:, :])
    return {
        "mae": float(mae),
        "rmse": float(mse.sqrt()),
        "psnr": float(-10.0 * torch.log10(mse.clamp_min(1e-12))),
        "gradient_mae": float((grad_x + grad_y) / 2.0),
    }


def tensor_panel(tensor: torch.Tensor, title: str, heatmap: bool = False) -> Image.Image:
    tensor = tensor.detach().cpu()
    if heatmap:
        value = tensor.abs().mean(0).clamp(0.0, 0.25) / 0.25
        rgb = torch.stack((value, value.square(), torch.zeros_like(value)))
    else:
        rgb = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
        if rgb.size(0) == 1:
            rgb = rgb.expand(3, -1, -1)
    panel = TF.to_pil_image(rgb)
    canvas = Image.new("RGB", (panel.width, panel.height + 24), "white")
    canvas.paste(panel, (0, 24))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def save_contact_sheet(panels: list[list[Image.Image]], output: Path) -> None:
    width = max(sum(panel.width for panel in row) for row in panels)
    height = sum(max(panel.height for panel in row) for row in panels)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in panels:
        x = 0
        row_height = max(panel.height for panel in row)
        for panel in row:
            sheet.paste(panel, (x, y))
            x += panel.width
        y += row_height
    sheet.save(output)


def summarize(records: list[dict]) -> dict:
    def aggregate(selected: list[dict]) -> dict[str, float | int]:
        keys = (
            "mae", "rmse", "psnr", "gradient_mae",
            "source_transform_mae", "output_transform_mae",
            "commutation_to_transform_ratio", "output_std",
        )
        return {
            "count": len(selected),
            **{key: float(np.mean([row[key] for row in selected])) for key in keys},
        }

    by_transform = {
        name: aggregate([row for row in records if row["transform"] == name])
        for name in sorted({row["transform"] for row in records})
    }
    by_direction = {
        name: aggregate([row for row in records if row["direction"] == name])
        for name in sorted({row["direction"] for row in records})
    }
    return {"overall": aggregate(records), "by_transform": by_transform,
            "by_direction": by_direction}


def main() -> None:
    args = parse_args()
    if args.max_rows <= 0 or args.valid_margin * 2 >= args.crop_size:
        raise ValueError("invalid sample count or valid margin")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint["config"]
    valid_fields = {item.name for item in fields(TransCUTConfig)}
    config_values = {key: value for key, value in saved_config.items() if key in valid_fields}
    config_values["gpu_ids"] = [device.index or 0] if device.type == "cuda" else []
    model = TransCUT(TransCUTConfig(**config_values))
    names = list(checkpoint["modality_names"])
    model.set_modality_names(names)
    model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
    model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
    model.mod_embed.load_state_dict(checkpoint["mod_embed"], strict=True)
    model.style_embed.load_state_dict(checkpoint["style_embed"], strict=True)
    model.to(device).eval()

    manifest = Path(args.paired_manifest).expanduser().resolve()
    root = Path(args.manifest_root).expanduser().resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = [f"{name}_path" for name in names]
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest is missing columns: {missing}")
        rows = [row for row in reader if not row.get("split") or row["split"] == args.split]
    rows = random.Random(args.seed).sample(rows, min(args.max_rows, len(rows)))
    if not rows:
        raise RuntimeError(f"no manifest rows selected for split {args.split!r}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grids = make_grids(args.crop_size, args.crop_size, args, device)
    records: list[dict] = []

    with torch.inference_mode():
        for row_index, row in enumerate(rows):
            visual_rows: list[list[Image.Image]] = []
            for src_id, src_name in enumerate(names):
                source = load_image(
                    root / row[f"{src_name}_path"], int(saved_config["input_nc"]),
                    args.load_size, args.crop_size,
                ).unsqueeze(0).to(device)
                target_ids = [index for index in range(len(names)) if index != src_id]
                target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
                source_batch = source.expand(len(target_ids), -1, -1, -1)
                source_ids = torch.full_like(target_tensor, src_id)
                generated = model(source_batch, source_ids, target_tensor)
                for transform_name, grid in grids.items():
                    transformed_source = warp(source, grid)
                    first_warp = model(
                        transformed_source.expand(len(target_ids), -1, -1, -1),
                        source_ids, target_tensor,
                    )
                    first_translate = warp(generated, grid)
                    mask = valid_mask(
                        grid, args.crop_size, args.crop_size, args.valid_margin,
                    )
                    source_transform_mae = pixel_mae(source, transformed_source, mask)
                    for batch_index, tgt_id in enumerate(target_ids):
                        result = metrics(
                            first_warp[batch_index:batch_index + 1],
                            first_translate[batch_index:batch_index + 1], mask,
                        )
                        target_slice = generated[batch_index:batch_index + 1]
                        warped_slice = first_translate[batch_index:batch_index + 1]
                        output_transform_mae = pixel_mae(
                            target_slice, warped_slice, mask,
                        )
                        output_pixels = display_tensor(target_slice)[mask.expand_as(target_slice)]
                        direction = f"{src_name}->{names[tgt_id]}"
                        records.append({
                            "row": row.get("patch_id", str(row_index)),
                            "direction": direction, "source": src_name,
                            "target": names[tgt_id], "transform": transform_name,
                            "source_transform_mae": source_transform_mae,
                            "output_transform_mae": output_transform_mae,
                            "commutation_to_transform_ratio": (
                                result["mae"] / max(output_transform_mae, 1e-8)
                            ),
                            "output_std": float(output_pixels.std()),
                            **result,
                        })
                        if row_index == 0:
                            visual_rows.append([
                                tensor_panel(source[0], f"{direction}: source"),
                                tensor_panel(transformed_source[0], transform_name),
                                tensor_panel(first_warp[batch_index], "G(T(x))"),
                                tensor_panel(first_translate[batch_index], "T(G(x))"),
                                tensor_panel(
                                    first_warp[batch_index] - first_translate[batch_index],
                                    f"abs diff; MAE={result['mae']:.4f}", heatmap=True,
                                ),
                            ])
            if visual_rows:
                save_contact_sheet(
                    visual_rows, output_dir / "commutation_contact_sheet.png",
                )

    fieldnames = list(records[0])
    with (output_dir / "commutation_metrics.csv").open(
        "w", encoding="utf-8", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    report = {
        "checkpoint": str(checkpoint_path), "manifest": str(manifest),
        "modalities": names, "sample_rows": len(rows),
        "transforms": list(grids), "valid_margin": args.valid_margin,
        **summarize(records),
    }
    with (output_dir / "commutation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report["overall"], indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
