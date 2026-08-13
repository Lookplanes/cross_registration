#!/usr/bin/env python3
"""Build a resumable offline TransCUT + known-flow Stage 2 dataset.

The builder freezes a TransCUT checkpoint, translates each selected source,
then applies a known diffeomorphic displacement to the image chosen as fixed.
Training subsequently reads only PNG/NPY files and never runs TransCUT online.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from crossreg.data.perturbation import generate_diffeomorphic_flow
from crossreg.registration.transmorph.model import SpatialTransformer, VecInt
from crossreg.translation.transcut import TransCUT, TransCUTConfig


MANIFEST_FIELDS = (
    "split", "sample_id", "direction", "translation_direction",
    "pair_direction", "moving_id", "fixed_id", "moving_modality",
    "fixed_modality", "source_path", "moving_path", "fixed_path",
    "flow_path", "valid_mask_path", "aligned_target_path", "seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--output-split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--directions",
        help="Optional comma-separated source-target ID pairs, e.g. 0-1,0-2,0-3",
    )
    parser.add_argument(
        "--artifact-suffix", default="",
        help="Suffix for shard config/manifest filenames; sample paths stay canonical",
    )
    parser.add_argument("--samples-per-direction", type=int, default=5000)
    parser.add_argument(
        "--pair-direction", choices=("alternating", "source-moving", "target-moving"),
        default="alternating",
    )
    parser.add_argument("--load-size", type=int, default=358)
    parser.add_argument(
        "--canvas-size", type=int, default=320,
        help="Translate and deform on this larger canvas before final cropping",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smooth-sigma", type=float, default=12.0)
    parser.add_argument("--max-displacement", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument(
        "--save-diagnostics", action="store_true",
        help="Also save the aligned target before deformation for visual diagnostics",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_manifest(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_png(tensor: torch.Tensor, path: Path, *, mask: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if mask:
        array = tensor.detach().cpu().squeeze().numpy().astype(np.uint8) * 255
        image = Image.fromarray(array, mode="L")
    else:
        tensor = ((tensor.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
        image = TF.to_pil_image(tensor)
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def atomic_flow(flow: torch.Tensor, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, flow.detach().cpu().numpy().astype(np.float16), allow_pickle=False)
    os.replace(temporary, path)


def read_source_pools(
    manifest: Path, root: Path, split: str, names: list[str],
) -> dict[str, list[Path]]:
    pools: dict[str, list[Path]] = defaultdict(list)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if {"modality", "patch_path"}.issubset(fieldnames):
            for row in reader:
                if row.get("split") and row["split"] != split:
                    continue
                if row["modality"] in names:
                    pools[row["modality"]].append((root / row["patch_path"]).resolve())
        elif all(f"{name}_path" in fieldnames for name in names):
            for row in reader:
                if row.get("split") and row["split"] != split:
                    continue
                for name in names:
                    pools[name].append((root / row[f"{name}_path"]).resolve())
        else:
            raise ValueError(
                "source manifest must be long-form modality/patch_path or contain "
                "one <modality>_path column per checkpoint modality"
            )
    for name in names:
        unique = sorted(set(pools[name]))
        invalid = [path for path in unique if not path.is_file()]
        if invalid:
            raise FileNotFoundError(f"source image not found: {invalid[0]}")
        if not unique:
            raise RuntimeError(f"source pool is empty for modality {name!r}")
        pools[name] = unique
    return dict(pools)


def select_sources(paths: list[Path], count: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    order = list(paths)
    selected: list[Path] = []
    while len(selected) < count:
        rng.shuffle(order)
        selected.extend(order[:min(len(order), count - len(selected))])
    return selected


def select_directions(value: str | None, count: int) -> list[tuple[int, int]]:
    all_directions = [
        (source, target) for source in range(count) for target in range(count)
        if source != target
    ]
    if not value:
        return all_directions
    selected = []
    for token in value.split(","):
        parts = token.strip().split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid direction token: {token!r}")
        pair = (int(parts[0]), int(parts[1]))
        if pair not in all_directions:
            raise ValueError(f"invalid direction pair: {pair}")
        if pair in selected:
            raise ValueError(f"duplicate direction pair: {pair}")
        selected.append(pair)
    if not selected:
        raise ValueError("--directions selected no pairs")
    return selected


def load_batch(paths: list[Path], load_size: int, canvas_size: int) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image, [load_size, load_size], interpolation=InterpolationMode.BICUBIC,
            )
            image = TF.center_crop(image, [canvas_size, canvas_size])
            tensor = TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)
        tensors.append(tensor)
    return torch.stack(tensors)


def center_crop_tensor(tensor: torch.Tensor, size: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    if size > height or size > width:
        raise ValueError(f"cannot crop {size} from tensor with shape {tensor.shape}")
    top = (height - size) // 2
    left = (width - size) // 2
    return tensor[..., top:top + size, left:left + size]


def load_generator(checkpoint: dict, device: torch.device) -> TransCUT:
    saved = checkpoint["config"]
    allowed = {item.name for item in fields(TransCUTConfig)}
    values = {key: value for key, value in saved.items() if key in allowed}
    values["gpu_ids"] = [device.index or 0] if device.type == "cuda" else []
    model = TransCUT(TransCUTConfig(**values))
    model.set_modality_names(list(checkpoint["modality_names"]))
    for name in ("encoder", "decoder", "mod_embed", "style_embed"):
        getattr(model, name).load_state_dict(checkpoint[name], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def ensure_space(output_dir: Path, minimum_gb: float) -> None:
    free = shutil.disk_usage(output_dir).free / 1024 ** 3
    if free < minimum_gb:
        raise RuntimeError(
            f"only {free:.1f} GiB free at {output_dir}; require {minimum_gb:.1f} GiB"
        )


def main() -> None:
    args = parse_args()
    if args.samples_per_direction <= 0 or args.batch_size <= 0:
        raise ValueError("sample and batch counts must be positive")
    if not 0 < args.crop_size <= args.canvas_size <= args.load_size:
        raise ValueError("sizes must satisfy 0 < crop_size <= canvas_size <= load_size")
    if (args.canvas_size - args.crop_size) % 2:
        raise ValueError("canvas_size - crop_size must be even for centered coordinates")
    canvas_margin = (args.canvas_size - args.crop_size) // 2
    if canvas_margin < args.max_displacement + 2:
        raise ValueError(
            f"canvas margin {canvas_margin}px is too small for "
            f"max displacement {args.max_displacement}px"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    source_manifest = Path(args.source_manifest).expanduser().resolve()
    manifest_root = Path(args.manifest_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    split_dir = output_dir / args.output_split
    output_names = ["moving", "fixed", "gt_flow", "valid_mask"]
    if args.save_diagnostics:
        output_names.append("aligned_target")
    for name in output_names:
        (split_dir / name).mkdir(parents=True, exist_ok=True)
    ensure_space(output_dir, args.min_free_gb)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    names = list(checkpoint["modality_names"])
    directions = select_directions(args.directions, len(names))
    if checkpoint["config"].get("input_nc") != 3:
        raise ValueError("offline HEMIT builder currently requires a 3-channel checkpoint")
    pools = read_source_pools(
        source_manifest, manifest_root, args.source_split, names,
    )
    model = load_generator(checkpoint, device)
    canvas_spatial = SpatialTransformer((args.canvas_size, args.canvas_size)).to(device)
    final_spatial = SpatialTransformer((args.crop_size, args.crop_size)).to(device)
    integrate = VecInt((args.canvas_size, args.canvas_size), nsteps=7).to(device)
    final_ones = torch.ones(1, 1, args.crop_size, args.crop_size, device=device)

    metadata = {
        "format_version": 2,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "source_manifest": str(source_manifest),
        "manifest_root": str(manifest_root),
        "source_split": args.source_split,
        "output_split": args.output_split,
        "modalities": names,
        "samples_per_translation_direction": args.samples_per_direction,
        "selected_translation_directions": [
            f"{source}-{target}" for source, target in directions
        ],
        "total_expected_samples": args.samples_per_direction * len(directions),
        "pair_direction": args.pair_direction,
        "preprocessing": {
            "mode": "RGB", "load_size": args.load_size,
            "canvas_size": args.canvas_size, "crop_size": args.crop_size,
            "procedure": "resize -> center canvas -> translate/deform -> center final crop",
            "canvas_margin": canvas_margin, "model_range": "[-1,1]",
        },
        "deformation": {
            "type": "integrated_stationary_velocity", "channels": ["dy", "dx"],
            "parameterization": "backward_sampling_displacement_pixels",
            "smooth_sigma": args.smooth_sigma,
            "max_displacement": args.max_displacement,
            "integration_steps": 7,
        },
        "storage": {"images": "RGB PNG", "flow": "float16 NPY", "mask": "1-bit PNG"},
        "save_diagnostics": args.save_diagnostics,
        "seed": args.seed,
    }
    suffix = f"_{args.artifact_suffix}" if args.artifact_suffix else ""
    config_path = output_dir / f"dataset_config_{args.output_split}{suffix}.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        immutable = (
            "checkpoint_sha256", "source_manifest", "source_split", "modalities",
            "samples_per_translation_direction", "pair_direction", "preprocessing",
            "deformation", "selected_translation_directions",
            "save_diagnostics", "seed",
        )
        changed = [key for key in immutable if existing.get(key) != metadata.get(key)]
        if changed:
            raise RuntimeError(
                f"refusing to resume with changed dataset contract: {changed}"
            )
    else:
        atomic_json(metadata, config_path)

    rows: list[dict] = []
    completed = generated = 0
    with torch.inference_mode():
        for src_id, src_name in enumerate(names):
            for tgt_id, tgt_name in enumerate(names):
                if (src_id, tgt_id) not in directions:
                    continue
                direction_seed = args.seed + src_id * 10_000 + tgt_id * 1_000
                selected = select_sources(
                    pools[src_name], args.samples_per_direction, direction_seed,
                )
                for start in range(0, len(selected), args.batch_size):
                    batch_paths = selected[start:start + args.batch_size]
                    sample_indices = list(range(start, start + len(batch_paths)))
                    sample_ids = [
                        f"{args.output_split}_s{src_id:02d}_t{tgt_id:02d}_{index:06d}"
                        for index in sample_indices
                    ]
                    records = []
                    missing_indices = []
                    for local_index, (sample_index, sample_id, source_path) in enumerate(
                        zip(sample_indices, sample_ids, batch_paths)
                    ):
                        if args.pair_direction == "alternating":
                            pair_direction = (
                                "source-moving" if sample_index % 2 == 0
                                else "target-moving"
                            )
                        else:
                            pair_direction = args.pair_direction
                        moving_id, fixed_id = (
                            (src_id, tgt_id) if pair_direction == "source-moving"
                            else (tgt_id, src_id)
                        )
                        relative = {
                            "moving_path": f"{args.output_split}/moving/{sample_id}.png",
                            "fixed_path": f"{args.output_split}/fixed/{sample_id}.png",
                            "flow_path": f"{args.output_split}/gt_flow/{sample_id}.npy",
                            "valid_mask_path": f"{args.output_split}/valid_mask/{sample_id}.png",
                            "aligned_target_path": (
                                f"{args.output_split}/aligned_target/{sample_id}.png"
                                if args.save_diagnostics else ""
                            ),
                        }
                        record = {
                            "split": args.output_split, "sample_id": sample_id,
                            "direction": f"{names[moving_id]}->{names[fixed_id]}",
                            "translation_direction": f"{src_name}->{tgt_name}",
                            "pair_direction": pair_direction,
                            "moving_id": moving_id, "fixed_id": fixed_id,
                            "moving_modality": names[moving_id],
                            "fixed_modality": names[fixed_id],
                            "source_path": str(source_path), **relative,
                            "seed": direction_seed + sample_index,
                        }
                        records.append(record)
                        required_output_keys = [
                            "moving_path", "fixed_path", "flow_path", "valid_mask_path",
                        ]
                        if args.save_diagnostics:
                            required_output_keys.append("aligned_target_path")
                        output_paths = [
                            output_dir / relative[key] for key in required_output_keys
                        ]
                        if all(path.is_file() for path in output_paths):
                            completed += 1
                        else:
                            missing_indices.append(local_index)
                    rows.extend(records)
                    if not missing_indices:
                        continue

                    source = load_batch(
                        [batch_paths[index] for index in missing_indices],
                        args.load_size, args.canvas_size,
                    ).to(device)
                    count = source.size(0)
                    source_ids = torch.full((count,), src_id, dtype=torch.long, device=device)
                    target_ids = torch.full((count,), tgt_id, dtype=torch.long, device=device)
                    fake = model(source, source_ids, target_ids)
                    velocities = []
                    for local_index in missing_indices:
                        np.random.seed(int(records[local_index]["seed"]))
                        velocity, _, _ = generate_diffeomorphic_flow(
                            (args.canvas_size, args.canvas_size),
                            smooth_sigma=args.smooth_sigma,
                            max_displacement=args.max_displacement,
                            affine_probability=0.0,
                        )
                        velocities.append(torch.from_numpy(velocity))
                    velocity = torch.stack(velocities).to(device=device, dtype=source.dtype)
                    canvas_flow = integrate(velocity)
                    warped_source = canvas_spatial(source, canvas_flow)
                    warped_fake = canvas_spatial(fake, canvas_flow)
                    source = center_crop_tensor(source, args.crop_size)
                    fake = center_crop_tensor(fake, args.crop_size)
                    warped_source = center_crop_tensor(warped_source, args.crop_size)
                    warped_fake = center_crop_tensor(warped_fake, args.crop_size)
                    flow = center_crop_tensor(canvas_flow, args.crop_size)
                    # Recompute observability in final-crop coordinates.  The
                    # large canvas removes visible zero-fill borders; this mask
                    # excludes final pixels whose correspondence lies outside
                    # the cropped moving image.
                    valid = final_spatial(
                        final_ones.expand(count, -1, -1, -1), flow,
                    ) > 0.999
                    for batch_index, local_index in enumerate(missing_indices):
                        record = records[local_index]
                        if record["pair_direction"] == "source-moving":
                            moving, fixed = source[batch_index], warped_fake[batch_index]
                        else:
                            moving, fixed = fake[batch_index], warped_source[batch_index]
                        atomic_png(moving, output_dir / record["moving_path"])
                        atomic_png(fixed, output_dir / record["fixed_path"])
                        if args.save_diagnostics:
                            atomic_png(
                                fake[batch_index],
                                output_dir / record["aligned_target_path"],
                            )
                        atomic_flow(flow[batch_index], output_dir / record["flow_path"])
                        atomic_png(
                            valid[batch_index], output_dir / record["valid_mask_path"],
                            mask=True,
                        )
                        generated += 1
                        if generated % args.print_freq == 0:
                            ensure_space(output_dir, args.min_free_gb)
                            print(
                                f"Generated {generated}; reused {completed}; "
                                f"latest={record['sample_id']}", flush=True,
                            )

    rows.sort(key=lambda row: row["sample_id"])
    manifest_path = output_dir / f"manifest_{args.output_split}{suffix}.csv"
    atomic_manifest(rows, manifest_path)
    metadata["completed_samples"] = len(rows)
    metadata["generated_this_run"] = generated
    metadata["reused_this_run"] = completed
    atomic_json(metadata, config_path)
    print(
        f"Complete: samples={len(rows)} generated={generated} reused={completed}\n"
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
