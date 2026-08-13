#!/usr/bin/env python3
"""Evaluate Stage 2 on real paired modalities with known/zero deformation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from crossreg.data.perturbation import generate_diffeomorphic_flow
from crossreg.registration.transmorph.conditioned_model import (
    ModalityConditionedTransMorph,
    config_from_transcut,
)
from crossreg.registration.transmorph.model import SpatialTransformer, VecInt


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class RealPairedDirectionDataset(Dataset):
    def __init__(
        self, moving_dir: Path, fixed_dir: Path, stems: list[str],
        image_size: tuple[int, int],
    ) -> None:
        self.moving = self._by_stem(moving_dir)
        self.fixed = self._by_stem(fixed_dir)
        self.stems = stems
        self.image_size = image_size

    @staticmethod
    def _by_stem(directory: Path) -> dict[str, Path]:
        return {
            path.stem: path for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }

    def _read(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize(
            (self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array)).mul(2.0).sub(1.0)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[index]
        return self._read(self.moving[stem]), self._read(self.fixed[stem])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--samples-per-direction", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _common_stems(directories: list[Path]) -> list[str]:
    common: set[str] | None = None
    for directory in directories:
        stems = {
            path.stem for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        common = stems if common is None else common & stems
    return sorted(common or set())


def _flow_batch(
    batch_size: int, size: tuple[int, int], integrate: VecInt,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    velocities = []
    for _ in range(batch_size):
        velocity, _, _ = generate_diffeomorphic_flow(
            size, smooth_sigma=12.0, max_displacement=15.0,
            affine_probability=0.0,
        )
        velocities.append(torch.from_numpy(velocity))
    velocity = torch.stack(velocities).to(device=device, dtype=dtype)
    return integrate(velocity)


def _accumulate(
    totals: dict[str, float], prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    errors = torch.linalg.vector_norm(prediction - target, dim=1)
    batch_size = prediction.size(0)
    totals["epe"] += float(errors.mean()) * batch_size
    totals["pck_1"] += float((errors <= 1).float().mean()) * batch_size
    totals["pck_2"] += float((errors <= 2).float().mean()) * batch_size
    totals["pck_4"] += float((errors <= 4).float().mean()) * batch_size
    totals["mean_dy"] += float(prediction[:, 0].mean()) * batch_size
    totals["mean_dx"] += float(prediction[:, 1].mean()) * batch_size
    totals["samples"] += batch_size
    if torch.count_nonzero(target):
        cosine = F.cosine_similarity(
            prediction.flatten(1), target.flatten(1), dim=1, eps=1e-8,
        )
        totals["flow_cosine"] += float(cosine.mean()) * batch_size


def _finalize(totals: dict[str, float], include_cosine: bool) -> dict[str, float]:
    samples = totals["samples"]
    keys = ["epe", "pck_1", "pck_2", "pck_4", "mean_dy", "mean_dx"]
    if include_cosine:
        keys.append("flow_cosine")
    result = {key: totals[key] / samples for key in keys}
    result["samples"] = int(samples)
    return result


@torch.inference_mode()
def evaluate_direction(
    model: ModalityConditionedTransMorph,
    loader: DataLoader,
    moving_id: int,
    fixed_id: int,
    spatial: SpatialTransformer,
    integrate: VecInt,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, float]]:
    empty = {
        "epe": 0.0, "pck_1": 0.0, "pck_2": 0.0, "pck_4": 0.0,
        "mean_dy": 0.0, "mean_dx": 0.0, "flow_cosine": 0.0,
        "samples": 0.0,
    }
    perturbed = dict(empty)
    identity = dict(empty)
    zero_baseline = dict(empty)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for moving, aligned_fixed in loader:
        moving = moving.to(device)
        aligned_fixed = aligned_fixed.to(device)
        batch_size = moving.size(0)
        moving_ids = torch.full(
            (batch_size,), moving_id, dtype=torch.long, device=device,
        )
        fixed_ids = torch.full(
            (batch_size,), fixed_id, dtype=torch.long, device=device,
        )

        zero = torch.zeros(
            batch_size, 2, *moving.shape[-2:], device=device, dtype=moving.dtype,
        )
        identity_prediction = model.predict_flow(
            moving, aligned_fixed, moving_ids, fixed_ids,
        )
        _accumulate(identity, identity_prediction, zero)

        target_flow = _flow_batch(
            batch_size, tuple(moving.shape[-2:]), integrate, device, moving.dtype,
        )
        perturbed_fixed = spatial(aligned_fixed, target_flow)
        prediction = model.predict_flow(
            moving, perturbed_fixed, moving_ids, fixed_ids,
        )
        _accumulate(perturbed, prediction, target_flow)
        _accumulate(zero_baseline, zero, target_flow)

    return {
        "known_perturbation": _finalize(perturbed, include_cosine=True),
        "zero_flow_baseline": _finalize(zero_baseline, include_cosine=True),
        "no_perturbation_expected_zero": _finalize(identity, include_cosine=False),
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_direction < 1:
        raise ValueError("--samples-per-direction must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
    saved = checkpoint["model_config"]
    stage1 = torch.load(
        saved["encoder_checkpoint"], map_location="cpu", weights_only=False,
    )
    names = saved.get("modality_names") or stage1.get("modality_names")
    if not names:
        raise ValueError("checkpoint does not contain a modality registry")
    image_size = tuple(saved["img_size"])
    trans_config = config_from_transcut(
        stage1["config"], image_size,
        input_nc=stage1["config"].get("input_nc", 1),
    )
    model = ModalityConditionedTransMorph(
        trans_config, num_modalities=len(names),
        id_embed_dim=stage1["config"].get("id_embed_dim", 64),
        image_channels=stage1["config"].get("input_nc", 1),
    ).to(device)
    model.load_state_dict(checkpoint["registration_model"], strict=True)
    model.eval()
    spatial = SpatialTransformer(image_size).to(device)
    integrate = VecInt(image_size, nsteps=7).to(device)

    data_root = Path(args.data_root)
    directories = [data_root / name for name in names]
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    common = _common_stems(directories)
    if len(common) < args.samples_per_direction:
        raise ValueError(
            f"requested {args.samples_per_direction} samples but only "
            f"{len(common)} common stems exist"
        )
    stems = sorted(random.Random(args.seed).sample(common, args.samples_per_direction))

    directions = {}
    for moving_id, moving_name in enumerate(names):
        for fixed_id, fixed_name in enumerate(names):
            if moving_id == fixed_id:
                continue
            dataset = RealPairedDirectionDataset(
                directories[moving_id], directories[fixed_id], stems, image_size,
            )
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=(device.type == "cuda"),
            )
            key = f"{moving_name}->{fixed_name}"
            directions[key] = evaluate_direction(
                model, loader, moving_id, fixed_id, spatial, integrate,
                device, args.seed + 100_000,
            )
            print(key, json.dumps(directions[key], sort_keys=True), flush=True)

    result = {
        "checkpoint": str(Path(args.stage2_ckpt).resolve()),
        "data_root": str(data_root.resolve()),
        "common_stems": len(common),
        "selected_stems": stems,
        "directions": directions,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
