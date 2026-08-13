#!/usr/bin/env python3
"""Check whether a Stage 2 model uses sample-level moving/fixed correspondence."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for path in (_ROOT, _SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crossreg.data.datasets import PairedImageFolderDataset
from crossreg.data.translation import MultiDomainTranslationDataset
from crossreg.registration.transmorph.model import VecInt
from scripts.train_synthetic_supervised import (
    _construct_cross_modal_pair,
    _generate_synthetic_pair,
    _normalize_stage1_input,
    _predict_registration_flow,
    _resize_displacement,
    _stage2_batch,
    build_stage2_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-ckpt", required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--data-dir")
    data.add_argument("--modality-dirs", nargs="+")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    return parser.parse_args()


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = torch.linalg.vector_norm(prediction - target, dim=1)
    cosine = F.cosine_similarity(
        prediction.flatten(1), target.flatten(1), dim=1, eps=1e-8,
    )
    return {
        "epe": float(error.mean()),
        "flow_cosine": float(cosine.mean()),
        "pck_1": float((error <= 1).float().mean()),
        "pck_2": float((error <= 2).float().mean()),
        "pck_4": float((error <= 4).float().mean()),
    }


def _within_direction_permutation(
    moving_ids: torch.Tensor, fixed_ids: torch.Tensor,
) -> torch.Tensor:
    """Shuffle samples without changing any sample's modality-ID pair."""
    permutation = torch.arange(moving_ids.numel(), device=moving_ids.device)
    for moving_id in torch.unique(moving_ids):
        for fixed_id in torch.unique(fixed_ids):
            indices = torch.nonzero(
                (moving_ids == moving_id) & (fixed_ids == fixed_id),
                as_tuple=False,
            ).flatten()
            if indices.numel() > 1:
                permutation[indices] = torch.roll(indices, shifts=1)
    return permutation


@torch.inference_mode()
def evaluate_direction(
    loader: DataLoader,
    modules: tuple,
    config: SimpleNamespace,
    direction: str,
    device: torch.device,
    max_iters: int,
) -> dict[str, dict[str, float]]:
    pair_encoder, pair_mod_embed, decoder, style_embed, encoder, model, spatial = modules
    integrate = VecInt(tuple(config.img_size), nsteps=7).to(device)
    model.eval()
    variants = ("correct", "shuffled_moving", "shuffled_fixed", "zero_moving")
    totals = {name: {} for name in variants}
    counts = 0
    shuffled_samples = 0
    random.seed(config.seed + 100_000)
    np.random.seed(config.seed + 100_000)
    torch.manual_seed(config.seed + 100_000)

    for index, batch in enumerate(loader):
        source, source_ids, target_ids = _stage2_batch(batch, config, device)
        fake, target_flow = _generate_synthetic_pair(
            source, pair_encoder, pair_mod_embed, decoder, style_embed,
            source_ids, target_ids, device, integrate,
        )
        moving, fixed, moving_ids, fixed_ids = _construct_cross_modal_pair(
            source, fake, target_flow, source_ids, target_ids, spatial, direction,
        )
        permutation = _within_direction_permutation(moving_ids, fixed_ids)
        shuffled_samples += int(torch.count_nonzero(
            permutation != torch.arange(source.size(0), device=device),
        ))
        inputs = {
            "correct": (moving, fixed, moving_ids, fixed_ids),
            "shuffled_moving": (
                moving[permutation], fixed, moving_ids[permutation], fixed_ids,
            ),
            "shuffled_fixed": (
                moving, fixed[permutation], moving_ids, fixed_ids[permutation],
            ),
            "zero_moving": (torch.zeros_like(moving), fixed, moving_ids, fixed_ids),
        }
        predictions = {}
        for name, (variant_moving, variant_fixed, variant_mids, variant_fids) in inputs.items():
            prediction = _predict_registration_flow(
                model, encoder, variant_moving, variant_fixed,
                variant_mids, variant_fids, config.registration_model,
            )
            prediction = _resize_displacement(prediction, target_flow.shape[-2:])
            predictions[name] = prediction
            batch_metrics = _metrics(prediction, target_flow)
            for key, value in batch_metrics.items():
                totals[name][key] = totals[name].get(key, 0.0) + value * source.size(0)
        for name in variants:
            change = torch.linalg.vector_norm(
                predictions[name] - predictions["correct"], dim=1,
            ).mean()
            totals[name]["prediction_change_from_correct"] = (
                totals[name].get("prediction_change_from_correct", 0.0)
                + float(change) * source.size(0)
            )
        counts += source.size(0)
        if index + 1 >= max_iters:
            break
    result = {
        name: {key: value / counts for key, value in metrics.items()}
        for name, metrics in totals.items()
    }
    result["diagnostic"] = {
        "samples": counts,
        "same_id_pair_shuffle_fraction": shuffled_samples / counts,
    }
    return result


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cli.stage2_ckpt, map_location=device, weights_only=False)
    saved = checkpoint["model_config"]
    config = SimpleNamespace(
        transcut_ckpt=saved["encoder_checkpoint"],
        generator_ckpt=saved["generator_checkpoint"],
        encoder_init=saved.get("encoder_init", "checkpoint"),
        img_size=saved["img_size"],
        num_modalities=None,
        embed_dim=saved["embed_dim"],
        src_modality=saved["src_modality"],
        tgt_modality=saved["tgt_modality"],
        seed=42,
        registration_model=saved["registration_model"],
        fusion_residual=saved.get("fusion_residual", "none"),
        modality_names=saved.get("modality_names"),
    )
    modules = build_stage2_model(config, device)
    modules[5].load_state_dict(checkpoint["registration_model"], strict=True)
    stage1 = torch.load(
        config.transcut_ckpt, map_location="cpu", weights_only=False,
    )
    input_nc = stage1.get("config", {}).get("input_nc", 1)
    if input_nc not in (1, 3):
        raise ValueError(f"unsupported Stage 1 input_nc={input_nc}")
    if cli.modality_dirs:
        if len(cli.modality_dirs) != stage1["config"]["num_modalities"]:
            raise ValueError("modality directory count does not match checkpoint")
        dataset = MultiDomainTranslationDataset(
            cli.modality_dirs, input_nc=input_nc,
            load_size=config.img_size[0], crop_size=config.img_size[0],
            pairing_mode="unpaired",
        )
    else:
        dataset = PairedImageFolderDataset(
            cli.data_dir, img_size=tuple(config.img_size),
            grayscale=(input_nc == 1),
        )
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False)
    result = {
        direction: evaluate_direction(
            loader, modules, config, direction, device, cli.max_iters,
        )
        for direction in ("source-moving", "target-moving")
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if cli.output:
        Path(cli.output).write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
