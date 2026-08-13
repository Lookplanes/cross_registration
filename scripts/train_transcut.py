#!/usr/bin/env python3
"""Stage 1: Train TransCUT for N-to-N cross-modality translation.

Each iteration randomly picks two DIFFERENT modalities, loads a random image
from each, and trains the translation src→tgt.  The shared Swin-Transformer
encoder and CNN decoder learn ALL N modalities simultaneously through
CLN/AdaIN modality-ID injection.

Data format::

    dataroot/
        2PM/              ← modality 0 images
        Confocal/         ← modality 1 images
        Fluorescence/     ← modality 2 images
        HE/               ← modality 3 images
        MACSima/          ← modality 4 images
        MSI/              ← modality 5 images

Usage::

    python scripts/train_transcut.py \\
        --dataroot /data2/wuyh/processed \\
        --save-dir /path/to/output \\
        --epochs 200 --device cuda

Or with explicit modality directory list::

    python scripts/train_transcut.py \\
        --modality-dirs /data/2PM,/data/Confocal,/data/Fluorescence,/data/HE,/data/MACSima,/data/MSI \\
        --save-dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from crossreg.data.modalities import (
    ModalitySpec, load_modality_registry, save_modality_registry,
)
from crossreg.data.translation import MultiDomainTranslationDataset
from crossreg.translation.transcut import TransCUT, TransCUTConfig
from crossreg.utils.metrics import AverageMeter


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransCUT (Stage 1 — N-to-N)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataroot", help="Root with modality subdirs")
    g.add_argument("--modality-dirs", help="Comma-separated paths to each modality")
    g.add_argument("--modality-config", help="YAML registry with stable modality IDs")
    p.add_argument("--modality-names",
                   default="2PM,Confocal,Fluorescence,HE,MACSima,MSI",
                   help="Subdir names under --dataroot (comma-separated)")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--input-nc", type=int, choices=[1, 3], default=None,
                   help="Common channel count; defaults to registry value or 1")
    p.add_argument("--pairing-mode", choices=["unpaired", "paired"],
                   default="unpaired")
    p.add_argument("--split-manifest",
                   help="patches.csv used to select only one declared split")
    p.add_argument("--manifest-root",
                   help="Root prepended to relative patch_path entries")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--paired-anchor-manifest",
        help=(
            "Optional long-form manifest of aligned samples used as sparse "
            "supervision inside unpaired training"
        ),
    )
    p.add_argument(
        "--paired-anchor-probability", type=float, default=0.0,
        help="Probability that an unpaired item uses an aligned anchor",
    )
    p.add_argument("--min-image-mean", type=float, default=5.0)
    p.add_argument("--min-image-std", type=float, default=5.0)
    p.add_argument("--max-dark-fraction", type=float, default=0.9,
                   help="Reject images with a larger fraction of pixels below 5")
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument(
        "--decoder-variant",
        choices=["legacy", "highres_content", "fullres_residual"],
        default="legacy",
        help=(
            "legacy preserves the original checkpoint-compatible decoder; "
            "highres_content adds shared normalized 128/256px content skips; "
            "fullres_residual learns target-conditioned changes on a direct "
            "full-resolution source carrier"
        ),
    )
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--num-patches", type=int, default=128,
                   help="NCE patches (64 for small feature maps)")
    p.add_argument("--load-size", type=int, default=286)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="Batches prefetched per DataLoader worker (workers>0 only)",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n-epochs-decay", type=int, default=200)
    p.add_argument("--max-iters-per-epoch", type=int, default=0,
                   help="Optional smoke-test cap; 0 uses the full dataset")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-D", type=float, default=None)
    p.add_argument("--d-update-freq", type=int, default=2)
    p.add_argument("--lambda-GAN", type=float, default=1.0)
    p.add_argument("--lambda-NCE", type=float, default=1.0)
    p.add_argument("--lambda-identity", type=float, default=1.0,
                   help="Independent same-domain L1 identity weight")
    p.add_argument(
        "--lambda-paired", type=float, default=0.0,
        help=(
            "Aligned target L1 anchor; requires paired training or a sparse "
            "paired-anchor manifest"
        ),
    )
    p.add_argument(
        "--lambda-cycle", type=float, default=0.0,
        help=(
            "EXPERIMENTAL source reconstruction after translation; assumes "
            "approximate cross-domain invertibility and requires "
            "--allow-experimental-cycle"
        ),
    )
    p.add_argument(
        "--allow-experimental-cycle", action="store_true",
        help="Acknowledge the non-invertibility risk of enabling --lambda-cycle",
    )
    p.add_argument("--lambda-structure", type=float, default=0.0,
                   help="Multi-scale gradient-magnitude preservation weight")
    p.add_argument(
        "--lambda-D-mismatch", type=float, default=0.0,
        help="Wrong-modality negative weight for the conditional discriminator",
    )
    p.add_argument("--nce-fake-modality", choices=["source", "target"],
                   default="target",
                   help="Condition used when encoding generated images for PatchNCE")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-freq", type=int, default=100)
    p.add_argument("--save-epoch-freq", type=int, default=20)
    p.add_argument("--sample-freq", type=int, default=5,
                   help="Export fixed original/translation samples every N epochs; 0 disables")
    p.add_argument("--sample-count", type=int, default=1,
                   help="Fixed source images exported per modality")
    p.add_argument(
        "--fixed-sample-manifest",
        help=(
            "Optional wide CSV for paired fixed samples; it must contain "
            "<modality>_path columns and one row per spatially paired sample"
        ),
    )
    p.add_argument(
        "--fixed-sample-root",
        help="Root prepended to relative paths in --fixed-sample-manifest",
    )
    p.add_argument(
        "--fixed-sample-split", default="test",
        help="Split selected from --fixed-sample-manifest (default: test)",
    )
    p.add_argument("--keep-sample-snapshots", type=int, default=3)
    p.add_argument("--keep-epoch-checkpoints", type=int, default=3)
    p.add_argument(
        "--milestone-freq", type=int, default=10,
        help="Permanently retain checkpoint/sample snapshots every N epochs; 0 disables",
    )
    p.add_argument(
        "--collapse-dark-gap", type=float, default=0.25,
        help="Warn when fake dark-pixel fraction exceeds real targets by this amount",
    )
    p.add_argument(
        "--collapse-min-samples", type=int, default=32,
        help="Minimum target-domain samples required before collapse warnings",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dist-backend", choices=["nccl", "gloo"], default=None,
                   help="DDP backend; defaults to nccl on CUDA and gloo on CPU")
    p.add_argument("--local-rank", "--local_rank", type=int, default=None,
                   help=argparse.SUPPRESS)
    checkpoint = p.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", action="store_true")
    checkpoint.add_argument("--init-checkpoint",
                            help="Initialize a new run from an existing checkpoint")
    p.add_argument("--expand-modalities", action="store_true",
                   help="Allow --init-checkpoint to add/remap modality embeddings")
    return p.parse_args()


class DistributedTransCUTStep(nn.Module):
    """Expose complete D/G loss graphs through one DDP-wrapped forward.

    ``TransCUT.optimize_parameters`` calls methods on the bare module and
    therefore cannot trigger DDP's forward/backward reducer.  This adapter is
    used only by the distributed training entry point; checkpoints continue
    to contain the unwrapped :class:`TransCUT` state.
    """

    def __init__(self, model: TransCUT):
        super().__init__()
        self.model = model

    def forward(
        self, real_src: torch.Tensor, real_tgt: torch.Tensor,
        src_id: torch.Tensor, tgt_id: torch.Tensor,
        paired_mask: torch.Tensor, phase: str,
    ) -> dict[str, torch.Tensor]:
        if phase == "D":
            with torch.no_grad():
                fake = self.model(real_src, src_id, tgt_id)
            return self.model.compute_D_loss_components(fake, real_tgt, tgt_id)
        if phase == "G":
            fake = self.model(real_src, src_id, tgt_id)
            return self.model.compute_G_loss_components(
                real_src, real_tgt, src_id, tgt_id, fake=fake,
                paired_mask=paired_mask,
            )
        raise ValueError(f"unsupported distributed phase: {phase}")


def _distributed_context(args: argparse.Namespace) -> tuple[bool, int, int, torch.device]:
    """Initialise torchrun-style DDP and select this process's device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get(
        "LOCAL_RANK", args.local_rank if args.local_rank is not None else 0,
    ))

    wants_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if distributed:
        backend = args.dist_backend or ("nccl" if wants_cuda else "gloo")
        if backend == "nccl" and not wants_cuda:
            raise RuntimeError("NCCL DDP requires CUDA")
        if wants_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        init_kwargs = {"backend": backend, "init_method": "env://"}
        if backend == "nccl":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)
    else:
        device = torch.device(args.device if wants_cuda else "cpu")
    return distributed, rank, world_size, device


def _distributed_optimize(
    ddp_step: DDP, model: TransCUT, real_src: torch.Tensor,
    real_tgt: torch.Tensor, src_id: torch.Tensor, tgt_id: torch.Tensor,
    paired_mask: torch.Tensor,
) -> dict[str, float]:
    """Run one synchronized discriminator/generator optimization step."""
    model.clear_batch_diagnostics()
    update_discriminator = model._optimization_step % model.config.d_update_freq == 0
    if update_discriminator:
        model.set_requires_grad(model.netD, True)
        model.optimizer_D.zero_grad()
        d_losses = ddp_step(
            real_src, real_tgt, src_id, tgt_id, paired_mask, "D",
        )
        d_losses["D"].backward()
        model.optimizer_D.step()
    else:
        model.set_requires_grad(model.netD, False)
        with torch.no_grad():
            d_losses = model.compute_D_loss_components(
                model(real_src, src_id, tgt_id), real_tgt, tgt_id,
            )

    model.set_requires_grad(model.netD, False)
    model.optimizer_G.zero_grad()
    if model.optimizer_F is not None:
        model.optimizer_F.zero_grad()
    g_losses = ddp_step(
        real_src, real_tgt, src_id, tgt_id, paired_mask, "G",
    )
    g_losses["G"].backward()
    model.optimizer_G.step()
    if model.optimizer_F is not None:
        model.optimizer_F.step()
    model._optimization_step += 1
    return {
        name: float(value.detach())
        for name, value in {**g_losses, **d_losses}.items()
    }


def _reduce_epoch_statistics(
    meters: dict[str, AverageMeter], pair_counts: Counter[tuple[int, int]],
    num_modalities: int, device: torch.device,
) -> None:
    """Convert rank-local epoch statistics into global sums and counts."""
    for meter in meters.values():
        values = torch.tensor([meter.sum, meter.count], dtype=torch.float64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        meter.sum = float(values[0].item())
        meter.count = int(values[1].item())
        meter.avg = meter.sum / meter.count if meter.count else 0.0

    counts = torch.zeros(
        (num_modalities, num_modalities), dtype=torch.long, device=device,
    )
    for (src, tgt), count in pair_counts.items():
        counts[src, tgt] = count
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    pair_counts.clear()
    for src in range(num_modalities):
        for tgt in range(num_modalities):
            if src != tgt:
                pair_counts[(src, tgt)] = int(counts[src, tgt].item())


GROUPED_METRICS = (
    "G_GAN", "NCE", "identity", "paired", "paired_anchor", "cycle",
    "D", "D_real", "D_fake", "D_mismatch",
    "fake_mean", "fake_std", "fake_dark_fraction",
    "real_mean", "real_std", "real_dark_fraction",
)


def _new_grouped_statistics(
    num_modalities: int, device: torch.device,
) -> dict[str, torch.Tensor]:
    metric_count = len(GROUPED_METRICS)
    return {
        "target_sums": torch.zeros(
            num_modalities, metric_count, dtype=torch.float64, device=device,
        ),
        "target_counts": torch.zeros(
            num_modalities, dtype=torch.long, device=device,
        ),
        "direction_sums": torch.zeros(
            num_modalities, num_modalities, metric_count,
            dtype=torch.float64, device=device,
        ),
        "direction_counts": torch.zeros(
            num_modalities, num_modalities, dtype=torch.long, device=device,
        ),
    }


def _update_grouped_statistics(
    statistics: dict[str, torch.Tensor],
    diagnostics: dict[str, torch.Tensor],
    src_id: torch.Tensor,
    tgt_id: torch.Tensor,
) -> None:
    missing = [name for name in GROUPED_METRICS if name not in diagnostics]
    if missing:
        raise RuntimeError(
            "TransCUT omitted per-sample diagnostics: " + ", ".join(missing)
        )
    values = torch.stack(
        [diagnostics[name].to(dtype=torch.float64) for name in GROUPED_METRICS],
        dim=1,
    )
    if values.size(0) != src_id.numel():
        raise RuntimeError("per-sample diagnostics do not match the training batch")
    statistics["target_sums"].index_add_(0, tgt_id, values)
    statistics["target_counts"].index_add_(
        0, tgt_id, torch.ones_like(tgt_id, dtype=torch.long),
    )
    num_modalities = statistics["target_counts"].numel()
    flat_direction = src_id * num_modalities + tgt_id
    statistics["direction_sums"].view(
        num_modalities * num_modalities, -1,
    ).index_add_(0, flat_direction, values)
    statistics["direction_counts"].view(-1).index_add_(
        0, flat_direction, torch.ones_like(flat_direction, dtype=torch.long),
    )


def _reduce_grouped_statistics(
    statistics: dict[str, torch.Tensor],
) -> None:
    for tensor in statistics.values():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _metric_record(
    sums: torch.Tensor, count: int,
) -> dict[str, float | int]:
    record: dict[str, float | int] = {"count": count}
    if count:
        means = sums / count
        record.update({
            name: float(means[index].item())
            for index, name in enumerate(GROUPED_METRICS)
        })
    return record


def _grouped_records(
    statistics: dict[str, torch.Tensor], names: list[str],
) -> tuple[dict[str, dict[str, float | int]],
           dict[str, dict[str, float | int]]]:
    target_records = {}
    direction_records = {}
    for target, name in enumerate(names):
        count = int(statistics["target_counts"][target].item())
        target_records[name] = _metric_record(
            statistics["target_sums"][target], count,
        )
    for source, source_name in enumerate(names):
        for target, target_name in enumerate(names):
            if source == target:
                continue
            count = int(
                statistics["direction_counts"][source, target].item()
            )
            direction_records[f"{source_name}->{target_name}"] = _metric_record(
                statistics["direction_sums"][source, target], count,
            )
    return target_records, direction_records


def _print_grouped_statistics(
    target_records: dict[str, dict[str, float | int]],
    direction_records: dict[str, dict[str, float | int]],
    collapse_dark_gap: float,
    collapse_min_samples: int,
) -> None:
    for name, record in target_records.items():
        if not record["count"]:
            continue
        print(
            f"Target {name}: n={record['count']} "
            f"GAN={record['G_GAN']:.4f} NCE={record['NCE']:.4f} "
            f"Idt={record['identity']:.4f} Pair={record['paired']:.4f} "
            f"Anchor={record['paired_anchor']:.3f} "
            f"Cycle={record['cycle']:.4f} "
            f"D={record['D']:.4f} Dreal={record['D_real']:.4f} "
            f"Dfake={record['D_fake']:.4f} "
            f"Dwrong={record['D_mismatch']:.4f} "
            f"fake(mean={record['fake_mean']:.4f},std={record['fake_std']:.4f},"
            f"dark={record['fake_dark_fraction']:.3f}) "
            f"real(mean={record['real_mean']:.4f},std={record['real_std']:.4f},"
            f"dark={record['real_dark_fraction']:.3f})"
        )
        dark_gap = (
            float(record["fake_dark_fraction"])
            - float(record["real_dark_fraction"])
        )
        if record["count"] >= collapse_min_samples and dark_gap > collapse_dark_gap:
            print(
                f"WARNING target={name}: fake dark-pixel fraction exceeds "
                f"real targets by {dark_gap:.3f}; possible dark-output collapse"
            )
    for direction, record in direction_records.items():
        if not record["count"]:
            continue
        print(
            f"Direction {direction}: n={record['count']} "
            f"GAN={record['G_GAN']:.4f} NCE={record['NCE']:.4f} "
            f"Pair={record['paired']:.4f} "
            f"Anchor={record['paired_anchor']:.3f} "
            f"Cycle={record['cycle']:.4f} "
            f"D={record['D']:.4f} "
            f"Dwrong={record['D_mismatch']:.4f} "
            f"fake_mean={record['fake_mean']:.4f} "
            f"fake_std={record['fake_std']:.4f} "
            f"fake_dark={record['fake_dark_fraction']:.3f}"
        )


def _fixed_tensor(path: Path, channels: int, load_size: int,
                  crop_size: int) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        image = image.convert(mode)
        image = TF.resize(
            image, [load_size, load_size], interpolation=InterpolationMode.BICUBIC,
        )
        image = TF.center_crop(image, [crop_size, crop_size])
        tensor = TF.to_tensor(image)
    return TF.normalize(tensor, [0.5] * channels, [0.5] * channels)


def select_manifest_files(
    specs: list[ModalitySpec], manifest_path: str, manifest_root: str,
    split: str,
) -> list[list[Path]]:
    root = Path(manifest_root).expanduser().absolute()
    selected = [[] for _ in specs]
    parents = {Path(spec.path).resolve(): index for index, spec in enumerate(specs)}
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split:
                continue
            path = (root / row["patch_path"]).absolute()
            modality_index = parents.get(path.parent.resolve())
            if modality_index is not None:
                selected[modality_index].append(path)
    for spec, paths in zip(specs, selected):
        if not paths:
            raise RuntimeError(
                f"manifest selected no {split!r} images for modality {spec.name!r}"
            )
    return selected


def filter_low_information_images(
    modality_files: list[list[Path]], min_mean: float, min_std: float,
    max_dark_fraction: float,
) -> list[list[Path]]:
    if not 0.0 <= max_dark_fraction <= 1.0:
        raise ValueError("--max-dark-fraction must be in [0, 1]")
    if min_mean <= 0.0 and min_std <= 0.0 and max_dark_fraction >= 1.0:
        # The HEMIT launchers intentionally retain legitimate dark marker
        # patches.  Avoid reopening every image when all rejection criteria
        # are disabled.
        return modality_files
    filtered: list[list[Path]] = []
    for paths in modality_files:
        kept = []
        for path in paths:
            with Image.open(path) as image:
                small = TF.resize(image.convert("L"), [64, 64])
                pixels = np.asarray(small, dtype=np.float32)
            if pixels.mean() < min_mean or pixels.std() < min_std:
                continue
            if float((pixels < 5).mean()) > max_dark_fraction:
                continue
            kept.append(path)
        if not kept:
            raise RuntimeError("quality filtering removed every image from a modality")
        filtered.append(kept)
    return filtered


def _save_normalized_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().cpu().clamp(-1, 1).add(1).div(2)
    TF.to_pil_image(image).save(path)


def _prune(paths: list[Path], keep: int) -> None:
    if keep < 0:
        raise ValueError("retention counts must be non-negative")
    for path in paths[:-keep] if keep else paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _prune_epoch_artifacts(
    paths: list[Path], keep: int, milestone_freq: int,
) -> None:
    """Keep recent artifacts plus periodic milestones for unstable GAN runs."""
    if keep < 0 or milestone_freq < 0:
        raise ValueError("retention counts and milestone frequency must be non-negative")
    ordered = sorted(
        paths, key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    recent = set(ordered[-keep:] if keep else [])
    for path in ordered:
        epoch = int(path.stem.rsplit("_", 1)[1])
        is_milestone = milestone_freq > 0 and epoch % milestone_freq == 0
        if path in recent or is_milestone:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def export_fixed_samples(
    model: TransCUT, fixed_sources: list[list[tuple[Path, torch.Tensor]]],
    names: list[str], output_root: Path, epoch: int, device: torch.device,
    keep: int, milestone_freq: int,
) -> None:
    snapshot = output_root / "samples" / f"epoch_{epoch:04d}"
    snapshot.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for src_id, sources in enumerate(fixed_sources):
            for sample_index, (source_path, source) in enumerate(sources):
                prefix = f"src_{src_id}_{names[src_id]}_sample_{sample_index}"
                _save_normalized_image(
                    source, snapshot / f"{prefix}_original.png",
                )
                batch = source.unsqueeze(0).to(device)
                for tgt_id, target_name in enumerate(names):
                    if tgt_id == src_id:
                        continue
                    translated = model(batch, src_id, tgt_id)[0]
                    _save_normalized_image(
                        translated,
                        snapshot / f"{prefix}_to_{tgt_id}_{target_name}.png",
                    )
                (snapshot / f"{prefix}_source.txt").write_text(
                    str(source_path) + "\n", encoding="utf-8",
                )
    model.train(was_training)
    snapshots = list((output_root / "samples").glob("epoch_*"))
    _prune_epoch_artifacts(snapshots, keep, milestone_freq)
    print(f"Exported fixed originals and translations: {snapshot}")


def load_paired_fixed_samples(
    manifest_path: str, manifest_root: str, split: str,
    names: list[str], channels: int, load_size: int, crop_size: int,
    count: int, seed: int,
) -> list[tuple[str, list[tuple[Path, torch.Tensor]]]]:
    """Load deterministic, spatially paired fixed samples from a wide CSV."""
    root = Path(manifest_root).expanduser().absolute()
    required = ["patch_id", *(f"{name}_path" for name in names)]
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"fixed sample manifest is missing columns: {missing}"
            )
        rows = [
            row for row in reader
            if not row.get("split") or row["split"] == split
        ]
    if not rows:
        raise RuntimeError(
            f"fixed sample manifest selected no rows for split {split!r}"
        )
    selected = random.Random(seed).sample(rows, min(count, len(rows)))
    samples = []
    for row in selected:
        modalities = []
        for name in names:
            path = (root / row[f"{name}_path"]).absolute()
            if not path.is_file():
                raise FileNotFoundError(f"paired fixed sample does not exist: {path}")
            modalities.append((
                path,
                _fixed_tensor(path, channels, load_size, crop_size),
            ))
        samples.append((row["patch_id"], modalities))
    return samples


def export_paired_fixed_samples(
    model: TransCUT,
    paired_samples: list[tuple[str, list[tuple[Path, torch.Tensor]]]],
    names: list[str], output_root: Path, epoch: int, device: torch.device,
    keep: int, milestone_freq: int,
) -> None:
    """Export aligned real references and every directed translation together."""
    snapshot = output_root / "samples" / f"epoch_{epoch:04d}"
    snapshot.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for sample_index, (patch_id, modalities) in enumerate(paired_samples):
            sample_dir = snapshot / f"paired_sample_{sample_index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            source_paths = {"patch_id": patch_id, "modalities": {}}
            for src_id, ((source_path, source), source_name) in enumerate(
                zip(modalities, names)
            ):
                _save_normalized_image(
                    source, sample_dir / f"real_{src_id}_{source_name}.png",
                )
                source_paths["modalities"][source_name] = str(source_path)
                batch = source.unsqueeze(0).to(device)
                for tgt_id, target_name in enumerate(names):
                    if tgt_id == src_id:
                        continue
                    translated = model(batch, src_id, tgt_id)[0]
                    _save_normalized_image(
                        translated,
                        sample_dir / (
                            f"fake_{src_id}_{source_name}"
                            f"_to_{tgt_id}_{target_name}.png"
                        ),
                    )
            (sample_dir / "paired_sources.json").write_text(
                json.dumps(source_paths, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    model.train(was_training)
    snapshots = list((output_root / "samples").glob("epoch_*"))
    _prune_epoch_artifacts(snapshots, keep, milestone_freq)
    print(f"Exported paired fixed references and translations: {snapshot}")


def prune_epoch_checkpoints(
    save_dir: Path, keep: int, milestone_freq: int,
) -> None:
    _prune_epoch_artifacts(
        list(save_dir.glob("transcut_epoch_*.pth")),
        keep,
        milestone_freq,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.sample_count < 1:
        raise ValueError("--sample-count must be at least 1")
    if args.milestone_freq < 0:
        raise ValueError("--milestone-freq must be non-negative")
    if not 0.0 <= args.collapse_dark_gap <= 1.0:
        raise ValueError("--collapse-dark-gap must be in [0, 1]")
    if args.collapse_min_samples < 1:
        raise ValueError("--collapse-min-samples must be at least 1")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be at least 1")
    if args.lambda_cycle > 0.0 and not args.allow_experimental_cycle:
        raise ValueError(
            "--lambda-cycle is experimental and assumes approximate cross-domain "
            "invertibility; add --allow-experimental-cycle only after documenting "
            "why that assumption is valid"
        )
    if not 0.0 <= args.paired_anchor_probability <= 1.0:
        raise ValueError("--paired-anchor-probability must be in [0, 1]")
    uses_sparse_anchors = (
        args.pairing_mode == "unpaired"
        and args.paired_anchor_manifest is not None
        and args.paired_anchor_probability > 0.0
    )
    if args.paired_anchor_probability > 0.0 and not uses_sparse_anchors:
        raise ValueError(
            "positive --paired-anchor-probability requires unpaired mode and "
            "--paired-anchor-manifest"
        )
    if args.paired_anchor_manifest and args.paired_anchor_probability <= 0.0:
        raise ValueError(
            "--paired-anchor-manifest requires positive "
            "--paired-anchor-probability"
        )
    if args.lambda_paired > 0.0 and not (
        args.pairing_mode == "paired" or uses_sparse_anchors
    ):
        raise ValueError(
            "--lambda-paired requires paired mode or a sparse paired-anchor "
            "manifest so source and target are spatially aligned"
        )
    if uses_sparse_anchors and args.lambda_paired <= 0.0:
        raise ValueError("sparse paired anchors require positive --lambda-paired")
    distributed, rank, world_size, device = _distributed_context(args)
    is_main = rank == 0
    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    if is_main:
        mode = f"DDP world_size={world_size}" if distributed else "single process"
        print(f"Device: {device} ({mode})")
        os.makedirs(args.save_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    # --- Resolve stable modality registry ---
    if args.modality_config:
        specs = load_modality_registry(args.modality_config)
        registry_channels = specs[0].channels
        if args.input_nc is not None and args.input_nc != registry_channels:
            raise RuntimeError(
                f"--input-nc={args.input_nc} conflicts with registry channels={registry_channels}"
            )
        args.input_nc = registry_channels
    elif args.dataroot:
        names = [n.strip() for n in args.modality_names.split(",")]
        modality_dirs = [os.path.join(args.dataroot, n) for n in names]
        for i, (n, d) in enumerate(zip(names, modality_dirs)):
            if not os.path.isdir(d):
                for entry in os.listdir(args.dataroot):
                    if entry.lower() == n.lower():
                        modality_dirs[i] = os.path.join(args.dataroot, entry)
                        break
        args.input_nc = args.input_nc or 1
        specs = [
            ModalitySpec(index, name, str(Path(directory).resolve()), args.input_nc)
            for index, (name, directory) in enumerate(zip(names, modality_dirs))
        ]
    else:
        modality_dirs = [d.strip() for d in args.modality_dirs.split(",")]
        args.input_nc = args.input_nc or 1
        specs = [
            ModalitySpec(index, Path(directory).name, str(Path(directory).resolve()), args.input_nc)
            for index, directory in enumerate(modality_dirs)
        ]

    modality_dirs = [spec.path for spec in specs]
    names = [spec.name for spec in specs]
    if is_main:
        save_modality_registry(
            specs, Path(args.save_dir) / "modality_registry.yaml",
        )

    num_modalities = len(modality_dirs)
    if is_main:
        print(f"N = {num_modalities} modalities:")
    for i, (name, d) in enumerate(zip(names, modality_dirs)):
        if is_main:
            print(f"  [{i}] {name}: {d}")
        if not os.path.isdir(d):
            raise RuntimeError(f"Directory not found: {d}")

    # --- Data ---
    modality_files = None
    if args.split_manifest:
        if not args.manifest_root:
            raise RuntimeError("--split-manifest requires --manifest-root")
        modality_files = select_manifest_files(
            specs, args.split_manifest, args.manifest_root, args.split,
        )
    else:
        modality_files = [
            sorted(
                path for path in Path(directory).iterdir()
                if path.is_file() and path.suffix.lower() in
                {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            )
            for directory in modality_dirs
        ]
    before_quality = [len(paths) for paths in modality_files]
    modality_files = filter_low_information_images(
        modality_files, args.min_image_mean, args.min_image_std,
        args.max_dark_fraction,
    )
    if is_main:
        print(
            "Selected images after split/QC: "
            + ", ".join(
                f"{name}={after}/{before}"
                for name, after, before in zip(
                    names, map(len, modality_files), before_quality,
                )
            )
        )
    paired_anchor_files = None
    if uses_sparse_anchors:
        if not args.manifest_root:
            raise RuntimeError(
                "--paired-anchor-manifest requires --manifest-root"
            )
        paired_anchor_files = select_manifest_files(
            specs, args.paired_anchor_manifest, args.manifest_root, args.split,
        )
    dataset = MultiDomainTranslationDataset(
        modality_dirs, input_nc=args.input_nc,
        load_size=args.load_size, crop_size=args.crop_size,
        pairing_mode=args.pairing_mode,
        modality_files=modality_files,
        paired_anchor_files=paired_anchor_files,
        paired_anchor_probability=args.paired_anchor_probability,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
        seed=args.seed, drop_last=False,
    ) if distributed else None
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options.update({
            "persistent_workers": True,
            "prefetch_factor": args.prefetch_factor,
        })
    loader = DataLoader(dataset, **loader_options)
    if is_main:
        per_rank = len(loader)
        print(f"Dataset: {len(dataset)} samples; {per_rank} iters/rank/epoch")
        if uses_sparse_anchors:
            print(
                f"Sparse paired anchors: {len(dataset.paired_anchor_stems)} "
                f"unique aligned patches; sample probability="
                f"{args.paired_anchor_probability:g}"
            )
    fixed_rng = random.Random(args.seed)
    fixed_sources: list[list[tuple[Path, torch.Tensor]]] = []
    paired_fixed_samples: list[
        tuple[str, list[tuple[Path, torch.Tensor]]]
    ] = []
    if is_main:
        if args.fixed_sample_manifest:
            if not args.fixed_sample_root:
                raise ValueError(
                    "--fixed-sample-root is required with "
                    "--fixed-sample-manifest"
                )
            paired_fixed_samples = load_paired_fixed_samples(
                args.fixed_sample_manifest, args.fixed_sample_root,
                args.fixed_sample_split, names, args.input_nc,
                args.load_size, args.crop_size, args.sample_count, args.seed,
            )
            print(
                f"Paired fixed samples: {len(paired_fixed_samples)} rows "
                f"from {args.fixed_sample_manifest}"
            )
        else:
            for paths in dataset.paths:
                selected = fixed_rng.sample(paths, min(args.sample_count, len(paths)))
                fixed_sources.append([
                    (path, _fixed_tensor(
                        path, args.input_nc, args.load_size, args.crop_size,
                    ))
                    for path in selected
                ])

    # --- Model ---
    config = TransCUTConfig(
        num_modalities=num_modalities, input_nc=args.input_nc,
        output_nc=args.input_nc, img_size=args.crop_size,
        embed_dim=args.embed_dim, decoder_variant=args.decoder_variant,
        ndf=args.ndf,
        num_patches=args.num_patches,
        nce_fake_modality=args.nce_fake_modality,
        lambda_GAN=args.lambda_GAN, lambda_NCE=args.lambda_NCE,
        lambda_identity=args.lambda_identity,
        lambda_paired=args.lambda_paired,
        lambda_cycle=args.lambda_cycle,
        lambda_structure=args.lambda_structure,
        lambda_D_mismatch=args.lambda_D_mismatch,
        lr=args.lr, lr_D=args.lr_D, d_update_freq=args.d_update_freq,
        n_epochs=args.epochs, n_epochs_decay=args.n_epochs_decay,
        gpu_ids=[device.index if device.index is not None else 0]
        if device.type == "cuda" else [],
    )
    if is_main:
        objective = {
            "pairing_mode": args.pairing_mode,
            "nce_fake_modality": args.nce_fake_modality,
            "decoder_variant": args.decoder_variant,
            "lambda_GAN": args.lambda_GAN,
            "lambda_NCE": args.lambda_NCE,
            "lambda_identity": args.lambda_identity,
            "lambda_paired": args.lambda_paired,
            "paired_anchor_manifest": args.paired_anchor_manifest,
            "paired_anchor_probability": args.paired_anchor_probability,
            "lambda_cycle_EXPERIMENTAL": args.lambda_cycle,
            "lambda_structure_EXPERIMENTAL": args.lambda_structure,
            "lambda_D_mismatch": args.lambda_D_mismatch,
        }
        print("Effective training objective: " + json.dumps(objective, sort_keys=True))
        with (Path(args.save_dir) / "run_config.json").open(
            "w", encoding="utf-8",
        ) as handle:
            json.dump({
                "argv": sys.argv,
                "cli": vars(args),
                "model": vars(config),
                "modalities": names,
                "world_size": world_size,
                "effective_global_batch_size": args.batch_size * world_size,
                "objective": objective,
            }, handle, indent=2, sort_keys=True)
            handle.write("\n")
    model = TransCUT(config)
    model.set_modality_names(names)

    if args.expand_modalities and not args.init_checkpoint:
        raise RuntimeError("--expand-modalities requires --init-checkpoint")

    # Resume
    start_epoch = 0
    ckpt_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
    if args.resume and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        init_batch = next(iter(loader))
        init_src = init_batch["A"].to(device)
        init_src_id = init_batch["src_id"].to(device=device, dtype=torch.long)
        model.initialize_netF(init_src, init_src_id)
        model.load_training_state(ck)
        start_epoch = ck.get("epoch", 0)
        if is_main:
            print(f"Resumed from epoch {start_epoch}")
    elif args.init_checkpoint:
        ck = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        init_batch = next(iter(loader))
        init_src = init_batch["A"].to(device)
        init_src_id = init_batch["src_id"].to(device=device, dtype=torch.long)
        model.initialize_netF(init_src, init_src_id)
        changed = model.load_training_state(
            ck, load_optimizers=False,
            allow_modality_expansion=args.expand_modalities,
        )
        action = "expanded/remapped modalities" if changed else "matching modalities"
        if is_main:
            print(f"Initialized from {args.init_checkpoint} ({action}); optimizers reset")

    # DDP must see netF's lazily-created MLP parameters before construction.
    # Initialising on every rank also creates optimizer_F before synchronization.
    if distributed and config.lambda_NCE > 0.0 and not model._F_initialized:
        init_batch = next(iter(loader))
        model.initialize_netF(
            init_batch["A"].to(device),
            init_batch["src_id"].to(device=device, dtype=torch.long),
        )
    ddp_step = None
    if distributed:
        ddp_step = DDP(
            DistributedTransCUTStep(model).to(device),
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # --- Training ---
    total_epochs = args.epochs + args.n_epochs_decay
    loss_names = (
        "G", "G_GAN", "NCE", "identity", "paired", "cycle", "structure",
        "D", "D_real", "D_fake", "D_mismatch",
    )
    meters = {name: AverageMeter() for name in loss_names}

    if is_main and args.sample_freq and start_epoch == 0:
        if paired_fixed_samples:
            export_paired_fixed_samples(
                model, paired_fixed_samples, names, Path(args.save_dir), 0,
                device, args.keep_sample_snapshots, args.milestone_freq,
            )
        else:
            export_fixed_samples(
                model, fixed_sources, names, Path(args.save_dir), 0,
                device, args.keep_sample_snapshots, args.milestone_freq,
            )

    for epoch in range(start_epoch, total_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for meter in meters.values():
            meter.reset()
        pair_counts: Counter[tuple[int, int]] = Counter()
        grouped_statistics = _new_grouped_statistics(
            num_modalities, device,
        )

        for idx, batch in enumerate(loader):
            non_blocking = device.type == "cuda"
            real_A = batch["A"].to(device, non_blocking=non_blocking)
            real_B = batch["B"].to(device, non_blocking=non_blocking)
            src_id = batch["src_id"].to(
                device=device, dtype=torch.long, non_blocking=non_blocking,
            )
            tgt_id = batch["tgt_id"].to(
                device=device, dtype=torch.long, non_blocking=non_blocking,
            )
            paired_mask = batch["is_paired"].to(
                device=device, dtype=real_A.dtype, non_blocking=non_blocking,
            )

            losses = (
                _distributed_optimize(
                    ddp_step, model, real_A, real_B, src_id, tgt_id,
                    paired_mask,
                )
                if ddp_step is not None
                else model.optimize_parameters(
                    real_A, real_B, src_id, tgt_id,
                    paired_mask=paired_mask,
                )
            )
            _update_grouped_statistics(
                grouped_statistics, model.batch_diagnostics(), src_id, tgt_id,
            )
            for name, meter in meters.items():
                meter.update(losses[name], real_A.size(0))
            pair_counts.update(zip(src_id.tolist(), tgt_id.tolist()))

            if is_main and (idx + 1) % args.print_freq == 0:
                print(f"  [{epoch+1}/{total_epochs}] iter {idx+1} "
                      f"G={meters['G'].avg:.4f} "
                      f"GAN={meters['G_GAN'].avg:.4f} "
                      f"NCE={meters['NCE'].avg:.4f} "
                      f"Idt={meters['identity'].avg:.4f} "
                      f"Pair={meters['paired'].avg:.4f} "
                      f"Cycle={meters['cycle'].avg:.4f} "
                      f"Struct={meters['structure'].avg:.4f} "
                      f"D={meters['D'].avg:.4f} "
                      f"Anchors={int(paired_mask.sum().item())}/{real_A.size(0)} "
                      f"pairs={list(zip(src_id.tolist(), tgt_id.tolist()))}")
            if args.max_iters_per_epoch and idx + 1 >= args.max_iters_per_epoch:
                if is_main:
                    print(f"  Reached smoke-test cap: {args.max_iters_per_epoch} iterations")
                break

        if distributed:
            _reduce_epoch_statistics(meters, pair_counts, num_modalities, device)
            _reduce_grouped_statistics(grouped_statistics)

        summary = " ".join(f"{name}={meters[name].avg:.4f}" for name in loss_names)
        current_lr_g = model.optimizer_G.param_groups[0]["lr"]
        current_lr_d = model.optimizer_D.param_groups[0]["lr"]
        if is_main:
            print(
                f"Epoch {epoch+1}/{total_epochs}  {summary} "
                f"lr_G={current_lr_g:.8g} lr_D={current_lr_d:.8g}"
            )
        pair_summary = " ".join(
            f"{names[src]}->{names[tgt]}={pair_counts[(src, tgt)]}"
            for src in range(num_modalities)
            for tgt in range(num_modalities) if src != tgt
        )
        if is_main:
            print(f"Direction counts: {pair_summary}")
        target_records, direction_records = _grouped_records(
            grouped_statistics, names,
        )
        if is_main:
            _print_grouped_statistics(
                target_records, direction_records,
                args.collapse_dark_gap, args.collapse_min_samples,
            )
            metrics_path = Path(args.save_dir) / "metrics.jsonl"
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "lr_G": current_lr_g,
                    "lr_D": current_lr_d,
                    "overall": {
                        name: meters[name].avg for name in loss_names
                    },
                    "targets": target_records,
                    "directions": direction_records,
                }, sort_keys=True) + "\n")

        if is_main and (epoch + 1) % args.save_epoch_freq == 0:
            model.save(os.path.join(args.save_dir, f"transcut_epoch_{epoch+1}.pth"))
            prune_epoch_checkpoints(
                Path(args.save_dir), args.keep_epoch_checkpoints,
                args.milestone_freq,
            )
        if is_main:
            state = model.training_state()
            state["epoch"] = epoch + 1
            torch.save(state, ckpt_path)
        should_sample = args.sample_freq and (
            (epoch + 1) % args.sample_freq == 0 or epoch + 1 == total_epochs
        )
        if is_main and should_sample:
            if paired_fixed_samples:
                export_paired_fixed_samples(
                    model, paired_fixed_samples, names, Path(args.save_dir),
                    epoch + 1, device, args.keep_sample_snapshots,
                    args.milestone_freq,
                )
            else:
                export_fixed_samples(
                    model, fixed_sources, names, Path(args.save_dir), epoch + 1,
                    device, args.keep_sample_snapshots, args.milestone_freq,
                )
        model.update_learning_rate(epoch + 1)

    if is_main:
        model.save(os.path.join(args.save_dir, "transcut_final.pth"))
        print("Stage 1 training finished (N-to-N).")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
