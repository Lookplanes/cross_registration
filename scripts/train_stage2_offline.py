#!/usr/bin/env python3
"""Train modality-conditioned TransMorph from an offline Stage 2 manifest."""

from __future__ import annotations

import argparse
import json
import os
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
from torch.optim import Adam
from torch.utils.data import DataLoader

from crossreg.data.stage2_offline import OfflineStage2Dataset
from crossreg.registration.transmorph.conditioned_model import (
    ModalityConditionedTransMorph,
    config_from_transcut,
)
from crossreg.registration.transmorph.losses import Grad
from crossreg.registration.visualization import (
    RegistrationSampleCollector,
    prune_sample_snapshots,
    update_sample_alias,
)
from crossreg.utils.metrics import AverageMeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcut-ckpt", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--val-root")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--reg-weight", type=float, default=0.5)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--max-iters-per-epoch", type=int, default=0)
    parser.add_argument("--max-val-iters", type=int, default=0)
    parser.add_argument(
        "--val-samples-per-direction", type=int, default=1,
        help="fixed validation examples exported per observed direction; 0 disables",
    )
    parser.add_argument(
        "--val-sample-freq", type=int, default=1,
        help="export validation sample sheet every N validation runs",
    )
    parser.add_argument("--keep-val-sample-snapshots", type=int, default=5)
    parser.add_argument("--val-sample-flow-limit", type=float, default=15.0)
    parser.add_argument("--milestone-freq", type=int, default=10)
    parser.add_argument("--keep-milestones", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_model(
    checkpoint: dict, img_size: tuple[int, int], device: torch.device,
) -> ModalityConditionedTransMorph:
    saved = checkpoint["config"]
    input_nc = int(saved.get("input_nc", 3))
    config = config_from_transcut(saved, img_size, input_nc=input_nc)
    model = ModalityConditionedTransMorph(
        config,
        num_modalities=int(saved["num_modalities"]),
        id_embed_dim=int(saved.get("id_embed_dim", 64)),
        image_channels=input_nc,
    ).to(device)
    report = model.initialize_from_transcut(checkpoint)
    print(f"Initialized conditioned TransMorph: {report['copied_tensors']} tensors")
    return model


def make_optimizer(
    model: ModalityConditionedTransMorph, lr: float, backbone_lr: float,
) -> Adam:
    backbone_ids = {
        id(parameter)
        for module in (model.transformer, model.modality_embedding, model.pair_cln)
        for parameter in module.parameters()
    }
    backbone = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids]
    new_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    return Adam([
        {"params": backbone, "lr": backbone_lr},
        {"params": new_parameters, "lr": lr},
    ], amsgrad=True)


def make_loader(
    dataset: OfflineStage2Dataset, args: argparse.Namespace, shuffle: bool,
    device: torch.device,
) -> DataLoader:
    options = {
        "batch_size": args.batch_size, "shuffle": shuffle,
        "num_workers": args.num_workers, "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        options.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, **options)


def flow_mse(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise only correspondences observable inside the final moving crop."""
    valid = valid_mask > 0.5
    valid = valid.expand(-1, prediction.size(1), -1, -1)
    if not torch.any(valid):
        raise ValueError("offline Stage 2 batch contains no valid flow pixels")
    return (prediction - target).square()[valid].mean()


@torch.inference_mode()
def validate(
    model: ModalityConditionedTransMorph, loader: DataLoader,
    device: torch.device, max_iters: int,
    sample_collector: RegistrationSampleCollector | None = None,
) -> dict[str, object]:
    model.eval()
    total = defaultdict(float)
    directions: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for index, batch in enumerate(loader):
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)
        target = batch["flow"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True).squeeze(1)
        moving_ids = batch["moving_id"].to(device, non_blocking=True)
        fixed_ids = batch["fixed_id"].to(device, non_blocking=True)
        prediction = model.predict_flow(moving, fixed, moving_ids, fixed_ids)
        if prediction.shape[-2:] != target.shape[-2:]:
            old_h, old_w = prediction.shape[-2:]
            new_h, new_w = target.shape[-2:]
            prediction = F.interpolate(
                prediction, size=(new_h, new_w), mode="bilinear", align_corners=False,
            )
            prediction[:, 0] *= new_h / old_h
            prediction[:, 1] *= new_w / old_w
        if sample_collector is not None:
            selected = [
                sample for sample in range(moving.size(0))
                if sample_collector.wants(str(batch["direction"][sample]))
            ]
            if selected:
                indices = torch.tensor(selected, device=device)
                warped = model.spatial_trans(
                    moving.index_select(0, indices),
                    prediction.index_select(0, indices),
                )
                for local_index, sample in enumerate(selected):
                    sample_collector.add(
                        sample_id=str(batch["sample_id"][sample]),
                        direction=str(batch["direction"][sample]),
                        moving=moving[sample], fixed=fixed[sample],
                        warped=warped[local_index], target_flow=target[sample],
                        predicted_flow=prediction[sample], valid_mask=mask[sample],
                    )
        error = torch.linalg.vector_norm(prediction - target, dim=1)
        zero = torch.linalg.vector_norm(target, dim=1)
        for sample in range(moving.size(0)):
            valid = mask[sample] > 0.5
            sample_error = error[sample]
            sample_zero = zero[sample]
            values = {
                "epe": float(sample_error.mean()),
                "valid_epe": float(sample_error[valid].mean()),
                "zero_epe": float(sample_zero.mean()),
                "valid_zero_epe": float(sample_zero[valid].mean()),
                "pck_1": float((sample_error[valid] <= 1).float().mean()),
                "pck_2": float((sample_error[valid] <= 2).float().mean()),
                "pck_4": float((sample_error[valid] <= 4).float().mean()),
            }
            key = str(batch["direction"][sample])
            for metric, value in values.items():
                total[metric] += value
                directions[key][metric] += value
            total["samples"] += 1
            directions[key]["samples"] += 1
        if max_iters and index + 1 >= max_iters:
            break
    samples = total["samples"]
    result = {
        key: (value if key == "samples" else value / samples)
        for key, value in total.items()
    }
    result["directions"] = {
        key: {
            metric: (value if metric == "samples" else value / values["samples"])
            for metric, value in values.items()
        }
        for key, values in sorted(directions.items())
    }
    return result


def prune_milestones(directory: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(
        directory.glob("checkpoint_epoch_*.pth"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    for path in checkpoints[:-keep]:
        path.unlink()


def main() -> None:
    args = parse_args()
    if bool(args.val_manifest) != bool(args.val_root):
        raise ValueError("--val-manifest and --val-root must be provided together")
    if args.val_interval < 1 or args.batch_size < 1:
        raise ValueError("validation interval and batch size must be positive")
    if args.val_samples_per_direction < 0 or args.val_sample_freq < 1:
        raise ValueError("sample count must be non-negative and sample frequency positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir).expanduser().resolve()
    experiments = save_dir / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    train_dataset = OfflineStage2Dataset(args.train_manifest, args.train_root)
    train_loader = make_loader(train_dataset, args, True, device)
    val_loader = None
    if args.val_manifest:
        val_dataset = OfflineStage2Dataset(args.val_manifest, args.val_root)
        val_loader = make_loader(val_dataset, args, False, device)
    print(f"Offline train samples: {len(train_dataset)}")
    if val_loader is not None:
        print(f"Offline validation samples: {len(val_loader.dataset)}")

    stage1_path = Path(args.transcut_ckpt).expanduser().resolve()
    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
    names = list(stage1["modality_names"])
    model = build_model(stage1, tuple(args.img_size), device)
    optimizer = make_optimizer(model, args.lr, args.backbone_lr)
    criterion_reg = Grad("l2", loss_mult=2).to(device)
    config = {
        **vars(args), "transcut_ckpt": str(stage1_path), "modality_names": names,
        "flow_parameterization": "backward_sampling_displacement_(dy,dx)_pixels",
        "objective": "MSE(flow_pred,flow_gt)+reg_weight*Grad(flow_pred)",
    }
    with (save_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    latest_path = experiments / "latest_checkpoint.pth"
    start_epoch = 0
    best = float("inf")
    if args.resume and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["registration_model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best = float(checkpoint.get("best_loss", best))
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        meter = AverageMeter()
        counts = torch.zeros(len(names), len(names), dtype=torch.long)
        for index, batch in enumerate(train_loader):
            moving = batch["moving"].to(device, non_blocking=True)
            fixed = batch["fixed"].to(device, non_blocking=True)
            target = batch["flow"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            moving_ids = batch["moving_id"].to(device, non_blocking=True)
            fixed_ids = batch["fixed_id"].to(device, non_blocking=True)
            prediction = model.predict_flow(moving, fixed, moving_ids, fixed_ids)
            if prediction.shape[-2:] != target.shape[-2:]:
                old_h, old_w = prediction.shape[-2:]
                new_h, new_w = target.shape[-2:]
                prediction = F.interpolate(
                    prediction, size=(new_h, new_w), mode="bilinear", align_corners=False,
                )
                prediction[:, 0] *= new_h / old_h
                prediction[:, 1] *= new_w / old_w
            loss_mse = flow_mse(prediction, target, valid_mask)
            loss_reg = criterion_reg(prediction, target) * args.reg_weight
            loss = loss_mse + loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meter.update(loss.detach().item(), moving.size(0))
            flat = moving_ids.detach().cpu() * len(names) + fixed_ids.detach().cpu()
            counts += torch.bincount(flat, minlength=len(names) ** 2).reshape(
                len(names), len(names),
            )
            if (index + 1) % args.print_freq == 0:
                print(
                    f"[{epoch + 1}/{args.epochs}] {index + 1}/{len(train_loader)} "
                    f"loss={loss.detach().item():.4f} "
                    f"mse={loss_mse.detach().item():.4f} "
                    f"reg={loss_reg.detach().item():.4f}", flush=True,
                )
            if args.max_iters_per_epoch and index + 1 >= args.max_iters_per_epoch:
                break
        print(f"Epoch {epoch + 1}: loss={meter.avg:.4f}")
        print("Directions: " + " ".join(
            f"{names[source]}->{names[target]}={int(counts[source, target])}"
            for source in range(len(names)) for target in range(len(names))
            if source != target
        ))

        validation = None
        sample_path = None
        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            validation_index = (epoch + 1) // args.val_interval
            export_samples = (
                args.val_samples_per_direction > 0
                and validation_index % args.val_sample_freq == 0
            )
            collector = (
                RegistrationSampleCollector(args.val_samples_per_direction)
                if export_samples else None
            )
            validation = validate(
                model, val_loader, device, args.max_val_iters,
                sample_collector=collector,
            )
            print("Validation: " + json.dumps(validation, sort_keys=True))
            if collector is not None:
                samples_dir = save_dir / "samples"
                sample_path = collector.save(
                    samples_dir / f"epoch_{epoch + 1:04d}.png",
                    flow_limit=args.val_sample_flow_limit,
                )
                update_sample_alias(sample_path, samples_dir / "latest.png")
                prune_sample_snapshots(
                    samples_dir, args.keep_val_sample_snapshots,
                )
                print(f"Validation samples: {sample_path}")
        with (save_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "epoch": epoch + 1, "train_loss": meter.avg,
                "validation": validation,
            }, sort_keys=True) + "\n")

        selection = validation["valid_epe"] if validation else meter.avg
        checkpoint = {
            "format_version": 2, "epoch": epoch + 1,
            "best_loss": min(best, selection),
            "model_type": "ModalityConditionedTransMorph",
            "model_config": {
                "img_size": list(args.img_size),
                "embed_dim": int(stage1["config"]["embed_dim"]),
                "flow_parameterization": "displacement",
                "registration_model": "conditioned_transmorph",
                "encoder_init": "checkpoint",
                "encoder_checkpoint": str(stage1_path),
                "generator_checkpoint": str(stage1_path),
                "offline_stage2": True,
                "train_manifest": str(Path(args.train_manifest).resolve()),
                "modality_names": names,
            },
            "registration_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if selection < best:
            best = selection
            checkpoint["best_loss"] = best
            torch.save(checkpoint, experiments / "model_best.pth")
            if sample_path is not None:
                update_sample_alias(sample_path, save_dir / "samples" / "best.png")
            print(f"Best model: selection={best:.4f}")
        torch.save(checkpoint, latest_path)
        if args.milestone_freq and (epoch + 1) % args.milestone_freq == 0:
            torch.save(checkpoint, experiments / f"checkpoint_epoch_{epoch + 1}.pth")
            prune_milestones(experiments, args.keep_milestones)

    print("Offline Stage 2 training finished.")


if __name__ == "__main__":
    main()
