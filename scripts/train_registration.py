#!/usr/bin/env python3
"""Train TransMorph for intra-modality registration (unsupervised).

Usage (crossreg conda env)::

    python scripts/train_registration.py \
        --train-dir /path/to/Train_CrossModal_unsup_full \
        --val-dir /path/to/Train_CrossModal/Val \
        --save-dir /path/to/output \
        --batch-size 64 \
        --epochs 400

The input data should be structured as::

    train_dir/
        ch0_to_ch1/
            moving/   fixed/   gt_flow/   valid_mask/
        ch1_to_ch0/
            moving/   fixed/   gt_flow/   valid_mask/
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

# Ensure src/ is on PYTHONPATH
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM

from crossreg.data.datasets import MultiModalityPairedDataset
from crossreg.config import parse_args_with_config, save_resolved_config
from crossreg.registration.transmorph.model import TransMorph, CONFIGS
from crossreg.registration.transmorph.losses import NCC_vxm, Grad
from crossreg.utils.metrics import AverageMeter, compute_epe, build_foreground_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransMorph (unsupervised)")
    p.add_argument("--train-dir", required=True, help="Root of training data")
    p.add_argument("--val-dir", default=None, help="Root of validation data")
    p.add_argument("--save-dir", required=True, help="Output directory")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--ncc-weight", type=float, default=1.0)
    p.add_argument("--reg-weight", type=float, default=1.0)
    p.add_argument("--print-freq", type=int, default=10)
    p.add_argument("--val-interval", type=int, default=1,
                   help="Validate every N epochs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    return parse_args_with_config(
        p, sections=("data", "model", "training", "loss", "validation", "checkpoint"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str) -> None:
    torch.save(state, path)


def _ncc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Zero-mean NCC computed over valid (mask > 0) pixels."""
    if mask is not None:
        valid = mask.sum()
        if valid == 0:
            return torch.tensor(0.0, device=pred.device)
        pred_mean = (pred * mask).sum() / valid
        tgt_mean = (target * mask).sum() / valid
        p_c = (pred - pred_mean) * mask
        t_c = (target - tgt_mean) * mask
    else:
        pred_mean = pred.mean()
        tgt_mean = target.mean()
        p_c = pred - pred_mean
        t_c = target - tgt_mean

    cross = (p_c * t_c).sum()
    p_var = (p_c ** 2).sum()
    t_var = (t_c ** 2).sum()
    return cross / (torch.sqrt(p_var * t_var) + eps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.join(args.save_dir, "experiments"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "logs"), exist_ok=True)
    save_resolved_config(args, os.path.join(args.save_dir, "resolved_config.yaml"), {
        "data": ("train_dir", "val_dir", "img_size", "num_workers"),
        "model": (),
        "training": ("save_dir", "batch_size", "epochs", "lr", "print_freq", "device"),
        "loss": ("ncc_weight", "reg_weight"),
        "validation": ("val_interval",),
        "checkpoint": ("resume",),
    })

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    config = copy.deepcopy(CONFIGS["TransMorph"])
    config.in_chans = 2
    config.img_size = tuple(args.img_size)
    model = TransMorph(config).to(device)

    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs — DataParallel")
        model = nn.DataParallel(model)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_set = MultiModalityPairedDataset(
        root_dir=args.train_dir,
        img_size=tuple(args.img_size),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    print(f"Train: {len(train_set)} pairs")

    val_loader = None
    if args.val_dir and os.path.isdir(args.val_dir):
        val_set = MultiModalityPairedDataset(
            root_dir=args.val_dir,
            img_size=tuple(args.img_size),
        )
        val_loader = DataLoader(val_set, batch_size=min(args.batch_size, 50),
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=(device.type == "cuda"))
        print(f"Val: {len(val_set)} pairs")

    # ------------------------------------------------------------------
    # Losses & Optimiser
    # ------------------------------------------------------------------
    criterion_sim = NCC_vxm(win=[9, 9]).to(device)
    criterion_reg = Grad("l2", loss_mult=2).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, amsgrad=True)
    ssim_calc = SSIM(data_range=1.0, size_average=True, channel=1)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_epe = 1e10
    top_best: list[dict] = []
    ckpt_path = os.path.join(args.save_dir, "experiments", "latest_checkpoint.pth")

    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"]
        best_epe = ck.get("best_epe", 1e10)
        top_best = ck.get("top_best_models", [])
        print(f"Resumed at epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_all = AverageMeter()
        loss_sim_m = AverageMeter()
        loss_reg_m = AverageMeter()

        for idx, data in enumerate(train_loader):
            moving = data["moving"].to(device)
            fixed = data["fixed"].to(device)

            warped, flow, _ = model(moving, fixed)

            loss_sim = criterion_sim(fixed, warped) * args.ncc_weight
            loss_reg = criterion_reg(flow, fixed) * args.reg_weight
            loss = loss_sim + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all.update(loss.item(), moving.size(0))
            loss_sim_m.update(loss_sim.item(), moving.size(0))
            loss_reg_m.update(loss_reg.item(), moving.size(0))

            if (idx + 1) % args.print_freq == 0:
                print(f"  [{epoch+1}/{args.epochs}] "
                      f"Iter {idx+1}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f} "
                      f"NCC: {loss_sim.item():.4f} "
                      f"Reg: {loss_reg.item():.4f}")

        print(f"Epoch {epoch+1}: loss={loss_all.avg:.4f} "
              f"(sim={loss_sim_m.avg:.4f} reg={loss_reg_m.avg:.4f})")

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            model.eval()
            eval_epe = AverageMeter()
            eval_ssim = AverageMeter()
            eval_ncc = AverageMeter()

            with torch.no_grad():
                for data in val_loader:
                    moving = data["moving"].to(device)
                    fixed = data["fixed"].to(device)
                    warped, flow, _ = model(moving, fixed)

                    # Mask
                    valid_mask = build_foreground_mask(fixed, threshold=0.01)
                    if "valid_mask" in data:
                        valid_mask = data["valid_mask"].to(device)

                    # EPE
                    if "flow" in data:
                        epe = compute_epe(flow, data["flow"].to(device), valid_mask)
                        eval_epe.update(epe, moving.size(0))

                    # SSIM
                    val_ssim = ssim_calc(warped, fixed)
                    eval_ssim.update(val_ssim.item(), moving.size(0))

                    # ZNCC
                    val_ncc = _ncc_loss(warped, fixed, valid_mask)
                    eval_ncc.update(val_ncc.item(), moving.size(0))

            # Use ZNCC (higher=better) as primary metric if EPE unavailable
            current = -eval_ncc.avg if eval_epe.count == 0 else eval_epe.avg
            print(f"  Val Epoch {epoch+1}: "
                  f"EPE={eval_epe.avg:.4f} "
                  f"SSIM={eval_ssim.avg:.4f} "
                  f"ZNCC={eval_ncc.avg:.4f}")

            # Maintain top-5 best models
            if len(top_best) < 5 or current < max(m["metric"] for m in top_best):
                fname = f"model_best_{current:.4f}_epoch_{epoch+1}.pth"
                save_path = os.path.join(args.save_dir, "experiments", fname)
                save_checkpoint({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_epe": min(best_epe, current),
                    "top_best_models": top_best,
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                top_best.append({"metric": current, "epoch": epoch + 1, "path": fname})
                top_best.sort(key=lambda x: x["metric"])
                if len(top_best) > 5:
                    old = top_best.pop()
                    old_path = os.path.join(args.save_dir, "experiments", old["path"])
                    if os.path.exists(old_path):
                        os.remove(old_path)
                print(f"  → Saved to top-5 (metric={current:.4f})")

            # Absolute best
            if current < best_epe:
                best_epe = current
                save_checkpoint({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_epe": best_epe,
                    "optimizer": optimizer.state_dict(),
                }, os.path.join(args.save_dir, "experiments", "model_best.pth"))

        # Latest checkpoint
        save_checkpoint({
            "format_version": 1,
            "model_type": "TransMorph",
            "model_config": {"name": "TransMorph", "img_size": list(args.img_size), "in_chans": 2},
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_epe": best_epe,
            "top_best_models": top_best,
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)

        # Periodic cold backup
        if (epoch + 1) % 20 == 0:
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_epe": best_epe,
                "optimizer": optimizer.state_dict(),
            }, os.path.join(args.save_dir, "experiments",
                            f"checkpoint_epoch_{epoch+1}.pth"))

    print("Training finished.")


if __name__ == "__main__":
    main()
