#!/usr/bin/env python3
"""Train CUT for cross-modality image translation.

Usage (crossreg conda env)::

    python scripts/train_translation.py \
        --dataroot /path/to/paired_images \
        --name experiment_name \
        --save-dir /path/to/output \
        --input-nc 1 --output-nc 1 \
        --n-epochs 200 --n-epochs-decay 200

The input ``dataroot`` should contain ``trainA/`` and ``trainB/``
subdirectories with matching filenames (or ``A/``, ``B/``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torch
from torch.utils.data import DataLoader

from crossreg.data.translation import TwoDomainTranslationDataset
from crossreg.config import parse_args_with_config, save_resolved_config
from crossreg.translation.cut import CUTConfig, CUTWrapper
from crossreg.utils.metrics import AverageMeter

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CUT for modality translation")
    p.add_argument("--dataroot", required=True, help="Root dir with trainA/ trainB/")
    p.add_argument("--name", default="cut_experiment")
    p.add_argument("--save-dir", default="./output/cut")
    p.add_argument("--input-nc", type=int, default=1)
    p.add_argument("--output-nc", type=int, default=1)
    p.add_argument("--pairing-mode", choices=["unpaired", "paired"],
                   default="unpaired")
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--netF-nc", type=int, default=256)
    p.add_argument("--netG", default="resnet_9blocks")
    p.add_argument("--nce-layers", type=int, nargs="+",
                   default=[0, 4, 8, 12, 16])
    p.add_argument("--lambda-GAN", type=float, default=1.0)
    p.add_argument("--lambda-NCE", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--n-epochs-decay", type=int, default=200)
    p.add_argument("--load-size", type=int, default=286)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--print-freq", type=int, default=100)
    p.add_argument("--save-epoch-freq", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    return parse_args_with_config(
        p, sections=("data", "model", "training", "checkpoint"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)
    save_resolved_config(args, os.path.join(save_dir, "resolved_config.yaml"), {
        "data": ("dataroot", "pairing_mode", "load_size", "crop_size", "num_workers"),
        "model": ("input_nc", "output_nc", "ngf", "ndf", "netF_nc", "netG", "nce_layers"),
        "training": ("name", "save_dir", "batch_size", "n_epochs", "n_epochs_decay", "lr",
                     "lambda_GAN", "lambda_NCE", "print_freq", "device"),
        "checkpoint": ("resume", "save_epoch_freq"),
    })

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = CUTConfig(
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        netG=args.netG,
        ngf=args.ngf,
        ndf=args.ndf,
        netF_nc=args.netF_nc,
        nce_layers=list(args.nce_layers),
        lambda_GAN=args.lambda_GAN,
        lambda_NCE=args.lambda_NCE,
        lr=args.lr,
        n_epochs=args.n_epochs,
        n_epochs_decay=args.n_epochs_decay,
        gpu_ids=[0] if device.type == "cuda" else [],
        nce_idt=True,
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    trainA_dir = os.path.join(args.dataroot, "trainA")
    trainB_dir = os.path.join(args.dataroot, "trainB")
    if not os.path.isdir(trainA_dir):
        # Fallback: look for A/ B/
        trainA_dir = os.path.join(args.dataroot, "A")
        trainB_dir = os.path.join(args.dataroot, "B")

    dataset = TwoDomainTranslationDataset(
        trainA_dir, trainB_dir,
        input_nc=args.input_nc, output_nc=args.output_nc,
        load_size=args.load_size,
        crop_size=args.crop_size,
        pairing_mode=args.pairing_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    print(f"Dataset: {len(dataset)} pairs")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = CUTWrapper(config)

    # Resume
    start_epoch = 0
    ckpt_path = os.path.join(save_dir, "latest_checkpoint.pth")
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        # PatchSampleF creates its MLPs lazily; materialise them before loading.
        init_batch = next(iter(loader))
        model.initialize_netF(init_batch)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_training_state(ck)
        start_epoch = ck.get("epoch", 0)
        print(f"Resumed at epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    total_epochs = args.n_epochs + args.n_epochs_decay
    loss_G_GAN_m = AverageMeter()
    loss_G_NCE_m = AverageMeter()
    loss_D_m = AverageMeter()
    t_start = time.time()

    for epoch in range(start_epoch, total_epochs):
        loss_G_GAN_m.reset()
        loss_G_NCE_m.reset()
        loss_D_m.reset()
        epoch_iter = 0

        for data in loader:
            epoch_iter += args.batch_size
            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_current_losses()
            loss_G_GAN_m.update(losses.get("G_GAN", 0), args.batch_size)
            loss_G_NCE_m.update(losses.get("NCE", 0), args.batch_size)
            loss_D_m.update(
                (losses.get("D_real", 0) + losses.get("D_fake", 0)) * 0.5,
                args.batch_size,
            )

            if epoch_iter % args.print_freq == 0:
                print(f"  [{epoch+1}/{total_epochs}] "
                      f"iter {epoch_iter} "
                      f"G_GAN={loss_G_GAN_m.avg:.4f} "
                      f"G_NCE={loss_G_NCE_m.avg:.4f} "
                      f"D={loss_D_m.avg:.4f}")

        t_epoch = time.time() - t_start
        print(f"Epoch {epoch+1}/{total_epochs} "
              f"({t_epoch:.0f}s) "
              f"G_GAN={loss_G_GAN_m.avg:.4f} "
              f"G_NCE={loss_G_NCE_m.avg:.4f} "
              f"D={loss_D_m.avg:.4f}")

        # Save periodically
        if (epoch + 1) % args.save_epoch_freq == 0:
            model.save_networks(save_dir, epoch + 1)

        # Latest checkpoint
        state = model.training_state()
        state["epoch"] = epoch + 1
        torch.save(state, ckpt_path)

        # LR decay
        model.update_learning_rate()

    # Final save
    model.save_networks(save_dir, total_epochs)
    print("Training finished.")


if __name__ == "__main__":
    main()
