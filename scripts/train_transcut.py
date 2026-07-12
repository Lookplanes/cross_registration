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
import os
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

from crossreg.translation.transcut import TransCUT, TransCUTConfig
from crossreg.utils.metrics import AverageMeter


# =============================================================================
# N-to-N translation dataset
# =============================================================================


class NtoNTranslationDataset(Dataset):
    """N-to-N unpaired translation dataset.

    Each ``__getitem__`` returns ``{A, B, src_id, tgt_id}`` where src_id and
    tgt_id are randomly chosen different modality indices.
    """

    _EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, modality_dirs: list[str],
                 input_nc: int = 1, load_size: int = 286, crop_size: int = 256):
        super().__init__()
        self.num_modalities = len(modality_dirs)
        self.paths: list[list[str]] = []
        for d in modality_dirs:
            p = sorted([os.path.join(d, f) for f in os.listdir(d)
                        if os.path.splitext(f)[1].lower() in self._EXTS])
            if not p:
                raise RuntimeError(f"No images in {d}")
            self.paths.append(p)

        t_list = [T.Resize(load_size), T.RandomHorizontalFlip(),
                  T.RandomCrop(crop_size), T.ToTensor()]
        t_list.append(T.Normalize((0.5,), (0.5,)) if input_nc == 1
                      else T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = T.Compose(t_list)
        self.mode = "L" if input_nc == 1 else "RGB"

    def __len__(self) -> int:
        return max(len(p) for p in self.paths) * self.num_modalities

    def __getitem__(self, _idx: int) -> dict:
        src_id = np.random.randint(0, self.num_modalities)
        tgt_id = np.random.randint(0, self.num_modalities)
        while tgt_id == src_id and self.num_modalities > 1:
            tgt_id = np.random.randint(0, self.num_modalities)

        src_path = self.paths[src_id][np.random.randint(len(self.paths[src_id]))]
        tgt_path = self.paths[tgt_id][np.random.randint(len(self.paths[tgt_id]))]

        return {
            "A": self.transform(Image.open(src_path).convert(self.mode)),
            "B": self.transform(Image.open(tgt_path).convert(self.mode)),
            "src_id": src_id,
            "tgt_id": tgt_id,
        }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransCUT (Stage 1 — N-to-N)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataroot", help="Root with modality subdirs")
    g.add_argument("--modality-dirs", help="Comma-separated paths to each modality")
    p.add_argument("--modality-names",
                   default="2PM,Confocal,Fluorescence,HE,MACSima,MSI",
                   help="Subdir names under --dataroot (comma-separated)")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--input-nc", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--num-patches", type=int, default=128,
                   help="NCE patches (64 for small feature maps)")
    p.add_argument("--load-size", type=int, default=286)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--n-epochs-decay", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-GAN", type=float, default=1.0)
    p.add_argument("--lambda-NCE", type=float, default=1.0)
    p.add_argument("--print-freq", type=int, default=100)
    p.add_argument("--save-epoch-freq", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Resolve modality directories ---
    if args.dataroot:
        names = [n.strip() for n in args.modality_names.split(",")]
        modality_dirs = [os.path.join(args.dataroot, n) for n in names]
        for i, (n, d) in enumerate(zip(names, modality_dirs)):
            if not os.path.isdir(d):
                for entry in os.listdir(args.dataroot):
                    if entry.lower() == n.lower():
                        modality_dirs[i] = os.path.join(args.dataroot, entry)
                        break
    else:
        modality_dirs = [d.strip() for d in args.modality_dirs.split(",")]

    num_modalities = len(modality_dirs)
    print(f"N = {num_modalities} modalities:")
    for i, d in enumerate(modality_dirs):
        print(f"  [{i}] {d}")
        if not os.path.isdir(d):
            raise RuntimeError(f"Directory not found: {d}")

    # --- Data ---
    dataset = NtoNTranslationDataset(
        modality_dirs, input_nc=args.input_nc,
        load_size=args.load_size, crop_size=args.crop_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2, pin_memory=True)
    print(f"Dataset: {len(dataset)} iters/epoch")

    # --- Model ---
    config = TransCUTConfig(
        num_modalities=num_modalities, input_nc=args.input_nc,
        embed_dim=args.embed_dim, ndf=args.ndf,
        num_patches=args.num_patches,
        lambda_GAN=args.lambda_GAN, lambda_NCE=args.lambda_NCE,
        lr=args.lr, n_epochs=args.epochs, n_epochs_decay=args.n_epochs_decay,
        gpu_ids=[0] if device.type == "cuda" else [],
    )
    model = TransCUT(config)

    # Resume
    start_epoch = 0
    ckpt_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
    if args.resume and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.encoder.load_state_dict(ck["encoder"])
        model.decoder.load_state_dict(ck["decoder"])
        model.mod_embed.load_state_dict(ck["mod_embed"])
        model.style_embed.load_state_dict(ck["style_embed"])
        start_epoch = ck.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    # --- Training ---
    total_epochs = args.epochs + args.n_epochs_decay
    loss_G_m = AverageMeter()
    loss_D_m = AverageMeter()

    for epoch in range(start_epoch, total_epochs):
        loss_G_m.reset()
        loss_D_m.reset()

        for idx, batch in enumerate(loader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            src_id = batch["src_id"].item()
            tgt_id = batch["tgt_id"].item()

            losses = model.optimize_parameters(real_A, real_B, src_id, tgt_id)
            loss_G_m.update(losses["G"], real_A.size(0))
            loss_D_m.update(losses["D"], real_A.size(0))

            if (idx + 1) % args.print_freq == 0:
                print(f"  [{epoch+1}/{total_epochs}] iter {idx+1} "
                      f"G={loss_G_m.avg:.4f} D={loss_D_m.avg:.4f} "
                      f"pair=({src_id}→{tgt_id})")

        print(f"Epoch {epoch+1}/{total_epochs}  G={loss_G_m.avg:.4f}  D={loss_D_m.avg:.4f}")

        if (epoch + 1) % args.save_epoch_freq == 0:
            model.save(os.path.join(args.save_dir, f"transcut_epoch_{epoch+1}.pth"))
        torch.save({
            "epoch": epoch + 1,
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "mod_embed": model.mod_embed.state_dict(),
            "style_embed": model.style_embed.state_dict(),
        }, ckpt_path)
        model.update_learning_rate()

    model.save(os.path.join(args.save_dir, "transcut_final.pth"))
    print("Stage 1 training finished (N-to-N).")


if __name__ == "__main__":
    main()
