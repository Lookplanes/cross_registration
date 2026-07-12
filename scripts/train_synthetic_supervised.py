#!/usr/bin/env python3
"""Stage 2: Synthetic supervised registration training.

Loads a pre-trained TransCUT encoder, freezes it, and trains a
Cross-Attention Registration Head with known deformation fields.

Workflow per iteration
----------------------
1. Take real image n₁ (source modality).
2. TransCUT decoder → n₂_fake (cross-modal, pixel-aligned to n₁).
3. Random deformation D_gt → n₂_fake' (perturbed).
4. Shared encoder: F₁ = enc(n₁, id₁),  F₂' = enc(n₂_fake', id₂).
5. RegHead(F₁, F₂') → D_pred.
6. Loss = MSE(D_pred, D_gt) + λ·Grad(D_pred).

Usage::

    python scripts/train_synthetic_supervised.py \
        --transcut-ckpt /path/to/transcut_encoder.pth \
        --data-dir /path/to/paired_images \
        --save-dir /path/to/output \
        --epochs 200 \
        --device cuda
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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from crossreg.data.datasets import PairedImageFolderDataset
from crossreg.data.perturbation import generate_diffeomorphic_flow
from crossreg.models.swin_transformer import SwinTransformer
from crossreg.registration.transmorph.cross_attn_head import CrossAttentionRegHead
from crossreg.registration.transmorph.losses import Grad
from crossreg.registration.transmorph.model import SpatialTransformer, VecInt
from crossreg.translation.transcut.transcut_model import SwinEncoderWithCLN
from crossreg.translation.transcut.cln_adain import ModalityIDEmbedding
from crossreg.translation.transcut.decoder import TransCUTDecoder
from crossreg.utils.metrics import AverageMeter, compute_epe


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: Synthetic Supervised Training")
    p.add_argument("--transcut-ckpt", required=True, help="TransCUT checkpoint")
    p.add_argument("--data-dir", required=True, help="Paired images for synthetic pairs")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-modalities", type=int, default=6)
    p.add_argument("--src-modality", type=int, default=0)
    p.add_argument("--tgt-modality", type=int, default=1)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--reg-weight", type=float, default=0.5)
    p.add_argument("--print-freq", type=int, default=10)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


# =============================================================================
# Model builder
# =============================================================================


def build_stage2_model(args: argparse.Namespace, device: torch.device):
    """Build encoder (from TransCUT ckpt, frozen) + RegHead (trainable)."""
    img_size = tuple(args.img_size)
    embed_dim = args.embed_dim

    # Modality ID embedding
    mod_embed = ModalityIDEmbedding(args.num_modalities, embed_dim // 2)
    mod_embed.load_state_dict(
        torch.load(args.transcut_ckpt, map_location=device, weights_only=True)["mod_embed"]
    )
    mod_embed.to(device)

    # Encoder (shared)
    encoder = SwinTransformer(
        pretrain_img_size=img_size[0], in_chans=1, embed_dim=embed_dim,
        depths=(2, 2, 4, 2), num_heads=(4, 4, 8, 8),
        window_size=(8, 8), patch_norm=False,
    )
    # Load TransCUT encoder weights
    ckpt = torch.load(args.transcut_ckpt, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"]["swin"])
    encoder.to(device)

    # Wrap with CLN (needed for modality-aware encoding)
    class EncoderWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = SwinEncoderWithCLN.__new__(SwinEncoderWithCLN)
            self.enc.swin = encoder
            self.enc.mod_embed = mod_embed
            self.enc.cln = nn.LayerNorm(embed_dim)  # simplified: use LayerNorm instead of CLN
        def forward(self, x, mod_id):
            return self.enc.swin(x)

    enc_wrapper = EncoderWrapper().to(device)

    # Decoder (for generating synthetic pairs — frozen)
    decoder = TransCUTDecoder(embed_dim=embed_dim, output_nc=1,
                              style_dim=embed_dim // 2, n_layers=4)
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
    decoder.to(device)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    # Style embedding (for decoder Adain)
    style_embed = nn.Linear(embed_dim // 2, embed_dim // 2)
    if "style_embed" in ckpt:
        style_embed.load_state_dict(ckpt["style_embed"])
    style_embed.to(device)

    # RegHead (trainable)
    reg_head = CrossAttentionRegHead(
        embed_dim=embed_dim, out_indices=(0, 1, 2, 3),
        reg_head_chan=16, num_heads=4,
    ).to(device)

    # Spatial transformer for warping
    spatial_trans = SpatialTransformer(img_size).to(device)

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, mod_embed, decoder, style_embed, reg_head, spatial_trans


# =============================================================================
# Synthetic data generation (on-the-fly)
# =============================================================================


def _generate_synthetic_pair(
    real_src: torch.Tensor,
    encoder: nn.Module,
    mod_embed: nn.Module,
    decoder: nn.Module,
    style_embed: nn.Module,
    src_id: int,
    tgt_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (fake, perturbed_fake, D_gt) from real source image.

    Returns
    -------
    fake : (B, 1, H, W)          Translated image (pixel-aligned to real_src).
    perturbed : (B, 1, H, W)     Deformed fake.
    flow_gt : (B, 2, H, W)       Ground-truth deformation field.
    """
    B, _, H, W = real_src.shape
    tgt_tensor = torch.full((B,), tgt_id, dtype=torch.long, device=device)
    src_tensor = torch.full((B,), src_id, dtype=torch.long, device=device)

    with torch.no_grad():
        # 1. Translate: real_src → fake (pixel-aligned, target modality style)
        feats = encoder(real_src)
        style = style_embed(mod_embed(tgt_tensor))
        fake = decoder(feats, style)
        # Upsample to match input resolution for deformation
        if fake.shape[-2:] != real_src.shape[-2:]:
            fake = F.interpolate(fake, size=real_src.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # 2. Generate random deformation on CPU (NumPy → Torch)
        flow_np, map_x, map_y = generate_diffeomorphic_flow(
            (H, W), smooth_sigma=12.0, max_displacement=15.0, affine_probability=0.8,
        )
        flow_gt = torch.from_numpy(flow_np).unsqueeze(0).to(device).float()  # (1, 2, H, W)
        map_x_t = torch.from_numpy(map_x).unsqueeze(0).unsqueeze(0).to(device).float()
        map_y_t = torch.from_numpy(map_y).unsqueeze(0).unsqueeze(0).to(device).float()

        # 3. Warp fake
        perturbed = F.grid_sample(
            fake, torch.cat([map_x_t, map_y_t], dim=1).permute(0, 2, 3, 1) * 2 / (W - 1) - 1,
            mode='bilinear', align_corners=True, padding_mode='zeros',
        )

    return fake, perturbed, flow_gt


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(os.path.join(args.save_dir, "experiments"), exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    ds = PairedImageFolderDataset(
        args.data_dir, img_size=tuple(args.img_size), grayscale=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)
    print(f"Training pairs: {len(ds)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    encoder, mod_embed, decoder, style_embed, reg_head, spatial_trans = \
        build_stage2_model(args, device)

    # Optimiser (only reg_head)
    optimizer = Adam(reg_head.parameters(), lr=args.lr, amsgrad=True)
    criterion_reg = Grad("l2", loss_mult=2).to(device)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_loss = 1e10
    ckpt_path = os.path.join(args.save_dir, "experiments", "latest_checkpoint.pth")
    if args.resume and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        reg_head.load_state_dict(ck["reg_head"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"]
        best_loss = ck.get("best_loss", 1e10)
        print(f"Resumed from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        reg_head.train()
        loss_meter = AverageMeter()

        for idx, batch in enumerate(loader):
            real_src = batch[0].to(device)  # moving image (source modality)

            # Generate synthetic supervision
            fake, perturbed, flow_gt = _generate_synthetic_pair(
                real_src, encoder, mod_embed, decoder, style_embed,
                args.src_modality, args.tgt_modality, device,
            )

            # Encode both images through shared (frozen) encoder
            with torch.no_grad():
                f1 = encoder(real_src)         # (B, C, H, W) at each scale
                f2 = encoder(perturbed)

            # RegHead predicts flow
            flow_pred = reg_head(f1, f2)  # (B, 2, H', W')

            # Resize flow_pred to match flow_gt
            if flow_pred.shape[-2:] != flow_gt.shape[-2:]:
                scale_h = flow_gt.shape[-2] / flow_pred.shape[-2]
                scale_w = flow_gt.shape[-1] / flow_pred.shape[-1]
                flow_pred = F.interpolate(flow_pred, size=flow_gt.shape[-2:],
                                          mode='bilinear', align_corners=False)
                flow_pred[:, 0] *= scale_h
                flow_pred[:, 1] *= scale_w

            # Loss: MSE + Grad regularisation
            loss_mse = F.mse_loss(flow_pred, flow_gt)
            loss_reg = criterion_reg(flow_pred, flow_gt) * args.reg_weight
            loss = loss_mse + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), real_src.size(0))

            if (idx + 1) % args.print_freq == 0:
                print(f"  [{epoch+1}/{args.epochs}] iter {idx+1}/{len(loader)} "
                      f"loss={loss.item():.4f} (MSE={loss_mse.item():.4f} reg={loss_reg.item():.4f})")

        print(f"Epoch {epoch+1}: avg_loss={loss_meter.avg:.4f}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1, "best_loss": best_loss,
            "reg_head": reg_head.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, ckpt_path)

        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            torch.save(ckpt, os.path.join(args.save_dir, "experiments", "model_best.pth"))
            print(f"  → Best model (loss={best_loss:.4f})")

        if (epoch + 1) % 20 == 0:
            torch.save(ckpt, os.path.join(
                args.save_dir, "experiments", f"checkpoint_epoch_{epoch+1}.pth"))

    print("Stage 2 training finished.")


if __name__ == "__main__":
    main()
