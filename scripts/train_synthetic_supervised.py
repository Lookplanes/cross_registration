#!/usr/bin/env python3
"""Stage 2: Synthetic supervised registration training.

Loads a pre-trained TransCUT encoder to initialize a modality-conditioned
TransMorph and trains it with known deformation fields.  The deprecated
Cross-Attention head remains available only for historical reproduction.

Workflow per iteration
----------------------
1. Take real image n₁ (source modality).
2. TransCUT decoder → n₂_fake (cross-modal, pixel-aligned to n₁).
3. Random displacement D_gt is applied to one aligned image.
4. Build a cross-modal pair satisfying warp(moving, D_gt) ≈ fixed.
5. Concatenate [fixed, moving] and condition joint patches with both IDs.
6. Original TransMorph Swin/decoder → D_pred.
7. Loss = MSE(D_pred, D_gt) + λ·Grad(D_pred).

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
import json
import os
import random
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
from crossreg.data.translation import MultiDomainTranslationDataset
from crossreg.data.perturbation import generate_diffeomorphic_flow
from crossreg.registration.transmorph.conditioned_model import (
    ModalityConditionedTransMorph,
    config_from_transcut,
)
from crossreg.registration.transmorph.deprecated_cross_attn_head import (
    CrossAttentionRegHead,
)
from crossreg.registration.transmorph.losses import Grad
from crossreg.registration.transmorph.model import SpatialTransformer, VecInt
from crossreg.translation.transcut.transcut_model import SwinEncoderWithCLN, TransCUTConfig
from crossreg.translation.transcut.cln_adain import ModalityIDEmbedding
from crossreg.translation.transcut.decoder import TransCUTDecoder
from crossreg.translation.transcut.highres_decoder import (
    HighResolutionContentDecoder,
)
from crossreg.translation.transcut.fullres_residual_decoder import (
    FullResolutionResidualDecoder,
)
from crossreg.utils.metrics import AverageMeter, compute_epe


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: Synthetic Supervised Training")
    p.add_argument("--transcut-ckpt", required=True, help="TransCUT checkpoint")
    p.add_argument(
        "--generator-ckpt",
        help=(
            "Optional fixed Stage 1 checkpoint used only to generate n2_fake; "
            "defaults to --transcut-ckpt"
        ),
    )
    p.add_argument(
        "--encoder-init", choices=("checkpoint", "random"),
        default="checkpoint",
        help="Registration Encoder initialization; Decoder always uses generator checkpoint",
    )
    p.add_argument("--data-dir", help="Single-source paired image directory")
    p.add_argument("--val-data-dir", help="Optional held-out paired image directory")
    p.add_argument(
        "--modality-dirs", nargs="+",
        help=("Ordered Stage 2 train directories for all-modality-pairs mode; "
              "order must match the Stage 1 modality registry"),
    )
    p.add_argument(
        "--val-modality-dirs", nargs="+",
        help="Ordered held-out directories for all-modality-pairs validation",
    )
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-modalities", type=int, default=None,
                   help="Defaults to the Stage 1 checkpoint value")
    p.add_argument("--src-modality", type=int, default=0)
    p.add_argument("--tgt-modality", type=int, default=1)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--embed-dim", type=int, default=None,
                   help="Defaults to the Stage 1 checkpoint value")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--reg-weight", type=float, default=0.5)
    p.add_argument(
        "--registration-model",
        choices=("conditioned_transmorph", "deprecated_cross_attention"),
        default="conditioned_transmorph",
        help="Registration architecture; Cross-Attention is reproduction-only",
    )
    p.add_argument(
        "--backbone-lr", type=float, default=None,
        help="Conditioned TransMorph backbone LR; defaults to 0.1 * --lr",
    )
    p.add_argument(
        "--fusion-residual", choices=("none", "fixed_query"), default="none",
        help=(
            "Cross-attention residual policy. 'none' prevents a direct "
            "fixed-only path; 'fixed_query' reproduces the legacy RegHead."
        ),
    )
    p.add_argument(
        "--pair-direction", choices=("random", "source-moving", "target-moving"),
        default="random",
        help=("Which aligned image is moving. random samples per item; "
              "source-moving uses n1 -> warped n2_fake; target-moving uses "
              "n2_fake -> warped n1."),
    )
    p.add_argument("--print-freq", type=int, default=10)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--max-iters-per-epoch", type=int, default=0,
                   help="Short-run cap; 0 uses the complete training loader")
    p.add_argument("--max-val-iters", type=int, default=0,
                   help="Validation cap; 0 uses the complete validation loader")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


# =============================================================================
# Model builder
# =============================================================================


def build_stage2_model(args: argparse.Namespace, device: torch.device):
    """Build the frozen pair generator and selected registration model."""
    # Keep programmatic callers compatible while making new Stage 2 builds use
    # the shortcut-free policy.  CLI parsing already supplies this field.
    args.fusion_residual = getattr(args, "fusion_residual", "none")
    args.registration_model = getattr(
        args, "registration_model", "conditioned_transmorph",
    )
    img_size = tuple(args.img_size)
    encoder_ckpt = torch.load(
        args.transcut_ckpt, map_location=device, weights_only=False,
    )
    generator_path = args.generator_ckpt or args.transcut_ckpt
    generator_ckpt = torch.load(
        generator_path, map_location=device, weights_only=False,
    )
    saved_cfg = encoder_ckpt.get("config", {})
    generator_cfg = generator_ckpt.get("config", {})
    saved_num_modalities = saved_cfg.get("num_modalities")
    if args.num_modalities is not None and saved_num_modalities is not None:
        if args.num_modalities != saved_num_modalities:
            raise ValueError(
                f"--num-modalities={args.num_modalities} does not match "
                f"Stage 1 checkpoint value {saved_num_modalities}"
            )
    saved_embed_dim = saved_cfg.get("embed_dim")
    if args.embed_dim is not None and saved_embed_dim is not None:
        if args.embed_dim != saved_embed_dim:
            raise ValueError(
                f"--embed-dim={args.embed_dim} does not match "
                f"Stage 1 checkpoint value {saved_embed_dim}"
            )
    num_modalities = args.num_modalities or saved_num_modalities
    embed_dim = args.embed_dim or saved_embed_dim
    if num_modalities is None or embed_dim is None:
        raise ValueError(
            "Stage 1 checkpoint lacks num_modalities/embed_dim; provide both CLI overrides"
        )
    if not (0 <= args.src_modality < num_modalities):
        raise ValueError(f"src modality {args.src_modality} is outside [0,{num_modalities})")
    if not (0 <= args.tgt_modality < num_modalities):
        raise ValueError(f"target modality {args.tgt_modality} is outside [0,{num_modalities})")
    if args.src_modality == args.tgt_modality:
        raise ValueError("Stage 2 requires distinct source and target modalities")
    # Persist resolved values for checkpoint metadata and resume diagnostics.
    args.num_modalities = num_modalities
    args.embed_dim = embed_dim

    for field in ("num_modalities", "input_nc", "output_nc"):
        if saved_cfg.get(field) != generator_cfg.get(field):
            raise ValueError(
                f"registration Encoder and pair generator disagree on {field}: "
                f"{saved_cfg.get(field)} != {generator_cfg.get(field)}"
            )

    cfg = TransCUTConfig(
        num_modalities=num_modalities,
        input_nc=saved_cfg.get("input_nc", 1),
        output_nc=saved_cfg.get("output_nc", 1),
        img_size=img_size[0],
        embed_dim=embed_dim,
        id_embed_dim=saved_cfg.get("id_embed_dim", 64),
        decoder_style_dim=saved_cfg.get("decoder_style_dim", 64),
        decoder_variant=saved_cfg.get("decoder_variant", "legacy"),
        patch_size=saved_cfg.get("patch_size", 4),
        depths=tuple(saved_cfg.get("depths", (2, 2, 4, 2))),
        num_heads=tuple(saved_cfg.get("num_heads", (4, 4, 8, 8))),
        window_size=tuple(saved_cfg.get("window_size", (8, 8))),
        mlp_ratio=saved_cfg.get("mlp_ratio", 4.0),
        drop_path_rate=saved_cfg.get("drop_path_rate", 0.3),
        ape=saved_cfg.get("ape", False),
        spe=saved_cfg.get("spe", False),
        rpe=saved_cfg.get("rpe", True),
        out_indices=tuple(saved_cfg.get("out_indices", (0, 1, 2, 3))),
    )
    if cfg.decoder_variant not in {
        "legacy", "highres_content", "fullres_residual",
    }:
        raise ValueError(
            f"unsupported Stage 1 decoder_variant: {cfg.decoder_variant}"
        )
    encoder = None
    mod_embed = None
    if args.registration_model == "deprecated_cross_attention":
        torch.manual_seed(args.seed + 500)
        mod_embed = ModalityIDEmbedding(
            cfg.num_modalities, cfg.id_embed_dim,
        ).to(device)
        encoder = SwinEncoderWithCLN(cfg, mod_embed).to(device)
        if args.encoder_init == "checkpoint":
            encoder.load_state_dict(encoder_ckpt["encoder"], strict=True)

    pair_cfg = TransCUTConfig(
        num_modalities=generator_cfg["num_modalities"],
        input_nc=generator_cfg.get("input_nc", 1),
        output_nc=generator_cfg.get("output_nc", 1),
        img_size=img_size[0],
        embed_dim=generator_cfg["embed_dim"],
        id_embed_dim=generator_cfg.get("id_embed_dim", 64),
        decoder_style_dim=generator_cfg.get("decoder_style_dim", 64),
        decoder_variant=generator_cfg.get("decoder_variant", "legacy"),
        patch_size=generator_cfg.get("patch_size", 4),
        depths=tuple(generator_cfg.get("depths", (2, 2, 4, 2))),
        num_heads=tuple(generator_cfg.get("num_heads", (4, 4, 8, 8))),
        window_size=tuple(generator_cfg.get("window_size", (8, 8))),
        mlp_ratio=generator_cfg.get("mlp_ratio", 4.0),
        drop_path_rate=generator_cfg.get("drop_path_rate", 0.3),
        ape=generator_cfg.get("ape", False),
        spe=generator_cfg.get("spe", False),
        rpe=generator_cfg.get("rpe", True),
        out_indices=tuple(generator_cfg.get("out_indices", (0, 1, 2, 3))),
    )
    if pair_cfg.decoder_variant not in {
        "legacy", "highres_content", "fullres_residual",
    }:
        raise ValueError(
            f"unsupported pair-generator decoder_variant: {pair_cfg.decoder_variant}"
        )
    pair_mod_embed = ModalityIDEmbedding(
        pair_cfg.num_modalities, pair_cfg.id_embed_dim,
    ).to(device)
    pair_encoder = SwinEncoderWithCLN(pair_cfg, pair_mod_embed).to(device)
    pair_encoder.load_state_dict(generator_ckpt["encoder"], strict=True)

    # Decoder (for generating synthetic pairs — frozen)
    decoder_kwargs = {
        "embed_dim": pair_cfg.embed_dim,
        "output_nc": pair_cfg.output_nc,
        "style_dim": pair_cfg.decoder_style_dim,
        "n_layers": len(pair_cfg.out_indices),
        "output_scale": pair_cfg.patch_size,
    }
    if pair_cfg.decoder_variant == "highres_content":
        decoder = HighResolutionContentDecoder(
            input_nc=pair_cfg.input_nc,
            **decoder_kwargs,
        )
    elif pair_cfg.decoder_variant == "fullres_residual":
        decoder = FullResolutionResidualDecoder(
            input_nc=pair_cfg.input_nc,
            **decoder_kwargs,
        )
    else:
        decoder = TransCUTDecoder(**decoder_kwargs)
    if "decoder" in generator_ckpt:
        decoder.load_state_dict(generator_ckpt["decoder"])
    decoder.to(device)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    # Style embedding (for decoder Adain)
    style_embed = nn.Linear(
        pair_cfg.id_embed_dim, pair_cfg.decoder_style_dim,
    )
    if "style_embed" in generator_ckpt:
        style_embed.load_state_dict(generator_ckpt["style_embed"])
    style_embed.to(device)

    # Registration model (trainable).  The main path reuses original
    # TransMorph joint encoding/decoder and only adds ordered pair CLN.
    torch.manual_seed(args.seed + 1000)
    if args.registration_model == "conditioned_transmorph":
        trans_cfg = config_from_transcut(
            saved_cfg, img_size, input_nc=cfg.input_nc,
        )
        registration_model = ModalityConditionedTransMorph(
            trans_cfg,
            num_modalities=num_modalities,
            id_embed_dim=cfg.id_embed_dim,
            image_channels=cfg.input_nc,
        ).to(device)
        if args.encoder_init == "checkpoint":
            report = registration_model.initialize_from_transcut(encoder_ckpt)
            print(
                "Initialized conditioned TransMorph from Stage 1: "
                f"{report['copied_tensors']} tensors"
            )
    else:
        registration_model = CrossAttentionRegHead(
            embed_dim=embed_dim, out_indices=(0, 1, 2, 3),
            reg_head_chan=16, num_heads=4,
            fusion_residual=args.fusion_residual,
        ).to(device)

    # Spatial transformer for warping
    spatial_trans = SpatialTransformer(img_size).to(device)

    # Freeze encoder
    frozen_modules = [pair_encoder, pair_mod_embed]
    if encoder is not None:
        frozen_modules.extend([encoder, mod_embed])
    for module in frozen_modules:
        for p in module.parameters():
            p.requires_grad = False
    for p in style_embed.parameters():
        p.requires_grad = False

    if encoder is not None:
        encoder.eval()
        mod_embed.eval()
    pair_encoder.eval()
    pair_mod_embed.eval()
    decoder.eval()
    style_embed.eval()

    return (
        pair_encoder, pair_mod_embed, decoder, style_embed,
        encoder, registration_model, spatial_trans,
    )


# =============================================================================
# Synthetic data generation (on-the-fly)
# =============================================================================


def _generate_synthetic_pair(
    real_src: torch.Tensor,
    encoder: nn.Module,
    mod_embed: nn.Module,
    decoder: nn.Module,
    style_embed: nn.Module,
    src_id: int | torch.Tensor,
    tgt_id: int | torch.Tensor,
    device: torch.device,
    integrate: VecInt,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate an aligned target-style image and a displacement field.

    Returns
    -------
    fake : (B, C, H, W)          Translated image (pixel-aligned to real_src).
    flow_gt : (B, 2, H, W)       Ground-truth deformation field.
    """
    B, _, H, W = real_src.shape
    if isinstance(tgt_id, torch.Tensor):
        tgt_tensor = tgt_id.to(device=device, dtype=torch.long)
    else:
        tgt_tensor = torch.full((B,), tgt_id, dtype=torch.long, device=device)
    if isinstance(src_id, torch.Tensor):
        src_tensor = src_id.to(device=device, dtype=torch.long)
    else:
        src_tensor = torch.full((B,), src_id, dtype=torch.long, device=device)
    if src_tensor.shape != (B,) or tgt_tensor.shape != (B,):
        raise ValueError(
            f"modality IDs must have shape ({B},), got "
            f"{tuple(src_tensor.shape)} and {tuple(tgt_tensor.shape)}"
        )

    with torch.no_grad():
        # 1. Translate: real_src → fake (pixel-aligned, target modality style)
        feats = encoder(real_src, src_tensor)
        style = style_embed(mod_embed(tgt_tensor))
        fake = decoder(feats, style, source=real_src)
        # Upsample to match input resolution for deformation
        if fake.shape[-2:] != real_src.shape[-2:]:
            fake = F.interpolate(fake, size=real_src.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # 2. Generate one smooth stationary velocity per sample.
        velocities = []
        for _ in range(B):
            velocity, _, _ = generate_diffeomorphic_flow(
                (H, W), smooth_sigma=12.0, max_displacement=15.0,
                affine_probability=0.0,
            )
            velocities.append(torch.from_numpy(velocity))
        velocity = torch.stack(velocities).to(device=device, dtype=fake.dtype)

        # Integrate a stationary velocity to obtain a smooth displacement.
        # The displacement is applied later to the image chosen as fixed, so
        # no inverse field is needed.
        flow_gt = integrate(velocity)

    return fake, flow_gt


def _construct_cross_modal_pair(
    real_src: torch.Tensor,
    fake_tgt: torch.Tensor,
    flow_gt: torch.Tensor,
    src_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    spatial_trans: SpatialTransformer,
    direction: str = "random",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a pair that exactly follows ``warp(moving, flow_gt) ≈ fixed``.

    ``real_src`` and ``fake_tgt`` are assumed to be geometrically aligned.
    We deform the image selected as fixed and apply the same displacement to
    the other modality during supervision.  This avoids an inverse-flow
    construction while retaining genuinely cross-modal encoder inputs.
    """
    if direction not in {"random", "source-moving", "target-moving"}:
        raise ValueError(f"Unsupported pair direction: {direction}")
    if real_src.shape != fake_tgt.shape:
        raise ValueError(
            f"Aligned images must have equal shapes, got {real_src.shape} and {fake_tgt.shape}"
        )

    warped_src = spatial_trans(real_src, flow_gt)
    warped_tgt = spatial_trans(fake_tgt, flow_gt)
    batch = real_src.shape[0]
    if direction == "source-moving":
        source_moving = torch.ones(batch, dtype=torch.bool, device=real_src.device)
    elif direction == "target-moving":
        source_moving = torch.zeros(batch, dtype=torch.bool, device=real_src.device)
    else:
        source_moving = torch.rand(batch, device=real_src.device) < 0.5

    image_mask = source_moving.view(batch, 1, 1, 1)
    moving = torch.where(image_mask, real_src, fake_tgt)
    fixed = torch.where(image_mask, warped_tgt, warped_src)
    moving_ids = torch.where(source_moving, src_ids, tgt_ids)
    fixed_ids = torch.where(source_moving, tgt_ids, src_ids)
    return moving, fixed, moving_ids, fixed_ids


def _normalize_stage1_input(image: torch.Tensor) -> torch.Tensor:
    """Convert Dataset images from ``[0,1]`` to Stage 1's ``[-1,1]``."""
    return image.mul(2.0).sub(1.0)


def _stage2_batch(
    batch: dict, args: argparse.Namespace, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized source images and their true source/target IDs."""
    if "src_id" in batch:
        source = batch["A"].to(device)
        src_ids = batch["src_id"].to(device=device, dtype=torch.long)
        tgt_ids = batch["tgt_id"].to(device=device, dtype=torch.long)
        return source, src_ids, tgt_ids

    source = _normalize_stage1_input(batch["moving"].to(device))
    src_ids = torch.full(
        (source.size(0),), args.src_modality,
        dtype=torch.long, device=device,
    )
    tgt_ids = torch.full(
        (source.size(0),), args.tgt_modality,
        dtype=torch.long, device=device,
    )
    return source, src_ids, tgt_ids


def _modality_label(args: argparse.Namespace, modality_id: int) -> str:
    names = getattr(args, "modality_names", None)
    if names is not None and modality_id < len(names):
        return str(names[modality_id])
    return str(modality_id)


def _resize_displacement(
    flow: torch.Tensor, target_size: tuple[int, int],
) -> torch.Tensor:
    """Resize ``(dy,dx)`` while preserving displacement pixel units."""
    if flow.shape[-2:] == target_size:
        return flow
    scale_h = target_size[0] / flow.shape[-2]
    scale_w = target_size[1] / flow.shape[-1]
    flow = F.interpolate(
        flow, size=target_size, mode="bilinear", align_corners=False,
    )
    flow[:, 0] *= scale_h
    flow[:, 1] *= scale_w
    return flow


def _predict_registration_flow(
    registration_model: nn.Module, encoder: nn.Module | None,
    moving: torch.Tensor, fixed: torch.Tensor,
    moving_ids: torch.Tensor, fixed_ids: torch.Tensor,
    registration_type: str,
) -> torch.Tensor:
    """Dispatch the main TransMorph path or deprecated feature head."""
    if registration_type == "conditioned_transmorph":
        return registration_model.predict_flow(
            moving, fixed, moving_ids, fixed_ids,
        )
    if encoder is None:
        raise RuntimeError("deprecated Cross-Attention requires its frozen encoder")
    with torch.no_grad():
        fixed_features = encoder(fixed, fixed_ids)
        moving_features = encoder(moving, moving_ids)
    return registration_model(fixed_features, moving_features)


@torch.inference_mode()
def _validate_stage2(
    loader: DataLoader, pair_encoder: nn.Module, pair_mod_embed: nn.Module,
    decoder: nn.Module, style_embed: nn.Module, encoder: nn.Module,
    registration_model: nn.Module, spatial_trans: SpatialTransformer,
    integrate: VecInt, args: argparse.Namespace, device: torch.device,
) -> dict[str, object]:
    """Evaluate against a repeatable held-out set of synthetic flows."""
    registration_model.eval()
    random.seed(args.seed + 100_000)
    np.random.seed(args.seed + 100_000)
    torch.manual_seed(args.seed + 100_000)
    epe_sum = zero_epe_sum = 0.0
    pck1_sum = pck2_sum = pck4_sum = 0.0
    sample_count = 0
    direction_totals: dict[str, dict[str, float]] = {}
    for index, batch in enumerate(loader):
        real_src, src_ids, tgt_ids = _stage2_batch(batch, args, device)
        fake, flow_gt = _generate_synthetic_pair(
            real_src, pair_encoder, pair_mod_embed, decoder, style_embed,
            src_ids, tgt_ids, device, integrate,
        )
        moving, fixed, moving_ids, fixed_ids = _construct_cross_modal_pair(
            real_src, fake, flow_gt, src_ids, tgt_ids,
            spatial_trans, args.pair_direction,
        )
        flow_pred = _predict_registration_flow(
            registration_model, encoder, moving, fixed,
            moving_ids, fixed_ids, args.registration_model,
        )
        flow_pred = _resize_displacement(flow_pred, flow_gt.shape[-2:])
        errors = torch.linalg.vector_norm(flow_pred - flow_gt, dim=1)
        batch_size = real_src.size(0)
        epe_sum += float(errors.mean()) * batch_size
        zero_epe_sum += compute_epe(torch.zeros_like(flow_gt), flow_gt) * batch_size
        pck1_sum += float((errors <= 1).float().mean()) * batch_size
        pck2_sum += float((errors <= 2).float().mean()) * batch_size
        pck4_sum += float((errors <= 4).float().mean()) * batch_size
        sample_count += batch_size
        for sample_index in range(batch_size):
            key = (
                f"{_modality_label(args, int(moving_ids[sample_index]))}->"
                f"{_modality_label(args, int(fixed_ids[sample_index]))}"
            )
            values = direction_totals.setdefault(
                key, {"epe": 0.0, "zero_epe": 0.0, "pck_1": 0.0,
                      "pck_2": 0.0, "pck_4": 0.0, "samples": 0.0},
            )
            sample_error = errors[sample_index]
            sample_zero = torch.linalg.vector_norm(
                flow_gt[sample_index], dim=0,
            )
            values["epe"] += float(sample_error.mean())
            values["zero_epe"] += float(sample_zero.mean())
            values["pck_1"] += float((sample_error <= 1).float().mean())
            values["pck_2"] += float((sample_error <= 2).float().mean())
            values["pck_4"] += float((sample_error <= 4).float().mean())
            values["samples"] += 1
        if args.max_val_iters and index + 1 >= args.max_val_iters:
            break
    result = {
        "epe": epe_sum / sample_count,
        "zero_epe": zero_epe_sum / sample_count,
        "pck_1": pck1_sum / sample_count,
        "pck_2": pck2_sum / sample_count,
        "pck_4": pck4_sum / sample_count,
        "samples": sample_count,
    }
    result["directions"] = {
        key: {
            metric: (value if metric == "samples" else value / values["samples"])
            for metric, value in values.items()
        }
        for key, values in sorted(direction_totals.items())
    }
    return result


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.val_interval < 1:
        raise ValueError("--val-interval must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(os.path.join(args.save_dir, "experiments"), exist_ok=True)

    if bool(args.data_dir) == bool(args.modality_dirs):
        raise ValueError("provide exactly one of --data-dir or --modality-dirs")
    if args.val_data_dir and args.val_modality_dirs:
        raise ValueError(
            "provide at most one of --val-data-dir or --val-modality-dirs"
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    stage1_meta = torch.load(args.transcut_ckpt, map_location="cpu",
                             weights_only=False)
    stage1_input_nc = stage1_meta.get("config", {}).get("input_nc", 1)
    modality_names = stage1_meta.get("modality_names")
    if modality_names is None:
        modality_names = [
            str(index)
            for index in range(stage1_meta["config"]["num_modalities"])
        ]
    if len(modality_names) != stage1_meta["config"]["num_modalities"]:
        raise ValueError("Stage 1 modality registry length is inconsistent")
    args.modality_names = list(modality_names)
    if stage1_input_nc not in (1, 3):
        raise ValueError(
            "PairedImageFolderDataset supports Stage 1 input_nc 1 or 3, "
            f"got {stage1_input_nc}"
        )
    if args.modality_dirs:
        if len(args.modality_dirs) != stage1_meta["config"]["num_modalities"]:
            raise ValueError(
                "--modality-dirs count must match the Stage 1 modality count"
            )
        ds = MultiDomainTranslationDataset(
            args.modality_dirs, input_nc=stage1_input_nc,
            load_size=args.img_size[0], crop_size=args.img_size[0],
            pairing_mode="unpaired",
        )
    else:
        ds = PairedImageFolderDataset(
            args.data_dir, img_size=tuple(args.img_size),
            grayscale=(stage1_input_nc == 1),
        )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    print(f"Training pairs: {len(ds)}")
    val_loader = None
    if args.val_modality_dirs:
        if len(args.val_modality_dirs) != stage1_meta["config"]["num_modalities"]:
            raise ValueError(
                "--val-modality-dirs count must match the Stage 1 modality count"
            )
        val_ds = MultiDomainTranslationDataset(
            args.val_modality_dirs, input_nc=stage1_input_nc,
            load_size=args.img_size[0], crop_size=args.img_size[0],
            pairing_mode="unpaired",
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Validation pairs: {len(val_ds)}")
    elif args.val_data_dir:
        val_ds = PairedImageFolderDataset(
            args.val_data_dir, img_size=tuple(args.img_size),
            grayscale=(stage1_input_nc == 1),
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Validation pairs: {len(val_ds)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    (
        pair_encoder, pair_mod_embed, decoder, style_embed,
        encoder, registration_model, spatial_trans,
    ) = build_stage2_model(args, device)
    integrate = VecInt(tuple(args.img_size), nsteps=7).to(device)

    if args.registration_model == "conditioned_transmorph":
        backbone_lr = args.backbone_lr or args.lr * 0.1
        backbone_ids = {
            id(parameter)
            for module in (
                registration_model.transformer,
                registration_model.modality_embedding,
                registration_model.pair_cln,
            )
            for parameter in module.parameters()
        }
        backbone = [
            parameter for parameter in registration_model.parameters()
            if id(parameter) in backbone_ids
        ]
        new_parameters = [
            parameter for parameter in registration_model.parameters()
            if id(parameter) not in backbone_ids
        ]
        optimizer = Adam([
            {"params": backbone, "lr": backbone_lr},
            {"params": new_parameters, "lr": args.lr},
        ], amsgrad=True)
    else:
        optimizer = Adam(registration_model.parameters(), lr=args.lr, amsgrad=True)
    criterion_reg = Grad("l2", loss_mult=2).to(device)
    with (Path(args.save_dir) / "run_config.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
        handle.write("\n")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_loss = 1e10
    ckpt_path = os.path.join(args.save_dir, "experiments", "latest_checkpoint.pth")
    if args.resume and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_type = ck.get("model_config", {}).get(
            "registration_model", "deprecated_cross_attention",
        )
        if saved_type != args.registration_model:
            raise ValueError(
                "resume checkpoint registration model mismatch: "
                f"checkpoint={saved_type!r}, CLI={args.registration_model!r}"
            )
        if args.registration_model == "conditioned_transmorph":
            registration_model.load_state_dict(ck["registration_model"])
        else:
            saved_residual = ck.get("model_config", {}).get(
                "fusion_residual", "fixed_query",
            )
            if saved_residual != args.fusion_residual:
                raise ValueError(
                    "resume checkpoint fusion_residual mismatch: "
                    f"checkpoint={saved_residual!r}, CLI={args.fusion_residual!r}"
                )
            registration_model.load_state_dict(ck["reg_head"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"]
        best_loss = ck.get("best_loss", 1e10)
        print(f"Resumed from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        registration_model.train()
        loss_meter = AverageMeter()
        pair_counts = torch.zeros(
            args.num_modalities, args.num_modalities, dtype=torch.long,
        )

        # Make on-the-fly train flows and pair directions identical across
        # Encoder-control runs with the same seed.
        random.seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)

        for idx, batch in enumerate(loader):
            real_src, src_ids, tgt_ids = _stage2_batch(batch, args, device)

            # Generate synthetic supervision
            fake, flow_gt = _generate_synthetic_pair(
                real_src, pair_encoder, pair_mod_embed, decoder, style_embed,
                src_ids, tgt_ids, device,
                integrate,
            )
            moving, fixed, moving_ids, fixed_ids = _construct_cross_modal_pair(
                real_src, fake, flow_gt, src_ids, tgt_ids,
                spatial_trans, args.pair_direction,
            )
            flat_pairs = (
                moving_ids.detach().cpu() * args.num_modalities
                + fixed_ids.detach().cpu()
            )
            pair_counts += torch.bincount(
                flat_pairs, minlength=args.num_modalities ** 2,
            ).reshape(args.num_modalities, args.num_modalities)

            flow_pred = _predict_registration_flow(
                registration_model, encoder, moving, fixed,
                moving_ids, fixed_ids, args.registration_model,
            )

            # Resize flow_pred to match flow_gt
            flow_pred = _resize_displacement(
                flow_pred, flow_gt.shape[-2:],
            )

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
            if args.max_iters_per_epoch and idx + 1 >= args.max_iters_per_epoch:
                print(
                    f"  Reached short-run cap: {args.max_iters_per_epoch} iterations"
                )
                break

        print(f"Epoch {epoch+1}: avg_loss={loss_meter.avg:.4f}")
        print("Registration direction counts: " + " ".join(
            f"{_modality_label(args, source)}->{_modality_label(args, target)}="
            f"{int(pair_counts[source, target])}"
            for source in range(args.num_modalities)
            for target in range(args.num_modalities)
            if source != target
        ))

        validation = None
        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            validation = _validate_stage2(
                val_loader, pair_encoder, pair_mod_embed, decoder,
                style_embed, encoder, registration_model, spatial_trans, integrate,
                args, device,
            )
            print("Validation: " + json.dumps(validation, sort_keys=True))
            with (Path(args.save_dir) / "metrics.jsonl").open(
                "a", encoding="utf-8",
            ) as handle:
                handle.write(json.dumps({
                    "epoch": epoch + 1,
                    "train_loss": loss_meter.avg,
                    "validation": validation,
                }, sort_keys=True) + "\n")

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1, "best_loss": best_loss,
            "format_version": 1,
            "model_type": (
                "ModalityConditionedTransMorph"
                if args.registration_model == "conditioned_transmorph"
                else "DeprecatedCrossAttentionRegHead"
            ),
            "model_config": {
                "img_size": list(args.img_size), "embed_dim": args.embed_dim,
                "flow_parameterization": "displacement",
                "registration_model": args.registration_model,
                "pair_direction": args.pair_direction,
                "src_modality": args.src_modality,
                "tgt_modality": args.tgt_modality,
                "encoder_init": args.encoder_init,
                "encoder_checkpoint": args.transcut_ckpt,
                "generator_checkpoint": args.generator_ckpt or args.transcut_ckpt,
                "all_modality_pairs": bool(args.modality_dirs),
                "modality_names": list(args.modality_names),
            },
            "optimizer": optimizer.state_dict(),
        }
        if args.registration_model == "conditioned_transmorph":
            ckpt["registration_model"] = registration_model.state_dict()
        else:
            ckpt["model_config"]["fusion_residual"] = args.fusion_residual
            ckpt["reg_head"] = registration_model.state_dict()

        selection_loss = validation["epe"] if validation else loss_meter.avg
        if selection_loss < best_loss:
            best_loss = selection_loss
            ckpt["best_loss"] = best_loss
            torch.save(ckpt, os.path.join(args.save_dir, "experiments", "model_best.pth"))
            print(f"  → Best model (selection={best_loss:.4f})")

        torch.save(ckpt, ckpt_path)

        if (epoch + 1) % 20 == 0:
            torch.save(ckpt, os.path.join(
                args.save_dir, "experiments", f"checkpoint_epoch_{epoch+1}.pth"))

    print("Stage 2 training finished.")


if __name__ == "__main__":
    main()
