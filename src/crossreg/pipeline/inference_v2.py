"""
Stage 3: End-to-end conditioned TransMorph inference.

The main path keeps original TransMorph joint spatial encoding and injects
the ordered moving/fixed modality IDs after patch embedding.  The historical
separate-Encoder/Cross-Attention path remains loadable for old checkpoints.

No generative decoder is used — this is the deployment-ready pipeline.

Usage::

    from crossreg.pipeline.inference_v2 import Stage3Inference

    pipeline = Stage3Inference(encoder, reg_head, img_size=(256, 256))
    warped, flow = pipeline(moving, fixed, moving_ids, fixed_ids)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crossreg.registration.transmorph.model import SpatialTransformer, VecInt


class Stage3Inference(nn.Module):
    """Encoder-only registration inference (Stage 3).

    Parameters
    ----------
    encoder : nn.Module
        Shared Swin-Transformer encoder.
    reg_head : nn.Module
        Cross-Attention Registration Head.
    img_size : tuple[int, int]
        Input image (H, W).  Used for spatial transformer and VecInt.
    flow_parameterization : str
        ``displacement`` when RegHead directly predicts the backward-sampling
        displacement used by ``warp``. ``velocity`` is reserved for a head
        explicitly trained against stationary velocities.
    """

    def __init__(self, encoder: nn.Module, reg_head: nn.Module,
                 img_size: tuple[int, int] = (256, 256),
                 flow_parameterization: str = "displacement"):
        super().__init__()
        self.encoder = encoder
        self.reg_head = reg_head
        if flow_parameterization not in {"displacement", "velocity"}:
            raise ValueError(f"Unsupported flow parameterization: {flow_parameterization}")
        self.flow_parameterization = flow_parameterization

        self.spatial_trans = SpatialTransformer(img_size)
        self.integrate = VecInt(img_size, nsteps=7) if flow_parameterization == "velocity" else None

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor,
                moving_ids: torch.Tensor, fixed_ids: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Register ``moving`` to ``fixed``.

        Parameters
        ----------
        moving, fixed : (B, C, H, W)
        moving_ids, fixed_ids : (B,) stable modality IDs.

        Returns
        -------
        warped : (B, 1, H, W)  Registered source.
        flow : (B, 2, H, W)    Deformation field.
        """
        # 1. Encode both images through shared Swin
        if moving.shape != fixed.shape:
            raise ValueError(f"moving and fixed shapes differ: {moving.shape} vs {fixed.shape}")
        f_moving = self.encoder(moving, moving_ids)
        f_fixed = self.encoder(fixed, fixed_ids)

        # 2. RegHead estimates flow
        raw_flow = self.reg_head(f_fixed, f_moving)

        # 3. Resize flow to match input resolution
        if raw_flow.shape[-2:] != moving.shape[-2:]:
            scale_h = moving.shape[-2] / raw_flow.shape[-2]
            scale_w = moving.shape[-1] / raw_flow.shape[-1]
            raw_flow = F.interpolate(raw_flow, size=moving.shape[-2:],
                                     mode='bilinear', align_corners=False)
            raw_flow[:, 0] *= scale_h
            raw_flow[:, 1] *= scale_w

        # 4. Optional diffeomorphic integration
        flow = self.integrate(raw_flow) if self.integrate is not None else raw_flow

        # 5. Warp
        warped = self.spatial_trans(moving, flow)
        return warped, flow

    @torch.no_grad()
    def infer(self, moving: torch.Tensor, fixed: torch.Tensor,
              moving_ids: torch.Tensor, fixed_ids: torch.Tensor):
        """Inference without gradient tracking."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(moving, fixed, moving_ids, fixed_ids)
        finally:
            if was_training:
                self.train()


class ConditionedStage3Inference(nn.Module):
    """Thin deployment wrapper around ``ModalityConditionedTransMorph``."""

    def __init__(self, registration_model: nn.Module):
        super().__init__()
        self.registration_model = registration_model
        self.flow_parameterization = "displacement"

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor,
                moving_ids: torch.Tensor, fixed_ids: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        warped, flow, _ = self.registration_model(
            moving, fixed, moving_ids, fixed_ids,
        )
        return warped, flow

    @torch.no_grad()
    def infer(self, moving: torch.Tensor, fixed: torch.Tensor,
              moving_ids: torch.Tensor, fixed_ids: torch.Tensor):
        was_training = self.training
        self.eval()
        try:
            return self.forward(moving, fixed, moving_ids, fixed_ids)
        finally:
            if was_training:
                self.train()


def build_stage3_from_checkpoints(
    encoder_ckpt: str,
    reg_head_ckpt: str,
    img_size: tuple[int, int] = (256, 256),
    embed_dim: int = 96,
    device: str = "cpu",
) -> Stage3Inference:
    """Factory: build Stage 3 pipeline from saved checkpoints.

    Parameters
    ----------
    encoder_ckpt : str   Path to TransCUT encoder weights (Stage 1 output).
    reg_head_ckpt : str  Path to RegHead weights (Stage 2 output).
    img_size : tuple
    embed_dim : int
    device : str
    """
    from crossreg.registration.transmorph.conditioned_model import (
        ModalityConditionedTransMorph,
        config_from_transcut,
    )
    from crossreg.registration.transmorph.deprecated_cross_attn_head import (
        CrossAttentionRegHead,
    )
    from crossreg.translation.transcut.cln_adain import ModalityIDEmbedding
    from crossreg.translation.transcut.transcut_model import SwinEncoderWithCLN, TransCUTConfig

    dev = torch.device(device)

    # Inspect Stage 2 first: new checkpoints contain the complete conditioned
    # TransMorph; missing metadata denotes the historical Cross-Attention path.
    ckpt2 = torch.load(reg_head_ckpt, map_location=dev, weights_only=False)
    model_cfg = ckpt2.get("model_config", {}) if isinstance(ckpt2, dict) else {}
    registration_type = model_cfg.get(
        "registration_model", "deprecated_cross_attention",
    )

    ckpt = torch.load(encoder_ckpt, map_location=dev, weights_only=False)
    saved_cfg = ckpt.get("config", {})
    if registration_type == "conditioned_transmorph":
        input_nc = saved_cfg.get("input_nc", 1)
        trans_cfg = config_from_transcut(
            saved_cfg, img_size, input_nc=input_nc,
        )
        registration_model = ModalityConditionedTransMorph(
            trans_cfg,
            num_modalities=saved_cfg.get("num_modalities", 2),
            id_embed_dim=saved_cfg.get("id_embed_dim", 64),
            image_channels=input_nc,
        )
        registration_model.load_state_dict(
            ckpt2["registration_model"], strict=True,
        )
        registration_model.to(dev)
        registration_model.eval()
        return ConditionedStage3Inference(registration_model).to(dev)

    # Historical path: rebuild the separate frozen Stage 1 Encoder.
    cfg = TransCUTConfig(
        num_modalities=saved_cfg.get("num_modalities", 2),
        input_nc=saved_cfg.get("input_nc", 1),
        output_nc=saved_cfg.get("output_nc", 1),
        img_size=img_size[0],
        embed_dim=saved_cfg.get("embed_dim", embed_dim),
        id_embed_dim=saved_cfg.get("id_embed_dim", 64),
        decoder_style_dim=saved_cfg.get("decoder_style_dim", 64),
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
    mod_embed = ModalityIDEmbedding(cfg.num_modalities, cfg.id_embed_dim)
    encoder = SwinEncoderWithCLN(cfg, mod_embed)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    encoder.to(dev)
    encoder.eval()

    # RegHead.  Checkpoints created before the residual policy was recorded
    # used the fixed-query residual, so missing metadata must retain that
    # legacy behaviour instead of silently changing inference semantics.
    fusion_residual = model_cfg.get("fusion_residual", "fixed_query")
    reg_head = CrossAttentionRegHead(
        embed_dim=cfg.embed_dim, out_indices=cfg.out_indices,
        reg_head_chan=16, num_heads=4,
        fusion_residual=fusion_residual,
    )
    reg_head.load_state_dict(ckpt2.get("reg_head", ckpt2))
    reg_head.to(dev)
    reg_head.eval()

    parameterization = model_cfg.get("flow_parameterization", "displacement")
    return Stage3Inference(
        encoder, reg_head, img_size,
        flow_parameterization=parameterization,
    ).to(dev)
