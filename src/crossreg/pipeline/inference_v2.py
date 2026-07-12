"""
Stage 3: End-to-end inference module (encoder-only, no decoder).

Takes two real images from different modalities, encodes both through the
shared Swin-Transformer, and estimates the deformation field via the
Cross-Attention Registration Head.

No generative decoder is used — this is the deployment-ready pipeline.

Usage::

    from crossreg.pipeline.inference_v2 import Stage3Inference

    pipeline = Stage3Inference(encoder, reg_head, img_size=(256, 256))
    warped, flow = pipeline(source_img, target_img)
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
    diffeomorphic : bool
        If True, apply VecInt integration for diffeomorphic flow.
    """

    def __init__(self, encoder: nn.Module, reg_head: nn.Module,
                 img_size: tuple[int, int] = (256, 256),
                 diffeomorphic: bool = True):
        super().__init__()
        self.encoder = encoder
        self.reg_head = reg_head
        self.diffeomorphic = diffeomorphic

        self.spatial_trans = SpatialTransformer(img_size)
        self.integrate = VecInt(img_size, nsteps=7) if diffeomorphic else None

    def forward(self, source: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Register *source* to *target*.

        Parameters
        ----------
        source : (B, 1, H, W)  Moving image (modality A).
        target : (B, 1, H, W)  Fixed image (modality B).

        Returns
        -------
        warped : (B, 1, H, W)  Registered source.
        flow : (B, 2, H, W)    Deformation field.
        """
        # 1. Encode both images through shared Swin
        f_src = self.encoder(source)
        f_tgt = self.encoder(target)

        # 2. RegHead estimates flow
        pos_flow = self.reg_head(f_tgt, f_src)  # fixed features as Q, moving as K/V

        # 3. Resize flow to match input resolution
        if pos_flow.shape[-2:] != source.shape[-2:]:
            scale_h = source.shape[-2] / pos_flow.shape[-2]
            scale_w = source.shape[-1] / pos_flow.shape[-1]
            pos_flow = F.interpolate(pos_flow, size=source.shape[-2:],
                                     mode='bilinear', align_corners=False)
            pos_flow[:, 0] *= scale_h
            pos_flow[:, 1] *= scale_w

        # 4. Optional diffeomorphic integration
        flow = self.integrate(pos_flow) if self.diffeomorphic else pos_flow

        # 5. Warp
        warped = self.spatial_trans(source, flow)
        return warped, flow

    @torch.no_grad()
    def infer(self, source: torch.Tensor, target: torch.Tensor):
        """Inference without gradient tracking."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(source, target)
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
    from crossreg.models.swin_transformer import SwinTransformer
    from crossreg.registration.transmorph.cross_attn_head import CrossAttentionRegHead

    dev = torch.device(device)

    # Encoder
    encoder = SwinTransformer(
        pretrain_img_size=img_size[0], in_chans=1, embed_dim=embed_dim,
        depths=(2, 2, 4, 2), num_heads=(4, 4, 8, 8),
        window_size=(8, 8),
    )
    ckpt = torch.load(encoder_ckpt, map_location=dev, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"]["swin"])
    encoder.to(dev)
    encoder.eval()

    # RegHead
    reg_head = CrossAttentionRegHead(
        embed_dim=embed_dim, out_indices=(0, 1, 2, 3),
        reg_head_chan=16, num_heads=4,
    )
    ckpt2 = torch.load(reg_head_ckpt, map_location=dev, weights_only=False)
    reg_head.load_state_dict(ckpt2.get("reg_head", ckpt2))
    reg_head.to(dev)
    reg_head.eval()

    return Stage3Inference(encoder, reg_head, img_size).to(dev)
