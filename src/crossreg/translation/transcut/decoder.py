"""
Lightweight CNN Decoder for TransCUT.

Takes Swin-Transformer feature maps and a target modality ID, producing
a translated image in the target modality style via AdaIN injection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cln_adain import AdaIN2d


class AdaINDecoderBlock(nn.Module):
    """Decoder block with AdaIN style injection.

    ``Conv → AdaIN → ReLU → Conv → AdaIN → ReLU`` with optional skip.
    """

    def __init__(self, in_ch: int, out_ch: int, style_dim: int,
                 skip_ch: int = 0, upsample: bool = True):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="nearest",
        ) if upsample else None
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.adain1 = AdaIN2d(out_ch, style_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.adain2 = AdaIN2d(out_ch, style_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        if self.upsample:
            x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.act(self.adain1(self.conv1(x), style))
        x = self.act(self.adain2(self.conv2(x), style))
        return x


class TransCUTDecoder(nn.Module):
    """Lightweight CNN decoder that reconstructs images from Swin features.

    Multi-scale features from the encoder are fused with skip connections
    (largest → smallest: f0..f3 from Swin out_indices).  Each block receives
    a target-modality style vector via AdaIN.

    Parameters
    ----------
    embed_dim : int    Swin embed dim (e.g. 96).
    output_nc : int    Output channels (1 for grayscale, 3 for RGB).
    style_dim : int    Dimension of the target-modality ID embedding.
    n_layers : int     Number of decoder blocks (default 4, matching Swin stages).
    """

    def __init__(self, embed_dim: int = 96, output_nc: int = 1,
                 style_dim: int = 64, n_layers: int = 4,
                 output_scale: int = 4):
        super().__init__()
        self.output_scale = output_scale
        dims = [embed_dim * (2 ** i) for i in range(n_layers)]  # [96, 192, 384, 768]
        dims = list(reversed(dims))  # [768, 384, 192, 96]

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            in_c = dims[i]
            out_c = dims[i + 1] if i + 1 < n_layers else embed_dim
            # Skip from features[-(i+2)] = dims_rev[i+1]
            skip_c = dims[i + 1] if i < n_layers - 1 else 0
            self.blocks.append(
                AdaINDecoderBlock(in_c, out_c, style_dim, skip_ch=skip_c,
                                  upsample=(i < n_layers - 1))
            )

        if output_scale < 1 or output_scale & (output_scale - 1):
            raise ValueError("output_scale must be a positive power of two")
        self.refine_blocks = nn.ModuleList()
        refine_channels = out_c
        for _ in range(int(math.log2(output_scale))):
            next_channels = max(embed_dim // 2, refine_channels // 2, 16)
            self.refine_blocks.append(
                AdaINDecoderBlock(
                    refine_channels, next_channels, style_dim,
                    skip_ch=0, upsample=True,
                )
            )
            refine_channels = next_channels

        self.head = nn.Sequential(
            nn.Conv2d(refine_channels, output_nc,
                      kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, features: list[torch.Tensor],
                style: torch.Tensor,
                source: torch.Tensor | None = None) -> torch.Tensor:
        """*features*: list of feature maps from Swin (largest→smallest).
           *style*: (B, style_dim) target modality vector.

        ``source`` is accepted for API compatibility with the optional
        high-resolution decoder and intentionally ignored by this legacy
        implementation.
        """
        # features are ordered largest first (from Swin out_indices)
        x = features[-1]  # highest-level (smallest spatial) feature
        for i, blk in enumerate(self.blocks):
            skip_idx = len(features) - 2 - i
            skip = features[skip_idx] if skip_idx >= 0 else None
            x = blk(x, style, skip)
        for blk in self.refine_blocks:
            x = blk(x, style)
        output = self.head(x)
        # Stage-0 Swin features are at patch_size=4 resolution.  Translation
        # outputs are part of the public model contract and must match the
        # source image resolution, not rely on callers to upsample ad hoc.
        target_size = (
            features[0].shape[-2] * self.output_scale,
            features[0].shape[-1] * self.output_scale,
        )
        if output.shape[-2:] != target_size:
            output = F.interpolate(output, size=target_size,
                                   mode="nearest")
        return output
