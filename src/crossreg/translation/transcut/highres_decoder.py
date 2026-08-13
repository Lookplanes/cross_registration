"""Optional TransCUT decoder with shared high-resolution content skips.

The legacy decoder only receives Swin features whose finest resolution is
``input / patch_size``.  This variant keeps the complete legacy coarse path,
and adds a small, domain-shared convolutional stem that supplies normalized
source-content features to the final upsampling blocks.

It is deliberately a separate class so old checkpoints and experiments retain
the exact legacy architecture.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import AdaINDecoderBlock, TransCUTDecoder


class HighResolutionContentDecoder(TransCUTDecoder):
    """TransCUT decoder with normalized high-resolution source-content skips.

    The detail stem is shared across every modality and contains no
    modality-specific head. Instance normalization removes per-image channel
    statistics before the detail features are fused; target appearance remains
    controlled by the existing AdaIN blocks.
    """

    def __init__(
        self,
        embed_dim: int = 96,
        input_nc: int = 1,
        output_nc: int = 1,
        style_dim: int = 64,
        n_layers: int = 4,
        output_scale: int = 4,
        detail_channels: int | None = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            output_nc=output_nc,
            style_dim=style_dim,
            n_layers=n_layers,
            output_scale=output_scale,
        )
        if input_nc < 1:
            raise ValueError("input_nc must be positive")
        self.input_nc = input_nc
        self.detail_channels = detail_channels or max(embed_dim // 2, 16)
        self.detail_stem = nn.Sequential(
            nn.Conv2d(input_nc, self.detail_channels, 3, padding=1),
            nn.InstanceNorm2d(self.detail_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.detail_channels, self.detail_channels, 3, padding=1),
            nn.InstanceNorm2d(self.detail_channels, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        num_refinements = int(math.log2(output_scale))
        self.detail_projections = nn.ModuleList([
            nn.Conv2d(self.detail_channels, self.detail_channels, 3, padding=1)
            for _ in range(num_refinements)
        ])

        # Replace only the legacy refinement path. The Swin/coarse decoder
        # blocks and output head stay conceptually identical.
        refine_blocks = []
        refine_channels = embed_dim
        for _ in range(num_refinements):
            next_channels = max(embed_dim // 2, refine_channels // 2, 16)
            refine_blocks.append(
                AdaINDecoderBlock(
                    refine_channels,
                    next_channels,
                    style_dim,
                    skip_ch=self.detail_channels,
                    upsample=True,
                )
            )
            refine_channels = next_channels
        self.refine_blocks = nn.ModuleList(refine_blocks)
        self.head = nn.Sequential(
            nn.Conv2d(refine_channels, output_nc, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        style: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source is None:
            raise ValueError(
                "HighResolutionContentDecoder requires the source image"
            )
        if source.size(1) != self.input_nc:
            raise ValueError(
                f"expected {self.input_nc} source channels, got {source.size(1)}"
            )

        target_size = (
            features[0].shape[-2] * self.output_scale,
            features[0].shape[-1] * self.output_scale,
        )
        if source.shape[-2:] != target_size:
            source = F.interpolate(
                source, size=target_size, mode="bilinear", align_corners=False,
            )
        detail = self.detail_stem(source)

        x = features[-1]
        for index, block in enumerate(self.blocks):
            skip_index = len(features) - 2 - index
            skip = features[skip_index] if skip_index >= 0 else None
            x = block(x, style, skip)

        for projection, block in zip(self.detail_projections, self.refine_blocks):
            skip_size = (x.shape[-2] * 2, x.shape[-1] * 2)
            detail_skip = F.interpolate(
                detail, size=skip_size, mode="bilinear", align_corners=False,
            )
            detail_skip = projection(detail_skip)
            x = block(x, style, detail_skip)

        output = self.head(x)
        if output.shape[-2:] != target_size:
            output = F.interpolate(output, size=target_size, mode="nearest")
        return output
