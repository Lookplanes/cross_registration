"""Geometry-biased TransCUT decoder with a direct full-resolution carrier.

This optional decoder keeps the source image on the output path and learns
only a target-conditioned residual.  The residual branch is shared by every
modality, uses the existing AdaIN conditioning, and also consumes the finest
Swin/CLN feature map for context.  It introduces no modality-specific heads
and leaves the legacy decoders checkpoint-compatible and unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cln_adain import AdaIN2d


class FullResolutionAdaINResidualBlock(nn.Module):
    """Stride-one residual block conditioned by the target style."""

    def __init__(self, channels: int, style_dim: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation,
        )
        self.adain1 = AdaIN2d(channels, style_dim)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation,
        )
        self.adain2 = AdaIN2d(channels, style_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.adain1(self.conv1(x), style))
        residual = self.adain2(self.conv2(residual), style)
        return self.act(x + residual)


class FullResolutionResidualDecoder(nn.Module):
    """Predict target appearance as a full-resolution residual on the source.

    The source carrier makes exact pixel geometry available at the output.
    The network is still free to change intensity, colour and local texture,
    but no downsample/upsample reconstruction is required for those pixels.
    The zero-initialized output layer makes the initial mapping exactly the
    identity and is trained normally from the first optimizer step.
    """

    def __init__(
        self,
        embed_dim: int = 96,
        input_nc: int = 1,
        output_nc: int = 1,
        style_dim: int = 64,
        n_layers: int = 4,
        output_scale: int = 4,
        hidden_channels: int | None = None,
    ):
        super().__init__()
        if input_nc != output_nc:
            raise ValueError(
                "fullres_residual requires equal input/output channels for "
                "the direct source carrier"
            )
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.output_scale = output_scale
        channels = hidden_channels or max(embed_dim // 2, 24)

        self.source_stem = nn.Conv2d(input_nc, channels, 3, padding=1)
        self.feature_projection = nn.Conv2d(embed_dim, channels, 1)
        dilations = (1, 2, 4, 1)
        self.blocks = nn.ModuleList([
            FullResolutionAdaINResidualBlock(
                channels, style_dim, dilation=dilations[index % len(dilations)],
            )
            for index in range(max(1, n_layers))
        ])
        self.residual_head = nn.Conv2d(channels, output_nc, 3, padding=1)
        self.reset_output_to_identity()

    def reset_output_to_identity(self) -> None:
        """Initialize the learned residual to zero without freezing it."""
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        features: list[torch.Tensor],
        style: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source is None:
            raise ValueError("FullResolutionResidualDecoder requires the source image")
        if source.size(1) != self.input_nc:
            raise ValueError(
                f"expected {self.input_nc} source channels, got {source.size(1)}"
            )
        if not features:
            raise ValueError("FullResolutionResidualDecoder requires encoder features")

        finest = F.interpolate(
            features[0], size=source.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        x = self.source_stem(source) + self.feature_projection(finest)
        for block in self.blocks:
            x = block(x, style)
        residual = self.residual_head(x)
        return torch.clamp(source + residual, -1.0, 1.0)
