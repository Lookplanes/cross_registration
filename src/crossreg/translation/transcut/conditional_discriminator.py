"""Target-modality-conditioned PatchGAN discriminator."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from crossreg.translation.cut.networks import NLayerDiscriminator


class ConditionalPatchDiscriminator(nn.Module):
    """PatchGAN with a projection term conditioned on the target modality.

    The convolutional trunk remains shared across modalities. Adding a new
    modality only adds one embedding row instead of a complete discriminator.
    """

    def __init__(self, input_nc: int, num_modalities: int,
                 ndf: int = 64, n_layers: int = 3):
        super().__init__()
        base = NLayerDiscriminator(input_nc, ndf, n_layers)
        layers = list(base.model.children())
        self.features = nn.Sequential(*layers[:-1])
        self.head = layers[-1]
        feature_channels = ndf * min(2 ** n_layers, 8)
        self.modality_embedding = nn.Embedding(num_modalities, feature_channels)
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=0.02)
        self.feature_channels = feature_channels

    def forward(self, image: torch.Tensor,
                modality_id: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        logits = self.head(features)
        condition = self.modality_embedding(modality_id).view(
            image.size(0), self.feature_channels, 1, 1,
        )
        projection = (features * condition).sum(dim=1, keepdim=True)
        projection = projection / math.sqrt(self.feature_channels)
        if projection.shape[-2:] != logits.shape[-2:]:
            projection = F.adaptive_avg_pool2d(projection, logits.shape[-2:])
        return logits + projection
