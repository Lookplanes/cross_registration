"""
Conditional Layer Normalization (CLN) and Adaptive Instance Normalization (AdaIN).

Used by TransCUT to inject modality identity into the shared Swin-Transformer
encoder (CLN) and the CNN decoder (AdaIN).

  - CLN:  ``output = norm(x) * γ(id) + β(id)`` —   inject at patch-embed layer.
  - AdaIN: ``output = σ(id) * (x - μ) / σ + μ(id)`` — inject in decoder blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityIDEmbedding(nn.Module):
    """Learnable embedding for modality identity.

    Maps an integer modality index (0..N-1) to a *dim*-dimensional vector.
    """

    def __init__(self, num_modalities: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_modalities, dim)

    def forward(self, mod_ids: torch.Tensor) -> torch.Tensor:
        """*mod_ids*: (B,) integer tensor."""
        return self.embedding(mod_ids)  # (B, dim)


class CLN2d(nn.Module):
    """Conditional Layer Normalization for 2D feature maps.

    Normalizes over (C, H, W), then applies a modality-conditioned
    affine transform: ``γ(id) * norm(x) + β(id)``.

    Parameters
    ----------
    num_features : int  Number of feature channels (C).
    cond_dim : int       Dimensionality of the conditioning vector.
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """*x*: (B, L, C) or (B, C, H, W).  *cond*: (B, cond_dim)."""
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # → (B, L, C)
        x_norm = self.norm(x)
        gamma = self.gamma(cond).unsqueeze(1)   # (B, 1, C)
        beta = self.beta(cond).unsqueeze(1)     # (B, 1, C)
        return x_norm * gamma + beta


class AdaIN2d(nn.Module):
    """Adaptive Instance Normalization for 2D feature maps.

    ``AdaIN(x, s) = σ(s) * norm(x) + μ(s)`` where *s* is the style (target
    modality) vector and *norm* is instance-normalization without learnable
    affine parameters.
    """

    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_bias = nn.Linear(style_dim, num_features)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """*x*: (B, C, H, W).  *style*: (B, style_dim)."""
        x_norm = self.norm(x)
        scale = self.style_scale(style).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        bias = self.style_bias(style).unsqueeze(-1).unsqueeze(-1)
        return x_norm * scale + bias
