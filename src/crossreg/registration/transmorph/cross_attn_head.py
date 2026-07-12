"""
Cross-Attention Registration Head for TransMorph v2.

Compares two feature streams (real image F₁, perturbed synthetic F₂')
via multi-scale cross-attention to estimate the deformation field D_pred.

Architecture
------------
::

    F₁ (real) ──┐                 F₂' (synthetic) ──┐
                 │                                    │
    ┌────────────▼────────────────────────────────────▼────────────┐
    │  Cross-Attention Fusion (per scale)                          │
    │    Q = F₁ · Wq    K,V = F₂' · Wk, Wv                        │
    │    attn = softmax(Q·Kᵀ/√d)                                  │
    │    fused = attn · V  +  F₁  (residual)                       │
    └──────────────────────────┬───────────────────────────────────┘
                               │
                        CNN Decoder (Conv+Upsample)
                               │
                               ▼
                         D_pred (flow field)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Cross-Attention Fusion Block
# =============================================================================


class CrossAttentionFusion(nn.Module):
    """Fuse two feature maps via cross-attention.

    Q = Linear(F₁), K/V = Linear(F₂')
    Output = Attention(Q, K, V) + F₁  (residual connection).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """*f1*, *f2*: (B, C, H, W).  Returns fused (B, C, H, W)."""
        B, C, H, W = f1.shape

        # Flatten spatial dims: (B, C, H, W) → (B, H*W, C)
        f1_flat = f1.flatten(2).transpose(1, 2)
        f2_flat = f2.flatten(2).transpose(1, 2)

        # Multi-head projections
        q = self.q_proj(f1_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(f2_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(f2_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out).transpose(1, 2).view(B, C, H, W)

        return out + f1  # residual


class MultiScaleCrossAttentionFuser(nn.Module):
    """Apply cross-attention fusion at multiple feature scales.

    Takes paired feature lists from the shared encoder (both images),
    fuses them at each scale, and returns fused features ready for
    the decoder.

    Parameters
    ----------
    feature_dims : list[int]   Channel dimensions at each scale, largest first.
    num_heads : int
    """

    def __init__(self, feature_dims: list[int], num_heads: int = 4):
        super().__init__()
        self.fusers = nn.ModuleList([
            CrossAttentionFusion(d, num_heads) for d in feature_dims
        ])

    def forward(self, feats1: list[torch.Tensor],
                feats2: list[torch.Tensor]) -> list[torch.Tensor]:
        """*feats1*, *feats2*: lists of (B, C, H, W) from shared encoder."""
        return [fuser(f1, f2) for fuser, f1, f2 in zip(self.fusers, feats1, feats2)]


# =============================================================================
# Conv2dReLU helper (copied from TransMorph — kept self-contained)
# =============================================================================


class _Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )


class _DecoderBlock(nn.Module):
    """Upsample + Conv blocks, optionally with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int = 0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = _Conv2dReLU(in_ch + skip_ch, out_ch, 3, padding=1)
        self.conv2 = _Conv2dReLU(out_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


# =============================================================================
# Cross-Attention Registration Head (top-level)
# =============================================================================


class CrossAttentionRegHead(nn.Module):
    """Registration head with cross-attention feature fusion + CNN decoder.

    Takes multi-scale features from two images (real + synthetic),
    fuses via cross-attention, and decodes to a 2-channel flow field.

    Parameters
    ----------
    embed_dim : int            Swin embed dim (e.g. 96).
    out_indices : tuple[int]   Which Swin stages to use. Default (0,1,2,3).
    reg_head_chan : int        Channels before final flow conv. Default 16.
    num_heads : int            Cross-attention heads. Default 4.
    """

    def __init__(self, embed_dim: int = 96, out_indices: tuple = (0, 1, 2, 3),
                 reg_head_chan: int = 16, num_heads: int = 4):
        super().__init__()
        n = len(out_indices)
        dims = [embed_dim * (2 ** i) for i in out_indices]  # feature dims at each stage

        # Cross-attention fusion at each scale
        self.fuser = MultiScaleCrossAttentionFuser(dims, num_heads)

        # CNN decoder (same pattern as TransMorph, but takes fused features)
        dims_rev = list(reversed(dims))  # [high_dim, ..., low_dim]
        self.decoder_blocks = nn.ModuleList()
        for i in range(n):
            in_c = dims_rev[i]
            out_c = dims_rev[i + 1] if i + 1 < n else reg_head_chan
            skip_c = dims_rev[i + 1] if i + 1 < n else 0
            self.decoder_blocks.append(
                _DecoderBlock(in_c, out_c, skip_ch=skip_c)
            )

        # Final flow prediction
        self.flow_head = nn.Conv2d(reg_head_chan, 2, kernel_size=3, padding=1)
        # Small initial weights for stable training start
        nn.init.normal_(self.flow_head.weight, 0, 1e-5)
        nn.init.zeros_(self.flow_head.bias)

    def forward(self, feats1: list[torch.Tensor],
                feats2: list[torch.Tensor]) -> torch.Tensor:
        """*feats1*, *feats2*: feature lists from shared encoder.

        Returns (B, 2, H, W) displacement field (pos_flow, before VecInt).
        """
        # Cross-attention fusion
        fused = self.fuser(feats1, feats2)  # same order as input (small→large)

        # Decode (from high-level to low-level with skip connections)
        x = fused[-1]  # smallest spatial, highest channel
        for i, blk in enumerate(self.decoder_blocks):
            skip_idx = len(fused) - 2 - i
            skip = fused[skip_idx] if skip_idx >= 0 else None
            x = blk(x, skip)

        return self.flow_head(x)
