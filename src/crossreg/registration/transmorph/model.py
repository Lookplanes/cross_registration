"""
TransMorph registration model.

The Swin-Transformer encoder is imported from
:mod:`crossreg.models.swin_transformer` so it can be shared with the
translation module (TransCUT).

References
----------
Liu, Z. et al. (2021). Swin Transformer. arXiv:2103.14030.
Chen, J. et al. (2022). TransMorph. arXiv:2111.10480.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from crossreg.models.swin_transformer import SwinTransformer

from . import config as configs

# =============================================================================
# Registration-specific building blocks
# =============================================================================


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=False)
        relu = nn.LeakyReLU(inplace=True)
        nm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.InstanceNorm2d(out_channels)
        super().__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels,
                                kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels,
                                kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                           padding=kernel_size // 2)
        conv2d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv2d.weight.shape))
        conv2d.bias = nn.Parameter(torch.zeros(conv2d.bias.shape))
        super().__init__(conv2d)


class SpatialTransformer(nn.Module):
    """N-D Spatial Transformer (from VoxelMorph)."""

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """Vector-field integration via scaling-and-squaring."""

    def __init__(self, inshape, nsteps=7):
        super().__init__()
        assert nsteps >= 0, f'nsteps should be >= 0, found: {nsteps}'
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


# =============================================================================
# TransMorph
# =============================================================================


class TransMorph(nn.Module):
    """TransMorph: Swin-Transformer encoder + CNN decoder + registration head."""

    def __init__(self, config):
        super().__init__()
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim

        self.transformer = SwinTransformer(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf,
        )

        self.up0 = DecoderBlock(embed_dim * 8, embed_dim * 4,
                                skip_channels=embed_dim * 4 if self.if_transskip else 0,
                                use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2,
                                skip_channels=embed_dim * 2 if self.if_transskip else 0,
                                use_batchnorm=False)
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim,
                                skip_channels=embed_dim if self.if_transskip else 0,
                                use_batchnorm=False)
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2,
                                skip_channels=embed_dim // 2 if self.if_convskip else 0,
                                use_batchnorm=False)
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if self.if_convskip else 0,
                                use_batchnorm=False)

        self.c1 = Conv2dReLU(config.in_chans, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv2dReLU(config.in_chans, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(in_channels=config.reg_head_chan, out_channels=2, kernel_size=3)

        self.if_diffeomorphic = getattr(config, 'if_diffeomorphic', False)
        if self.if_diffeomorphic:
            self.integrate = VecInt(config.img_size, nsteps=7)
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def _conv_skip_features(self, joint_input):
        """Return the original TransMorph shallow joint-input skips."""
        if not self.if_convskip:
            return None, None
        shallow_half = self.c1(self.avg_pool(joint_input))
        shallow_full = self.c2(joint_input)
        return shallow_half, shallow_full

    def _decode_flow(self, out_feats, shallow_half=None, shallow_full=None):
        """Decode Swin features with the original TransMorph decoder."""
        f1 = out_feats[-2] if self.if_transskip else None
        f2 = out_feats[-3] if self.if_transskip else None
        f3 = out_feats[-4] if self.if_transskip else None

        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, shallow_half)
        x = self.up4(x, shallow_full)
        return self.reg_head(x)

    def _finish_registration(self, source, pos_flow):
        flow = self.integrate(pos_flow) if self.if_diffeomorphic else pos_flow
        warped = self.spatial_trans(source, flow)
        return warped, flow, pos_flow

    def forward(self, moving, fixed=None):
        """Warp ``moving`` into ``fixed`` coordinates.

        The explicit two-tensor interface is preferred.  For compatibility,
        a concatenated ``[fixed, moving]`` tensor is also accepted.
        """
        if fixed is None:
            x = moving
            src_ch = x.shape[1] // 2
            source = x[:, src_ch:, :, :]
        else:
            if moving.shape != fixed.shape:
                raise ValueError(
                    f"moving and fixed must have equal shapes, got "
                    f"{tuple(moving.shape)} and {tuple(fixed.shape)}"
                )
            source = moving
            x = torch.cat([fixed, moving], dim=1)

        f4, f5 = self._conv_skip_features(x)
        out_feats = self.transformer(x)
        pos_flow = self._decode_flow(out_feats, f4, f5)
        return self._finish_registration(source, pos_flow)


# =============================================================================
# Pre-defined configurations
# =============================================================================

CONFIGS = {
    'TransMorph': configs.get_2DTransMorph_config(),
    'TransMorph-No-Conv-Skip': configs.get_2DTransMorphNoConvSkip_config(),
    'TransMorph-No-Trans-Skip': configs.get_2DTransMorphNoTransSkip_config(),
    'TransMorph-No-Skip': configs.get_2DTransMorphNoSkip_config(),
    'TransMorph-Lrn': configs.get_2DTransMorphLrn_config(),
    'TransMorph-Sin': configs.get_2DTransMorphSin_config(),
    'TransMorph-Sin-RGB': configs.get_2DTransMorphSinRGB_config(),
    'TransMorph-No-RelPosEmbed': configs.get_2DTransMorphNoRelativePosEmbd_config(),
    'TransMorph-Large': configs.get_2DTransMorphLarge_config(),
    'TransMorph-Small': configs.get_2DTransMorphSmall_config(),
    'TransMorph-Tiny': configs.get_2DTransMorphTiny_config(),
}


def get_affine_net(config):
    raise NotImplementedError("TransMorphAffine2D is not migrated yet.")
