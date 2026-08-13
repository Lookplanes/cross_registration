"""A modality-conditioned TransMorph with minimal architectural changes.

The original TransMorph joint spatial path is preserved:
``[fixed, moving] -> patch_embed -> Swin -> decoder -> flow``.  The only
model addition is ordered fixed/moving modality conditioning immediately
after patch embedding.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

from crossreg.translation.transcut.cln_adain import CLN2d, ModalityIDEmbedding

from .model import TransMorph


def config_from_transcut(
    saved: dict, img_size: tuple[int, int], *, input_nc: int,
) -> SimpleNamespace:
    """Build the TransMorph fields from a saved TransCUT configuration."""
    return SimpleNamespace(
        img_size=tuple(img_size),
        patch_size=saved.get("patch_size", 4),
        in_chans=2 * input_nc,
        embed_dim=saved.get("embed_dim", 96),
        depths=tuple(saved.get("depths", (2, 2, 4, 2))),
        num_heads=tuple(saved.get("num_heads", (4, 4, 8, 8))),
        window_size=tuple(saved.get("window_size", (8, 8))),
        mlp_ratio=saved.get("mlp_ratio", 4.0),
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=saved.get("drop_path_rate", 0.3),
        ape=saved.get("ape", False),
        spe=saved.get("spe", False),
        rpe=saved.get("rpe", True),
        patch_norm=False,
        use_checkpoint=False,
        out_indices=tuple(saved.get("out_indices", (0, 1, 2, 3))),
        pat_merg_rf=2,
        if_convskip=True,
        if_transskip=True,
        reg_head_chan=16,
        if_diffeomorphic=False,
    )


class ModalityConditionedTransMorph(TransMorph):
    """Original TransMorph with ordered pair-conditioned patch normalization."""

    def __init__(self, config, *, num_modalities: int, id_embed_dim: int,
                 image_channels: int):
        cfg = copy.deepcopy(config)
        cfg.in_chans = 2 * image_channels
        cfg.patch_norm = False
        super().__init__(cfg)
        self.num_modalities = num_modalities
        self.id_embed_dim = id_embed_dim
        self.image_channels = image_channels
        self.modality_embedding = ModalityIDEmbedding(
            num_modalities, id_embed_dim,
        )
        self.fixed_projection = nn.Linear(id_embed_dim, id_embed_dim, bias=False)
        self.moving_projection = nn.Linear(id_embed_dim, id_embed_dim, bias=False)
        self.pair_cln = CLN2d(cfg.embed_dim, id_embed_dim)
        with torch.no_grad():
            identity = torch.eye(id_embed_dim) * 0.5
            self.fixed_projection.weight.copy_(identity)
            self.moving_projection.weight.copy_(identity)

    def _validate_inputs(self, moving, fixed, moving_ids, fixed_ids):
        if moving.shape != fixed.shape:
            raise ValueError(
                f"moving and fixed must have equal shapes, got "
                f"{tuple(moving.shape)} and {tuple(fixed.shape)}"
            )
        if moving.shape[1] != self.image_channels:
            raise ValueError(
                f"expected {self.image_channels} image channels, "
                f"got {moving.shape[1]}"
            )
        batch = moving.shape[0]
        if moving_ids.shape != (batch,) or fixed_ids.shape != (batch,):
            raise ValueError("moving_ids and fixed_ids must both have shape (B,)")

    def predict_flow(self, moving, fixed, moving_ids, fixed_ids):
        """Predict moving-to-fixed displacement through original TransMorph."""
        self._validate_inputs(moving, fixed, moving_ids, fixed_ids)
        joint = torch.cat([fixed, moving], dim=1)
        shallow_half, shallow_full = self._conv_skip_features(joint)

        patch_map = self.transformer.patch_embed(joint)
        batch, channels, height, width = patch_map.shape
        tokens = patch_map.flatten(2).transpose(1, 2)
        fixed_condition = self.fixed_projection(
            self.modality_embedding(fixed_ids)
        )
        moving_condition = self.moving_projection(
            self.modality_embedding(moving_ids)
        )
        tokens = self.pair_cln(tokens, fixed_condition + moving_condition)
        patch_map = tokens.transpose(1, 2).reshape(
            batch, channels, height, width,
        )

        out_feats = self.transformer.forward_from_patch_map(patch_map)
        return self._decode_flow(out_feats, shallow_half, shallow_full)

    def forward(self, moving, fixed, moving_ids, fixed_ids):
        pos_flow = self.predict_flow(moving, fixed, moving_ids, fixed_ids)
        return self._finish_registration(moving, pos_flow)

    def initialize_from_transcut(self, checkpoint: dict) -> dict[str, int]:
        """Transfer compatible Stage 1 Encoder parameters.

        The single-image patch projection is copied into both role halves at
        half scale.  Swin, CLN and modality embedding parameters are copied
        exactly when shapes agree.
        """
        encoder_state = checkpoint["encoder"]
        own_state = self.state_dict()
        copied = 0

        with torch.no_grad():
            old_weight = encoder_state["swin.patch_embed.proj.weight"]
            new_weight = own_state["transformer.patch_embed.proj.weight"]
            if old_weight.shape[1] != self.image_channels:
                raise ValueError(
                    "Stage 1 patch channels do not match conditioned model: "
                    f"{old_weight.shape[1]} != {self.image_channels}"
                )
            if new_weight.shape[1] != 2 * old_weight.shape[1]:
                raise ValueError("conditioned patch projection is not a 2x expansion")
            new_weight[:, :self.image_channels].copy_(old_weight * 0.5)
            new_weight[:, self.image_channels:].copy_(old_weight * 0.5)
            copied += 1
            bias_key = "swin.patch_embed.proj.bias"
            if bias_key in encoder_state:
                own_state["transformer.patch_embed.proj.bias"].copy_(
                    encoder_state[bias_key]
                )
                copied += 1

            for old_key, value in encoder_state.items():
                if not old_key.startswith("swin.") or "patch_embed" in old_key:
                    continue
                new_key = "transformer." + old_key[len("swin."):]
                if new_key in own_state and own_state[new_key].shape == value.shape:
                    own_state[new_key].copy_(value)
                    copied += 1

            for old_prefix, new_prefix in (
                ("cln.", "pair_cln."),
                ("mod_embed.", "modality_embedding."),
            ):
                for old_key, value in encoder_state.items():
                    if not old_key.startswith(old_prefix):
                        continue
                    new_key = new_prefix + old_key[len(old_prefix):]
                    if new_key in own_state and own_state[new_key].shape == value.shape:
                        own_state[new_key].copy_(value)
                        copied += 1

        return {"copied_tensors": copied}
