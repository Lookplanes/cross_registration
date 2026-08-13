"""
End-to-end cross-modality registration pipeline inference.

Pipeline:
  source (modality A) --> CUT translate --> pseudo-B --> TransMorph register --> warped + flow
"""

from __future__ import annotations

import torch
import torch.nn as nn
import copy
from dataclasses import dataclass, field


@dataclass
class PipelineOutput:
    """Output of a pipeline forward pass."""

    translated: torch.Tensor  # (B, C, H, W) — CUT output (source translated to target modality)
    warped: torch.Tensor      # (B, C, H, W) — TransMorph warped result
    flow: torch.Tensor        # (B, 2, H, W) — displacement field
    pos_flow: torch.Tensor    # (B, 2, H, W) — raw flow before VecInt (if diffeomorphic)


class PipelineInference(nn.Module):
    """End-to-end inference: CUT translation -> TransMorph registration.

    Usage::

        pipeline = PipelineInference(cut_model, transmorph_model)
        result = pipeline(source_img, target_img)
        # result.translated:  CUT output
        # result.warped:      registered result
        # result.flow:        displacement field
    """

    def __init__(self, cut_model: nn.Module, transmorph_model: nn.Module):
        super().__init__()
        self.cut = cut_model
        self.transmorph = transmorph_model

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> PipelineOutput:
        """Run the full pipeline.

        Args:
            source: Source domain image, shape (B, C, H, W).
            target: Target domain image, shape (B, C, H, W).

        Returns:
            PipelineOutput with translated, warped, flow, pos_flow.
        """
        # Step 1: Translation (source -> target modality)
        translated = self.cut(source)

        # Step 2: Registration
        # TransMorph expects concatenated [fixed, moving] = [target, translated]
        registration_input = torch.cat([target, translated], dim=1)
        warped, flow, pos_flow = self.transmorph(registration_input)

        return PipelineOutput(
            translated=translated,
            warped=warped,
            flow=flow,
            pos_flow=pos_flow,
        )

    @torch.no_grad()
    def infer(self, source: torch.Tensor, target: torch.Tensor) -> PipelineOutput:
        """Run inference without gradient tracking."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(source, target)
        finally:
            if was_training:
                self.train()


def build_pipeline(
    cut_input_nc: int = 1,
    cut_output_nc: int = 1,
    cut_ngf: int = 64,
    cut_netG: str = "resnet_9blocks",
    transmorph_img_size: tuple[int, int] = (256, 256),
    transmorph_config_name: str = "TransMorph",
    device: str = "cpu",
) -> PipelineInference:
    """Factory: build a PipelineInference with specified sub-models (random init).

    Args:
        cut_input_nc: CUT input channels.
        cut_output_nc: CUT output channels.
        cut_ngf: CUT generator base filters.
        cut_netG: CUT generator architecture.
        transmorph_img_size: (H, W) for TransMorph spatial transformer.
        transmorph_config_name: Key in TransMorph CONFIGS dict.
        device: "cpu" or "cuda".

    Returns:
        Configured PipelineInference (random weights).
    """
    from ..translation.cut import CUTInference
    from ..registration.transmorph.model import TransMorph, CONFIGS

    # Build CUT inference model
    cut_model = CUTInference(
        input_nc=cut_input_nc,
        output_nc=cut_output_nc,
        ngf=cut_ngf,
        netG=cut_netG,
        gpu_ids=[0] if device == "cuda" else [],
    ).to(device)

    # Build TransMorph
    config = copy.deepcopy(CONFIGS[transmorph_config_name])
    config.in_chans = cut_output_nc * 2  # 2 * C for concatenated input
    config.img_size = transmorph_img_size
    transmorph_model = TransMorph(config).to(device)

    return PipelineInference(cut_model, transmorph_model).to(device)
