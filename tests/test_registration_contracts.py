"""Numerical contracts for deformation fields and registration inputs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_spatial_transformer_identity_and_xy_direction() -> None:
    from crossreg.registration.transmorph.model import SpatialTransformer

    transform = SpatialTransformer((16, 16))
    moving = torch.zeros(1, 1, 16, 16)
    moving[0, 0, 8, 9] = 1.0

    identity = transform(moving, torch.zeros(1, 2, 16, 16))
    assert torch.allclose(identity, moving, atol=1e-6)

    # backward sampling: dx=+1 makes output[x=8] sample moving[x=9]
    flow = torch.zeros(1, 2, 16, 16)
    flow[:, 1] = 1.0
    shifted = transform(moving, flow)
    assert shifted[0, 0, 8, 8] > 0.999
    assert shifted[0, 0, 8, 9] < 1e-6


def test_registration_sample_collector_exports_and_prunes(tmp_path: Path) -> None:
    from crossreg.registration.visualization import (
        RegistrationSampleCollector,
        prune_sample_snapshots,
        update_sample_alias,
    )

    collector = RegistrationSampleCollector(samples_per_direction=1)
    image = torch.zeros(3, 16, 16)
    flow = torch.zeros(2, 16, 16)
    flow[1] = 1.0
    collector.add(
        sample_id="sample-a", direction="he->dapi",
        moving=image, fixed=image, warped=image,
        target_flow=flow, predicted_flow=torch.zeros_like(flow),
        valid_mask=torch.ones(16, 16),
    )
    # The direction quota makes validation snapshots stable and bounded.
    collector.add(
        sample_id="sample-b", direction="he->dapi",
        moving=image, fixed=image, warped=image,
        target_flow=flow, predicted_flow=flow,
        valid_mask=torch.ones(16, 16),
    )
    assert len(collector.samples) == 1

    samples = tmp_path / "samples"
    first = collector.save(samples / "epoch_0001.png", flow_limit=2, arrow_step=8)
    assert first.is_file()
    with Image.open(first) as sheet:
        assert sheet.size == (7 * 16, 16 + 34)
    update_sample_alias(first, samples / "latest.png")
    assert (samples / "latest.png").is_file()

    for epoch in (2, 3):
        collector.save(samples / f"epoch_{epoch:04d}.png", flow_limit=2)
    prune_sample_snapshots(samples, keep=2)
    assert not (samples / "epoch_0001.png").exists()
    assert (samples / "epoch_0002.png").is_file()
    assert (samples / "epoch_0003.png").is_file()
    assert (samples / "latest.png").is_file()


def test_flow_resize_preserves_pixel_units(tmp_path: Path) -> None:
    from crossreg.data.datasets import PairedImageFolderDataset

    for name in ("moving", "fixed", "gt_flow"):
        (tmp_path / name).mkdir()
    image = np.zeros((16, 16), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "moving" / "a.png")
    Image.fromarray(image).save(tmp_path / "fixed" / "a.png")
    flow = np.stack([
        np.ones((16, 16), dtype=np.float32),
        np.full((16, 16), 2.0, dtype=np.float32),
    ])
    np.save(tmp_path / "gt_flow" / "a.npy", flow)

    sample = PairedImageFolderDataset(
        str(tmp_path), flow_subdir="gt_flow", img_size=(32, 32),
        grayscale=True, require_flow=True,
    )[0]
    resized = sample["flow"]
    assert resized.shape == (2, 32, 32)
    assert torch.allclose(resized[0], torch.full((32, 32), 2.0))
    assert torch.allclose(resized[1], torch.full((32, 32), 4.0))


def test_transmorph_zero_flow_returns_moving_not_fixed() -> None:
    from crossreg.registration.transmorph.config import get_2DTransMorphTiny_config
    from crossreg.registration.transmorph.model import TransMorph

    cfg = get_2DTransMorphTiny_config()
    cfg.img_size = (64, 64)
    cfg.in_chans = 2
    model = TransMorph(cfg).eval()
    for parameter in model.reg_head.parameters():
        parameter.data.zero_()

    moving = torch.rand(1, 1, 64, 64)
    fixed = torch.zeros_like(moving)
    with torch.no_grad():
        warped, flow, _ = model(moving, fixed)
    assert torch.count_nonzero(flow) == 0
    assert torch.allclose(warped, moving, atol=1e-5)
    assert not torch.allclose(warped, fixed)


def test_cross_attention_residual_policy_has_no_implicit_fixed_path() -> None:
    from crossreg.registration.transmorph.cross_attn_head import (
        CrossAttentionFusion,
    )

    fixed = torch.randn(1, 4, 4, 4)
    moving = torch.randn_like(fixed)
    no_residual = CrossAttentionFusion(4, num_heads=1, residual="none")
    legacy = CrossAttentionFusion(4, num_heads=1, residual="fixed_query")
    legacy.load_state_dict(no_residual.state_dict())

    # Eliminate the learned attention branch.  Only the explicitly requested
    # legacy policy may still expose fixed features to the decoder.
    for module in (no_residual, legacy):
        for parameter in module.parameters():
            parameter.data.zero_()

    assert torch.count_nonzero(no_residual(fixed, moving)) == 0
    assert torch.allclose(legacy(fixed, moving), fixed)


def test_no_residual_cross_attention_uses_both_feature_streams() -> None:
    from crossreg.registration.transmorph.cross_attn_head import (
        CrossAttentionFusion,
    )

    fusion = CrossAttentionFusion(4, num_heads=1, residual="none")
    fixed = torch.randn(1, 4, 3, 3, requires_grad=True)
    moving = torch.randn(1, 4, 3, 3, requires_grad=True)
    fusion(fixed, moving).square().mean().backward()

    assert fixed.grad is not None and fixed.grad.abs().sum() > 0
    assert moving.grad is not None and moving.grad.abs().sum() > 0


def test_integrated_velocity_has_usable_inverse() -> None:
    from crossreg.registration.transmorph.model import SpatialTransformer, VecInt

    size = (32, 32)
    transform = SpatialTransformer(size)
    integrate = VecInt(size, nsteps=7)
    fixed = torch.zeros(1, 1, *size)
    fixed[:, :, 8:24, 8:24] = 1.0
    velocity = torch.zeros(1, 2, *size)
    velocity[:, 1] = 1.5

    gt_flow = integrate(velocity)
    inverse_flow = integrate(-velocity)
    moving = transform(fixed, inverse_flow)
    reconstructed = transform(moving, gt_flow)
    # Ignore the outer sampling border; the central synthetic anatomy should invert.
    assert torch.mean(torch.abs(reconstructed[:, :, 3:-3, 3:-3] - fixed[:, :, 3:-3, 3:-3])) < 0.03


def test_stage2_cross_modal_pair_uses_direct_flow_without_inverse() -> None:
    from scripts.train_synthetic_supervised import _construct_cross_modal_pair
    from crossreg.registration.transmorph.model import SpatialTransformer

    size = (24, 24)
    transform = SpatialTransformer(size)
    aligned_src = torch.zeros(2, 1, *size)
    aligned_src[:, :, 7:17, 8:16] = 1.0
    # Equal pixels isolate the geometry contract; distinct IDs verify that the
    # two encoder streams remain cross-modal.
    aligned_tgt = aligned_src.clone()
    flow = torch.zeros(2, 2, *size)
    flow[:, 1] = 2.0
    src_ids = torch.tensor([0, 0])
    tgt_ids = torch.tensor([3, 3])

    moving, fixed, moving_ids, fixed_ids = _construct_cross_modal_pair(
        aligned_src, aligned_tgt, flow, src_ids, tgt_ids,
        transform, direction="source-moving",
    )
    assert torch.allclose(transform(moving, flow), fixed, atol=1e-6)
    assert torch.equal(moving_ids, src_ids)
    assert torch.equal(fixed_ids, tgt_ids)

    moving, fixed, moving_ids, fixed_ids = _construct_cross_modal_pair(
        aligned_src, aligned_tgt, flow, src_ids, tgt_ids,
        transform, direction="target-moving",
    )
    assert torch.allclose(transform(moving, flow), fixed, atol=1e-6)
    assert torch.equal(moving_ids, tgt_ids)
    assert torch.equal(fixed_ids, src_ids)


def test_stage2_normalizes_dataset_range_for_stage1_encoder() -> None:
    from scripts.train_synthetic_supervised import _normalize_stage1_input

    image = torch.tensor([0.0, 0.5, 1.0])
    assert torch.equal(
        _normalize_stage1_input(image), torch.tensor([-1.0, 0.0, 1.0]),
    )


def test_stage2_can_separate_pair_generator_and_registration_encoder(
    tmp_path: Path,
) -> None:
    from scripts.train_synthetic_supervised import build_stage2_model
    from crossreg.translation.transcut import TransCUT, TransCUTConfig

    cfg = TransCUTConfig(
        num_modalities=4, img_size=32, input_nc=1, output_nc=1,
        embed_dim=16, id_embed_dim=8, decoder_style_dim=8,
        depths=(1, 1, 1, 1), num_heads=(4, 4, 8, 8),
        window_size=(4, 4), out_indices=(0, 1, 2, 3),
        drop_path_rate=0.0, ndf=8, netF_nc=8, num_patches=4,
        gpu_ids=[],
    )
    generator = TransCUT(cfg)
    registration = TransCUT(cfg)
    with torch.no_grad():
        generator.encoder.swin.patch_embed.proj.weight.fill_(0.25)
        registration.encoder.swin.patch_embed.proj.weight.fill_(0.75)
    generator_path = tmp_path / "generator.pth"
    encoder_path = tmp_path / "encoder.pth"
    torch.save(generator.training_state(), generator_path)
    torch.save(registration.training_state(), encoder_path)
    args = SimpleNamespace(
        transcut_ckpt=str(encoder_path), generator_ckpt=str(generator_path),
        encoder_init="checkpoint", img_size=[32, 32], num_modalities=None,
        embed_dim=None, src_modality=0, tgt_modality=1, seed=7,
        registration_model="deprecated_cross_attention",
    )
    pair_encoder, _, _, _, encoder, _, _ = build_stage2_model(
        args, torch.device("cpu"),
    )
    assert torch.all(
        pair_encoder.swin.patch_embed.proj.weight == 0.25
    )
    assert torch.all(encoder.swin.patch_embed.proj.weight == 0.75)

    args.registration_model = "conditioned_transmorph"
    pair_encoder, _, _, _, encoder, registration_model, _ = (
        build_stage2_model(args, torch.device("cpu"))
    )
    assert encoder is None
    assert torch.all(
        pair_encoder.swin.patch_embed.proj.weight == 0.25
    )
    conditioned_patch = registration_model.transformer.patch_embed.proj.weight
    assert torch.all(conditioned_patch[:, :1] == 0.375)
    assert torch.all(conditioned_patch[:, 1:] == 0.375)


def test_conditioned_transmorph_reuses_backbone_and_expands_patch_input() -> None:
    from crossreg.registration.transmorph.conditioned_model import (
        ModalityConditionedTransMorph,
        config_from_transcut,
    )
    from crossreg.registration.transmorph.model import TransMorph
    from crossreg.translation.transcut import TransCUT, TransCUTConfig

    cfg = TransCUTConfig(
        num_modalities=4, img_size=32, input_nc=1, output_nc=1,
        embed_dim=16, id_embed_dim=8, decoder_style_dim=8,
        depths=(1, 1, 1, 1), num_heads=(4, 4, 8, 8),
        window_size=(4, 4), out_indices=(0, 1, 2, 3),
        drop_path_rate=0.0, ndf=8, netF_nc=8, num_patches=4,
        gpu_ids=[],
    )
    stage1 = TransCUT(cfg)
    checkpoint = stage1.training_state()
    trans_cfg = config_from_transcut(
        checkpoint["config"], (32, 32), input_nc=1,
    )
    model = ModalityConditionedTransMorph(
        trans_cfg, num_modalities=4, id_embed_dim=8, image_channels=1,
    )
    assert isinstance(model, TransMorph)
    report = model.initialize_from_transcut(checkpoint)
    assert report["copied_tensors"] > 0

    old_patch = checkpoint["encoder"]["swin.patch_embed.proj.weight"]
    new_patch = model.transformer.patch_embed.proj.weight.detach()
    assert torch.allclose(new_patch[:, :1], old_patch * 0.5)
    assert torch.allclose(new_patch[:, 1:], old_patch * 0.5)

    moving = torch.rand(2, 1, 32, 32)
    fixed = torch.rand_like(moving)
    warped, flow, pos_flow = model(
        moving, fixed, torch.tensor([0, 1]), torch.tensor([2, 3]),
    )
    assert warped.shape == moving.shape
    assert flow.shape == pos_flow.shape == (2, 2, 32, 32)


def test_stage3_uses_conditioned_ids_and_displacement_directly() -> None:
    from crossreg.pipeline.inference_v2 import Stage3Inference

    class RecordingEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_ids: list[torch.Tensor] = []

        def forward(self, image: torch.Tensor, ids: torch.Tensor):
            self.seen_ids.append(ids.detach().clone())
            return [image]

    class ZeroHead(torch.nn.Module):
        def forward(self, fixed_features, moving_features):
            image = fixed_features[0]
            return torch.zeros(image.shape[0], 2, *image.shape[-2:])

    encoder = RecordingEncoder()
    model = Stage3Inference(
        encoder, ZeroHead(), img_size=(16, 16),
        flow_parameterization="displacement",
    )
    assert model.integrate is None
    moving = torch.rand(1, 1, 16, 16)
    fixed = torch.zeros_like(moving)
    moving_ids = torch.tensor([2])
    fixed_ids = torch.tensor([1])
    warped, flow = model(moving, fixed, moving_ids, fixed_ids)
    assert torch.allclose(warped, moving, atol=1e-6)
    assert torch.count_nonzero(flow) == 0
    # Stage 3 encodes moving first and fixed second, each with its own ID.
    assert torch.equal(encoder.seen_ids[0], moving_ids)
    assert torch.equal(encoder.seen_ids[1], fixed_ids)


def test_stage3_factory_restores_conditioned_encoder(tmp_path: Path) -> None:
    from crossreg.pipeline.inference_v2 import build_stage3_from_checkpoints
    from crossreg.registration.transmorph.cross_attn_head import CrossAttentionRegHead
    from crossreg.translation.transcut import TransCUT, TransCUTConfig

    cfg = TransCUTConfig(
        num_modalities=3, img_size=32, input_nc=1, output_nc=1,
        embed_dim=16, id_embed_dim=8, decoder_style_dim=8,
        depths=(1, 1), num_heads=(4, 4), window_size=(4, 4),
        out_indices=(0, 1), drop_path_rate=0.0,
        ndf=8, netF_nc=8, num_patches=4, gpu_ids=[],
    )
    transcut = TransCUT(cfg)
    encoder_path = tmp_path / "stage1.pth"
    torch.save(transcut.training_state(), encoder_path)

    reg_head = CrossAttentionRegHead(
        embed_dim=16, out_indices=(0, 1), reg_head_chan=16, num_heads=4,
        fusion_residual="none",
    )
    reg_path = tmp_path / "stage2.pth"
    torch.save({
        "reg_head": reg_head.state_dict(),
        "model_config": {
            "flow_parameterization": "displacement",
            "fusion_residual": "none",
        },
    }, reg_path)

    model = build_stage3_from_checkpoints(
        str(encoder_path), str(reg_path), img_size=(32, 32),
        device="cpu",
    )
    moving = torch.rand(1, 1, 32, 32)
    fixed = torch.rand_like(moving)
    with torch.no_grad():
        warped, flow = model(
            moving, fixed, torch.tensor([0]), torch.tensor([2]),
        )
    assert warped.shape == moving.shape
    assert flow.shape == (1, 2, 32, 32)
    assert model.flow_parameterization == "displacement"
    assert model.reg_head.fusion_residual == "none"

    legacy_path = tmp_path / "stage2_legacy.pth"
    torch.save({
        "reg_head": reg_head.state_dict(),
        "model_config": {"flow_parameterization": "displacement"},
    }, legacy_path)
    legacy_model = build_stage3_from_checkpoints(
        str(encoder_path), str(legacy_path), img_size=(32, 32),
        device="cpu",
    )
    assert legacy_model.reg_head.fusion_residual == "fixed_query"


def test_stage3_factory_restores_conditioned_transmorph(tmp_path: Path) -> None:
    from crossreg.pipeline.inference_v2 import (
        ConditionedStage3Inference,
        build_stage3_from_checkpoints,
    )
    from crossreg.registration.transmorph.conditioned_model import (
        ModalityConditionedTransMorph,
        config_from_transcut,
    )
    from crossreg.translation.transcut import TransCUT, TransCUTConfig

    cfg = TransCUTConfig(
        num_modalities=3, img_size=32, input_nc=1, output_nc=1,
        embed_dim=16, id_embed_dim=8, decoder_style_dim=8,
        depths=(1, 1, 1, 1), num_heads=(4, 4, 8, 8),
        window_size=(4, 4), out_indices=(0, 1, 2, 3),
        drop_path_rate=0.0, ndf=8, netF_nc=8, num_patches=4,
        gpu_ids=[],
    )
    transcut = TransCUT(cfg)
    stage1_state = transcut.training_state()
    encoder_path = tmp_path / "conditioned_stage1.pth"
    torch.save(stage1_state, encoder_path)

    trans_cfg = config_from_transcut(
        stage1_state["config"], (32, 32), input_nc=1,
    )
    registration = ModalityConditionedTransMorph(
        trans_cfg, num_modalities=3, id_embed_dim=8, image_channels=1,
    )
    registration.initialize_from_transcut(stage1_state)
    stage2_path = tmp_path / "conditioned_stage2.pth"
    torch.save({
        "registration_model": registration.state_dict(),
        "model_config": {
            "registration_model": "conditioned_transmorph",
            "flow_parameterization": "displacement",
        },
    }, stage2_path)

    model = build_stage3_from_checkpoints(
        str(encoder_path), str(stage2_path), img_size=(32, 32),
        device="cpu",
    )
    assert isinstance(model, ConditionedStage3Inference)
    moving = torch.rand(1, 1, 32, 32)
    fixed = torch.rand_like(moving)
    warped, flow = model(
        moving, fixed, torch.tensor([0]), torch.tensor([2]),
    )
    assert warped.shape == moving.shape
    assert flow.shape == (1, 2, 32, 32)
