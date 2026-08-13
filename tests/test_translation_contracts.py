"""Training and checkpoint contracts for standalone translation models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_paired_translation_uses_shared_geometry(tmp_path: Path) -> None:
    from crossreg.data.translation import TwoDomainTranslationDataset
    dir_a, dir_b = tmp_path / "A", tmp_path / "B"
    dir_a.mkdir()
    dir_b.mkdir()
    image = np.arange(80 * 80, dtype=np.uint16).reshape(80, 80)
    image = (image % 256).astype(np.uint8)
    Image.fromarray(image).save(dir_a / "same.png")
    Image.fromarray(image).save(dir_b / "same.png")

    torch.manual_seed(7)
    sample = TwoDomainTranslationDataset(
        str(dir_a), str(dir_b), input_nc=1, output_nc=1,
        load_size=72, crop_size=64, pairing_mode="paired",
    )[0]
    assert torch.equal(sample["A"], sample["B"])


def test_unpaired_translation_does_not_require_matching_stems(tmp_path: Path) -> None:
    from crossreg.data.translation import TwoDomainTranslationDataset

    dir_a, dir_b = tmp_path / "A", tmp_path / "B"
    dir_a.mkdir()
    dir_b.mkdir()
    for index in range(2):
        Image.fromarray(np.full((32, 32), index, dtype=np.uint8)).save(dir_a / f"a_{index}.png")
    for index in range(3):
        Image.fromarray(np.full((32, 32), index, dtype=np.uint8)).save(dir_b / f"b_{index}.png")
    dataset = TwoDomainTranslationDataset(
        str(dir_a), str(dir_b), load_size=32, crop_size=32,
        pairing_mode="unpaired",
    )
    assert len(dataset) == 3
    assert dataset[0]["A_paths"].endswith("a_0.png")


def test_multidomain_sparse_anchor_uses_aligned_shared_geometry(
    tmp_path: Path,
) -> None:
    from crossreg.data.translation import MultiDomainTranslationDataset

    domains = [tmp_path / name for name in ("A", "B")]
    for directory in domains:
        directory.mkdir()
        Image.fromarray(np.zeros((80, 80), dtype=np.uint8)).save(
            directory / f"unpaired_{directory.name}.png"
        )
        aligned = (np.arange(80 * 80).reshape(80, 80) % 256).astype(np.uint8)
        Image.fromarray(aligned).save(directory / "anchor.png")

    dataset = MultiDomainTranslationDataset(
        [str(directory) for directory in domains],
        load_size=72, crop_size=64, pairing_mode="unpaired",
        modality_files=[
            [directory / f"unpaired_{directory.name}.png"]
            for directory in domains
        ],
        paired_anchor_files=[
            [directory / "anchor.png"] for directory in domains
        ],
        paired_anchor_probability=1.0,
    )
    sample = dataset[0]
    assert sample["is_paired"] is True
    assert Path(sample["A_paths"]).stem == "anchor"
    assert Path(sample["B_paths"]).stem == "anchor"
    assert torch.equal(sample["A"], sample["B"])


def test_cut_inference_loads_complete_training_checkpoint(tmp_path: Path) -> None:
    from crossreg.translation.cut import CUTConfig, CUTInference, CUTWrapper

    config = CUTConfig(
        input_nc=1, output_nc=1, ngf=8, ndf=8,
        nce_layers=[0, 4], netF_nc=16, gpu_ids=[],
    )
    trained = CUTWrapper(config)
    path = tmp_path / "cut_training.pth"
    torch.save(trained.training_state(), path)

    inference = CUTInference(input_nc=1, output_nc=1, ngf=8, gpu_ids=[])
    inference.load_weights(str(path))
    expected = trained.netG.state_dict()
    actual = inference.netG.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


def test_transcut_supports_batched_modality_ids_and_trains_style() -> None:
    from crossreg.translation.transcut import TransCUT, TransCUTConfig

    config = TransCUTConfig(
        num_modalities=3, img_size=64, input_nc=1, output_nc=1,
        embed_dim=32, depths=(1, 1, 1, 1), num_heads=(4, 4, 8, 8),
        window_size=(4, 4), drop_path_rate=0.0, ndf=8,
        netF_nc=16, num_patches=8, gpu_ids=[],
        lambda_identity=0.7, lambda_paired=0.5, lambda_cycle=0.5,
        lambda_structure=1.0, lambda_D_mismatch=1.0,
    )
    model = TransCUT(config)
    source = torch.randn(2, 1, 64, 64)
    src_ids = torch.tensor([0, 1])
    tgt_ids = torch.tensor([1, 2])
    model.eval()
    with torch.no_grad():
        output = model(source, src_ids, tgt_ids)
    assert output.shape[0] == 2
    assert output.shape[1] == 1

    optimized = {id(p) for group in model.optimizer_G.param_groups for p in group["params"]}
    assert all(id(p) in optimized for p in model.style_embed.parameters())

    model.train()
    before = [p.detach().clone() for p in model.style_embed.parameters()]
    losses = model.optimize_parameters(
        source, torch.randn_like(source), src_ids, tgt_ids,
    )
    assert {"identity", "paired", "cycle", "structure", "D_mismatch"} <= set(losses)
    assert losses["paired"] > 0.0
    assert all(np.isfinite(value) for value in losses.values())
    diagnostics = model.batch_diagnostics()
    assert set(diagnostics) == {
        "G_GAN", "NCE", "identity", "paired", "paired_anchor", "cycle",
        "D", "D_real", "D_fake", "D_mismatch",
        "fake_mean", "fake_std", "fake_dark_fraction",
        "real_mean", "real_std", "real_dark_fraction",
    }
    assert all(value.shape == (2,) for value in diagnostics.values())
    assert torch.equal(diagnostics["paired_anchor"], torch.ones(2))
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    assert any(not torch.equal(old, new) for old, new in zip(before, model.style_embed.parameters()))


def test_highres_decoder_is_opt_in_and_uses_source_detail() -> None:
    from crossreg.translation.transcut import (
        HighResolutionContentDecoder,
        TransCUT,
        TransCUTConfig,
    )
    from crossreg.translation.transcut.decoder import TransCUTDecoder

    common = dict(
        num_modalities=2, img_size=32, input_nc=1, output_nc=1,
        embed_dim=16, depths=(1, 1), num_heads=(4, 4),
        window_size=(4, 4), out_indices=(0, 1), drop_path_rate=0.0,
        ndf=8, netF_nc=8, num_patches=4, gpu_ids=[],
    )
    legacy = TransCUT(TransCUTConfig(**common))
    assert type(legacy.decoder) is TransCUTDecoder

    model = TransCUT(TransCUTConfig(
        decoder_variant="highres_content", **common,
    ))
    assert isinstance(model.decoder, HighResolutionContentDecoder)

    source = torch.randn(2, 1, 32, 32)
    output = model(source, torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert output.shape == source.shape
    output.mean().backward()
    detail_parameters = list(model.decoder.detail_stem.parameters())
    assert detail_parameters
    assert all(parameter.grad is not None for parameter in detail_parameters)

    features = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
    ]
    style = torch.randn(1, 64)
    zeros = torch.zeros(1, 1, 32, 32)
    pattern = zeros.clone()
    pattern[..., ::2, ::2] = 1.0
    with torch.no_grad():
        flat_output = model.decoder(features, style, source=zeros)
        detailed_output = model.decoder(features, style, source=pattern)
    assert not torch.allclose(flat_output, detailed_output)


def test_fullres_residual_decoder_starts_as_exact_source_carrier() -> None:
    from crossreg.translation.transcut import (
        FullResolutionResidualDecoder,
        TransCUT,
        TransCUTConfig,
    )

    config = TransCUTConfig(
        num_modalities=2, img_size=32, input_nc=1, output_nc=1,
        embed_dim=16, depths=(1, 1), num_heads=(4, 4),
        window_size=(4, 4), out_indices=(0, 1), drop_path_rate=0.0,
        ndf=8, netF_nc=8, num_patches=4, gpu_ids=[],
        decoder_variant="fullres_residual",
    )
    model = TransCUT(config)
    assert isinstance(model.decoder, FullResolutionResidualDecoder)

    source = torch.empty(2, 1, 32, 32).uniform_(-0.9, 0.9)
    output = model(source, torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert output.shape == source.shape
    assert torch.equal(output, source)

    output.mean().backward()
    assert model.decoder.residual_head.weight.grad is not None


def test_transcut_rejects_decoder_variant_checkpoint_mismatch() -> None:
    from crossreg.translation.transcut import TransCUT, TransCUTConfig

    common = dict(
        num_modalities=2, img_size=32, input_nc=1, output_nc=1,
        embed_dim=16, depths=(1, 1), num_heads=(4, 4),
        window_size=(4, 4), out_indices=(0, 1), drop_path_rate=0.0,
        ndf=8, netF_nc=8, num_patches=4, gpu_ids=[],
    )
    legacy = TransCUT(TransCUTConfig(**common))
    state = legacy.training_state()
    highres = TransCUT(TransCUTConfig(
        decoder_variant="highres_content", **common,
    ))
    with pytest.raises(RuntimeError, match="decoder variant"):
        highres.load_training_state(state, load_optimizers=False)


def test_modality_registry_has_stable_contiguous_ids(tmp_path: Path) -> None:
    from crossreg.data.modalities import load_modality_registry

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    registry = tmp_path / "modalities.yaml"
    registry.write_text(yaml.safe_dump({"modalities": [
        {"id": 1, "name": "second", "path": "second", "channels": 1},
        {"id": 0, "name": "first", "path": "first", "channels": 1},
    ]}), encoding="utf-8")
    specs = load_modality_registry(registry)
    assert [spec.id for spec in specs] == [0, 1]
    assert [spec.name for spec in specs] == ["first", "second"]
    assert Path(specs[0].path) == first.resolve()

    registry.write_text(yaml.safe_dump({"modalities": [
        {"id": 0, "name": "first", "path": "first"},
        {"id": 2, "name": "second", "path": "second"},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        load_modality_registry(registry)


def test_transcut_expands_embedding_by_stable_modality_name() -> None:
    from crossreg.translation.transcut import TransCUT, TransCUTConfig

    common = dict(
        img_size=32, input_nc=1, output_nc=1, embed_dim=16,
        depths=(1, 1), num_heads=(4, 4), window_size=(4, 4),
        out_indices=(0, 1), drop_path_rate=0.0, ndf=8,
        netF_nc=8, num_patches=4, gpu_ids=[],
    )
    old = TransCUT(TransCUTConfig(num_modalities=2, **common))
    old.set_modality_names(["two_photon", "confocal"])
    with torch.no_grad():
        old.mod_embed.embedding.weight[0].fill_(1.25)
        old.mod_embed.embedding.weight[1].fill_(2.5)
    state = old.training_state()

    new = TransCUT(TransCUTConfig(num_modalities=3, **common))
    new.set_modality_names(["confocal", "brightfield_he", "two_photon"])
    new_row = new.mod_embed.embedding.weight[1].detach().clone()
    changed = new.load_training_state(
        state, load_optimizers=False, allow_modality_expansion=True,
    )
    assert changed is True
    assert torch.all(new.mod_embed.embedding.weight[0] == 2.5)
    assert torch.equal(new.mod_embed.embedding.weight[1], new_row)
    assert torch.all(new.mod_embed.embedding.weight[2] == 1.25)


def test_transcut_lr_stays_constant_before_decay() -> None:
    from crossreg.translation.transcut import TransCUT, TransCUTConfig

    config = TransCUTConfig(
        num_modalities=2, img_size=32, input_nc=1, output_nc=1,
        embed_dim=16, depths=(1, 1), num_heads=(4, 4),
        window_size=(4, 4), out_indices=(0, 1), drop_path_rate=0.0,
        ndf=8, netF_nc=8, num_patches=4, gpu_ids=[],
        lr=2e-4, lr_D=1e-4, n_epochs=2, n_epochs_decay=2,
    )
    model = TransCUT(config)
    model.update_learning_rate(1)
    assert model.optimizer_G.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert model.optimizer_D.param_groups[0]["lr"] == pytest.approx(1e-4)
    model.update_learning_rate(2)
    assert model.optimizer_G.param_groups[0]["lr"] == pytest.approx(2e-4)
    model.update_learning_rate(3)
    assert model.optimizer_G.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert model.optimizer_D.param_groups[0]["lr"] == pytest.approx(5e-5)


def test_conditional_discriminator_uses_target_modality() -> None:
    from crossreg.translation.transcut.conditional_discriminator import (
        ConditionalPatchDiscriminator,
    )

    discriminator = ConditionalPatchDiscriminator(1, num_modalities=2, ndf=8)
    image = torch.randn(2, 1, 64, 64)
    ids = torch.tensor([0, 1])
    output = discriminator(image[:1].expand(2, -1, -1, -1), ids)
    assert output.shape[0] == 2
    assert not torch.equal(output[0], output[1])


def test_transcut_structure_loss_preserves_edges_across_contrast_reversal() -> None:
    from crossreg.translation.transcut import TransCUT

    source = torch.zeros(1, 1, 32, 32)
    source[:, :, 8:24, 8:24] = 1.0
    contrast_reversed = 1.0 - source
    shifted = torch.roll(source, shifts=3, dims=-1)

    same_edges = TransCUT.multiscale_gradient_loss(contrast_reversed, source)
    shifted_edges = TransCUT.multiscale_gradient_loss(shifted, source)
    assert same_edges.item() < 1e-7
    assert shifted_edges.item() > 0.01


def test_transcut_loss_weights_are_independent() -> None:
    from crossreg.translation.transcut import TransCUTConfig

    config = TransCUTConfig(
        lambda_GAN=0.5, lambda_NCE=2.0,
        lambda_identity=1.0, lambda_cycle=0.75,
        lambda_structure=1.5, lambda_D_mismatch=0.25,
    )
    assert config.lambda_GAN == 0.5
    assert config.lambda_NCE == 2.0
    assert config.lambda_identity == 1.0
    assert config.lambda_cycle == 0.75
    assert config.lambda_structure == 1.5
    assert config.lambda_D_mismatch == 0.25
