#!/usr/bin/env python3
"""Smoke-test all cross_registration modules with random-initialised weights.

Usage (crossreg conda env)::

    source /data2/xujr/miniconda3/etc/profile.d/conda.sh
    conda activate crossreg
    python tests/test_all_modules.py

All models use random initialisation; no real data or GPU required.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure src/ is on PYTHONPATH for local module imports
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_VERBOSE = True


def _ok(msg: str) -> None:
    if _VERBOSE:
        print(f"  ✓ {msg}")


def _section(title: str) -> None:
    if _VERBOSE:
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _make_random_image(
    path: str,
    size: tuple[int, int] = (64, 64),
    mode: str = "L",
) -> None:
    """Write a random uint8 image to *path*."""
    if mode in ("RGB", "RGBA"):
        channels = 3 if mode == "RGB" else 4
        arr = np.random.randint(0, 256, (*size, channels), dtype=np.uint8)
    else:
        arr = np.random.randint(0, 256, size, dtype=np.uint8)
    Image.fromarray(arr, mode=mode).save(path)


def _temp_dataset(prefix: str = "test_ds") -> str:
    """Create a temporary paired dataset directory with 3 image pairs."""
    d = tempfile.mkdtemp(prefix=prefix)
    for sub in ("moving", "fixed", "valid_mask", "gt_flow"):
        os.makedirs(os.path.join(d, sub))
    for i in range(3):
        fname = f"{i:04d}.png"
        _make_random_image(os.path.join(d, "moving", fname), (64, 64))
        _make_random_image(os.path.join(d, "fixed", fname), (64, 64))
        # Mask
        Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 255).save(
            os.path.join(d, "valid_mask", fname)
        )
        # Flow
        np.save(os.path.join(d, "gt_flow", f"{i:04d}.npy"),
                np.random.randn(2, 64, 64).astype(np.float32))
    return d


# ======================================================================
# 1. CUT Translation
# ======================================================================
def test_cut_inference() -> None:
    _section("1. CUTInference (generator-only forward)")

    from crossreg.translation.cut import CUTInference

    # Grayscale (1-channel)
    model = CUTInference(input_nc=1, output_nc=1, gpu_ids=[])
    img = torch.randn(1, 256, 256)
    out = model.translate(img)
    assert out.shape == img.shape, f"shape {out.shape} != {img.shape}"
    _ok(f"Grayscale: {tuple(img.shape)} -> {tuple(out.shape)}")

    # RGB (3-channel)
    model_rgb = CUTInference(input_nc=3, output_nc=3, gpu_ids=[])
    batch = torch.randn(4, 3, 128, 128)
    out = model_rgb.translate(batch)
    assert out.shape == batch.shape, f"shape {out.shape} != {batch.shape}"
    _ok(f"RGB batch: {tuple(batch.shape)} -> {tuple(out.shape)}")

    # encode_only
    feats = model.netG(img.unsqueeze(0), layers=[0, 4, 8], encode_only=True)
    assert len(feats) == 3, f"expected 3 feats, got {len(feats)}"
    _ok(f"encode_only: {len(feats)} feature maps")


def test_cut_wrapper() -> None:
    _section("2. CUTWrapper (full training step)")

    from crossreg.translation.cut import CUTConfig, CUTWrapper

    config = CUTConfig(
        input_nc=1, output_nc=1,
        netG="resnet_9blocks", netD="basic", netF="mlp_sample",
        ngf=16, ndf=16, netF_nc=64,        # tiny model
        nce_layers=[0, 4],
        lambda_GAN=1.0, lambda_NCE=1.0,
        nce_idt=True,
        gpu_ids=[0] if torch.cuda.is_available() else [],
    )
    model = CUTWrapper(config)
    _ok(f"CUTWrapper created (G ngf={config.ngf}, D ndf={config.ndf})")

    # Single training iteration with random data
    real_A = torch.randn(1, 1, 128, 128)
    real_B = torch.randn(1, 1, 128, 128)
    data = {"A": real_A, "B": real_B, "A_paths": ["/fake/a.png"]}

    model.set_input(data)
    model.optimize_parameters()
    losses = model.get_current_losses()
    _ok(f"Training step OK, losses: { {k: f'{v:.4f}' for k, v in losses.items()} }")

    # Inference mode
    from crossreg.translation.cut import CUTInference
    infer = CUTInference(input_nc=1, output_nc=1, gpu_ids=[])
    with torch.no_grad():
        out = infer.translate(torch.randn(1, 256, 256))
    assert out.shape == (1, 256, 256)
    _ok(f"CUTInference: {tuple(out.shape)}")


# ======================================================================
# 3. TransMorph Registration
# ======================================================================
def test_transmorph() -> None:
    _section("3. TransMorph (registration forward)")

    from crossreg.registration.transmorph.model import TransMorph, CONFIGS

    for name in ["TransMorph", "TransMorph-Small", "TransMorph-Large"]:
        cfg = CONFIGS[name]
        cfg.in_chans = 2
        cfg.img_size = (128, 128)
        model = TransMorph(cfg).to(_DEVICE)

        # Input: [fixed, moving] concatenated
        x = torch.randn(1, 2, 128, 128).to(_DEVICE)
        warped, flow, pos_flow = model(x)
        assert warped.shape == (1, 1, 128, 128), f"warped {warped.shape}"
        assert flow.shape == (1, 2, 128, 128), f"flow {flow.shape}"
        _ok(f"{name}: warped {tuple(warped.shape)}, flow {tuple(flow.shape)}")

    # Batch test
    cfg = CONFIGS["TransMorph"]
    cfg.in_chans = 2
    model = TransMorph(cfg).to(_DEVICE)
    x = torch.randn(2, 2, 128, 128).to(_DEVICE)
    warped, flow, pos_flow = model(x)
    assert warped.shape == (2, 1, 128, 128)
    _ok(f"Batch-2: warped {tuple(warped.shape)}")


# ======================================================================
# 4. Pipeline (CUT + TransMorph)
# ======================================================================
def test_pipeline() -> None:
    _section("4. PipelineInference (CUT -> TransMorph)")

    from crossreg.pipeline.inference import build_pipeline

    pipeline = build_pipeline(
        cut_input_nc=1, cut_output_nc=1,
        cut_ngf=16,
        transmorph_img_size=(128, 128),
        device=str(_DEVICE),
    )
    source = torch.randn(1, 1, 128, 128).to(_DEVICE)
    target = torch.randn(1, 1, 128, 128).to(_DEVICE)

    result = pipeline.infer(source, target)
    assert result.translated.shape == (1, 1, 128, 128)
    assert result.warped.shape == (1, 1, 128, 128)
    assert result.flow.shape == (1, 2, 128, 128)
    _ok(f"translated {tuple(result.translated.shape)}, "
        f"warped {tuple(result.warped.shape)}, "
        f"flow {tuple(result.flow.shape)}")


# ======================================================================
# 5. Data Perturbation
# ======================================================================
def test_perturbation() -> None:
    _section("5. Data Perturbation Utilities")

    from crossreg.data.perturbation import (
        generate_diffeomorphic_flow,
        generate_ffd_like_flow,
        apply_synthmorph_appearance,
        set_random_zero_borders,
        center_crop_pair_and_flow,
        sample_param_value,
    )

    shape = (128, 128)

    # --- Flow generation ---
    flow, map_x, map_y = generate_diffeomorphic_flow(shape, smooth_sigma=12.0)
    assert flow.shape == (2, 128, 128), f"flow {flow.shape}"
    assert map_x.shape == (128, 128)
    assert np.min(flow) < 0 and np.max(flow) > 0, "flow should have non-zero values"
    _ok(f"diffeomorphic_flow: {flow.shape}, range [{flow.min():.1f}, {flow.max():.1f}]")

    flow2, _, _ = generate_ffd_like_flow(shape, grid_resolution=8)
    assert flow2.shape == (2, 128, 128)
    _ok(f"ffd_flow: {flow2.shape}, range [{flow2.min():.1f}, {flow2.max():.1f}]")

    # --- Appearance perturbation ---
    img = np.random.randint(0, 256, shape, dtype=np.uint8)
    pert = apply_synthmorph_appearance(img)
    assert pert.shape == img.shape and pert.dtype == np.uint8
    _ok(f"appearance: shape {pert.shape}, dtype {pert.dtype}")

    # --- Zero borders ---
    bordered = set_random_zero_borders(img, max_border_frac=1 / 8, apply_prob=1.0)
    assert bordered.shape == img.shape
    _ok(f"zero_borders: shape {bordered.shape}")

    # --- Center crop ---
    mask = np.ones(shape, dtype=np.uint8) * 255
    m_c, f_c, mk_c, fl_c = center_crop_pair_and_flow(
        img, pert, mask, flow, keep_fraction=0.75,
    )
    expected_h = int(round(128 * 0.75))
    assert m_c.shape == (expected_h, expected_h)
    assert fl_c.shape == (2, expected_h, expected_h)
    _ok(f"center_crop: {(expected_h, expected_h)}, flow {fl_c.shape}")

    # --- Param sampling ---
    v = sample_param_value(5.0, (1.0, 10.0))
    assert 1.0 <= v <= 10.0, f"value {v} out of range"
    v = sample_param_value(5.0)
    assert v == 5.0
    _ok("sample_param_value OK")

    # --- cv2.remap integration ---
    import cv2
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    assert warped.shape == img.shape
    _ok(f"cv2.remap integration: {warped.shape}")


# ======================================================================
# 6. Datasets
# ======================================================================
def test_datasets() -> None:
    _section("6. Datasets")

    d = _temp_dataset()

    from crossreg.data.datasets import PairedImageFolderDataset

    # PairedImageFolderDataset
    ds = PairedImageFolderDataset(d, img_size=(64, 64))
    assert len(ds) == 3, f"expected 3 pairs, got {len(ds)}"
    sample = ds[0]
    moving, fixed = sample[0], sample[1]
    assert moving.shape == (3, 64, 64), f"moving {moving.shape}"
    assert fixed.shape == (3, 64, 64)
    _ok(f"PairedImageFolderDataset: {len(ds)} pairs, shape {tuple(moving.shape)}")

    # With flow
    ds_flow = PairedImageFolderDataset(d, flow_subdir="gt_flow", img_size=(64, 64))
    sample = ds_flow[0]
    assert len(sample) == 3, f"expected 3 (m, f, flow), got {len(sample)}"
    assert sample[2].shape == (2, 64, 64)
    _ok(f"With flow: {tuple(sample[2].shape)}")

    # With mask + flow
    ds_full = PairedImageFolderDataset(d, mask_subdir="valid_mask", flow_subdir="gt_flow", img_size=(64, 64))
    sample = ds_full[0]
    assert len(sample) == 4, f"expected 4 (m, f, mask, flow), got {len(sample)}"
    _ok(f"With mask+flow: mask {tuple(sample[2].shape)}, flow {tuple(sample[3].shape)}")

    # ------------------------------------------------------------------
    # MultiModalityPairedDataset (requires subfolder layout)
    # ------------------------------------------------------------------
    from crossreg.data.datasets import MultiModalityPairedDataset
    md = tempfile.mkdtemp(prefix="multi_mod_")
    for task in ("ch0_to_ch1", "ch1_to_ch0"):
        for sub in ("moving", "fixed", "gt_flow", "valid_mask"):
            os.makedirs(os.path.join(md, task, sub))
        for i in range(2):
            fname = f"{i:04d}.png"
            _make_random_image(os.path.join(md, task, "moving", fname), (64, 64))
            _make_random_image(os.path.join(md, task, "fixed", fname), (64, 64))
            Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 255).save(
                os.path.join(md, task, "valid_mask", fname))
            np.save(os.path.join(md, task, "gt_flow", f"{i:04d}.npy"),
                    np.random.randn(2, 64, 64).astype(np.float32))
    mds = MultiModalityPairedDataset(md, img_size=(64, 64))
    assert len(mds) >= 2, f"expected >=2 pairs, got {len(mds)}"
    s = mds[0]
    assert len(s) == 4, f"expected 4-tuple, got {len(s)}"
    _ok(f"MultiModalityPairedDataset: {len(mds)} pairs, shapes "
        f"{tuple(t.shape for t in s)}")


# ======================================================================
# 7. Modality Analyzer
# ======================================================================
def test_modality_analyzer() -> None:
    _section("7. Modality Analyzer (features)")

    # --- Handcrafted features ---
    from modality_analyzer.features import extract_all_features

    img_2d = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    feats = extract_all_features(img_2d)
    assert isinstance(feats, dict) and len(feats) > 0
    _ok(f"Handcrafted features: {len(feats)} dims "
        f"(sample: {list(feats.keys())[:3]}...)")

    # All values should be finite
    for k, v in feats.items():
        assert np.isfinite(v), f"Non-finite value in {k}: {v}"
    _ok("All feature values finite")

    # --- CORE_FEATURES subset exists ---
    from modality_analyzer import CORE_FEATURES, FEAT_LABELS
    core_present = [c for c in CORE_FEATURES if c in feats]
    assert len(core_present) >= 10, f"only {len(core_present)} core features found"
    _ok(f"CORE_FEATURES: {len(core_present)}/{len(CORE_FEATURES)} present")

    # --- ResNet extractor ---
    from modality_analyzer.features.resnet import (
        build_resnet_extractor, load_image, FEAT_DIM,
    )
    assert FEAT_DIM == 512

    model = build_resnet_extractor("cpu")
    # Create a temp image for ResNet
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _make_random_image(f.name, (256, 256), "RGB")
        f.flush()
        tensor = load_image(f.name)
        os.unlink(f.name)

    assert tensor.shape == (3, 224, 224), f"ResNet input {tensor.shape}"
    with torch.no_grad():
        vec = model(tensor.unsqueeze(0))
    assert vec.shape == (1, 512), f"ResNet output {vec.shape}"
    _ok(f"ResNet18: {tuple(vec.shape)}")


def test_modality_analyzer_io() -> None:
    _section("8. Modality Analyzer I/O")

    from modality_analyzer.io import (
        discover_images, load_checkpoint, save_checkpoint,
    )

    # Discover images in temp dir
    d = tempfile.mkdtemp(prefix="mod_io_")
    for i in range(5):
        _make_random_image(os.path.join(d, f"img_{i:02d}.png"), (32, 32))
    _make_random_image(os.path.join(d, "extra.jpg"), (32, 32))

    images = discover_images(data_root=os.path.dirname(d),
                             source_dir=os.path.basename(d),
                             label="test_mod")
    assert len(images) == 6, f"expected 6, got {len(images)}"
    assert images[0]["modality_name"] == "test_mod"
    _ok(f"discover_images: {len(images)} found, label='{images[0]['modality_name']}'")

    # Checkpoint
    cp_path = os.path.join(d, "checkpoint.json")
    save_checkpoint(cp_path, {"a", "b", "c"})
    loaded = load_checkpoint(cp_path)
    assert loaded == {"a", "b", "c"}, f"checkpoint mismatch: {loaded}"
    _ok("checkpoint save/load OK")


# ======================================================================
# 8. Config
# ======================================================================
def test_configs() -> None:
    _section("9. Configuration Loaders")

    from crossreg.registration.transmorph.model import CONFIGS
    assert "TransMorph" in CONFIGS
    assert "TransMorph-Small" in CONFIGS
    assert "TransMorph-Large" in CONFIGS
    _ok(f"TransMorph CONFIGS: {len(CONFIGS)} variants loaded")

    from crossreg.translation.cut.cut_model import CUTConfig
    cfg = CUTConfig()
    assert cfg.lr == 2e-4
    assert cfg.lr_policy == "linear"
    assert cfg.lr_decay_iters == 50
    _ok(f"CUTConfig defaults: lr={cfg.lr}, policy={cfg.lr_policy}")


# ======================================================================
# Main
# ======================================================================
def main() -> int:
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Device: {_DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy:   {np.__version__}")

    tests = [
        ("CUTInference",              test_cut_inference),
        ("CUTWrapper",                test_cut_wrapper),
        ("TransMorph",                test_transmorph),
        ("PipelineInference",         test_pipeline),
        ("Data Perturbation",         test_perturbation),
        ("Datasets",                  test_datasets),
        ("Modality Analyzer (feats)", test_modality_analyzer),
        ("Modality Analyzer (I/O)",   test_modality_analyzer_io),
        ("Configs",                   test_configs),
    ]

    failed: list[str] = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            failed.append(name)
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    if failed:
        print(f"✗ {len(failed)}/{len(tests)} tests FAILED: {', '.join(failed)}")
        return 1
    else:
        print(f"✓ All {len(tests)} tests PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
