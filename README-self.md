# Cross Registration - Internal Notes

This file is for internal development coordination and agent context.
Keep it short, factual, and up to date.

## Project Goals
- Cross-modality medical image registration.
- Pipeline: modality translation (M -> N) then registration in same modality.
- Current focus: 2D single-channel PNG; keep interfaces open for multi-channel and 3D.
- No C/CUDA or acceleration work for now.

## Current Directory Structure

```
cross_registration/
├── README-self.md
├── requirements.txt            # ✅ pip dependencies
├── configs/
│   ├── analysis/
│   │   └── idr_channels.yaml   # ✅ IDR channel mapping for feature extraction
│   ├── pipelines/              # TODO: empty
│   ├── registration/           # TODO: empty, need TransMorph config YAMLs
│   └── translation/            # TODO: empty, need CUT config YAMLs
├── scripts/                    # ✅ 2 CLI scripts done
│   ├── extract_features.py     # ✅ Scan TIFF datasets → features CSV
│   └── analyze_domain_gap.py   # ✅ CSV → domain gap / PCA / radar figures
├── src/crossreg/
│   ├── __init__.py
│   ├── config/                 # TODO: empty, need config schema/loader
│   ├── data/                   # ✅ migrated
│   │   ├── __init__.py
│   │   ├── data_utils.py       # ✅ migrated (pkload, etc.)
│   │   ├── datasets.py         # ✅ migrated (PairedImageFolder, MultiModalityPaired, etc.)
│   │   ├── rand.py             # ✅ migrated (Uniform, Gaussian, Constant)
│   │   └── transforms.py       # ✅ migrated (Flip, Rotate, Crop, Compose, etc.)
│   ├── pipeline/               # ✅ core pipeline
│   │   ├── __init__.py
│   │   └── inference.py        # ✅ PipelineInference: CUT -> TransMorph end-to-end
│   ├── registration/
│   │   ├── __init__.py
│   │   └── transmorph/         # ✅ migrated
│   │       ├── __init__.py
│   │       ├── config.py       # ✅ 2D TransMorph config variants
│   │       ├── losses.py       # ✅ NCC, SSIM, Grad, MIND, MI, etc.
│   │       ├── model.py        # ✅ TransMorph + SwinTransformer + decoder
│   │       └── utils.py        # ✅ SpatialTransformer, metrics, register_model
│   ├── translation/
│   │   ├── __init__.py
│   │   └── cut/                # ✅ migrated (stripped from official CUT)
│   │       ├── __init__.py
│   │       ├── networks.py     # ✅ ResnetGenerator, NLayerDiscriminator, PatchSampleF, GANLoss
│   │       ├── patchnce.py     # ✅ PatchNCELoss (decoupled from argparse)
│   │       ├── cut_model.py    # ✅ CUTWrapper (train) + CUTInference (gen-only inference)
│   │       └── test.py         # ✅ smoke test (random weights, all passes)
│   └── utils/                  # TODO: empty, need paths.py, io helpers
│
├── src/modality_analyzer/      # ✅ independent sub-package
│   ├── requirements.txt        # ✅ standalone deps (numpy, scipy, sklearn, ...)
│   ├── __init__.py             # ✅ CORE_FEATURES, FEAT_LABELS (single source of truth)
│   ├── features/
│   │   ├── __init__.py         # ✅ extract_all_features() aggregator
│   │   ├── intensity.py        # ✅ mean/std/percentiles/skew/kurtosis/SNR
│   │   ├── histogram.py        # ✅ entropy, peak, active bin count
│   │   ├── texture.py          # ✅ GLCM contrast/homogeneity/energy/correlation
│   │   ├── gradient.py         # ✅ Sobel gradient stats + edge density
│   │   └── frequency.py        # ✅ FFT radial band energies + low/high ratio
│   └── visualize/
│       ├── __init__.py
│       ├── domain_gap.py       # ✅ Z-score heatmap (Hub vs Source)
│       ├── pca.py              # ✅ PCA scatter + top loading bar chart
│       └── radar.py            # ✅ per-study radar chart
│
└── tests/                      # TODO: empty, no tests
```

## ✅ Completed
| Module | Location | Notes |
|--------|----------|-------|
| TransMorph model | src/crossreg/registration/transmorph/model.py | SwinTransformer + decoder + deformable head; `get_affine_net` not migrated |
| TransMorph configs | src/crossreg/registration/transmorph/config.py | 11 config variants (2D only) |
| Loss functions | src/crossreg/registration/transmorph/losses.py | NCC, SSIM, Grad, MIND, MI, PCC, etc. |
| Registration utils | src/crossreg/registration/transmorph/utils.py | SpatialTransformer (device-adaptive), register_model, metrics |
| Datasets | src/crossreg/data/datasets.py | PairedImageFolder, MultiModalityPaired, SingleModalityPaired, RaFD |
| Transforms | src/crossreg/data/transforms.py | Flip, Rotate, Crop, Compose, etc. |
| Data utils | src/crossreg/data/data_utils.py | pkload, etc. |
| Rand utils | src/crossreg/data/rand.py | Uniform, Gaussian, Constant samplers |
| CUT networks | src/crossreg/translation/cut/networks.py | ResnetGenerator, NLayerDiscriminator, PatchSampleF, GANLoss (stylegan removed) |
| PatchNCE loss | src/crossreg/translation/cut/patchnce.py | Decoupled from argparse, clean config-driven API |
| CUT model | src/crossreg/translation/cut/cut_model.py | CUTWrapper (train) + CUTInference (generator-only inference) |
| Smoke test | src/crossreg/translation/cut/test.py | 4 tests: inference, training, factory funcs, save/load — all pass |
| Pipeline inference | src/crossreg/pipeline/inference.py | End-to-end: CUT translate -> TransMorph register; single + batch OK |
| Modality features | src/modality_analyzer/features/*.py | 5 categories: intensity, histogram, texture (GLCM), gradient, frequency |
| Modality visualize | src/modality_analyzer/visualize/*.py | 3 charts: domain gap heatmap, PCA overview, per-study radar |
| Feature extraction | scripts/extract_features.py | CLI: scan TIFF data → features CSV with resume/checkpoint |
| Domain gap analysis | scripts/analyze_domain_gap.py | CLI: features CSV → figures (selectable plots) |

## 🚧 TODO
| Priority | Task | Where |
|----------|------|-------|
| 🔴 High | Define config schema + loader | src/crossreg/config/ |
| 🔴 High | Create minimal default config YAMLs | configs/{registration,translation,pipelines}/ |
| 🔴 High | Train/infer scripts (registration) | scripts/train_registration.py, scripts/infer_registration.py |
| 🟡 Med | Train/infer scripts (translation) | scripts/train_translation.py, scripts/infer_translation.py |
| 🟡 Med | End-to-end pipeline script | scripts/run_pipeline.py |
| 🟡 Med | Path resolution helper | src/crossreg/utils/paths.py |
| 🟡 Med | Evaluation script | scripts/evaluate.py |
| 🟡 Med | Reuse Microsolve dataset layout (trainA/B) | scripts/prepare_dataset.py |
| 🟢 Low | Migrate affine registration (TransMorphAffine2D) | src/crossreg/registration/transmorph/ |
| 🟢 Low | Unit tests | tests/ |
| 🟢 Low | Dataset scanner | scripts/scan_dataset.py |

## Data and Model Storage
- Data and model weights are NOT stored on this disk.
- All paths should be provided via config or environment variables.
- Use a single helper (planned in src/crossreg/utils/paths.py) to resolve paths.

## Conventions
- Training and inference are separate entrypoints.
- Keep translation and registration modules decoupled.
- Prefer registry/adapter patterns to swap models.
- Default image format: PNG. Avoid format-specific logic outside data/io.
- Leave hooks for 3D volumes and multi-channel images.

## Environment
- Conda env: `crossreg` (Python 3.10)
- Activate: `source /data/xujr/miniconda3/etc/profile.d/conda.sh && conda activate crossreg`
- Install: `pip install -r requirements.txt`
- CUDA: driver 12010 too old for torch 2.12+cu130; runs on CPU. Install CPU torch if GPU not needed: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Run tests: `python3 src/crossreg/translation/cut/test.py`

