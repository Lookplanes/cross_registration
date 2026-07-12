# Cross Registration - Internal Notes

This file is for internal development coordination and agent context.
Keep it short, factual, and up to date.

## Project Goals
- Cross-modality medical image registration.
- Pipeline: modality translation (M -> N) then registration in same modality.
- Hub-spoke architecture: M modalities → N center modalities (N ≪ M), route through centers.
- Current focus: 2D single-channel PNG; keep interfaces open for multi-channel and 3D.
- No C/CUDA or acceleration work for now.
- **Data inventory**: see [DATA_INVENTORY.md](DATA_INVENTORY.md)

## 6 Center Modalities (all processed ✅)

| # | Modality | Images | Source | Status |
|---|----------|:---:|------|:---:|
| 1 | 2PM | 232 | ImmuneMAP Movie7_EP | ✅ |
| 2 | Confocal | 360 | IDR0056 (.flex) | ✅ |
| 3 | Fluorescence | 1715 | DeepLIIF (.zip) | ✅ |
| 4 | H&E | 600 | HistoPlexer-Ultivue (.ndpi) | ✅ |
| 5 | MACSima | 24 | S-BIAD1116 (.tif) | ✅ |
| 6 | MSI | 83 | MSIFlow (imzML) | ✅ (外部处理) |

## Current Directory Structure

```
cross_registration/
├── README-self.md
├── DATA_INVENTORY.md          # ✅ 数据集资产清单
├── requirements.txt           # ✅ pip dependencies
├── configs/
│   ├── analysis/
│   │   ├── modality_sources.yaml  # ✅ 6 模态数据源配置
│   │   └── idr_channels.yaml      # ✅ IDR 通道映射
│   ├── pipelines/              # TODO: empty
│   ├── registration/           # TODO: empty
│   └── translation/            # TODO: empty
├── scripts/                    # ✅ 4 CLI scripts
│   ├── extract_features.py         # ✅ 手工特征 → CSV
│   ├── extract_resnet_features.py  # ✅ ResNet18 特征 → CSV (新增)
│   └── analyze_domain_gap.py       # ✅ CSV → distance/tsne/pca/radar 图
├── src/crossreg/               # 核心配准包
│   ├── __init__.py
│   ├── data/                   # ✅ migrated
│   │   ├── data_utils.py, datasets.py, rand.py, transforms.py
│   │   └── perturbation.py     # ✅ SynthMorph-style appearance & deformation
│   ├── pipeline/
│   │   └── inference.py        # ✅ CUT → TransMorph 端到端
│   ├── registration/
│   │   └── transmorph/         # ✅ SwinTransformer + 解码器 + 损失函数
│   ├── translation/
│   │   └── cut/                # ✅ ResnetGenerator + PatchNCE
│   ├── config/                 # TODO: empty
│   └── utils/                  # TODO: empty
│
├── src/modality_analyzer/      # ✅ 特征分析工具包
│   ├── __init__.py             # ✅ CORE_FEATURES (19维) + FEAT_LABELS
│   ├── features/               # ✅ 5类: intensity/histogram/texture/gradient/frequency
│   └── visualize/              # ✅ 4图: domain_gap/pca/tsne/radar
│
└── results/                    # ✅ 分析输出
    ├── features.csv            # 手工特征 907行×19维
    ├── resnet_features.csv     # ResNet特征 907行×512维
    ├── figures/                # 手工特征: distance + tsne + radar
    ├── figures_resnet/         # ResNet: distance + tsne
    └── figures_resnet-ch{0-3}/ # 精选子通道 (含MSI)
```

## ✅ Completed
| Module | Location | Notes |
|--------|----------|-------|
| TransMorph model | src/crossreg/registration/transmorph/model.py | SwinTransformer + decoder |
| CUT model | src/crossreg/translation/cut/cut_model.py | CUTWrapper + CUTInference |
| Pipeline inference | src/crossreg/pipeline/inference.py | CUT → TransMorph end-to-end |
| Datasets / transforms / data utils | src/crossreg/data/ | PairedImageFolder, transforms, pkload |
| Data perturbation | src/crossreg/data/perturbation.py | SynthMorph-style: appearance, diffeomorphic/FFD flow, test-suite builder |
| Modality features | src/modality_analyzer/features/ | 5 categories, 19-dim handcrafted |
| Modality visualize | src/modality_analyzer/visualize/ | domain_gap (Z-score), PCA, t-SNE, radar |
| Feature extraction | scripts/extract_features.py | Scan data → handcrafted CSV |
| ResNet extraction | scripts/extract_resnet_features.py | ResNet18 → 512-dim CSV (新增) |
| Domain gap analysis | scripts/analyze_domain_gap.py | Auto-detect feature type, selectable plots |
| 6-modality data | /data2/wuyh/ | All 6 center modalities processed |
| 2PM sub-cluster analysis | results/2pm_subclusters.png | 4 channels by t-SNE |
| Confocal sub-cluster analysis | results/confocal_subclusters.png | 4 channels by t-SNE |
| Channel selection analysis | results/figures_resnet-ch{0-3}/ | 2PM each channel vs Conf-Tubulin + others |

## 🚧 TODO
| Priority | Task | Where |
|----------|------|-------|
| 🔴 High | Define config schema + loader | src/crossreg/config/ |
| 🔴 High | Create minimal default config YAMLs | configs/{registration,translation,pipelines}/ |
| 🔴 High | Train/infer scripts (registration) | scripts/train_registration.py |
| 🟡 Med | Train/infer scripts (translation) | scripts/train_translation.py |
| 🟡 Med | End-to-end pipeline script | scripts/run_pipeline.py |
| 🟡 Med | Path resolution helper | src/crossreg/utils/paths.py |
| 🟡 Med | Evaluation script | scripts/evaluate.py |
| 🟢 Low | Migrate affine registration | src/crossreg/registration/transmorph/ |
| 🟢 Low | Unit tests | tests/ |

## Environment
- Conda env: `crossreg` (Python 3.10)
- Location: `/data2/xujr/conda-envs/crossreg`
- Activate: `source /data2/xujr/miniconda3/etc/profile.d/conda.sh && conda activate crossreg`
- Install: `pip install -r requirements.txt`
- Analysis deps: `pip install seaborn scikit-learn openslide-python openslide-bin h5py`
- CUDA: unavailable (driver too old); torch 2.12.0+cpu
- Run tests: `python3 src/crossreg/translation/cut/test.py`

## Conventions
- Training and inference are separate entrypoints.
- Keep translation and registration modules decoupled.
- Prefer registry/adapter patterns to swap models.
- Default image format: PNG. Avoid format-specific logic outside data/io.
- Leave hooks for 3D volumes and multi-channel images.
- Data paths configured via `configs/analysis/modality_sources.yaml`.

