#!/usr/bin/env python3
"""
特征提取脚本 —— 针对 IDR 多通道医学影像数据集。
从每个 Study 的 Hub（细胞核通道）与 Source（其他模态）图像中提取多维特征，
输出统一 CSV，便于后续配准难度评估和模态差异分析。

特性:
  - 自动识别 Hub/Source 通道映射（基于预定义规则）
  - 多维度特征: 亮度统计、直方图、纹理 (GLCM)、梯度、频域、图像质量
  - 支持断点续跑 (checkpoint)
  - 可配置参数（百分位截断、GLCM 距离/角度、频域带数等）

数据路径约定:
  /data2/xujr/idr_data/test_feature/
      <StudyID>/
          <Screen>/
              channel_<N>/
                  *.tiff

输出: feature_test/features.csv (可配置)
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage, stats, signal
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 全局配置 — 按 Study 定义 Hub 与 Source 通道索引
# ──────────────────────────────────────────────────────────────

STUDY_CHANNEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "idr0003-breker-plasticity": {
        "hub_channels": [2],           # Ch2 = Cherry/H2B
        "hub_label": "H2B",
        "source_channels": {
            0: "GFP",
            1: "Transmitted",
        },
    },
    "idr0037-vigilante-hipsci": {
        "hub_channels": [0],           # Ch0 = DAPI
        "hub_label": "DAPI",
        "source_channels": {
            1: "Alexa488",
            2: "CellMask",
            3: "Brightfield",
        },
    },
    "idr0056-stojic-lncrnas": {
        "hub_channels": [3],           # Ch3 = Hoechst (DNA) — universal hub
        "hub_label": "Hoechst",
        # screenA 与 screenB/C 的 Source 通道物理含义不同
        "source_channels": {
            0: "Tubulin",
            1: "CEP215",
            2: "Phalloidin",
        },
        "screen_configs": {
            "screenB": {0: "alpha-Tubulin", 1: "gamma-Tubulin", 2: "phospho-H3"},
            "screenC": {0: "alpha-Tubulin", 1: "gamma-Tubulin", 2: "phospho-H3"},
        },
    },
    "idr0080": {
        "hub_channels": [0],           # Ch0 = DNA
        "hub_label": "DNA",
        "source_channels": {1: "Ch1", 2: "Ch2", 3: "Ch3", 4: "Ch4"},
    },
    "idr0081": {
        "hub_channels": [0],           # Ch0 = DAPI
        "hub_label": "DAPI",
        "source_channels": {1: "FITC"},
    },
    "idr0086": {
        "hub_channels": [1],           # Ch1 = DAPI
        "hub_label": "DAPI",
        "source_channels": {0: "488"},
    },
    "idr0161": {
        "hub_channels": "auto_nuclei",
        "hub_label": "Nuclei",
        "source_channels": "auto_protein",
    },
    "idr0163": {
        "hub_channels": "auto_nuclei",
        "hub_label": "DAPI/Hoechst",
        "source_channels": "auto_protein",
    },
}


# ──────────────────────────────────────────────────────────────
# 特征提取函数
# ──────────────────────────────────────────────────────────────

def percentile_clip(img: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """百分位截断，避免异常值干扰后续统计。"""
    pl, ph = np.percentile(img, [low, high])
    return np.clip(img, pl, ph)


def extract_intensity_features(img: np.ndarray, clip_percentile: bool = True) -> Dict[str, float]:
    """亮度统计特征。"""
    if clip_percentile:
        img = percentile_clip(img, 1.0, 99.0)
    feats = {}
    feats["int_mean"] = float(np.mean(img))
    feats["int_std"] = float(np.std(img))
    feats["int_min"] = float(np.min(img))
    feats["int_max"] = float(np.max(img))
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        feats[f"int_p{p:02d}"] = float(np.percentile(img, p))
    feats["int_skewness"] = float(stats.skew(img.ravel()))
    feats["int_kurtosis"] = float(stats.kurtosis(img.ravel()))
    feats["int_dynamic_range"] = feats["int_p99"] - feats["int_p01"]
    # SNR 估计 (均值 / 标准差)
    feats["int_snr"] = feats["int_mean"] / (feats["int_std"] + 1e-8)
    return feats


def extract_histogram_features(img: np.ndarray, bins: int = 256) -> Dict[str, float]:
    """直方图形状特征。"""
    hist, _ = np.histogram(img.ravel(), bins=bins, density=True)
    hist = hist / (hist.sum() + 1e-12)
    feats = {}
    feats["hist_entropy"] = float(-np.sum(hist * np.log(hist + 1e-12)))
    # 峰度（直方图最大值的倒数，衡量多峰性）
    feats["hist_max_peak"] = float(np.max(hist))
    # 超过均值 10% 的 bin 数量
    feats["hist_active_bins"] = float(np.sum(hist > 0.1 * np.mean(hist)))
    return feats


def extract_glcm_features(img: np.ndarray, levels: int = 64,
                          distances: List[int] = None,
                          angles: List[float] = None) -> Dict[str, float]:
    """GLCM 纹理特征。"""
    if distances is None:
        distances = [1, 3]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    # 量化到 levels 级
    img_clip = percentile_clip(img, 1.0, 99.0)
    img_q = np.floor((img_clip - img_clip.min()) / (img_clip.max() - img_clip.min() + 1e-8) * (levels - 1)).astype(np.uint8)

    glcm = graycomatrix(img_q, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    feats = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        vals = graycoprops(glcm, prop).ravel()
        feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"glcm_{prop}_std"] = float(np.std(vals))
    return feats


def extract_gradient_features(img: np.ndarray) -> Dict[str, float]:
    """梯度 / 边缘特征。"""
    img_f = img.astype(np.float32)
    grad = sobel(img_f)
    feats = {}
    feats["grad_mean"] = float(np.mean(grad))
    feats["grad_std"] = float(np.std(grad))
    feats["grad_p90"] = float(np.percentile(grad, 90))
    feats["grad_p95"] = float(np.percentile(grad, 95))
    # 边缘密度: 梯度幅值 > 均值 + 1*std 的像素占比
    edge_mask = grad > (feats["grad_mean"] + feats["grad_std"])
    feats["edge_density"] = float(np.mean(edge_mask))
    return feats


def extract_frequency_features(img: np.ndarray, num_bands: int = 6) -> Dict[str, float]:
    """频域能量分布特征。"""
    fft = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
    power = np.abs(fft) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    # 构建径向距离图
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    feats = {}
    for i in range(1, num_bands + 1):
        r_low = max_r * (i - 1) / num_bands
        r_high = max_r * i / num_bands
        mask = (r >= r_low) & (r < r_high)
        band_energy = power[mask].sum()
        feats[f"freq_band_{i:02d}_energy"] = float(band_energy)
    total_energy = power.sum()
    for i in range(1, num_bands + 1):
        feats[f"freq_band_{i:02d}_ratio"] = feats[f"freq_band_{i:02d}_energy"] / (total_energy + 1e-12)
    feats["freq_total_energy"] = float(total_energy)
    # 低频 vs 高频能量比
    low_energy = sum(feats[f"freq_band_{i:02d}_energy"] for i in range(1, 3))
    high_energy = sum(feats[f"freq_band_{i:02d}_energy"] for i in range(5, num_bands + 1))
    feats["freq_low_high_ratio"] = float(low_energy / (high_energy + 1e-12))
    return feats


def extract_all_features(img: np.ndarray) -> Dict[str, float]:
    """聚合全部特征。"""
    features = {}
    features.update(extract_intensity_features(img))
    features.update(extract_histogram_features(img))
    features.update(extract_glcm_features(img))
    features.update(extract_gradient_features(img))
    features.update(extract_frequency_features(img))
    return features


# ──────────────────────────────────────────────────────────────
# 数据发现与配对
# ──────────────────────────────────────────────────────────────

def discover_studies(data_root: Path) -> List[Path]:
    """发现所有 Study 目录。"""
    return sorted([p for p in data_root.iterdir() if p.is_dir()])


def get_study_config(study_name: str) -> Optional[Dict[str, Any]]:
    """根据 Study 名获取通道配置（支持前缀匹配）。"""
    for key, cfg in STUDY_CHANNEL_CONFIG.items():
        if study_name.startswith(key):
            return cfg
    return None


def auto_detect_nuclei_channels(screen_path: Path) -> Tuple[List[int], Dict[int, str]]:
    """
    自动检测: Hub 通道名含 'hoechst'/'nuclei'/'dapi'/'dna'/'h2b'（不区分大小写），
    其余为 Source 通道。
    
    注意: 此函数需要通道名元数据（需要额外的映射文件或 API），
    这里提供框架——如果无法自动推断，则跳过。
    """
    # 这里只是占位，实际需要从 IDR metadata 读取通道名
    # 如果没有通道名，则使用启发式：
    # 通常通道编号最小的 DAPI/DNA/Hoechst 在低索引
    channel_dirs = sorted([d for d in screen_path.iterdir() if d.is_dir() and d.name.startswith("channel_")])
    hub, source = [], {}
    for d in channel_dirs:
        idx = int(d.name.split("_")[1])
        source[idx] = f"Ch{idx}"
    # 默认: 最后一个通道为 Hub (Hoechst 常在高索引)
    return [max(source.keys())], {k: v for k, v in source.items() if k != max(source.keys())}


def pair_images(study_path: Path, study_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    配对一个 Study 下的 Hub-Source 图像对。
    支持 per-screen 通道名称覆盖 (screen_configs)。
    """
    pairs = []
    study_name = study_path.name
    hub_label = study_config.get("hub_label", "Hub")
    screen_configs = study_config.get("screen_configs", {})

    for screen_path in sorted(study_path.iterdir()):
        if not screen_path.is_dir():
            continue

        screen_name = screen_path.name

        # 确定该 screen 的通道映射（优先用 screen_configs 覆盖）
        hub_ids = study_config["hub_channels"]
        if screen_name in screen_configs:
            src_map = screen_configs[screen_name]
        else:
            src_map = study_config["source_channels"]

        if hub_ids == "auto_nuclei":
            hub_ids, src_map = auto_detect_nuclei_channels(screen_path)
            if not hub_ids:
                print(f"  [WARN] {study_name}/{screen_name}: 无法自动推断 Nuclei 通道，跳过")
                continue

        # 收集所有通道的文件名集合
        channel_files: Dict[int, set] = {}
        for ch_idx in list(hub_ids) + list(src_map.keys()):
            ch_dir = screen_path / f"channel_{ch_idx}"
            if not ch_dir.exists():
                print(f"  [WARN] {ch_dir} 不存在，跳过该通道")
                continue
            tiffs = set(f.name for f in ch_dir.glob("*.tiff")) | set(f.name for f in ch_dir.glob("*.tif"))
            channel_files[ch_idx] = tiffs

        if not channel_files:
            continue

        # 求所有通道的共同文件名（配对）
        all_sets = list(channel_files.values())
        common_files = all_sets[0].intersection(*all_sets[1:]) if len(all_sets) > 1 else all_sets[0]

        for hid in hub_ids:
            if hid in channel_files:
                common_files = common_files & channel_files[hid]

        if not common_files:
            print(f"  [WARN] {study_name}/{screen_name}: 无共同配对文件")
            continue

        for fname in sorted(common_files):
            pair = {
                "study": study_name,
                "screen": screen_name,
                "image_id": Path(fname).stem,
                "hub_label": hub_label,
                "hub_paths": {hid: str(screen_path / f"channel_{hid}" / fname) for hid in hub_ids},
                "source_paths": {sid: (str(screen_path / f"channel_{sid}" / fname), sname)
                                 for sid, sname in src_map.items()},
            }
            pairs.append(pair)

    return pairs


# ──────────────────────────────────────────────────────────────
# 主处理流程
# ──────────────────────────────────────────────────────────────

def process_dataset(data_root: str, output_csv: str, checkpoint_path: str,
                    max_pairs_per_study: int = 500, seed: int = 42):
    """主入口: 遍历数据、提取特征、输出 CSV。"""
    random.seed(seed)
    np.random.seed(seed)

    data_root = Path(data_root)
    output_csv = Path(output_csv)
    checkpoint_path = Path(checkpoint_path)

    # 断点续跑: 加载已处理的 image_id 集合
    processed = set()
    if checkpoint_path.exists():
        processed = set(json.loads(checkpoint_path.read_text()))
        print(f"[Checkpoint] 已恢复 {len(processed)} 条已处理记录")

    studies = discover_studies(data_root)
    print(f"发现 {len(studies)} 个 Study: {[s.name for s in studies]}")

    all_rows: List[Dict[str, Any]] = []

    for study_path in studies:
        study_name = study_path.name
        config = get_study_config(study_name)
        if config is None:
            print(f"[SKIP] {study_name}: 未找到通道配置")
            continue
        print(f"\n{'='*60}\n处理 Study: {study_name}\n{'='*60}")

        pairs = pair_images(study_path, config)
        print(f"  共配对 {len(pairs)} 对图像")

        # 随机采样
        if len(pairs) > max_pairs_per_study:
            pairs = random.sample(pairs, max_pairs_per_study)
            print(f"  随机采样 {max_pairs_per_study} 对")

        for pair in tqdm(pairs, desc=f"  {study_name}", unit="pair"):
            # 生成唯一 key 用于断点续跑
            hub_key = ",".join(pair["hub_paths"].values())
            src_key = ",".join(v[0] for v in pair["source_paths"].values())
            unique_key = f"{pair['study']}|{pair['screen']}|{pair['image_id']}"
            if unique_key in processed:
                continue

            try:
                # --- Hub 特征 ---
                for hid, hpath in pair["hub_paths"].items():
                    img = tifffile.imread(hpath)
                    feats = extract_all_features(img)
                    row = {
                        "study": pair["study"],
                        "screen": pair["screen"],
                        "image_id": pair["image_id"],
                        "channel_type": "hub",
                        "channel_index": hid,
                        "channel_name": pair.get("hub_label", f"hub_ch{hid}"),
                    }
                    row.update(feats)
                    all_rows.append(row)

                # --- Source 特征 ---
                for sid, (spath, sname) in pair["source_paths"].items():
                    img = tifffile.imread(spath)
                    feats = extract_all_features(img)
                    row = {
                        "study": pair["study"],
                        "screen": pair["screen"],
                        "image_id": pair["image_id"],
                        "channel_type": "source",
                        "channel_index": sid,
                        "channel_name": sname,
                    }
                    row.update(feats)
                    all_rows.append(row)

                processed.add(unique_key)

            except Exception as e:
                print(f"\n  [ERROR] {pair['study']}/{pair['screen']}/{pair['image_id']}: {e}")

        # 每处理完一个 Study 就保存 checkpoint
        checkpoint_path.write_text(json.dumps(list(processed)))
        print(f"  [Checkpoint] 已保存 ({len(processed)} 条)")

    # 最终保存 CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ 特征已保存到: {output_csv}")
        print(f"  - 总行数: {len(df)}")
        print(f"  - Study 数: {df['study'].nunique()}")
        print(f"  - 特征维度: {len(df.columns) - 5}")  # 减去元数据列
    else:
        print("\n[WARN] 无数据输出")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IDR 多通道图像特征提取")
    parser.add_argument("--data-root", type=str,
                        default="/data2/xujr/idr_data/test_feature",
                        help="数据根目录")
    parser.add_argument("--output-csv", type=str,
                        default="/home/xujr/cross_registration/feature_test/features.csv",
                        help="输出 CSV 路径")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/xujr/cross_registration/feature_test/checkpoint.json",
                        help="断点续跑文件路径")
    parser.add_argument("--max-pairs", type=int, default=500,
                        help="每个 Study 最大采样对数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process_dataset(
        data_root=args.data_root,
        output_csv=args.output_csv,
        checkpoint_path=args.checkpoint,
        max_pairs_per_study=args.max_pairs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
