#!/usr/bin/env python3
"""
特征可视化脚本 —— 分析 features.csv，生成多维度对比图表。
用于回答: Hub vs Source 通道在特征空间中的差异模式是怎样的？

输出图表:
  1. feature_distribution.png     —— 核心特征的小提琴图/箱线图（按通道类型分面）
  2. pca_cluster.png              —— PCA 降维散点图（按 Study + 通道着色）
  3. hub_vs_source_radar.png      —— 各 Study 的 Hub vs Source 雷达图
  4. correlation_heatmap.png      —— 特征间相关性热力图
  5. per_study_comparison.png     —— 每个 Study 内 Hub vs 各 Source 的逐特征对比
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
CSV_PATH = Path(__file__).resolve().parent / "features.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# 全局 matplotlib 风格
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "figure.titlesize": 13,
})
sns.set_style("whitegrid")

# 用于可视化的核心特征子集（去掉冗余的频域绝对能量）
CORE_FEATURES = [
    "int_mean", "int_std", "int_skewness", "int_kurtosis",
    "int_dynamic_range", "int_snr",
    "hist_entropy", "hist_max_peak", "hist_active_bins",
    "glcm_contrast_mean", "glcm_dissimilarity_mean",
    "glcm_homogeneity_mean", "glcm_energy_mean", "glcm_correlation_mean",
    "grad_mean", "grad_std", "grad_p90", "edge_density",
    "freq_low_high_ratio",
]
# 核心特征（剔除频域绝对能量——数量级差异太大不适合一起标准化）
ALL_NON_ENERGY = [c for c in CORE_FEATURES if not c.startswith("freq_band_")]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 创建复合标签列
    df["study_channel"] = df["study"].str.replace("idr", "").str[:4] + "|" + df["channel_name"]
    df["channel_label"] = df.apply(
        lambda r: f"{'HUB' if r['channel_type']=='hub' else 'SRC'}: {r['channel_name']}", axis=1
    )
    return df


# ═══════════════════════════════════════════
# 图 1: 核心特征分布（小提琴 + 箱线混合）
# ═══════════════════════════════════════════
def plot_feature_distributions(df: pd.DataFrame):
    """按 study 分面，每个特征 hub vs source 的分组小提琴图。
    当 Study 较多时，每个 Study 单独一张图，避免 subplot 过多导致 tight_layout 卡死。
    """
    id_cols = ["study", "screen", "image_id", "channel_type", "channel_name", "study_channel", "channel_label"]
    df_long = df.melt(id_vars=id_cols, value_vars=ALL_NON_ENERGY,
                      var_name="feature", value_name="value")

    studies = sorted(df["study"].unique())
    features = ALL_NON_ENERGY
    n_feats = len(features)
    n_cols = min(3, max(1, n_feats // 5))

    for study in studies:
        study_short = study.replace("idr", "").split("-")[0]
        sub_long = df_long[df_long["study"] == study]
        n_rows = int(np.ceil(n_feats / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(3.8 * n_cols, 2.2 * n_rows),
                                  constrained_layout=True)
        axes = np.atleast_1d(axes).flatten()

        for fi, feat in enumerate(features):
            ax = axes[fi]
            feat_data = sub_long[sub_long["feature"] == feat]
            if feat_data.empty:
                ax.set_visible(False)
                continue
            hub_vals = feat_data[feat_data["channel_type"] == "hub"]["value"].dropna()
            src_vals = feat_data[feat_data["channel_type"] == "source"]["value"].dropna()

            if len(hub_vals) == 0 and len(src_vals) == 0:
                ax.set_visible(False)
                continue
            data = [v.values for v in [hub_vals, src_vals] if len(v) > 0]
            labels = []
            if len(hub_vals): labels.append("Hub")
            if len(src_vals): labels.append("Src")
            colors = ["#2196F3"][:len(hub_vals)>0] + ["#FF9800"][:len(src_vals)>0]
            positions = list(range(len(data)))

            parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(feat, fontsize=7)
            ax.tick_params(labelsize=6)

        # 隐藏多余 subplot
        for ax in axes[n_feats:]:
            ax.set_visible(False)

        fig.suptitle(f"{study}: Feature Distributions (Hub vs Source)", fontweight="bold", fontsize=12)
        fig.savefig(OUT_DIR / f"feature_distribution_{study_short}.png", bbox_inches="tight")
        plt.close(fig)
    print(f"  ✓ feature_distribution_*.png ({len(studies)} studies)")


# ═══════════════════════════════════════════
# 图 2: PCA 降维散点图
# ═══════════════════════════════════════════
def plot_pca(df: pd.DataFrame):
    """2D PCA: 看不同 Study 和通道类型的聚类情况。"""
    X = df[ALL_NON_ENERGY].values.astype(np.float64)
    # 处理 NaN / Inf（如 SNR 除零、频域比值等可能导致）
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = df[["study", "channel_type", "channel_name"]].copy()
    df_pca["PC1"] = X_pca[:, 0]
    df_pca["PC2"] = X_pca[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # 左: 按 Study 着色
    ax = axes[0]
    studies = sorted(df_pca["study"].unique())
    palette = sns.color_palette("tab10", len(studies))
    for i, s in enumerate(studies):
        sub = df_pca[df_pca["study"] == s]
        ax.scatter(sub["PC1"], sub["PC2"], c=[palette[i]], label=s, alpha=0.6, s=20, edgecolors="none")
        # 标注中心
        ax.scatter(sub["PC1"].mean(), sub["PC2"].mean(), c=[palette[i]], marker="X",
                   s=120, edgecolors="black", linewidths=0.8)
    ax.set_title("PCA Colored by Study")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=7, loc="upper right")

    # 右: 按 Hub/Source 着色
    ax = axes[1]
    colors = {"hub": "#2196F3", "source": "#FF9800"}
    markers = {"hub": "o", "source": "^"}
    for ct in ["hub", "source"]:
        sub = df_pca[df_pca["channel_type"] == ct]
        ax.scatter(sub["PC1"], sub["PC2"], c=colors[ct], marker=markers[ct],
                   label=ct.capitalize(), alpha=0.4, s=20, edgecolors="none")
        ax.scatter(sub["PC1"].mean(), sub["PC2"].mean(), c=colors[ct], marker="X",
                   s=150, edgecolors="black", linewidths=1)
    ax.set_title("PCA Colored by Hub / Source")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=8)

    fig.suptitle("PCA of Multi-Channel Image Features", fontweight="bold")
    fig.savefig(OUT_DIR / "pca_cluster.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ pca_cluster.png")


# ═══════════════════════════════════════════
# 图 3: 相关性热力图
# ═══════════════════════════════════════════
def plot_correlation_heatmap(df: pd.DataFrame):
    """特征间 Pearson 相关性。"""
    X = df[ALL_NON_ENERGY]
    corr = X.corr()

    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.3, cbar_kws={"shrink": 0.7},
                ax=ax, annot_kws={"fontsize": 6})
    ax.set_title("Feature Correlation Matrix (Pearson)", fontweight="bold")
    fig.savefig(OUT_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ correlation_heatmap.png")


# ═══════════════════════════════════════════
# 图 4: 每个 Study 的 Hub vs Source 雷达图
# ═══════════════════════════════════════════
def plot_radar(df: pd.DataFrame):
    """每个 Study，将 Hub 和各 Source 通道的特征均值画成雷达图。"""
    studies = sorted(df["study"].unique())
    # 选择 6-8 个最有区分度的特征做雷达图
    radar_feats = [
        "int_mean", "int_std", "int_skewness", "hist_entropy",
        "glcm_contrast_mean", "glcm_homogeneity_mean",
        "grad_mean", "edge_density",
    ]

    n_studies = len(studies)
    n_cols = min(3, n_studies)
    n_rows = int(np.ceil(n_studies / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                              subplot_kw={"projection": "polar"}, constrained_layout=True)
    if n_studies == 1:
        axes = np.array([axes])

    # 计算全局 min/max 用于归一化
    global_min, global_max = {}, {}
    for feat in radar_feats:
        vals = df[feat].values
        global_min[feat] = np.percentile(vals, 5)
        global_max[feat] = np.percentile(vals, 95)

    angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for si, study in enumerate(studies):
        ax = axes.flat[si] if n_studies > 1 else axes[0]
        sub = df[df["study"] == study]

        # Hub
        hub_sub = sub[sub["channel_type"] == "hub"]
        hub_vals = []
        for feat in radar_feats:
            v = hub_sub[feat].mean()
            hub_vals.append((v - global_min[feat]) / (global_max[feat] - global_min[feat] + 1e-8))
        hub_vals += hub_vals[:1]
        ax.fill(angles, hub_vals, alpha=0.3, color="#2196F3", label="Hub")
        ax.plot(angles, hub_vals, color="#2196F3", linewidth=2)

        # 各 Source
        src_colors = sns.color_palette("Oranges", 6)[2:]
        src_names = sorted(sub[sub["channel_type"] == "source"]["channel_name"].unique())
        for sni, sname in enumerate(src_names):
            src_sub = sub[(sub["channel_type"] == "source") & (sub["channel_name"] == sname)]
            src_vals = []
            for feat in radar_feats:
                v = src_sub[feat].mean()
                src_vals.append((v - global_min[feat]) / (global_max[feat] - global_min[feat] + 1e-8))
            src_vals += src_vals[:1]
            color = src_colors[sni % len(src_colors)]
            ax.fill(angles, src_vals, alpha=0.15, color=color)
            ax.plot(angles, src_vals, color=color, linewidth=1.5, label=sname)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_feats, fontsize=7)
        ax.set_title(study, fontsize=10, fontweight="bold", pad=15)
        ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # 隐藏多余的 subplot
    for ax in axes.flat[n_studies:]:
        ax.set_visible(False)

    fig.suptitle("Hub vs Source Channel Feature Profiles (Radar)", fontweight="bold", y=1.02)
    fig.savefig(OUT_DIR / "hub_vs_source_radar.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ hub_vs_source_radar.png")


# ═══════════════════════════════════════════
# 图 5: 每个 Study 内逐特征条形对比
# ═══════════════════════════════════════════
def plot_per_study_bar(df: pd.DataFrame):
    """每个 Study 一张大图: 每个特征 Hub 均值 vs 各 Source 均值的条形图。"""
    studies = sorted(df["study"].unique())
    compare_feats = ALL_NON_ENERGY

    for study in studies:
        sub = df[df["study"] == study]
        hub = sub[sub["channel_type"] == "hub"]
        hub_means = hub[compare_feats].mean()

        src_channels = sorted(sub[sub["channel_type"] == "source"]["channel_name"].unique())
        n_src = len(src_channels)

        fig, axes = plt.subplots(len(compare_feats), 1, figsize=(max(6, 2 * n_src), 2.8 * len(compare_feats)),
                                  sharex=True, constrained_layout=True)
        if len(compare_feats) == 1:
            axes = [axes]

        bar_width = 0.8 / (n_src + 1)
        colors_src = sns.color_palette("Set2", n_src)

        for fi, feat in enumerate(compare_feats):
            ax = axes[fi]
            positions = np.arange(n_src + 1)
            vals = [float(hub_means[feat])]
            for si, sname in enumerate(src_channels):
                src_sub = sub[(sub["channel_type"] == "source") & (sub["channel_name"] == sname)]
                vals.append(float(src_sub[feat].mean()))

            bars = ax.bar(positions, vals, width=bar_width * (n_src + 1) * 0.8)
            bars[0].set_color("#2196F3")
            for si in range(n_src):
                bars[si + 1].set_color(colors_src[si])

            ax.set_xticks(positions)
            ax.set_xticklabels(["Hub"] + src_channels, fontsize=7)
            ax.set_ylabel(feat, fontsize=7)
            # 添加数值标注
            for bi, (pos, val) in enumerate(zip(positions, vals)):
                ax.text(pos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=5.5,
                        rotation=45)

        fig.suptitle(f"{study}: Hub vs Source Feature Comparison", fontweight="bold")
        fig.savefig(OUT_DIR / f"per_study_{study}.png", bbox_inches="tight")
        plt.close(fig)
    print("  ✓ per_study_*.png")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    print(f"Loading: {csv_path}")
    df = load_data(csv_path)
    print(f"  Rows: {len(df)}, Studies: {df['study'].nunique()}, Features: {len(ALL_NON_ENERGY)}")

    print("\nGenerating figures...")
    plot_feature_distributions(df)
    plot_pca(df)
    plot_correlation_heatmap(df)
    plot_radar(df)
    plot_per_study_bar(df)

    print(f"\nAll figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
