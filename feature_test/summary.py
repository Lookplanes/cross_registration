#!/usr/bin/env python3
"""
汇总仪表盘 —— 两张图回答核心问题：
  1. dashboard_domain_gap.png  — 各 Study 的 Hub↔Source 特征距离热力图（谁最需要配准？）
  2. dashboard_pca_overview.png — PCA + top 特征贡献（什么特征在区分模态？）
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

CSV_PATH = Path(__file__).resolve().parent / "features.csv"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 150})
sns.set_style("whitegrid")

# 核心特征（与 main.py 输出一致）
CORE_FEATURES = [
    "int_mean", "int_std", "int_skewness", "int_kurtosis",
    "int_dynamic_range", "int_snr",
    "hist_entropy", "hist_max_peak", "hist_active_bins",
    "glcm_contrast_mean", "glcm_dissimilarity_mean",
    "glcm_homogeneity_mean", "glcm_energy_mean", "glcm_correlation_mean",
    "grad_mean", "grad_std", "grad_p90", "edge_density",
    "freq_low_high_ratio",
]
# 用于显示的简短标签
FEAT_LABELS = {
    "int_mean": "Mean", "int_std": "Std", "int_skewness": "Skew",
    "int_kurtosis": "Kurt", "int_dynamic_range": "DynRange", "int_snr": "SNR",
    "hist_entropy": "Entropy", "hist_max_peak": "Peak", "hist_active_bins": "ActiveBins",
    "glcm_contrast_mean": "GLCM_Contrast", "glcm_dissimilarity_mean": "GLCM_Dissim",
    "glcm_homogeneity_mean": "GLCM_Homog", "glcm_energy_mean": "GLCM_Energy",
    "glcm_correlation_mean": "GLCM_Corr",
    "grad_mean": "GradMean", "grad_std": "GradStd", "grad_p90": "GradP90",
    "edge_density": "EdgeDens", "freq_low_high_ratio": "Low/HighFreq",
}

# Hub 类型映射（兼容新旧 CSV 格式：新版用 "Hoechst"/"DAPI"，旧版用 "hub_ch3" 等）
HUB_TYPE_MAP = {
    "idr0003": "H2B", "idr0037": "DAPI", "idr0056": "Hoechst",
    "idr0080": "DNA", "idr0081": "DAPI", "idr0086": "DAPI",
    "idr0161": "Nuclei", "idr0163": "DAPI/Hoechst",
}

def _get_hub_type(row_or_series, study: str = "") -> str:
    """推断 Hub 类型：优先用 channel_name（新格式），否则 fallback 映射表。"""
    if hasattr(row_or_series, "get"):
        cn = row_or_series.get("channel_name", "")
    else:
        cn = str(row_or_series)
    if cn and not cn.startswith("hub_ch") and cn not in ("unknown",):
        return cn
    s = study or (row_or_series.get("study", "") if hasattr(row_or_series, "get") else "")
    for key, ht in HUB_TYPE_MAP.items():
        if s.startswith(key):
            return ht
    return "Hub"


def load_data():
    df = pd.read_csv(CSV_PATH)
    # 处理 NaN
    df[CORE_FEATURES] = df[CORE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    return df


# ═══════════════════════════════════════════
# 图 1: Hub-Source 特征距离矩阵
# ═══════════════════════════════════════════
def plot_domain_gap(df: pd.DataFrame):
    """热力图: 每行 = Study+Source通道，每列 = 特征。
    Z-score: (Source_mean - Hub_mean) / Hub_std。
    左侧标注 Hub 类型（DNA/Hoechst/DAPI...），行按 Study 分组。
    """
    studies = sorted(df["study"].unique())

    rows_data = []
    row_labels = []
    hub_types = []      # 每行对应的 Hub 类型
    study_boundaries = [0]  # Study 分组分隔线位置

    for study in studies:
        sub = df[df["study"] == study]
        hub = sub[sub["channel_type"] == "hub"]
        if hub.empty:
            continue
        hub_mean = hub[CORE_FEATURES].mean()
        hub_std = hub[CORE_FEATURES].std().replace(0, 1e-8)
        hub_type = _get_hub_type(hub["channel_name"].iloc[0] if len(hub) else "", study)

        src_channels = sorted(sub[sub["channel_type"] == "source"]["channel_name"].unique())
        if not src_channels:
            continue

        for sname in src_channels:
            src = sub[(sub["channel_type"] == "source") & (sub["channel_name"] == sname)]
            if src.empty:
                continue
            src_mean = src[CORE_FEATURES].mean()
            diff = (src_mean - hub_mean) / hub_std
            rows_data.append(diff.values)
            study_short = study.replace("idr", "").split("-")[0]
            row_labels.append(f"{study_short} | {sname}")
            hub_types.append(hub_type)

        study_boundaries.append(len(rows_data))

    if not rows_data:
        print("  [WARN] 无数据可绘制")
        return

    mat = np.array(rows_data)
    mat = np.clip(mat, -3, 3)

    n_rows = len(row_labels)
    fig, ax = plt.subplots(figsize=(max(15, len(CORE_FEATURES) * 0.55),
                                    max(7, n_rows * 0.38)),
                            constrained_layout=True)

    # 主热力图
    sns.heatmap(mat, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                vmin=-3, vmax=3, linewidths=0.5, linecolor="#eee",
                xticklabels=[FEAT_LABELS.get(f, f) for f in CORE_FEATURES],
                yticklabels=row_labels,
                cbar_kws={"label": "Z-score vs Hub  (red=src>hub, blue=src<hub)", "shrink": 0.5},
                ax=ax, annot_kws={"fontsize": 6})

    # 左侧标注 Hub 类型
    hub_colors = {"DAPI": "#1B5E20", "Hoechst": "#0D47A1", "DNA": "#4A148C",
                  "H2B": "#B71C1C", "Nuclei": "#E65100", "DAPI/Hoechst": "#004D40"}
    for i, ht in enumerate(hub_types):
        c = hub_colors.get(ht, "#616161")
        ax.annotate(ht, xy=(-0.06, i + 0.5), xycoords=("axes fraction", "data"),
                    ha="right", va="center", fontsize=6, fontweight="bold", color=c,
                    annotation_clip=False)

    # Study 分组分隔线
    for b in study_boundaries[1:-1]:
        ax.axhline(b, color="black", linewidth=1.5)

    # 标题 + 说明
    ax.set_title("Domain Gap: Source vs Hub (★ DNA/Hoechst = Universal Anchor ★)\n"
                 "Darker color = larger gap from Hub → harder cross-modality registration",
                 fontweight="bold", fontsize=10, loc="left")
    ax.tick_params(axis='x', labelsize=7, rotation=45)
    ax.tick_params(axis='y', labelsize=7)

    # 图例: Hub 类型
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=f"Hub={ht}") for ht, c in hub_colors.items()
                      if ht in set(hub_types)]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper right",
              bbox_to_anchor=(1.15, 1.0), title="Hub Channel Type", title_fontsize=7)

    fig.savefig(OUT_DIR / "dashboard_domain_gap.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ dashboard_domain_gap.png")


# ═══════════════════════════════════════════
# 图 2: PCA + Top 判别特征
# ═══════════════════════════════════════════
def plot_pca_overview(df: pd.DataFrame):
    """左: PCA 散点图（Hub vs Source，按 Study 着色）。
    右: 对 Hub/Source 分类贡献最大的 Top-10 特征（ANOVA F-score）。
    """
    X = df[CORE_FEATURES].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                              gridspec_kw={"width_ratios": [1.2, 1]})

    # ── 左: PCA ──
    ax = axes[0]
    studies = sorted(df["study"].unique())
    palette = sns.color_palette("tab10", len(studies))
    marker_map = {"hub": "o", "source": "^"}
    alpha_map = {"hub": 0.8, "source": 0.25}

    for ct in ["hub", "source"]:
        sub = df[df["channel_type"] == ct]
        for si, study in enumerate(studies):
            ssub = sub[sub["study"] == study]
            if ssub.empty:
                continue
            idx = ssub.index
            ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                       c=[palette[si]], marker=marker_map[ct],
                       alpha=alpha_map[ct], s=18 if ct == "source" else 35,
                       edgecolors="none",
                       label=f"{study}" if ct == "hub" else "")

    # 标注 Hub 中心
    for si, study in enumerate(studies):
        idx = df[(df["study"] == study) & (df["channel_type"] == "hub")].index
        if len(idx):
            ax.scatter(X_pca[idx, 0].mean(), X_pca[idx, 1].mean(),
                       c=[palette[si]], marker="D", s=80,
                       edgecolors="black", linewidths=0.8, zorder=10)

    # 自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                               markersize=8, label='Hub (nuclei)'),
                       Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                               markersize=8, label='Source')]
    for si, study in enumerate(studies):
        legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                       markerfacecolor=palette[si], markersize=8,
                                       label=study))
    ax.legend(handles=legend_elements, fontsize=7, loc="upper left",
              framealpha=0.9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA: Hub (●) vs Source (▲)\n★ All Hubs = DNA / Hoechst / DAPI / H2B (nuclear stains)",
                 fontweight="bold", fontsize=10)

    # ── 右: Top 判别特征 ──
    ax = axes[1]
    from sklearn.feature_selection import f_classif
    y_binary = (df["channel_type"] == "hub").astype(int).values
    f_scores, _ = f_classif(X_scaled, y_binary)
    top_k = 12
    top_idx = np.argsort(f_scores)[-top_k:][::-1]
    top_names = [CORE_FEATURES[i] for i in top_idx]
    top_labels = [FEAT_LABELS.get(n, n) for n in top_names]

    colors = ["#2196F3" if df[df["channel_type"] == "hub"][n].mean() >
              df[df["channel_type"] == "source"][n].mean() else "#FF9800"
              for n in top_names]

    bars = ax.barh(range(top_k), f_scores[top_idx][::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_labels[::-1], fontsize=8)
    ax.set_xlabel("ANOVA F-score (higher = better discriminator)")
    ax.set_title("Top Features Separating Hub vs Source\n(★ Hub = nuclear DNA stain across all Studies)",
                 fontweight="bold", fontsize=10)
    # 图例
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2196F3", label="Hub > Source"),
        Patch(color="#FF9800", label="Source > Hub"),
    ], fontsize=7, loc="lower right")

    fig.suptitle("Global Overview — DNA/Hoechst as Universal Registration Hub", fontweight="bold", fontsize=13)
    fig.savefig(OUT_DIR / "dashboard_pca_overview.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ dashboard_pca_overview.png")


# ═══════════════════════════════════════════
def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    print(f"Loading: {csv_path}")
    df = load_data()
    print(f"  Studies={df['study'].nunique()}, Rows={len(df)}, "
          f"Hub={len(df[df['channel_type']=='hub'])}, Source={len(df[df['channel_type']=='source'])}")

    print("\nGenerating summary dashboard...")
    plot_domain_gap(df)
    plot_pca_overview(df)
    print(f"\nDone → {OUT_DIR}/dashboard_*.png")


if __name__ == "__main__":
    main()
