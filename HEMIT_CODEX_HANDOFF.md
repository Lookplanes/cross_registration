# HEMIT 数据收集与接入交接说明

> **状态：已完成阶段的交接记录。** 本文用于当时独立窗口执行下载与接入，
> 不是当前训练规范；现行入口与数据路径见 `README-self.md`。

更新日期：2026-07-27

## 1. 项目目标

主项目位于 `/home/xujr/cross_registration_2`。项目已从“通用医学显微图像配准模型”收缩为聚焦组织病理图像域配准的完整论文。

当前路线：

1. 使用条件化多域 TransCUT 学习 H&E、DAPI、panCK、CD3 之间的转换；
2. 生成与源图像结构近似一致的目标域伪图；
3. 施加已知人工非刚性形变，构造 displacement 监督；
4. 训练统一跨域可变形配准模型；
5. 在真实配对数据上验证结构保持和配准迁移能力。

HEMIT 是受控核心数据：

- 第一阶段：H&E、DAPI、panCK 三域；
- 第二阶段：加入 CD3，形成四域；
- TransCUT 主实验在同一 split 内打散配对，保持非配对训练；
- validation/test 必须保留真实配对；
- 还需训练 paired translation upper bound；
- SHIFT 后续用于外部验证，不属于本数据窗口的首要任务。

## 2. 已完成工作

官方仓库已经部署到：

```text
/data2/wuyh/HEMIT
```

来源：

```text
https://github.com/BianChang/HEMIT-DATASET
```

已确认：

- 分支：`main`
- 提交：`e56a56ee0a6e40802b66c343c810f14f2f7d2b4e`
- 仓库约 73 MB；
- 仓库只有说明、许可证、示例图和评估脚本，不含真实数据。

数据来源：

```text
https://data.mendeley.com/datasets/3gx53zm49d/1
DOI: 10.17632/3gx53zm49d.1
```

许可证为 CC BY 4.0。使用、修改和发表时必须正确引用、附许可证链接并说明修改。

## 3. 已确认的数据事实

HEMIT 是来自 8 张结直肠癌 WSI 的同切片、细胞级配准 H&E–mIHC 图像对。

| Split | 配对 patch 数 |
|---|---:|
| train | 3717 |
| val | 630 |
| test | 945 |
| 总计 | 5292 |

- patch：`1024 × 1024`
- 格式：TIF
- 每对：一张 H&E RGB、一张三通道 mIHC
- input/label 使用相同文件名，可按 stem 恢复配对
- 官方结构为 `train|val|test/input|label`

### 3.1 通道顺序冲突

README 写作：

```text
DAPI, panCK, CD3
```

官方 `evaluation_metrics.py` 实际使用：

```text
channel 0 = DAPI
channel 1 = CD3
channel 2 = panCK
```

下载样本后必须：

1. 检查 TIF shape、dtype、photometric 和 compression；
2. 分别导出三个通道；
3. 对照官方示例图判断 DAPI/CD3/panCK；
4. 将证据和最终顺序写入 manifest；
5. 确认前禁止批量预处理。

## 4. 磁盘与体积

此前 `/data2` 状态：

```text
总容量约 7.0 TB
已用约 6.5 TB
可用约 117 GB
使用率 99%
```

执行任务前必须重新检查：

```bash
df -h /data2/wuyh
du -sh /data2/wuyh/HEMIT
```

Mendeley 对该账户的数据集发布上限为 10 GB，但这不是 HEMIT 的真实下载大小。尚未取得官方文件列表、Content-Length 和 checksum。

8-bit 三通道图像完全展开的理论像素体积：

```text
5292 × 2 × 1024 × 1024 × 3 bytes
= 33.294 GB
= 31.008 GiB
```

16-bit 时约 62.016 GiB。解压后实际体积取决于 TIF 内部压缩。

不得把三个 marker 各自复制为 RGB 后保存。应保存单通道 marker，在 Dataset 中动态扩展为三通道。

## 5. 首要任务：下载前验证

在大规模下载前确认：

1. Mendeley 提供一个大压缩包、多个 split 包，还是独立 TIF/目录；
2. 各文件精确大小和总大小；
3. MD5/SHA checksum；
4. 固定下载入口和最终重定向 URL；
5. HTTP Range 支持；
6. 临时 S3 URL 过期后能否从官方入口继续同一文件。

此前终端网络权限阻止了 Mendeley API 连接；尚未完成 Range 验证，也没有下载数据分片。

用约 2 MiB 验证：

```bash
curl -IL '<official-download-url>'

curl -L --range 0-1048575 \
  '<official-download-url>' \
  -o /tmp/hemit.range.0

curl -L --range 1048576-2097151 \
  '<official-download-url>' \
  -o /tmp/hemit.range.1
```

验收要求：

- 返回 `206 Partial Content`；
- `Content-Range` 与请求一致；
- 两段各 1 MiB；
- 记录 `Accept-Ranges`、`Content-Length`、ETag、Last-Modified；
- 完成后删除测试分片；
- 不得仅因为底层使用 S3 就推定续传可靠。

## 6. 可恢复下载要求

确认后优先使用：

```bash
aria2c \
  -c \
  -x 4 \
  -s 4 \
  --file-allocation=none \
  --auto-file-renaming=false \
  --dir=/data2/wuyh/HEMIT/dataset/archive \
  --out='<official-file-name>' \
  '<official-download-url>'
```

要求：

- 日志放入 `dataset/logs/`；
- 保留 aria2 控制文件；
- 中断后先核对 ETag/Last-Modified；
- 完成后验证官方 checksum；
- 无官方 checksum 时记录本地 SHA-256；
- 校验前不解压、不改名、不删除控制文件。

不要人为截断大压缩包到 10 GB。不完整 ZIP/7z 通常不能可靠测试。应完整下载归档，再选择性解压少量样本。

## 7. 小规模测试数据

建议首批：

| Split | 数量 |
|---|---:|
| train | 100–300 对 |
| val | 30–50 对 |
| test | 30–50 对 |

要求：

- input/label 成对；
- 保留三个官方 split；
- 尽可能覆盖多个 WSI；
- 不只选择排序后的连续前 N 张；
- 固定随机种子；
- 保存原始相对路径、split 和 pair ID；
- `raw` 数据只读。

检查项目：

- shape、dtype、位深、通道数；
- TIFF compression 和 photometric；
- H&E 是否为 RGB；
- mIHC 通道顺序；
- stem 和尺寸是否匹配；
- 损坏、全黑、低信息 patch；
- 明显残余错位；
- split 是否按 WSI/患者隔离；
- 单张与总样本实际体积。

## 8. 推荐目录

```text
/data2/wuyh/HEMIT/
├── .git/
├── README.md
├── LICENSE
├── evaluation_metrics.py
├── images/
├── CODEX_HANDOFF.md
└── dataset/
    ├── archive/
    ├── raw/
    ├── sample/
    ├── processed/
    ├── manifests/
    ├── previews/
    └── logs/
```

将 `dataset/` 写入该仓库本地 `.git/info/exclude`，不要修改官方 `.gitignore`，也不要将数据、日志和预览加入 Git。

## 9. 预处理规范

| Domain ID | 图像域 | 磁盘保存 |
|---:|---|---|
| 0 | H&E | RGB |
| 1 | DAPI | 单通道 |
| 2 | panCK | 单通道 |
| 3 | CD3 | 单通道 |

Domain ID 是项目定义，不等于原始 TIF channel index。

- H&E 保留 RGB；
- marker 在加载时动态复制为 RGB；
- 不保存重复 RGB marker；
- 不添加伪彩色；
- 不覆盖原始数据；
- 不跨 split 打散；
- 训练可在 split 内进行 unpaired 采样；
- val/test 始终保留 pair ID。

manifest 至少包含：

```text
split
pair_id
source_wsi_id
patient_id
he_path
mihc_path
dapi_channel_index
cd3_channel_index
panck_channel_index
height
width
dtype
tiff_compression
file_size
sha256
```

没有 patient ID 时留空并说明，不得从坐标文件名无依据推测。

## 10. 主项目兼容性

当前数据读取：

```text
/home/xujr/cross_registration_2/src/crossreg/data/translation.py
```

已支持：

- TIF/TIFF；
- 多域目录；
- unpaired/paired；
- RGB 输入；
- paired 按公共 stem 配对；
- paired shared geometry。

不能直接把三通道 mIHC 复合图当作三个独立域。必须分离通道，或新增 manifest/channel-view Dataset。优先采用不复制数据的 channel-view 方案。

正式接入前只生成小样本 manifest 和预览，不修改模型结构，不启动正式训练。

## 11. 交付物

1. 官方文件列表、精确大小和 checksum；
2. Range/断点续传测试记录；
3. 可恢复下载命令或脚本；
4. 下载状态和日志路径；
5. 小规模配对样本；
6. dtype、位深、压缩方式统计；
7. mIHC 通道顺序的证据和结论；
8. split/pair/WSI manifest；
9. H&E、mIHC 合成图及 marker 预览；
10. 实际磁盘占用与完整解压估计；
11. 接入当前 TransCUT/配准流程的结论；
12. 未解决问题列表。

大规模下载、完整解压或大体积预处理前，先向用户报告：

- 精确下载量；
- 峰值空间；
- 最终保留体积；
- 清理策略；
- 续传能力；
- 预计耗时。

## 12. 禁止事项

- 不把 10 GB 平台上限写成真实下载大小；
- 不在未实测 `206` 时承诺续传；
- 不把不完整归档当成可用数据；
- 不先完整解压再考虑空间；
- 不保存 marker 的 RGB 重复副本；
- 不跨 split 打散；
- 不丢失真实配对；
- 不在通道未确认前批量处理；
- 不将数据、归档、日志和预览加入 Git；
- 不在小样本检查完成前启动正式训练。
