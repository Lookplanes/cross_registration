# Cross Registration — Engineering Contract

> **状态：现行工程契约。** 本文只记录模块职责、接口、数值约定、启动方式
> 和兼容边界。研究范围见 `AAA-document/0725Report/0725.md`，历次实验及
> 成败结论见 [EXPERIMENTS.md](EXPERIMENTS.md)。

代码行为以自动测试和每次运行保存的 `run_config.json` 为准。

## 工程范围

- Stage 1：条件化多域翻译与共享 Encoder 预训练。
- Stage 2：用模态转换图和已知人工形变训练配准头。
- Stage 3：移除 Decoder，直接对真实跨模态图像预测位移场。
- 当前代码重点支持 2D 图像；同一次 TransCUT 训练的所有域必须具有相同
  通道数，通道适配在离线数据预处理完成。
- 当前不进行自定义 C/CUDA 算子开发。

## 当前结构

```text
cross_registration_2/
├── configs/
│   ├── analysis/                 # 数据源、IDR 通道与全模态分析配置
│   ├── experiments/              # 实验编排示例（仍属初版）
│   ├── translation/              # CUT YAML 示例
│   └── registration/             # TransMorph YAML 示例
├── scripts/
│   ├── train_translation.py      # 两模态 CUT 训练
│   ├── train_transcut.py         # N-to-N TransCUT 训练（Stage 1）
│   ├── train_synthetic_supervised.py # 合成监督 Conditioned TransMorph（Stage 2）
│   ├── build_stage2_offline_dataset.py # 固定 Stage 1 后离线生成/扰动
│   ├── train_stage2_offline.py   # 从离线 manifest 训练 Conditioned TransMorph
│   ├── train_registration.py     # TransMorph 同模态无监督训练
│   ├── evaluate.py               # CUT + TransMorph / TransMorph-only 评估
│   ├── validate_dataset.py       # 独立、只读的数据预检查
│   ├── run_experiment.py         # 初版实验编排器
│   └── extract_*.py, analyze_*.py, run_full_tsne.py
├── src/crossreg/
│   ├── data/                     # Dataset、变换与 SynthMorph 风格扰动
│   ├── config/                   # YAML 默认值 + CLI override + resolved config
│   ├── models/                   # 共享 Swin Transformer
│   ├── translation/
│   │   ├── cut/                  # CUT、PatchNCE、GAN 网络
│   │   └── transcut/             # CLN/AdaIN 条件翻译
│   ├── registration/transmorph/  # TransMorph、条件主线、弃用 Head、损失
│   ├── pipeline/                 # 两阶段和 Stage 3 推理原型
│   └── utils/metrics.py
├── src/modality_analyzer/        # 手工/ResNet 特征与可视化
├── tests/                        # smoke、数值契约与脚本级 E2E
└── results/                      # 分析产物，不纳入 Git
```

### 关键文件职责

| 文件或目录 | 唯一职责 |
|---|---|
| `scripts/train_transcut.py` | Stage 1 单卡/DDP 训练、日志、样例与 checkpoint 生命周期 |
| `scripts/train_synthetic_supervised.py` | Stage 2 合成跨模态 pair 构造与 Conditioned TransMorph 监督训练 |
| `scripts/build_stage2_offline_dataset.py` | 可恢复地离线执行 TransCUT、扰动和 pair 构造；不训练模型 |
| `scripts/train_stage2_offline.py` | 只读取离线 moving/fixed/flow/ID，训练 Conditioned TransMorph；不加载生成器 |
| `scripts/visualize_stage2_offline_samples.py` | 将源图、对齐转换图、扰动图、flow分量及mask汇总为诊断图 |
| `scripts/export_stage2_validation_samples.py` | 从已训练的Stage 2 checkpoint独立导出固定验证预测样例与指标，不修改训练状态 |
| `scripts/merge_stage2_offline_manifests.py` | 合并互斥方向分片，并严格检查数量、方向平衡、ID唯一性与文件存在性 |
| `scripts/verify_stage2_offline_dataset.py` | 穷举读取正式PNG/flow/mask，审计schema、集合、数值和磁盘统计 |
| `scripts/train_registration.py` | 独立 TransMorph 同模态无监督训练，不代替 Stage 2 |
| `scripts/evaluate.py` | 严格加载 checkpoint 后执行评估 |
| `scripts/validate_dataset.py` | 独立只读数据预检查，不自动清洗或修改训练集 |
| `scripts/train_transcut_*.sh` | 固化可复现实验参数；不得在 Python 模型中硬编码数据路径 |
| `src/crossreg/data/translation.py` | paired/unpaired 多域采样与共享几何增强 |
| `src/crossreg/data/modalities.py` | 稳定的模态名称、ID、路径和通道注册表 |
| `src/crossreg/data/perturbation.py` | 生成 Stage 2 的速度场与 displacement |
| `src/crossreg/data/stage2_offline.py` | 离线 Stage 2 manifest、路径、归一化、flow 与模态 ID 契约 |
| `src/crossreg/translation/transcut/transcut_model.py` | 组装 TransCUT、计算 loss、保存/恢复完整训练状态 |
| `src/crossreg/translation/transcut/decoder.py` | checkpoint-compatible `legacy` Decoder |
| `src/crossreg/translation/transcut/highres_decoder.py` | 可选 `highres_content` Decoder；不得冒充 legacy 加载 |
| `src/crossreg/translation/transcut/fullres_residual_decoder.py` | 可选 `fullres_residual` Decoder；源图直连并学习条件残差 |
| `src/crossreg/translation/transcut/conditional_discriminator.py` | 共享 Projection PatchGAN 与模态条件打分 |
| `src/crossreg/translation/cut/patchnce.py` | PatchNCE 损失本体，不负责 GAN 判别 |
| `src/crossreg/registration/transmorph/model.py` | SpatialTransformer、VecInt 与 TransMorph 主模型 |
| `src/crossreg/registration/transmorph/conditioned_model.py` | 主线：在原版 TransMorph joint patch 后注入 fixed/moving ID |
| `src/crossreg/registration/transmorph/deprecated_cross_attn_head.py` | 弃用：仅复现历史 Cross-Attention Stage 2 |
| `src/crossreg/registration/visualization.py` | 配准验证固定样例收集、flow/EPE综合图导出及快照轮转；供训练/评估入口复用 |
| `src/crossreg/registration/transmorph/cross_attn_head.py` | 弃用路径的兼容导入，新代码禁止使用 |
| `src/crossreg/pipeline/inference_v2.py` | 当前 Stage 3 推理契约；旧 `inference.py` 仅为早期基线 |
| `tests/test_translation_contracts.py` | 翻译模型、条件 ID、Decoder/checkpoint 契约 |
| `tests/test_registration_contracts.py` | flow 方向、warp、resize、Stage 2/3 数值契约 |

配置文件只保存可复用默认值，Shell 脚本固化一次运行的参数组合，输出目录的
`run_config.json` 保存最终解析结果。三者发生冲突时，以 `run_config.json`
描述已经发生的运行，以本工程契约判断该运行是否有效。

## 训练与推理链路

本节是项目的核心契约。修改 TransCUT、扰动、Stage 2、Reg-Head 或
Stage 3 时，必须先核对本节，并运行 `tests/test_registration_contracts.py`。

项目最终目标是训练直接跨模态配准模型，而不是部署模态生成器：

```text
Stage 1: TransCUT scaffold pre-training
  real n1/src_id -> CLN-Swin Encoder -> shared features
                                      -> AdaIN Decoder/tgt_id -> n2_fake
  必须满足：n1 与 n2_fake 几何对齐；Decoder 只在训练脚手架中使用。

Stage 2: synthetic supervised cross-modal registration
  从对齐的 (n1, n2_fake) 采样 backward-sampling displacement D_gt；
  构造真正跨模态的 (moving, fixed)，且必须满足：
      warp(moving, D_gt) ≈ fixed
  [fixed, moving] channel concat -> original TransMorph Patch Embedding
  (fixed_id, moving_id) -> ordered pair conditioner -> CLN on joint patches
  joint patches -> original Swin + skips + Decoder -> D_pred
  Loss = MSE(D_pred, D_gt) + lambda * Grad(D_pred)

Stage 3: direct real-image inference
  real fixed/moving + their IDs -> ModalityConditionedTransMorph -> D_pred
  warped = warp(moving, D_pred)

  理想情况下，warped 应该与 Fixed 完成对齐
```

### Stage 1：TransCUT 脚手架

TransCUT 有两个产物：共享 Encoder，以及仅用于构造 Stage 2 数据和视觉
诊断的 Decoder。生成图不要求成为可诊断的虚拟染色结果，但必须保持用于
配准的坐标与结构。若某一转换方向生成、删除或移动明显结构，该方向不得
进入 Stage 2，即使 GAN/NCE loss 数值正常。

Source ID 通过 CLN 进入 Encoder，Target ID 通过 AdaIN 进入 Decoder。
Stage 2 使用 Stage 1 的 Swin/CLN/modality embedding 权重初始化
`ModalityConditionedTransMorph`，但配准训练中允许主干以较小学习率
微调。这是参数迁移，不是将 Stage 1 Encoder 作为两个冻结黑盒
分别运行。

### Stage 2：无需显式逆场的双向构造

设 `n1` 与 `n2_fake` 几何对齐，`D_gt` 是本项目约定的 backward-sampling
displacement。当前实现按样本随机选择以下一种：

```text
# 源模态作为 moving
moving = n1
fixed  = warp(n2_fake, D_gt)
moving_id = src_id
fixed_id  = tgt_id

# 目标伪图作为 moving
moving = n2_fake
fixed  = warp(n1, D_gt)
moving_id = tgt_id
fixed_id  = src_id
```

两种构造都直接满足 `warp(moving,D_gt)≈fixed`，因此不需要显式计算
`D_gt` 的逆场。只有先用 `D` 生成 perturbed、再要求把 perturbed 恢复到
未扰动 base 时，监督标签才需要 `D^{-1}`；当前 Stage 2 不使用这种构造。

### Stage 2 离线数据契约

正式规模优先使用离线入口，在线入口只保留用于短实验和历史复现。离线
构建顺序必须是 `source -> TransCUT -> aligned fake -> 对 fixed 施加 D_gt`，
禁止改成“先形变再翻译”，否则生成器不满足交换性时会额外破坏监督标签。
HEMIT正式入口先将原始patch变为320×320画布，在该画布完成转换与扰动，
最后同步中心裁剪图像和flow为256×256。每侧32 px上下文大于当前15 px最大
位移，避免 `grid_sample` 越界填零形成可泄漏位移方向的人工黑边。

每条 manifest 记录必须包含：

```text
moving_path / fixed_path       RGB PNG，读取后恢复到 [-1,1]
flow_path                      float16 NPY，加载后转 float32
valid_mask_path                有效采样区域
moving_id / fixed_id           checkpoint 模态注册表中的整数 ID
pair_direction                 source-moving 或 target-moving
flow                           shape=(2,H,W)，顺序严格为 (dy,dx)，单位为像素
```

`valid_mask` 必须在最终256×256裁剪坐标中根据裁剪后的flow重新计算，而不是
沿用大画布mask。flow MSE只监督mask内可由moving观测的对应；Grad平滑正则
仍作用于完整预测场。大画布负责消除可见的填零捷径，mask负责排除最终裁剪
边缘不可观测的对应，两者不能互相替代。

构建器以 `sample_id` 生成确定路径；四个文件全部存在才视为完成。重复执行时
复用完整样本，缺少任一文件则重新生成该样本。`dataset_config_<split>.json`
固定 checkpoint SHA256、预处理、扰动参数与随机种子；这些字段变化时拒绝
在同一目录恢复，避免混合两套数据语义。

`--save-diagnostics` 仅用于小规模预览，额外保存扰动前的 aligned target；
正式60k入口默认关闭，避免重复保存中间图。诊断图脚本按行显示 source、
aligned target、warped fixed、扰动差异、flow幅值/方向、dy、dx和valid mask。

当前正式首轮配置为12个方向各2,000对，共24,000 train；paired test 固定
构建12个方向各64对，共768 val。保留的60k脚本用于后续扩大规模，而不是
首轮默认入口。`alternating` 在每个 translation direction
内交替两种 moving/fixed 角色，使最终12个配准方向严格平衡。离线化只消除
训练时生成器开销并固定监督数据，不修复 TransCUT 自身的结构生成错误。
24k脚本默认将四个source ID分配到GPU 4–7，每个分片写独立config/manifest
和日志，文件sample ID互斥；仅当四个子进程全部成功后才合并canonical
`manifest_train.csv`和`dataset_config_train.json`。任何单个分片失败都必须
使父脚本失败，不得使用未合并的目录启动训练。

Conditioned TransMorph 的入参顺序固定为：

```python
flow_pred = model.predict_flow(moving, fixed, moving_ids, fixed_ids)
# 内部联合输入始终为 concat([fixed, moving], dim=1)
```

HEMIT 主线的每张图像固定为 3 通道，因此 joint input 固定为
`(B,6,H,W)`。灰度 marker 在离线/数据入口复制为三通道，禁止让同一
模型的 pair input 在 2/4/6 通道之间变化。

Stage 1 权重迁移契约：

```text
Swin layers / norm / RPE       -> 严格同形复制
CLN + modality embedding       -> 严格同形复制
PatchEmbed C-channel weight    -> fixed/moving 两半各 0.5 * weight
Pair fixed/moving projections  -> 新增，各初始化为 0.5 * I
Original TransMorph Decoder    -> 保留原版结构，不使用 TransCUT Decoder
```

`--registration-model conditioned_transmorph` 是新训练默认值。
`deprecated_cross_attention` 只允许复现 T10/T11 等历史实验；其同步 K/V
空间置换不变，无法把 moving 匹配坐标传给 Decoder，不得恢复为主线。
历史 checkpoint 缺少 `registration_model` 字段时，推理端按弃用路径
加载，避免静默改变语义。新 checkpoint 必须写入：

```text
registration_model = conditioned_transmorph
```

`PairedImageFolderDataset` 输出 `[0,1]`，但 Stage 1 Encoder/Decoder 的输入
契约是 `[-1,1]`；`train_synthetic_supervised.py` 必须先执行 `2*x-1`，
禁止将 Dataset tensor 直接送入冻结的 Stage 1 Encoder。该入口可通过
`--generator-ckpt` 固定伪图生成器，同时用 `--transcut-ckpt` 和
`--encoder-init checkpoint|random` 独立选择待评估 Encoder；默认不传时
仍由同一个 Stage 1 checkpoint 完成生成与编码。

禁止重新写成 `fake/perturbed_fake` 两路相同 target ID。那会退化为同模态
配准，无法训练项目所需的跨模态模型。

当前 Reg-Head **直接回归 displacement**，checkpoint 写入：

```text
flow_parameterization = displacement
registration_model = conditioned_transmorph
```

Stage 3 对该输出直接执行 `warp`，不得再次通过 `VecInt` 积分。只有未来
明确将 Reg-Head 改为监督 stationary velocity 时，才可将参数化设置为
`velocity` 并在推理端执行一次 `VecInt`。

### Stage 3：直接跨模态推理

部署入口是 `src/crossreg/pipeline/inference_v2.py` 中的
`build_stage3_from_checkpoints()`；新 checkpoint 返回
`ConditionedStage3Inference`，历史 checkpoint 才返回 `Stage3Inference`。
调用必须显式传入 moving/fixed 和各自模态 ID：

```python
warped, flow = model(
    moving, fixed,
    moving_ids=moving_ids,
    fixed_ids=fixed_ids,
)
```

返回关系始终为：

```text
warped = warp(moving, flow)
warped ≈ fixed
```

`build_stage3_from_checkpoints()` 必须根据 Stage 2 checkpoint 的
`registration_model` 重建完整 `ModalityConditionedTransMorph`并严格加载。
不得只加载 Swin 或遗漏 pair conditioner。

`pipeline/inference.py` 中的 `CUT -> TransMorph` 是早期“先翻译再配准”基线，
不是本文档定义的最终 Stage 3 部署链路。最终模型以 `inference_v2.py` 为准。

### 链路修改的强制检查

任何涉及 moving/fixed、扰动或 flow 的修改必须同时满足：

1. 用非零已知位移验证 `warp(moving,flow_gt)≈fixed`；
2. 检查 `flow[0]=dy`、`flow[1]=dx`；
3. 检查 resize 后位移像素单位同步缩放；
4. 检查两路 Encoder 使用各自稳定 modality ID；
5. 检查联合通道顺序为 `[fixed,moving]`，函数入参为
   `(moving,fixed,moving_ids,fixed_ids)`；
6. 检查 checkpoint 的 flow parameterization 与推理一致；
7. 导出 moving、fixed、GT-warped、pred-warped、flow 和差异图；
8. 运行：

```bash
python -m pytest -q tests/test_registration_contracts.py
```

9. 在固定模态方向的验证 batch 内打乱 moving 特征。有效的双图配准模型
   必须出现明确性能下降；若只打乱 fixed 才下降，说明配准模型退化为
   fixed-only 位移预测，不能把较低 EPE 解释为跨图像配准成功。

## 核心数据与形变契约

所有新代码必须遵守以下定义：

```text
moving: 待变换图像
fixed:  参考图像
flow:   backward-sampling displacement，shape=(2,H,W)
flow[0]: dy
flow[1]: dx
warped = warp(moving, flow)
目标关系：warped ≈ fixed
```

`PairedImageFolderDataset` 和 `MultiModalityPairedDataset` 返回命名字典：

```python
{
    "moving": Tensor,
    "fixed": Tensor,
    "flow": Tensor,          # 可选
    "valid_mask": Tensor,    # 可选
    "moving_path": str,
    "fixed_path": str,
}
```

禁止在新代码中依赖 tuple 下标推断 flow/mask。flow resize 时必须同时缩放位移像素单位：`dy` 按高度比例缩放，`dx` 按宽度比例缩放。

TransMorph 推荐调用方式：

```python
warped, flow, velocity = model(moving, fixed)
```

为兼容现有代码，模型仍接受拼接的 `[fixed, moving]`，但新代码不应继续使用该形式。

## 已完成与已验证

| 模块 | 当前状态 |
|---|---|
| CUT | 独立训练、完整 checkpoint、恢复训练及推理加载已通过合成数据验证 |
| TransCUT | batch modality ID、style 优化、PatchNCE、原尺寸输出已通过单步训练测试 |
| 翻译 Dataset | CUT/TransCUT 默认 unpaired，可用 `--pairing-mode paired` 切换 |
| 扰动生成 | 统一 `(dy,dx)`；Stage 2 用 `exp(v)` 得到平滑 displacement，并以直接双向构造避免求逆 |
| Dataset | 统一字典接口；flow 格式检查与 resize 像素单位缩放 |
| 数据预检查 | 独立 `validation.py` + `validate_dataset.py`，不耦合训练 |
| 配置 | CUT/TransMorph 支持 YAML 默认值、CLI 覆盖及 resolved config 保存 |
| TransMorph | 已修复错误扭曲 fixed 的问题；现在明确执行 `warp(moving, flow)` |
| Conditioned TransMorph | 复用原版 joint Swin/skip/Decoder，仅新增有序模态对 CLN |
| Cross-Attention Head | 已弃用；仅保留历史 checkpoint 与实验复现 |
| 评估 | flow/mask 字段、严格 checkpoint 加载和最终 warped 指标已修复 |
| 模态分析 | 手工特征、ResNet18、PCA、t-SNE、雷达图和 domain-gap 分析可用 |

当前自动测试：

```bash
/data2/xujr/conda-envs/crossreg/bin/python -m pytest -p no:cacheprovider -q
# 包含 Stage 2 输入范围、Stage 1 权重迁移、Conditioned TransMorph
# 及历史 checkpoint 兼容契约；通过数以当次 pytest 输出为准。
```

已执行的运行验证：

- CUT CPU 小数据训练与 checkpoint 恢复成功。
- TransMorph CPU 合成数据两轮训练成功。
- CUT 与 TransMorph checkpoint 严格加载后评估成功。
- 数值测试覆盖 identity flow、已知 `dx` 方向、flow resize、moving/fixed 防反转、互逆速度场、Stage 2 双向跨模态 pair 和 Stage 3 模态 ID。

以上仅证明工程链路和数值契约成立，不代表真实数据上的模型效果已经得到验证。

## 功能状态与防误用边界

本节是长期维护的事实来源。代码中“存在某个参数”不代表项目主路线已经
接受该假设；启动实验前必须同时检查本节、启动脚本和输出目录中的
`run_config.json`。若三者不一致，以停止训练并核对为先，禁止根据参数名
自行推断用途。

### 已添加功能的状态清单

| 功能 | 位置/入口 | 状态与用途 | 使用边界 |
|---|---|---|---|
| paired/unpaired Dataset | `data/translation.py`、`--pairing-mode` | 通用能力；CUT/TransCUT 默认 unpaired | paired 依赖公共文件 stem，并共享几何增强；不得因数据本身配对就自动切换 |
| 稳定模态注册表 | `data/modalities.py`、`--modality-config` | 主线使用；固定 ID、名称、路径和通道数 | ID 必须连续稳定；更换顺序不能只改目录顺序 |
| 新模态扩展加载 | `--init-checkpoint --expand-modalities` | 可选迁移能力；按名称复制旧 embedding，新域行随机初始化 | 不是“零成本加入模态”；必须重置优化器并重新训练新域 |
| Projection 条件判别器 | `conditional_discriminator.py` | 共享卷积主体，用目标 ID 调整判别分数 | 提供条件不等于训练时一定会使用条件 |
| 条件反例判别 | `--lambda-D-mismatch` | 可选：真实图配错误 ID 时作为负例 | 增加判别器任务和计算；不能当作内容保持或生物监督 |
| 高分辨率内容 Decoder | `highres_decoder.py`、`--decoder-variant highres_content` | 可选架构；向最后两级上采样提供共享、归一化的源图内容特征 | 不改变 CLN、目标 AdaIN、域数或 loss；checkpoint 记录 variant 并拒绝交叉加载 |
| 全分辨率残差 Decoder | `fullres_residual_decoder.py`、`--decoder-variant fullres_residual` | 可选架构；源图直达输出，只学习目标条件残差，同时使用最细 Swin/CLN 特征 | 不是新 loss，也不使用分域 head；初始化为恒等映射 |
| 独立 loss 权重 | `--lambda-GAN/NCE/identity/...` | 主线能力；避免一个参数隐式控制多个 loss | 改权重必须形成独立实验记录，不能覆盖旧输出目录 |
| 配对重建锚点 | `--lambda-paired` | 可选：生成图与空间对齐真实目标的 L1 | 默认 0；可用于全 paired，或与 `--paired-anchor-manifest`、正的 `--paired-anchor-probability` 联用；loss 只作用于掩码标记的对齐样本 |
| 稀疏配对锚点 | `--paired-anchor-manifest`、`--paired-anchor-probability` | 在主 unpaired Dataset 中按概率插入少量同 stem、共享几何增强的配对样本 | 两项必须联用且要求 `lambda_paired>0`；概率为 0 时历史 unpaired 行为不变；日志 `Anchor` 记录实际比例 |
| 轻量 cycle | `--lambda-cycle` | 实验性、默认关闭 | 隐含近似可逆假设；CLI 非零时还必须显式给出 `--allow-experimental-cycle` |
| 多尺度梯度 loss | `--lambda-structure` | 实验性、默认关闭 | 只比较梯度幅值；不是几何等变 loss |
| 分域/分方向诊断 | `metrics.jsonl`、epoch 日志 | 仅观测，不改变梯度或采样 | 指标异常只用于定位问题，不能据此自动删域、加权或停止 |
| 暗图告警 | `--collapse-dark-gap` | 仅告警 | marker 真实目标本来可能很暗；告警不等于样本无效或模型必然塌缩 |
| 固定样例导出 | `samples/epoch_*` | 主线诊断；同时保留原图和全部目标域结果 | 视觉样例不能替代分域统计，固定样例也可能恰好是阴性 patch |
| HEMIT 5% 信号清单 | `prepare_hemit_signal_manifests.py`、`train_transcut_hemit_4domain_signal05.sh` | panCK/CD3 训练池独立要求信号面积不低于 5%，模型和 loss 不变 | 训练仍为 unpaired；清单不能描述为完整 marker 分布 |
| HEMIT 严格配对样例 | `--fixed-sample-manifest` | 同一 test patch 同时导出四域真实参考和 12 个有向转换 | 当前清单要求 DAPI、panCK、CD3 均不低于 5%；仅用于观察，不向训练提供配对监督 |
| 条件判别器探针 | `analyze_transcut_discriminator.py` | 对真实配对图做 tile/pixel shuffle、模糊和局部对比度扰动，测正确/错误 ID 间隔与 marker saliency | 只读诊断，不是生成质量指标；诊断结论记录到实验账本 |
| checkpoint 清理 | `--keep-*`、`--milestone-freq` | 主线工程能力 | 最近项会轮转删除，但周期里程碑、latest 和 final 保留 |
| DDP | `torchrun` | 已验证；梯度与 epoch 统计跨 rank 同步 | `--batch-size` 是每卡 batch；不会自动缩放学习率 |
| DataLoader 加速 | workers、persistent workers、prefetch、pin/non-blocking | 已启用在 HEMIT 脚本 | 只改变吞吐，不应被描述为模型效果优化 |
| 独立数据预检查 | `validate_dataset.py` | 只读、独立运行 | 当前不自动接入训练，也不修改或清洗数据 |
| 合成扰动 | `data/perturbation.py` | Stage 2 数据生成能力 | flow 统一为 `(dy,dx)`；与 Stage 1 翻译 loss 无关 |
| Stage 2 双向构造 | `train_synthetic_supervised.py` | 已修复形变方向契约 | 使用直接生成的正/负速度场，不要求从 displacement 数值求逆 |
| Stage 3 推理 | `pipeline/inference_v2.py` | 当前正式链路定义 | 旧 `pipeline/inference.py` 只是早期基线，不能混作最终部署流程 |

每次 TransCUT 启动会在日志打印 `Effective training objective`，并在输出
目录写入 `run_config.json`，其中保存完整 CLI、模型配置、模态顺序、
world size、全局 batch size 和实际 loss 权重。Checkpoint 内也保存模型
配置和模态名称。正式结果必须连同这两个配置来源一起保留。

## 训练入口

```bash
# CUT：两模态翻译
python scripts/train_translation.py \
  --dataroot /path/to/trainA_trainB_root \
  --pairing-mode unpaired \
  --name example --save-dir /path/to/output --device cpu

# CUT：也可从 YAML 启动，显式 CLI 参数优先
python scripts/train_translation.py \
  --config configs/translation/cut_example.yaml \
  --dataroot /real/data/path --save-dir /real/output/path

# TransCUT：N-to-N 翻译（Stage 1）
python scripts/train_transcut.py \
  --dataroot /path/to/modality_dirs \
  --save-dir /path/to/output --device cpu

# TransCUT：单机多卡 DDP；每卡 batch-size=1 时全局 batch-size=GPU数
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc-per-node=2 \
  scripts/train_transcut.py \
  --modality-config /path/to/modalities.yaml \
  --save-dir /path/to/output --batch-size 1 --device cuda

# 四模态正式预备实验（默认 GPU 0，可用 GPU_ID/OUTPUT_DIR 覆盖）
bash scripts/train_transcut_4modal_formal.sh

# 四模态结构保持实验：从头训练，独立 GAN/NCE/identity/structure 权重
bash scripts/train_transcut_4modal_structure.sh

# HEMIT H&E/DAPI/panCK/CD3 四域优化入口；默认 GPU 4-7、每卡 batch=4
bash scripts/train_transcut_hemit_4domain.sh

# HEMIT 四域 5% marker 信号数据消融；loss/模型不变，样例为严格配对 test
bash scripts/train_transcut_hemit_4domain_signal05.sh

# HEMIT signal05 + full-resolution residual Decoder
bash scripts/train_transcut_hemit_4domain_fullres_signal05.sh

# HEMIT 四域 paired 能力上限/可行性控制
bash scripts/train_transcut_hemit_4domain_paired_upper_bound.sh

# HEMIT 四域 5% 配对锚点 + 95% unpaired，50 epoch
bash scripts/train_transcut_hemit_4domain_semipaired05.sh

# 合成监督 Conditioned TransMorph 配准（Stage 2）
python scripts/train_synthetic_supervised.py \
  --transcut-ckpt /path/to/transcut.pth \
  --data-dir /path/to/paired_images \
  --src-modality 0 --tgt-modality 1 \
  --pair-direction random \
  --registration-model conditioned_transmorph \
  --lr 1e-4 --backbone-lr 1e-5 \
  --save-dir /path/to/output --device cpu

# 首轮正式 Stage 2：可断点恢复地构建24k train + 768 val
bash scripts/build_stage2_hemit_offline_24k.sh

# 正式 Stage 2：离线四域 Conditioned TransMorph，默认20 epoch
bash scripts/train_stage2_hemit_offline_4domain.sh

# TransMorph 同模态无监督配准
python scripts/train_registration.py \
  --train-dir /path/to/train \
  --val-dir /path/to/val \
  --save-dir /path/to/output --device cpu

# 独立数据预检查（不修改数据）
python scripts/validate_dataset.py \
  --task translation --data-dir /path/to/data \
  --pairing-mode unpaired --output /path/to/report.json
```

离线 Stage 2 每次验证默认从每个已观察方向固定导出1例，综合图位于
`<save-dir>/samples/epoch_*.png`，同时维护 `latest.png` 和按验证
`valid_epe` 选择的 `best.png`。每行依次为 moving、fixed、预测 warp、
GT flow、预测 flow、endpoint-error 图和有效监督 mask。编号快照默认只保留
最近5份；`latest/best` 不参与轮转。可用 `--val-samples-per-direction 0`
关闭，或用 `--val-sample-freq` 与 `--keep-val-sample-snapshots` 控制频率和
磁盘占用。该导出复用验证推理结果，不会进行第二遍模型预测。

通用训练/评估入口的 `--num-workers` 默认是`0`，适合受限环境；离线
Stage 2入口默认为`4`。正式服务器脚本会显式覆盖该值，实际配置以输出目录
的`run_config.json`为准。

`train_transcut.py` 在普通 `python` 启动时保持单进程行为；检测到
`torchrun` 提供的 `WORLD_SIZE>1` 后自动启用单机 DDP。每个 rank 绑定自己的
`LOCAL_RANK`，使用 `DistributedSampler` 划分数据，并同步生成器、条件判别器
和 PatchNCE `netF` 的梯度。`netF` 在包装 DDP 前完成懒初始化。只有 rank 0
写日志、固定样例、模态注册表和 checkpoint；epoch loss、逐目标域统计、
逐方向统计及方向计数在所有 rank 间求和。DDP checkpoint 不带 `module.`
前缀，可直接用于单卡恢复和推理。
`--batch-size` 是每个 rank 的 batch size；全局 batch size 为
`batch_size × WORLD_SIZE`，当前实现不自动按全局 batch size 缩放学习率。
当 `--num-workers>0` 时，DataLoader 自动使用 persistent workers 和
`--prefetch-factor` 预取；CUDA 搬运使用 pinned memory 与 non-blocking copy。

### TransCUT 输出目录契约

- `run_config.json`：最终 CLI、模型配置、模态顺序、world/global batch。
- `metrics.jsonl`：逐 epoch、逐目标域和逐方向的可机读统计。
- `train.log`：人类可读日志与告警；告警不会自动删样本或改变权重。
- `samples/epoch_*`：固定原图、真实配对参考（若提供）及全部转换方向。
- `latest_checkpoint.pth`：恢复入口；不参与轮转删除。
- `transcut_epoch_*.pth`：普通快照受 `--keep-epoch-checkpoints` 管理，
  `--milestone-freq` 指定的里程碑除外。
- `transcut_final.pth`：正常完成后的最终状态；不参与轮转删除。

不同 `decoder_variant` 的 checkpoint 禁止交叉加载。实验采用哪个入口、
参数、结果和结论，全部记录在 `EXPERIMENTS.md`，不在本文维护。

## 环境

- Conda 环境：`crossreg`，Python 3.10。
- Python：`/data2/xujr/conda-envs/crossreg/bin/python`。
- 激活：`source /data2/xujr/miniconda3/etc/profile.d/conda.sh && conda activate crossreg`。
- CPU 回归环境：PyTorch `2.12.0+cpu`。
- GPU/DDP 验证环境：`/data2/xujr/conda-envs/transmorph`，两张 V100S，NCCL。
- `requirements.txt` 是最低依赖清单，分析脚本还使用 pandas、matplotlib、seaborn、scikit-learn、tqdm 和 OpenCV。

## 开发约定

- 翻译、扰动生成、配准训练和推理保持解耦。
- CUT/TransCUT 默认使用非配对数据；显式 `--pairing-mode paired` 时所有样本要求公共 stem 和共享几何增强。稀疏锚点模式仅对被抽中的样本施加这一契约，并在 batch 中传递 `is_paired` 掩码。
- 训练和推理使用不同入口，但共享严格的 checkpoint loader。
- 默认 PNG；格式特定逻辑限制在 data/io 层。
- 评估时 checkpoint 缺失或结构不匹配必须失败，禁止静默随机初始化。
- 新增配准逻辑必须同时提供数值契约测试，不能只断言 tensor shape。
- 真实数据路径通过配置传入，不在模型代码中硬编码。
