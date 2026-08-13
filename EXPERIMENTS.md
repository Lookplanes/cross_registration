# CrossReg Experiment Ledger

> **用途：实验事实账本。** 成功、失败、中断和仅诊断的尝试都必须记录。
> 工程接口与文件职责见 `README-self.md`，论文范围见
> `AAA-document/0725Report/0725.md`。

## 记录规则

每次实验使用独立 ID 和输出目录，至少记录：日期、问题、相对基线的唯一
变化、数据、状态、观察、结论和产物。loss 数值下降不能单独写成“成功”；
模态转换必须结合固定样例和分域统计判断。运行中的瞬时 epoch 不持续写入
本文，结束或异常中止后再补最终状态。

状态只使用：`运行中`、`完成/有收益`、`完成/无明确收益`、`失败`、
`中断`、`仅诊断`。

## 实验索引

| ID | 日期 | 唯一主要变化 | 状态 | 简要结论 |
|---|---|---|---|---|
| T01 | 2026-07-14 | 早期异质四域 TransCUT | 完成/无明确收益 | 首版中断、修复版完成，但数据域不构成可信论文实验 |
| T02 | 2026-07-17 | 增加多尺度梯度 structure loss | 完成/无明确收益 | loss 可训练，未解决跨域结构失真 |
| T03 | 2026-07-29 | HEMIT 四域 legacy baseline | 中断 | 暴露 marker 稀疏与目标域输出问题 |
| T04 | 2026-07-30 | 条件反例判别 | 完成/有收益 | 强化模态条件，未使生成质量达到 Stage 2 要求 |
| T05 | 2026-07-31 | `highres_content` Decoder | 完成/无明确收益 | 高分辨率浅层 skip 仍不足以稳定保持结构 |
| T06 | 2026-07-31 | signal05 marker 清单，短训 20 epoch | 完成/无明确收益 | 稀疏性是影响因素，但不是唯一瓶颈 |
| T07 | 2026-08-01 | `fullres_residual` Decoder + signal05 | **失败** | 直接源图路径形成复制捷径，几乎没有完成目标域转换 |
| T08 | 2026-08-01 | 四域 paired L1 能力上限 | **完成/有收益** | 证明模型能响应目标监督；纯 unpaired 欠约束是主要问题之一 |
| T09 | 2026-08-01 | 5% 配对锚点 + 95% unpaired，50 epoch | **完成/有收益** | 少量锚点显著推动目标域转换，但结构伪影仍不满足 Stage 2 |
| T10 | 2026-08-02 | 短 Stage 2 冻结 Encoder 对照 | **完成/无明确收益** | 当前 RegHead 未把 T09 表征优势转化为更低 flow EPE |
| T11 | 2026-08-02 | RegHead 关闭 fixed-query residual | **完成/无明确收益** | 消除了完全 fixed-only 直通，但仍未学到样本级两图对应 |
| T12 | 2026-08-02 | 原版 TransMorph 联合主干 + 成对模态条件 | **完成/有收益** | flow EPE 显著下降，且打乱任一输入均严重恶化，已学到样本级两图对应 |
| T13 | 2026-08-02 | 单模型联合训练 HEMIT 四域 12 个方向 | **完成/有收益** | 总体 EPE 改善 47.1%，12 个方向全部优于 zero-flow，但 CD3 相关方向较弱 |
| T14 | 2026-08-12 | 大画布扰动的24k离线Stage 2数据 | **完成/有收益** | 24,000 train＋768 val穷举审计通过，12方向严格平衡且无填零边缘捷径 |
| T15 | 2026-08-13 | T14离线数据训练四域Conditioned TransMorph | **完成/无明确收益** | 合成验证改善25.1%，但同一真实配对测试仅改善4.7%，不优于T13 |
| D01 | 2026-07-31 | NCE 模态 ID 条件检查 | 仅诊断 | 没有证据表明不同 CLN ID 是当前主要问题 |
| D02 | 2026-07-31 | 配对真实跨域 MIND 分布 | 仅诊断 | 不支持把全局 MIND loss 加入主训练 |
| D03 | 2026-07-31 | 条件判别器扰动探针 | 仅诊断 | 不支持用“空间无序纹理 D”替换 PatchGAN |
| D04 | 2026-08-01 | Encoder 跨模态空间对应探针 | 仅诊断 | T09 学到显著对应但对已知平移仍不够稳健；epoch 30 优于 final |
| D05 | 2026-08-02 | RegHead moving 空间置换不变性 | 仅诊断 | 当前 attention 丢弃 moving 坐标，结构上无法直接表达位移对应 |
| D06 | 2026-08-03 | 四域真实配对＋已知/零扰动零微调测试 | 仅诊断 | H&E↔DAPI 可迁移，panCK 有限，CD3 相关方向失败；无扰动时存在明显虚假位移 |

## 详细记录

### T01：早期异质四域 TransCUT

- 数据域：2PM、Confocal、brightfield H&E、Cell Painting，unpaired。
- 目的：验证一个条件化模型能否完成四域训练及全部有向转换。
- 产物：
  - `/data2/xujr/crossreg/transcut_4modal_formal_20260714`
  - `/data2/xujr/crossreg/transcut_4modal_formal_v2`
- 结果：首版训练中判别器长期为零并在 epoch 51 中断，暴露训练代码问题；
  修复后的 v2 完成 100 epoch，但视觉转换仍不理想。
- 结论：工程链路得到修复，但这些数据的生物对象、尺度和成像语义差异过
  大，不再作为当前论文路线的数据依据。

### T02：多尺度梯度 structure loss

- 唯一主要变化：在既有 GAN/NCE/identity 外启用多尺度梯度幅值约束。
- 产物：`/data2/xujr/crossreg/transcut_4modal_structure_20260717`
- 结果：完成 100 epoch，loss 数值正常；没有形成足以支持结构明显改善的
  视觉证据。
- 结论：不同模态的真实边缘表达并不必然一致，该约束主观性较强，默认
  保持关闭；它也不是几何等变 loss。

### T03：HEMIT 四域 legacy baseline

- 数据域：H&E、DAPI、panCK、CD3；训练使用 unpaired 关系。
- 产物：`/data2/xujr/crossreg/transcut_hemit_4domain_20260729`
- 结果：运行至 epoch 97 后异常中断；已有样例显示整体转换不理想，稀疏
  marker 方向尤为明显。
- 结论：HEMIT 提供了合适的配对观察条件，但此 checkpoint 不具备可靠
  构造 Stage 2 训练伪图的能力。

### T04：条件反例判别

- 唯一主要变化：真实目标图配正确 ID 判真，配随机错误 ID 判假；
  `lambda_D_mismatch=1`，cycle 保持关闭。
- 产物：
  `/data2/xujr/crossreg/transcut_hemit_4domain_condneg_bs4_20260730_run2`
- 结果：完成 100 epoch，`D_mismatch` 可观测，判别器不再只接收正确标签；
  固定样例仍未达到可供 Stage 2 使用的结构与目标外观质量。
- 结论：这是针对条件判别退化的有效工程约束，但不是结构保持方案。

### T05：highres_content Decoder

- 唯一主要变化：在原有粗尺度 Decoder 后段融合共享、归一化的源图浅层
  高分辨率特征；loss 与四域设定不变。
- 产物：`/data2/xujr/crossreg/transcut_hemit_4domain_highres_20260731`
- 结果：完成 100 epoch；视觉结果仍存在结构改动和目标域质量问题。
- 结论：仅在上采样阶段补充 source detail skip 不足以解决问题。

### T06：signal05 数据分布消融

- 唯一主要变化：H&E/DAPI 保持原训练池，panCK/CD3 各自仅使用
  `signal_fraction >= 5%` 的 patch；训练仍为 unpaired。
- 固定观察：使用四域严格配对、所有 marker 均达到 5% 的 test patch。
- 产物：
  `/data2/xujr/crossreg/transcut_hemit_4domain_highres_signal05_short20_20260731`
- 结果：20 epoch 短训完成；样例仍不理想。
- 结论：marker 稀疏性会影响训练，但提高有效信号比例没有单独解决结构
  与风格转换问题，不能把失败完全归因于纯黑 patch。

### T07：fullres_residual Decoder

- 唯一主要变化：用 `fullres_residual` 替换 `highres_content`。源图直接
  连接输出，网络在完整分辨率学习目标 AdaIN 条件残差，并融合最细
  Swin/CLN 特征；没有新增 loss。
- 不变项：四域、12 个 unpaired 方向、signal05 清单、GAN/NCE/identity、
  条件反例判别、4 卡 DDP。
- 启动：2026-08-01，50 epoch 恒定学习率 + 50 epoch 线性衰减。
- 产物：
  `/data2/xujr/crossreg/transcut_hemit_4domain_fullres_signal05_20260801`
- 状态：100 epoch 正常完成；CPU 24 项回归测试和两卡 DDP 冒烟均通过，
  checkpoint、最终样例和指标文件完整。
- 最终数值：epoch 100 为 `G=0.8093`、`GAN=0.7073`、`NCE=0.1002`、
  `identity=0.0017`、`D=0.0812`。这些 loss 有限且训练稳定，但不代表转换
  成功。
- 分域证据：真实 H&E 的均值约 `0.740`，而 marker→H&E 生成图均值仅
  `0.031–0.057`；真实 marker 均值约 `0.028–0.053`，H&E→marker 生成图
  均值却约 `0.736–0.748`。输出统计主要跟随源域而非目标域。
- 固定样例证据：三组配对样例中，H&E→三个 marker 仍保持粉紫色 H&E，
  三个 marker→H&E 仍保持灰度荧光外观。sample 00 的 12 个方向上，最终
  `MAE(fake, source)` 平均仅为 `0.01767`；作为行为参照，T06 的
  highres 短训对应值为 `0.37936`。该比较只用于证明输出过度接近源图，
  不将更大的 MAE 本身解释为更好的转换。
- 结论：本实验在像素几何保持上形成了过强的直接捷径。PatchNCE 和
  identity 同样奖励源内容保留，而现有 GAN 没能推动残差跨越巨大的域
  外观差异，最终退化为近似复制源图。该 checkpoint **不得用于 Stage 2
  伪图构造**；`fullres_residual` 保留为失败消融实现，不作为下一版默认。

### T08：四域 paired 能力上限

- 要回答的问题：在目标 patch 明确给出的情况下，现有共享四域架构是否有
  能力学习跨域映射，从而区分“模型能力不足”和“纯 unpaired 欠约束”。
- 唯一训练目标变化：新增统一的
  `lambda_paired * L1(fake, aligned_real_target)`，权重为 10；该项默认关闭
  且 T08 运行时仅允许全 paired 模式。T09 后扩展为：unpaired 模式只有在
  显式提供配对锚点清单、采样概率和逐样本掩码时才允许启用。
- 模型：复用 `highres_content`、CLN/AdaIN、GAN/NCE/identity、条件反例
  判别器和四卡 DDP；没有模态白名单、方向权重或专用 head。
- 数据：HEMIT train 中四域同 stem 且 DAPI/panCK/CD3 signal fraction 均
  不低于 5% 的 1,466 个配对 patch；测试观察仍使用独立的 256 个配对
  test patch。训练增强对 source/target 使用相同 resize、flip 和 crop。
- 启动：10 epoch，从头训练；脚本
  `scripts/train_transcut_hemit_4domain_paired_upper_bound.sh`。
- 产物：
  `/data2/xujr/crossreg/transcut_hemit_4domain_paired_upper_bound_20260801`
- 状态：10 epoch 正常完成，用时约 13 分钟；24 项 CPU 回归和两卡 DDP
  冒烟通过，checkpoint、日志、指标和三组独立 test 固定样例完整。
- 最终训练数值：`G=2.3099`、`GAN=0.5102`、`NCE=0.4845`、
  `identity=0.0391`、加权 `paired=1.2762`、`D=0.1820`。
- 独立 test 的三组固定样例中，相比直接拿 source 当预测，所有 12 个方向
  的 `MAE(fake,target)` 均下降。H&E→DAPI/panCK/CD3 分别下降
  `98.1%/91.1%/94.6%`，marker→H&E 下降 `72.2%–86.5%`；marker 间
  下降 `10.0%–66.2%`。该指标会被 marker 大面积黑背景影响，只证明输出
  向配对目标移动，不能单独证明生物结构预测正确。
- 视觉观察：H&E→DAPI 已形成较合理的核信号外观；H&E→panCK/CD3 和
  marker 间能够切换亮度/稀疏模式；marker→H&E 能生成粉紫色组织外观。
  但输出仍有模糊、RGB 色边和明显网格/棋盘伪影，稀疏 marker→丰富 H&E
  也无法恢复源图未包含的组织信息。
- 结论：现有共享四域架构具备响应明确目标监督的基本表达能力，T03–T07
  的失败不能只归因于“模型完全不会生成”；纯 unpaired 下缺少样本级映射
  锚点是主要瓶颈之一。该 10 epoch checkpoint 是诊断性 upper-bound，
  **仍不得直接用于 Stage 2**。下一项应在保持大部分 unpaired 数据的同时，
  比较少量配对锚点比例，而不是继续盲改 Decoder。

### T09：5% 配对锚点半监督四域训练

- 要回答的问题：保持 TransCUT 多域统一结构和主要 unpaired 数据设定时，
  极少的空间配对监督能否避免纯 unpaired 映射欠约束，并改善固定配对 test
  上的目标外观与结构对应。
- 相对 T06/T08 的变化：主训练池仍为 signal05 的 unpaired 四域清单；从
  1,466 个合格配对 train patch 中用 seed 42 固定抽取 73 个（约 5%）作为
  唯一可见配对标签池，并以 5% 采样概率插入训练。配对 L1 权重为 10，
  只乘在这些样本的掩码上；其余样本仍只使用 GAN/NCE/identity。
- 模型保持 `highres_content`、CLN/AdaIN、统一条件判别器和四域 12 方向；
  不使用 cycle、MIND、几何等变 loss、方向白名单或模态专用 head。
- 训练：从头开始，25 epoch 恒定学习率 + 25 epoch 线性衰减，四卡 DDP，
  全局 batch 16。启动脚本：
  `scripts/train_transcut_hemit_4domain_semipaired05.sh`。
- 产物：
  `/data2/xujr/crossreg/transcut_hemit_4domain_semipaired05_50e_20260801`
- 状态：50 epoch 正常完成；25 项 CPU 回归与 2 卡 DDP 强制锚点短测
  通过。最终四个目标域实际 `Anchor` 比例为 `4.7%–5.3%`，总体
  `G=0.6595`、`GAN=0.4244`、`NCE=0.1409`、`identity=0.0206`、
  加权且按 batch 稀释后的 `paired=0.0735`、`D=0.1622`，没有数值异常。
- 固定三组独立 paired test 的 36 个方向样例：epoch 30 的平均
  `MAE(fake,target)=0.0731`，epoch 50 为 `0.0769`；直接使用 source 的
  基线为 `0.3847`。作为参照，全 paired T08 为 `0.0714`。该 MAE 会受到
  marker 大面积黑背景影响，只说明输出明显转向目标域分布。
- 视觉结论：H&E→marker 和 marker→H&E 都发生明确目标域转换，不再复制
  source，证明少量配对锚点对纯 unpaired 欠约束有实质帮助；但
  marker→H&E 仍虚构源图不存在的丰富组织纹理，并存在网格、涂抹和局部
  结构错配，marker 间部分方向主要学习亮度/稀疏度。epoch 30 的平均配对
  MAE 略优于 epoch 50，也说明继续拟合没有自动改善结构可靠性。
- 结论：T09 是正向诊断结果，但最终 checkpoint **仍不得直接用于 Stage 2
  伪图构造**。下一步应围绕“配对锚点比例/采样方式与结构可靠性的关系”做
  有限消融，而不是继续增加 epoch 或重改网络主体。

### T10：短 Stage 2 冻结 Encoder 对照

- 问题：T09-e30 的跨模态特征对应改善，能否直接帮助当前
  Cross-Attention RegHead 回归已知人工 displacement。
- 公平控制：三组都由固定的 T09-e30 Encoder/Decoder 生成完全相同的
  H&E→DAPI `n2_fake`、训练形变和验证形变；只替换冻结的注册 Encoder 为
  随机初始化、纯 unpaired T05 或 T09-e30。RegHead 初始化、seed、数据
  顺序、loss 和优化参数一致。
- 规模：单方向、单 seed；每组 `5×200=1000` 次更新，每个 epoch 在固定的
  256 个未见 HEMIT test source/flow 上验证。该规模只作可学习性探针，不是
  完整四域 Stage 2 训练。
- 运行前修复：`PairedImageFolderDataset` 返回 `[0,1]`，而 Stage 1 使用
  `[-1,1]`。旧入口此前直接把 `[0,1]` 送入冻结 Encoder/Decoder；现已统一
  执行 `2*x-1`，并新增独立 `--generator-ckpt`、随机 Encoder、固定验证
  EPE 和短迭代上限。27项回归测试通过。
- 最终验证：zero-flow EPE 为 `3.1097 px`；随机、T05、T09-e30 分别为
  `2.9005/2.8609/2.9160 px`，相对 zero-flow 改善
  `6.7%/8.0%/6.2%`；PCK@4 分别为 `76.58%/77.79%/76.63%`。
- 结论：三组只形成小幅且相近的改善，T09没有领先；本实验不支持“当前
  RegHead 已经能利用 T09 表征优势”。这不否定 T09 的生成与D04特征对应
  结果。由于只有单方向、单seed和短训练，也不能据此否定完整Stage 2；更
  直接的暴露是当前head/监督设置可能主要学习形变先验，尚未充分依赖输入
  对应。下一步应先检查预测flow与GT的样本相关性/打乱输入对照，而不是扩大
  训练规模。
- 输入依赖复查：在固定 `source-moving` 或 `target-moving` 方向后再做
  batch内特征打乱，保证打乱前后模态ID不变。三种Encoder中，打乱moving
  仅使EPE增加 `0.0000–0.0016 px`，置零moving也只增加
  `0.0022–0.0345 px`；而打乱fixed使EPE增加 `0.343–0.628 px`，并将
  `flow cosine` 从 `0.31–0.45` 降至约 `0.00`。整对输入一起打乱与只打乱
  fixed几乎相同。
- 修正结论：当前head不是单纯输出固定的平均flow，但几乎完全依赖fixed
  特征，未使用moving/fixed对应。直接原因候选包括Cross-Attention融合的
  `attention_output + fixed_query` 残差捷径，以及当前合成构造始终让fixed
  成为被人工warp的一支。T10不能作为跨图像配准成功证据；在消除fixed-only
  捷径前不得扩大正式Stage 2训练。
- 产物：`/data2/xujr/crossreg/stage2_encoder_probe_20260802`（233 MB）。

### T11：RegHead 去除 fixed-query residual

- 问题：T10 的 moving 无关退化是否主要由
  `Attention(F_fixed,F_moving,F_moving) + F_fixed` 中的单支 residual
  造成。
- 唯一主要变化：保持 T10 T09-e30 组的数据、伪图生成器、
  冻结 Encoder、seed、loss 及 `5×200` 次更新不变，设置
  `fusion_residual=none`。
- 验证：zero-flow EPE `3.1097 px`；最终 EPE `2.9657 px`，
  改善 `4.63%`，弱于 T10 旧 residual T09 组的 `2.9160 px / 6.2%`。
- 输入依赖：`source-moving` 正确输入 EPE `2.9169`，打乱 moving
  后 `2.9179`，置零 moving 后 `3.2260`；`target-moving` 分别为
  `3.0160/3.0171/3.2260`。两方向打乱 fixed 后均恶化至约
  `3.26–3.28`，flow cosine 降至约零。
- 结论：删除 residual 后，模型不再能完全不读 moving，因为置零
  moving 会明显恶化。但替换为同域的错误样本几乎不影响 EPE，
  表明 moving 当前只提供通用域特征/值字典，并未提供有效的
  fixed–moving 样本级对应。fixed residual 是真实捷径，但不是唯一
  原因；T11 仍不能视为跨图像配准成功。
- 产物：`/data2/xujr/crossreg/stage2_no_fixed_residual_t09_20260802`，
  包含 checkpoint、训练日志、配置与两个方向的输入依赖日志。
- 完整训练复查：取消每轮 200 次上限，完整训练
  `5×1181=5905` 次更新。验证 EPE 持续降至 `2.8097 px`，相对
  zero-flow 改善 `9.65%`，说明无 residual 模型能继续优化。但
  source-moving 正确/打乱 moving EPE 仍为 `2.7636/2.7645`，
  target-moving 为 `2.8612/2.8626`；延长训练没有建立样本级对应
  依赖。产物：
  `/data2/xujr/crossreg/stage2_no_fixed_residual_t09_full5_20260802`。

### T12：成对条件化 TransMorph

- 要回答的问题：在尽量复用原版 TransMorph 联合输入、Swin、Decoder 和
  RegHead 的前提下，仅增加 moving/fixed 模态条件，能否消除 T10/T11 的
  单支依赖并学习样本级对应。
- 数据与控制：沿用 T10/T11 的 T09 epoch 30 生成器、HEMIT train/val、
  H&E→DAPI、随机双向构造、seed 42 和固定 256 个验证样本；训练 20 个
  完整 epoch，每轮 1181 次更新。
- 结果：验证 EPE 从 zero-flow `3.1097 px` 降至 `1.0809 px`，改善
  `65.24%`；PCK@1/2/4 为 `61.46%/88.64%/97.76%`。20轮中验证 EPE
  持续下降，最佳为 epoch 20，未观察到后期反弹。
- 输入依赖：source-moving/target-moving 正确输入 EPE 分别为
  `1.0536/1.1138 px`；只打乱 moving 后升至 `4.9524/5.0178 px`，只打乱
  fixed 后均约 `5.58 px`。正确输入 flow cosine 为 `0.924/0.909`，打乱
  moving 后仅 `0.205/0.196`。因此本次改善确实依赖两图的样本级对应，
  不是 fixed-only 捷径或平均 flow。
- 边界：这是单一 H&E↔DAPI 方向、合成已知形变上的可学习性验证，证明
  新 Stage 2 结构解决了旧 RegHead 的核心退化，但尚不能代表真实错位、
  其他模态组合或完整四域配准性能。
- 启动脚本：`scripts/train_stage2_conditioned_t09_20e.sh`；诊断脚本：
  `scripts/analyze_stage2_input_dependence.py`。
- 产物：
  `/data2/xujr/crossreg/stage2_conditioned_transmorph_t09_e30_20e_20260802`。

### T13：四域联合 Stage 2

- 要回答的问题：T12 在单一 H&E↔DAPI 方向上的收益，能否扩展到 H&E、
  DAPI、panCK、CD3 的全部 12 个有向组合，并由同一个条件化 TransMorph
  统一学习。
- 数据：HEMIT `processed_1200_150_150_512` 的四个真实 source 域；目标图
  由固定 T09 epoch 30 按逐样本 target ID 生成。训练/验证仍使用隔离的
  train/val split。
- 规模：20 个完整 epoch，batch 16，每轮 18,896 个 source 样本和 1,181
  次更新；每轮验证固定 1,024 个样本并分别报告 12 个实际配准方向。
- 启动脚本：`scripts/train_stage2_conditioned_t09_4domain_20e.sh`。
- 产物：
  `/data2/xujr/crossreg/stage2_conditioned_transmorph_t09_4domain_20e_20260802`。
- 结果：20轮完整结束，总体验证 EPE 从 zero-flow `3.2228 px` 降至
  `1.7060 px`，改善 `47.07%`；PCK@1/2/4 为
  `35.02%/70.84%/93.51%`。验证 EPE 持续下降，epoch 20 为最佳。
- 逐方向：12个方向均优于各自 zero-flow。较强方向为 H&E→DAPI
  `1.2721 px`、DAPI→H&E `1.3768 px`、panCK→DAPI `1.3984 px`；较弱
  方向为 panCK→CD3 `2.1503 px`、H&E→CD3 `2.0794 px`、DAPI→CD3
  `1.9760 px`。弱方向仍有约 `32.7%–39.7%` 的 zero-flow 改善，并非完全
  失败。
- 与 T12 的边界：四域模型 H&E→DAPI 为 `1.2721 px`，弱于单方向 T12
  的约 `1.08 px`，符合单模型共享容量和每方向更新量减少的预期；但不能仅
  据此区分容量竞争与 TransCUT 假图质量影响。
- 输入依赖复查：在保持每个样本 `moving域→fixed域` ID 完全不变的条件下，
  只在同一方向组内交换图片，`99.41%` 的1024个验证样本实际发生交换。
  source-moving/target-moving 的正确 EPE 为 `1.7186/1.7324 px`；打乱
  moving 后升至 `3.2249/3.4045 px`，打乱 fixed 后升至
  `4.1118/4.1833 px`。正确 flow cosine 为 `0.818/0.807`，打乱 moving
  后降至 `0.377/0.328`，打乱 fixed 后约为零。四域低 EPE 因此不能解释为
  仅识别模态 ID、fixed-only 捷径或平均形变先验。
- 结论：同一个成对条件化 TransMorph 能联合学习四域12方向的合成已知
  形变，旧 RegHead 的结构性失败没有在四域训练中重现。下一步需要真实
  数据评估，不能把合成监督结果直接表述为真实配准性能。

### T14：大画布扰动的24k离线 Stage 2 数据

- 日期：2026-08-12至2026-08-13。
- 目的：固定完整HEMIT训练得到的TransCUT，将模态转换与已知扰动离线化，
  消除Stage 2训练时的生成器开销和每轮随机数据变化；同时避免扰动越界
  填零给配准模型泄漏flow方向。
- 生成契约：原始patch先resize到358并中心裁为320；TransCUT转换和速度场
  积分/warp都在320画布进行，再将源图、生成图、扰动图和flow同步中心裁为
  256。最终valid mask在256坐标系重算，flow MSE只监督mask内像素。
- 规模：train为12个translation方向各2,000对，共24,000；paired test来源
  的val为12个方向各64对，共768。每个translation方向内交替
  source-moving/target-moving，因此最终registration方向也严格平衡。
- 构建：`scripts/build_stage2_hemit_offline_24k.sh`，四张GPU按source ID分成
  4个互斥分片；一次早期单卡运行中断，其完整文件由分片0按sample ID复用。
  四分片最终均完成6,000条，canonical manifest只在全部成功后合并。
- 产物：
  `/data2/xujr/crossreg_data/hemit_stage2_offline_24k_v1`。
- 穷举审计：train/val的moving、fixed、gt_flow、valid_mask分别为
  `24,000×4`和`768×4`个文件；sample ID唯一，manifest与目录文件集合完全
  相等，无孤儿或`.tmp`文件。所有PNG/NPY均实际读取，图像为RGB 256²，flow
  为float16 `(2,256,256)`且无NaN/Inf，mask为二值256²且非空。
- 统计：train/val valid fraction均值为`0.9811/0.9812`；flow平均幅值均值为
  `3.012/2.987 px`。数据文件约`7.96+0.26 GiB`，目录总占用`8.23 GiB`，
  完成时文件系统剩余`87.43 GiB`。
- 审计产物：`audit_train.json`、`audit_val.json`和
  `completion_summary.json`。结论：数据构建与工程契约通过，可作为下一版
  Conditioned TransMorph正式训练输入；这不等价于证明生成图生物学正确。

### T15：24k离线四域 Stage 2 正式训练

- 日期：2026-08-13。
- 要回答的问题：保持T13的Conditioned TransMorph主体不变，使用T14离线
  数据、更完整的HEMIT训练池和新版TransCUT，能否提高真实跨模态配准能力。
- 训练：20 epoch、batch 16、每轮24,000个固定离线pair；12个方向各2,000
  个。主干学习率`1e-5`、其余参数`1e-4`，flow目标为有效mask内MSE加
  `0.5×Grad`。
- 产物：
  `/data2/xujr/crossreg/stage2_offline_fulltranscut_4domain_20e_v1`。
- 合成验证：epoch 20最佳；valid EPE由zero-flow `2.9736 px`降至
  `2.2267 px`，改善`25.11%`，PCK@2/4为`54.13%/87.60%`。H&E↔DAPI
  改善`45.9%–47.3%`，panCK→CD3仅改善`5.9%`。
- 固定样例：预测flow不再接近全零，能跟随GT的主要空间趋势，但整体更
  平滑且幅度偏小，复杂局部形变恢复不足。综合图位于`samples/best.png`。
- 真实配对检查：复用D06完全相同的64个test stem和随机flow。总体EPE由
  zero-flow `3.1366 px`降至`2.9895 px`，仅改善`4.69%`，差于T13的
  `2.7105 px`。H&E→DAPI/DAPI→H&E仍改善`41.0%/44.9%`；6个方向不优于
  zero-flow，主要涉及CD3。
- 零扰动检查：预测位移平均EPE为`1.0338 px`，优于T13的`1.7023 px`，
  说明本版更少乱动；但结合已知扰动结果，模型整体变得过于保守。
- 结论：离线工程链路、平衡采样和边缘处理均成立，但没有转化为更好的
  真实四域性能。本实验不能替代T13作为当前最佳已知扰动结果；它暴露了
  固定伪图/真实图域差距和位移幅度收缩问题，状态记为“无明确收益”。
- 评估：`scripts/export_stage2_validation_samples.py`与
  `scripts/evaluate_stage2_real_pairs.py`；指标分别保存于T15目录的
  `samples/best_metrics.json`和`real_paired_test_64.json`。

## 诊断记录

### D01：NCE 模态 ID 条件

检查了源图按 source ID、生成图按 target ID 编码时的特征对应。不同 ID
确实会显著改变 CLN 特征，但训练后生成图在 target ID 编码下仍能表现出
较强空间对应，因此没有证据将当前主要失败归因于“ID 使 NCE 无法比较”。
主线继续使用真实 target ID，不改成伪装的 source ID。

### D02：真实配对跨域 MIND 分布

在 HEMIT 配对真实不同域图像上检查 MIND。真实 H&E/marker 与 marker 间
的 MIND 差异本身分布较宽，说明对所有方向施加统一强 MIND 约束会把真实
模态差异误当成生成错误。因此没有把全局 MIND loss 加入训练。

### D03：条件判别器扰动探针

脚本：`scripts/analyze_transcut_discriminator.py`。对真实配对图执行 tile
shuffle、pixel shuffle、模糊和局部对比度扰动，观察正确/错误模态 ID 的
判别间隔。结果没有支持“现有 PatchGAN 主要依赖有序空间结构，因此应换
成空间无序纹理判别器”的假设，故没有替换判别器。

产物：
`/data2/xujr/crossreg/transcut_hemit_4domain_highres_signal05_short20_20260731/discriminator_probe.json`

### D04：Encoder 跨模态空间对应探针

- 数据：从未参与训练的 HEMIT paired test 固定抽取 16 个四域 patch；每个
  模型、每个方向和层级抽取 64 个 query。比较同位置特征与随机位置的余弦
  间隔，并在全特征图中做最近邻定位。PCK@2 表示最近邻落在真实位置两个
  feature cell 内；layer 1 的 stride 为 8，即约 16 像素容差。
- 对照：随机初始化、纯 unpaired T05、全 paired T08、T09 epoch 30 与
  epoch 50。该探针直接使用真实跨模态图的 Encoder 特征，不使用 Decoder
  输出或像素 MAE。
- 对齐图 layer 1：随机、T05、T08、T09-e30、T09-e50 的余弦间隔分别为
  `0.0001/0.0016/0.1004/0.1200/0.1037`，PCK@2 分别为
  `1.21%/1.93%/12.90%/14.97%/9.89%`。因此配对锚点确实让同一空间内容
  在不同模态特征中更可辨，T09-e30 在此检查中最佳；不能再把 T09 的改善
  只解释为 Decoder 学会目标颜色分布。
- 已知平移检查：将目标图向下、向右各平移 16 像素后，T09-e30 的跨模态
  layer 1 PCK@2 降至 `4.58%`；同模态平移控制为 `50.81%`。这既说明 Swin
  特征本身并非完全平移等变，也说明当前跨模态全局最近邻仍有较强歧义。
- 边界：全局最近邻比最终 Cross-Attention RegHead 更严格，且相似细胞会
  形成多个合理候选，因此低 PCK 不能直接判定配准失败；反过来，正余弦
  间隔也不能证明能回归位移。结论是 T09 已形成可检出的跨模态对应，值得
  进入一个以已知 flow 误差为终点的短 Stage 2 对照，而不是继续做简单的
  anchor 比例或 loss 权重调参。

### D05：RegHead moving 空间置换不变性

- 当前融合为 `softmax(Q_fixed K_moving^T) V_moving`，且不将
  attention map 或 moving 坐标传给 Decoder。
- 对 moving 的空间 token 使用同一随机排列同步重排 K 和 V，
  输出的最大绝对变化仅 `5.96e-08`，属于浮点误差。这是 attention
  公式的置换不变性，不是训练不足。
- 因此当前 RegHead 能利用 moving token 的内容集合，却不知道被匹配
  token 在 moving 中的坐标，无法从 attention 本身直接计算
  `moving_coordinate - fixed_coordinate`。这与 T11 中“置零 moving 有影响，
  但换成错误 moving 几乎无影响”一致。
- 结论：在扩大 Stage 2 前，RegHead 必须以最小改动显式保留对应
  位置（如位置编码+对应坐标，或局部 correlation volume）；仅继续
  增加 epoch 不能消除该结构不变性。

### D06：真实配对模态的已知与零扰动测试

- 数据与协议：从未参与训练/验证的 HEMIT test split 中固定抽取64个四域
  共同 stem，共 `12×64=768` 个真实跨模态方向；不经过 TransCUT。对每对
  真实图分别测试已知人工 flow，以及不加扰动且以零 flow 为唯一最优答案。
- 已知扰动总体：zero-flow `3.1366 px`，T13 为 `2.7105 px`，仅改善
  `13.59%`，明显弱于其 fake-target 合成验证的 `47.07%`。
- 分方向：H&E→DAPI/DAPI→H&E 分别为 `1.5206/1.4873 px`，改善
  `51.5%/52.6%`；四个 panCK 相关非 CD3 方向改善约 `13.5%–16.4%`；
  CD3→panCK/H&E/DAPI 仅改善 `2.1%–5.2%`；panCK/DAPI/H&E→CD3
  分别为 `-0.1%/-2.5%/-9.5%`，不优于 zero-flow。
- 零扰动：预测位移幅度总体为 `1.7023 px`；H&E↔DAPI 为
  `0.7022/0.6486 px`，其余方向约 `1.47–2.40 px`。因此模型会把真实
  跨模态外观差异的一部分误判为几何形变，尤其是 CD3 目标方向。
- 结论：T13 已证明配准结构能学习两图对应，但由 TransCUT fake target
  训练得到的能力没有完整迁移到真实四域。当前主要边界是生成域与真实域的
  差距、marker 稀疏/对应信息不足以及训练中缺少明确零形变样本，不能将
  T13 的合成 EPE 直接作为真实配准性能。
- 脚本：`scripts/evaluate_stage2_real_pairs.py`；产物：T13目录下
  `real_paired_test_64.json` 与 `real_paired_test_64.log`。

### D07：TransCUT 形变—转换交换性短验证

- 日期：2026-08-11。
- 问题：检查最新版 TransCUT 是否满足
  `G(T(x), source→target) ≈ T(G(x), source→target)`；只做只读验证，未将
  几何等变项加入训练。
- 数据与规模：HEMIT 未参与训练的 paired test，固定抽取2个四域共同
  patch；覆盖12个转换方向和8 px平移、3°旋转、约4 px平滑局部形变，共
  72项比较。指标只在去除16 px边界后的有效区域计算。
- 模型：
  `/data2/xujr/crossreg/transcut_hemit_4domain_semipaired05_fulltrain_100e_20260803/transcut_final.pth`。
- 总体：0–1强度范围内交换 MAE `0.0309`、gradient MAE `0.00945`；交换
  MAE 与形变实际引起的输出变化之比为 `1.12`。平移/旋转/局部形变的该
  比值分别为 `0.47/1.12/1.78`，模型对局部非刚性形变的交换性最弱。
- 方向差异：H&E→DAPI 最好（MAE `0.0075`，比值 `0.35`）；
  panCK→H&E 与 CD3→H&E 的绝对误差最高（MAE `0.0828/0.0815`）。
  H&E→panCK 的绝对 MAE 虽低（`0.0180`），但输出方差也低且相对比值
  `2.10`，说明不能把低误差直接解释为结构保持良好。
- 结论边界：结果提供了“当前生成器并非普遍几何等变”的初步证据，且弱点
  与方向和形变类型有关；但仅2个测试 patch，不能据此直接增加训练 loss，
  更不能证明生成内容具有生物学真实性。若要做训练改动，应先扩展独立
  test 统计，并将该指标与 Stage 2 各方向真实配准误差关联。
- 脚本：`scripts/evaluate_transcut_commutation.py`；产物：
  `/data2/xujr/crossreg/transcut_commutation_eval_20260811`。

## 已讨论但未作为主实验执行

| 方案 | 决定 | 原因 |
|---|---|---|
| 几何等变 loss | 暂不加入；保留独立交换性验证 | D07 已发现交换误差，但样本量尚小，且等变不等于生物学正确；先扩展验证并检查其与 Stage 2 的相关性 |
| cycle consistency | HEMIT 主线关闭 | 跨模态映射未必可逆，可能鼓励隐藏信息或保留错误源域内容 |
| 全局 MIND loss | 不加入 | D02 不支持统一约束所有真实跨域方向 |
| 将生成图按 source ID 做主线 NCE | 不采用 | ID 不应伪装；D01 未证明 target-ID NCE 是主要瓶颈 |
| RGB/GRAY 模态专用输出 head | 不采用 | 依赖当前四域的人工硬编码，破坏统一多域设计 |
| 删除 panCK/CD3 或只做三域 | 不采用 | 四域统一转换是当前多域验证的一部分 |
| 用空间无序纹理 D 替换 PatchGAN | 不采用 | D03 的实证探针不支持该假设 |

## 新实验模板

```markdown
### TXX：名称

- 日期：
- 要回答的问题：
- 相对基线的唯一主要变化：
- 数据与 split：
- 完整启动脚本/`run_config.json`：
- 输出目录：
- 状态：
- 定量与固定样例观察：
- 结论：成功、无明确收益、失败或仍需验证；说明证据边界。
```
