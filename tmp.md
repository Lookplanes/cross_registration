# 初始项目设计（历史基线）

> **状态：历史设计。** 本文保留最初的通用 Hub–Spoke 目标和脚手架动机；
> 当前论文范围已经收缩，现行规划见
> `AAA-document/0725Report/0725.md`，工程契约见 `README-self.md`。

### 核心背景与挑战
*   **核心任务**：实现 $M$ 种显微模态到 $N$ 个中心锚点模态（H&E、Fluorescence、MSI 等）的高精度非线性配准。这里仅解决$N$ 个中心锚点模态之间的配准问题
*   **关键难点**：跨模态图像之间存在显著的外观差异与内容不对应问题，且缺乏真实的形变场真值（Ground Truth），难以直接构建监督信号。
*   **解决思路**：采用"脚手架（Scaffold）"式训练策略，以模态转换（Translation）作为中间代理任务，构造精确的监督信号，最终演进为端到端的特征驱动配准模型。

---

### 系统架构组件

#### TransCUT（模态转换模块）
*   **Encoder**：采用 **Swin-Transformer** 架构（与 TransMorph 保持一致），负责提取跨模态的通用结构表征。
    *   输入 $n_1$ + **Source ID**（通过 **CLN** 注入）。
*   **Decoder**：采用轻量级 **CNN** 结构，负责将抽象特征映射回像素空间。
    *   输入特征 + **Target ID**（通过 **AdaIN** 注入）。

*   **Loss**：依然使用 CUT 的 Contrastive Loss，确保结构不失真。



#### TransMorph（配准模块）
*   **共享 Encoder**：复用 TransCUT 阶段预训练得到的 Swin-Transformer 权重。
*   **配准头（Registration Head, Reg-Head）**：接收两组特征流，通过交叉注意力机制估计非线性形变场（Deformation Field）。

*   **输入流**：
    1. 真图 $n_1$ + **ID_1** $\to$ Encoder $\to$ 特征 $F_1$。
    2. 扰动伪图 $n_{2\_fake}'$ + **ID_2** $\to$ Encoder $\to$ 特征 $F_{2}'$。
*   **Reg-Head**：对比 $F_1$ 和 $F_{2}'$ 的几何差异，回归位移场 $D_{gt}$。

---

### 训练流程

#### 1.特征空间预训练（TransCUT Pre-training）
*   **目标**：使 Encoder 学会捕捉不同模态间的生物学结构共性。
*   **流程**：
    1. 输入源模态图像 $n_1$ 与目标模态 ID $n_j$。
    2. Encoder 提取特征表示，Decoder 据此生成目标模态风格的合成图像 $n_{j\_fake}$。
    3. 采用 **分块对比损失（Patch-wise Contrastive Loss, CUT）** 约束转换过程中的结构保真度。
*   **产物**：掌握 $N$ 种模态“生物学语义”的 Swin Encoder。

#### 2.合成数据驱动的有监督训练（Synthetic Supervised Training）
*   **目标**：借助已知形变场真值（$D_{gt}$）训练配准头的几何推断能力。
*   **流程**：
    1. **数据“造影”**：取真实图像 $n_1$，经 TransCUT 生成与之像素级对齐的跨模态合成图像 $n_{2\_fake}$。
    2. **人工形变模拟**：对 $n_{2\_fake}$ 施加随机非刚性形变场 $D_{gt}$，得到扰动后的图像 $n_{2\_fake}'$。
    3. **有监督训练**：
       - TransMorph 的 Encoder 加载 TransCUT 预训练权重并冻结。
       - 输入：$(n_1, n_{2\_fake}')$。
       - 监督标签：$D_{gt}$。
       - 损失函数：$Loss = \text{MSE}(D_{\text{pred}}, D_{gt})$。
*   **意义**：从根本上解决了跨模态配准任务中“形变场真值缺失”这一核心难题。

#### 3.端到端精简与部署（End-to-End Inference）
*   **流程**：
    1. 输入两幅真实组织切片：$x$（模态 1）与 $y$（模态 2）。
    2. Encoder 分别提取二者的特征图。
    3. Reg-Head 基于特征图之间的相关性直接输出形变场。
*   **优势**：推理阶段无生成伪影引入，延迟低，精度高。

---

### 方案优势

1.  **架构一致性（Architectural Consistency）**：预训练任务（模态转换）与下游任务（图像配准）共享同一 Swin-Transformer Encoder，从根源上消除领域偏移（Domain Shift）。
2.  **形变场真值引导（Deformation-GT Guidance）**：以像素级模态转换作为“脚手架”，合成已知形变场 $D_{gt}$，确保模型学习的是确定的几何映射规律，而非缺乏可解释性的隐式特征关联。
3.  **模块化可解释性**：在研发阶段保留 Decoder，便于生物医学工程研究人员对模态转换质量进行可视化评估；在部署阶段移除 Decoder，保障最终配准结果的可靠性与临床安全性。
4.  **任务解耦设计**：将“模态不变特征学习”交由 Encoder 承担，将“空间形变估计”交由 Reg-Head 承担，实现多模态场景下各项子任务的负载均衡。

---

### 评价指标
*   **结构相似性**：SSIM、NCC（在模态转换后的图像空间上计算，衡量图像内容的对齐程度）。
*   **目标配准误差（TRE, Target Registration Error）**：基于人工标注的解剖特征点对，计算配准后对应点之间的欧氏距离，反映物理定位精度。
*   **形变场拓扑保真度**：Jacobian 行列式（Jacobian Determinant），用于检验形变场的平滑性与可微同胚性，确保不发生折叠或撕裂）。
*   
