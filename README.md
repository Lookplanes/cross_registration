# Cross Registration

本页是项目文档入口，不重复记录实现细节。发生冲突时，按下表中的优先级
判断；代码行为仍需以测试和实际运行配置为准。

## 当前文档

| 优先级 | 文档 | 用途 | 状态 |
|---:|---|---|---|
| 1 | `AAA-document/0725Report/0725.md`（本地） | 当前论文范围、HEMIT/SHIFT 数据设计和完整实验路线 | **现行研究规划**，会议目录不由Git跟踪 |
| 2 | [README-self](README-self.md) | 代码结构、训练/推理契约、功能边界、启动入口 | **现行工程规范** |
| 3 | [EXPERIMENTS](EXPERIMENTS.md) | 成功、失败、中断及诊断性尝试的统一账本 | **现行实验记录** |
| 4 | [HEMIT sample report](/data2/wuyh/HEMIT/dataset/manifests/HEMIT_SAMPLE_REPORT.md) | 已下载 HEMIT 数据、通道映射和质量信息 | **现行数据事实**，位于外部数据盘 |
| 5 | [HEMIT 官方说明](/data2/wuyh/HEMIT/README.md) | 数据集官方来源与格式 | 外部参考 |

当前范围已经从早期“通用医学/显微图像配准”收缩为 H&E、DAPI、panCK、
CD3 等组织病理图像域上的完整论文验证。早期六中心模态内容只保留为项目
演进记录，不应覆盖 0725 规划。

## 历史与专项材料

| 文档 | 定位 |
|---|---|
| [tmp.md](tmp.md) | 最初的通用 Hub–Spoke/脚手架设计；保留核心动机，范围已被 0725 规划收缩 |
| `AAA-document/0715Report/`（本地） | 2026-07-15 阶段复盘、会议报告和内部附录 |
| `AAA-document/0725Report/`（本地） | 0725 规划、旧版备份和HEMIT论文评价摘记 |
| [HEMIT_CODEX_HANDOFF](HEMIT_CODEX_HANDOFF.md) | 下载与接入阶段交接记录；该阶段已经完成 |
| `DATA_INVENTORY.md`（本地） | HEMIT 之前的数据资产盘点，当前不作为训练数据定义 |

`AAA-document/*Report/` 是本地会议材料，按约定不由 Git 跟踪；训练数据、
日志、样例和 checkpoint 也不写入 Git。

## 维护规则

1. 项目目标或论文范围只在 `0725.md` 更新。
2. 代码契约、CLI 和文件职责只在 `README-self.md` 更新。
3. 数据数量、通道和质量事实写入对应数据目录的 manifest/report。
4. 带日期的会议材料视为快照，不在其中持续追加工程状态。
5. 新实验不新建说明文档；参数进入脚本和 `run_config.json`，过程与结论
   追加到 `EXPERIMENTS.md`。
