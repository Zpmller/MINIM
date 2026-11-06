# MINIM（ICML 2026）复现报告

> 完整项目回顾已整理到[项目全程总结](MINIM_项目全程总结.md)，覆盖截至2026-11-04的所有阶段、试错与阈值版本。本文件保留早期复现记录，不应将其根目录结果表视为最新模型结果。

> 2026-11-02 机制定位：[27 任务塌缩机制实验](MINIM_27任务塌缩机制实验.md)。保持完整 27 任务、划分、旧编码、batch=32、10 epochs 不变，仅将 CI-L1 换为 CI-CE，真实关键节点保留率由 0% 恢复为 99.56%，TISL=0.0379。此为非官方训练目标的受控实验。官方公开入口未找到作者预训练 checkpoint；本文此前所有模型结果均来自本地重训，不能表述为跑过作者原权重。

> **论文参数对齐复跑（后续补充）**：原始复现使用用户指定的 hidden 496、batch 16。本次新增独立的论文参数复跑，使用 27 个核心任务（清洗后 5,400 变体）、GATv2 hidden 256、4 heads、batch 32、10 epochs、AdamW、lr 3e-4、weight decay 1e-5、α=λ=1。新 checkpoint 位于 [`runs/paper_aligned/model.pt`](../runs/paper_aligned/model.pt)，原始数据未删除，清洗视图在 [`data_paper_aligned/`](../data_paper_aligned/)。
>
> 清洗后的 5,400 条全部通过完整性校验，共 5,048,216 个节点。仓库分组划分结果为 train 4,356、val 592、test 452。训练 loss 从 2.1818 降至约 0.985，但验证 loss 约 3.8586 不降；固定论文阈值 τnec=1 的测试结果为 TCNP 0.0354、TCNP-I 0.0354、TISL 0.0000（几乎全移除）。按论文的验证目标 `TCNP−0.5×TISL` 重新选择阈值，验证集选出 τnec=0；测试结果变成 TCNP/TCNP-I/TISL 均为 1.0（全保留）。模型仍未复现论文的隐私效用折中。
>
> 仍无法完全对齐：公开数据过滤 7 个额外 `T_` ID 后为 5,400 条，比论文报告的 5,403 少 3 条；公开仓库没有论文固定 split/seed，当前仓库按 80/10/10 样本分组切分，测试集为 452 而非论文的 662。**固定阈值结果**在 [`results/paper_aligned/`](../results/paper_aligned/)，**验证集选阈值结果**在 [`results/paper_aligned_selected/`](../results/paper_aligned_selected/)，阈值记录见 [`selected_threshold.json`](../results/paper_aligned/selected_threshold.json)。

> 最新对照：[官方设置对齐与扩大训练结果](MINIM_对齐与扩大训练结果.md)。按论文可核对设置、冻结官方模型/损失的全量对照仍全移除；修复版在同一 484 变体测试集上，6/15/30 棵训练树的真实保留率为 90.45%/97.99%/97.81%，TISL 为 0.0524/0.0786/0.0893。这是修复方法的扩展结果，不能称作原论文配方的成功复现。

> 修复进展：[小规模修复与验证报告](MINIM_小规模修复报告.md)已完成六组受控对照。修复版已恢复有效预测与三种策略动作，小测试集真实关键节点保留率 65.06%、TISL=0.1173，尚未达到论文效用水平。

> 后续复核（2026-10-31）：详见 [失败原因诊断](MINIM_失败原因诊断.md)。全测试集实际全部移除，TCNP=0.0331 和 TCNP-I=0.0393 完全来自无关键节点样本的满分约定；实际关键节点保留率为零。第 2–10 轮训练损失均吻合全零预测器，另已确认任务哈希碰撞以及论文 hidden=256 与公开代码 496 的差异。以下为原始运行记录，原因判断以该诊断为准。

**执行日期：** 2026-10-31（Asia/Shanghai）  
**复现状态：** 官方工程、全量数据、GPU 训练与主要评估均已实际运行；训练流程完整，但测试效用结果未能复现论文。  
**代码版本：** 官方仓库 `yyyyyhx/MINIM`，HEAD `456c60b6fd47941d1e5849db573dbbcb488e3ae7`。

## 1. 论文方法

MINIM 对无障碍树（AX tree）的每个节点预测敏感度 `s` 和任务必要性 `n`。官方推理策略使用两个阈值：`n < 1` 时移除（R），否则 `s >= 5` 时抽象敏感文本（A），其余保留（K）。训练器通过加权 L1 损失分别拟合敏感度和必要性；GATv2 输入为 512 维节点特征，由 384 维 MiniLM 文本嵌入和 128 维 UI/结构特征组成。

评估指标按仓库实现计算：TCNP 衡量必要性标签至少 7 的关键节点没有被移除的比例；TCNP-I 限于交互节点；TISL 是低必要性节点上敏感信息暴露相对完整观察的比率。TISL 越低表示隐私暴露越少，但若同时 TCNP 很低，可能是过度移除而不是有效的隐私—效用折中。

论文与代码：[PMLR 论文](https://proceedings.mlr.press/v306/yu26bq.html)、[官方代码](https://github.com/yyyyhx/MINIM)、[官方数据集](https://huggingface.co/datasets/Chaoyu112358/MINIM-data)。

## 2. 环境与数据

| 项目 | 实际配置/结果 |
|---|---|
| 系统 | WSL2，Ubuntu 22.04.5，Linux 6.18.54.1-microsoft-standard-WSL2 |
| Python | CPython 3.10.12，项目虚拟环境 `.venv` |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6,144 MiB |
| PyTorch / CUDA wheel | PyTorch 2.10.0+cu128，CUDA runtime 12.8 |
| PyG | torch-geometric 2.8.1 |
| 其他主要依赖 | sentence-transformers 6.2.0、NumPy 2.2.6、Matplotlib 3.10.9 |
| CUDA 验证 | MLP 与 GATv2 CUDA 前向、损失和反向传播通过；GAT witness loss 49.664780 |

依赖清单在 [`environment-freeze.txt`](environment-freeze.txt)，环境说明在 [`system_info.txt`](system_info.txt)。网络下载 Hugging Face 大文件时发生 TLS/Xet 连接问题，后续通过分段下载、SHA256 校验与本地模型目录完成，不需要管理员安装系统包。

数据下载自官方仓库固定提交 `fd1786cb08dc42173905588e7ca015de968c8134`。完整性校验实际通过：5,750 个 manifest 条目与唯一树文件一一对应；Gmail 2,950、Reddit 1,300、Shopping 1,500；共检查 5,440,118 个节点，无缺文件、坏 JSON、缺失或越界标签。敏感度/必要性标签均为 0–10 数值。标签高度偏斜：敏感度为 0 的节点 5,272,844 个，必要性为 0 的节点 4,675,702 个。

数据明细见 [`dataset_validation.json`](../results/dataset_validation.json) 和 [`dataset_validation.log`](../logs/dataset_validation.log)。论文补充材料报告的总变体数为 5,403、测试集 662；公开数据仓库实际有 5,750 个变体。本次按官方代码的样本分组随机切分，得到 train 4,637、val 629、test 484。测试集域分布为 Gmail 118、Reddit 156、Shopping 210。与论文报告的测试规模及 Table 10 域样本数（472/130/60）不同，因此绝对数值不可视为完全同分布的严格对照。

## 3. 小规模流程与显存

执行了 MLP 1 epoch 与 GATv2 1 epoch 小规模 CUDA 训练、checkpoint 保存/加载及测试推理；GATv2 batch size 4、8、16 均通过。显存抽样中小规模最高约 428 MiB，因此全量训练选 batch size 16。隐私策略 K/A/R 的三个分支单元检查通过。小规模随机初始化/短训推理输出几乎全为 K，不能当作效用验证。

全量特征如果全部保留在内存，仅 512 维 float32 节点向量就约需 11 GiB，超过 WSL 可用内存。为保留官方模型和计算公式，本地加了可选的逐图磁盘特征缓存，按 batch 回读；缓存最终约 14.3 GB，当前训练代码可通过 `MINIM_FEATURE_CACHE_DIR` 重复使用缓存。该处理只改变特征的暂存位置，不改变特征值、标签、模型或损失。

## 4. 全量训练配置与过程

使用官方 GATv2：hidden dimension 496、4 heads、10 epochs、batch size 16、learning rate 3e-4、weight decay 1e-5、CI lambda 1.0、alpha 1.0、seed 0、CUDA；三个额外 boost 参数遵循官方 CLI 默认值 0。训练配置保存在 [`train_config.json`](../runs/minim/train_config.json)，checkpoint 为 [`model.pt`](../runs/minim/model.pt)，划分为 [`split.json`](../runs/minim/split.json)。

| Epoch | Train loss | Val loss | Val TCNP | Val TISL |
|---:|---:|---:|---:|---:|
| 1 | 1.4615 | 3.8468 | 0.1272 | 0.0000 |
| 2 | 1.0252 | 3.8468 | 0.1272 | 0.0000 |
| 3 | 1.0273 | 3.8468 | 0.1272 | 0.0000 |
| 4 | 1.0380 | 3.8468 | 0.1272 | 0.0000 |
| 5 | 1.0277 | 3.8468 | 0.1272 | 0.0000 |
| 6 | 1.0248 | 3.8468 | 0.1272 | 0.0000 |
| 7 | 1.0256 | 3.8468 | 0.1272 | 0.0000 |
| 8 | 1.0257 | 3.8468 | 0.1272 | 0.0000 |
| 9 | 1.0229 | 3.8468 | 0.1272 | 0.0000 |
| 10 | 1.0259 | 3.8468 | 0.1272 | 0.0000 |

训练损失下降后很快停滞，验证损失与隐私/效用指标十轮完全不变。全量 GPU 采样峰值 5,978 MiB，训练正常结束但显存余量很小。曲线：[`training_loss.png`](../results/training_loss.png)；逐轮数据：[`training_history.csv`](../results/training_history.csv)；日志：[`train_full.log`](../logs/train_full.log)。

## 5. 主要实验结果

所有官方表格和 Figure 2/3 使用同一个 checkpoint 和 `split.json` 测试集（484 变体）。Table 2 的 LLM 基线未运行。

### Table 1：主要对比

| 方法 | 论文 TCNP | 本次 TCNP | 论文 TCNP-I | 本次 TCNP-I | 论文 TISL | 本次 TISL |
|---|---:|---:|---:|---:|---:|---:|
| Full Observation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Budget | 0.2284 | 0.2284 | 0.2346 | 0.2346 | 0.1971 | 0.1971 |
| Sensitivity-Only | 0.0401 | 0.0522 | 0.0393 | 0.0582 | 0.3799 | 0.0268 |
| Necessity-Only | 0.9445 | 0.3858 | 0.9730 | 0.3829 | 0.2032 | 0.7247 |
| MINIM | **0.9491** | **0.0331** | **0.9931** | **0.0393** | **0.1010** | **0.0000** |

本次 MINIM 的 TCNP 比论文低 0.9160，TCNP-I 低 0.9538；TISL 虽降为 0，但伴随关键节点保留率近乎为零，不能据此声称实现了良好隐私效用平衡。Random Budget 的值与论文表格相同；监督模型的数值明显偏离。对照图：[`table1_paper_vs_reproduction.png`](../results/table1_paper_vs_reproduction.png)。原始表：[`table1.csv`](../results/table1.csv)、论文参考：[`paper_reference.csv`](../results/paper_reference.csv)。

### Table 3：任务语义

| 任务语义 | 论文 Keep | 本次 Keep | 论文 Abstract | 本次 Abstract | 论文 TCNP / TCNP-I | 本次 TCNP / TCNP-I |
|---|---:|---:|---:|---:|---:|---:|
| Informational | 8.66% | 0.00% | 0.83% | 0.00% | 0.9156 / 0.9886 | 0.0000 / 0.0103 |
| Sensitive/Admin | 15.84% | 0.00% | 9.84% | 0.00% | 1.0000 / 1.0000 | 0.3333 / 0.3333 |
| Transactional | 10.88% | 0.00% | 3.52% | 0.00% | 1.0000 / 1.0000 | 0.0476 / 0.0476 |

官方 Table 3 脚本原来只输出 TCNP-I，没有 TCNP；本次已补充并重跑，结果文件含两个指标。实际 Keep 与 Abstract 都为 0%，说明策略没有学会论文中任务语义差异。见 [`table3.csv`](../results/table3.csv)。

### Table 4：Abstract 消融

| 策略 | 论文 TCNP / TCNP-I / TISL | 本次 TCNP / TCNP-I / TISL |
|---|---:|---:|
| MINIM 自适应 | 0.9491 / 0.9931 / 0.1010 | 0.0331 / 0.0393 / 0.0000 |
| Conservative (A→R) | 0.9050 / 0.9650 / 0.0989 | 0.0331 / 0.0393 / 0.0000 |
| Radical (A→K) | 0.9491 / 0.9931 / 0.1014 | 0.0331 / 0.0393 / 0.0000 |

三个本次结果完全相同，因为模型没有产生 Abstract 动作；本次消融不能支持 Abstract 策略有效性。见 [`table4.csv`](../results/table4.csv)。

### Table 10：领域结果

| 领域 | 论文测试 N | 本次测试 N | 论文 TCNP / TISL | 本次 TCNP / TISL |
|---|---:|---:|---:|---:|
| Gmail | 472 | 118 | 0.9661 / 0.0188 | 0.1356 / 0.0000 |
| Reddit | 130 | 156 | 0.9538 / 0.1350 | 0.0000 / 0.0000 |
| Shopping | 60 | 210 | 0.9360 / 0.0085 | 0.0000 / 0.0000 |

样本数差异来自公开数据规模与本次按样本分组的随机划分；本次域间测试占比也不同。见 [`table10.csv`](../results/table10.csv)。

### Figure 2 与 Figure 3

Figure 2 在 `tau_nec=0` 时等价于全保留，得到 TCNP/TISL 均为 1；阈值提高到 0.5 后，TCNP 直接降至 0.0331、TISL 降至 0，并在之后阈值保持不变。该阶跃与必要性预测几乎全为 0 相符。Figure 3 在 Hard Conflict、Safe Utility、Privacy Risk、Irrelevant 四组中均为 Remove 100%，Keep 和 Abstract 均为 0%。

图表及可复核数据：[`figure2_threshold_sweep.png`](../results/figure2_threshold_sweep.png)、[`pareto_test_sweep.csv`](../results/pareto_test_sweep.csv)、[`figure3_policy_distribution.png`](../results/figure3_policy_distribution.png)、[`figure3_policy_distribution.csv`](../results/figure3_policy_distribution.csv)。

## 6. 结果诊断与差异分析

对 checkpoint 的诊断抽样（manifest 前 24 个变体，共 1,800 节点）发现：敏感度预测均值约 `1.89e-7`，必要性预测均值约 `1.05e-5`；两者最大值分别约 `1.14e-6` 和 `4.31e-5`。阈值策略因此把几乎所有节点判为 R。此诊断与 Figure 3 的全移除结果一致。

较可能的原因是节点标签严重偏斜、零标签占绝大多数，且训练时 CI loss boost 参数按官方 CLI 的默认 0 使用；但现有运行只能证明分布偏斜与输出塌缩同时存在，尚不能单独证明因果。训练/验证 loss 曲线也显示验证 loss 不变，故当前 checkpoint 不适合作为有效隐私过滤器。

复现差异还包括：公开数据比论文补充材料多 347 个变体；本次测试集为 484 而论文报告 662；Table 10 域分布不同。论文未完全给出代码默认值与公开仓库数据之间的映射细节。尽管如此，Random Budget 对照能复现，而监督模型明显失败，核心问题不能只归因于测试样本数量差异。

## 7. 复现中遇到的问题

1. WSL 未预装 `python3-venv`，使用用户目录安装的 uv 创建 `.venv`，未要求管理员权限。
2. CUDA wheel、数据与 Hugging Face 模型下载出现 TLS/Xet 连接中断；通过重试、分段下载、固定数据提交及模型 SHA256 校验完成。
3. 全量特征常驻会超过可用 RAM；采用逐图磁盘缓存和 batch 回读后完成全量训练。
4. 全量训练峰值显存接近 6 GB 上限（5,978 MiB），batch 16 可完成但余量有限。
5. 论文指标未复现：模型输出塌缩，全量评估结果反映了失败而非方法优势。

## 8. 后续改进方向

1. 先审计训练目标与标签：逐领域统计正负样本、必要性高分比例、模型 logits/softmax 温度，并检查预测期望分数是否应按类别期望计算。
2. 在不改测试集的前提下，单独对照官方 loss boost（敏感高分、必要性低/高分加权）与标签重采样，观察验证 loss 和 TCNP 是否恢复；保留原始官方配置作为基线。
3. 用多随机种子和论文一致的测试划分重跑，报告均值/方差；首先解释公开数据 5,750 与论文 5,403 的差异。
4. 加入早停、校准与阈值敏感性分析，防止只优化节点级标签损失却得到“全移除”策略。
5. 对 Abstract 策略做独立覆盖率约束，确保其在验证集真实触发后再讨论隐私—效用收益。

## 9. 可重复执行入口

- 环境：[`scripts/install_env.sh`](../scripts/install_env.sh)
- 数据下载与校验：[`scripts/download_dataset.py`](../scripts/download_dataset.py)、[`scripts/validate_dataset.py`](../scripts/validate_dataset.py)
- 冒烟流程：[`scripts/run_smoke_tests.sh`](../scripts/run_smoke_tests.sh)
- 全量训练：[`scripts/train_full.sh`](../scripts/train_full.sh)
- Table 1/3/4/10 与 Figure 2/3：[`scripts/run_experiments.sh`](../scripts/run_experiments.sh)
- 训练曲线：[`scripts/plot_training.py`](../scripts/plot_training.py)

复现实验均有日志保存在 `logs/`，权重与划分保存在 `runs/`，CSV 和图片保存在 `results/`。最终结论：**工程链路和实验脚本已完整跑通，论文核心隐私—效用结论没有在本次运行中复现；当前结果应作为失败复现基线和后续诊断起点。**

## 10. 论文参数对齐复跑的完整结果

本节是后续补充的第二次全量运行，使用清洗视图 `data_paper_aligned/`；原始 `data/` 未修改。清洗规则保留公开 manifest 中 27 个不带 `T_` 前缀的任务 ID，排除另外 7 个 ID。结果是 5,400 个变体：Gmail 2,800、Reddit 1,200、Shopping 1,400；节点总数 5,048,216。清洗规则与排除 ID 记录在 [`cleaning_manifest.json`](../data_paper_aligned/cleaning_manifest.json)，完整校验在 [`dataset_validation.json`](../results/paper_aligned/dataset_validation.json)。

训练配置：GATv2 3 层，hidden 256，4 heads；AdamW、batch 32、10 epochs、lr 3e-4、weight decay 1e-5、alpha 1.0、CI lambda 1.0。论文没有报告随机种子，故沿用仓库默认 seed 0。额外 boost 参数为 0，使损失对应论文给出的 CI 加权 L1 公式。配置在 [`train_config.json`](../runs/paper_aligned/train_config.json)，命令在 [`train_paper_aligned.sh`](../scripts/train_paper_aligned.sh)。

仓库默认按样本分组、80/10/10 切分，得到 train 4,356、val 592、test 452；同一样本的变体没有跨 split。训练 loss 从 2.1818 降至约 0.985，验证 loss 约 3.8586，验证 TCNP 0.1351、TISL 0。逐轮记录和曲线见 [`paper_aligned_training_history.csv`](../results/paper_aligned_training_history.csv) 与 [`paper_aligned_training_loss.png`](../results/paper_aligned_training_loss.png)。

论文先在验证集按 `TCNP−0.5×TISL` 选择必要性阈值。本次验证扫描选出 `tau_nec=0.0`，因为该模型在任何正阈值下几乎全部移除节点。测试结果因此分别报告两种阈值：

| 评估阈值 | MINIM TCNP | MINIM TCNP-I | MINIM TISL | 观察 |
|---|---:|---:|---:|---|
| 固定论文值 `tau_nec=1.0` | 0.0354 | 0.0354 | 0.0000 | 几乎全移除，效用极低 |
| 验证集选择值 `tau_nec=0.0` | 1.0000 | 1.0000 | 1.0000 | 全保留，没有隐私降低 |

验证集阈值曲线和选择记录见 [`validation_threshold_sweep.csv`](../results/paper_aligned/validation_threshold_sweep.csv) 与 [`selected_threshold.json`](../results/paper_aligned/selected_threshold.json)。固定阈值与验证选择阈值下的 Table 1/3/4/10、Figure 2/3 分别保存在 [`results/paper_aligned/`](../results/paper_aligned/) 和 [`results/paper_aligned_selected/`](../results/paper_aligned_selected/)。两者都没有复现论文的折中点（TCNP 0.9491、TCNP-I 0.9931、TISL 0.1010）。

### 仍未对齐的部分

1. **变体总数差 3 条：**论文报告 5,403；移除 7 个额外 ID 后，公开仓库只剩 5,400。公开数据无法提供这缺少的 3 条，未人为复制或合成。
2. **固定 split 不可用：**论文报告 train 4,741/test 662，但未公开可直接复用的 split 文件或随机种子。仓库脚本生成 train 4,356/val 592/test 452。Table 10 域测试数也不同：本次 112/144/196，论文 472/130/60。
3. **阈值选择结果不同：**论文选出 `tau_nec=1.0`；本次验证集选出 0.0。阈值 0 对本模型等价于全保留，阈值 1 则几乎全移除。
4. **硬件与随机种子：**本次在 RTX 3060 CUDA 上运行；论文给出 AdamW 等超参数，但没有报告 seed 或硬件。batch 32 在本机显存约 6 GB 的卡上运行成功，但峰值余量有限。
5. **核心结果仍不同：**即使把任务 ID、hidden size、heads、batch、epoch、optimizer、学习率、weight decay、alpha 和 CI lambda 调到论文报告值，模型仍未学到论文展示的自适应 K/A/R 策略。因此本次复跑不能支持论文的隐私—效用结论。
