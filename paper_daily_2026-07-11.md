# Paper Daily - 2026-07-11

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
- 本次黑名单论文标题：
  - Adaptive Time Encoding for Irregular Multivariate Time-Series Classification
  - Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification
  - FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification
  - One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification
  - PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks
  - SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data
  - GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care
  - DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification
  - Fault Diagnosis of Irregular Sequences by Adjoint Learning in Continuous-Time Model Space
  - Beyond Observations: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning
  - Can we generate portable representations for clinical time series data using LLMs?
  - QuITE: Query-based Irregular Time-series Embedding
  - Generative Modeling of Irregular Time Series via SDE-Induced Continuous-Discrete Variational Inference
  - MTM: A Multi-Scale Token Mixing Transformer for Irregular Multivariate Time Series Classification
  - MedMamba: Multi-View State Space Models with Adaptive Graph Learning for Medical Time Series Classification
  - MedSpaformer: A Transferable Transformer with Multi-Granularity Token Sparsification for Medical Time Series Classification
  - MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling
  - StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Stars
  - Rethinking Large Language Models for Irregular Time Series Classification in Critical Care
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / clinical irregular time series classification 的顶会或顶级方向论文，重点核对 NeurIPS 2025、ICML 2026、ICLR 2026、AAAI 2026、ICASSP 2026、OpenReview、IEEE ICASSP 页面、arXiv 与 DOI 页面。
- 已排除全部黑名单论文；同时排除偏 forecasting、普通规则时序分类、仅 journal 而非会议、或与异步采样分类关系较弱的条目。本次保留全新工作 2 篇：RoMAE 是 NeurIPS 2025 poster，虽为跨模态 MAE 框架，但明确评估 irregularly sampled multivariate time-series classification，并提供连续位置编码路线；VP-GNN 是 ICASSP 2026 MLSP poster，直接面向 irregular clinical time series 的风险预测/分类。

## 1. Rotary Masked Autoencoders are Versatile Learners

- 简称：RoMAE
- 会议：NeurIPS 2025 Poster
- 作者：Uros Zivanovic, Serafina Di Gioia, Andre Scaffidi, Martin Emilio de los Rios, Gabriella Contardo, Roberto Trotta
- OpenReview：https://openreview.net/forum?id=nfZmQgxyyN
- 论文：https://arxiv.org/abs/2505.20535
- 关键词：irregular time-series, rotary positional embeddings, masked autoencoder, continuous positions, multivariate time-series classification, cross-modal representation learning

### 场景、任务与核心难点

RoMAE 面向一个比单一医疗 IMTS 更宽的场景：不规则多变量时序、图像和音频都被看作带连续位置的 token 集合。与许多专门为不规则时序设计的模型不同，它关心的是标准 Transformer / MAE 能否在不引入复杂时序专用结构的情况下，直接处理非均匀时间戳和多维连续位置。论文在 irregularly sampled multivariate time-series classification、插值、图像和音频任务上验证，尤其强调天文光变曲线等极端不规则科学时序。

核心难点是位置表达。普通 Transformer 往往假设离散、规则、整数位置；不规则采样下，时间戳可能是连续值，多个变量或波段还可能有各自的观测坐标。RoMAE 将 Rotary Positional Embedding 扩展到连续/轴向位置，并嵌入 MAE 的 mask-reconstruct 预训练范式，让模型在保留标准注意力结构的同时消费连续位置 token。换言之，它不是先把 IMTS 补齐成规则网格，而是让 backbone 原生理解“这个值出现在什么连续位置”。

### 审稿人视角：价值与不足

最有价值的思想是把不规则采样建模的复杂性压缩到 positional encoding 与自监督预训练层面。许多 IMTS 方法通过 ODE、图、插值器或专用 attention 处理不规则性，但这会带来工程复杂度和跨模态迁移成本；RoMAE 证明，只要连续位置编码处理得足够好，标准 MAE 也可以成为强不规则时序表示学习器。这对需要统一处理不同观测坐标、不同模态和不同采样密度的数据平台很有吸引力。

不足在于，RoMAE 的“通用性”并不等价于“采样政策鲁棒性”。连续 RoPE 让模型更准确地利用观测位置，但如果观测时间本身由医院流程、巡天 cadence、设备告警或主动采样策略决定，模型同样可能把位置分布当成类别捷径。论文还分析了 `[CLS]` token 对绝对位置恢复的影响，这既是能力，也是风险：一旦模型能从表示中恢复强位置/采样轨迹，它就可能在跨策略环境中放大 cadence shortcut。实验覆盖不规则分类，但尚未系统评估同一底层轨迹在不同采样政策下的表示稳定性。

### 对 Sampling-Policy Shift 的启发

RoMAE 的横向启发是：sampling-policy shift 可以被视为 continuous positional distribution shift。我们不应只把时间戳作为辅助数值拼进输入，而要检查位置编码层是否把采样策略编码成了可被分类头直接利用的信号。对我们的研究，可以把 RoMAE 风格连续 RoPE 作为强前端，再显式拆成 state positional channel 与 policy positional channel：前者服务于状态轨迹重建，后者解释采样 cadence、观测窗口和变量可见性。

纵向深化上，可以把 MAE 预训练目标改造成 policy-contrastive reconstruction：对同一潜在轨迹生成不同采样策略视图，要求 masked reconstruction 的 state representation 保持一致，同时允许 policy head 恢复采样策略。还可以借鉴论文中关于 `[CLS]` 恢复绝对位置的诊断，把“从表示中能否预测采样策略”作为策略泄漏度量。如果 RoMAE 表示在分类上很强、但 policy-only probe 同样很强，就说明模型仍在利用策略性位置分布；反之，若 state 表示跨采样策略稳定，就可能成为通用 Transformer 路线下的 policy-robust irregular classifier。

## 2. VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series

- 会议：ICASSP 2026, Machine Learning for Signal Processing Poster
- 作者：Yurong He, Boya Zhang, Longfei Liu, Guosheng Cui, Dan Wu
- 官方页：https://cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=18058
- DOI：https://doi.org/10.1109/icassp55912.2026.11461796
- 代码：https://github.com/bursonz/VP-GNN
- 关键词：irregular clinical time series, EHR classification, selective message passing, variable-wise graph, patch-wise graph, PhysioNet 2012, Sepsis 2019

### 场景、任务与核心难点

VP-GNN 面向 EHR/ICU 中的不规则临床时序分类，任务包括院内死亡风险预测和早期脓毒症检测。输入通常由生命体征、化验、用药或其他临床变量组成，各变量采样频率不同，观测时间异步，缺失率高，并且变量间依赖会随时间和病情阶段变化。论文在 PhysioNet 2012 与 Sepsis 2019 上报告性能，直接对应 irregular clinical time series 的高风险分类场景。

核心难点在于同时建模“变量关系”和“多尺度时间模式”。只做变量图可能捕捉到哪些指标相互影响，却忽略病情演化中的局部片段；只做时间 patch 又可能把不同变量的异步关系压平。VP-GNN 因此提出两阶段图框架：先在 variable-wise stage 通过 selective message passing 捕捉异步变量依赖，再通过 hierarchical Patch-GNN aggregation 建模多尺度时间结构。这个设计试图让模型既能看到跨变量异步共现，也能保留时间片段层级上的动态变化。

### 审稿人视角：价值与不足

最有价值的技术思想是把 irregular clinical time series 拆成变量图与时间补丁图两个层次，而不是强行构造完整的 time-by-variable 规则矩阵。selective message passing 适合临床数据：某些变量之间只在特定病程阶段或观测上下文中有意义，固定全连接或静态相关图容易引入噪声；Patch-GNN 则给模型提供从局部变化到全局病程的层级聚合路径。相比单纯 decay RNN 或普通 Transformer，这种分层图结构更贴近 EHR 的异步、多尺度和变量异质性。

不足在于，VP-GNN 仍默认观测图样可以被直接用作预测证据。临床变量之间的“选择性消息传递”很可能受采样政策影响：某些化验被联测，可能是病理机制，也可能是医院 order set、医生习惯或费用流程。Patch 聚合也可能把策略性密集采样片段当成病情恶化片段。论文在 P12/P19 类 benchmark 上报告增益，但这些评估仍主要是同分布切分，缺少跨医院、跨采样协议或反事实采样策略下的 graph stability 检验。

### 对 Sampling-Policy Shift 的启发

VP-GNN 对 sampling-policy shift 的横向启发是：策略偏移可以表现为 variable-wise graph 与 patch-wise graph 的联合漂移。不同医院可能不改变真实生理变量关系，却改变哪些变量被联测、哪些时间窗口更密集、哪些 patch 被模型认为重要。因此，我们可以把边选择频率、patch 聚合权重和跨变量消息路径作为策略偏移诊断指标，而不只看最终 AUROC。

纵向深化上，可以把 VP-GNN 改造成 policy-aware hierarchical graph。变量层面将边分解为 state edges 与 protocol edges：state edges 代表跨策略稳定的生理耦合并进入分类主路径，protocol edges 解释检查流程和联测习惯；patch 层面则将病程片段与采样密度片段解耦。训练时可对同一患者轨迹构造不同采样策略视图，约束 state graph、state patch embedding 和 logits 保持一致，同时允许 policy graph 区分医院协议或主动采样规则。这样能把 VP-GNN 的临床图建模能力推进到“采样政策变化下的图结构不变性”。
