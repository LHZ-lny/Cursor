# Paper Daily - 2026-06-17

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification 的顶会论文，重点核对 ICML 2026 官方页面、OpenReview、arXiv、AAAI Proceedings 与 ICLR 官方页面。
- 已排除黑名单论文；同时排除 MedFuse 这类 withdrawn submission、STAR Set Transformer 与 RAxSS 这类 workshop 条目、MedMamba/MedSpaformer 这类更偏规则或弱不规则医疗时序分类的工作。本次保留全新工作 2 篇。

## 10. QuITE: Query-based Irregular Time-series Embedding

- 会议：ICML 2026 Poster
- 作者：Junghoon Lim
- 官方页：https://icml.cc/virtual/2026/poster/64962
- OpenReview：https://openreview.net/forum?id=ILQGHFvEoo
- 论文：https://arxiv.org/abs/2605.28166
- 代码：https://github.com/Meaningfull9502/QuITE
- 关键词：irregular multivariate time series, query tokens, input embedding, backbone-agnostic module, classification/forecasting

### 场景、任务与核心难点

QuITE 面向不规则多变量时序的通用建模，其中分类实验覆盖医疗和活动识别等 IMTS benchmark。现实场景中的观测并不在统一网格上发生：不同变量有不同采样频率，变量之间存在时间错位，单条序列内部也会出现长短不一的观测间隔。现有路线通常有两种：一是重新设计专门的 IMTS 架构，代价是难以复用 PatchTST、iTransformer、S-Mamba 等成熟 MTS backbone；二是先插值或重采样到规则网格，代价是引入人工值并扭曲真实时间结构。

这篇工作的核心难点是：如何让标准规则时序 backbone 在不改主干结构的情况下直接消费非规则观测。QuITE 将问题定位在输入嵌入层，而不是整个模型架构。它用一组 learnable query tokens 通过 masked self-attention 聚合不规则观测，把原始 IMTS 转成 backbone-compatible latent representations。这样既不需要生成伪观测值，也不需要为每个 backbone 重新设计连续时间模块。

### 审稿人视角：价值与不足

最有价值的技术思想是把“不规则时序建模”从架构级问题降维为 embedding-level adapter 问题。这个视角很实用：如果一个轻量、可插拔的 query-based embedding 能稳定接入不同 MTS backbone，那么 IMTS 研究可以直接复用规则时序模型生态中的长序列建模、patching、channel mixing 和 foundation-style pretraining 能力。相比固定插值，query tokens 的聚合更像任务自适应的观测摘要，能够按数据分布学习哪些时间片段、变量和观测密度对下游分类更重要。

不足在于，QuITE 的优势也可能带来新的捷径风险。query tokens 会学习如何从观测集合中抽取信息，但它们并不天然区分“状态导致的 informative observation”和“采样政策导致的 observation pattern”。如果训练集中某些类别被更频繁采样，或某些变量只在特定流程中被测，query embedding 可能把采样密度、观测位置和变量共现直接编码成分类证据。论文强调跨 backbone 的性能增益，但对跨医院、跨设备、跨主动采样策略的 policy shift 尚未给出显式诊断。

### 对 Sampling-Policy Shift 的启发

QuITE 对我们的问题有很强的横向应用价值：它提供了一个低侵入的 policy-robust adapter 插入点。我们可以在 query-based embedding 中引入两类 query：state queries 聚合跨策略稳定的状态动力学，policy queries 聚合采样频率、时间间隔、变量共现等观测机制信息；分类头主要使用 state queries，而 policy queries 用于环境识别、不确定性校准或偏移诊断。

纵向深化上，可以把 QuITE 的 query tokens 变成 policy-invariant anchors。对同一潜在轨迹施加多种反事实采样策略，要求 state-query representation 和 logits 保持一致，同时允许 policy-query representation 变化。还可以对 query attention 做跨环境稳定性约束：如果某个 query 在训练医院主要关注“被频繁检测的变量”，而在测试医院该检测策略改变，就应能通过 policy branch 捕获这种变化，而不是让它污染主分类边界。这样 QuITE 不只是让规则 backbone 能处理 IMTS，也能成为我们研究 Sampling-Policy Shift 的模块化实验平台。

## 11. Generative Modeling of Irregular Time Series via SDE-Induced Continuous-Discrete Variational Inference

- 简称：SDEVI / SDE-VI
- 会议：ICML 2026 Poster / Spotlight
- 作者：Zexin Yuan, Qinliang Su, Junxi Xiao
- 官方页：https://icml.cc/virtual/2026/poster/64934
- 代码：https://github.com/SamoyedY/SDE-VI/
- 关键词：irregular time series, sparse asynchronous observations, continuous-discrete state-space model, SDE, variational inference, classification

### 场景、任务与核心难点

SDEVI 面向稀疏、异步观测的不规则时序生成建模，并在医疗、物理、气候和 IoT benchmark 上覆盖插值、外推、回归与分类任务。它处理的核心场景是：真实系统背后可能有连续时间动力学，但我们只能看到离散、非均匀、不同变量异步到达的观测。对于分类任务而言，关键不是简单补齐缺失值，而是从不完整观测中恢复足够稳定的潜在动态状态，使分类器依赖连续状态机制而不是观测时间表面的偶然模式。

论文针对的核心难点是连续-离散状态空间模型中的推断效率与后验表达能力。许多连续时间模型依赖 path-based variational inference，需要沿潜在路径做复杂近似，计算成本高，且容易受到后验假设限制。SDEVI 改为直接在离散观测的联合分布上进行 variational inference，同时保证与底层 SDE 连续过程一致；它用 linear time-varying SDE 诱导的变分后验作为可扩展推断骨架，并进一步引入 non-linear-SDE-induced variational inference 与复数域扩展，以增强真实复杂动态的表达。

### 审稿人视角：价值与不足

最有价值的思想是把不规则观测下的推断对象从“补全一条具体路径”转向“在离散观测联合分布上做连续过程一致的推断”。这对非规则分类很重要：如果模型能在不枚举整条潜在路径的情况下学习连续动力学一致的表示，就有机会降低求解器成本、减少插值路径假设，并在不同观测时间网格下保持更稳定的 latent state。SDE 视角还天然带有不确定性表达，适合医疗或 IoT 中稀疏观测导致的分类置信度校准。

不足在于，它是通用生成建模框架，分类只是多个下游任务之一；因此论文的分类分析可能不如专门的 IMTS classifier 那样细致。更关键的是，SDEVI 保证的是与连续随机过程的一致性，但不自动保证采样机制外生或策略不变。如果观测时间本身由病情严重程度、医生干预、设备节能策略或事件触发机制决定，那么离散观测联合分布会同时混入状态动力学与采样政策。模型越擅长拟合联合分布，越可能在没有约束时吸收 policy-induced observation pattern。

### 对 Sampling-Policy Shift 的启发

SDEVI 对 Sampling-Policy Shift 的纵向启发是：可以把“策略偏移”纳入连续-离散生成模型，而不是只在判别表征层做对抗或一致性。具体地，可以扩展为双过程模型：一个 SDE 描述真实状态动力学，另一个 point process 或 policy process 描述何时观测、观测哪个变量。分类器只依赖 state SDE 的后验表示，而 policy process 输出用于偏移诊断和置信度校准。

横向应用上，SDEVI 的联合分布推断可以用于构造反事实采样评估：固定底层 SDE 状态轨迹，替换不同采样策略，检查后验 state representation 与分类 logits 是否稳定。如果同一状态过程在不同观测政策下被推断到不同类别，就说明模型仍然把 policy 当作 label evidence。进一步，可以在 ELBO 或对比目标中加入跨策略约束：对同一潜在状态、多种采样策略的观测集合，最大化状态后验一致性，同时允许策略后验区分环境。这样能把 SDEVI 的连续时间不确定性优势转化为对 sampling-policy shift 更可控的建模框架。
