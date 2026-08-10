# Paper Daily - 2026-08-10

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
- 额外纳入自动化记忆中 2026-08-03 至 2026-08-09 已覆盖但当前分支未出现的标题，避免因分支缺失历史文件而重复推荐。
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
  - Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing
  - Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction
  - Status-Aware Self-Supervised Forecasting for Irregular Clinical Time Series
  - LLM4EHR: Aligning Clinical Time Series with Medical Event Sequences via Large Language Models
  - Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series
  - VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series
  - PULSE: Benchmarking Large Language Models for ICU Time Series Classification
  - Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time
  - Context-Aware Neural SDEs for Robust Irregular Time-Series Classification
  - Context-Informed Sequence Classification: A Multimodal Approach to Vehicle Diagnostics
  - Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness
  - RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification
  - NeurOCNN: A Neural-Operator-Based Model for Physiological Time Series
  - Cached Foundation Model Summaries for Memory-Efficient Clinical Time Series Inference
  - DeNOTS: Stable Deep Neural ODEs for Time Series
  - Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time
  - PathwayLLM: Explainable Clinical Trajectory Modeling with Structured Pathways for Sepsis Prediction
  - ArcTimeSDE: Aligning Compute with Information Via ARC Length Time in Neural SDEs
  - MoRGen: Mixture-of-Resolutions Generative Forecasting for Irregularly Sampled Medical Time-Series Data
  - STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data
  - Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / medical time series classification / sampling-policy shift 的顶会论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、KDD 2025/2026、NeurIPS 2025 官方页、OpenReview、ACM DOI、AAAI Proceedings、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 TreeText-CTS、TRIAGE、CT-Former、SPAM 这类虽直接命中 irregular clinical time series 但暂无可验证顶会录用信息的预印本/稿件，排除 LakeFM、Under-Cali、TFMixer、MOSES、ReIMTS、ASTGI、MIRA 等偏 forecasting/generation 的工作，也排除 MambaSL、TimeSliver、UniShape、OpenTSLM、HEARTS 等普通规则时序分类或 reasoning benchmark。由于新的“原生 irregular sampled classification”正会 direct hits 已基本被历史日报和记忆覆盖，本次保留 2 篇全新 ICLR 2026 医疗时序分类工作：FORMED 强调跨数据集/通道/任务异质下的可泛化分类，TeCh/CoTAR 强调医疗多通道信号的中心化依赖结构；二者不是纯事件级 IMTS 方法，但对采样策略偏移中的通道可用性、观测协议和跨环境适配有可迁移启发。

## 1. Repurposing Foundation Model for Generalizable Medical Time Series Classification

- 简称：FORMED
- 会议：ICLR 2026 Poster
- 作者：Nan Huang, Haishuai Wang, Zihuai He, Marinka Zitnik, Xiang Zhang
- OpenReview：https://openreview.net/forum?id=wNEzRYiyZM
- 官方页：https://iclr.cc/virtual/2026/poster/10006735
- 论文：https://arxiv.org/abs/2410.03794
- 代码：https://github.com/DL4mHealth/FORMED
- 关键词：medical time series classification, foundation model repurposing, cross-dataset generalization, channel embeddings, label queries, lightweight adaptation

### 场景、任务与核心难点

FORMED 面向医疗时序分类中的跨数据集泛化问题。真实部署时，医疗信号数据往往来自不同设备、队列和诊断任务：通道数量不同，序列长度不同，标签空间不同，患者群体与采集协议也不同。一个在单一 EEG/ECG 数据集上训练的分类器，即使同分布表现强，也很容易在换医院、换设备或换任务后退化。

论文的核心难点是：如何把主要为 forecasting 训练的通用时序 foundation model，改造成能处理任意通道数和类别数的医疗分类器，同时避免为每个新数据集全量微调。作者冻结 TimesFM 这类预训练 backbone，把适配集中到一个轻量分类头：task-specific channel embeddings 表示当前数据集的通道结构，label queries 表示当前任务的类别证据探针，shared decoding attention 在多数据集联合训练中沉淀医疗领域的共享判别知识。适配新数据集时只训练约 0.1% 参数，使模型能够在 unseen MedTS datasets 上快速迁移。

### 审稿人视角：价值与不足

最有价值的技术思想是把“医疗时序分类泛化”拆成三层：冻结的通用时间表征、跨数据集共享的医疗判别注意力、以及数据集/任务特异的通道和标签查询。这比直接微调整套 foundation model 更符合真实医疗部署，因为新医院可能只有少量标注，且输入通道和标签空间经常与训练源不同。label query 的设计也很干净：类别不再只是最后线性层的 index，而是主动从时间表示中读取证据的可学习查询。

不足在于，FORMED 处理的是多通道医疗信号分类的结构异质性，而不是原生事件级 irregular sampled EHR。输入仍默认可表示为 channel-by-time 的序列，采样时间戳、变量级异步、告警触发式测量和 missingness pattern 并不是模型主干中的显式一等对象。更重要的是，channel embeddings 可能吸收数据集的采集协议、设备布置、通道选择和重采样策略；如果这些协议与标签相关，轻量适配模块可能学到的是 site/protocol shortcut，而不仅是可迁移的疾病证据。论文证明了跨数据集分类泛化，但还没有系统报告跨采样政策、跨缺失机制或反事实通道可用性变化下的稳定性。

### 对 Sampling-Policy Shift 的启发

FORMED 对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以被建模为“适配参数空间”的偏移。不同医院或设备的采样频率、通道选择、窗口长度和任务标签不同，都会反映到 channel embeddings 与 label queries 的学习轨迹上。因此，这些轻量适配参数本身可以成为 policy shift 的诊断对象：如果同一疾病任务在不同采样政策下需要完全不同的 channel embedding 才能工作，就说明 backbone 表征并未真正策略不变。

纵向深化上，可以把 FORMED 改造成 state-policy adapter：共享 backbone 和 shared decoding attention 负责学习跨策略稳定的状态表征；channel embeddings 分解为 state-channel embedding 与 policy-channel embedding；label queries 只允许读取 state embedding，policy embedding 用于校准和偏移诊断。训练时可对同一潜在轨迹构造不同采样策略视图，约束 state queries、分类 logits 和共享注意力分布保持一致，同时允许 policy embeddings 解释通道缺失、采样频率和设备协议差异。这样能把“跨数据集轻量适配”推进到“跨采样政策可解释适配”。

## 2. Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series

- 方法名：TeCh / CoTAR
- 会议：ICLR 2026 Oral Session / OpenReview
- 作者：Guoqi Yu, Juncheng Wang, Chen Yang, Jing Qin, Angelica Aviles-Rivero, Shujun Wang
- OpenReview：https://openreview.net/forum?id=oZJFY2BQt2
- 官方页：https://iclr.cc/virtual/2026/session/10012007
- 论文：https://arxiv.org/html/2602.18473v1
- 代码：https://github.com/Levi-Ackman/TeCh
- 关键词：medical time series classification, channel dependency, centralized signal structure, core token aggregation-redistribution, EEG, ECG, efficient transformer alternative

### 场景、任务与核心难点

这篇工作面向 EEG、ECG 等医疗多通道时序分类，例如脑部或心脏疾病诊断。此类信号的一个核心结构是通道之间并非完全去中心化交互：EEG 常受中枢神经活动协调，ECG 常受窦房结等中心节律驱动，不同通道共享一个相对集中的生理源。标准 Transformer attention 让每个 token 与所有 token 做 peer-to-peer 交互，表达力强，但也容易让噪声通道、局部伪差或冗余通道稀释核心同步模式。

论文解决的核心难点是：如何用更符合医疗信号生成结构的方式建模跨通道依赖，同时降低计算成本。作者提出 CoTAR（Core Token Aggregation-Redistribution），用一个 global core token 替代全连接式 attention：所有通道/时间 token 先聚合到核心 token，再由核心 token 将全局信息重新分发回各 token，形成类似 star topology 的中心化通信。该模块被整合为 TeCh 框架，在多个 MedTS benchmark 上相比 Transformer 类 baseline 提升分类性能，并把复杂度从二次级注意力降到更接近线性。

### 审稿人视角：价值与不足

最有价值的思想是明确指出“医疗时序的跨通道结构”和“Transformer 的默认通信拓扑”之间可能存在错配。很多时序模型默认 self-attention 是通用答案，但在 EEG/ECG 这类通道强同步、中心源明显的信号中，全通道两两交互不一定是最佳 inductive bias。CoTAR 用 core token 作为信息瓶颈，既能减少噪声通道直接污染其他通道，也让全局节律或中心驱动模式更容易形成稳定表征；同时带来显著效率收益，这对长医疗信号很实用。

不足在于，中心化假设并非所有医疗时序都成立。ICU/EHR 中的化验、生命体征、用药和护理记录往往是异步事件流，变量之间可能由多种病理机制与医院流程共同驱动，未必存在单一核心源。即便在 EEG/ECG 中，设备布置、导联缺失、采样频率、滤波流程和伪差处理也会影响 core token 聚合到的“中心信号”。如果某些通道在某医院更常缺失，或某些设备只在高风险患者中使用，core token 可能成为采样政策和设备协议的压缩器，而不是纯生理状态的压缩器。论文重点验证同类 MedTS benchmark 的准确率、效率和鲁棒性，对跨设备、跨采样率、跨通道缺失政策下 core token 的语义稳定性仍缺少系统分析。

### 对 Sampling-Policy Shift 的启发

TeCh/CoTAR 对 Sampling-Policy Shift 的横向启发是：采样策略偏移不仅改变哪些观测可见，也会改变模型内部的信息拓扑。若模型使用全连接 attention，策略性高频通道可能通过大量 pairwise interaction 放大 shortcut；若模型使用 core token，策略信息可能被更集中地压缩进核心瓶颈。因此，研究采样策略偏移时，应同时监控通道可用性、core-token activation、聚合/分发权重以及下游 logits 的跨策略变化。

纵向深化上，可以把 CoTAR 扩展为双核心结构：state core 聚合跨采样策略稳定的生理/系统状态，policy core 聚合通道缺失、设备布置、采样频率和预处理协议等观测机制信息。分类头主要读取 state core，policy core 用于不确定性校准、OOD 检测或策略识别。训练时对同一底层信号施加不同通道遮蔽、重采样频率和设备协议增强，约束 state core 与 logits 保持一致，同时允许 policy core 区分采样策略。这样可以把 CoTAR 的“中心化医疗信号归纳偏置”推进为“状态中心与策略中心可分离”的非规则采样分类框架。
