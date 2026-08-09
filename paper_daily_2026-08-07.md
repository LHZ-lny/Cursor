# Paper Daily - 2026-08-07

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`。另参考自动化记忆中 2026-08-03 至 2026-08-06 已追加但当前分支未出现为独立日期文件的标题，以扩大去重黑名单。
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
  - Context-Aware Neural SDEs for Robust Irregular Time-Series Classification
  - Context-Informed Sequence Classification: A Multimodal Approach to Vehicle Diagnostics
  - Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness
  - RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification
  - NeurOCNN: A Neural-Operator-Based Model for Physiological Time Series
  - Cached Foundation Model Summaries for Memory-Efficient Clinical Time Series Inference
  - DeNOTS: Stable Deep Neural ODEs for Time Series
  - Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / clinical trajectory / ICU time-series classification 的顶会或顶会相关论文，重点核对 ICLR 2026、ICML 2026、NeurIPS 2025、AAAI 2026、KDD 2025/2026 页面、OpenReview、arXiv 与会议官方页面。
- 已排除全部黑名单论文；同时排除 MIRA 这类 NeurIPS 2025 但主评估为 forecasting-only 的工作、Time-IMM / ReIMTS 这类偏 forecasting 或 benchmark-library 的候选、ReTAMamba / Handling Missing Modalities 这类暂无顶会录用信息的近期预印本，以及 CauKer 这类分类强但不直接处理 irregular sampling 的工作。本次保留全新工作 1 篇：PathwayLLM 是 ICML 2026 Poster，面向 ICU 患者级脓毒症风险预测，利用时间生理信号、图结构证据与 pathway-level 临床信息做轨迹分类/风险评分，并包含 MIMIC-IV 到 eICU 的外部验证。

## 1. PathwayLLM: Explainable Clinical Trajectory Modeling with Structured Pathways for Sepsis Prediction

- 会议：ICML 2026 Poster
- 作者：Zhengqiu Yu, Yueping Ding, Xiangrong Liu
- 官方页：https://icml.cc/virtual/2026/poster/61641
- 关键词：clinical trajectory modeling, sepsis prediction, EHR time series, structured pathways, graph-structured evidence, LLM contextual embeddings, external validation

### 场景、任务与核心难点

PathwayLLM 面向 ICU 中患者级脓毒症早期风险预测：模型需要从常规 EHR 观察窗口中追踪生理状态随时间恶化的轨迹，并将生命体征、实验室指标、诊断、用药和统计依赖发现得到的 clinical pathways 结合起来，输出患者级风险分数。官方 ICML 页面报告其在 MIMIC-IV 15,410 名 ICU 患者、8.45% 脓毒症患病率设置下达到 AUROC 0.891 / AUPRC 0.724，并在 eICU 外部验证中取得 zero-shot AUROC 0.842、轻量微调后 AUROC 0.867。

这个任务的核心难点不只是二分类标签稀缺或正负样本不平衡，而是 ICU 轨迹证据高度异质且时间结构复杂：早期脓毒症信号可能分散在不同时间窗口、不同变量组和不同治疗路径中；单一生理指标往往不足以解释风险，诊断-用药图和 pathway 证据又会随临床流程动态变化。PathwayLLM 采用三阶段框架：先对每个 observation window 的 physiological measurements、temporal dynamics、patient-diagnosis-medication graph 和 dependency-derived pathway signals 做多视角编码；再把这些结构化表示作为辅助上下文嵌入注入预训练语言模型，使风险预测和 evidence-conditioned explanation 联合学习；最后用 Clinical Trajectory LSTM 与 Deterioration Attention 聚合窗口级表示，突出关键恶化时点并产生患者级风险分数。

### 审稿人视角：价值与不足

最有价值的技术思想是把临床轨迹分类从“单一路径的时序编码”扩展为“多视角结构化证据 + 语言模型上下文推理 + 轨迹级恶化注意力”。相比只用 Transformer/RNN 读取时间窗口，PathwayLLM 显式把统计依赖发现得到的 pathway signals 和诊断-用药异构图放入主干，让模型既能捕捉局部生理动态，也能把风险解释锚定到更高层的临床机制和证据链。对审稿人而言，MIMIC-IV 到 eICU 的外部验证也很重要，因为它至少部分检验了模型在跨中心数据差异下的可迁移性，而不只是同院随机划分的性能提升。

不足在于，PathwayLLM 并不是一个专门为非规则采样机制设计的 encoder。论文摘要强调 temporal signals 和 observation windows，但没有清楚说明原始异步 EHR 事件如何被窗口化、聚合或对齐；如果前处理阶段已经把不规则时间戳压成固定窗口统计量，那么采样频率、未测模式和 value-pending 信息可能被部分丢失，也可能以不可控方式混入 pathway / graph 特征。更关键的是，诊断、用药和 pathway 证据本身高度受医院流程影响：某些医嘱、检查或编码可能代表真实病情，也可能代表机构特定的筛查协议、记录习惯和治疗路径。因此，跨 eICU 验证虽然是加分项，但仍不足以证明模型已经分离 patient-state signal 与 sampling / treatment policy shortcut；还需要按医院、科室、检查触发规则或反事实采样策略做更细粒度的偏移评估。

### 对 Sampling-Policy Shift 的启发

PathwayLLM 对 sampling-policy shift 的横向启发是：策略偏移不只体现在观测值和时间戳上，还会通过“结构化临床证据图”进入分类器。诊断-用药边、dependency-derived pathways 和 deterioration attention 的高权重窗口，可能同时反映患者病程与医院如何检查、编码、开药和记录。因而，在非规则采样分类中，我们不应只监测 mask ratio、delta-t 或变量共现，还应监测 pathway edge distribution、用药/检查触发模式、窗口级 attention 峰值与外部环境之间的相关性。

纵向深化上，可以把 PathwayLLM 改造成 state-policy 双通道 clinical trajectory model：state pathway branch 只学习跨医院稳定的生理恶化链和疾病机制，进入风险分类主路径；policy pathway branch 学习机构流程诱导的检查、用药、编码和记录路径，用于偏移诊断、置信度校准和解释中的 caveat。训练时可对同一患者潜在轨迹施加不同反事实采样/记录策略，要求 state pathway representation、Deterioration Attention 的关键病程时点和患者级 logits 保持一致，同时允许 policy branch 预测不同的观测密度、检查触发和医嘱路径。这样能把 PathwayLLM 的“可解释临床证据链”进一步推进到“可解释哪些证据链是病程稳定信号，哪些只是采样政策或治疗流程信号”。
