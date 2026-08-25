# Paper Daily - 2026-08-25

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`；同时读取兼容入口 `paper_daily.md` 的标题索引，并参考自动化记忆中 2026-08-03 至 2026-08-21 的新增标题。
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
  - Repurposing Foundation Model for Generalizable Medical Time Series Classification
  - Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series
  - Investigating a Model-Agnostic and Imputation-Free Approach for Irregularly-Sampled Multivariate Time-Series Modeling
  - TreeText-CTS: Compact, Source-Traceable Tree-Path Evidence for Irregular Clinical Time-Series Prediction
  - Random Controlled Differential Equations
  - An Automated Data Engineering Pipeline for Time Series Classification Via Text Embeddings
  - CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data
  - JETS: A Self-Supervised Joint Embedding Time Series Foundation Model for Behavioral Data in Healthcare
  - TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs
  - MATA-Former & SIICU: Semantic Aware Temporal Alignment for High-Fidelity ICU Risk Prediction
  - Cross-Representation Benchmarking in Time-Series Electronic Health Records for Clinical Outcome Prediction
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / informative irregularity / clinical workflow metadata / sampling policy shift 的顶会或顶会 workshop 论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、ICASSP 2026、NeurIPS 2025、OpenReview、ICML virtual site、AAAI Proceedings、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 TimEE、SGN、FORMED 等不以 irregular sampling 为核心或已被历史日报覆盖的工作，以及 MIRA/ReDiTT/Time-IMM 等偏 forecasting/generation 的条目。严格的“顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 2 篇未在黑名单中的 ICML 2026 Structured Data for Health workshop 新工作：`Informative Irregularity` 直接把临床观测时间与工作流元数据用于预测鲁棒性诊断；`SBRD` 虽更偏临床决策/动作预测而非传统时序分类，但其 shared benchmark + switching regime 分解对 Sampling-Policy Shift 的状态-政策拆分非常有启发。

## 1. Informative Irregularity as a Diagnostic for Model Robustness

- 会议/状态：ICML 2026 Workshop on Structured Data for Health Poster
- 作者：Tamara Krafft, Bernhard Bauer
- 官方页：https://icml.cc/virtual/2026/71781
- 关键词：irregularly sampled EHR, structural workflow metadata, robustness stratification, clinical observation timing, MIMIC-IV-ED, AUPRC

### 场景、任务与核心难点

这篇工作面向急诊/临床 EHR 中的不规则观测建模。输入不是理想化的规则生命体征矩阵，而是 MIMIC-IV-ED 中 3,497 次就诊的临床观测序列；论文关注的核心任务是临床预测模型在不同 irregularity regime 下的鲁棒性，并检验“何时被测量、是否偏离本地常规节律、工作流结构如何变化”本身是否能补充生命体征数值。

核心难点在于，临床观测时间不是随机噪声。患者病情恶化、医生关注度、科室工作流、检查排队和本地 routine rhythm 都会改变观测时间；这些信号对 acuity 有预测价值，但也可能是医院特定采样政策的产物。论文因此抽取 structural workflow metadata，并按 distinct irregularity regimes 做分层评估。结果显示，加入结构元数据相对 native baseline 带来 +4.1% AUPRC；特征重要性分析显示 workflow metadata 与生命体征 top features 的平均相关绝对值仅约 0.041，说明其提供了相对独立的诊断信号；当采样不遵循本地 routine rhythm 时，结构元数据收益进一步达到 +8.0% AUPRC。

### 审稿人视角：价值与不足

最有价值的思想是把 irregularity 从“需要被模型忍受的缺陷”转化为“可以被量化、分层和审计的工作流信号”。很多 IMTS 论文只报告随机 missing ratio 或同分布测试集性能，而这篇论文要求模型回答更细的问题：在常规节律、非例行采样、局部工作流异常等不同 regime 下，性能为何变化？这种 stratification 对审稿人很有吸引力，因为它能把平均 AUPRC 背后的 failure mode 暴露出来，也能避免把采样信息简单归为有用或有害。

不足在于，它是 workshop 工作，贡献更偏诊断框架和特征分析，而不是一个完整的新分类架构。实验数据集中在 MIMIC-IV-ED 的单一场景，structural workflow metadata 的定义也可能依赖本地记录系统；若换医院、换急诊流程或换变量 schema，这些 metadata 是否可迁移仍需验证。此外，AUPRC 提升证明采样/工作流信息有预测力，但不等于它是跨环境稳定的病理信号；它也可能是高价值但高风险的 policy shortcut。

### 对 Sampling-Policy Shift 的启发

这篇论文对 Sampling-Policy Shift 的横向启发非常直接：采样政策偏移应该被显式变成评估分层，而不仅是隐藏在整体测试集中的分布变化。我们可以借鉴其 workflow metadata 思路，为每条不规则序列构造 policy descriptors，例如 routine-rhythm deviation、变量联测强度、告警后采样密度、long-gap pattern、夜班/日班观测差异和 value-pending 状态，然后分别报告 state-only、policy-only 与 full-model 的分类表现。

纵向深化上，可以设计 policy-diagnostic IMTS classifier：主干只用观测值和稳定时间结构学习 patient-state representation；并行 policy head 预测 workflow/irregularity regime，用于偏移告警和校准，而不直接进入最终分类边界。训练时对同一潜在轨迹施加不同采样策略增强，要求 state logits 稳定，同时允许 policy descriptors 改变。这样能把“informative irregularity 可提升 AUPRC”推进到“哪些 irregularity 是可迁移病情信号，哪些只是当前医院采样政策”的可审计区分。

## 2. Shared-Benchmark Regime Decomposition for Nonstationary Clinical Decisions

- 简称：SBRD
- 会议/状态：ICML 2026 Workshop on Structured Data for Health Poster
- 作者：Yuting Yan, Haozhou Gao, Xinye Chen, Yinghao Fu, Shuang Li
- 官方页：https://icml.cc/virtual/2026/71707
- 关键词：irregularly sampled physiological signals, clinical decision sequences, nonstationarity, latent regimes, continuous-time latent dynamics, ICU sepsis treatment

### 场景、任务与核心难点

SBRD 面向结构化临床决策序列，包括 tabular EHR 与不规则采样 physiological signals。论文的目标不是传统“给整条时间序列打疾病类别标签”，而是预测 ICU sepsis treatment 与 CKD-MBD management 中的临床动作，并解释为什么相似 measured states 在不同时间或不同约束下会导致不同决策。这可视为一种 policy-sensitive sequential classification：模型需要把患者当前 belief state、可选动作价值和临床流程约束同时纳入判断。

核心难点是部分可观测和非平稳性。相同的生理观测值并不总是对应相同治疗动作，因为 ICU 资源、协议压力、治疗复杂度、阶段性约束和医生行为 regime 会改变动作分布。SBRD 提出 shared-benchmark regime decomposition：用 continuous-time latent dynamics layer 从结构化健康记录中构造 belief states 和 benchmark action values，再用 sparse regime layer 恢复 persistent、可解释的 constraint-regime deviations，也就是相对共享 benchmark 的策略性 wedge。实验显示该分解能提升 held-out action prediction，并给出 clinically meaningful regime profiles。

### 审稿人视角：价值与不足

最有价值的思想是把“稳定状态价值”与“环境/约束 regime 偏差”拆开。对 sampling-policy shift 而言，这比单纯提高分类 accuracy 更关键：很多模型失败不是因为无法表示患者状态，而是把某一环境下的观察/治疗政策当成了状态本身。SBRD 的 shared benchmark 相当于寻找跨 regime 可共享的临床决策基线，sparse regime layer 则把协议、资源或时间阶段造成的偏差显式化。这个结构为“状态机制 vs 政策机制”的分解提供了可操作模板。

不足也很明确：它不是标准 irregular time series classification 正会论文，而是 ICML workshop 中偏临床决策建模的工作；任务中心是 action prediction 和决策 regime 解释，不是 P12/P19/PAM 这类 IMTS 分类 benchmark。latent regime 的可识别性依赖建模假设，若未观测混杂同时影响病情和决策，benchmark value 与 regime wedge 仍可能纠缠。论文也尚未系统评估换医院采样协议、换测量频率或反事实 observation policy 后，belief state 与 regime layer 是否保持预期分工。

### 对 Sampling-Policy Shift 的启发

SBRD 对 Sampling-Policy Shift 的纵向启发是：采样策略本身可以被视为一种 switching constraint regime。临床系统并不是被动记录患者，而是在资源、协议和风险判断下主动选择“何时测、测什么、何时治疗”；因此，不规则采样分类器也应分解为 shared state benchmark 与 policy-regime deviation。前者承载跨采样政策稳定的病程表征，后者解释为什么当前环境产生特定观测密度、变量可见性或治疗动作。

横向应用上，可以把 SBRD 的 regime decomposition 移植到 IMTS 分类：先用连续时间 latent dynamics 从不规则观测中构造 patient belief state，再用稀疏 policy-regime layer 捕捉医院/科室/时间段/采样密度诱导的偏差。分类头只允许依赖 shared state 与经校准的 regime uncertainty，而不是直接把 regime wedge 当作类别证据。评估时报告 cross-regime classification drop、policy-only predictability、regime-wedge stability 和 counterfactual-policy consistency，判断模型是否真正抵抗了 Sampling-Policy Shift。
