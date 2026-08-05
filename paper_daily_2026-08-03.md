# Paper Daily - 2026-08-03

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / variable-length / event-sequence time series classification 的顶会或顶会 workshop 论文，重点核对 ICLR 2026 TSALM、ICML 2026 SPIGM、ICLR 2026 OpenReview、ICML 2026、AAAI 2026、KDD 2026、NeurIPS 2025/TS4H 页面、OpenReview、arXiv 与 workshop accepted paper 页面。
- 已排除全部黑名单论文；同时排除 MedFuse/HT-Transformer/ReasonTSC 等 withdrawn submission，ReDiTT/FITS/Under-Cali 等偏 forecasting 或 generation 的工作，ClinPRISM 这类 QA 而非分类主任务且暂无顶会录用信息的 arXiv 预印本，ReTAMamba/TANDEM/统计特征建模等暂无顶会状态的预印本，以及 RAxSS 这类虽相关但主要落在 NeurIPS 2025 workshop/时间窗口略偏旧的候选。本次保留全新工作 2 篇：CA-NSDE 是 ICML 2026 SPIGM workshop poster，直接面向 robust irregular time-series classification；BiCarFormer 是 ICLR 2026 TSALM workshop poster，面向异步诊断事件序列与连续环境传感信号的多模态序列分类。

## 1. Context-Aware Neural SDEs for Robust Irregular Time-Series Classification

- 简称：CA-NSDE
- 会议：ICML 2026 SPIGM Workshop Poster
- 作者：YongKyung Oh, Alex Bui
- OpenReview：https://openreview.net/forum?id=LwHuqdidzW
- Workshop accepted list：https://spigmworkshop2026.github.io/papers/
- 关键词：irregular time-series classification, Neural SDE, context-aware dynamics, missingness robustness, stochastic continuous-time model

### 场景、任务与核心难点

CA-NSDE 面向不规则采样和缺失观测下的时序分类。典型场景包括医疗监测、科学观测和传感器系统：训练时可见的观测密度、时间间隔和缺失结构，与部署时传感器掉线、观测成本变化或环境条件改变后的观测过程并不一致。相比只追求同分布 accuracy，这类任务更关心模型在更稀疏、更不均匀或缺失机制变化时能否保持分类稳定，并且能否给出合理的不确定性。

论文从 Neural SDE 路线切入：连续时间随机微分方程适合处理任意时间戳和不确定动力学，但朴素 Neural SDE 容易受 drift/diffusion 设计、数值稳定性和缺失输入扰动影响。CA-NSDE 的核心思想是在稳定 Neural SDE backbone 上加入 time-varying context，让模型不只根据观测值更新潜在状态，也根据当前观测环境、稀疏程度或上下文条件调整动力学。这样做试图把 irregularity 从单纯噪声提升为影响连续时间演化的条件变量，从而提高 sparse / missing setting 下的分类鲁棒性。

### 审稿人视角：价值与不足

最有价值的思想是把“稳定连续时间动力学”和“上下文条件化”合在一起处理不规则分类。既有 Neural CDE / Neural ODE 方法常把观测路径构造好以后交给统一 dynamics；稳定 Neural SDE 进一步建模随机扰动，但如果 diffusion 或 drift 不看采样上下文，仍可能在采样密度变化时给出过度自信或错误的状态演化。CA-NSDE 把 context 显式放进动力学层，提供了一个更自然的接口来表达“同样的观测值在不同观测可靠性、缺失程度或时间间隔下含义不同”。

不足在于，公开 workshop 信息显示其关注 robustness，但采样政策与真实状态的可分性仍需要更强实证拆解。context-aware 机制若只以 missingness、delta-t 或观测密度作为输入，可能提升在缺失扰动下的平均性能，却也可能把训练环境中的采样政策编码进 drift/diffusion。尤其当观测频率本身由病情、设备告警或机构流程触发时，context 既是有用的可靠性信号，也是潜在 shortcut。作为 workshop poster，它还需要更完整的跨环境、反事实采样策略和 policy-only baseline 来证明鲁棒性来自状态建模，而不是来自更灵活地利用采样模式。

### 对 Sampling-Policy Shift 的启发

CA-NSDE 对我们的问题有直接纵向启发：sampling-policy shift 可以进入连续时间模型的动力学方程，而不仅是 encoder 的输入层。可以把 Neural SDE 写成 state drift/diffusion 与 policy drift/diffusion 两部分：前者描述真实潜在状态随时间演化，后者描述采样政策导致的观测可靠性、噪声和时间间隔变化。分类头只依赖 state posterior，而 policy component 用于校准不确定性和诊断分布偏移。

横向应用上，CA-NSDE 提示我们评估 policy shift 时应同时看 accuracy、calibration 和 dynamics sensitivity。对同一潜在轨迹施加不同采样策略时，若 state drift、state diffusion 和分类 logits 大幅变化，说明模型仍把采样策略当作状态证据；若 policy diffusion 或观测噪声分支变化但 state posterior 稳定，则更符合 sampling-policy-invariant 的目标。这个方向也适合和反事实采样增强结合：固定连续状态轨迹，改变观测时刻和变量可见性，约束 state SDE 解的一致性，同时允许 context/policy 分支解释不同采样制度。

## 2. Context-Informed Sequence Classification: A Multimodal Approach to Vehicle Diagnostics

- 方法名：BiCarFormer
- 会议：ICLR 2026 TSALM Workshop Poster
- 作者：Hugo Math, Rainer Lienhart
- OpenReview：https://openreview.net/forum?id=G4iAE9xOpb
- arXiv/DOI：https://doi.org/10.48550/arxiv.2602.01109
- 关键词：asynchronous event sequences, multimodal time series, vehicle diagnostics, Diagnostic Trouble Codes, co-attention, multi-label sequence classification

### 场景、任务与核心难点

BiCarFormer 面向车辆故障诊断中的多标签序列分类：输入包括异步出现的 Diagnostic Trouble Codes (DTCs) 事件序列，以及温度、压力、湿度等连续环境传感信号。真实车队数据不是规则采样的单一传感器矩阵，而是高维离散故障码、事件发生时间、里程进程和噪声较大的环境变量共同组成的异步记录。任务是把这些记录分类到 360 类 error patterns，数据规模包含 22,137 个 error codes。

核心难点在于，单独依赖 DTC 序列会丢失专家诊断中常用的环境上下文：同一个故障码组合在不同温度、压力或湿度条件下可能对应不同故障模式；但直接拼接连续传感器又会遇到模态尺度不一致、采样时间不同步、噪声和高维上下文冗余的问题。BiCarFormer 使用 bidirectional Transformer 处理诊断事件序列，并通过 tokenized sensory data、special embeddings 和 co-attention 机制融合环境上下文，使模型能在离散事件模式和连续环境波动之间建立交互关系。

### 审稿人视角：价值与不足

最有价值的技术思想是把异步事件序列分类从“只看事件 token”推进到“事件-环境上下文共同判别”。在医疗 EHR、工业告警、车辆诊断等场景中，事件是否出现只是表层记录，事件发生时的上下文才决定其语义。BiCarFormer 的 co-attention 使 DTC 与环境变量互相查询，避免把环境信号当作简单附加特征；这种设计比后期拼接 embedding 更有可能捕捉“某个故障码在某种环境条件下才有判别力”的条件模式。

不足在于，这篇是 industry/application workshop 论文，方法和评估主要围绕单一大规模车队数据展开。它证明多模态融合优于单模态基线，但尚未系统评估跨车厂、跨传感器记录频率、跨维修策略或跨地理环境下的泛化。环境传感器本身也可能携带 policy shortcut：某些车辆、区域或维修流程会改变哪些传感器被记录、DTC 何时触发以及数据上传频率。若 co-attention 未区分稳定物理状态与机构/设备采样政策，模型可能学到的是车队运维流程而不是真正可迁移的故障机理。

### 对 Sampling-Policy Shift 的启发

BiCarFormer 对 Sampling-Policy Shift 的横向启发是：非规则采样分类中的 policy 不一定只表现为 missing mask，也可能表现为“事件通道与上下文通道的联合可见性”。DTC 触发、环境传感器采样、上传频率和里程记录共同定义了观测政策；同一潜在故障在不同车队或不同维护规则下，可能产生不同的事件-环境共现模式。医疗场景中的化验下单、护理记录和生命体征监测也有类似结构。

纵向深化上，可以把 BiCarFormer 改造成 state-context-policy 三分支模型：事件和连续值先分别编码，state co-attention 只学习跨策略稳定的故障/病程状态关系，policy co-attention 学习哪些事件与上下文为什么被共同观测，最终分类头对 policy branch 做不变性或门控约束。评估时可以构造反事实采样：固定 DTC 或临床事件语义，改变环境/生命体征采样频率、上传延迟和上下文缺失模式，检查 state attention 与 logits 是否稳定。这样能把“多模态异步序列分类”推进到“跨观测政策稳定的多模态事件分类”。
