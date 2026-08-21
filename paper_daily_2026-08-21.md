# Paper Daily - 2026-08-21

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md` 和自动化记忆，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / clinical irregular time series classification 的顶会或顶会相关论文，重点核对 ICLR 2026、NeurIPS 2025 TS4H、OpenReview、ICLR virtual site、arXiv 与相关论文索引。
- 已排除全部黑名单论文；同时排除偏 forecasting / imputation-only、时间窗口偏旧或已在记忆中标记为重叠的条目。本次保留全新工作 2 篇：Random Controlled Differential Equations 是 ICLR 2026 Poster，虽不以 irregular sampling 为标题主轴，但其连续时间 reservoir 与 rough-path 表征天然适配非均匀采样路径，并在多变量时序分类上验证；ADEPT 是 ICLR 2026 submission，直接面向 time series classification 中缺失、格式异常和 irregular timestamps 的自动化表示学习，对采样政策信息是否会被原始记录表示吸收很有启发。

## 1. Random Controlled Differential Equations

- 简称：R-CDE / RF-CDE / R-RDE
- 会议：ICLR 2026 Poster
- 作者：Francesco Piatti, Thomas Cass, William F. Turner
- 官方页：https://iclr.cc/virtual/2026/poster/10007804
- OpenReview：https://openreview.net/forum?id=kHqt0ZSbKT
- arXiv：https://arxiv.org/html/2512.23670
- 代码：https://github.com/FrancescoPiatti/RandomSigJax
- 关键词：controlled differential equations, random features, rough paths, signature kernels, time-series classification, continuous-time reservoirs

### 场景、任务与核心难点

这篇工作面向一般时序学习，重点实验包括 UEA 多变量时序分类和 fractional Brownian motion 的 roughness / Hurst-exponent 分类。它不是专门为 ICU EHR 这类临床异步事件流设计的模型，但 CDE/RDE 的输入对象是连续时间路径，天然能接收非均匀时间戳构成的 control path，因此对 irregular sampled signal classification 有底层方法论意义。实际场景包括医疗传感器、工业监测和科学观测中“采样点不均、路径粗糙、标签依赖轨迹几何”的分类任务。

论文解决的核心难点是：Neural CDE/RDE 有很强的连续时间归纳偏置，但端到端训练昂贵；显式 signature kernel 又有特征计算和 Gram matrix 反演成本。作者把大规模随机参数化 CDE/RDE 当成 frozen continuous-time reservoir，只训练最后的线性 readout；并提出 RF-CDE 与 R-RDE 两个变体，前者先用 random Fourier features 提升输入信号再驱动 CDE，近似 RBF-lifted signature kernel，后者直接在 rough-path 输入上通过 log-ODE discretisation 和 log-signatures 捕捉高阶路径交互。这样在保留路径签名归纳偏置的同时，显著降低训练成本。

### 审稿人视角：价值与不足

最有价值的思想是把 continuous-time deep architecture、random feature reservoir 和 signature kernel theory 放进同一个框架。对审稿人而言，本文不是又提出一个更深的 CDE，而是给出了一个更可扩展的折中：随机 reservoir 固定、线性头可训练、无限宽极限可证明收敛到相应 signature kernel。这个设计尤其适合低数据或需要快速重训的分类场景，因为它把大部分时序几何建模能力封装进随机连续时间特征，避免每个任务都重新优化复杂 vector field。

不足在于，模型对采样机制的区分仍是隐式的。CDE/RDE 看到的是由观测点构造出的路径；若某一类别或某一医院策略导致观测更密集、路径更粗糙或某些事件更容易被记录，random reservoir 会忠实地把这些路径几何编码进特征，但不会自动判断它们是 patient/system state 还是 sampling policy artifact。随机特征提升了效率和理论可解释性，却牺牲了任务自适应的策略分解能力；论文也主要在通用分类 benchmark 和随机缺失鲁棒性上评估，尚未系统测试跨医院、跨采样频率或事件触发采样规则变化下的表示稳定性。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发在于：策略偏移可以被视为 control path signature / reservoir response 的分布偏移。不同采样政策会改变路径的局部增量、log-signature、高阶迭代积分和 reservoir 激活分布；这些量比简单 mask ratio 或 delta-t 均值更接近模型实际用于分类的连续时间几何。因此，可以把 RF-CDE/R-RDE 的随机特征响应、roughness 指标和 readout 权重稳定性作为策略偏移诊断工具。

纵向深化上，可以设计 state-policy 双 reservoir：一组 reservoir 只接收经反事实采样增强后仍稳定的 state path 表征，进入分类 readout；另一组 reservoir 专门接收采样时间、观测密度、变量可见性和路径粗糙度，用于 policy classification 或不确定性校准。对同一潜在轨迹施加不同采样策略时，要求 state reservoir 的线性可分结构和 logits 稳定，同时允许 policy reservoir 响应变化。这样能把 Random CDE 的效率和路径签名理论优势推进到“可审计的策略不变连续时间分类器”。

## 2. An Automated Data Engineering Pipeline for Time Series Classification Via Text Embeddings

- 简称：ADEPT
- 会议/状态：Submitted to ICLR 2026（OpenReview submission，尚非已确认正会录用）
- 作者：Anonymous authors under double-blind review
- OpenReview：https://openreview.net/forum?id=bX0Rw2uZgV
- 关键词：time series classification, text embeddings, raw format representation, variational information bottleneck, missing data, irregular timestamps, automated data engineering

### 场景、任务与核心难点

ADEPT 面向跨领域 time series classification 中高度工程化、格式不稳定的真实数据管线：医疗、金融、科学和工业 IoT 数据常常带有缺失、异常记录、损坏格式、非统一采样时间戳和多视图窗口。传统做法需要人工完成数据清洗、时间对齐、插补、归一化、特征提取和特征工程；这些步骤不仅成本高，而且每个领域、每种采样协议都需要重新定制，容易把数据工程假设变成隐形模型假设。

论文的核心想法是绕开大量手工数据工程：直接把原始时间序列文件或表格转成 textually dense raw format representation，再用 LLM-oriented text embedding model 生成表示；随后用 variational information bottleneck 过滤 text embedding 中的噪声和冗余，并接多头注意力分类器。作者的主张是，原始格式文本嵌入的熵可以在信息论意义上承载与数值化特征工程结果相当甚至更强的时空相关信息，从而在缺失、格式异常和 irregular timestamps 下仍能完成端到端分类。

### 审稿人视角：价值与不足

最有价值的思想是把“时序分类前处理”本身作为可替代的学习对象。许多 irregular time series 方法在模型结构上很复杂，但仍默认输入已经被清洗、对齐和规范化；ADEPT 则挑战这一点，尝试用 raw-format text embeddings 直接吸收原始记录中的时间、变量、缺失和格式信息，并用信息瓶颈控制表示熵。若这一方向成立，它可以显著降低跨数据集部署时的工程成本，也为异构时序数据提供一个统一入口。

不足同样明显。首先，ADEPT 当前是 ICLR 2026 submission，仍需等待评审确认；其次，text embedding 是强黑箱，数值精度、时间间隔、单位和变量名如何被编码并不透明。更重要的是，原始记录格式会完整保留采样政策痕迹：哪些变量出现、何时出现、字段顺序、缺失占位符、医院模板和设备导出格式都可能成为分类 shortcut。VIB 能压缩噪声，但若采样政策与标签高度相关，它不一定会主动丢弃这类高互信息但不可迁移的策略信号。论文强调对 irregular timestamps 和 data integrity issues 的鲁棒性，但尚未充分证明跨采样政策、跨医院记录模板或反事实观测流程下的稳定性。

### 对 Sampling-Policy Shift 的启发

ADEPT 对 Sampling-Policy Shift 的横向启发是：采样政策偏移不只存在于数值张量的 mask / delta-t 中，也存在于“数据被记录和序列化的方式”里。raw-format representation 会把时间戳、空值、列顺序、单位、注释和格式异常一起交给 embedding model；这可能提升鲁棒性，也可能让模型更容易利用医院或设备特定的数据管线痕迹。因此，研究策略偏移时应把 serialization / data-engineering layer 纳入偏移源，而不是只检查模型主干。

纵向深化上，可以把 ADEPT 改造成 policy-aware raw-format classifier：将原始记录解析为 state text 和 policy text 两个视图，前者只描述跨策略稳定的观测值趋势、异常和临床/系统状态，后者描述采样频率、缺失占位符、字段可见性和数据源格式。VIB 可从单一压缩目标扩展为双瓶颈：state bottleneck 保留对类别稳定有用的信息，policy bottleneck 保留观测流程信息但限制其直接进入分类头。再配合反事实 serialization augmentation，例如同一轨迹用不同采样时间、缺失标记和字段顺序重写，约束 state embedding 与 logits 稳定，就能把 ADEPT 的自动数据工程能力转化为面向 Sampling-Policy Shift 的表示压力测试和稳健学习框架。
