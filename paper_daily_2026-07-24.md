# Paper Daily - 2026-07-24

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / clinical or medical time-series classification / ICU prediction 的顶会论文与顶会 workshop 论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、OpenReview、ICLR virtual site、AAAI Proceedings、arXiv 与相关代码/项目页。
- 已排除全部黑名单论文；同时排除 PULSE 这类尚未确认 ICLR 正会录用的投稿、PULSE-ICU 这类仅有 arXiv 预印本的候选、TSPulse/CauKer 这类偏通用规则时序基础模型而非非规则分类主任务的候选，以及 LLapDiff、ASTGI、HELIX、T1 等偏 forecasting/imputation/generation 的条目。本次保留全新工作 2 篇：FORMED 是 ICLR 2026 正会 Poster，虽不显式解决稀疏 EHR 异步采样，但围绕医疗时序分类的跨通道、跨长度、跨任务泛化，与采样策略偏移下的可迁移分类头设计密切相关；STAR-Set 是 ICLR 2026 TSALM Workshop Poster，不是正会论文，但它直接面向 asynchronous clinical time series，并在摘要中明确指出 grid/imputation 可能引入 sampling-policy shortcuts，因此作为直接相关的前沿补充纳入。

## 1. Repurposing Foundation Model for Generalizable Medical Time Series Classification

- 简称：FORMED
- 会议：ICLR 2026 Poster
- 作者：Nan Huang, Haishuai Wang, Zihuai He, Marinka Zitnik, Xiang Zhang
- 官方页：https://iclr.cc/virtual/2026/poster/10006735
- OpenReview：https://openreview.net/forum?id=wNEzRYiyZM
- 论文：https://arxiv.org/abs/2410.03794
- 代码：https://github.com/DL4mHealth/FORMED
- 关键词：medical time series classification, foundation model repurposing, channel embeddings, label queries, cross-dataset generalization, lightweight adaptation

### 场景、任务与核心难点

FORMED 面向医疗时序分类的跨数据集泛化问题。典型应用是 ECG、EEG 等多通道医学信号中的诊断分类：不同数据集的通道数、采样长度、标签集合、患者群体和任务定义差异很大，导致一个在单一数据集上训练良好的分类器，迁移到另一个医疗场景时往往需要重训输入层、分类头甚至整个模型。

论文的核心难点不是传统 ICU/EHR 中变量级时间戳完全异步的 irregular sampling，而是更广义的医疗时序部署异质性：同一类医学分类任务可能有不同通道配置、不同长度窗口和不同标签空间。作者将通用时序基础模型作为冻结 backbone，再设计可动态适配的分类头：task-specific channel embeddings 负责接纳任意数量的通道，label queries 负责适配不同类别集合，shared decoding attention 在多数据集联合训练中沉淀医疗领域知识。新任务适配时只训练约 0.1% 参数，使模型可以在未见医疗数据集上轻量迁移。

### 审稿人视角：价值与不足

最有价值的思想是把医疗时序分类的泛化瓶颈从“重新训练一个任务专用模型”改写为“冻结通用表示，学习可组合的通道-标签查询接口”。很多医疗时序方法把通道、长度和标签固定在训练数据集内，FORMED 则把 channel embedding 和 label query 设计成任务可变的接口层，这对真实部署很重要：医院/设备/诊断任务变化时，分类器可以通过极少量参数更新完成适配，而不是推倒重来。

不足也需要明确：FORMED 主要面向连续或较规则的医学波形，不是专门为稀疏、异步、高缺失的 EHR/ICU 表格事件序列设计。换句话说，它解决的是跨数据集结构异质性，而不是完整的 observation process 建模。若不同医院的通道缺失、记录长度或数据预处理流程本身由采样政策决定，task-specific channel embeddings 可能吸收策略性可见性差异；shared decoding attention 也可能把某些数据集特有的采集协议当成医学领域知识。论文证明了跨数据集 MedTS 适配能力，但还没有系统评估在变量级异步采样、事件触发测量或跨医院采样政策变化下的稳健性。

### 对 Sampling-Policy Shift 的启发

FORMED 对 Sampling-Policy Shift 的横向启发是：分类器的“适配层”可以成为隔离采样政策差异的自然位置。与其让所有观测时间、通道可用性和标签语义直接进入统一 backbone，可以把可变通道、可变标签和可变采样协议集中到轻量 query/interface 层中处理；backbone 只承载更稳定的状态表征，策略差异则由独立的 policy queries 或 channel-policy embeddings 解释。

纵向深化上，可以把 FORMED 的 label query / channel embedding 扩展为 state-policy 双查询机制。state queries 学习跨采样策略稳定的诊断语义，进入分类主路径；policy queries 学习医院协议、设备配置、观测密度和通道缺失模式，只用于偏移诊断和校准。对同一潜在病程施加不同采样策略增强时，约束 state query 表征和 logits 保持一致，同时允许 policy query 区分采样环境。这样可以把 FORMED 的“跨任务轻量适配”推进到“跨采样政策轻量适配”。

## 2. Structure-Aware Set Transformers: Temporal and Variable-type Attention Biases for Asynchronous Clinical Time Series

- 简称：STAR-Set / Structure-Aware Set Transformer
- 会议：ICLR 2026 TSALM Workshop Poster
- 作者：Joohyung Lee, Kwanhyung Lee, Changhun Kim, Eunho Yang
- 官方页：https://iclr.cc/virtual/2026/10013883
- OpenReview：https://openreview.net/forum?id=AxXNor3Kd2
- 论文：https://arxiv.org/abs/2603.06605
- 关键词：asynchronous clinical time series, EHR, point-set tokenization, temporal attention bias, variable-type attention bias, sampling-policy shortcuts

### 场景、任务与核心难点

STAR-Set 面向电子健康记录中的异步临床时序预测/分类。EHR 通常不是规则矩阵，而是一组带时间戳的事件：化验、生命体征、用药或护理记录在不同时间、不同变量上稀疏出现。若把它们重采样成 time-by-variable 网格，需要插补和 missingness mask，容易引入误差或让模型利用采样政策捷径；若直接把每条观测当成 set token，又会丢失同一变量的局部轨迹、时间邻近关系和跨变量上下文。

论文解决的核心难点是如何在不强行网格化的情况下，把规则网格中有用的结构先验重新注入 point-set Transformer。作者提出两类参数高效的 soft attention biases：一类是 temporal locality penalty，根据时间间隔惩罚远距离 token，并学习不同时间尺度 tau；另一类是 variable-type affinity，通过可学习变量兼容矩阵刻画不同临床变量之间的交互倾向。它在 MIMIC-IV 的 CPR、mortality、vasopressor use 等 ICU 任务上优于 regular-grid、event-time grid 和已有 set baseline，同时 learned tau 和变量兼容矩阵提供一定可解释性。

### 审稿人视角：价值与不足

最有价值的技术思想是把“set tokenization 的灵活性”和“grid layout 的结构归纳偏置”用 attention bias 连接起来。相比重新设计复杂 EHR backbone，STAR-Set 的修改很轻量：只在注意力 logits 上加入时间局部性和变量类型偏置，就能缓解纯 set 表示丢失轨迹结构的问题。更重要的是，论文明确指出 grid/imputation 可能制造 sampling-policy shortcuts，这一点与非规则采样下的偏移风险高度一致。

不足在于，STAR-Set 目前是 workshop 短文，实验和消融深度有限；它虽然识别了 sampling-policy shortcuts 的风险，但方法本身仍会从观测时间和变量类型共现中学习预测信号。若某些变量在训练医院中只在高风险患者上被测，variable-type affinity 可能学到流程共现而非稳定生理关系；temporal locality tau 也可能混合病程速度与记录频率。现有评估主要说明结构偏置提升同数据集 ICU 任务表现，还没有跨医院、跨记录频率、跨检查触发规则的系统鲁棒性测试。

### 对 Sampling-Policy Shift 的启发

STAR-Set 对 Sampling-Policy Shift 的横向启发非常直接：采样策略偏移可以被看作 attention bias 分布的偏移。不同医院或设备会改变时间邻近 token 的密度、变量共现矩阵和有效上下文尺度；如果模型的 tau、变量兼容矩阵或注意力权重在环境间大幅漂移，就说明分类器可能依赖了策略特定的观测结构。

纵向深化上，可以把 STAR-Set 的 temporal / variable-type biases 分解成 state bias 与 policy bias。state bias 只建模跨策略稳定的临床时间尺度和生理变量关系，进入分类注意力；policy bias 建模采样密度、联测习惯和记录流程，只用于偏移检测或不确定性校准。训练时可对同一潜在病程生成多种采样策略视图，约束 state bias 下的 attention map 与 logits 稳定，同时允许 policy bias 识别环境。这样能把“给 set Transformer 注入结构先验”推进到“给 set Transformer 注入可分解的状态-策略先验”。
