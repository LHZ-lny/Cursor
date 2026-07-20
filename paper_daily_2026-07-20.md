# Paper Daily - 2026-07-20

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / EHR / clinical time series classification / sampling-policy shift / multi-center clinical prediction 的顶会论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、KDD 2025/2026、OpenReview、ICML/ICLR virtual site、AAAI Proceedings、ACM DOI 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除 Time-CoT 这类普通规则 MTS classification，ReIMTS、ASTGI、LLapDiff、HELIX、ReDiTT 等偏 forecasting / imputation / generation 的候选，STAR-Set、EHR-SPC 等 ICLR 2026 workshop 条目，以及 CT-Former 这类未能在官方 ICML virtual 页面核实正会身份的候选。本次保留全新工作 2 篇：Time-Conditioned Foreseeing 是 ICML 2026 正会 EHR foundation model，直接处理 irregular timestamps 与 calendrical time；AdaTTT 是 ICLR 2026 正会多中心 ICU clinical prediction 工作，虽不以 irregular sampling 为标题主轴，但正面处理 EHR 部署中的跨机构临床流程/系统偏移，对 Sampling-Policy Shift 有直接方法论启发。

## 1. Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time

- 会议：ICML 2026
- OpenReview 标题：Time Conditioned Foreseeing: Temporal Generative Pretraining for EHR foundation models
- 作者：Bong Gyun Kang, Junyong Ahn, Hyeongrok Han, Sungroh Yoon
- OpenReview：https://openreview.net/forum?id=Z8Hu7CJfZy
- 官方页：https://icml.cc/virtual/2026/day/7/9
- 关键词：EHR foundation model, irregular timestamps, calendrical time, time-conditioned foreseeing, temporal generative pretraining, clinical prediction

### 场景、任务与核心难点

这篇工作面向纵向 EHR 序列建模与临床预测。EHR 与普通文本序列的关键差异在于，token 不是等间距出现：化验、诊断、用药和生命体征事件由临床流程、患者状态、门急诊节律和医院资源共同触发；相邻事件之间的时间间隔可能从分钟到数月不等，绝对日历时间也有临床含义，例如昼夜节律、工作日/周末、门诊复诊周期和治疗计划窗口。下游任务覆盖多种临床预测/分类与事件预测场景，因此模型需要同时理解“发生了什么”和“何时发生/何时会发生”。

核心难点是，现有 EHR foundation model 往往照搬 NLP 的 next-token prediction，把事件当作离散 token 流，却弱化了连续时间间隔和日历上下文。作者提出三件事：Pathology-Focused Binning 将数值变量离散化时更强调临床关键范围，而不是平均切分数值空间；Dual-Calendar RoPE 同时编码相对时间间隔与绝对日历上下文；Time-Conditioned Foreseeing (TCF) 目标则不只预测下一个 token，而是联合学习下一事件何时发生以及给定未来时间条件下会出现什么事件，从而把 EHR 预训练目标改造成更贴近临床计划和长期随访的时间生成任务。

### 审稿人视角：价值与不足

最有价值的思想是把 EHR 中的时间从“位置编码附属变量”提升为预训练目标的一部分。Dual-Calendar RoPE 解决的是输入表达问题，TCF 解决的是训练信号问题：如果模型必须预测事件时间和未来时间条件下的事件内容，它就被迫学习临床轨迹中的节律、等待时间、检查周期和病程演化，而不是只学 code 共现。对审稿人而言，这比单纯在 Transformer 上拼接 delta-t 更有意义，因为它把 irregular timestamp 的统计结构纳入生成式因果链条。

不足在于，这种优势也可能放大采样政策信号。EHR 中“下一次何时测量”常常不是单纯由患者生理状态决定，还受到医院排班、检查套餐、医保、科室习惯和告警流程影响。TCF 若在单一或少数机构上预训练，可能把特定机构的复诊周期、周末低采样、某些检查的触发规则学成通用病程知识。论文强调 irregular dynamics 与 calendrical time，但还需要更强的跨医院、跨科室、跨采样协议评估，才能证明模型学到的是可迁移临床状态，而不是部署环境特有的测量节律。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的纵向启发是：采样政策本身可以被建成一个显式的 time-to-observation / marked event process，而不是只作为 mask 或 delta-t 特征输入分类器。TCF 的“何时发生 + 发生什么”联合目标可被扩展为 state-policy factorization：state process 解释真实病程和风险，policy process 解释医院为何在某时刻测量某变量、记录某事件或延迟某检查。

横向应用上，可以设计反事实采样版 TCF：固定同一潜在病程，替换不同医院或不同资源约束下的观测时间表，要求 state representation 和临床分类 logits 保持稳定，同时允许 policy representation 预测不同的下一次观测时间与变量类型。Dual-Calendar RoPE 也提示我们，sampling-policy shift 不只是 delta-t 分布漂移，还包括日历制度漂移：夜班、周末、节假日、门诊周期和 ICU 常规化验时间都会改变可观测轨迹。若能把这些制度性时间因素从状态表征中剥离出来，模型对非规则采样下的策略偏移会更稳健。

## 2. Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts

- 简称：AdaTTT
- 会议：ICLR 2026 Poster
- 作者：Xiaolei Lu, Shamim Nemati
- 官方页：https://iclr.cc/virtual/2026/poster/10007659
- 论文：https://arxiv.org/html/2512.06652v2
- 关键词：EHR clinical prediction, invasive mechanical ventilation, multi-center cohorts, domain shift, test-time training, dynamic masking, partial optimal transport

### 场景、任务与核心难点

这篇工作面向 ICU 多中心队列中的有创机械通气需求预测，本质上是 EHR-based clinical classification / risk prediction：给定患者近期临床记录，提前预测是否需要 IMV，以支持干预和资源分配。困难不只来自标签稀缺或类别不平衡，更来自部署时的跨中心偏移：不同医院的人群构成、EHR 字段、护理流程、检查频率、通气阈值和记录习惯都可能不同，使源医院训练好的模型到目标医院后显著退化。

论文解决的核心难点是：目标医院推理时通常没有标签，也不现实重新完整训练模型，但模型又必须适应新的 EHR 分布。AdaTTT 在训练阶段联合主任务分类与自监督辅助任务，在测试阶段对每个目标样本进行少量 test-time updates。辅助任务包括 reconstruction 与 masked feature modeling，并通过 dynamic masking 强调与主任务更相关的特征；理论上，作者用主任务与辅助任务不确定性之间的关系推导 test-time error bound；实践上，再结合 prototype learning 与 Partial Optimal Transport (POT)，使目标样本表征只与临床相关的源域原型做柔性对齐，而不是强行全分布匹配。

### 审稿人视角：价值与不足

最有价值的技术思想是把临床 EHR 部署偏移从“训练前要解决的问题”推进到“推理时可自适应的问题”。很多 domain generalization 方法要求训练时预先见过足够多环境，或需要目标域无标签批数据；AdaTTT 更贴近医院上线场景：每个新患者到来时，用自监督目标做局部更新，然后再输出风险预测。dynamic masking 也比随机 masked modeling 更合理，因为它让辅助任务与 IMV 分类任务对齐，减少 test-time training 学到与主任务无关的重构细节。

不足在于，论文主要把跨中心差异建模为表征分布偏移，还没有显式拆解其中哪些来自患者状态、哪些来自采样政策。dynamic masking 强调 task-critical features，但如果某些特征在源医院中因特定通气流程或检查协议而变得“任务关键”，测试时的局部更新可能进一步强化这种 policy shortcut。POT 对齐源域原型有助于稳住临床语义，但如果原型本身混入医院测量习惯，对齐也可能把目标医院患者拉回源医院政策空间。此外，per-sample test-time update 带来计算、稳定性和安全性问题，临床部署需要额外监控错误更新与置信度漂移。

### 对 Sampling-Policy Shift 的启发

AdaTTT 对 Sampling-Policy Shift 的横向启发是：采样政策偏移很适合在 test-time adaptation 框架下处理，因为真实部署中我们往往只能看到目标医院的无标签观测流，不能提前知道其完整采样规则。可以把 dynamic masking 改造成 policy-aware masking：一部分 mask 任务学习恢复跨策略稳定的生理状态，另一部分 mask 任务专门预测观测是否出现、何时出现和哪些变量联测，用于估计目标环境采样政策。

纵向深化上，POT 原型对齐可以从“全表示对齐”改为“state prototype 对齐 + policy residual 保留”。也就是说，目标样本在测试时只把 state representation 对齐到源域临床状态原型，而把观测密度、变量可用性、周末/夜班采样、医院流程等 policy residual 留在单独分支，用于校准和偏移报警，不直接进入分类边界。这样能把 AdaTTT 的部署适应能力与 sampling-policy disentanglement 结合起来：模型可以快速适应新医院的观测制度，同时避免把目标医院的采样规则误更新成疾病证据。
