# Paper Daily - 2026-07-19

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / sparse healthcare time series classification / event detection 的顶会论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、KDD 2025/2026、OpenReview、ICML virtual site、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除 MedFuse 这类 withdrawn submission、STAR-Set 这类 ICLR workshop 条目，以及 LLapDiff、ASTGI、APN、ReIMTS、MOSES、MN-Diff 等偏 forecasting/generation/imputation 而非分类主任务的候选。本次保留全新工作 1 篇：Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction 是 ICLR 2026 正会 Poster，虽然任务形式是事件检测而非整条序列分类，但它明确联合定位事件边界与分类事件类型，且面向临床稀疏事件，在异步/非规则医疗信号分类与 sampling-policy shift 问题上有较强横向价值。

## 1. Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction

- 会议：ICLR 2026 Poster
- 作者：Beomjun Bark, Yun Kwan Kim
- OpenReview：https://openreview.net/forum?id=DulnZ7Dv82
- 官方页：https://iclr.cc/virtual/2026/poster/10010733
- 代码：https://github.com/hbumjj/CDI-TS-Event-Detection
- 关键词：sparse healthcare time-series, event detection, event type classification, boundary localization, adaptive gating, context-detail interaction

### 场景、任务与核心难点

这篇工作面向医疗时序中的稀疏临床事件检测，任务不是只判断一整条序列的类别，而是同时定位事件起止边界并分类事件类型。论文评估的场景包括心律失常检测、情绪识别和活动监测等 healthcare time-series：真正有诊断价值的片段在长序列中占比极低，事件边界模糊，类别分布稀疏，临床上又要求模型给出可操作的时间位置，而不是只输出一个全局风险分数。

核心难点在于，DETR 类检测框架在图像目标检测中能通过 query 匹配定位对象，但直接迁移到医疗时序时会遇到极端事件稀疏：大部分时间窗口是背景，局部高频细节容易被全局上下文淹没；如果始终启用细粒度检测分支，又会在大量无事件区域引入噪声和计算浪费。作者因此提出 coarse-to-fine 框架，由 global context explorer 先建模长程背景和事件可能性，local detail inspector 负责精细边界与事件形态，再用 Adaptive Gating Module (AGM) 作为上下文-细节交互开关。AGM 利用 transformed labels，把事件是否存在、事件位置和原始类别标签转成多视角监督，使模型只在事件可能出现时强化局部细节提取，从而提升极稀疏事件的检测和分类能力。

### 审稿人视角：价值与不足

最有价值的技术思想是把“稀疏事件分类”显式拆成全局筛查与局部精查两种计算模式，并用标签变换驱动的 gate 学习二者何时交互。相比在所有时间点平均施加同样的注意力或检测 query，AGM 更符合医疗监测流程：先判断是否存在可疑片段，再在可疑区域进行边界级和类型级判别。对审稿人而言，这个设计的优势不只是指标提升，还在于它把稀疏性从数据缺陷转化为模型结构先验，使 rare-event learning 不再完全依赖 loss reweighting 或更多负样本采样。

不足在于，论文主要处理“事件在时间轴上稀疏”的问题，并没有把观测过程本身的非规则采样、传感器缺失或医院测量政策作为显式变量。心律失常、活动或情绪数据中的稀疏事件不一定等同于 ICU/EHR 中由医生决策触发的异步化验；如果事件片段更容易被设备高频记录、人工标注或特定监测策略覆盖，AGM 学到的 gate 可能同时反映真实临床事件和采样/标注流程。论文证明了 sparse event detection 的有效性，但还缺少跨设备、跨医院、跨采样频率或跨告警触发规则下 gate 稳定性的系统评估。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以被看作一种“门控触发分布”的偏移。现实医疗系统中，医生或设备并不是均匀观察病人，而是先由粗粒度风险、报警阈值或资源约束触发更密集的局部测量；这与论文中的 global context explorer 触发 local detail inspector 在结构上非常相似。因此，我们可以把采样政策建模为一个 policy gate：它决定哪些时间段、哪些变量会进入高分辨率观测，而分类模型需要区分 gate 是由真实状态驱动，还是由环境/医院流程驱动。

纵向深化上，可以把 AGM 扩展为 state-policy 双门控框架。state gate 负责捕捉跨采样策略稳定的真实事件或病程变化，进入分类主路径；policy gate 负责解释为何某些时间段被密集观测、为何某些变量被联测或为何某些事件更容易被标注，只用于偏移诊断和不确定性校准。训练时可对同一潜在病程施加不同采样策略增强，约束 state gate、事件类型 logits 和关键边界表示保持稳定，同时允许 policy gate 随观测策略改变。这样能把稀疏事件检测中的 context-detail interaction 推进到非规则采样下的策略不变事件分类。
