# Paper Daily - 2026-07-13

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / continuous-time irregular sequence classification 的顶会论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、NeurIPS 2025、KDD 2025/2026 官方页、OpenReview、ACM DOI 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除偏 forecasting、普通规则时序分类、ICML/KDD workshop 条目，以及 KDD 2025 中时间窗口偏旧但相关的 ISTS-PLM/HCIB 候选。本次保留全新工作 1 篇：Efficient Neural CDE via Attentive Kernel Smoothing 是 ICML 2026 正会论文，虽然标题不直接写 classification，但论文问题设定和实验明确覆盖监督分类，并聚焦不规则观测下 Neural CDE 的控制路径构造与求解效率。

## 1. Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing

- 简称：MV-CDE / MVC-CDE
- 会议：ICML 2026 Poster
- 作者：Egor Serov, Ilya Kuleshov, Alexey Zaytsev
- 官方页：https://icml.cc/virtual/2026/poster/62701
- 论文：https://arxiv.org/html/2602.02157
- 关键词：irregularly sampled time series, Neural CDE, kernel smoothing, Gaussian Process smoothing, multi-view attention, classification efficiency

### 场景、任务与核心难点

这篇工作面向不规则观测序列上的监督学习，实验主要评估多变量时序分类，并以 CharacterTrajectories、SpokenArabicDigits、UWaveGestureLibrary 等 UEA/UCR 分类数据为 benchmark。它关注的不是传统“如何补齐缺失值”，而是 Neural CDE 在处理不规则采样时一个更底层的问题：离散观测必须先被提升为连续 control path，而常用线性/三次样条插值会强行穿过每个观测点，把噪声、高频抖动和采样不均匀性一起变成非常粗糙的驱动路径。

这种粗糙路径会让自适应 ODE solver 为了控制局部误差而频繁缩小步长，导致 Number of Function Evaluations 和推理时间显著上升。作者因此用 Kernel / Gaussian Process smoothing 替代精确插值，显式控制 control path 的 regularity；为了避免过度平滑丢失判别性细节，再引入 learnable queries 的 Multi-View CDE 和卷积版 MVC-CDE，让多个平滑视图分别捕捉不同时间尺度或局部模式。最终目标是在保持甚至提升分类准确率的同时，显著降低 Neural CDE 的求解成本。

### 审稿人视角：价值与不足

最有价值的技术思想是把 Neural CDE 的效率瓶颈从“solver 本身慢”进一步定位到“control path 几何太粗糙”。很多连续时间模型默认把插值视为无害前处理，然后再优化 vector field 或 solver tolerance；这篇论文指出，只要驱动路径继承了噪声和非均匀采样带来的高频变化，solver 就会被迫沿着复杂几何前进。用平滑路径降低几何复杂度，再用多视图注意力补回高频信息，是一个很清晰的 accuracy-efficiency trade-off 设计。

不足在于，论文主要把 irregularity 看作数值求解和路径光滑性问题，还没有显式区分“真实状态变化导致的高频事件”与“采样政策/传感器策略导致的观测粗糙”。如果某些类别在训练环境中被更密集采样，或者某些变量只在告警后出现，平滑路径可能把策略诱导的观测密度差异压缩成看似稳定的低频趋势；多视图 attention 也可能学习到与采样政策相关的局部可见性模式。论文证明 MVC-CDE 在分类准确率和 NFE 上更优，但尚未系统测试换采样策略、换观测触发规则后，平滑视图和 attention head 是否保持语义稳定。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：采样策略偏移会改变 control path roughness，从而同时影响表示学习和数值求解成本。也就是说，策略偏移不只体现在 mask ratio、delta-t 分布或变量共现图上，还可能体现在 solver 需要多少步、哪些时间段产生高曲率路径、哪些平滑带宽最有效。NFE、路径曲率、GP smoothing bandwidth、multi-view attention 分布都可以成为诊断采样政策偏移的辅助指标。

纵向深化上，可以把 MVC-CDE 改造成 policy-aware path smoothing：一组 state views 用于保留跨策略稳定的真实动力学，一组 policy views 用于解释观测调度、噪声水平和采样密度变化。对同一潜在轨迹生成不同采样策略视图时，可约束 state control path、分类 logits 和关键 attention head 保持一致，同时允许 policy view 的粗糙度和不确定性变化。这样既保留平滑路径带来的可扩展性，又避免把“某种策略下更容易求解/更平滑”的路径误当成可迁移类别证据。
