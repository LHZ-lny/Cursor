# Paper Daily - 2026-07-14

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / medical time series classification / sampling-policy shortcut 的顶会论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、NeurIPS 2025 Datasets & Benchmarks、OpenReview、ICLR/ICML virtual site、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除偏 forecasting-only 的 ASTGI、MoRGen、Time-IMM，偏 imputation-only 的 T1/HELIX，ICLR workshop 的 STAR-Set，以及 withdrawn 的 MedFuse/MLLM4TS。本次保留全新工作 1 篇：FORMED 是 ICLR 2026 正会医疗时序分类论文，虽然标题不直接写 irregular sampling，但其核心问题是医疗时序在通道数、长度、任务定义和患者/数据集差异下的跨数据集泛化，与非规则采样下的 Sampling-Policy Shift 高度相邻。

## 1. Repurposing Foundation Model for Generalizable Medical Time Series Classification

- 简称：FORMED
- 会议：ICLR 2026 Poster
- 作者：Nan Huang, Haishuai Wang, Zihuai He, Marinka Zitnik, Xiang Zhang
- 官方页：https://iclr.cc/virtual/2026/poster/10006735
- OpenReview：https://openreview.net/forum?id=wNEzRYiyZM
- 论文：https://arxiv.org/html/2410.03794v2
- 关键词：medical time series classification, foundation model repurposing, cross-dataset generalization, channel heterogeneity, label queries, lightweight adaptation

### 场景、任务与核心难点

FORMED 面向医疗时序分类在真实部署中的跨数据集泛化问题。典型场景不是单一 ICU irregular EHR benchmark，而是不同医疗数据集之间存在通道数不同、信号长度不同、任务标签不同、患者群体不同和诊断目标不同的异质性：一个模型若只针对某个数据集的固定通道和固定标签训练，迁移到新医院、新设备或新疾病任务时往往需要重新设计输入层、分类头和微调流程。

论文解决的核心难点是如何把通用时间序列 foundation model 重新用于医疗分类，并让它在 unseen MedTS datasets 上只需极少参数更新即可适配。FORMED 保留一个在通用时序上预训练的 backbone，在其上加入两类可动态伸缩的任务组件：task-specific channel embeddings 与 label queries，用于匹配任意数量的通道和类别；同时加入 shared decoding attention layer，在多数据集联合训练中学习 task-agnostic 的医学特征-标签交互。部署到新数据集时，主要训练轻量 label query，论文报告只需约 0.1% 参数即可适配，并在 5 个医疗时序数据集、11 个 task-specific model 和 4 个 task-specific adaptation baseline 上验证。

### 审稿人视角：价值与不足

最有价值的技术思想是把医疗时序分类中的“泛化难”拆成两个层次：底层表示尽量复用通用时序 foundation model，高层任务接口则用可变 channel embeddings 和 label queries 吸收数据集/任务差异。相比为每个医疗数据集训练专用模型，FORMED 更接近真实部署需求：新医院的通道配置、信号长度和标签空间变化时，不必重新训练整个 backbone，而是通过轻量查询和共享解码层完成任务重绑定。这种“domain-invariant representation + task-specific query adaptation”的设计，对医疗时序分类的可扩展性比单个 benchmark accuracy 更有价值。

不足在于，FORMED 主要处理的是医疗时序的结构异质性和任务异质性，对变量级异步时间戳、事件触发采样和 missingness mechanism 的显式建模仍然较弱。换言之，它能适配“有哪些通道、多少标签、序列多长”这类可见结构变化，但不一定能区分“为什么某个通道在某个时刻被测”。如果训练集中的某些通道配置、采样频率或标签定义本身携带医院政策信息，channel embeddings 和 label queries 可能把这种政策差异编码成任务知识。论文强调跨数据集泛化和轻量适配，但还没有系统评估跨采样政策、跨测量协议或反事实观测调度下的分类稳定性。

### 对 Sampling-Policy Shift 的启发

FORMED 对 Sampling-Policy Shift 的横向启发在于：采样策略偏移不只是 mask/delta-t 的低层统计偏移，也常常与任务接口变化绑定出现，例如医院 A 有某些常规化验通道、医院 B 没有；某个队列标签定义更粗，另一个队列标签定义更细；某些设备产生更长或更短的观测窗口。因此，policy-robust irregular classification 需要同时解决表示层不变性和任务接口可适配性，而不是只在固定输入维度上做 mask augmentation。

纵向深化上，可以把 FORMED 的 channel embeddings 和 label queries 改造成 policy-aware adaptation 层：state channel queries 负责抽取跨采样政策稳定的生理/系统状态，policy channel queries 负责解释通道可用性、采样频率、观测窗口和医院协议差异；label queries 则在适配新任务时被约束为主要读取 state queries，而不是直接依赖 policy queries。训练时可对同一底层病程生成不同采样策略视图，要求 backbone state representation 与主要 logits 稳定，同时允许 policy queries 区分环境/协议。这样可以把 FORMED 的“跨数据集轻量适配”推进到“跨采样政策轻量适配”：新环境只需校准策略支路和标签查询，而不重新学习疾病状态表示。
