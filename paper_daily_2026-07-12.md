# Paper Daily - 2026-07-12

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / asynchronous clinical time series 的顶会或顶会 workshop 论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、KDD 2026、WWW 2026、OpenReview、ICLR/ICML virtual site、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除偏 forecasting 的 U-Former ODE、偏 regular MTSC 且非原生异步采样建模的 AAAI 2026 SVGL、时间窗口偏旧的 Hi-Patch、以及已被历史日报覆盖的 MTM/MILM/GSNF/QuITE/DBGL 等条目。本次仅保留全新 direct hit 1 篇：STAR-Set 是 ICLR 2026 TSALM Workshop Poster，虽然不是主会正会，但直接面向异步 EHR/ICU 分类，并且在摘要中明确指出 grid 与 mask 方案可能引入 sampling-policy shortcuts，因此与本轮主题高度相关。

## 1. Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series

- 简称：STAR-Set Transformer
- 会议/状态：ICLR 2026 TSALM Workshop Poster
- 作者：Joohyung Lee, Kwanhyung Lee, Changhun Kim, Eunho Yang
- ICLR 页面：https://iclr.cc/virtual/2026/10013883
- OpenReview：https://openreview.net/forum?id=AxXNor3Kd2
- 论文：https://arxiv.org/html/2603.06605
- 关键词：asynchronous clinical time series, EHR, set transformer, temporal locality bias, variable-type attention bias, ICU prediction, sampling-policy shortcuts

### 场景、任务与核心难点

STAR-Set 面向电子健康记录中的异步临床时序分类/预测任务，实验覆盖 ICU 场景下的 CPR、院内死亡风险和 vasopressor use 等结局预测。输入不是规则矩阵，而是由不同变量、不同时间戳组成的事件集合：生命体征、化验、治疗记录和其他临床事件以不同频率出现，且观测是否发生本身常受病情、医嘱和医院流程影响。

论文抓住的核心难点是输入布局选择。规则网格能显式保留 time x variable 结构，但需要插补、离散化或 mask，容易引入误差和 sampling-policy shortcut；点集 tokenization 则避免了强制对齐，却丢掉变量内轨迹、局部时间上下文和变量类型之间的结构先验。STAR-Set 在 set transformer 的 attention logits 中加入两类参数高效的 soft biases：一类是基于 `-|delta t| / tau` 的 temporal locality penalty，并让时间尺度 `tau` 可学习；另一类是基于变量类型兼容矩阵的 variable-type affinity。这样模型仍可直接处理异步事件集合，同时恢复一部分网格结构所提供的时间邻近性和变量关系归纳偏置。

### 审稿人视角：价值与不足

最有价值的技术思想是把“不规则 EHR 到底该用 grid 还是 set”这个设计矛盾，转化为 attention bias 层面的可控折中。它没有回到插补/重采样，也没有把所有事件当作无结构 token，而是在 set flexibility 上叠加时间局部性和变量类型亲和性。对审稿人而言，这种设计的优势在于轻量、可插拔、解释性较好：学到的 `tau` 可以反映任务需要的时间上下文范围，变量亲和矩阵也能提示哪些临床变量组合对预测更有用。

不足在于，它目前是 workshop 短文，实验规模和机制验证还不如主会完整论文充分。论文展示了三项 ICU 预测任务上的收益，也比较了不同 depth-wise bias fusion schedules，但尚未系统做跨医院、跨采样协议或反事实采样策略评估。更关键的是，temporal locality 与 variable-type affinity 可能同时学习到稳定生理关系和机构特定测量流程；例如某些变量在训练医院中总是被联测，attention bias 可能把这种流程共现误读为可迁移的临床耦合。

### 对 Sampling-Policy Shift 的启发

STAR-Set 对 Sampling-Policy Shift 的横向启发非常直接：它把 sampling-policy shortcut 写进了问题设定，说明 grid/mask 表示虽然方便，却可能让分类器利用“谁被测、何时被测、如何被对齐”这样的政策痕迹。对我们而言，时间局部 bias 和变量类型 bias 可以被扩展为 policy-diagnostic objects：如果同一类状态在不同医院或不同采样策略下学到的 `tau`、变量亲和矩阵或 attention mass 明显变化，就说明模型仍在吸收采样政策。

纵向深化上，可以把 STAR-Set 改造成 policy-aware set transformer：保留一套 state attention biases 学跨策略稳定的时间邻近性和变量关系，另设一套 policy attention biases 专门解释观测密度、联测模式和事件触发流程。训练时对同一潜在病程构造多种反事实采样视图，约束 state bias、state representation 和 logits 稳定，同时允许 policy bias 区分采样策略。这样既能避免粗暴丢弃 informative missingness，又能限制 sampling-policy shortcut 直接进入分类边界。
