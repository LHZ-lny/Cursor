# Paper Daily - 2026-07-15

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification 的顶会或顶级方向论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、ICASSP 2026、KDD 2025/2026、OpenReview、IEEE/ACM DOI、官方会议页面与 arXiv 页面。
- 已排除全部黑名单论文；同时排除偏 forecasting、普通规则时序分类、ICLR workshop/preprint 条目以及已在历史记录中明确排除的 STAR-Set 类候选。本次保留全新工作 1 篇：VP-GNN 是 ICASSP 2026 会议论文，直接面向 EHR 不规则临床时序分类，在变量级异步依赖与多尺度 patch 聚合之间建立统一图框架。

## 1. VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series

- 会议：ICASSP 2026，MLSP-P33.14
- 作者：Yurong He, Boya Zhang, Longfei Liu, Guosheng Cui, Dan Wu
- 会议页：https://cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=18058
- DOI：https://doi.org/10.1109/icassp55912.2026.11461796
- 代码：https://github.com/bursonz/VP-GNN
- 关键词：irregular clinical time series, EHR classification, selective variable-wise message passing, patch-wise graph aggregation, mortality prediction, early sepsis detection

### 场景、任务与核心难点

VP-GNN 面向电子健康记录中的不规则临床时序分类，实验覆盖 PhysioNet 2012 与 Sepsis 2019，任务包括住院死亡风险预测和早期脓毒症检测。此类 EHR 数据的观测不是规则同步网格：生命体征、实验室检查和临床指标在不同时间被异步记录，变量之间采样密度差异大，且缺失和稀疏本身常常携带病情与临床流程信息。

论文解决的核心难点是如何同时建模“变量之间的异步依赖”和“时间局部片段中的多尺度动态”。如果只做变量级图消息传递，模型容易忽略不同时间窗口内的局部演化；如果只把序列切成 patch，则可能弱化临床变量之间由非同步观测形成的关系。VP-GNN 因此设计两阶段统一图框架：先在 variable-wise stage 通过 selective message passing 捕捉异步变量依赖，再在 patch-wise stage 通过 hierarchical Patch-GNN aggregation 建模多尺度时间模式，使患者表征同时保留跨变量结构和局部时序演化。

### 审稿人视角：价值与不足

最有价值的技术思想是把 EHR 不规则性拆成两个互补图问题：变量图负责“哪些临床指标应相互交流”，patch 图负责“哪些时间片段和尺度对当前风险更关键”。相比直接将 irregular EHR 填补到规则表格，VP-GNN 更尊重观测稀疏和变量异步；相比只用全局 attention，它又通过图消息传递显式约束了变量级和片段级的信息流。对临床风险分类而言，这种 variable-wise + patch-wise 的结构分解比单一序列 encoder 更贴近真实诊疗数据。

不足在于，模型仍主要在同分布公开 EHR benchmark 上验证，尚未显式把医院、病区、设备或检查协议作为采样政策环境来做跨策略评估。Selective message passing 可能把“哪些变量被频繁观测或联测”学习成高权重边；Patch-GNN 也可能偏向某些由临床流程触发的时间片段，而这些片段在另一家医院未必同样可见或同样语义稳定。论文报告了稀疏和不规则条件下的性能提升，但还缺少对 learned variable graph、patch importance 与采样政策变化之间关系的系统诊断。

### 对 Sampling-Policy Shift 的启发

VP-GNN 对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以同时改变变量图和时间 patch 图。不同医院若采用不同的化验联测协议、报警触发规则或记录频率，模型看到的变量依赖边、patch 密度和多尺度时间模式都会变化。因此，评估策略偏移时不应只统计 mask ratio 或 delta-t 分布，还应跟踪 variable-message edge weights、patch selection frequency、hierarchical aggregation path 在不同策略环境下是否稳定。

纵向深化上，可以把 VP-GNN 改造成 policy-aware dual-graph learner：变量图分解为 state-driven clinical edges 与 policy-induced observation edges，patch 图分解为病程关键片段与采样流程片段。训练时对同一潜在病程施加不同采样策略增强，约束 state graph、state patch representation 和分类 logits 保持一致，同时允许 policy graph 解释观测频率、联测习惯和检查触发规则。这样既保留 EHR 中 informative sampling 的有用性，又避免分类头把特定医院的采样政策当作可迁移的疾病证据。
