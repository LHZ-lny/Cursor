# Paper Daily - 2026-07-17

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / partially observed / incomplete multivariate time series classification 的顶会或顶会入口论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、KDD 2025/2026、OpenReview、AAAI Proceedings、ACM DOI 与 arXiv/期刊页面。
- 已排除全部黑名单论文；同时排除偏 forecasting、anomaly detection、imputation/tutorial、普通规则时序分类、ICML/ICLR workshop 条目，以及时间窗口明显偏旧的 KDD 2025 HCIB/ISTS-PLM 候选。本次保留全新工作 1 篇：MUDRA 是 AAAI 2026 Journal Track / Abstract Reprint，原文发表于 Machine Learning 期刊；它不是普通 Technical Track 新论文，但在 2026 顶会入口中直接面向 partially-observed time series classification，并且提供了与深度 IMTS 模型不同的统计函数判别视角。

## 1. Multivariate Functional Linear Discriminant Analysis for Partially-Observed Time Series

- 简称：MUDRA
- 会议入口：AAAI 2026 Journal Track / Abstract Reprint
- 原文期刊：Machine Learning, 2025
- 作者：Rahul Bordoloi, Clemence Reda, Orell Trautmann, Saptarshi Bej, Olaf Wolkenhauer
- AAAI 页面：https://ojs.aaai.org/index.php/AAAI/article/view/41371
- DOI：https://doi.org/10.1609/aaai.v40i47.41371
- 期刊论文：https://doi.org/10.1007/s10994-025-06741-0
- 关键词：partially-observed time series, heterogeneous sampling times, missing features, multivariate functional data analysis, linear discriminant analysis, classification

### 场景、任务与核心难点

MUDRA 面向短程、多变量且部分观测的时序分类，典型场景包括生物医学、心理行为记录、语音/动作轨迹等样本长度不长但变量之间相关性强的数据。输入并不是完整规则网格，而是每条样本在不同时间点只观测到部分变量：同一变量可能缺失，不同变量的采样时间也可能不一致。论文在 Articulatory Words 等任务上评估分类和降维能力，重点关注高缺失比例下是否仍能得到稳定的低维判别表示。

它解决的核心难点有两层。第一，传统 FLDA 主要处理单变量函数的碎片化观测，难以直接建模多变量函数之间的依赖；第二，如果把部分观测时序先 padding、插值或补齐再分类，会把异步采样和缺失结构硬压成规则矩阵，在短序列和高缺失场景下很容易引入伪信号。MUDRA 将每条多变量时序视为高维函数曲线的碎片化观测，用 B-spline 基底表示时间函数，并通过 reduced-rank regression 与 PARAFAC 张量分解扩展 FLDA；参数推断上使用 ECM 算法，避免高维张量逆运算，从而在不显式插补完整网格的情况下完成判别式降维和分类。

### 审稿人视角：价值与不足

最有价值的思想是把“非规则/部分观测时序分类”重新拉回到统计函数判别分析框架中。近年的 IMTS 工作大多用 Transformer、CDE、Neural Flow 或图网络来吸收时间戳、mask 和变量关系；MUDRA 则提供了一个更可解释的替代基线：先假设样本来自连续多变量函数，再在函数空间里学习类别可分方向。它的贡献不在于更复杂的神经结构，而在于说明对于短时序、高缺失、需要低维解释的场景，显式函数基底、低秩结构和判别分析可能比黑盒插补-分类流水线更稳健。

不足也很明显。首先，MUDRA 依赖较强的函数平滑和类别分布假设；当真实过程存在突发事件、非平稳跳变或强非线性交互时，线性判别方向和 B-spline 平滑可能不足。其次，它主要在短序列和有限 benchmark 上展示优势，尚未覆盖 ICU/EHR 中那类长窗口、高维、事件触发式异步观测。第三，AAAI 入口是 Journal Track / Abstract Reprint，而非普通 Technical Track 新方法；因此它更适合作为“被顶会重新展示的统计视角”纳入前沿追踪，而不是与 ICML/ICLR 新深度模型直接等量比较。

### 对 Sampling-Policy Shift 的启发

MUDRA 对 Sampling-Policy Shift 的横向启发在于：策略偏移不一定只能通过更复杂的神经 encoder 解决，也可以先把不规则观测投影到可解释的函数空间，再检查类别判别方向是否随采样策略改变而漂移。若同一潜在轨迹在策略 A 和策略 B 下对应的 spline 系数、低秩因子或 FLDA 判别坐标差异很大，就说明模型仍把观测政策当作状态差异。

纵向深化上，可以把 MUDRA 改造成 policy-aware functional discriminant analysis：函数表示分成 state basis 与 policy basis 两组。state basis 解释跨采样策略稳定的连续状态轨迹，并进入分类判别方向；policy basis 解释变量何时被观测、哪些特征缺失以及采样时间异质性，只用于偏移诊断或不确定性校准。训练时可对同一底层函数生成不同采样策略视图，约束 state-side spline coefficients 和分类 logits 保持一致，同时允许 policy-side coefficients 变化。这个方向与深度 CDE/Transformer 的不变性目标互补：它提供了低维、可检验、可解释的采样策略泄漏诊断工具。
