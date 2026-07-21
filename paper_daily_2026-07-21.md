# Paper Daily - 2026-07-21

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / sparse medical time series classification / continuous-time classification / sampling-pattern explainability 的顶会论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、NeurIPS 2025 官方页、OpenReview、NeurIPS Proceedings、ICLR virtual site、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除 SITS 这类 ICML 2025 workshop 条目、LLapDiff/ReIMTS/ASTGI/HyperIMTS/VIMTS 等偏 forecasting/generation/imputation 的候选，以及 Mantis/Time-CoT/FORMED 这类主要面向规则或一般医疗时序分类、未显式处理非规则采样机制的候选。本次保留全新工作 2 篇：Random Controlled Differential Equations 是 ICLR 2026 正会 Poster，虽不以 irregular sampling 为标题主轴，但其 CDE/RDE 连续时间 reservoir 和 missing-observation robustness 与非规则采样分类直接相关；Contimask 是 NeurIPS 2025 正会 Poster，聚焦不规则时序模型解释，能直接揭示分类器是否利用 time intensity / informative missingness，是 sampling-policy shift 诊断的强相关工具。

## 1. Random Controlled Differential Equations

- 会议：ICLR 2026 Poster
- 作者：Francesco Piatti, Thomas Cass, William F. Turner
- OpenReview：https://openreview.net/forum?id=kHqt0ZSbKT
- 官方页：https://iclr.cc/virtual/2026/poster/10007804
- 论文：https://arxiv.org/html/2512.23670v1
- 代码：https://github.com/FrancescoPiatti/RandomSigJax
- 关键词：time-series classification, controlled differential equations, rough differential equations, random features, path signatures, reservoir computing, missing-observation robustness

### 场景、任务与核心难点

这篇工作面向连续时间序列学习和多变量时序分类。它不是专门为 ICU 异步化验表设计的 IMTS 架构，但使用 CDE/RDE 将离散观测路径映射到连续时间动力系统表示，因此天然适合讨论非均匀时间戳、插值路径、粗糙路径和缺失观测下的分类表示问题。论文在 UEA 多变量时序分类 benchmark 上评估，并额外检查随机缺失观测下的鲁棒性。

核心难点在于，Neural CDE、signature kernel 和 rough path 方法具有很强的连续时间归纳偏置，但通常训练成本或核矩阵成本较高；显式 signature 计算、kernel inversion 或端到端训练 vector field 都可能限制其在大规模分类任务上的可用性。作者提出 Random Controlled Differential Equations，将大规模随机参数化的 CDE/RDE 作为 continuous-time reservoir，只训练线性 readout。具体包含 RF-CDE 与 R-RDE 两个变体：前者先用 random Fourier features 提升输入，再驱动随机 CDE，在无限宽极限下逼近 RBF-lifted signature kernel；后者在 rough path 输入上通过 log-ODE discretization 捕捉高阶时间交互，在无限宽极限下对应 rough signature kernel。

### 审稿人视角：价值与不足

最有价值的技术思想是把 path-signature / CDE 的表达能力与 random feature reservoir 的训练效率连接起来。对审稿人而言，这类工作的重要性不只是平均准确率，而是给出一个清晰的理论桥梁：有限宽随机 reservoir 是可扩展模型，无限宽极限又能解释为结构化路径核。这样既保留连续时间路径模型对不规则观测的归纳偏置，又避免对每个任务训练复杂的神经微分方程主干。

不足在于，论文的主问题仍是高效时间序列分类和路径核近似，而不是显式的 irregular sampled classification benchmark。它能处理 piecewise-linear 或 rough path 表示，也展示 missing observations 下稳定退化，但并没有把采样时间本身的生成机制、变量级异步采样、医生/设备触发式观测或跨机构 sampling policy shift 作为主实验变量。换言之，Random CDE 提供了稳健的连续时间表示框架，但它默认输入路径构造已经足以表达观测过程；若训练环境中的采样密度、缺失模式或路径粗糙度与类别相关，线性 readout 仍可能利用这些策略性痕迹。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：可以把策略稳健性问题放到 path-kernel / random reservoir 的几何层面考察。采样政策改变会改变路径的签名特征、roughness、log-signature 局部项和 RF-CDE reservoir 激活分布；如果同一潜在状态在不同采样政策下投影到相距很远的随机路径特征，分类头再简单也会受到策略偏移污染。

纵向深化上，可以构建 policy-invariant random CDE：固定随机 reservoir 的高效优势不变，但在训练 readout 或 reservoir scaling 时加入反事实采样一致性。对同一连续轨迹生成多种观测策略视图，要求 RF-CDE/R-RDE 的 state subspace 和 logits 稳定，同时允许单独的 policy subspace 解释路径粗糙度、观测密度和缺失诱导的 signature residual。这样能把随机连续时间 reservoir 从“高效分类器”推进到“采样政策可诊断、状态表示可约束”的非规则时序分类前端。

## 2. Contimask: Explaining Irregular Time Series Models via Perturbations in Continuous Time

- 会议：NeurIPS 2025 Poster
- 作者：Max Moebus, Bjoern Braun, Christian Holz
- OpenReview：https://openreview.net/forum?id=Jzr9VOiJYd
- Proceedings：https://proceedings.neurips.cc/paper_files/paper/2025/file/4eb5daabc45b45a9a312aa2c8fca8a74-Paper-Conference.pdf
- 项目页：https://siplab.org/projects/Contimask
- 关键词：irregular time series, post-hoc explanation, continuous-time perturbation, informative missingness, time intensity, sepsis prediction

### 场景、任务与核心难点

Contimask 面向不规则时序模型的事后解释，典型场景是医疗预测/分类模型，例如真实 sepsis prediction 中 90% 数据缺失、观测时间不均匀、某些检查是否出现本身携带临床决策信息。它不是一个新的分类器，而是解释已经训练好的 irregular time series classifier：模型的预测可能同时依赖观测值、观测时间、缺失结构和 time intensity，因此传统规则时序 saliency 方法只遮蔽固定网格片段会漏掉关键机制。

核心难点在于，对不规则时序做扰动不能只改变 value。若一个模型把“某项指标被测了”或“某段时间观测很密集”当作风险证据，那么把数值置零、平滑或替换为基线并不能判断模型是否依赖 sampling pattern。Contimask 先将规则时序的 perturbation explainer 推到连续时间设定，再引入可以改变观测结构的 non-differentiable deletion perturbation：通过 NeuroEvolution 学习掩码，模拟某些观测“从未被采集”。这种扰动能揭示 value-independent 的结构性 saliency，包括 informative missingness 和 time intensity。

### 审稿人视角：价值与不足

最有价值的思想是把“解释不规则时序模型”从 value saliency 扩展到 observation-process saliency。对于医疗分类模型，很多高风险预测并不只来自异常数值，而来自医生为何下单、何时下单、哪些变量被连续监测。Contimask 的 deletion perturbation 正好能测试这类结构性依赖：如果删除某些观测事件本身比改变其数值更影响预测，就说明模型可能在使用采样行为作为证据。这对审稿人尤其重要，因为它把 informative missingness 从概念讨论变成了可操作的解释工具。

不足在于，Contimask 是 post-hoc explainer，并不直接给出如何训练 policy-robust classifier 的方法。NeuroEvolution 处理非可微删除扰动有灵活性，但也带来计算成本和解释稳定性问题；解释质量还依赖被解释模型、扰动预算和保真度目标的设定。如果底层模型已经强烈混合了状态信号与采样政策信号，Contimask 能指出模型依赖了观测结构，却未必能自动判断这种依赖是临床合理的 informative sampling，还是环境特定的 policy shortcut。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发非常直接：我们可以把 Contimask 用作策略泄漏诊断器。在训练环境和目标环境分别解释同一个分类器，比较 deletion mask、time-intensity saliency 和 value saliency 的差异；若模型在源环境中高度依赖某些观测事件的出现，而这些事件在目标环境中由不同医院流程触发，就能提前发现 sampling-policy shortcut。

纵向深化上，可以把 Contimask 从事后解释推进到训练时约束。具体做法是把 deletion perturbation 生成的 policy-sensitive observations 作为负证据：分类主路径对这些结构性扰动应保持稳定，policy 诊断分支则需要准确预测采样环境或观测流程。还可以设计 state-value saliency 与 policy-structure saliency 的分解目标，让模型既承认 informative missingness 的存在，又避免把训练医院的观测强度直接写入类别边界。这样，Contimask 不只是解释工具，也能成为 sampling-policy shift benchmark 中的稳定性评价和正则化信号来源。
