# Paper Daily - 2026-08-06

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`，并合并自动化记忆中 2026-08-03 至 2026-08-05 已总结标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / continuous-time irregular sequence modeling / informative missingness explanation 的顶会论文，重点核对 ICLR 2026 OpenReview、ICML 2026、AAAI 2026、KDD 2026 MILETS、NeurIPS 2025 官方页、OpenReview、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 Hi-Patch 这类 ICML 2025 时间窗口偏旧条目、WaveGNN 这类非指定顶会条目、CISM 这类 KDD workshop 且尚非正会论文、Learning Clinical Representations Under Systematic Distribution Shift 这类暂无顶会录用信息的 arXiv 条目，以及偏生成/插补而非分类或模型审计主线的候选。本次保留全新工作 2 篇：DeNOTS 是 ICLR 2026 正会 Poster，面向 Neural CDE/ODE 类连续时间模型并覆盖不规则时序分类；Contimask 是 NeurIPS 2025 Poster，虽然不是新分类器，但直接审计不规则时序分类/预测模型是否利用观测时机与 informative missingness，对 Sampling-Policy Shift 有强方法论价值。

## 1. DeNOTS: Stable Deep Neural ODEs for Time Series

- 会议：ICLR 2026 Poster
- 作者：Ilya Kuleshov, Evgenia Romanenkova, Vladislav Andreevich Zhuzhel, Galina Boeva, Evgeni Vorsin, Alexey Zaytsev
- OpenReview：https://openreview.net/forum?id=SFoDJZ1sSk
- 代码：https://github.com/Ilykuleshov/denots_iclr2025
- 关键词：irregular time series, Neural CDE, Neural ODE, time scaling, negative feedback, stability, classification

### 场景、任务与核心难点

DeNOTS 面向不规则时间戳序列上的连续时间建模，实验覆盖分类、回归和预测任务，其中包括 UWGL、InsectSound 与 Sepsis 等分类/临床风险预测场景。其关注点不是再设计一个新的输入 token 化方式，而是 Neural CDE / Neural ODE 这类连续时间模型的表达力与稳定性瓶颈：在不规则观测中，离散样本通常先被构造成连续 control path，隐藏状态再沿微分方程演化；如果仅通过收紧 solver tolerance 增加 Number of Function Evaluations，模型得到的往往是更高数值精度，而不一定是更强函数表达能力。

论文提出的核心做法是把积分时间跨度本身作为“深度”来放大。扩大 integration horizon 可以让 Neural CDE 获得更多有效演化步数，类似加深离散神经网络；但普通 vector field 在长时间积分下容易爆炸或漂移。因此作者引入 Negative Feedback / Anti-NF 机制，在保持表达灵活性的同时提供 input-to-state stability，并进一步用 Gaussian Process 理论量化插值与积分误差。整体目标是在不规则时序分类和预测中，让连续时间模型既能变深，又不会因为长积分区间而失稳。

### 审稿人视角：价值与不足

最有价值的技术思想是把 Neural CDE 的“深度”从 solver 精度中解耦出来。很多连续时间模型默认把 NFE 当作数值求解副产品，靠 tolerance 间接调控；DeNOTS 明确指出，增加数值精度不等于增加表示能力，真正类似网络加深的操作应是拉长隐藏动力学的有效演化时间。Negative Feedback 机制则补上了这一操作最关键的稳定性缺口，使 time scaling 不只是经验 trick，而有稳定性和误差界支撑。

不足在于，DeNOTS 主要解决的是连续时间模型的表达力、稳定性与数值鲁棒性，还没有显式建模 observation process。对 ICU 或传感器数据而言，control path 的粗糙度、插值误差和有效积分轨迹不仅由真实状态决定，也由采样政策决定：某些变量在告警后被密集观测，某些医院有固定化验节奏，都会改变 Neural CDE 看到的驱动路径。DeNOTS 可以让长时间动力学更稳，但如果没有 state/policy 分解，它仍可能稳定地利用采样政策 shortcut。

### 对 Sampling-Policy Shift 的启发

DeNOTS 对 Sampling-Policy Shift 的横向启发是：策略偏移会进入连续时间模型的数值层，而不只是进入输入 mask 或 delta-t 统计。不同采样政策会改变 control path 的插值误差、局部曲率、solver NFE 分布和长积分下的隐藏状态漂移；因此 NFE、time-scaling sensitivity、negative-feedback 激活强度和插值误差界都可以作为采样政策偏移诊断指标。

纵向深化上，可以把 DeNOTS 改造成 policy-aware continuous-time model：state vector field 学跨策略稳定的真实动力学，policy vector field 或 policy feedback term 解释观测密度、触发式采样和插值不确定性。对同一潜在轨迹生成不同采样策略视图时，约束 state trajectory、分类 logits 与 state-level feedback 保持一致，同时允许 policy branch 的 NFE、误差估计和反馈强度变化。这样可以把 DeNOTS 的稳定深连续时间建模推进到“采样政策改变时仍稳定且不泄漏策略证据”的分类框架。

## 2. Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time

- 会议：NeurIPS 2025 Poster
- 作者：Max Moebus, Bjoern Braun, Christian Holz
- OpenReview：https://openreview.net/forum?id=Jzr9VOiJYd
- 项目页：https://siplab.org/projects/Contimask
- 代码：https://github.com/eth-siplab/contimask
- 关键词：irregular time series, explainability, perturbation, informative missingness, continuous time, sepsis prediction

### 场景、任务与核心难点

Contimask 面向不规则时序模型的事后解释，尤其关注医院记录这类高缺失、非随机观测的预测/分类任务。论文中的代表场景是 sepsis prediction：真实 EHR 中高达 90% 的数据可能缺失，某项检查是否被医生下单、何时被测量，本身就可能反映临床怀疑和病情严重程度。传统时序 saliency 方法大多假设规则网格，只扰动观测值或连续片段；但在不规则时序中，模型可能依赖的不是数值本身，而是“这个时间点出现了观测”或“某段时间没有观测”。

论文解决的核心难点是如何解释模型对观测结构的依赖。Contimask 先把规则时序 perturbation 方法迁移到连续时间，再指出仅扰动数值仍会漏掉 informative missingness。它进一步使用 NeuroEvolution 搜索非可微扰动，例如模拟某些数据点从未被观测，从而生成能够同时覆盖 value saliency 与 timing/missingness saliency 的解释。最终目标不是提升分类准确率，而是揭示黑箱不规则时序模型是否利用了观测时机、缺失模式和结构性采样差异。

### 审稿人视角：价值与不足

最有价值的思想是把不规则时序解释从“哪些观测值重要”推进到“哪些观测行为重要”。这对医疗分类模型尤其关键：如果模型因为医生下单某项检查而预测某种疾病，它可能只是学到了临床流程 shortcut；如果模型利用的是更早、更稳定的病理信号，则更可能有部署价值。Contimask 提供了一种直接审计这种差异的工具，通过连续时间扰动和“未被观测”反事实，让 missingness 和 time intensity 也进入 saliency 分析。

不足在于，Contimask 是解释器而不是稳健学习算法。它可以发现模型依赖结构性缺失，但本身不保证分类器会避免这种依赖；NeuroEvolution 搜索也可能带来计算开销、随机性和扰动空间设计敏感性。更重要的是，扰动“移除观测点”是否对应真实可行的反事实采样政策，需要领域约束支持。若扰动生成的缺失模式不符合医院流程或传感器机制，得到的 saliency 可能解释的是模型对不真实输入的反应。

### 对 Sampling-Policy Shift 的启发

Contimask 对 Sampling-Policy Shift 的横向应用非常直接：它可以作为 policy leakage 的审计工具。我们可以在训练好的不规则时序分类器上分别扰动观测值、观测时间和观测存在性，比较 state saliency 与 policy saliency；如果模型对“是否测过某变量”比对实际数值更敏感，就说明分类边界可能被采样政策污染。

纵向深化上，可以把 Contimask 从事后解释扩展为训练期正则：先用连续时间 perturbation 生成多种反事实采样策略视图，再约束模型对 policy-only perturbation 的 logits 不敏感，同时保留对 state-changing perturbation 的敏感性。也可以定义 sampling-policy saliency gap：同一类别在不同医院/设备/采样协议下，状态相关 saliency 应稳定，而缺失与观测时机 saliency 可以变化但不应主导分类。这样，Contimask 不只是解释工具，还可以转化为 Sampling-Policy Shift 下的模型选择、诊断和鲁棒训练信号。
