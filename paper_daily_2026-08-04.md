# Paper Daily - 2026-08-04

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`，并结合自动化记忆中 2026-08-03 的新增标题，纳入所有历史总结标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / sparse clinical time series classification / informative missingness / variable-length medical time series classification 的顶会或顶会 workshop 论文，重点核对 ICLR 2026、ICML 2026、ACL 2026、NeurIPS 2025、KDD 2026、WWW 2026、OpenReview、ACL Anthology、ICLR/ICML/NeurIPS 官方页面、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 FORMED、CoTAR/TeCh、CauKer、AnchorMoE、VisualTimeAnomaly 等更偏规则医疗时序、通用时序分类或异常检测的强相关但非原生 irregular sampled classification 候选。本次保留全新工作 2 篇：OPL-MT-MNAR 是 ACL 2026 Findings 论文，直接把多模态 ICU 记录中的 informative missingness 作为动态状态与离线策略学习信号；RAxSS 是 NeurIPS 2025 TS4H Workshop Poster，面向多中心 iEEG 变长医疗时序分类，显式围绕 sparse sampling、窗口选择和可解释聚合展开。

## 1. Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness

- 方法名：OPL-MT-MNAR
- 会议：Findings of ACL 2026
- 作者：Zihan Liang, Ziwen Pan, Ruoxuan Xiong
- ACL Anthology：https://aclanthology.org/2026.findings-acl.1313/
- DOI：https://doi.org/10.18653/v1/2026.findings-acl.1313
- arXiv：https://arxiv.org/html/2604.21235v1
- 关键词：multimodal clinical time series, informative missingness, MNAR observation process, Bayesian filtering, ICU sepsis, mortality prediction, offline treatment policy learning

### 场景、任务与核心难点

这篇工作面向 ICU sepsis 场景中的多模态临床时间序列：结构化生命体征/化验值与临床文本记录都随时间产生，但两类模态的观测过程并不相同。结构化测量往往由医嘱、告警、病情严重度和医院流程触发；临床 notes 则受查房、护理记录和文书习惯影响。论文在 MIMIC-III、MIMIC-IV 和 eICU 上评估，任务包括 post-72-hour mortality prediction 等不良结局预测，以及基于学习到的患者状态做 offline treatment policy learning。

核心难点是 missingness not at random：数据是否被记录本身依赖患者潜在状态，且不同模态有不同记录机制。传统方法要么把缺失当作噪声处理，要么只把 mask/delta-t 拼进 encoder；OPL-MT-MNAR 则显式利用 observation process。方法包含三部分：多模态 encoder 同时吸收结构化数值、文本和观测模式；Bayesian filtering module 随时间更新潜在患者状态；下游模块基于该状态做 outcome prediction 和 off-policy learning。这样，模型不是简单补齐 ICU 轨迹，而是把“何时被观察、哪种模态被记录”作为动态病情推断的一部分。

### 审稿人视角：价值与不足

最有价值的思想是把 informative missingness 从辅助特征提升为动态表征学习和策略学习的核心对象。对审稿人而言，这篇论文的亮点在于它没有停留在“missingness 有预测力”的经验观察，而是把观测过程、潜在患者状态、临床文本记录和治疗策略联系到一个 sequential decision framework 中。尤其是 Bayesian filtering 的引入，使模型能够按时间更新对 latent state 的信念，比静态地聚合 mask 统计更贴近 ICU 决策过程。

不足也很直接：论文承认并利用观测过程的信息，但对“哪些观测信息是病情驱动、哪些是医院政策驱动”的分离还不充分。跨 MIMIC/eICU 的验证增强了可信度，但 ICU 数据中的用药、化验和文书记录都可能携带站点特定流程；如果模型把某家医院的记录习惯当作治疗价值或死亡风险信号，offline policy learning 会进一步放大这种偏差。另一个限制是，论文的下游目标包含策略学习与 outcome prediction，并非专门为 irregular sampled time-series classification 设计，因此分类任务上的结论需要迁移到更明确的跨采样策略评估中再验证。

### 对 Sampling-Policy Shift 的启发

OPL-MT-MNAR 对我们的 Sampling-Policy Shift 问题有直接横向价值：它把 observation process 明确视作一类动态政策，而不是缺失噪声。我们可以借鉴其 MNAR 特征和 Bayesian filtering，把非规则采样下的输入分解为 state evidence 与 policy evidence：前者反映真实病程或系统状态，后者反映何时记录、记录什么、由谁触发记录。若 policy-only 表征在不同医院或不同采样协议下仍能强预测标签，就说明存在严重 sampling-policy shortcut。

纵向深化上，可以在该框架上加入反事实采样策略训练：对同一潜在病程生成不同观测流程，要求 filtered state posterior、下游分类 logits 和治疗价值估计保持稳定，同时允许 policy posterior 区分不同记录制度。还可以把结构化测量与 notes 的 observation processes 拆成多个 policy heads，分别诊断化验触发、生命体征监测、文本记录和治疗动作中的策略偏移。这样能把“利用 informative missingness”推进到“使用但隔离 sampling-policy information”，更贴近我们要解决的跨策略稳健分类。

## 2. RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification

- 全称：Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification
- 会议：NeurIPS 2025 TS4H Workshop Poster
- 作者：Aydin Javadov, Samir Garibov, Tobias Hoesli, Qiyang Sun, Joseph Ollier, Florian von Wangenheim, Björn Schuller
- OpenReview：https://openreview.net/forum?id=RKVLsB0Ciu
- arXiv：https://arxiv.org/abs/2510.02936
- NeurIPS workshop 页面：https://neurips.cc/virtual/2025/132323
- 关键词：variable-length medical time series classification, sparse sampling, retrieval augmentation, explainable AI, iEEG, seizure onset zone localization, multi-center evaluation

### 场景、任务与核心难点

RAxSS 面向变长医疗时序分类，具体任务是基于多中心 iEEG 记录做 seizure onset zone localization。该类数据的困难不在于每个变量都有任意时间戳的 EHR 异步观测，而在于记录长度高度可变、可判别片段稀疏、信号噪声强，且不同医疗中心的采集流程和患者群体不同。若把整条长序列直接压成固定长度输入，模型容易被大量无关背景淹没；若只随机抽窗，分类证据又可能不稳定且难解释。

论文在 stochastic sparse sampling (SSS) 框架上加入 retrieval-informed aggregation。模型先按长度比例采样窗口并用 backbone 得到局部预测，再在同一记录内部计算窗口间相似度，基于 top-m 相似邻居形成 support score，最后在 probability space 中对窗口预测做凸聚合。这样每条序列的最终分类分数可以分解为若干窗口贡献，并为医生提供“哪些窗口支持了判断、它们与哪些窗口相似”的 evidence trail。

### 审稿人视角：价值与不足

最有价值的技术思想是把稀疏窗口采样和检索式证据聚合结合起来。相比 SSS 中对窗口预测做近似均匀或简单平均，RAxSS 用 within-recording retrieval 让相互支持的片段获得更高权重；相比黑箱 attention，它的 probability-space convex aggregation 天然给出可加性解释，便于追踪序列级预测由哪些局部证据构成。这对于医疗分类尤其重要，因为临床用户通常不只需要一个 SOZ 预测结果，还需要知道模型依据了哪些时间片段。

不足在于，RAxSS 仍是一篇 workshop 论文，实验集中在 iEEG 变长分类，对 ICU/EHR 异步多变量场景的覆盖有限。它处理的是 sparse window sampling 与 variable length，而不是完整的变量级非规则采样、missing-not-at-random 或事件触发观测过程。检索聚合也可能放大采样政策偏差：如果某中心更常记录某些状态、某些手术候选患者拥有更长或更密集的监测窗口，那么相似窗口检索会把中心特定证据聚成更强的分类信号。论文展示了多中心数据潜力，但还没有把采样协议、中心、设备设置或记录长度机制作为显式 policy shift 来做反事实评估。

### 对 Sampling-Policy Shift 的启发

RAxSS 对 Sampling-Policy Shift 的横向启发是：采样策略不仅决定哪些原始点可见，也决定哪些局部窗口会进入证据池、哪些窗口会通过相似性检索彼此增强。换句话说，sampling policy shift 可以表现为 evidence retrieval graph 的偏移：同一潜在病程在不同采样策略下，可能得到不同的关键窗口、不同的邻居支持和不同的概率聚合权重。

纵向深化上，可以把 RAxSS 改造成 state-policy evidence decomposition。state evidence windows 是跨采样策略稳定、真正承载病理或系统状态的片段；policy evidence windows 则解释记录长度、窗口密度、中心流程或设备设置带来的可见性差异。训练时可对同一序列构造不同 sparse sampling views，要求 state evidence leaderboard、聚合后 logits 和关键局部表征保持一致，同时允许 policy evidence graph 改变。评估时报告 retrieval-neighbor stability、window-contribution shift 和 policy-only aggregation performance，可以直接衡量模型是否把采样流程当成了分类证据。
