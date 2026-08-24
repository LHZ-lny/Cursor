# Paper Daily - 2026-08-24

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`；同时读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-08-21 的新增标题。
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
  - DeNOTS: Stable Deep Neural ODEs for Time Series
  - Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time
  - PathwayLLM: Explainable Clinical Trajectory Modeling with Structured Pathways for Sepsis Prediction
  - ArcTimeSDE: Aligning Compute with Information Via ARC Length Time in Neural SDEs
  - MoRGen: Mixture-of-Resolutions Generative Forecasting for Irregularly Sampled Medical Time-Series Data
  - STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data
  - Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment
  - Repurposing Foundation Model for Generalizable Medical Time Series Classification
  - Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series
  - Investigating a Model-Agnostic and Imputation-Free Approach for Irregularly-Sampled Multivariate Time-Series Modeling
  - TreeText-CTS: Compact, Source-Traceable Tree-Path Evidence for Irregular Clinical Time-Series Prediction
  - Random Controlled Differential Equations
  - An Automated Data Engineering Pipeline for Time Series Classification Via Text Embeddings
  - CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data
  - JETS: A Self-Supervised Joint Embedding Time Series Foundation Model for Behavioral Data in Healthcare
  - TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / EHR event-stream clinical prediction 的顶会或顶会相关论文，重点核对 AAAI 2026、ICML 2026、ICLR 2026、ICASSP 2026、KDD 2025/2026、OpenReview、AAAI Proceedings、ICML/ICLR virtual site、IEEE/ICASSP 页面、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 ViTST/SEFT/KEDGN 等时间窗口过旧条目、Hi-Patch/GRUwE/ASTGI/ReIMTS 等偏 forecasting 或 event prediction 而非分类主任务条目、RC-GRF/Transformer autoencoder 等暂无顶会来源的预印本，以及 TreeText-CTS/TRIAGE 等已由自动化记忆标记过的条目。本次保留全新工作 2 篇：MATA-Former & SIICU 是 AAAI 2026 相关 ICU 风险预测工作，直接处理 text-intensive irregular clinical time series；Cross-Representation Benchmarking 是 ICASSP 2026 论文，系统比较 EHR 多变量时序、事件流和文本事件流表征，对非规则采样下的表示选择与 sampling-policy shift 评估有直接方法论价值。

## 1. MATA-Former & SIICU: Semantic Aware Temporal Alignment for High-Fidelity ICU Risk Prediction

- 方法名：Medical-semantics Aware Time-ALiBi Transformer (MATA-Former)
- 会议/状态：AAAI 2026 相关记录；arXiv 2026-04 版本
- 作者：Zhichong Zheng, Xiaohang Nie, Xueqi Wang, Yuanjin Zhao, Haitao Zhang, Yichao Tang
- 论文：https://arxiv.org/abs/2604.01727
- 佐证记录：https://www.csauthors.net/yichao-tang/
- 关键词：irregular clinical time series, ICU risk prediction, semantic-aware temporal alignment, text-intensive EHR, soft labeling, event-wise risk modeling

### 场景、任务与核心难点

MATA-Former 面向 ICU 临床早期预警和多风险预测。输入不是单纯规则化生命体征矩阵，而是由结构化 vitals/labs 与非结构化护理记录、影像报告、操作记录等文本密集事件交织而成的 ICU 轨迹。论文强调 SIICU 数据具有高度异质和稀疏特征：每位患者事件数重尾分布，时间间隔跨越秒、分钟、小时到天，既有自动监护产生的高频结构化数据，也有人工干预和临床叙事产生的 sporadic events。

核心难点在于，ICU 风险并不总是随物理时间距离单调衰减。某些事件的有效期很短，例如急性生命体征变化；某些诊疗、用药或文本事件的病理影响可能持续很久，并且其重要性取决于当前要预测的风险类型。常规 time-aware attention 或 ALiBi 类方法通常把时间距离作为统一衰减项，容易低估远期但语义上仍有效的证据。MATA-Former 因此用医学事件语义来动态参数化时间注意力，使注意力窗口由 risk query 与 clinical event semantics 决定，而不是只由时间戳距离决定。论文还提出 Plateau-Gaussian Soft Labeling，将粗粒度二分类标签转为连续多时间窗风险轨迹，以更细地监督风险从潜伏、发展到恢复的过程。

### 审稿人视角：价值与不足

最有价值的思想是把“时间对齐”从几何距离问题提升为“医学语义有效期”问题。在很多 irregular EHR 模型中，时间戳只是 positional encoding 或 decay feature；MATA-Former 则明确指出，同样过去 12 小时的事件，在不同风险预测任务中可能有完全不同的相关性。用 event semantics 生成 query-dependent temporal bias，可以把临床知识和连续时间建模结合起来，尤其适合 text-intensive ICU 场景中远期病程记录、用药和护理事件的长效影响。SIICU 数据集的人机协同细粒度标注也有价值，因为它试图避免只用 discharge-level ICD 或粗二分类标签带来的 look-ahead bias 和监督稀疏。

不足在于，MATA-Former 将医学语义和时间偏置绑定得更紧，但还没有充分分解“病理语义”与“记录/采样政策语义”。ICU 文本事件、护理记录和操作记录本身高度受医院流程、科室习惯、告警阈值和人力资源影响；某条记录的出现可能说明病情，也可能说明某个班次更勤于记录、某类患者进入特定诊疗流程或某院区有更细的文书规范。MATA-Former 的语义注意力可能更好地捕获这些事件，但不保证捕获的是跨机构稳定病理机制。另一个风险是 SIICU 的 LLM 预标注 + 人工校验流程虽提高了细粒度标签密度，但其标签体系和事件语义是否能跨医院迁移仍需要外部验证。

### 对 Sampling-Policy Shift 的启发

MATA-Former 对 sampling-policy shift 的横向启发是：采样策略偏移不仅改变观测时间和缺失模式，还会改变“哪些语义事件被写入记录、何时写入、以什么粒度写入”。如果一个模型把文本事件语义直接作为时间注意力的调度器，那么采样/记录政策变化会通过 temporal bias 进入分类边界。我们可以借鉴其 semantic-aware temporal bias，但需要把 bias 分解为 state-validity bias 与 policy-documentation bias：前者表示某类病理证据的真实有效期，后者表示医院记录流程和采样习惯。

纵向深化上，可以设计 policy-aware MATA：对同一潜在病程构造不同记录密度、不同 note timing、不同 lab-ordering policy 的反事实视图，要求 state temporal bias、风险轨迹和 logits 保持一致，同时允许 policy bias 解释文书频率、检查触发和事件可见性差异。MATA-Former 的 PSL 也可用于评估策略偏移：如果换采样/记录政策后，连续风险曲线的形状大幅改变，但临床状态没有改变，就说明模型仍把 observation process 当成 disease process。

## 2. Cross-Representation Benchmarking in Time-Series Electronic Health Records for Clinical Outcome Prediction

- 会议：ICASSP 2026
- 作者：Tianyi Chen, Mingcheng Zhu, Zhiyao Luo, Tingting Zhu
- 官方记录：https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=16742
- 论文：https://arxiv.org/abs/2510.09159
- 代码：https://github.com/BrandonC8310/EHR-Cross-Representation-Benchmarks
- 关键词：EHR clinical outcome prediction, multivariate time series, event streams, textual event streams, missingness-based feature pruning, representation benchmark

### 场景、任务与核心难点

这篇工作面向 EHR 临床结局预测中的表示选择问题。它比较三类输入范式：将 EHR 聚合成规则多变量时序矩阵、保留时间戳的事件流，以及把事件流转成可供 LLM 使用的文本事件流。任务覆盖 MIMIC-IV ICU mortality、ICU phenotyping，以及 EHRSHOT 中的 30-day readmission 和 1-year pancreatic cancer 等临床预测/分类任务。

核心难点不是提出一个更复杂的单模型，而是回答一个更基础的问题：在非规则 EHR 中，究竟应把数据整理成哪种表示，才能在公平评估下得到可靠结论？传统 ICU benchmark 常把事件按小时或天聚合、前后向填补、再做 population-median imputation；这种做法便于复用 LSTM/Transformer/MLP，却会扭曲原始时间信息，并把 missingness 和 sampling density 压成难以解释的网格 artifact。事件流表示保留了不规则时间和原始事件顺序，文本事件流进一步给 LLM 提供可读语义，但三者过去常在不同数据切分、不同任务定义和不同预处理下比较，难以判断性能差异来自模型能力还是 representation/curation choices。

### 审稿人视角：价值与不足

最有价值的贡献是把 EHR 表示选择做成统一 benchmark，而不是默认某一种表示天然最优。论文在相同 cohorting、labeling、split 和 metric 下比较多变量时序模型、CLMBR/计数类事件流模型，以及 8-20B LLM 文本流模型，并进一步分析 missingness-based feature pruning。结果显示 event stream models 整体表现最强，CLMBR 在 few-shot 下样本效率高，而稀疏特征剪枝对 ICU 与纵向任务的影响不同：ICU 中剪掉高度缺失特征常能简化模型且损失较小，纵向任务中稀疏特征反而可能关键。

不足是，benchmark 仍然偏经验比较，并没有显式建模采样政策因果机制。多变量时序表示被规则化到固定网格，event stream 表示保留原始时间，但论文尚未把医院、科室、设备、记录流程或医生下单策略作为环境变量进行 cross-policy split。文本事件流中的 LLM 表现也可能受 prompt、上下文长度和事件 verbalization 影响；如果文本化规则把采样密度、缺失或记录习惯转成显著语言线索，LLM 可能学习到 policy shortcut。换言之，它能告诉我们不同表示在统一协议下谁更强，但还不能完全解释强弱差异中有多少来自 state information、有多少来自 observation policy。

### 对 Sampling-Policy Shift 的启发

这篇工作对 sampling-policy shift 的横向启发非常直接：研究策略偏移前，必须先把 representation choice 当作实验变量。相同 EHR 原始数据可以被转换成网格时序、事件流或文本事件流；不同转换会保留或抹平不同类型的采样政策信息。例如网格化和 imputation 可能把策略信息压缩为填充值和 mask，事件流显式保留事件可见性，文本流可能把缺失和检查行为转成语义描述。若不控制表示选择，所谓 policy robustness 可能只是某种表示恰好隐藏或暴露了 policy shortcut。

纵向深化上，可以在该 benchmark 框架上增加 policy-aware evaluation：为每个表示同时构造 state-only、policy-only 和 full-input 三组模型，并按医院/科室/采样密度/联测模式/缺失率分层报告跨环境性能。还可以比较同一反事实采样策略在三种表示中的传播路径：网格表示中表现为 mask 和 imputed value 变化，事件流中表现为 token sequence 变化，文本流中表现为叙述证据变化。这样能帮助我们判断 Sampling-Policy Shift 的解决方案到底应发生在数据表示层、encoder 层，还是分类/校准层。
