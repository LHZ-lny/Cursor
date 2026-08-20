# Paper Daily - 2026-08-20

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregularly sampled / asynchronous / incomplete / clinical time-series classification 的顶会、顶会 workshop 与高相关同行评审论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、KDD 2026、NeurIPS 2025、ICLR TSALM、OpenReview、ACM DOI、arXiv 与项目页。
- 已排除全部黑名单论文；同时排除 LakeFM 这类 KDD 2026 但主任务偏生态 forecasting/representation 的候选、Under-Cali 这类 KDD 2026 online forecasting 工作、Sparse Physio-Attention 这类期刊论文、以及 HCIB/ISTS-PLM 这类时间窗口偏旧或更偏 incomplete/imputation 的 KDD 2025 工作。严格的近月正会 direct hits 多数已被历史日报覆盖，因此本次保留 2 篇全新且与异步/非规则采样分类最贴近的工作：MissTSM 是 TMLR 2026 同行评审论文，明确覆盖 IMTS classification；TreeText-CTS 是 2026-05 的 TSALM 相关预印本/候选，直接处理 irregular clinical time-series mortality 与 sepsis prediction，并强调可追溯证据。

## 1. Investigating a Model-Agnostic and Imputation-Free Approach for Irregularly-Sampled Multivariate Time-Series Modeling

- 方法名：MissTSM / Missing Feature-aware Time Series Modeling
- 会议/状态：TMLR 2026；arXiv 2025-02，OpenReview 录用记录
- 作者：Abhilash Neog, Arka Daw, Sepideh Fatemi Khorasgani, Medha Sawhney, Aanish Pradhan, Mary E. Lofton, Bennett J. McAfee, Adrienne Breef-Pilz, Heather L. Wander, Dexter W. Howard, Cayelan C. Carey, Paul Hanson, Anuj Karpatne
- OpenReview：https://openreview.net/forum?id=HgJ0DMVAA3
- arXiv：https://arxiv.org/abs/2502.15785
- 代码：https://github.com/abhilash-neog/SparseTimeSeriesModeling
- 关键词：irregularly-sampled multivariate time series, imputation-free, model-agnostic adapter, missing feature-aware attention, classification and forecasting

### 场景、任务与核心难点

MissTSM 面向不规则采样多变量时序中的通用建模问题，实验同时覆盖 forecasting 与 classification；分类侧包含 Epilepsy、EMG、Gesture，以及 PhysioNet-2012、P12、P19 等真实健康监测数据。它关注的不是单一临床任务，而是一个更基础的输入接口问题：当每个时间步只观测到部分变量、不同变量在不同时间缺失时，常规 Transformer 把“整行时间步”或“整列变量轨迹”当作 token 的做法都会失效，因为 token 本身不再完整。

论文解决的核心难点是如何在不插补、不重采样、也不替换主干模型的情况下，让已有 MTS backbone 能消费 IMTS。MissTSM 把每个 time-feature combination 独立嵌入为 token，只对真实观测值计算 embedding；缺失 token 通过 mask 排除。随后 Missing Feature-Aware Attention 用可学习 query 在同一时间步内对已观测变量做 masked cross-attention，得到该时间步的潜在摘要，再交给后续 backbone 学习长程时间依赖。这样它既避免了插补伪影，也避免为每个任务重新设计专用 ODE/RNN/Graph 架构。

### 审稿人视角：价值与不足

最有价值的技术思想是把 IMTS 建模的切入点从“设计一个新的不规则时序主干”下移到“修复标准主干的输入 tokenization”。这个判断很务实：许多强 MTS 模型已经具备长程建模、mask modeling 或自监督能力，真正阻塞它们处理 IMTS 的，是 embedding 层默认完整时间步或完整变量轨迹。Time-Feature Independent embedding 加 MFAA 让 MissTSM 成为 plug-and-play 适配层，也使实验能更公平地比较 imputation-based、imputation-free specialized 与 model-agnostic 三类路线。

不足在于，MissTSM 主要把缺失视为“同一时间步内哪些变量不可用”的结构问题，尚未系统区分缺失来自随机传感器损坏、成本约束、医生触发测量，还是医院流程差异。MFAA 对同一时间步的已观测变量做聚合，可能在训练环境中学习到策略性联测模式：例如某些变量一旦共同出现就高度暗示标签，但这种共现也许只是某家机构的检查协议。论文的 synthetic masking 和真实 IMTS benchmark 能说明鲁棒性，但还不足以证明跨 sampling policy 时 attention 权重与分类证据稳定。

### 对 Sampling-Policy Shift 的启发

MissTSM 对我们的横向启发是：sampling-policy shift 可以在输入适配层就被显式隔离，而不必等到深层 representation 才做对抗或不变性约束。它的 time-feature tokenization 很适合扩展成 state tokens 与 policy tokens：前者嵌入真实观测值及其连续状态证据，后者只编码观测是否出现、变量联测、时间步内可见性和采样密度。分类主路径主要消费 state summary，policy summary 则用于偏移诊断、校准或环境识别。

纵向深化上，可以把 MFAA 改造成 policy-aware masked attention：同一潜在轨迹在多种反事实采样策略下，约束 state query 输出和 logits 保持一致，同时允许 policy query 对不同缺失/联测机制敏感。还可以把 MissTSM 的“无需插补”优势用于构造更干净的 policy-shift benchmark：不再先把不同采样策略都补齐到同一网格，而是在原始观测 token 层直接测量模型是否依赖策略性可见性。这样能避免插补器先把 policy shortcut 写进数据，再让分类器被动学习。

## 2. TreeText-CTS: Compact, Source-Traceable Tree-Path Evidence for Irregular Clinical Time-Series Prediction

- 会议/状态：arXiv 2026-05；ICLR 2026 TSALM 相关预印本/候选，未确认正会录用
- 作者：Kwanhyung Lee, Juhwan Choi, Jongheon Kim, Joohyung Lee, Hyeongwon Jang, Eunho Yang
- arXiv：https://arxiv.org/abs/2605.20292
- 关键词：irregular clinical time series, ICU prediction, source-traceable evidence, tree-path verbalization, language-model encoder, mortality prediction, sepsis onset forecasting

### 场景、任务与核心难点

TreeText-CTS 面向 ICU/EHR 中的不规则临床时序预测，任务包括 PhysioNet 2012 院内死亡预测、MIMIC-III 院内死亡预测，以及 PhysioNet 2019 提前 6 小时脓毒症 onset forecasting。输入是变量、时间戳和值组成的 irregular EHR trajectory，同时包含 missingness、静态协变量和多尺度临床窗口。该类任务的难点不只是分类准确率，还包括可解释性：数值模型能利用 timestamps、masks 和 missingness patterns，但很难把每次风险预测背后的测量条件和时间窗口转成可审计证据。

论文的核心做法是把不规则 EHR 轨迹先汇总成多尺度窗口统计，再通过冻结的 window-specific XGBoost 模型产生 activated tree paths；Tree-to-Evidence Mapper 把这些路径转成确定性的、可追溯到源窗口和变量的文本谓词，例如“16-24 小时内 MAP 最小值不超过 65，最后一次 HR 高于 110”。Compact Evidence Selector 再从大量候选证据中选择有限条，交给 BioClinical ModernBERT 一类语言模型编码器做最终二分类。这样它避免了把整条 EHR 原始序列粗暴序列化给 LLM，也避免了推理时生成自由文本摘要。

### 审稿人视角：价值与不足

最有价值的思想是把“语言模型用于临床时序”从 free-form summarization 转向 source-traceable evidence composition。对审稿人而言，这一点很关键：许多 LLM-EHR 方法的可读性来自生成摘要，但摘要是否忠实、是否遗漏关键异常值很难保证；TreeText-CTS 的每个输入 span 都来自固定树路径和确定性阈值谓词，具备可追溯性。实验中它在三项 ICU 预测 benchmark 上优于既有 text-based EHR time-series interfaces，且不需要 patient-level autoregressive decoding，说明可解释接口不一定要牺牲性能和效率。

不足在于，TreeText-CTS 的证据库存来自训练集上拟合的 XGBoost 阈值和窗口统计，因此这些阈值仍可能吸收训练环境的采样政策。比如“某窗口内 lactate count 较高”或“time since last observation 较短”可能反映病情恶化，也可能反映 ICU 的检查流程、告警触发或资源配置。论文强调 source traceability，但 source-traceable 不等于 causal-stable；如果跨医院采样协议变化，树路径证据仍可能从临床状态证据滑向 policy evidence。此外，它依赖固定窗口统计，可能弱化原始异步事件的细粒度顺序和连续时间结构。

### 对 Sampling-Policy Shift 的启发

TreeText-CTS 对我们的问题有很强的横向启发：采样策略偏移不仅应在 representation distance 或 AUROC drop 中被测量，也应在“模型引用了哪些证据”中被审计。它的 tree-path evidence 可以天然拆分为 state predicates 与 policy predicates：前者描述稳定临床状态，如低血压、乳酸升高、肾功能异常；后者描述观测过程，如某变量 count、missingness indicator、time since last observation 或窗口内检查密度。跨环境部署时，如果 policy predicates 被频繁选入且对 logits 贡献很大，就能直接暴露 sampling-policy shortcut。

纵向深化上，可以把 TreeText-CTS 改造成 policy-auditable classifier：证据选择器在训练时被约束优先选择跨策略稳定的 state predicates，同时把 policy predicates 保留在单独报告通道，用于解释数据质量和采样流程，而不是直接驱动分类边界。对同一患者轨迹构造不同反事实采样视图时，可以要求 selected state evidence 与最终 logits 稳定，同时允许 selected policy evidence 改变。这样能把“可追溯证据”进一步推进为“可区分状态证据和采样政策证据”的诊断工具，正好服务于非规则采样下的 Sampling-Policy Shift 研究。
