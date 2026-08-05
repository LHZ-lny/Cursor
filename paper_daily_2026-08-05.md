# Paper Daily - 2026-08-05

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 与 2026-08-04 的已覆盖标题，纳入黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / sampling-rate shift 的顶会或顶会相关论文，重点核对 ICML 2026、ICLR 2026 TSALM Workshop、AAAI 2026、NeurIPS 2025 官方页面、OpenReview、arXiv 与论文/项目页。
- 已排除全部黑名单论文；同时排除偏 forecasting/generation、已被历史日报覆盖、或与分类任务关系较弱的候选。本次保留全新工作 2 篇：NeurOCNN 是 ICML 2026 正会论文，直接处理生理时序分类在未见采样率下的零样本稳健性；Cached Foundation Model Summaries 是 ICLR 2026 TSALM Workshop 论文，面向长程不规则 EHR 事件流的内存受限推理与临床预测，适合作为采样策略偏移下“历史上下文压缩”问题的补充视角。

## 1. NeurOCNN: A Neural-Operator-Based Model for Physiological Time Series

- 会议：ICML 2026 Poster
- 作者：Daya Kumar, Uday Devulapalli, Aarat Satsangi, Apurva Narayan
- 官方页：https://icml.cc/virtual/2026/poster/62493
- 代码：https://github.com/Idsl-group/NeurOCNN
- 关键词：physiological time series, neural operators, continuous-time convolution, function-to-label mapping, zero-shot sampling-rate shift, classification

### 场景、任务与核心难点

NeurOCNN 面向 EEG、眼动、心电等生理信号的时序分类。现实部署中，同一种生理信号可能由不同设备、不同采样频率或不同记录协议采集；一个在固定采样率数据上训练的分类器，换到新设备或新采样率后可能出现明显退化。论文把这一问题表述为 function-to-label mapping：模型应识别连续时间信号中的局部形态和瞬态事件，而不是依赖某个固定离散网格上的采样点排列。

核心难点在于，神经算子通常擅长学习离散化一致的算子，但生理信号高度非平稳，分类线索往往来自短暂、局部、形态敏感的事件；普通 CNN/Transformer 又容易绑定到训练采样率。NeurOCNN 因此结合 continuous-time spline-parameterized convolutions 与 Fourier projection pooling：前者在真实时间轴上捕捉局部形态，后者把可变分辨率信号映射成固定维表示，从而在多个未见 evaluation sampling rates 上保持较稳定的分类准确率。

### 审稿人视角：价值与不足

最有价值的技术思想是把“采样率变化”从数据增强或后验鲁棒性问题，提升为离散化不变的函数学习问题。相比把信号重采样到统一频率再分类，NeurOCNN 让模型直接在连续时间卷积和投影池化中学习跨分辨率稳定的形态表示。这对生理信号特别重要，因为异常波形、心拍形态、眼动瞬态等模式的语义由真实时间尺度决定，而不是由采样点索引决定。

不足在于，论文处理的主要是 sampling-rate shift / discretization shift，而不是 EHR/ICU 中更复杂的变量级异步观测和策略性 missingness。若采样率改变来自设备差异，NeurOCNN 的连续时间归纳偏置很合适；但若采样频率由病情、医生决策或告警流程触发，那么“更密集采样”本身可能带标签信息。当前方法强调跨采样率准确率稳定，却没有显式分离真实状态动力学与采样政策选择，也缺少对主动采样、事件触发采样和跨机构观测协议变化的因果诊断。

### 对 Sampling-Policy Shift 的启发

NeurOCNN 对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以被拆成两个层面，一是离散化分辨率变化，二是观测行为选择变化。NeurOCNN 主要解决前者，提示我们在建模不规则采样时应尽量把状态表示定义在真实连续时间上，而不是固定网格上；否则模型很容易把训练采样率当作类别边界的一部分。

纵向深化上，可以把 NeurOCNN 的连续时间卷积扩展为 policy-conditioned operator：state operator 学习跨采样率、跨设备稳定的生理形态；policy operator 学习采样频率、设备协议和观测触发规则。对同一连续轨迹生成不同采样策略视图时，可约束 state operator 输出和分类 logits 稳定，同时允许 policy operator 捕捉分辨率、噪声和触发机制差异。这样能把“零样本采样率稳健”推进到“采样政策改变下的状态-政策解耦”。

## 2. Cached Foundation Model Summaries for Memory-Efficient Clinical Time Series Inference

- 会议：ICLR 2026 TSALM Workshop Poster
- 作者：Rafi Al Attrach, Rajna Fani, David Restrepo, Yugang Jia, Leo Anthony Celi, Peter J. Schüffler
- OpenReview：https://openreview.net/forum?id=kdoFfrlOZj
- 论文 PDF：https://openreview.net/pdf?id=kdoFfrlOZj
- 项目记录：https://mcml.ai/publications/afr+26/
- 关键词：long irregular clinical time series, EHR event streams, cached foundation model summaries, memory-efficient inference, MIMIC-IV, AUROC

### 场景、任务与核心难点

这篇工作面向长程临床 EHR 时间序列推理。患者历史可以包含数千个不规则间隔的事件，每个事件通常由 medical code、距离前一事件的 time delta 和可选数值组成；在部署时，Transformer 类模型无法无限制读取完整历史，因为显存、延迟和上下文长度都会成为瓶颈。任务目标是在 MIMIC-IV 等 ICU/EHR 场景中，让轻量预测模型在只读取短近期窗口时，仍能利用更长历史中的关键信息。

论文提出的做法是把长历史压缩成 cached foundation model summary：先用预训练基础模型离线生成固定长度历史摘要，在线推理时只让轻量模型处理最近 N 个事件，并通过摘要进行条件化。作者系统做了 252 组实验，比较近期窗口长度、摘要位置和融合方式；核心发现是，当近期窗口很短时，cached summaries 可以带来显著 AUROC 增益，而当近期窗口扩到 256 个事件后收益接近消失；此外，用 FiLM 调制事件表示比把摘要作为额外 token 更有效，近期历史摘要通常也比很久远的历史摘要更有用。

### 审稿人视角：价值与不足

最有价值的思想不是提出更大的 EHR 模型，而是把长程不规则临床序列部署中的“上下文预算分配”问题系统化。对审稿人而言，这类实验很实用：它回答了什么时候值得缓存基础模型摘要、摘要应如何注入、最近事件窗口多长时历史压缩还有边际价值。很多 EHR foundation model 论文默认越长上下文越好，但这篇工作显示在内存受限推理中，短窗口、摘要质量和调制机制之间存在明确 trade-off。

不足在于，cached summary 可能把历史中的采样政策信息也一起压缩进去。EHR 长历史不仅包含病人状态，还包含医院如何观察病人、何时开检查、哪些变量被反复记录、哪些阶段记录稀疏。若摘要模型没有区分 patient-state memory 与 care-process memory，FiLM 调制可能把训练医院特定的记录习惯直接注入近期事件表示。论文主要评估内存效率和同数据源预测收益，还没有系统测试跨医院、跨采样政策或反事实记录策略下，缓存摘要是否保持稳定语义。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发在于：策略偏移不只发生在当前观测窗口，也可能被长期历史摘要放大。一个 summary 若压缩了“过去哪些变量经常被测、多久未测、何时密集记录”等政策信息，就可能在下游分类时提供强 shortcut；而且这种 shortcut 被缓存后更难从模型内部诊断。

纵向深化上，可以把 cached summary 拆成 state cache 与 policy cache。state cache 只保留跨机构稳定的病程、诊断和趋势信息，用于分类主路径；policy cache 则描述历史采样密度、联测模式、记录延迟和数据质量，用于不确定性校准或偏移告警。训练时可对同一患者历史施加不同截断和反事实采样策略，要求 state cache 调制后的 logits 稳定，同时允许 policy cache 变化。这样能把“内存高效临床推理”推进到“在有限上下文预算下隔离采样政策记忆”，避免长历史摘要成为隐蔽的 sampling-policy shortcut。
