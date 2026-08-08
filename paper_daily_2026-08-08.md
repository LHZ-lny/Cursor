# Paper Daily - 2026-08-08

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / risk forecasting 的顶会或顶级信号处理会议论文，重点核对 ICML 2026、ICASSP 2026、ICLR 2026、NeurIPS 2025、KDD 2025、OpenReview、会议官网、ACM/IEEE DOI 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除 `Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data` 这类时间窗口过旧的 ICLR 2024 工作、`TANDEM` 这类相关但会议时间偏旧的 CIKM 2025 工作、`TRIAGE` / `TreeText-CTS` / `SPAM` 这类当前主要是 2026 arXiv 或 manuscript 且未确认顶会录用的候选，以及 `Time-IMM` / `ChannelTokenFormer` 这类更偏 forecasting 或 benchmark、分类主任务较弱的候选。本次保留全新工作 2 篇：`ArcTimeSDE` 是 ICASSP 2026 中直接覆盖 irregular/asynchronous data 上 classification 与 forecasting 的 Neural SDE 方法；`MoRGen` 是 ICML 2026 Poster，面向不规则采样临床时序的多分辨率零样本风险预测，虽以 risk forecasting 命名，但用 AUROC/BCE 评估二元临床结局，对分类式临床决策和 sampling-policy shift 有直接参考价值。

## 1. ArcTimeSDE: Aligning Compute with Information Via ARC Length Time in Neural SDEs

- 会议：ICASSP 2026
- 作者：Zongyao Yin, Xianchuan Yu
- DOI：https://doi.org/10.1109/icassp55912.2026.11461413
- 会议记录：https://cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=5723
- 关键词：irregularly sampled time series, asynchronous data, Neural SDE, arc-length time, sampling sparsity, time-distribution shift, classification

### 场景、任务与核心难点

ArcTimeSDE 面向 ICU monitoring、finance、astronomy 等不规则且异步采样的时序任务，论文明确在 irregular and asynchronous data 上评估 classification 与 forecasting。这里的关键问题不是简单缺值，而是观测密度本身随时间和通道剧烈变化：某些区域被密集采样，某些区域长期空缺；若 Neural SDE 按物理时间推进，数值求解和表示学习会被采样间隔牵着走，在密集段过拟合局部噪声，在稀疏段又欠拟合真实变化。

论文的核心做法是把时间轴从 physical time 重参数化为 arc-length time，使计算密度更接近信息密度。模型先用 exponential-decay imputation 构造 control features，再通过 kernel interpolation 对齐到均匀的 arc-length 网格，最后用 control-driven Neural SDE 演化 latent states，并支持 diagonal 或 low-rank diffusion。这样，模型不是在每个真实时间间隔上平均分配计算，而是让状态变化较强、信息含量更高的位置获得更多建模资源，从而减轻 resampling bias，并在 sampling sparsity 与 time-distribution shift 下保持分类/预测稳健性。

### 审稿人视角：价值与不足

最有价值的技术思想是把“非规则采样导致的计算错配”显式化。许多连续时间模型强调可以处理任意时间戳，但默认求解仍沿 physical time 前进；ArcTimeSDE 指出，采样政策改变后，physical-time solver 的步长、控制路径粗糙度和局部信息密度都会一起变化，进而影响模型容量分配。arc-length time 提供了一个简洁的中间坐标，让计算预算对齐到轨迹变化而不是原始采样间隔，这对异步信号分类尤其有价值。

不足在于，ArcTimeSDE 仍需要由观测构造 control features，且其中包含 exponential-decay imputation 和 kernel interpolation。若观测密度本身是标签相关的采样政策结果，例如 ICU 中高危患者被更频繁测量，那么 arc-length 可能同时反映真实状态变化和策略性测量强度。论文展示其在 sparsity 与 time-distribution shift 下的稳健性，但还需要更明确的 state/process 分解实验，证明 arc-length 坐标捕捉的是病程或系统动力学的信息密度，而不是训练环境的测量积极性。

### 对 Sampling-Policy Shift 的启发

ArcTimeSDE 对 Sampling-Policy Shift 的横向启发是：采样政策偏移不仅改变输入 mask 和 delta-t，也改变模型内部的计算时间。若不同医院或设备策略让同一类状态产生不同观测密度，physical-time 与 observation-time 表示都会发生系统漂移；arc-length time 可作为一个可诊断变量，衡量“模型把哪里看成信息密集区域”。

纵向深化上，可以把 arc-length 拆成 state arc-length 与 policy arc-length：前者由观测值变化、跨变量稳定动力学和临床/物理状态变化定义，进入分类主路径；后者由观测密度、联测模式、告警触发和缺失结构定义，用于策略诊断与置信度校准。训练时对同一潜在轨迹施加不同反事实采样策略，要求 state arc-length 表示和 logits 保持一致，同时允许 policy arc-length 改变。这样能把 ArcTimeSDE 的“计算对齐信息”进一步推进到“计算对齐跨策略稳定的信息”。

## 2. MoRGen: Mixture-of-Resolutions Generative Forecasting for Irregularly Sampled Medical Time-Series Data

- 会议：ICML 2026 Poster
- 作者：Nassim Oufattole, Matthew McDermott, Collin Stultz
- 官方页：https://icml.cc/virtual/2026/poster/64999
- 关键词：irregularly sampled clinical time series, zero-shot risk forecasting, generative forecasting, temporal resolution, binary clinical outcomes, AUROC

### 场景、任务与核心难点

MoRGen 面向不规则采样临床时间序列上的 zero-shot risk forecasting：模型需要基于历史病历和时序观测估计未来 readmission、mortality、abnormal lab results 等临床事件风险。虽然论文标题使用 forecasting，但其评估以 binary cross-entropy 和 AUROC 为核心，实际对应多种二元临床结局的分类式风险决策。临床时序的核心难点在于，不同终点的时间尺度差异很大：急性异常可能在小时级发生，死亡或再入院风险可能跨天到月演化；单一固定时间分辨率的自回归生成模型很容易与目标动态不匹配。

论文提出 Mixture-of-Resolutions Generation：训练或使用多个不同 temporal resolution 的生成式 forecaster，再用低容量、任务特异的 mixture 融合这些专家的预测。其关键观察是，单个生成 forecaster 的 zero-shot accuracy 会随时间分辨率显著变化；当分辨率与终点动态不匹配时，性能会下降。MoRGen 因此不再假设一个细粒度 tokenization 适合所有临床任务，而是让模型按任务自动组合粗细不同的时间尺度，在三个独立临床数据集、多个 horizon 与 outcome 上取得更低 BCE 和显著 AUROC 提升。

### 审稿人视角：价值与不足

最有价值的思想是把不规则临床时序中的“时间分辨率选择”从预处理超参数提升为模型组合对象。许多 EHR 生成/预测模型会先选一个固定 token 粒度，再讨论模型架构；MoRGen 直接指出，分辨率本身决定了模型能否看见目标终点的合适动态。低容量 mixture 也很审慎：它不是用一个大模型重新学习所有结局，而是让不同时间尺度专家保留各自归纳偏置，再为每个任务学习组合权重。

不足在于，MoRGen 的多分辨率专家可能仍然混合 patient state 与 care process。临床记录的时间分辨率不只是疾病变化速度，也反映随访制度、检查频率、医嘱流程和出院后数据可见性。若某家医院对高风险患者采用更细粒度的记录或随访，mixture 权重可能学习到“某类任务应该依赖细粒度专家”，但这个规律未必能跨采样政策迁移。论文证明多分辨率融合提升了风险预测，但还需要跨医院、跨随访策略和反事实采样粒度扰动下的系统评估。

### 对 Sampling-Policy Shift 的启发

MoRGen 对 Sampling-Policy Shift 的横向启发是：采样政策偏移可以被看作“最佳时间分辨率”的偏移。不同政策改变了哪些临床变化可在细粒度中被观察到、哪些只能在粗粒度中被稳定估计；如果模型的风险预测强依赖某一分辨率专家，部署环境换采样策略后就可能出现系统性失配。

纵向深化上，可以借鉴 MoRGen 设计 policy-aware resolution mixture：把 mixture 权重分成 state-driven weights 与 policy-driven weights。state-driven weights 反映目标终点真实动态需要的时间尺度，应在反事实采样策略下保持稳定；policy-driven weights 解释观测制度、随访频率和记录粒度差异，只用于校准或偏移告警。进一步可把不同采样策略下的同一潜在病程输入多分辨率专家，约束 state mixture、风险 logits 和校准曲线稳定，同时允许 policy mixture 指示“当前环境在哪些时间尺度上信息不足”。这能把多分辨率生成式风险预测扩展为面向 Sampling-Policy Shift 的时间尺度不变学习框架。
