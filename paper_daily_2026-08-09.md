# Paper Daily - 2026-08-09

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-08-08 已覆盖但当前根目录未出现的标题，以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / clinical prediction / multimodal longitudinal EHR 的顶会论文，重点核对 NeurIPS 2025 Proceedings 与 OpenReview、ICLR 2026 OpenReview、ICML 2026 官方页、AAAI 2026 Proceedings、arXiv 与项目页。
- 已排除全部黑名单论文；同时排除 MedFuse 这类 withdrawn submission、TRIAGE 和 A Statistical Approach 这类暂无顶会录用信息的近期 arXiv 候选、MIRA 这类主任务偏 forecasting 的基础模型、OsciFormer 这类分类证据不足或仍为 submission 的候选。本次保留全新工作 2 篇：STaRFormer 是 NeurIPS 2025 正会论文，虽是通用 sequential framework，但明确在 P19/P12/PAM 不规则采样分类上评估；DiPro 是 NeurIPS 2025 Spotlight，面向稀疏不规则 CXR 与异步 EHR 的多模态临床预测，覆盖 disease progression identification、length-of-stay classification 和 in-hospital mortality prediction。

## 1. STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data

- 会议：NeurIPS 2025
- 作者：Maximilian Forstenhausler, Daniel Kulzer, Christos Anagnostopoulos, Shameem Puthiya Parambath, Natascha Weber
- 论文：https://proceedings.neurips.cc/paper_files/paper/2025/file/1eb7e71099cbbeba47afbbeb0804e820-Paper-Conference.pdf
- 项目页：https://star-former.github.io
- 关键词：irregularly sampled time series, non-stationary sequential data, dynamic attention-based regional masking, semi-supervised contrastive learning, P19/P12/PAM classification

### 场景、任务与核心难点

STaRFormer 的原始动机来自智能车钥匙/车旁设备的用户意图识别：轨迹数据由 UWB、BLE、GPS 或传感器测量产生，真实环境中的遮挡、信号干扰、设备姿态和测距策略会让序列同时呈现非平稳和不规则采样。论文把任务表述为序列级分类，并在大规模 DKT、Geolife 以及 P19、P12、PAM 等不规则采样分类数据上验证。

核心难点在于，很多 Transformer 或 SSL 时序模型默认输入是规则、充分观测、统计稳定的序列；但现实采样中，真正有判别力的片段可能只在局部时间段出现，且这些片段的均值、方差、频率和采样密度会随环境变化。STaRFormer 提出 Dynamic Attention-based Regional Masking (DAReM)：先利用注意力聚合找出任务相关区域，再对这些区域做动态 masking 和统计/采样扰动，形成 masked 与 unmasked 两个视图；随后用共享 Transformer tower 与半监督对比学习，把 batch-wise 自监督相似性、class-wise 监督相似性和下游分类目标耦合起来。它不是简单重构被 mask 的点，而是让表征在任务相关扰动下保持可分且稳定。

### 审稿人视角：价值与不足

最有价值的技术思想是把“不规则采样鲁棒性”放进任务耦合的表示学习过程，而不是只在输入层做 delta-t embedding 或在预训练阶段做通用 mask reconstruction。DAReM 的区域选择来自模型注意力，扰动集中在可能影响分类的片段；半监督对比损失又同时利用同一样本的视图一致性和类别结构，使模型学到的 latent space 更直接服务下游分类。论文在 56 个数据集上评估，并在 P19/P12/PAM 上报告对 irregular baselines 的稳定提升，这使其作为通用鲁棒表征框架具有较强说服力。

不足在于，DAReM 主要把不规则性当作可增强的输入扰动，还没有显式建模“谁决定何时采样、采样哪个变量”。注意力选出的任务区域可能是真实状态事件，也可能是训练环境中特定传感器策略或医疗流程造成的可见性模式；对这些区域施加强化式对比约束，有时会进一步巩固 sampling-policy shortcut。临床部分虽然覆盖 P19/P12/PAM，但并未按医院、设备协议、测量触发规则或采样密度环境做 cross-policy split；静态属性经文本编码后拼接，也可能引入额外的环境线索。

### 对 Sampling-Policy Shift 的启发

STaRFormer 对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以被转化为“任务相关区域在采样扰动下是否稳定”的诊断问题。DAReM 类模块不仅可以作为增强器，也可以作为 stress test：对同一潜在轨迹施加不同采样频率、局部遮蔽、延迟观测或变量联测策略，观察模型注意力选择、区域 mask 和 logits 是否保持一致。

纵向深化上，可以把 DAReM 扩展成 state-region 与 policy-region 双区域机制。state-region 承载跨采样策略稳定的病程/系统状态，参与分类主路径；policy-region 专门解释哪些区域因为设备、医院流程或主动测量策略而变得可见，只用于偏移诊断和不确定性校准。对比学习目标也可拆成 state consistency 与 policy distinguishability：同一真实轨迹在多种采样政策下，state embedding 和分类 logits 应保持一致，而 policy embedding 应能识别采样策略。这样能把 STaRFormer 的任务驱动鲁棒表示进一步推进到策略不变表示。

## 2. Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment

- 简称：DiPro
- 会议：NeurIPS 2025 Spotlight
- 作者：Chen Liu, Wenfang Yao, Kejing Yin, William K. Cheung, Jing Qin
- OpenReview：https://openreview.net/forum?id=2afhRWVb6p
- 论文：https://proceedings.neurips.cc/paper_files/paper/2025/file/bfe27c959fda8112520479539c69360f-Paper-Conference.pdf
- 代码：https://github.com/Chenliu-svg/DiPro
- 关键词：longitudinal multimodal fusion, sparse irregular imaging, asynchronous EHR, disease progression identification, mortality prediction, length-of-stay classification

### 场景、任务与核心难点

DiPro 面向 ICU 中纵向多模态疾病进展建模，输入同时包含连续或高频 EHR 生命体征/化验记录，以及稀疏、不规则出现的序列胸片 CXR。论文在 MIMIC 相关数据上评估 disease progression identification、length-of-stay classification 和 in-hospital mortality prediction。这个场景的异步性比纯数值 IMTS 更复杂：EHR 可以在小时级更新，而 CXR 往往由病情、医生判断或流程触发，时间点稀疏且不均匀。

核心难点有两层。第一，连续胸片中大量静态解剖结构会淹没真正随时间变化的病理进展；第二，CXR 的稀疏时间点与 EHR 的连续时间线很难直接对齐。DiPro 因此设计 Spatiotemporal Disentanglement，把区域级 CXR 特征拆成静态解剖和动态病理变化；再用 Progression-Aware Enhancement 通过反转 CXR pair 的顺序来强化进展方向语义；最后用 Multiscale Multimodal Fusion 在局部 interval-level 与全局 sequence-level 同步 CXR 动态特征和 EHR 表征，从而同时服务进展识别和 ICU 风险分类。

### 审稿人视角：价值与不足

最有价值的思想是把多模态临床时序中的“时间错位”拆成可建模的局部与全局对齐问题，并且先把影像中的静态背景和动态病理分开，再与 EHR 对齐。很多多模态临床模型直接拼接最新影像与 EHR，或者把所有纵向影像当作普通序列聚合；DiPro 明确指出，只有动态病理变化才应该承担 disease progression 的主要语义，而 EHR-CXR 融合也不能只靠单一时间尺度。Spotlight 级别的实验证明了 STD、PAE、MMF 三个模块在不同任务上的贡献，attention 可视化也能部分对应临床区域知识。

不足在于，DiPro 仍没有显式建模 CXR acquisition policy。现实中，哪些患者会被反复拍片、多久拍一次、是否刚好覆盖恶化窗口，往往由医生怀疑、床旁资源、治疗流程和医院制度决定；这些因素会影响样本是否有至少两张 CXR，也会改变局部 EHR-CXR interval 的语义。论文自己也指出排除了只有单张 CXR 的 visits，这会引入选择偏差。区域级解剖标注或自动框也带来可扩展性约束；而跨医院、跨影像调度策略下的 multiscale alignment 稳定性尚未系统评估。

### 对 Sampling-Policy Shift 的启发

DiPro 对 Sampling-Policy Shift 的横向启发是：采样政策偏移不只发生在数值变量的 mask/delta-t 上，也发生在“哪种模态何时被采集”。CXR 的稀疏 acquisition cadence 可以被看作一种高成本、事件触发式采样政策；EHR 的高频生命体征则是另一种低成本连续监测政策。二者之间的时间错位本身携带了护理流程和临床怀疑信息，若直接进入分类器，可能形成跨机构不可迁移的 shortcut。

纵向深化上，可以把 DiPro 改造成 policy-aware multimodal alignment：CXR dynamic state branch 学习跨采样策略稳定的影像病理进展，EHR state branch 学习稳定的生理状态；另设 policy branch 解释为何这个时间点有 CXR、为何某段 EHR 更密集、为何某些模态缺失。训练时可对同一潜在病程模拟不同 CXR 拍摄 cadence 和 EHR 观测频率，约束 state alignment、progression representation 和分类 logits 稳定，同时允许 policy alignment 表征变化。这样能把 DiPro 的多尺度对齐从“同步异步模态”推进到“区分真实疾病进展与观测流程变化”。
