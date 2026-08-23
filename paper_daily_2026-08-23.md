# Paper Daily - 2026-08-23

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`；同时读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-08-21 的新增标题，纳入黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregularly sampled medical time series / ICU time-series classification / informative sampling / wearable irregular health time series 的顶会、顶会 workshop、OpenReview 与 arXiv 页面，重点核对 ICLR 2026、ICML 2026、AAAI 2026、NeurIPS 2025 TS4H、ICLR 2026 TSALM、OpenReview、arXiv 与项目页。
- 已排除全部黑名单论文。严格的“顶会正会 + 直接 IMTS 分类”新增命中基本已被历史日报覆盖；本次保留 2 篇未在黑名单中的新跟踪对象：JETS 是 NeurIPS 2025 TS4H workshop 工作，直接使用 irregular triplet wearable time series 做诊断预测；TRIAGE 是 2026-06 新预印本，尚未确认顶会录用，但明确面向 irregularly sampled medical time series 的临床风险分类、校准和解释，对 sampling-policy shift 的可解释诊断价值较高。

## 1. JETS: A Self-Supervised Joint Embedding Time Series Foundation Model for Behavioral Data in Healthcare

- 会议/状态：NeurIPS 2025 Workshop on Learning from Time-Series for Health (TS4H)
- 作者：Erik Xie, Raquel Rodriguez Martinez, Wyatt Chang, Brandon Ballinger
- OpenReview：https://openreview.net/forum?id=i4epRiMy8z
- 参考解读：https://www.empirical.health/blog/wearable-foundation-models
- 关键词：irregular wearable time series, behavioral health data, JEPA, triplet tokenization, self-supervised foundation model, diagnostic prediction

### 场景、任务与核心难点

JETS 面向可穿戴设备产生的真实世界健康行为时间序列。输入不是 ICU 化验表，而是 Apple Watch、Samsung Galaxy、Fitbit 等设备上的 63 通道行为/生理指标，例如血氧、静息心率、睡眠阶段、HRV 等；这些数据以 `(timestamp, value, metric type)` triplets 表示，而不是强行对齐到规则网格。下游任务包括 hypertension、sick sinus syndrome、ME/CFS、atrial flutter 等个体级诊断预测，以及 HbA1c、glucose、HDL、hs-CRP 等 biomarker 预测。

核心难点在于，消费级可穿戴数据天然稀疏、碎片化且强噪声：用户摘表、充电、设备型号、系统采样策略、传感器质量和日常行为都会改变哪些指标被记录、何时记录、持续多久。传统 MAE 式重构容易把大量传感器噪声和缺测 artifact 当作目标；规则化网格又会把真实的 wearing / charging / device policy 差异抹平成插补值。JETS 采用 JEPA 风格的 joint-embedding predictive architecture：一个 encoder 看完整序列，另一个 encoder 看随机 30% token，predictor 在 latent space 中预测完整视图表示；模型关注可迁移的行为-生理结构，而不是逐点重构原始噪声。

### 审稿人视角：价值与不足

最有价值的思想是把 wearable health foundation model 从“规则窗口上的传感器重构”推进到“非规则 triplet 序列上的 latent predictive learning”。这对审稿人有吸引力，因为它同时处理了三个现实部署问题：输入异构、观测稀疏、标签稀缺。JEPA 的 latent-space prediction 比 raw reconstruction 更适合可穿戴数据，因为很多缺口和短期尖峰并不是稳定健康状态，而是设备和用户行为造成的观测噪声；让模型预测高层表示有助于忽略不可迁移细节。

不足在于，JETS 目前是 workshop 级别工作，公开细节和系统消融相对有限；它的“不规则性”主要来自 wearable usage 与设备采样，而不是 ICU/EHR 中由医生决策触发的化验和治疗记录。更重要的是，wearable observation policy 本身高度带有健康和社会行为信息：病人越不舒服可能越少佩戴设备，某些设备或地区的采样配置不同，睡眠/运动指标的可见性也受用户合规性影响。若没有跨设备、跨用户群、跨佩戴策略的环境划分，模型学到的 embedding 可能仍混合了 health state 与 device/user policy。

### 对 Sampling-Policy Shift 的启发

JETS 对 Sampling-Policy Shift 的横向启发是：采样政策不一定来自医院，也可能来自用户-设备交互。可穿戴数据中的 missingness、metric availability、charging gaps、sensor duty cycle 和 firmware-level aggregation 都是 observation policy 的组成部分；这些政策变化会影响诊断分类器看到的行为轨迹，甚至比真实生理状态更容易被模型捕捉。

纵向深化上，可以把 JETS 改造成 state-policy 双嵌入的 wearable foundation model。state branch 用 JEPA 预测跨佩戴策略稳定的健康状态表示，policy branch 则预测设备型号、佩戴/充电模式、采样密度和通道可用性。训练时对同一连续行为轨迹模拟不同设备采样策略、不同缺测块和不同 metric subsets，约束 state embedding 与诊断 logits 稳定，同时允许 policy embedding 改变。这样能把 JETS 的 triplet-token JEPA 推进到“可穿戴采样政策变化下仍保持诊断语义稳定”的不规则健康时序分类器。

## 2. TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs

- 会议/状态：arXiv 2026-06 预印本，尚未确认顶会录用
- 作者：Hyeongwon Jang, Gyouk Chu, Changhun Kim, Joonhyung Park, Hangyul Yoon, Eunho Yang
- 论文：https://arxiv.org/abs/2606.09030
- 代码：https://github.com/HyeongWon-Jang/TRIAGE
- 关键词：irregularly sampled medical time series, clinical risk prediction, LLM reasoning, calibration, explainability, risk polarization

### 场景、任务与核心难点

TRIAGE 面向基于 EHR 的临床早期预警系统。输入是 irregularly sampled medical time series：生命体征、化验、病程记录等临床观测以不规则时间出现，模型需要输出可用于 triage 的连续风险分数，并给出临床人员可以核查的解释。与只追求 AUROC/AUPRC 的分类器不同，这类系统还要求校准良好、风险可比较、解释可追溯，因为过度自信的二分类判断会直接影响分诊优先级。

论文指出 LLM 用于临床时序风险预测时容易出现 risk polarization：模型把本应是连续概率的问题压成过度自信的阳性/阴性结论，导致校准差、跨患者风险不可比。TRIAGE 的核心做法是 dialectical reasoning：不让 LLM 只为单一结果找理由，而是为 competing clinical outcomes 分别生成 outcome-specific rationales，再由这些相互竞争的理由导出连续风险分数。训练包括 Dialectical Reasoning Supervision 和 Self-Refinement；论文在 3 个 ISMTS benchmark 上报告平均 AUPRC 提升 3.3%、calibration error 降低 81%，并用 LLM-as-a-judge 评估 rationale 质量。

### 审稿人视角：价值与不足

最有价值的思想是把 irregular clinical time-series classification 的 LLM 路线从“把序列文本化后直接问标签”推进到“风险估计必须同时比较正反临床证据”。这很重要，因为临床 triage 本质上是连续风险排序和证据权衡，而不是简单二分类。Dialectical rationale 也提供了一个可以审查的中间层：模型为什么认为 sepsis / mortality / deterioration 更可能，为什么相反结果仍有一定概率，这比单一链式解释更接近医生讨论风险的方式。

不足在于，TRIAGE 尚未确认顶会录用，当前应作为高相关新预印本跟踪，而不是已验证的正会工作。技术上，它仍依赖 LLM 对数值时序的序列化理解；若输入摘要或 tokenization 已经混入采样政策 shortcut，dialectical reasoning 可能只是把这些 shortcut 解释得更流畅。校准提升主要说明输出概率更可用，但不等于解释是因果正确的；跨医院、跨采样协议、value-pending、变量联测规则改变时，rationale 是否仍指向稳定病程证据，还需要系统实验。

### 对 Sampling-Policy Shift 的启发

TRIAGE 对 Sampling-Policy Shift 的横向启发是：解释层可以成为采样政策泄漏的诊断窗口。若模型在 rationale 中频繁把“某项检查被下单”“近期观测更密集”“某变量长时间未测”作为直接风险证据，我们就能更明确地区分它是在利用真实病程，还是在利用医院观察流程。相比只看 logits，rationale 更容易暴露 policy shortcut。

纵向深化上，可以设计 policy-audited dialectical reasoning：要求 LLM 分别输出 state rationale 与 policy rationale。state rationale 只能引用跨采样策略稳定的病情变化、数值趋势和临床机制；policy rationale 则解释观测为何出现、为何缺失、为何某些变量联测，只用于校准和偏移警报。对同一患者潜在轨迹生成不同反事实采样策略时，要求 state rationale、连续风险分数和分类排序保持稳定，同时允许 policy rationale 改变。这样能把 TRIAGE 的“可解释风险校准”进一步变成 sampling-policy shift 下的可审计分类决策。
