# Paper Daily - 2026-08-22

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-08-21 的新增标题，纳入黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / clinical irregular time series classification 的顶会论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、NeurIPS 2025、KDD 2025/2026、OpenReview、ICLR/ICML virtual site、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时继续排除历史日报已明确标记的 MedFuse withdrawn submission、VITAL/Mind the Missing 这类暂无顶会录用信息的预印本、TranSCANE/MUSE-Net 等时间窗口或 venue 不匹配候选，以及 ReDiTT/Time-IMM 等偏 forecasting/generation 的工作。本次保留全新工作 1 篇：CauKer 是 ICLR 2026 Oral，严格来说不是 IMTS 专用模型，但它面向 time series classification foundation models，提出可控、因果一致的合成预训练数据生成机制；在直接 IMTS 顶会命中几乎已被历史日报覆盖的情况下，它对构造 sampling-policy shift 的反事实预训练与评测最有新增价值。

## 1. CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data

- 会议：ICLR 2026 Oral
- 作者：Shifeng Xie, Vasilii Feofanov, Jianfeng Zhang, Themis Palpanas, Ievgen Redko
- OpenReview：https://openreview.net/forum?id=xBW2FIfswU
- 官方页：https://iclr.cc/virtual/2026/poster/10006662
- 论文：https://arxiv.org/abs/2508.02879
- 代码：https://github.com/ShifengXIE/CauKer
- 关键词：time series classification, foundation model pretraining, synthetic data, Gaussian Process kernel composition, Structural Causal Models, scaling laws

### 场景、任务与核心难点

CauKer 面向 time series classification foundation models 的预训练问题。近两年 TSFM 的主流路线常依赖大规模真实时序语料，但分类任务的数据来源高度碎片化：UCR/UEA、医疗、生理、工业和科学数据之间的采样频率、变量数、噪声、类别语义和长度分布差异极大。若直接堆叠真实数据做预训练，成本高、版权和隐私约束重，而且论文观察到真实数据上的 scaling law 很不规则：数据量或模型规模变大并不稳定带来更好的 zero-shot 分类能力。

论文的核心做法是用合成数据替代大规模真实预训练语料。CauKer 先通过 Gaussian Process kernel composition 组合趋势、周期、平滑性、非平稳性等时间结构，再用 Structural Causal Models 将多个生成过程按因果图传播和耦合，从而生成具有现实形态、非线性交互和因果一致性的分类时序。作者用这些合成序列预训练不同架构的分类 TSFM，并在多个真实分类 benchmark 上做 zero-shot / transfer 评估；结果显示，合成数据不仅能达到或接近真实数据预训练效果，还呈现更清晰的数据规模与模型规模 scaling law。

### 审稿人视角：价值与不足

最有价值的思想是把 time series classification 的基础模型瓶颈从“再收集更多真实序列”转向“能否设计足够结构化、可控、覆盖广泛机制的合成世界”。这对审稿人很有吸引力，因为它不是单纯的数据增强，而是把 GP kernel 的时间形态组合能力和 SCM 的变量生成依赖结合起来，让预训练数据具备可解释的生成因素。它也说明分类 TSFM 的泛化能力可能更依赖训练分布覆盖到足够多的形态机制，而不一定依赖真实数据本身。

不足在于，CauKer 主要面向规则或标准化后的通用时序分类 benchmark，还没有把不规则采样、变量异步、informative missingness 或医院观测策略显式作为生成因子。GP kernel composition 能产生丰富的状态轨迹，SCM 能产生变量间依赖，但当前 observation process 更像隐含或固定的采样层；如果目标是 ICU/EHR 这类非规则采样分类，真实难点恰恰在于“哪些值被看见、何时被看见、为什么被看见”本身与标签和环境耦合。因此，这篇论文证明了可控合成预训练的潜力，但尚未证明其合成世界覆盖了 sampling-policy shift 下最关键的观测机制变化。

### 对 Sampling-Policy Shift 的启发

CauKer 对 Sampling-Policy Shift 的横向启发是：我们可以把采样策略偏移从被动 benchmark 现象，升级为可控合成预训练和评测维度。现有许多 IMTS 方法只在真实数据或随机 mask 下测试鲁棒性，难以知道模型失败来自状态动力学改变、噪声改变，还是采样政策改变。CauKer 的 GP + SCM 框架提供了一个生成底座：先生成同一潜在连续状态和类别机制，再叠加多套 observation policy，例如固定间隔采样、阈值触发采样、告警后密集采样、成本约束下变量选择、医院特定联测规则等。

纵向深化上，可以设计 policy-aware CauKer：把生成过程拆成 state generator、label generator 和 sampling-policy generator 三层。预训练时要求模型在不同 observation policy 下保持 state representation 和分类 logits 一致，同时显式预测 policy metadata 或 sampling trace，用来做偏移诊断。评测时则报告 in-policy、cross-policy、counterfactual-policy 三类指标，并检验模型是否在只给时间戳/mask 的 policy-only 设置下泄漏标签。这样能把 CauKer 的“合成数据可替代真实预训练语料”推进到“合成采样政策可系统训练和审计 policy-invariant irregular time-series classifiers”。
