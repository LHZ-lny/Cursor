# Paper Daily - 2026-07-22

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / medical time series generalization 的顶会论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、KDD 2026、OpenReview、ICLR virtual site、AAAI Proceedings、arXiv 与论文项目页。
- 已排除全部黑名单论文；同时排除 STAR-Set、Status-Aware SSL 这类 ICLR workshop 条目，排除 ReIMTS、TFMixer、Time-IMM、Rethinking Irregular Time Series Forecasting 等偏 forecasting / benchmark 而非分类主任务的候选，排除 APSIPA Transactions 期刊论文 `A Statistical Approach for Modeling Irregular Multivariate Time Series with Missing Observations`，以及官方 ICLR virtual 页面未确认的 LLM4EHR 候选。本次保留全新工作 1 篇：FORMED 是 ICLR 2026 正会 Poster，虽然不以 irregular sampling 为标题主轴，但它直接处理医疗时序分类中跨数据集、跨通道配置、跨长度和跨任务定义的泛化问题；这些异质性常由设备、队列、采样协议和临床任务共同造成，对 Sampling-Policy Shift 有明确横向启发。

## 1. Repurposing Foundation Model for Generalizable Medical Time Series Classification

- 方法名：FORMED / Foundation Model Repurposed for Medical Time Series Classification
- 会议：ICLR 2026 Poster
- 作者：Nan Huang, Haishuai Wang, Zihuai He, Marinka Žitnik, Xiang Zhang
- 官方页：https://iclr.cc/virtual/2026/poster/10006735
- 论文：https://arxiv.org/abs/2410.03794
- 代码：https://github.com/DL4mHealth/FORMED
- 关键词：medical time series classification, foundation model repurposing, cross-dataset generalization, variable channels, task-specific label queries, lightweight adaptation

### 场景、任务与核心难点

FORMED 面向医疗时序分类中的真实部署泛化问题，典型输入包括 EEG、ECG 等多通道医学信号，任务覆盖不同疾病、不同诊断标签和不同数据集。它关注的核心不是单一 benchmark 上的同分布准确率，而是模型在新数据集、新通道配置、新序列长度和新标签空间下能否低成本适配。现实医疗时序往往来自不同设备、采集协议、患者群体和实验流程；即使信号本身看起来是规则采样，背后的采样频率、通道选择、记录时长和诊断任务也已经体现出强烈的策略差异。

论文指出，通用 time-series foundation models 多数预训练目标偏 forecasting，且常以 channel-independent 的方式处理多变量序列；直接拿来做医疗分类时，既不能充分建模跨通道诊断模式，也容易需要大量 task-specific adapter 或全模型微调。FORMED 因此提出“repurposing”而不是普通 fine-tuning：冻结通用 backbone，用 task-specific channel embeddings 与 label queries 动态适配任意通道数和类别数，再用共享 decoding attention layer 在多个医疗分类任务上学习可迁移的领域知识。适配新数据集时只需训练轻量 label query 等约 0.1% 参数，从而降低小样本医疗任务中过拟合和重训成本。

### 审稿人视角：价值与不足

最有价值的技术思想是把医疗时序 foundation model 的问题从“再预训练一个更大的分类模型”转化为“如何把 forecasting backbone 重新目的化为跨任务分类器”。task-specific channel embeddings 解决不同数据集通道数和通道含义不一致的问题，label queries 解决标签空间变化的问题，共享 decoding attention 则承担跨任务可迁移医疗知识的聚合。对审稿人而言，这个设计的亮点在于边界清晰：冻结 backbone 保留通用时序表征，少量可变 query 处理任务接口，跨数据集训练的共享 decoder 承载医疗分类归纳偏置。

不足在于，FORMED 对 irregular sampling / asynchronous observation 的建模仍是间接的。论文强调 channel configuration、sample length、diagnostic task 和 patient heterogeneity，但没有像 IMTS/EHR 模型那样显式消费任意时间戳、变量级 mask、事件触发观测或 missingness pattern。若不同数据集的通道配置和记录长度本身来自采样政策，例如某医院只在高风险患者上采集更长 EEG，或某设备协议决定某些导联缺失，那么 channel embeddings 与 label queries 可能会把策略差异当成任务差异吸收。论文展示了跨数据集泛化能力，但还缺少在同一潜在任务下改变采样频率、观测窗口、通道选择政策后的反事实评估。

### 对 Sampling-Policy Shift 的启发

FORMED 对 Sampling-Policy Shift 的横向启发是：策略偏移并不总是表现为 EHR 式的“某个变量什么时候被测”，也可能表现为任务接口层面的结构变化，包括可用通道集合、记录窗口长度、采样频率、标签粒度和数据集协议。我们研究非规则采样时，不能只在 encoder 内部做 mask/delta-t 不变性，还需要在“通道-标签-任务接口”层面设计可迁移机制。FORMED 的 channel embeddings 与 label queries 提示我们，可以把不同采样政策映射为可控的 policy/task query，而不是让模型把所有结构差异混入状态表征。

纵向深化上，可以把 FORMED 扩展为 policy-query repurposing 框架：state queries 聚合跨策略稳定的生理或系统状态，label queries 负责目标任务语义，policy queries 专门解释采样频率、通道可用性、记录时长和医院/设备协议。训练时对同一潜在病程施加不同采样策略增强，约束 state queries 与分类 logits 保持稳定，同时允许 policy queries 区分策略环境；适配新医院或新设备时，只更新少量 policy/label query，而不破坏共享 state decoder。这样可以把 FORMED 的跨数据集医疗分类泛化进一步推进到“跨采样政策、跨任务接口”的稳健分类。
