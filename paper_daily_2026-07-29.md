# Paper Daily - 2026-07-29

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / variable-length medical time series classification 的顶会或顶会相关论文，重点核对 NeurIPS 2025 官方论文页、OpenReview、ICLR/ICML/AAAI/ICASSP 2026 页面与 arXiv。
- 已排除全部黑名单论文；同时排除 withdrawn 的 MedFuse、偏 forecasting/generation/imputation 的 ReIMTS/Time-IMM 等条目，以及仅为 workshop 且与本轮主会候选相比证据强度较弱的 SITS、RAxSS 等候选。本次保留全新工作 2 篇：RoMAE 是 NeurIPS 2025 正会论文，明确评估 irregularly sampled multivariate time-series classification；STaRFormer 是 NeurIPS 2025 正会论文，在 P19/P12/PAM 等 irregularly sampled classification benchmark 上报告了系统结果。

## 1. Rotary Masked Autoencoders are Versatile Learners

- 简称：RoMAE
- 会议：NeurIPS 2025 Poster
- 作者：Uros Zivanovic, Serafina Di Gioia, Andre Scaffidi, Martin de los Rios, Gabriella Contardo, Roberto Trotta
- 官方页：https://proceedings.neurips.cc/paper_files/paper/2025/hash/c2626ef6cdaaa6a18927832820079e1d-Abstract-Conference.html
- 论文：https://arxiv.org/abs/2505.20535
- 关键词：irregular multivariate time series, masked autoencoder, rotary positional embedding, continuous positions, ELAsTiCC classification, data-efficient representation learning

### 场景、任务与核心难点

RoMAE 面向一类跨模态但对不规则时序尤其关键的问题：如何让标准 Transformer/MAE 在不规则、多变量、连续位置输入上工作，而不为每个数据形态单独设计专门架构。论文的 irregular time-series classification 实验覆盖 DESC ELAsTiCC 这类天文多波段光变曲线以及 UEA 中被构造为不规则采样的多变量时序；这类数据的共同难点是时间戳不是离散网格索引，变量或波段观测不完全同步，且样本数量、观测密度和时间尺度差异很大。

核心难点在于，标准 MAE/ViT 的位置编码通常假设规则网格或离散 token 序列；如果直接把不规则时间戳离散化，会丢失真实间隔和多维连续位置关系。RoMAE 的做法是把 Rotary Positional Embedding 扩展到多维连续位置，使模型可以直接把时间、波段或其他坐标作为连续位置输入，并在 MAE 式遮蔽重构和分类微调中复用标准 Transformer 组件。论文还特别指出，输入中加入 learned embeddings 会破坏 RoPE 的相对位置性质，因此需要明确处理连续位置重构与 [CLS] token 的角色。

### 审稿人视角：价值与不足

最有价值的思想是把“不规则时序是否必须使用专门 continuous-time / graph / interpolation 架构”这个问题重新打开。RoMAE 表明，只要连续位置编码处理得足够谨慎，标准 MAE 加 RoPE 就能在困难的不规则分类任务上超过一些专门模型。这对审稿人很有吸引力，因为它把方法复杂度从新架构转移到位置表示的正确性，并给出跨图像、音频和不规则时序的一致证据。

不足在于，RoMAE 主要把不规则性视为连续位置编码和表示学习问题，而不是观测过程问题。对 ELAsTiCC 或医疗 IMTS 来说，哪些时间点被观测、哪些波段或变量缺失，往往来自望远镜巡天 cadence、天气、设备策略或临床测量政策。RoPE 能表达连续位置，却不会自动区分“状态本身的相位/动态差异”和“采样政策造成的位置覆盖差异”。此外，论文目标是证明 MAE 的通用性，针对 sampling-policy shift、跨观测策略稳健性和 policy shortcut 的分析仍然不足。

### 对 Sampling-Policy Shift 的启发

RoMAE 对我们的横向启发是：采样策略偏移可以被重新表述为 continuous positional distribution 的偏移。不同医院测量协议或不同天文巡天 cadence 会改变时间位置、变量/波段位置和遮蔽位置的联合分布；如果模型的位置编码过强地绑定这些分布，分类器可能学习到策略特定的相对位置模式。

纵向深化上，可以把 RoMAE 改造成 policy-aware positional MAE：state positional branch 学跨策略稳定的连续动力学位置关系，policy positional branch 学采样日程、变量可见性和观测窗口覆盖。对同一潜在轨迹生成多种反事实采样视图时，约束 state RoPE representation、MAE latent 和分类 logits 保持一致，同时允许 policy branch 重构具体采样位置。这样能把“标准 Transformer 也能处理不规则位置”推进到“标准 Transformer 能否分离状态位置与采样政策位置”。

## 2. STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data

- 会议：NeurIPS 2025 Poster
- 作者：Maximilian Forstenhausler, Daniel Kulzer, Christos Anagnostopoulos, Shameem A. Puthiya Parambath, Natascha Weber
- 官方页：https://neurips.cc/virtual/2025/poster/116860
- OpenReview：https://openreview.net/forum?id=fDR4hzavDF
- 项目页：https://star-former.github.io/
- 关键词：irregularly sampled time series, semi-supervised contrastive learning, dynamic attention-based regional masking, P19/P12/PAM classification, task-informed representation

### 场景、任务与核心难点

STaRFormer 面向更广义的 sequential data prediction，但论文明确把 non-stationary and irregularly sampled time series 作为核心实验场景之一，并在 P19、P12、PAM 等典型 irregular sampled classification benchmark 上报告结果。其应用设定包括医疗、生理活动和用户意图等序列任务：真实数据中有效信号常集中在少数区域，采样频率和可见性受传感器限制、环境干扰或记录流程影响，标注又往往有限。

论文解决的核心难点不是单纯处理任意时间戳，而是如何在半监督条件下让模型找到“对任务有用的序列区域”。STaRFormer 提出 Dynamic Attention-based Regional Masking (DAReM)：利用模型注意力动态识别任务相关区域，再通过区域遮蔽构造更有针对性的增强视图；同时结合 batch-wise/self 与 class-wise 的 semi-supervised contrastive learning，使表示既保留实例内部结构，也靠少量标签形成任务相关聚类。对 irregular sampled classification 来说，这相当于把随机 mask/augmentation 改造成任务感知的区域级采样。

### 审稿人视角：价值与不足

最有价值的技术思想是把 masking 从“预训练时随机遮住一些 token”推进到“根据任务相关性动态遮住区域”。这对不规则时序尤其有意义：观测稀疏且长短不一时，随机增强很容易遮掉少量关键事件，或者反复保留大量无信息背景；DAReM 则试图把注意力已经发现的关键区域纳入增强策略，迫使模型学习更稳定的 task-informed representation。再加上半监督对比目标，论文在标签稀缺但无标签序列较多的场景下具有实用价值。

不足在于，STaRFormer 的通用性也带来机制不够细的问题。它证明了在 P19/P12/PAM 等 benchmark 上有效，但没有显式建模变量级异步、delta-t 分布、missingness pattern 或医院/设备采样策略。动态注意力可能发现真实任务区域，也可能发现训练环境中特定采样政策留下的区域：例如某类患者被密集监测的窗口、某类活动更容易被传感器捕捉的片段，都会成为 attention 和 regional masking 的高权重对象。若没有跨策略验证，task-informed masking 可能变成 policy-informed masking。

### 对 Sampling-Policy Shift 的启发

STaRFormer 对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以表现为“哪些区域被认为是任务相关”的分布偏移。传统做法常比较 mask ratio、时间间隔或变量共现；STaRFormer 提醒我们还应比较 attention-selected regions、DAReM mask regions 和 contrastive positive/negative structure 在不同采样策略下是否稳定。

纵向深化上，可以把 DAReM 扩展为 state-policy 双区域遮蔽：state regions 是跨采样策略稳定的病程、故障或行为片段，进入分类主路径；policy regions 解释为何某些窗口被更密集观测、为何某些片段更容易出现噪声或缺失，只用于偏移诊断。训练时可对同一底层轨迹施加不同采样策略增强，要求 state-region attention、state contrastive embedding 和 logits 保持一致，同时允许 policy-region attention 变化。这样能把“任务感知增强”推进到“策略不变任务区域学习”。
