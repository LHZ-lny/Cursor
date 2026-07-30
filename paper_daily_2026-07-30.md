# Paper Daily - 2026-07-30

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / clinical time-series inference 的顶会或顶会 workshop 论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、ICASSP 2026、OpenReview、ICML virtual site、AAAI Proceedings、TSALM Workshop 与 arXiv/机构论文页。
- 已排除全部黑名单论文；同时排除 SITS 这类 ICML 2025 workshop 且时间窗口偏旧的候选、MedFuse 这类 ICLR 2026 withdrawn submission、ReDiTT/TFMixer/ASTGI 等偏 forecasting/generation 的条目，以及 MLLM4TS/TiViT/Time-CoT 这类主要面向通用或规则时序分类、与异步采样机制关系较弱的工作。本次保留全新工作 1 篇：Cached Foundation Model Summaries 是 ICLR 2026 TSALM Workshop Poster，面向 MIMIC-IV 长程不规则临床事件序列的内存受限预测/分类式推理，虽不是正会主论文，但它直接研究长历史不规则事件、近期高分辨率窗口和临床风险 AUROC 之间的取舍，对 sampling-policy shift 下的上下文预算与历史压缩问题有明确启发。

## 1. Cached Foundation Model Summaries for Memory-Efficient Clinical Time Series Inference

- 会议：ICLR 2026 TSALM Workshop Poster
- 作者：R. Al Attrach, R. Fani, D. Restrepo, Y. Jia, L. A. Celi, P. J. Schüffler
- OpenReview：https://openreview.net/forum?id=kdoFfrlOZj
- 机构页：https://mcml.ai/publications/afr+26/
- 关键词：irregular clinical time series, long EHR event history, memory-efficient inference, cached foundation-model summary, recent-window prediction, FiLM conditioning, MIMIC-IV

### 场景、任务与核心难点

这篇工作面向临床 EHR 中很长、非规则间隔的患者事件序列。真实部署时，Transformer 或临床 foundation model 往往无法把数千个历史事件全部放进显存或延迟预算中；但只看最近几个事件，又可能丢掉慢性病史、既往治疗、长期异常趋势等对急性风险预测有用的信息。论文在 MIMIC-IV 上研究一个内存受限的临床推理任务：预训练 foundation model 离线把长历史压缩成固定大小 cached summary，轻量预测模型在线只处理短近期窗口，并用该 summary 条件化近期事件表示来预测临床风险。

核心难点不是再设计一个更大的 irregular encoder，而是回答“在不规则临床序列中，有限上下文预算应该分给长历史摘要还是近期高分辨率观测”。作者通过 252 组实验发现，cached summary 在近期窗口极短时最有价值：当 recent window 只有 8 个事件时，AUROC 相对提升约 6.5%，但窗口扩展到 256 个事件后收益几乎消失。集成方式上，Feature-wise Linear Modulation (FiLM) 优于把 summary 当作额外 token 注入；历史位置上，近期历史摘要比远期历史摘要更有信息量。

### 审稿人视角：价值与不足

最有价值的思想是把不规则临床时序的部署瓶颈具体化为“历史压缩 + 近期窗口”的可测量取舍。很多长上下文 EHR 模型默认更多上下文一定更好，但这篇工作给出一个更工程化也更科学的问题：当硬件只能容纳几十个事件时，离线 summary 能补回多少历史信息？当近期窗口足够长时，summary 是否仍有必要？FiLM 优于 token injection 的结果也很有启发，因为它说明长历史更适合作为对近期事件表示的条件调制，而不一定适合作为一个与原始事件竞争注意力的普通 token。

不足在于，它主要是经验性部署研究，还没有充分解释 cached summary 中究竟压缩了 patient state、医院流程，还是采样政策痕迹。MIMIC-IV 单数据源实验能说明内存预算下的平均收益，但不能证明换医院、换 ICU 记录制度或换事件触发规则后 summary 仍然稳定。另一个限制是，离线 foundation-model summary 的质量和偏差被当作给定条件；如果 foundation model 本身已经学习了某家医院的采样 shortcut，FiLM 调制可能只是更高效地把这种 shortcut 注入近期窗口模型。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：采样策略偏移不只改变原始观测密度，也改变“近期高分辨率上下文”和“压缩历史上下文”的相对价值。某些医院如果更依赖密集近期监测，短 recent window 可能已经含有强策略信号；另一些医院如果记录更稀疏，cached summary 可能承载更多由流程和长期测量习惯诱导的 policy trace。因此，summary gain 随窗口长度的曲线、FiLM gate 强度、近期/远期 summary 的边际收益，都可以作为采样政策偏移诊断指标。

纵向深化上，可以把 cached summary 分解为 state summary 与 policy summary。state summary 只压缩跨采样策略稳定的病程、慢性状态和长期异常趋势，用于条件化分类主路径；policy summary 专门描述观测频率、事件类型覆盖、近期窗口可见性和医院记录习惯，只用于校准、偏移检测或不确定性估计。训练时可对同一患者轨迹构造不同反事实采样窗口，要求 state summary 条件下的 logits 保持一致，同时允许 policy summary 和 FiLM 调制残差变化。这样能把“内存高效推理”推进到“在上下文预算受限时仍控制采样政策泄漏”的不规则时序分类框架。
