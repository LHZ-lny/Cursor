# Paper Daily - 2026-06-25

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / multimodal irregular / medical time series classification 的顶会或顶会相关论文，重点核对 AAAI 2026、ICLR 2026、ICML 2026、KDD 2025/2026、WWW 2026、OpenReview、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除偏 forecasting、causal discovery、普通规则时序分类或仅为 workshop 且已有更强正会替代的条目。本次保留全新工作 2 篇：MedSpaformer 是 AAAI 2026 正会医疗时序分类工作，虽不以 irregular sampling 为标题主轴，但其可变长度/通道维度、跨数据集迁移和 token sparsification 与采样策略偏移高度相关；MILM 直接面向 multimodal irregular time series classification 与 informative sampling，是近月对 sampling pattern 可预测性的最直接建模之一。

## 1. MedSpaformer: A Transferable Transformer with Multi-Granularity Token Sparsification for Medical Time Series Classification

- 会议：AAAI 2026 Technical Track
- 作者：Jiexia Ye, Weiqi Zhang, Ziyue Li, Jia Li, Fugee Tsung
- 链接：https://ojs.aaai.org/index.php/AAAI/article/view/40001
- 论文：https://arxiv.org/html/2503.15578v3
- DOI：https://doi.org/10.1609/aaai.v40i33.40001
- 关键词：medical time series classification, transferable transformer, token sparsification, multi-granularity cross-channel encoding, variable lengths/channels, cross-dataset transfer

### 场景、任务与核心难点

MedSpaformer 面向医疗时序分类中的跨数据集、少标注和异质输入问题。典型场景包括不同医院、不同设备或不同诊断任务下的 ECG/EEG/临床多通道时序：序列长度、通道数、标签空间和任务定义并不一致，直接训练一个任务专用模型会导致泛化差、迁移成本高，也难以在少样本或零样本诊断中复用已有知识。

论文解决的核心难点不是传统意义上“每个变量带任意时间戳”的 irregular EHR 建模，而是医疗时序在部署时常见的结构异质性与冗余观测：多通道信号中有大量非关键 token，不同粒度的局部形态和跨通道关系都可能影响分类，但它们在不同数据集中的出现频率、长度和通道配置并不稳定。MedSpaformer 因此提出 sparse token-based dual-attention mechanism，在全局上下文建模时动态聚焦 informative tokens；同时用 multi-granularity cross-channel encoding 捕捉不同时间粒度下的通道内/通道间依赖，并通过 adaptive label encoder 缓解跨数据集标签空间不一致。

### 审稿人视角：价值与不足

最有价值的思想是把医疗时序分类中的“可迁移性”与“稀疏 token 选择”放在同一个框架内处理。许多医疗时序模型在单一 benchmark 上很强，但换数据集、换通道定义或换标签空间后需要重新设计输入层和分类头；MedSpaformer 的多粒度 token sparsification 让模型在不同长度、不同通道维度下仍能抽取任务相关片段，adaptive label encoder 则把标签语义显式纳入迁移过程。这对真实医疗部署比单纯提升同分布 accuracy 更有价值。

不足在于，它对非规则采样机制的建模仍是间接的。论文强调 variable lengths 与 channel dimensions，但没有像 IMTS/EHR 模型那样显式处理变量级异步时间戳、事件触发采样和 missingness pattern。token sparsification 也可能引入新的 shortcut：如果某些通道或片段在训练医院中因为采样流程更频繁出现，模型可能把“被选中的 token”与疾病标签绑定，而不是学习稳定的生理状态。跨数据集迁移实验能部分说明鲁棒性，但还不足以证明在采样政策改变时 token selection 仍保持语义稳定。

### 对 Sampling-Policy Shift 的启发

MedSpaformer 对 sampling-policy shift 的横向启发在于：采样策略偏移可以表现为 token 预算和 token 重要性分布的偏移。不同医院或设备可能让某些时间段、通道或诊断片段更容易被记录，sparsification 模块若不受约束，就会把这种可见性差异放大为分类证据。因此，我们可以把 token selection frequency、multi-granularity attention map 和跨通道 token 共现率作为策略偏移诊断指标。

纵向深化上，可以把 token sparsification 改造成 policy-aware selection：一组 state tokens 承载跨策略稳定的生理/系统状态，进入分类主路径；另一组 policy tokens 解释观测密度、通道可用性和数据集协议差异，只用于校准或偏移诊断。训练时可对同一潜在轨迹施加不同采样策略增强，约束 state-token representation 与 logits 稳定，同时允许 policy-token 分布变化。这样能把 MedSpaformer 的跨数据集迁移能力进一步推进到“跨采样政策迁移”。

## 2. MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling

- 全称：Multimodal Irregular time series Language Model
- 会议/状态：WWW / ACM Web Conference 2026 方向；arXiv 2026-05 版本
- 作者：Hsing-Huan Chung, Shijun Li, Yoav Wald, Xing Han, Suchi Saria, Joydeep Ghosh
- 论文：https://arxiv.org/html/2605.13711v1
- 关键词：multimodal irregular time series, informative sampling, LLM, XML triplets, value-redacted training, EHR classification

### 场景、任务与核心难点

MILM 面向 multimodal irregular time series classification，尤其是 EHR 中数值化验、生命体征和临床文本共同组成的异步记录。与纯数值 IMTS 不同，MITS 的观测不仅有值和时间戳，还包含模态差异：某些时刻可能只有文本记录，某些时刻有化验值，某些检查的“被测量”本身就携带医生怀疑、治疗流程或病情变化的信息。任务通常是 ICU 或住院风险预测/分类，例如死亡风险、再入院或临床事件预测。

论文抓住的核心难点是 informative sampling：在不规则医疗记录中，什么时候观测、观测哪个通道，往往和标签高度相关。传统方法要么把 sampling pattern 当作缺失掩码辅助特征，要么在插补中隐式吸收；MILM 则直接让 LLM 学习这种结构。它把 MITS 序列化为按时间排序的 XML triplets，用两阶段 fine-tuning：第一阶段在 value-redacted MITS 上训练，仅凭时间和通道采样模式预测；第二阶段再加入观测值，让模型联合利用 sampling pattern 与 values。论文还设计 value-pending evaluation，模拟预测时某些值尚未返回但其时间/通道信息已知的临床场景。

### 审稿人视角：价值与不足

最有价值的思想是把 sampling pattern 从“可能有用的 side information”提升为主监督对象。第一阶段 value-redacted training 是一个很清晰的诊断实验：如果模型在没有数值的情况下仍能预测，说明采样行为本身具有强标签信息；第二阶段再融合数值，使模型不会只依赖观测值而忽略采样机制。XML triplet serialization 也提供了一个简单统一的多模态接口，让 LLM 可以同时消费时间、通道、值和文本，而不必为每类模态单独设计复杂融合模块。

不足同样来自它的优势：显式学习 informative sampling 很容易变成显式学习 sampling-policy shortcut。如果训练医院中某些检查只在高风险患者中触发，MILM 会合理地利用这种模式；但换到另一家医院、另一套检查协议或另一个成本约束后，同样的未测/已测模式可能有完全不同的语义。论文已经意识到跨医院 sampling distribution shift 是未来方向，但当前主要证明 sampling pattern 有预测力，并未充分回答哪些 sampling pattern 是可迁移的状态信号、哪些只是环境特定政策信号。此外，LLM 序列化会带来上下文长度、数值精度和可解释性控制问题。

### 对 Sampling-Policy Shift 的启发

MILM 对我们的 Sampling-Policy Shift 研究几乎是直接横向启发：在非规则采样下，不能简单问“是否应该使用 missingness”，而要问“哪些 missingness 来自状态，哪些来自政策”。value-redacted training 可以被改造成策略泄漏检测器：在不同环境中只用采样时间和通道训练一个 policy-only classifier，如果它能强预测标签，就说明当前任务存在强 sampling-policy shortcut。

纵向深化上，可以把 MILM 的两阶段训练改为三分解目标：state branch 从观测值和文本中学习跨策略稳定状态，policy branch 从时间/通道/缺失中学习采样政策，classification head 通过不变性或对抗约束限制对 policy branch 的直接依赖。对同一病程生成不同采样策略视图时，要求 state summary 和 logits 一致，同时允许 policy summary 区分医院协议、检查触发规则或 value-pending 机制。这样既承认 informative sampling 的存在，又避免模型把训练环境中的采样政策当成可迁移病理证据。
