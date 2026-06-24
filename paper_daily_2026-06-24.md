# Paper Daily - 2026-06-24

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification 的顶会或顶级应用会议论文，重点核对 NeurIPS 2025、AAAI 2026、ICLR 2026、ICML 2026、KDD 2025/2026、ICASSP 2026、OpenReview、AAAI Proceedings 与 arXiv/会议页面。
- 已排除全部黑名单论文；同时排除 ICLR 2026 workshop 条目 STAR Set Transformer、ICLR withdrawn submission MedFuse、偏 forecasting-only 的 TFMixer、以及普通规则时序分类且缺少非规则采样证据的候选。本次保留全新工作 2 篇。

## 1. MedSpaformer: A Transferable Transformer with Multi-Granularity Token Sparsification for Medical Time Series Classification

- 会议：AAAI 2026 Technical Track
- 作者：Jiexia Ye, Weiqi Zhang, Ziyue Li, Jia Li, Fugee Tsung
- 链接：https://ojs.aaai.org/index.php/AAAI/article/view/40001
- 论文：https://arxiv.org/html/2503.15578
- DOI：https://doi.org/10.1609/aaai.v40i33.40001
- 关键词：medical time series classification, token sparsification, multi-granularity cross-channel encoding, transferable classification, variable length/channel dimension

### 场景、任务与核心难点

MedSpaformer 面向医疗多通道时序分类，目标是在跨数据集、少标签甚至零样本诊断场景下复用同一类模型。它不以 irregular sampling 为标题主轴，但 AAAI 摘要明确强调医疗时序分类中的复杂多通道依赖、信息冗余、标签稀缺，以及可适配 variable lengths 和 channel dimensions 的输入形态；这些特征与真实医疗采样策略变化高度相邻：不同医院、设备和任务往往有不同观测窗口、通道集合和记录密度。

论文解决的核心难点是：传统 Transformer 在医疗分类中容易把所有 token 都纳入全局注意力，既带来冗余计算，也容易在跨数据集迁移时学习到数据集特定的局部模式。MedSpaformer 通过 sparse token-based dual-attention 在全局上下文中动态筛选更有判别力的 token；再用 multi-granularity cross-channel encoding 同时建模不同粒度内和粒度间的时间依赖、通道相关；最后用 adaptive label encoder 缓解跨数据集标签空间不一致，使模型更适合迁移和少样本诊断。

### 审稿人视角：价值与不足

最有价值的思想是把“哪些 token 值得进入分类决策”作为模型主干的一部分，而不是简单依赖完整序列注意力或后处理特征选择。对于医疗时序，很多观测点是冗余、噪声或任务无关的，token sparsification 可以同时提高效率与判别性；multi-granularity cross-channel encoding 则避免只在单一时间尺度上理解通道交互。自适应标签编码也有现实价值，因为跨疾病、跨设备、跨数据集迁移时，标签语义不完全一致是常见障碍。

不足在于，它对不规则采样机制的建模仍是间接的。模型可以适配 variable length 和 channel dimension，但没有显式区分“观测 token 重要”究竟来自真实状态异常，还是来自医院流程、设备协议、医生触发测量等采样政策。token sparsification 也可能放大 sampling-policy shortcut：如果某些类别在训练集中天然拥有更多关键化验、更多高质量片段或更长监测窗口，模型可能学会选择这些策略性 token，而不是跨环境稳定的状态 token。论文强调 transferability、few-shot 和 zero-shot 能力，但尚缺少跨采样频率、跨观测触发规则或跨机构采样协议的反事实评估。

### 对 Sampling-Policy Shift 的启发

MedSpaformer 对我们的横向启发是：sampling-policy shift 可以被看成 token candidate set 与 token selection distribution 的变化。相比只统计 mask ratio 或 delta-t，我们还应统计不同策略下哪些 token 被 sparse attention 保留、哪些通道/时间粒度成为分类瓶颈，以及 token selection 是否随采样政策而非状态语义漂移。

纵向深化上，可以把 token sparsification 改造成 policy-aware selection：一组 state tokens 进入分类主路径，另一组 policy tokens 解释观测窗口、通道可用性、联测习惯和标签空间差异。对同一潜在病程生成不同采样策略视图时，约束 state-token selection、multi-granularity representation 和 logits 保持稳定，同时允许 policy-token selection 变化。adaptive label encoder 也可用于跨策略校准：当同一临床标签在不同机构的观测协议下语义边界不同，标签编码应显式吸收环境差异，而不是让分类器隐式依赖采样 shortcut。

## 2. Rethinking Large Language Models for Irregular Time Series Classification in Critical Care

- 会议：ICASSP 2026
- 作者：Feixiang Zheng, Yu Wu, Cecilia Mascolo, Ting Dang
- 论文：https://arxiv.org/html/2601.16516
- 机构页：https://www.repository.cam.ac.uk/items/86268aaf-8188-4d63-a2d5-7b8f67746841
- 关键词：irregular ICU time series classification, LLM-based time-series modeling, multimodal alignment, encoder comparison, critical care prediction

### 场景、任务与核心难点

这篇工作面向 ICU critical-care 场景中的不规则时序分类，典型任务是基于 PhysioNet / MIMIC 类 ICU 数据进行死亡风险或临床结局预测。ICU 数据具有高缺失率、变量异步采样、观测频率受病情和流程触发影响等特点；同时，近期大量 LLM-for-time-series 方法声称可通过重编程、语义锚点或跨模态对齐提升泛化，但这些方法多在规则采样数据上验证，是否适合 irregular ICU classification 并不清楚。

论文的核心难点不是再提出一个单一分类器，而是系统拆解 LLM-based time-series modeling 在不规则 ICU 数据上的两个关键组件：time-series encoder 与 multimodal alignment strategy。作者把 Time-LLM、S2S2IP、CALF、FSCA 等代表性 LLM 方法放入统一测试床，并与传统监督/自监督 irregular baselines 比较。结果显示，显式建模 irregularity 的 encoder 比 vanilla Transformer 带来显著 AUPRC 增益；alignment strategy 有一定作用，但影响小于 encoder；同时，LLM 方法训练成本高，且在 few-shot 数据稀缺设置下不如预期。

### 审稿人视角：价值与不足

最有价值的贡献是提供了一个负责任的“降温型”评测：它没有默认 LLM 一定优于专用 irregular time-series 模型，而是把性能来源拆成 encoder 与 alignment 两部分。结论很有实践意义：在不规则 ICU 分类中，先把时间戳、缺失和异步观测建模好，往往比追求更复杂的 LLM 对齐更关键。这对审稿人而言很重要，因为它能帮助社区避免把规则时序或 NLP 场景中的 LLM 成功经验机械迁移到临床不规则时序。

不足在于，它主要是经验评测和组件归因，方法创新不如架构型论文强；同时，它评估的是当前几类 LLM-based baselines，并不代表未来经过专门设计的临床时间序列 LLM 上限。另一个关键限制是，论文虽强调 ICU 数据 irregular 和 missing，但还没有把 sampling-policy shift 作为显式实验变量：例如不同医院采样协议、事件触发测量规则、监测频率变化或主动采样策略改变。因而它能说明 LLM 方法当前在 irregular ICU classification 上的性价比问题，但还不能直接回答 LLM 表示是否更能抵抗采样政策偏移。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：不要把“引入 LLM/语义对齐”误认为自动解决采样偏移。若前端 encoder 没有正确处理不规则观测，LLM alignment 可能只是把采样 shortcut 映射到语言或嵌入空间中。我们的实验设计也应像该论文一样做组件级消融：分别比较 irregularity-aware encoder、policy-aware encoder、LLM alignment、语义摘要和最终分类头对跨策略泛化的贡献。

纵向深化上，可以把它的测试床扩展成 policy-shift benchmark：在同一 ICU 数据上构造多种反事实采样策略，比较传统 irregular models、LLM-reprogramming models、summary-based models 和 policy-invariant models 的 in-policy / cross-policy / counterfactual-policy 表现。尤其值得加入“encoder-first”的诊断：如果只替换采样策略就导致 LLM 输入 token 或语义锚点大幅变化，说明模型仍在依赖观测流程；若 policy-aware encoder 能稳定生成状态表征，再交给 LLM 做语义推理，才可能真正获得采样策略偏移下的稳健性。
