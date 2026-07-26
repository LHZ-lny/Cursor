# Paper Daily - 2026-07-26

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / ICU event-stream prediction / EHR time-series representation 的顶会或顶会 workshop 论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、KDD 2025/2026、OpenReview、ICLR virtual site 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除 MedFuse 这类先前已标记为 withdrawn 的候选、ReTAMamba 这类暂无顶会来源的近期 arXiv 预印本，以及偏 forecasting/generation/imputation、普通规则时序或与异步分类关系较弱的条目。本次保留全新工作 2 篇：EHR-SPC 是 ICLR 2026 TSALM Workshop Poster，直接面向不规则 EHR event streams 的下游 ICU 预测表征；LLM4EHR 是 ICLR 2026 OpenReview submission / 2026-07 arXiv 新预印本，尚未确认正会录用，但其临床事件序列与时序对齐思想对非规则采样分类和 sampling-policy shift 有较强跟踪价值。

## 1. Status-Aware Self-Supervised Forecasting for Irregular Clinical Time Series

- 方法名：EHR-SPC / EHR Set Predictive Coding
- 会议：ICLR 2026 TSALM Workshop Poster
- 作者：Kwanhyung Lee, Joohyung Lee, Jong-Heon Kim, Sangchul Hahn, Eunho Yang
- 官方页：https://iclr.cc/virtual/2026/10013833
- OpenReview：https://openreview.net/forum?id=lx98lmQ14i
- 关键词：irregular EHR event streams, self-supervised learning, ICU prediction, future status forecasting, query-based set prediction, masked event modeling

### 场景、任务与核心难点

这篇工作面向 ICU/EHR 中天然不规则的临床事件流：每条患者轨迹由观测值、时间戳和变量类型三元组组成，不同化验、生命体征、干预记录在不同时间异步出现。下游任务是面向未来临床状态的预测，例如 CPR、死亡风险或其他 ICU outcome；训练难点来自标签稀缺、类别不平衡，以及传统 SSL 方法通常先把事件流离散化到固定网格再做预测，导致原始事件集合结构、真实时间间隔和变量级异步性被抹平。

EHR-SPC 的核心做法是直接在 irregular event sets 上做自监督预训练。模型先把短事件窗口聚合为 status token，再用过去 status context 去预测未来 status representations；未来事件集合大小和组成是不固定的，因此作者借鉴 DETR 风格的 learnable queries，用 query-based Transformer decoder 生成未来 status tokens，并用 EMA momentum status encoder 提供稳定目标。同时，EHR-MAE 辅助目标对过去事件进行遮蔽重构，增强局部观测鲁棒性。整体目标不是补齐一个规则时间网格，而是学习一个与“未来临床状态预测”对齐的事件级表征。

### 审稿人视角：价值与不足

最有价值的技术思想是把不规则 EHR 预训练目标从“恢复被遮蔽的局部值”推进到“预测未来状态表征”。这更贴近临床分类/预后任务的实际需求：医生关心的是患者未来风险状态，而不仅是某个缺失化验值。query-based set prediction 也很适合 EHR，因为未来窗口中会出现哪些变量、出现多少事件本身不固定，强行做逐 token 对齐会引入虚假的顺序和数量假设。

不足在于，这仍是一篇 workshop 短文，实验和偏移分析深度有限。更重要的是，未来 status target 本身可能混合真实病程和医院采样/记录政策：如果某些未来事件集合是由告警、医生怀疑或科室流程触发的，那么模型预测到的“future status”可能部分是 protocol status，而不完全是 patient state。论文展示了对 ICU prediction 的下游收益，但尚未系统评估跨医院、跨记录制度或反事实采样策略下 status token 的语义稳定性。

### 对 Sampling-Policy Shift 的启发

EHR-SPC 对我们的横向启发是：采样策略偏移可以被提升到“未来状态预测目标是否稳定”的层面来分析。若同一潜在病程在不同采样政策下产生不同的 future event set，那么普通 SSL 会鼓励模型学习策略特定的未来观测集合；这解释了为什么一些 EHR 表征在同院同策略有效、跨院后退化。

纵向深化上，可以把 EHR-SPC 改造成 state-policy 双状态预测框架：state status token 预测跨采样策略稳定的病程状态，policy status token 预测未来会被记录哪些变量、何时记录、记录密度如何。训练时对同一患者轨迹构造多种反事实采样视图，约束 state status 和分类 logits 保持一致，同时允许 policy status 区分医院流程、告警触发和资源约束。这样能把“未来状态自监督”从单纯提升表征质量推进到显式约束采样政策不变性。

## 2. LLM4EHR: Aligning Clinical Time Series with Medical Event Sequences via Large Language Models

- 会议/状态：ICLR 2026 OpenReview submission；arXiv 2026-07 预印本（尚未确认正会录用）
- 作者：Jingteng Li, Alexander Capstick, Louise Rigny, Iona Biggart, Neil J. Sebire, Payam Barnaghi
- OpenReview：https://openreview.net/forum?id=pym3JRajmW
- arXiv：https://arxiv.org/html/2607.15447v1
- 关键词：clinical time series, EHR event sequences, LLM alignment, contrastive learning, ICU outcome prediction, transferable embeddings

### 场景、任务与核心难点

LLM4EHR 面向 ICU 临床预测中的多模态 EHR 表征学习。输入不只包括连续或半连续的 clinical time series，也包括药物、医嘱、操作、诊断等 itemized medical event sequences；这些事件天然带时间戳且不规则出现，和生命体征/监护时序之间存在共享但并不显式对齐的时间结构。下游任务包括 mortality、phenotyping、decompensation 和 remaining length-of-stay 等临床预测/分类任务，并强调少样本和跨队列适配。

核心难点在于，已有临床基础模型常把 EHR 事件和 TS 观测分开建模，或者只做粗粒度拼接，未充分利用二者在时间上的互补关系。LLM4EHR 使用 domain-adapted LLM 编码 EHR event sequence，用 Transformer TS encoder 编码临床时序，再通过 time- and semantic-aware regularized contrastive objective 对齐两种表示。这样得到的 TS embedding 不只是从数值轨迹中学习，还被事件语义和临床流程上下文条件化，从而提升多个下游任务和 k-shot 跨队列适配表现。

### 审稿人视角：价值与不足

最有价值的思想是把不规则临床事件序列当作时序表征的语义锚点，而不是只把它们作为额外协变量。对审稿人来说，contrastive alignment 的价值在于它给出了一个比较清晰的中间目标：临床 TS 表征应与同一患者同一时间上下文中的医学事件语义一致。相比直接把所有 EHR token 丢进 LLM，LLM4EHR 保留了专门的 TS encoder，同时利用 LLM 对事件文本、代码和临床语义的表达能力，具有更好的模块化和迁移潜力。

不足也同样明显。首先，当前状态是 submission / 新预印本，尚不能按已录用正会论文看待。其次，医学事件序列本身高度受医院政策影响：哪些药物被开、哪些检查被下单、何时记录护理事件，往往反映病情和流程的混合结果。如果 contrastive objective 不区分 patient-state semantics 与 hospital-policy semantics，TS embedding 可能被迫对齐到策略特定事件模式，跨医院时反而带来新的 shortcut。论文强调 transferable embeddings，但仍需要更强的 cross-policy、cross-site 和 counterfactual sampling 评估来证明对采样政策偏移的稳健性。

### 对 Sampling-Policy Shift 的启发

LLM4EHR 对 Sampling-Policy Shift 的横向启发是：事件序列可以作为采样政策的显式观测窗口。EHR 中的检查、用药、转科、护理记录不仅描述患者状态，也描述医院如何观察和干预患者；它们可以帮助解释为什么某些时段被密集采样、为什么某些变量被联测、为什么某些值尚未返回。因此，处理非规则采样分类时，不应只把 event sequence 当作增强分类性能的语义特征，还应把它作为 policy context 来建模。

纵向深化上，可以把 LLM4EHR 的对齐目标改造成三分支：state TS encoder 学跨策略稳定的病程表征；policy/event encoder 学采样、医嘱和记录流程；semantic alignment 只约束 state branch 与跨环境稳定的临床语义对齐，而通过对抗或条件不变性限制分类头直接利用 policy branch。对同一病程生成不同采样/记录策略时，要求 state embedding 与 logits 一致，同时允许 policy embedding 改变。这样能把 LLM 语义对齐从“增强 EHR 表征”推进到“识别并隔离采样政策语义”。
