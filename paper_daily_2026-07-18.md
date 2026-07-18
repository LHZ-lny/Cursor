# Paper Daily - 2026-07-18

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / sparse medical time series classification 的顶会或顶会 workshop 论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、NeurIPS 2025、KDD 2025/2026、WWW/ACL 相关页面、OpenReview、ACM DOI、PMLR 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除偏 forecasting、规则时序分类、时间窗口偏旧或已被历史日报明确排除的 ISTS-PLM/HCIB/LAST SToP 等候选。严格的主会 direct hits 基本已被历史日报覆盖，因此本次保留 2 篇未在黑名单中的顶会 workshop/相关论文：STAR-Set 直接面向异步临床时序分类，并在摘要中显式讨论 sampling-policy shortcuts；RAxSS 面向稀疏、变长医疗时序分类，虽不是典型变量级 IMTS，但其“采样窗口选择 + 检索式聚合”的机制对采样策略偏移具有直接启发。

## 1. Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series

- 简称：STAR-Set / STAR Set Transformer
- 会议/状态：ICLR 2026 TSALM Workshop Poster
- 作者：Joohyung Lee, Kwanhyung Lee, Changhun Kim, Eunho Yang
- OpenReview：https://openreview.net/forum?id=AxXNor3Kd2
- 论文：https://arxiv.org/html/2603.06605
- 关键词：asynchronous clinical time series, EHR, point-set tokenization, temporal attention bias, variable-type attention bias, ICU prediction, sampling-policy shortcut

### 场景、任务与核心难点

这篇工作面向 ICU/EHR 中的异步多变量临床时序预测与分类任务，实验覆盖 CPR、死亡风险和升压药使用等 ICU outcome prediction。输入不是规则网格，而是一组事件 token：每个观测由时间戳、变量类型和值构成。此类 EHR 数据的核心难点在于，不同变量被测量的时间完全不同；如果重采样到规则网格，需要插补或 mask，容易引入错误并放大采样政策捷径；如果直接用 point-set tokenization，则避免了插补，但又丢掉了规则网格天然暴露的两条结构轴：同一变量的时间轨迹和不同变量在相近时间内的上下文关系。

STAR-Set 的方法很克制：保留 set encoder 的灵活性，不把事件硬塞进规则网格，而是在 Transformer attention logits 中加入两个轻量 bias。第一个是 temporal locality penalty，用可学习时间尺度惩罚时间距离远的 token 交互；第二个是 variable-type affinity bias，用可学习变量兼容矩阵恢复同变量或相关变量之间的结构先验。作者系统比较 10 种不同层级注入策略，最终显示同时使用 temporal 与 variable-type bias 的 STAR-Set 在三个 ICU 任务上优于 regular-grid、event-time grid 和已有 set baseline。

### 审稿人视角：价值与不足

最有价值的思想是，它没有在“规则网格 vs. 事件集合”之间二选一，而是把二者的优点拆开重组：point-set 负责避免插补和离散化，attention bias 负责补回网格中原本显式存在的结构归纳偏置。这对异步临床时序很实用，因为许多 foundation-style event tokenizer 正在转向“所有观测都是 token”的输入形式，但如果没有时间局部性和变量类型结构，模型必须从有限数据中重新学到医疗变量关系，样本效率和可解释性都会受影响。论文还把 sampling-policy shortcuts 写进问题设定，说明作者意识到 mask/缺失不是中性信息。

不足在于，STAR-Set 仍然主要是在同一数据分布下证明更好的 ICU 任务表现，而不是专门评估跨医院、跨监测协议或跨采样触发规则的稳健性。temporal bias 和 variable-type bias 能减少插补依赖，但它们也可能把训练环境中的采样频率、变量联测习惯和护理流程共现压进 attention 结构。例如某医院只在高危患者中频繁记录某项化验时，学习到的变量亲和矩阵可能反映的是测量协议，而不一定是稳定生理关系。论文的解释对象是 learned timescale 和 variable affinity，但还缺少反事实采样或跨机构切分来验证这些解释是否具有策略不变语义。

### 对 Sampling-Policy Shift 的启发

STAR-Set 对 Sampling-Policy Shift 的横向启发非常直接：策略偏移可以被看成“事件集合结构先验”的偏移。换医院、换监测频率或换告警触发规则时，时间局部性尺度、变量类型亲和矩阵、以及不同层的 attention bias 使用方式都可能变化。因此，除了比较 mask ratio 和 delta-t 分布，我们还可以监控 learned temporal timescale、variable affinity matrix 和 bias-conditioned attention map 是否跨策略稳定。

纵向深化上，可以把 STAR-Set 改造成 policy-aware set transformer：将 temporal/variable bias 分解为 state bias 与 policy bias。state bias 表示跨策略稳定的生理时间尺度和变量耦合，进入分类主路径；policy bias 表示某个环境中的测量流程、联测习惯或护理协议，只用于偏移诊断和不确定性校准。训练时对同一潜在病程施加不同采样策略增强，约束 state-biased attention 与 logits 稳定，同时允许 policy-biased attention 区分采样环境。这样既保留 set tokenization 避免插补的优势，又能防止 attention 结构把采样政策当成类别证据。

## 2. RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification

- 全称：Retrieval-Augmented Sparse Sampling
- 会议/状态：NeurIPS 2025 TS4H Workshop Poster
- 作者：Aydin Javadov, Samir Garibov, Tobias Hoesli, Qiyang Sun, Joseph Ollier, Florian von Wangenheim, Björn Schuller
- OpenReview：https://openreview.net/forum?id=RKVLsB0Ciu
- NeurIPS 页面：https://neurips.cc/virtual/2025/132323
- 预印本：https://arxiv.org/pdf/2510.02936
- 关键词：variable-length medical time series classification, sparse sampling, retrieval-augmented aggregation, iEEG, explainability, cross-center clinical signals

### 场景、任务与核心难点

RAxSS 面向变长、稀疏且噪声较强的医疗时序分类，实验任务是多中心 intracranial EEG 记录中的 seizure onset zone localization。与 ICU 表格型 IMTS 不同，这里每条记录可能是长时段生理信号，长度差异很大，关键模式只出现在少量窗口中，并且跨中心采集流程、信号质量和个体差异都可能影响局部片段的可用性。固定长度模型需要截断或 padding，直接全序列建模又成本高且解释性弱。

论文建立在 Stochastic Sparse Sampling (SSS) 之上：先从长序列中按长度比例抽取固定窗口，用 backbone 得到窗口级预测，再聚合成序列级分类。RAxSS 的核心改动是用检索式聚合替代均匀平均。对每个窗口，它在同一记录/通道内检索最相似的非同一邻居窗口，计算平均支持度，再通过 softmax 得到窗口权重，最终在概率空间做凸组合。这样，模型不仅知道“哪个窗口给出高分”，还知道“这个窗口为什么被信任”：每个高贡献窗口都有一个相似邻居排行榜作为 evidence trail。

### 审稿人视角：价值与不足

最有价值的思想是把采样、检索和聚合统一到分类决策中。许多稀疏窗口方法默认随机抽到的片段同等重要，或者只用局部 score heatmap 解释“哪里重要”；RAxSS 进一步用 within-recording retrieval 给每个窗口的贡献提供支持证据，使聚合权重由片段之间的一致性决定。对于罕见、短暂、弱相关的医学事件，这种 similarity-weighted aggregation 比简单平均更符合临床直觉：孤立噪声窗口应被降权，能在同一记录中找到相似支持的模式应被提升。

不足是，RAxSS 当前主要解决变长与稀疏窗口选择问题，还没有充分处理变量级异步采样或真实临床政策导致的 missingness。检索严格限制在同一记录/通道内，降低了隐私和跨中心依赖，但也限制了对跨患者、跨医院可迁移模式的学习。更重要的是，相似窗口的出现频率本身可能受采样策略影响：某中心的记录更长、某类患者被更密集采集、某些事件前后更常保留高质量窗口，都会改变检索支持度和聚合权重。论文报告了多中心 iEEG 上的竞争性性能和解释链，但尚未用采样策略反事实或中心外测试来验证 retrieval evidence 是否稳定。

### 对 Sampling-Policy Shift 的启发

RAxSS 对 Sampling-Policy Shift 的横向启发是：采样策略偏移不只改变“观测了哪些点”，还改变“哪些片段有机会互相支持”。在窗口化时序中，policy shift 会表现为窗口长度分布、关键片段被抽中的概率、同类邻居密度、相似度支持度和最终聚合权重的联合偏移。因此，检索支持分布、top-m neighbor composition、window influence entropy 可以成为采样策略偏移诊断指标。

纵向深化上，可以把 RAxSS 扩展为 policy-aware retrieval aggregation：检索库或邻居集合分成 state neighbors 与 policy neighbors。state neighbors 支持跨策略稳定的动态模式，决定分类主路径；policy neighbors 解释某个采样策略为何让这些窗口更常出现或更容易被信任，只进入校准/诊断分支。对同一底层序列生成不同窗口抽样策略时，可以约束 state-neighbor 支持度和序列级 logits 稳定，同时允许 policy-neighbor 分布变化。这样可以把“可解释稀疏采样分类”推进为“可解释且采样政策稳健的稀疏时序分类”。
