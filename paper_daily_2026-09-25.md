# Paper Daily - 2026-09-25

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`、`paper_daily_2026-09-24.md`；同时用标题检索补读兼容入口 `paper_daily.md`，并参考自动化记忆中的 2026-08 至 2026-09 补充黑名单。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`Status-Aware Self-Supervised Forecasting for Irregular Clinical Time Series`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`、`Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models`、`MedFuse: Multiplicative Embedding Fusion for Irregular Clinical Time Series`、`ReDiTT: Retrieval Augmented Conditional Diffusion Transformers for Asynchronous Time Series` 与 `DynaMamba: Multi-scale Dynamic Interacting Mamba Network for Irregular Clinical Time Series Classification` 等全部已总结论文。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous clinical time series / online time-series monitoring / ICU risk classification / sampling-policy shift / explanation of prediction changes 的 ICLR、ICML、NeurIPS、ACL/health workshop、OpenReview、arXiv 与论文代码页。
- 已排除全部黑名单论文；同时排除 `STAR-Set`、`SLAN`、`GSNF`、`ATENet`、`TCF`、`MILM`、`ORA` 等重复命中，排除 `Hi-Patch` 这类 ICLR 2025 under-review/时间窗口偏旧候选，排除 `HT-Transformer` 这类 2025-08 预印本且并非近 3-7 个月顶会论文的事件序列分类工作；`ReTAMamba` 虽是 2026-05/06 高相关不规则临床时序 Mamba 预印本，但历史日报已因暂无顶会来源将其作为候选排除，本次继续不作为主推荐。严格的“近月顶会正会 + 直接 IMTS 分类”新增命中仍很稀缺，因此本次保留 1 篇未在黑名单中的新跟踪对象：`Delta-XAI` 是 ICLR 2026 正会 Poster，核心任务是解释在线时序分类器的预测变化，实验覆盖 MIMIC-III decompensation 与 PhysioNet 2019 sepsis 等临床风险分类/监测任务；它不是新的 IMTS encoder，但明确讨论真实时序中的 irregular sampling 与 imputation attribution failure，对 Sampling-Policy Shift 下“模型为何因新观测或补值而改变风险判断”的审计有直接横向价值。

## 1. Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring

- 会议：ICLR 2026 Poster
- 作者：Changhun Kim, Yechan Mun, Hyeongwon Jang, Eunseo Lee, Sangchul Hahn, Eunho Yang
- OpenReview：https://openreview.net/forum?id=ZHW5pp5nE5
- 论文：https://arxiv.org/abs/2511.23036
- 代码：https://github.com/AITRICS/Delta-XAI
- 关键词：online time-series monitoring, clinical risk classification, prediction-change explanation, irregular sampling, imputation attribution, SWING, Sampling-Policy Shift

### 场景、任务与核心难点

Delta-XAI 面向在线时序监测中的分类器解释问题，尤其是医疗场景中风险分数随时间更新的监测任务。论文把模型形式化为在线 time-series classifier：每个时间步接收一个 lookback window，输出二分类或多分类概率。实验覆盖 MIMIC-III decompensation 与 PhysioNet 2019 sepsis monitoring 等临床风险分类/预警任务，并比较 LSTM、Transformer 等不同 backbone 上的解释质量。

它解决的核心难点不是如何设计新的不规则采样 encoder，而是解释“为什么风险从上一个时间点变成现在这样”。传统时序 XAI 多解释某个单独预测，例如当前 sepsis risk 为 80% 时哪些变量重要；但临床决策更常问的是风险为什么从 10% 升到 50%，或从 90% 降到 50%。如果直接比较两个时间点的单点 attribution，解释会因为非线性模型、滑动窗口重叠、延迟效应和 imputation artifact 而失真。论文特别指出，真实世界时序常有 irregular sampling 和 forward filling：当最新窗口中的值全是补值时，普通解释方法可能错误地把补值时间点当作高贡献，而不是追溯到真正触发风险变化的实际观测。

Delta-XAI 因此提出 PredictionDifferenceWrapper，把解释目标从单点预测 `f(x_T)` 改为预测差值 `f(x_T2)-f(x_T1)`；再提出 SWING (Shifted Window Integrated Gradients)，用历史窗口作为积分路径的一部分，避免从零基线到当前窗口的 OOD 路径，并把新增观测、旧观测移出窗口和中间观测的延迟效应分解出来。这样解释对象从“当前风险由什么决定”变成“风险变化由哪些时间-变量证据驱动”。

### 审稿人视角：价值与不足

最有价值的思想是把在线时序解释的基本对象从 static attribution 改成 delta attribution。对审稿人而言，这一点很重要：在 ICU/EHR 监测中，风险变化本身比绝对风险更贴近医生操作，例如是否需要升级监护、追加化验或解除警报。PredictionDifferenceWrapper 让 14 类既有 XAI 方法可以被统一改造成 prediction-change explainer；SWING 又给出更符合时序语义的 Integrated Gradients 路径，并提出 faithfulness、sufficiency、completeness、coherence、efficiency 等在线解释评估维度。相比单纯报告一个新的 saliency map，它更像是给在线风险模型提供了一套审计协议。

不足在于，Delta-XAI 仍是解释框架而不是采样策略鲁棒分类器。它默认已有分类模型和窗口化输入，主要评价 attribution 是否忠实解释模型行为；如果底层模型已经把医院采样政策、补值模式或变量联测当成风险证据，Delta-XAI 会揭示这种依赖，但不会自动消除它。另一个限制是，论文实验主要在窗口化临床监测 benchmark 上进行；对原生事件流、变量级异步 token、value-pending 和跨医院 observation policy shift 的解释稳定性，还需要进一步验证。

### 对 Sampling-Policy Shift 的启发

Delta-XAI 对 Sampling-Policy Shift 的横向启发是：策略偏移不仅要看分类性能下降，还要看“风险变化的归因来源”是否从 patient-state evidence 漂移到 observation-policy evidence。若跨医院后模型风险上升主要归因于某变量刚被测、某个缺失被 forward-filled、某类窗口刚进入/移出，而不是归因于稳定的生理数值变化，就说明分类器可能依赖采样政策 shortcut。Delta-XAI/SWING 可以作为 policy leakage audit：比较同一患者状态在不同反事实采样策略下，delta attribution 是否仍指向相同的 state features。

纵向深化上，可以把 Delta-XAI 接入 policy-aware IMTS 训练闭环。训练阶段对同一潜在病程生成多种 observation policy 视图，要求分类 logits 稳定；验证阶段用 SWING 分析 prediction change，把 attribution 分成 state attribution 与 policy attribution。state attribution 应集中在真实观测值、趋势和跨策略稳定事件上；policy attribution 则包括 delta-t、mask、补值、变量联测和采样密度，只能进入不确定性或偏移告警。进一步可以设计 attribution consistency loss：反事实采样改变时，state-delta attribution 保持一致，policy-delta attribution 可以改变但不得主导分类风险变化。这样能把“解释风险为何变化”推进到“约束风险变化不能主要由采样政策变化驱动”。
