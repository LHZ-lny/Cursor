# Paper Daily - 2026-09-21

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`；同时用标题检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-09-20 的新增标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`Beyond Observations: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`，以及自动化记忆中记录的 `Auditing Black-Box Trends: Structural Inductive Bias Facilitates Causal Interpretability in Clinical Time Series`、`Pretraining EHR Foundation Models with Patient-Aware Sampling` 等近期已总结工作。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / ICU event-stream prediction / sampling-policy shift 的顶会、顶会 workshop、OpenReview、ICLR/ICML/AAAI/NeurIPS/KDD 官方页面、arXiv 与代码页。
- 已排除全部黑名单论文。严格的“近月顶会正会 + 直接 IMTS 分类”新增命中在当前检索中基本已被历史日报覆盖；本次仅保留 1 篇未在黑名单中的高相关新工作：`INTERVenE` 是 arXiv 2026-08 预印本，尚未确认顶会录用，但它直接面向稀疏、不规则 ICU EHR 事件流上的短期并发症预测/风险分类，并把原始点观测转为可解释的知识型时间抽象区间，对 Sampling-Policy Shift 下的“观测流程语义化”有新增价值。

## 1. INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction

- 会议/状态：arXiv 2026-08 预印本，尚未确认顶会录用
- 作者：Shahar Oded, Yuval Shahar
- 论文：https://arxiv.org/abs/2608.29901
- 关键词：irregular ICU EHR, temporal abstraction intervals, knowledge-based temporal abstraction, short-horizon complication prediction, interpretable clinical event prediction

### 场景、任务与核心难点

INTERVenE 面向 ICU 中稀疏、不规则、异构 EHR 事件流的短期临床并发症预测。论文聚焦 MIMIC-IV 中 57,078 次 diabetes-related ICU admissions：模型观察入院前 48 小时的实验室测量、用药、干预和静态上下文，预测 48-336 小时内是否发生 death、kidney complication、hyperglycemia、severe hyperglycemia、hypoglycemia、severe hypoglycemia 等 6 个目标事件，并同时处理 length-of-stay 相关目标。这里的任务形式不是传统单一二分类 benchmark，而是面向多个未来并发症的短期风险分类/时间到事件预测。

核心难点在于，原始 ICU EHR 不是规则时间网格：测量、用药和干预以病人特异的时间到达，既有点事件，也有持续状态和趋势；同一个数值在不同持续时间、上下文和临床阶段下语义不同。直接把原始 triplet 输入 GRU-D、STraTS 或普通 Transformer，虽然能处理时间戳和缺失，但解释通常仍停留在变量或 token 层，很难回答“哪个临床状态、趋势或上下文区间导致风险升高”。INTERVenE 因此先用 Knowledge-Based Temporal Abstraction (KBTA) 把原始点观测转换为有医学命名的区间 token，例如状态、趋势、事件和上下文，再构建两类 Transformer：INTERVenE-Enc 做 single-pass joint risk / time-to-event prediction；INTERVenE-Ar 以自回归方式生成未来 abstraction trajectory，并在每一步读出风险曲线。实验中 INTERVenE-Enc 在 held-out admissions 上达到 support-weighted AUPRC 0.672、AUROC 0.901，相比最强神经基线 AUPRC 提升约 0.041。

### 审稿人视角：价值与不足

最有价值的思想是把“不规则临床时序如何被解释”前移到表示层：先将原始观测转成临床知识可命名的 interval vocabulary，再让 Transformer 在这些区间 token 上学习风险。这样做的贡献不只是提升 AUPRC，而是让归因天然落到 named clinical concepts 上，避免事后解释在 raw value / bin index 层绕一圈。INTERVenE-Ar 的自回归风险曲线也很有意义，因为它不仅输出 admission-level 风险，还尝试说明风险在未来 abstraction trajectory 中何时、随哪些事件上升。

不足也比较明显。首先，它目前是 arXiv 预印本，尚不能按已录用顶会论文看待。其次，KBTA 的优势依赖人工医学知识库和 abstraction rule，迁移到其他病种、其他医院或更弱结构化的 EHR 时成本不低。更关键的是，KBTA interval 本身可能把医院流程和采样政策语义固化进 token：某些 lab state、trend 或 intervention interval 的出现，既可能反映病程，也可能反映下单习惯、监测频率、护理协议或资源约束。论文证明了 interval abstraction 是可解释 substrate，但还没有系统评估跨医院、跨测量政策或反事实采样策略下，这些 named intervals 是否仍代表稳定病理机制。

### 对 Sampling-Policy Shift 的启发

INTERVenE 对 Sampling-Policy Shift 的横向启发是：采样政策偏移不一定只表现为 mask、delta-t 或变量共现图变化，也可能表现为“被抽象出的临床区间词汇”发生偏移。若一个医院更频繁监测血糖，KBTA 可能生成更多 glucose state / trend intervals；若另一个医院只在危重时密集测量，同样的 interval token 就可能携带不同风险语义。因此，interval abstraction 可以作为可解释接口，但必须审计哪些区间是 patient-state interval，哪些区间其实是 observation-policy interval。

纵向深化上，可以把 INTERVenE 改造成 policy-aware temporal-abstraction Transformer：state abstraction branch 只保留跨采样策略稳定的病程状态、趋势和临床上下文，进入风险分类主路径；policy abstraction branch 则解释哪些 interval 是由测量频率、检查触发、护理记录习惯或 value-pending 机制产生，用于校准和偏移告警。训练时可对同一潜在病程生成不同采样策略下的 KBTA interval streams，约束 state intervals、risk logits 和 state rationale 保持稳定，同时允许 policy intervals 和不确定性随观测政策改变。这样能把 INTERVenE 的“可解释区间 token”推进到“区分病程语义与采样政策语义”的非规则时序分类框架。
