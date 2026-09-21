# Paper Daily - 2026-09-19

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`；同时用标题检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors`、`OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data`、`DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification`、`Continuum Dropout for Neural Differential Equations`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings`、`A novel approach to classification of ECG arrhythmia types with latent ODEs`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`，以及自动化记忆中记录的 `LERD`、`WIPSNet`、`Context-Aware Neural SDEs`、`RAxSS` 等已总结工作。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / clinical time-series foundation model audit / sampling-policy shift 的顶会、顶会 workshop、OpenReview、ICLR/ICML/AAAI/NeurIPS 官方页面、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 `ReTAMamba` 这类暂无顶会来源且历史日报已排除的 arXiv 候选、`PhASER` 这类分类泛化强但历史日报已判断为不直接处理 irregular sampling 的候选、`Interpretable Graph Learning on Irregular Clinical Time Series` / `GMAN` 这类与已总结 `SuperMAN` 高度重叠的候选，以及 `Benchmarking LLM Summaries of Multimodal Clinical Time Series for Remote Monitoring` 这类主要任务是摘要评测而非分类/风险预测的工作。严格的“近月顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 1 篇未在黑名单中的 ICLR 2026 TSALM Workshop Poster：它不提出新的同分布分类器，但直接审计临床时序基础模型在反事实风险推断中的混杂与政策捷径，对 Sampling-Policy Shift 的方法论增量明确。

## 1. Auditing Black-Box Trends: Structural Inductive Bias Facilitates Causal Interpretability in Clinical Time Series

- 会议/状态：ICLR 2026 Workshop on Time Series in the Age of Large Models (TSALM) Poster
- 作者：Aditya Kumar Karna, Trina Dutta Barlow
- OpenReview：https://openreview.net/forum?id=lBLE2GWcS0
- 论文 PDF：https://openreview.net/pdf?id=lBLE2GWcS0
- 关键词：clinical time series, foundation model audit, causal interpretability, confounding by indication, Causal Hallucination Score, MIMIC-IV ICU, propensity-regularized GRU-D

### 场景、任务与核心难点

这篇工作面向高风险 ICU 临床时序中的基础模型审计。它关注的不是再训练一个 P12/P19 上的普通 IMTS 分类器，而是一个更贴近部署的任务：当 Lag-Llama、Chronos-T5 等黑盒时间序列基础模型被用于临床“what-if”推断或风险解释时，它们是否真的理解干预语义，还是只复现观测数据中的混杂关联。论文使用 MIMIC-IV ICU 入院数据，围绕 vasopressor 这类强治疗指示变量，评估模型在反事实预测中的因果方向是否可信。

核心难点是 observational clinical time series 中的 confounding by indication。危重患者更可能接受升压药，所以朴素预测模型容易学到“使用升压药与死亡风险升高相关”；但这并不意味着升压药本身有害。对于临床时序基础模型，这个问题尤其危险：模型可能在常规 forecasting likelihood 上表现合理，却在反事实提示下把救命干预解释为风险原因。作者将这种预测准确性与干预有效性之间的断裂称为 Alignment Gap，并提出 Causal Hallucination Score (CHS)，用基础模型的零样本反事实输出与结构化参考工具之间的偏差来量化这一 gap。论文还构建 propensity-regularized GRU-D 作为审计仪器，结合 doubly robust estimation 和 placebo falsification 检查方向一致性。

### 审稿人视角：价值与不足

最有价值的思想是把临床时序基础模型的风险从“预测不准”推进到“预测看似合理但因果语义反了”。这对审稿人很重要，因为很多时间序列大模型评估仍停留在 likelihood、AUROC 或 summarization fluency；而这篇工作明确要求模型在反事实语境下不要把观测流程、治疗选择和疾病严重程度的混杂关系当成干预效应。CHS 的价值在于给出一个可操作的审计指标：即便模型是黑盒，只要能生成 counterfactual trajectory / risk response，就可以检查其是否复现了观测偏差。Propensity-regularized GRU-D 作为结构化参考模型也很合适，因为 GRU-D 原本就面向缺失、不规则临床时序，加入 propensity 约束后更适合作为临床时间序列中的混杂审计基线。

不足在于，它不是一个完整的新型 irregular time-series classifier，也没有直接给出跨医院 sampling-policy shift benchmark。审计结论依赖参考工具和因果假设：propensity model、DR estimator、placebo falsification 的质量会影响 CHS 的解释。论文聚焦 vasopressor 这类治疗变量，尚不能自动推广到所有观测变量、化验下单、记录频率或远程监测采样策略。另一个限制是，CATE 信号本身可能很小，审计指标更适合发现严重方向性错误，而未必能精细校准每个患者层面的干预效应。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：采样政策偏移与治疗政策混杂在机制上非常相似。非规则采样中的“某项检查被测量”“某段时间观测密度升高”“某变量长期未测”，就像治疗数据中的“某药物被使用”一样，往往由潜在病情和环境政策共同决定。若模型只在观测分布上优化分类，它可能学到“被密集观测 => 高风险”这类相关性，却无法判断这是病情恶化触发的稳定信号，还是医院流程、资源约束或下单习惯造成的 policy shortcut。

纵向深化上，可以借鉴 CHS 设计 Sampling-Policy Hallucination Score：对同一潜在患者状态构造多种反事实 observation policy，例如固定节律采样、告警后密集采样、低资源变量选择或医院特定联测规则；再检查分类器的 risk response 是否把纯采样政策改变误解释为病情改变。结构化参考工具可以是 policy-regularized GRU-D / CDE / Set Transformer：一方面建模缺失和时间间隔，另一方面通过 propensity 或 environment regularization 剥离观测政策。最终目标不是禁止模型使用 informative sampling，而是要求它把 state-driven informative sampling 与 policy-driven shortcut 分开：前者可进入分类证据，后者进入偏移告警、不确定性和拒识机制。
