# Paper Daily - 2026-09-28

## 检索与去重记录

- 已强制读取根目录下全部已发现的日期版 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`、`paper_daily_2026-09-24.md`、`paper_daily_2026-09-25.md`、`paper_daily_2026-09-26.md`、`paper_daily_2026-09-27.md`；同时用标题级检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充黑名单。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`、`Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models`、`MedFuse: Multiplicative Embedding Fusion for Irregular Clinical Time Series`、`ReDiTT: Retrieval Augmented Conditional Diffusion Transformers for Asynchronous Time Series`、`DynaMamba: Multi-scale Dynamic Interacting Mamba Network for Irregular Clinical Time Series Classification`、`Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring`、`Temporally Detailed Hypergraph Neural ODE for Disease Progression Modeling` 与 `ReTAMamba: Reliability-Aware Temporal Aggregation with Mamba for Irregular Clinical Time Series Prediction` 等全部已总结论文。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / longitudinal multimodal patient journey / irregular clinical time series classification / zero-shot clinical outcome classification / sampling-policy shift 的 ICLR、ICML、NeurIPS、AAAI、OpenReview、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 `DBGL`、`STAR-Set`、`MedFuse`、`GSNF`、`QuITE`、`MILM`、`TCF`、`ReTAMamba` 等重复命中，排除 `ALSET`（通用 time-series open-set recognition，非原生不规则采样主线）、`DynSHAP`（动态生存分析解释，非分类主任务）和 `K-STAMM`（期刊论文且更偏肺炎多模态诊断）等候选。严格的“近月顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 1 篇未在黑名单中的高相关新工作：`NOAH` 是 2026-09 arXiv 预印本，尚未确认顶会录用，但它直接处理 longitudinal multimodal patient records 的不规则时间动态，并支持 zero-shot classification、time-to-event prediction 与 counterfactual intervention simulation，对 Sampling-Policy Shift 下的“病人状态轨迹 vs 观测/干预流程轨迹”分解有新增价值。

## 1. NOAH: Learning the Full Patient Journey. A Longitudinal Multimodal Time-Aware Model for Representation and Forecasting

- 会议/状态：arXiv 2026-09 预印本，尚未确认顶会录用
- 作者：Tobias Susetzky, Raphael Rehms, Dmitrii Seletkov, Ozgun Turgut, Michelle Espranita Liman, Lisa Steinhelfer, Rickmer Braren, Daniel Rueckert
- 论文：https://arxiv.org/abs/2609.09140
- 关键词：longitudinal multimodal patient journey, irregular temporal dynamics, time-aware generative transformer, variational latent space, zero-shot classification, counterfactual intervention simulation

### 场景、任务与核心难点

NOAH 面向“完整病人旅程”的纵向多模态建模。输入不是单一 ICU 表格或固定窗口时序，而是跨 MIMIC 数据族整合的 559M timestamped clinical events，覆盖 299K patients、431K hospital stays，以及医学影像、时间序列/数值信号、类别事件、结构化与非结构化临床记录等多种模态。论文目标也不只是训练一个特定二分类器，而是学习一个 time-aware、task-agnostic、generative patient trajectory model，用于自回归预测、可控未来时间条件下的 forecast、zero-shot classification、time-to-event prediction 和临床干预反事实模拟。

核心难点在于，真实医疗轨迹既多模态又强不规则：生命体征、化验、用药、诊断、影像、报告和 ECG 等事件不会以固定频率出现；同一历史事件对不同预测 horizon 的有效性也不同。例如某个慢病诊断可能对数月后的并发症仍有意义，但对几天内的急性事件帮助有限；相反，近期化验异常可能对短期风险很重要，却不能简单用单调 recency bias 表示。NOAH 因此提出 bidirectional time integration，把前向/后向时间间隔和预测 horizon 注入 attention，使模型能根据“距离上一次事件多久、距离下一次事件多久、要预测多远的未来”来调整历史证据权重；同时用 context-conditional variational latent space 建模 patient state 的随机演化，避免把未来轨迹压成单一确定路径。

### 审稿人视角：价值与不足

最有价值的思想是把不规则 EHR/医疗时序从“下游分类输入”提升为“可生成、可查询、可反事实模拟的病人状态轨迹”。许多既有 EHR foundation model 仍偏 next-token、masked reconstruction 或单任务监督；NOAH 的贡献在于同时强调时间不规则性、预测 horizon 条件化和未来轨迹随机性。作为审稿人，我认为它的核心价值不是某个单项分类指标，而是架构层面的统一性：同一个 time-aware generative transformer 能消费多模态真实内容，并通过 latent patient state 支持 probed outcomes、ICD chapter / comorbidity classification、time-to-event prediction 和 intervention simulation。这为非规则采样分类提供了一个更上游的表示学习视角：分类器可以读取“病人状态轨迹”，而不是直接读取混杂的观测事件流。

不足也很明显。首先，NOAH 目前是大规模 arXiv 预印本，尚不能按已录用顶会正会结论看待；其规模化训练、数据整合和评估协议也需要独立复现。其次，生成式 forecast 的目标天然会同时学习 patient state 与 care process：未来会出现哪些化验、影像、诊断和干预，既由真实病程决定，也由医院流程、医生下单、资源可及性、检查返回延迟和编码习惯决定。若不显式分解，NOAH 的 patient state latent 可能把“这个医院如何观察和干预病人”也编码为状态本身。第三，虽然 zero-shot classification 和 counterfactual intervention simulation 很有吸引力，但反事实干预是否因果可信，取决于训练数据中 policy confounding 是否被处理；单纯能生成合理事件序列不等于能可靠回答“换一种采样/干预政策后风险是否改变”。

### 对 Sampling-Policy Shift 的启发

NOAH 对 Sampling-Policy Shift 的横向启发是：采样政策偏移可以被放到完整 patient journey 生成框架中建模，而不只是下游分类器的鲁棒性问题。在 NOAH 这类模型里，“未来会被观察到什么事件、何时观察、以何种模态出现”本身就是生成目标的一部分；这意味着 sampling policy 不再是外部噪声，而是 patient journey 的一个显式维度。我们可以借鉴其 bidirectional time integration，但把时间信息拆成 state time 与 policy time：前者描述病程相位和生理证据有效期，后者描述医院何时记录、测量、复查、转运或干预。

纵向深化上，可以设计 policy-factorized patient-journey model。state latent 负责生成跨采样政策稳定的潜在病程、风险状态和分类 logits；policy latent 负责生成观测事件时间、变量/模态可见性、value-pending、检查触发和干预记录。训练时对同一潜在病程构造多种 observation / intervention policy 视图，要求 state latent、zero-shot classification score 和 time-to-event risk 保持一致，同时允许 policy latent 预测不同医院/科室/设备下的事件密度与模态组合。评估上可以报告 state-logit counterfactual consistency、policy-only classification leakage、cross-policy forecast divergence，以及替换采样政策后 NOAH rollout 中风险变化是否主要来自病程事件而非观测流程事件。这样能把 NOAH 的“完整病人旅程建模”推进到“显式区分病程轨迹与采样政策轨迹”的非规则时序分类框架。
