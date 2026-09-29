# Paper Daily - 2026-09-29

## 检索与去重记录

- 已强制读取根目录下全部已发现的日期版 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`、`paper_daily_2026-09-24.md`、`paper_daily_2026-09-25.md`、`paper_daily_2026-09-26.md`、`paper_daily_2026-09-27.md`；同时分段读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-09-28 的 `NOAH` 等补充黑名单。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`、`Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models`、`MedFuse: Multiplicative Embedding Fusion for Irregular Clinical Time Series`、`ReDiTT: Retrieval Augmented Conditional Diffusion Transformers for Asynchronous Time Series`、`DynaMamba: Multi-scale Dynamic Interacting Mamba Network for Irregular Clinical Time Series Classification`、`Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring`、`Temporally Detailed Hypergraph Neural ODE for Disease Progression Modeling`、`ReTAMamba: Reliability-Aware Temporal Aggregation with Mamba for Irregular Clinical Time Series Prediction` 与 `NOAH: Learning the Full Patient Journey. A Longitudinal Multimodal Time-Aware Model for Representation and Forecasting` 等全部已总结论文。
- 检索范围：近月围绕 irregular sampled / asynchronous / missingness-aware time-series classification / clinical time-series prediction / sampling-policy shift 的 CIKM、ICLR、ICML、NeurIPS、EACL/ACL health workshop、OpenReview、arXiv、ACM 与 ACL Anthology 页面。
- 已排除全部黑名单论文；同时排除 `STAR-Set`、`SLAN`、`ATENet`、`ReTAMamba`、`NOAH`、`TRIAGE`、`Rethinking LLMs`、`MedFuse`、`MuSiCNet` 等重复、venue 不确定或已覆盖条目。本次保留 2 篇未在黑名单中的新跟踪对象：`TANDEM` 是 CIKM 2025 正会论文，直接面向缺失/不规则时序分类，并用 NDE + attention/gating 融合原始观测、控制路径和连续潜动态；`LLM Plug-ins Are Not a Free Lunch for Clinical Time-Series Prediction` 是 2026 HeaLing/EACL workshop 论文，虽不是原生 IMTS 架构，但针对 MIMIC-III ICU 时序预测系统性检验 frozen LLM plug-in 的边界，对我们判断“LLM 先验能否修复采样策略偏移”有新增警示价值。

## 1. TANDEM: Temporal Attention-guided Neural Differential Equations for Missingness in Time Series Classification

- 会议：CIKM 2025, Proceedings of the 34th ACM International Conference on Information and Knowledge Management
- 作者：YongKyung Oh, Dong-Young Lim, Sungil Kim, Alex A. T. Bui
- 论文：https://arxiv.org/abs/2508.17519
- CIKM accepted list：https://cikm2025.org/program/accepted-papers
- DOI：10.1145/3746252.3760996
- 关键词：time-series classification, missing data, irregular sampling, Neural ODE/CDE/SDE, temporal attention, Gumbel-Sigmoid gating

### 场景、任务与核心难点

TANDEM 面向带缺失值和不规则观测的时序分类。论文的问题设定覆盖 30 个 UCR/UEA benchmark 以及一个真实医疗数据集：每条样本是多维时间序列，某些时间点或变量维度缺失，模型需要输出序列级类别。这里的核心难点不是简单补齐缺失值，而是在不同缺失强度下判断哪类信息更可信：原始观测虽然最接近真实证据，但很稀疏；插值 control path 能给连续时间模型提供完整轨迹，却可能引入平滑假设；NDE latent dynamics 能建模连续演化，但也可能在观测不足时过度依赖隐式先验。

论文因此提出一个三路融合框架。第一路保留 raw incomplete observations，让模型直接看到真实观测和 missingness pattern；第二路构造 interpolated control path，服务于 Neural CDE 这类连续时间模型；第三路用 Neural ODE/CDE/SDE backbone 生成 continuous latent dynamics。随后，TANDEM 对每一路分别做 feature-wise multi-head attention，再用 Gumbel-Sigmoid stream-wise gates 自适应选择 raw / control-path / latent-dynamics 三类表征对分类的贡献。这样模型不是固定相信某一种表示，而是在缺失程度、动态复杂度和任务需求变化时，学习不同信息源的权重。

### 审稿人视角：价值与不足

最有价值的技术思想是把缺失/不规则时序分类中的“信息源选择”显式建模。很多 NDE 方法默认 control path 或 latent dynamics 足够好，很多插补方法默认补齐后的序列可直接分类；TANDEM 则承认三类表征各有偏差，并用 attention + differentiable gating 在端到端分类目标下选择它们。这个设计对审稿人有吸引力，因为它不是单个 backbone trick，而是一个可插拔融合层：同一框架可以接 Neural ODE、CDE 或 SDE，并通过消融验证 raw、control path、latent stream 的互补性。

不足在于，TANDEM 主要把 missingness / irregularity 当作“分类信息如何融合”的问题，还没有显式分解 state-driven missingness 与 policy-driven missingness。Gumbel gate 学到“在什么情况下相信 raw observation、interpolated path 或 latent dynamics”，但这些 gate 可能受训练环境中的采样政策支配：如果某类病人或某类设备总是更稀疏，模型可能把“该相信 latent dynamics 而非 raw observation”本身学成类别 shortcut。论文评估了多缺失水平下的分类准确率，但缺少跨医院、跨设备、跨主动采样规则或反事实 observation policy 下 gate 稳定性的系统实验。

### 对 Sampling-Policy Shift 的启发

TANDEM 对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以被看作“三类证据流可信度”的偏移。训练医院中 raw observation 可靠，换到目标医院后同样的缺失间隔可能代表不同采样流程；训练设备下 control path 平滑合理，换到事件触发采样设备后插值轨迹可能制造伪动态。因此，除最终 logits 外，我们还应审计 raw/control/latent gate 的跨策略稳定性。

纵向深化上，可以把 TANDEM 改造成 state-policy dual-gating NDE。state gate 学习跨采样政策稳定的证据融合规则，进入分类主路径；policy gate 学习当前观测制度下 raw observation、插值路径和 latent dynamics 的可靠性残差，用于偏移告警和不确定性校准。训练时对同一潜在轨迹生成多种采样策略视图，要求 state gate、state latent representation 和分类 logits 保持一致，同时允许 policy gate 预测采样策略、缺失机制和观测密度。这样能把 TANDEM 的“缺失条件下自适应融合”推进到“策略偏移条件下区分状态证据与观测政策证据”的非规则时序分类框架。

## 2. LLM Plug-ins Are Not a Free Lunch for Clinical Time-Series Prediction

- 会议/状态：Proceedings of the 1st Workshop on Linguistic Analysis for Health (HeaLing 2026), EACL 2026 workshop
- 作者：Juhwan Choi, Kwanhyung Lee, Sangchul Hahn, Eunho Yang
- 论文：https://aclanthology.org/2026.healing-1.17/
- DOI：10.18653/v1/2026.healing-1.17
- 关键词：clinical time-series prediction, ICU prediction, frozen LLM plug-in, MIMIC-III, representation compatibility, medical-domain LLM

### 场景、任务与核心难点

这篇工作面向 ICU 临床时序预测中的 LLM 集成问题。输入来自 MIMIC-III 的结构化临床时间序列，任务覆盖两个 ICU prediction settings；模型不是把 EHR 序列文本化后直接交给 LLM，而是在已有 clinical time-series backbone 聚合出的表示和 prediction head 之间插入一层 frozen LLM Transformer layer，检验这种 plug-in 是否能作为低成本 inductive prior 提升临床时序预测。

核心难点在于，近期很多工作默认“LLM 层或医学 LLM 先验”能改善医疗预测，但临床时间序列和语言 token 的结构并不天然兼容。ICU 时序表示中混合了生命体征、化验、缺失、时间窗口统计和采样流程；frozen LLM layer 既不能看到原始时间戳，也不能主动修复前端 encoder 对不规则采样的处理方式。论文系统比较不同 backbone、不同任务以及 general-purpose / medical-domain LLM plug-in，发现收益高度异质：某些组合有小幅提升，但整体并不稳定，也不足以证明 frozen LLM plug-in 是临床时序预测的通用增益模块。

### 审稿人视角：价值与不足

最有价值的贡献是给“把 LLM 组件插进临床时序模型”降温，并把失败模式定位到 representation compatibility。对审稿人而言，负结果很重要：如果一个 frozen language Transformer layer 只是在 aggregated time-series representation 后面做有限变换，它并不会神奇理解 ICU 中的采样频率、missingness、变量联测和时序因果。论文的实验设计也有现实意义，因为 plug-in 方案确实符合低计算、低改动的部署需求；它证明这种低改动方案的效果取决于 backbone、任务和数据规模，而不是稳定的免费午餐。

不足在于，这篇工作不是原生 irregular sampled time-series classification 架构，也没有把采样策略偏移作为主实验变量。它主要评估 frozen LLM layer 是否改善 ICU 时序预测，对前端如何处理异步观测、缺失机制、value-pending 和跨医院测量制度讨论较少。由于 plug-in 放在聚合表示之后，如果前端已经把 sampling-policy shortcut 编入 embedding，LLM layer 可能只是重新混合这些 artifact，而不会识别其不可迁移性。论文也只覆盖 MIMIC-III 中有限任务，需要跨医院和反事实采样评估来判断其结论能否推广。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：LLM 先验不能替代采样机制建模。若 state encoder 在输入层已经把“某变量是否被测、何时被测、多久未测”当作类别证据，后接一个 frozen LLM plug-in 并不会自动把 policy signal 还原成 patient-state signal；它甚至可能用更强的非线性变换放大这种 shortcut。因此，解决非规则采样下的策略偏移，应优先发生在输入表示、连续时间 encoder 和 state-policy disentanglement 层，而不是指望后端 LLM 模块事后修正。

纵向深化上，可以把 plug-in 评估改造成 policy-aware compatibility test：分别训练 state-only encoder、policy-only encoder 和 full encoder，再观察 frozen LLM layer 对三者的增益是否不同。如果 plug-in 对 policy-only embedding 也显著提升预测，说明它可能在放大采样政策信号；如果只对 state embedding 有稳定收益，才说明语言/医学先验与病程表征兼容。进一步可设计双通道 plug-in：state representation 接受医学知识层增强，policy representation 只进入校准、拒识和偏移解释。这样能把“LLM plug-in 是否有用”的经验问题转化为“LLM 是否帮助隔离而非放大采样政策 shortcut”的可审计问题。
