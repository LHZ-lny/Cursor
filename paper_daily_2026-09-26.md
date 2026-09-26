# Paper Daily - 2026-09-26

## 检索与去重记录

- 已强制读取根目录下全部已发现的日期版 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`、`paper_daily_2026-09-24.md`、`paper_daily_2026-09-25.md`；同时分段读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充黑名单。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`、`Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models`、`MedFuse: Multiplicative Embedding Fusion for Irregular Clinical Time Series`、`ReDiTT: Retrieval Augmented Conditional Diffusion Transformers for Asynchronous Time Series`、`DynaMamba: Multi-scale Dynamic Interacting Mamba Network for Irregular Clinical Time Series Classification` 与 `Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring` 等全部已总结论文。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / longitudinal EHR disease progression / continuous-time clinical prediction / sampling-policy shift 的 ICLR、ICML、NeurIPS、AAAI、OpenReview、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 `MUSE-Net`（时间窗口与 venue 不匹配，且历史日报已标记排除）、`A Statistical Approach for Modeling Irregular Multivariate Time Series with Missing Observations`（期刊/非顶会来源，历史日报已标记排除）、`HeLoM`（ICLR 2026 withdrawn submission，且更偏纵向 EHR LLM 记忆而非原生 IMTS 分类）、`ChorusTIC/TimEE`（通用规则 MTS in-context classification，非不规则采样主线）等候选。严格的“近月顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 1 篇未在黑名单中的高相关新工作：`Temporally Detailed Hypergraph Neural ODE for Disease Progression Modeling` 是 ICLR 2026 Poster，虽任务是下一次就诊并发症 marker 向量预测而非 P12/P19 式单标签分类，但它明确从 irregular-time longitudinal EHR 学习连续时间病程动力学，并用二元 marker 预测评估分类性能，对 Sampling-Policy Shift 下的“病程时间 vs 观测时间/就诊政策”分解有新增价值。

## 1. Temporally Detailed Hypergraph Neural ODE for Disease Progression Modeling

- 简称：TD-HNODE
- 会议：ICLR 2026 Poster
- 作者：Tingsong Xiao, Yao An Lee, Zelin Xu, Yupu Zhang, Zibo Liu, Yu Huang, Jiang Bian, Jingchuan Guo, Zhe Jiang
- 官方页：https://iclr.cc/virtual/2026/poster/10011634
- OpenReview：https://openreview.net/forum?id=3XRAkZtMPK
- 论文：https://arxiv.org/abs/2510.17211
- 关键词：irregularly sampled clinical events, longitudinal EHR, disease progression modeling, neural ODE, temporally detailed hypergraph, next-visit complication prediction

### 场景、任务与核心难点

TD-HNODE 面向慢病 longitudinal EHR 中的疾病进展建模，尤其是 2 型糖尿病及其心血管相关并发症轨迹。每个患者由不规则就诊序列构成；每次 encounter 包含风险因子向量，例如化验、生命体征、用药，以及并发症 marker 向量，例如 hypertension、atrial fibrillation、heart failure、cerebrovascular disease、stroke 等。任务是在给定历史不规则就诊、风险因子和当前并发症状态后，预测下一次就诊的二元 complication marker vector。因此它不是传统整条序列单标签分类，但本质上是 irregular EHR 上的多标签下一状态分类/风险预测。

核心难点有三层。第一，慢病状态在连续时间中演化，但 EHR 只在不规则就诊或检查时间点被观察；相邻 encounter 的时间间隔本身反映病情、随访制度和医疗可及性。第二，许多并发症并不是任意共现，而是沿临床知识认可的 progression pathways 逐步出现，例如从 hypertension 到 atrial fibrillation 再到 heart failure；普通 RNN/Transformer 或 pairwise graph 难以表达整条路径内的高阶依赖。第三，不同患者的进展速度和路径不同，需要模型能在共享医学路径先验的同时适配个体化时间动态。

论文将临床 progression trajectories 表示为 hypergraph：节点是 complication markers，hyperedge 是一条医学路径。进一步，它把每位患者已经发生的 marker 及其首次发生时间写入 temporally detailed hyperedge，使 hypergraph 不只是静态知识图，而是带患者时间戳的动态病程结构。TD-HNODE 用 Neural ODE 在不规则 encounter 之间积分 disease-state hidden dynamics，并通过 learnable TD-Hypergraph Laplacian 建模路径内 marker 重要性和路径间相关性。实验在 University Hospital 与 MIMIC-IV 两个真实 EHR 数据集上进行；在 MIMIC-IV 上，TD-HNODE 报告 87.9% accuracy、85.7% recall、42.9% F1，优于 T-LSTM、ContiFormer、NODE、CODE-RNN、TGNE、HyperTime 等基线。

### 审稿人视角：价值与不足

最有价值的思想是把“临床进展路径”与“不规则时间动力学”真正耦合起来，而不是把医学知识作为静态先验或事后解释。TD-Hypergraph 的设计让模型知道哪些 marker 属于同一条临床路径，也知道它们在某个患者身上何时首次出现；attention-based incidence matrix 则让不同 marker 在不同当前进展点上的重要性可变。作为审稿人，我认为这个结构比普通 temporal graph 更适合慢病进展：它能表达路径级高阶依赖，而 Neural ODE 又能处理 encounter 间隔不均带来的连续时间演化。

不足在于，TD-HNODE 依赖预定义且较可靠的疾病进展路径，这在糖尿病/心血管并发症中较合理，但迁移到 sepsis、ICU acute deterioration 或多病种混合风险预测时，路径知识可能不完整甚至有争议。另一个关键不足是，它把 marker 首次出现时间视为病程时间信号，但 EHR 中“何时首次记录某个并发症/异常”也可能受诊断流程、筛查频率、医保规则和就诊可及性影响。若某医院更早筛查视网膜病变，或某类患者更频繁复诊，TD-Hypergraph 上的时间戳会同时包含真实病程和 observation policy。论文展示了分类性能和 sub-phenotyping 可解释性，但还没有系统评估跨医院随访制度、筛查政策或反事实 visit schedule 改变后，learned Laplacian 与 marker prediction 是否保持稳定。

### 对 Sampling-Policy Shift 的启发

TD-HNODE 对 Sampling-Policy Shift 的横向启发是：在临床不规则时序中，采样策略偏移不只表现为变量缺失或 delta-t 分布变化，还会改变“疾病进展时间戳”的表观语义。同一个 complication marker 的首次记录时间，可能是病理真正进展到该阶段，也可能是医院更早筛查、更密集随访或更严格编码造成的观测提前。因此，我们在建模非规则采样分类时，应把 disease clock 与 observation clock 分离：前者描述跨策略稳定的病程相位，后者描述就诊、检查和记录政策。

纵向深化上，可以借鉴 TD-HNODE 构造 policy-aware progression hypergraph。state hypergraph 只编码跨采样/就诊政策稳定的病理路径和 marker 依赖，进入分类主路径；policy hypergraph 则编码就诊间隔、筛查频率、首次记录延迟、科室/医院流程和编码习惯，用于偏移诊断与不确定性校准。训练时可对同一潜在病程生成不同 visit schedule、screening policy 和 marker-recording delay，约束 state Laplacian、patient embedding 和下一次 marker logits 保持一致，同时允许 policy Laplacian 预测观测制度。这样能把 TD-HNODE 的“时间细化病程超图”推进到“区分真实病程进展与观测政策进展”的非规则采样分类框架。
