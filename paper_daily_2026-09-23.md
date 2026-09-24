# Paper Daily - 2026-09-23

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`；同时用标题检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`、`Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models` 与 `MedFuse: Multiplicative Embedding Fusion for Irregular Clinical Time Series` 等全部已总结工作。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / missingness shift / irregular clinical time series classification / cross-hospital distribution shift / sampling-policy shift 的顶会、顶会 workshop、ICML/NeurIPS 官方页面、OpenReview、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 `PhASER`（TMLR 2025 / NeurIPS TS4H Spotlight，但非原生不规则采样主线）、`Mamba-IVP`（ICLR 2026 submission 且主任务偏 forecasting / node classification）、`Interpretable Graph Learning on Irregular Clinical Time Series`（与历史记忆中已标记的 SuperMAN/GMAN 路线高度重叠）、以及 `DBGL/STAR-Set/SLAN/RAxSS/TimeCHEAT/INTERVenE/SLiCE/MedFuse` 等已总结条目。本次仅保留 1 篇未在黑名单中的新跟踪对象：`Missingness-Aware Conformal Prediction Under Cross-Hospital Distribution Shift` 是 ICML 2026 Structured Data for Health Poster，虽然不是新的 IMTS 分类主干，但直接处理跨医院 missingness / observation-policy shift 下临床预测覆盖率失效的问题，对“非规则采样下的 Sampling-Policy Shift”尤其有方法论价值。

## 1. Missingness-Aware Conformal Prediction Under Cross-Hospital Distribution Shift

- 会议/状态：ICML 2026 Workshop on Structured Data for Health Poster
- 作者：Liang You, Dongwen Ou, Hengyu Shi
- 官方页：https://icml.cc/virtual/2026/71725
- OpenReview：https://openreview.net/forum?id=h2I7GN4hTh
- 关键词：missingness shift, conformal prediction, cross-hospital distribution shift, clinical machine learning, subgroup coverage

### 场景、任务与核心难点

这篇工作面向多医院临床预测模型的部署可靠性问题。典型场景是：模型在某些医院或 pooled hospital data 上训练和校准，然后部署到 missingness pattern、变量可见性、检查流程和采样密度都不同的新医院。任务层面，它关注的不是重新设计一个 ICU 时序分类 encoder，而是在已有预测模型输出之上，用 conformal prediction 给出可靠的预测集合或不确定性控制，使模型在跨医院分布偏移下仍尽量维持目标覆盖率。

核心难点是，split conformal prediction 的边际覆盖保证依赖 exchangeability；跨医院部署时，missingness 本身会发生系统性偏移，这个条件被破坏。论文指出，pooled calibration 可以让整体 90% coverage 看起来达标，却掩盖某些 missingness subgroup 的严重 under-coverage。作者将 subgroup coverage gap 分解为 calibration heterogeneity 与 within-group shift 两部分，并提出一个 label-free selection rule：先找出跨站点 missingness shift 最大的变量，再基于该 missingness 变量做二分 Mondrian conformal calibration。结果显示，在 GOSSIS 和 MIMIC-IV 上，这种缺失感知分组校准可以显著缩小最大 subgroup gap，且预测集合大小变化很小。

### 审稿人视角：价值与不足

最有价值的思想是把 missingness 从“输入缺陷”提升为“校准分层变量”。很多不规则临床时序分类工作只问模型 AUROC/AUPRC 是否跨医院下降，而这篇论文问的是：即使平均准确率或平均覆盖率看起来正常，某些采样/缺失制度下的患者是否正在被系统性低覆盖？这种视角对医疗部署很关键，因为 sampling policy shift 的风险经常先表现为不确定性失真，而不是立即表现为整体指标崩盘。它的 label-free selection rule 也很实用：部署医院未必有足够标签重新训练或重校准复杂模型，但通常能观测 missingness / availability pattern。

不足在于，它主要是后验校准和覆盖率修复方法，并不直接学习 sampling-policy-invariant 的时序表征。如果底层分类器已经把医院测量政策当成病理证据，missingness-aware Mondrian calibration 可以缓解 subgroup under-coverage，却不能从根本上移除分类边界中的 policy shortcut。另一个限制是二分 missingness variable 的分组较轻量：它有利于稳健和可部署，但对多变量联测、时间依赖 missingness、value-pending、告警后密集采样等复杂策略机制表达不足。作为 workshop 工作，它还需要更多任务、医院和真实部署场景验证。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发非常直接：我们不能只报告跨策略 AUROC/AUPRC，还必须报告跨采样政策的校准和覆盖率分层。若训练医院与目标医院的采样制度不同，模型即使预测分数排序尚可，也可能对某些 missingness subgroup 给出过窄、不可信的预测集合。因而，在我们的 IMTS 分类研究中，应把 missingness descriptors、变量可见性、采样密度、联测模式、routine-rhythm deviation 和 value-pending 状态纳入 calibration audit。

纵向深化上，可以把这篇论文的 missingness-aware Mondrian calibration 与前端 state-policy disentanglement 结合：state encoder 负责学习跨采样政策稳定的病程表征，policy encoder 预测 missingness / observation-policy descriptors；分类头主要依赖 state representation，而 conformal calibration 按 policy descriptors 做分层覆盖率控制。训练和评估时，不仅比较 cross-policy accuracy drop，还要报告 policy subgroup coverage gap、set-size inflation、policy-only classifier predictability 和 counterfactual-policy consistency。这样能把 Sampling-Policy Shift 的目标从“平均性能不掉太多”推进到“不同采样政策患者都获得可靠且可解释的不确定性保障”。
