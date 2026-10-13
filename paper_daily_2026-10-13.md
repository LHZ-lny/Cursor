# Paper Daily - 2026-10-13

## 检索与去重记录

- 已强制读取根目录下全部已发现的日期版 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`、`paper_daily_2026-09-24.md`、`paper_daily_2026-09-25.md`、`paper_daily_2026-09-26.md`、`paper_daily_2026-09-27.md`、`paper_daily_2026-09-29.md`；同时读取兼容入口 `paper_daily.md` 的标题索引，并参考自动化记忆中的补充黑名单。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`、`Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models`、`ReDiTT: Retrieval Augmented Conditional Diffusion Transformers for Asynchronous Time Series`、`ReTAMamba: Reliability-Aware Temporal Aggregation with Mamba for Irregular Clinical Time Series Prediction`、`TANDEM: Temporal Attention-guided Neural Differential Equations for Missingness in Time Series Classification` 与 `LLM Plug-ins Are Not a Free Lunch for Clinical Time-Series Prediction` 等全部已总结论文。
- 检索范围：近月围绕 irregular sampled / asynchronous / irregular clinical time series classification / sampling pattern shift / non-random missingness / clinical workflow-driven modality availability 的 OpenReview、NeurIPS TS4H/CauScien、EMNLP/ACL Anthology、arXiv、ICLR/ICML/NeurIPS 页面与论文索引。
- 已排除全部黑名单论文；同时排除 `DBGL`、`ATENet`、`SLAN`、`ReTAMamba`、`Rethinking LLMs`、`STAR-Set`、`PRISM` 的同名误检索片段以外的非论文页面、`PhASER`（此前已多次因非原生不规则采样主线被排除）和 `CT-HEG`（高度相关但暂无顶会/顶会 workshop 录用，当前仅 arXiv/BMC 投稿线索）。本次保留 2 篇未在黑名单中的新跟踪对象：`Domain Generalization under Sampling Pattern Shifts in Irregular Time Series` / PRISM 是 2026-09 新预印本，venue 尚未确认，但它直接以 ISMTS 的 sampling-pattern shift 为研究对象，和我们的问题高度同构；`Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness` 是 EMNLP 2025 主会论文，并被 NeurIPS 2025 TS4H/CauScien workshop 收录，虽不是传统单一数值 IMTS 分类器，但明确建模临床决策导致的模态可见性偏差，对 Sampling-Policy Shift 的因果表征分解有直接启发。

## 1. Domain Generalization under Sampling Pattern Shifts in Irregular Time Series

- 方法名：PRISM
- 会议/状态：arXiv 2026-09 预印本，尚未确认顶会录用
- 作者：Changhun Kim, Joohyung Lee, Kwanhyung Lee, Donghwee Yoon, Grigorios Chrysos, Eunho Yang
- 论文：https://arxiv.org/html/2609.33279v1
- 关键词：irregularly sampled multivariate time series, sampling pattern shift, domain generalization, feature-centric representation, sampling-centric representation, HAR-C benchmark

### 场景、任务与核心难点

PRISM 面向 irregularly sampled multivariate time series (ISMTS) 的跨域分类/预测问题。它关注的不是普通缺失率提升，而是更贴近部署现实的 sampling pattern shift：不同域中 observation times 和 available measurements 的组合方式不同，模型在源域学到的采样模式 shortcut 到目标域会失效。论文构造 HAR-C controlled sampling shift benchmark，并在 controlled 与 real-world ISMTS benchmarks 上评估 unseen sampling shifts 下的鲁棒性。

核心难点在于，近年的 IMTS 模型越来越善于利用采样信息，例如 mask、delta-t、变量可见性和观测密度；这些信号在同分布测试时确实有预测力，却可能在采样制度改变时变成脆弱捷径。PRISM 的思路是先用无标签双视图表征学习，把 feature-centric representation 与 sampling-centric representation 拆开；随后在监督训练阶段引入 adversarial 与 diversity-enhanced 机制，让分类器在多种采样变化下减少对特定 sampling pattern 的依赖。换言之，它不否认采样模式有信息，而是要求模型知道哪些信息属于内在状态、哪些属于当前采样制度。

### 审稿人视角：价值与不足

最有价值的思想是把 sampling pattern shift 从泛泛的 domain shift 中单独拎出来，并给出 controlled benchmark 与方法闭环。对审稿人而言，这一点非常关键：过去很多 IMTS 鲁棒性实验只做随机 mask、随机 drop 或同数据集高缺失测试，难以说明模型是否抵抗真实采样制度变化；PRISM 明确证明 sampling shifts alone 就能导致性能退化、诱发 sampling-specific shortcuts，并让现有 DG 方法仍感到困难。feature-centric / sampling-centric 的无监督拆分也很贴合实际，因为目标域标签通常稀缺，而采样轨迹本身在部署时天然可得。

不足首先是论文当前仍是预印本，尚不能按已录用顶会结论使用。其次，feature 与 sampling 的可分性依赖数据构造和增强假设：在临床场景中，采样往往由真实病情、医生判断和医院流程共同决定，过强地压制 sampling representation 可能也会丢掉 state-driven informative sampling。论文需要更细的医疗/EHR 环境实验来证明：PRISM 去掉的是不可迁移的 policy shortcut，而不是稳定病程触发的观测行为。此外，sampling-centric representation 若只作为 adversarial nuisance，也可能低估它在校准、拒识和数据质量解释中的价值。

### 对 Sampling-Policy Shift 的启发

PRISM 与我们要解决的 Sampling-Policy Shift 几乎同构。横向应用上，可以直接借鉴其 HAR-C 式 controlled benchmark 思路，为医疗 IMTS 构造同一潜在病程下的多种采样政策：固定节律采样、阈值触发采样、告警后密集采样、成本约束变量选择和医院特定联测规则。评估时不只报告平均 AUROC/AUPRC，还应报告 unseen policy、policy-only classifier、sampling-centric shortcut reliance 和 state-feature stability。

纵向深化上，可以把 PRISM 从 domain generalization 扩展为 clinical state-policy dual encoder。state encoder 学跨采样政策稳定的病程表征，进入主分类路径；policy encoder 学观测时间、变量可见性、联测模式和采样密度，用于偏移告警、校准和拒识，而不是被简单对抗掉。训练时要求同一潜在轨迹在多种 observation policy 下的 state representation 与 logits 保持一致，同时允许 policy representation 解释采样制度差异。这样能把 PRISM 的“减少 sampling shortcut”推进到“保留并隔离 sampling policy，用于可审计鲁棒分类”。

## 2. Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness

- 会议/状态：EMNLP 2025 Main；NeurIPS 2025 TS4H Workshop Poster；NeurIPS 2025 CauScien Workshop Poster
- 作者：Zihan Liang, Ziwen Pan, Ruoxuan Xiong
- 论文：https://aclanthology.org/2025.emnlp-main.1465/
- arXiv：https://arxiv.org/abs/2509.17228
- OpenReview：https://openreview.net/forum?id=YUfWZtvfsI
- 关键词：multimodal clinical records, non-random modality missingness, MMNAR, causal representation learning, MIMIC-IV, eICU, hospital readmission, ICU admission

### 场景、任务与核心难点

这篇工作面向多模态临床记录中的 outcome prediction。输入包含结构化数据、影像、临床文本、出院小结、放射报告等模态；任务覆盖 MIMIC-IV 与 eICU 上的 hospital readmission、ICU admission 等临床预测。它不是传统意义上只处理数值 `(time, variable, value)` triplets 的 IMTS 分类器，但问题本质非常接近非规则采样：哪些模态可见、哪些文本或影像被记录，本身由临床决策、检查流程和患者状态共同决定，属于 modality missing-not-at-random (MMNAR)。

核心难点是，模态缺失并非随机噪声。比如是否有 discharge summary、是否拍胸片、是否生成 radiology report，既反映患者病情，也反映医院流程、医生判断和资源配置。若模型直接把“某模态是否存在”当成强预测信号，它在源医院可能有效，换到模态记录政策不同的医院后就可能失效。论文提出一个 causal representation learning 框架：MMNAR-aware modality fusion 在融合结构化数据、影像和文本时显式条件化 missingness patterns，以同时捕捉 patient health 与 clinician-driven assignment；modality reconstruction + contrastive learning 维持语义充分性；最后用带 rectifier 的 multitask predictor 修正特定模态观测模式带来的残余偏差。实验显示相对强基线在 hospital readmission 和 ICU admission 上分别最高有 13.8% 与 13.1% AUC 相对提升。

### 审稿人视角：价值与不足

最有价值的思想是把“模态是否出现”从普通缺失变量提升为临床决策驱动的因果问题。对审稿人来说，这比仅仅把 missingness mask 拼进模型更进一步：论文承认模态可见性同时包含健康状态和医生/机构 assignment mechanism，并试图在表征、重构、对比学习和预测校正中处理这种混合来源。它对多模态 EHR 特别重要，因为 LLM/影像/结构化数据融合越强，模型越容易把记录流程当成病情证据；MMNAR-aware fusion 和 rectifier 提供了一个可操作的偏差修正接口。

不足在于，论文主轴是多模态临床记录和文本/影像模态缺失，而不是原生连续时间 IMTS encoder；对变量级异步采样、delta-t、value-pending、变量联测和反事实采样策略的处理还需要进一步落地。另一个限制是，MMNAR 分解依赖可观测模态和任务监督；如果医院政策、医生偏好或资源约束没有被充分观测，rectifier 可能只能校正一部分偏差。最后，利用 missingness patterns 提升预测性能并不自动等于跨策略稳健，仍需要跨医院模态记录政策、跨科室检查协议和反事实 modality assignment 下的评估。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：非规则采样可以被看作更细粒度的 MMNAR。数值变量是否被测、某项化验是否被下单、文本记录是否出现、影像是否生成，本质上都是 observation assignment decisions。我们可以借鉴其 causal missingness view，把 sampling policy 写成显式 assignment mechanism：它既受 patient state 影响，也受医院流程影响。分类器需要从观测值中提取 state evidence，同时把 assignment evidence 放进 policy branch 做校准和偏移诊断。

纵向深化上，可以把 MMNAR-aware fusion 改造成 time-aware state-policy fusion。state fusion 聚合跨采样政策稳定的观测值、趋势和临床语义；policy fusion 聚合 mask、delta-t、变量可见性、模态可见性、value-pending 和文书/检查流程。训练时对同一潜在病程施加不同 variable-level 与 modality-level observation policy，要求 state representation、clinical rationale 和 logits 稳定，同时允许 policy representation 预测观测制度和不确定性。这样能把多模态 MMNAR 的因果表征思想纵向深化到非规则采样时序分类中的 Sampling-Policy Shift。
