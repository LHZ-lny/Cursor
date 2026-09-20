# Paper Daily - 2026-09-20

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`；同时从兼容入口 `paper_daily.md` 提取标题索引，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings`、`A novel approach to classification of ECG arrhythmia types with latent ODEs`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`，以及自动化记忆中记录的 `LERD`、`WIPSNet`、`Context-Aware Neural SDEs`、`RAxSS`、`CauKer` 等已总结工作。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / EHR foundation model / patient-aware sampling / observation policy / sampling-policy shift 的顶会、顶会 workshop、ICML virtual 页面、OpenReview、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 `DBGL`、`STAR-Set`、`SLAN`、`RAxSS`、`ORA` 等已总结条目，以及 temporal resampling for clinical RL、统计特征提取、federated imputation 等分类相关性或顶会来源不足的候选。本次保留 1 篇全新工作：`Pretraining EHR Foundation Models with Patient-Aware Sampling` 是 ICML 2026 Structured Data for Health workshop 工作，虽不是传统 IMTS 架构论文，但它直接指出 EHR 预训练窗口构造会因患者轨迹长度和跨患者拼接产生训练信号偏置，并在多个 ICU/住院分类任务上提升 Macro AUROC/AUPRC；对“非规则采样下的采样策略偏移”尤其有方法论价值。

## 1. Pretraining EHR Foundation Models with Patient-Aware Sampling

- 会议/状态：ICML 2026 Workshop on Structured Data for Health；arXiv 2026-07
- 作者：Joshua Placidi, Yuxuan Liu, Jinpei Han, Marek Rei, A. Aldo Faisal
- ICML 页面：https://icml.cc/virtual/2026/71790
- 论文：https://arxiv.org/abs/2607.22114
- 关键词：EHR foundation model, patient-aware sampling, autoregressive pretraining, sequence construction, ICU mortality, ICU readmission, hospital mortality, Macro AUROC/AUPRC

### 场景、任务与核心难点

这篇工作面向自回归 EHR foundation model 的预训练数据构造。输入是每位患者的纵向 EHR token 序列：化验、诊断、用药、就诊和 ICU/ED 事件等按时间组成长度极不均衡的 patient timeline。下游评估覆盖 MIMIC-IV v2.2/v3.1 与 ED 模块上的 ICU mortality、ICU readmission、ICU admission 和 hospital mortality 等分类任务，指标包括 Macro AUROC 与 Macro AUPRC。

论文抓住的核心难点不是新的 encoder，而是一个很容易被忽略的预训练采样策略：很多 EHR 自回归模型沿用语言模型的 global stream，把所有患者轨迹拼成一个长 token 流，再切固定长度窗口训练。这个做法在 NLP 中主要是工程选择，但在 EHR 中会引入两类偏差：一是窗口可能跨越患者边界，把无关患者事件放进同一上下文；二是长轨迹患者贡献更多训练窗口，使住院更久、记录更多、病情更复杂或流程更密集的患者在优化中被过度代表。作者比较 Global Stream、确定性的 Patient Chunks，以及随机 Patient Sampling。Patient Sampling 先按 `p_alpha(i) proportional |W(i)|^alpha` 抽患者，再在该患者内部抽窗口；`alpha=0` 接近患者均匀采样，`alpha=1` 接近按可用窗口数加权。实验中 `alpha=0.6` 整体最好：在 MIMIC-IV v2.2 + ED 上 Macro AUROC/AUPRC 从 0.834/0.467 提升到 0.842/0.478，在 MIMIC-IV v3.1 + ED 上从 0.816/0.466 提升到 0.825/0.477。

### 审稿人视角：价值与不足

最有价值的思想是把 EHR foundation model 的“采样策略”从数据加载细节提升为建模变量。很多论文把改进重点放在 tokenizer、Transformer/Mamba backbone 或 pretraining objective 上，却默认窗口切分方式无害；这篇工作说明，在患者轨迹长度高度长尾、临床事件密度与病情/流程相关的 EHR 中，训练窗口如何采样会直接改变模型看到的患者分布和优化信号。Patient Sampling 的优点在于简单、架构无关、可调：不需要改模型，只通过 `alpha` 控制“每位患者公平”与“长轨迹信息充分利用”之间的权衡，并且在四个临床分类任务上带来稳定的宏平均收益。

不足在于，它仍主要处理 pretraining data construction 的统计偏置，尚未完整刻画 EHR 中更复杂的 observation policy。患者长轨迹为何更长，可能来自真实病情复杂、住院时间更久、医院记录制度更密、检查频率更高或资源使用更多；Patient Sampling 能缓解长轨迹在训练中的过度权重，但不能区分“应该保留的病程信息”和“医院记录/采样流程导致的 shortcut”。论文也主要在单一数据家族 MIMIC-IV v2.2/v3.1 + ED 上评估，尚未证明在跨医院、跨科室、跨测量政策时最优 `alpha` 是否稳定。另一个限制是，该方法作用在 token-window 层面，若 token 本身已经编码了强 policy signal，例如某些检查只在特定风险流程中出现，单纯重加权窗口并不能阻止模型利用这些策略性事件。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发非常直接：采样策略偏移不只发生在模型输入的时间戳、mask 和变量可见性上，也发生在预训练/训练过程本身。若训练窗口更频繁来自长住院、高记录密度或高干预患者，模型学到的“常见状态”其实可能是“常被观察的状态”。这会让下游分类器在面对记录稀疏、住院流程不同或采样资源不同的目标环境时产生偏移。Patient-aware sampling 提醒我们，构造 foundation model 语料时应报告并控制 patient-level sampling distribution，而不只是 token 数、窗口数和总体 loss。

纵向深化上，可以把 Patient Sampling 扩展为 policy-aware pretraining sampler：先为每位患者或每个窗口估计 policy descriptors，例如轨迹长度、事件密度、变量联测强度、long-gap pattern、routine-rhythm deviation、value-pending 比例和医院/科室 ID；再用多维采样权重控制不同 policy regime 在预训练中的贡献。训练目标可分成 state objective 与 policy objective：state branch 在不同采样权重、不同窗口视图和不同反事实 observation policy 下保持表示与分类 logits 稳定；policy branch 则预测轨迹长度、记录密度和采样流程，用于偏移诊断与校准。这样可以把该论文的“按患者公平分配训练信号”推进到“按采样政策公平覆盖并显式隔离 policy shortcut”的非规则时序 foundation model 训练框架。
