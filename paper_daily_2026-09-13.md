# Paper Daily - 2026-09-13

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`；同时用标题检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-09-12 的新增标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness`、`Shared-Benchmark Regime Decomposition for Nonstationary Clinical Decisions`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification` 与 `WIPSNet: Deep Learning for Paediatric Wheeze Detection from Impedance Pneumography`。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / ICU time-series domain adaptation / medical time-series language model 的顶会、顶会 workshop、OpenReview、ICML virtual 页面、arXiv 与论文项目页。
- 已排除全部黑名单论文；同时排除 MedFuse 这类历史日报已判断为 withdrawn/submission 风险的候选、Under-Cali/MoRGen 等偏 forecasting 或 generation 的候选、PRIME 这类期刊而非顶会候选，以及 HEARTS 这类 benchmark 价值高但与非规则采样分类联系较弱的候选。严格的“顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 2 篇全新工作：`INPUTADAPTER` 是 ICML 2026 Foundation Models for Structured Data workshop 工作，直接面向跨医院 ICU 时序分类器在目标域退化的问题；`OpenTSLM` 是 ICML 2026 主会 Poster，虽不以 irregular sampling 为标题主轴，但其面向多变量医疗时序、可变长度/频率信号、分类与自然语言推理的原生时序-语言融合，对非规则采样分类的可解释接口和采样策略偏移诊断有新增价值。

## 1. Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors

- 方法名：INPUTADAPTER
- 会议/状态：ICML 2026 Workshop on Foundation Models for Structured Data
- 作者：Omer Gotfrid, Dvir Aran, Barak Gahtan, Alex Bronstein
- 官方页：https://icml.cc/virtual/2026/71728
- Workshop accepted list：https://icml-structured-fm-workshop.github.io/accepted-papers/
- 关键词：ICU time-series classification, domain adaptation, frozen predictor, MIMIC-IV, eICU, HiRID, input-space transformation, cross-hospital shift

### 场景、任务与核心难点

INPUTADAPTER 面向真实医疗部署中很常见但经常被 IMTS benchmark 弱化的问题：一个已经验证、监管备案或嵌入硬件的软件医疗器械模型，在源医院训练后不能随意改权重，但目标医院的 ICU 时序数据分布已经变化。论文评估 MIMIC-IV 到 eICU、HiRID 的跨医院迁移，覆盖 5 个 ICU classification tasks，并同时在 AdaTime 时间序列适配基准上比较。这里的核心任务不是从零训练一个更强 encoder，而是在下游 predictor 完全冻结的情况下，让目标域输入被转换成源域模型能够理解的形式。

核心难点在于，跨医院退化往往不是单一 covariate shift：变量量纲、采样频率、时间窗口统计、缺失模式、告警后测量密度、检查流程和患者结构都可能同时变化。常规 domain adaptation 假设 encoder 可训练，alignment loss 可以反向传播到表征层；但在 SaMD 迁移、嵌入式固件或已验证临床模型场景中，预测器权重被锁定，目标域标签可能有限，且监管上更希望“适配数据”而不是“改模型”。INPUTADAPTER 因此学习一个 input-space adapter，把目标域时间窗口变换成 frozen source predictor 可消费的输入；可选地，它还在源 encoder latent space 中按时间步检索 k 个近邻源样本，并通过 cross-attention 融合上下文。论文报告其在 MIMIC-IV->eICU/HiRID ICU 任务上优于或追平多类 frozen-backbone、end-to-end DA 与 test-time adaptation baseline，其中 eICU AKI 任务 AUCPR 提升最大达到 +14.1 点。

### 审稿人视角：价值与不足

最有价值的思想是把临床时序迁移从“改模型参数”转成“学习一个可审计的输入域变换”。这在医疗 AI 中很现实：很多模型一旦验证和上线，重新训练主干的成本、审计负担和风险都很高。INPUTADAPTER 关注 frozen predictor 这个被低估但重要的部署约束，并通过输入变换 + 源域实例检索，把目标域数据尽量映射回已知模型的工作区间。对审稿人而言，它的贡献不只在指标提升，也在于提出了一种更贴合监管和硬件约束的 adaptation regime。

不足是，INPUTADAPTER 仍主要把跨医院差异作为可学习的输入分布变换来处理，尚未充分因果分解 patient state shift 与 observation policy shift。若目标医院的采样政策本身不同，例如某些化验只在更危重患者中下单、某些变量在 eICU/HiRID 中记录粒度不同，adapter 可能把目标域的政策痕迹“翻译”成源域中同样带标签相关性的痕迹，而不一定恢复真正稳定的病理状态。另一个风险是可解释性：输入空间变换若改变时间窗口统计、缺失模式或数值尺度，临床上需要知道它是在校正设备/流程差异，还是在抹掉目标医院中真实有意义的护理差异。

### 对 Sampling-Policy Shift 的启发

INPUTADAPTER 对 Sampling-Policy Shift 的横向启发非常直接：采样政策变化可以被看作 frozen classifier 前的输入空间错配。现实中我们可能无法重训已部署的 ICU 分类器，但可以学习一个 policy-aware adapter，把不同医院、设备或科室采样策略下的输入映射到分类器训练时的观测协议附近。这提示我们不仅要做 policy-invariant encoder，也可以研究 policy-normalizing input adapter，尤其适合 legacy model 或监管锁定模型的迁移。

纵向深化上，可以把 INPUTADAPTER 拆成 state adapter 与 policy adapter 两层。state adapter 只校正跨域稳定的数值尺度、时间对齐和变量表示，使 patient-state evidence 被 frozen predictor 正确读取；policy adapter 则显式建模目标域的采样密度、变量联测、value-pending 和告警后测量模式，仅用于校准、偏移告警或拒识。训练时可加入反事实采样增强：同一潜在病程经过不同 observation policy 后，adapter 输出给 frozen predictor 的 state-relevant logits 应保持稳定，而 policy head 应能区分采样制度。这样能把“适配输入”推进到“适配采样政策但不把政策 shortcut 注入分类边界”。

## 2. OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data

- 会议：ICML 2026 Poster
- 作者：Patrick Langer, Thomas Kaar, Max Rosenblattl, Maxwell A. Xu, Winnie Chow, Martin Maritsch, Robert Jakob, Ning Wang, Juncheng Liu, Aradhana Verma, Brian Han, Daniel Kim, Henry Chubb, Scott Ceresnak, Aydin Zahedivash, Alexander Sandhu, Fatima Rodriguez, Daniel McDuff, Elgar Fleisch, Oliver Aalami, Filipe Barata, Paul Schmiedmayer
- 官方页：https://icml.cc/virtual/2026/poster/65261
- 论文：https://arxiv.org/abs/2510.02410
- 代码：https://github.com/OpenTSLM/OpenTSLM
- 关键词：medical time-series language model, multivariate signals, text-time-series reasoning, sleep staging, human activity recognition, ECG question answering, variable length and frequency

### 场景、任务与核心难点

OpenTSLM 面向医疗与可穿戴场景中的多变量时序信号和文本联合推理。论文关注的不是 EHR 化验事件流，而是更接近原始传感器的 ECG、EEG/sleep、activity 等多变量信号；下游任务包括 Sleep-CoT 的睡眠阶段分类、HAR-CoT 的人体活动识别以及 ECG-QA-CoT 的心电问答/解释。它要解决的核心问题是：通用 LLM 很擅长处理文本、图像和音频，但通常不能原生消费长序列、多通道、不同长度和不同频率的医学时间序列；把信号转成文本或图片会丢失精细时序结构，也会带来上下文长度和格式偏差。

论文提出两个 TSLM 架构。OpenTSLM-SoftPrompt 将时序 encoder 产生的可学习 token 接入 LLM token 序列，参数效率高，适合较短序列；OpenTSLM-Flamingo 则用 Perceiver-style resampler 和 gated cross-attention，把时序作为独立模态显式接入 LLM，因而在多变量、长序列输入上内存扩展更稳定。实验显示 OpenTSLM 在 sleep staging 和 HAR 上明显优于文本化/图像化 baselines，sleep staging F1 达到约 69.88%，HAR F1 约 65.44%-67.64%，并在 ECG-QA 中获得心脏科专家对解释质量的正面评估。

### 审稿人视角：价值与不足

最有价值的思想是坚持“时间序列应该作为原生模态进入语言模型”，而不是被粗暴序列化为文本或转成图像。对医疗时序而言，波形形态、局部节律、多通道同步关系和长程上下文往往同时决定分类标签；OpenTSLM-Flamingo 的 cross-attention 路线把 LLM 的语言推理能力和专门的时序 encoder 解耦，给出了比 prompt-only 更可扩展的接口。它还通过 CoT 数据集把分类、解释和问答放在同一框架中，有助于把“模型预测了什么”推进到“模型如何解释信号证据”。

不足在于，它不是专门针对 irregular EHR/IMTS 的方法，主要实验数据更接近连续传感器或规则窗口化信号；论文虽然强调多条时间序列可具有不同长度和频率，但对 missingness、异步变量观测、医生下单式采样政策和跨医院 observation process 的处理并不充分。另一个问题是，CoT 解释可能提升可读性，却不自动保证因果正确：如果输入格式、窗口选择或采样频率本身与类别相关，LLM 可能生成看似合理但实际依赖采样 shortcut 的解释。

### 对 Sampling-Policy Shift 的启发

OpenTSLM 对 Sampling-Policy Shift 的横向启发是：采样策略偏移最终也会进入“解释接口”。当我们把不规则采样信号交给 TSLM 或 LLM 辅助诊断时，模型不仅输出类别，还会生成自然语言理由；如果采样密度、变量可见性或传感器频率发生变化，解释文本可能把 policy artifact 包装成临床证据。因此，解决 Sampling-Policy Shift 不能只看 AUROC/AUPRC，还要审计 explanation 是否把“被测了什么、何时被测、采样多密”当作疾病证据。

纵向深化上，可以把 OpenTSLM 改造成 policy-audited TSLM：前端时序 encoder 分出 state tokens 和 policy tokens，前者进入分类与医学解释，后者进入数据质量、采样制度和不确定性说明。LLM 生成时被要求分别输出 state rationale 与 policy rationale，并在反事实采样视图下保持 state rationale 和诊断 logits 稳定，同时允许 policy rationale 描述不同采样协议。这样能把 OpenTSLM 的“原生时序-语言融合”推进到“可解释地隔离采样政策偏移”的临床时序分类助手。
