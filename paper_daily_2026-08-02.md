# Paper Daily - 2026-08-02

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
- 本次黑名单论文标题：
  - Adaptive Time Encoding for Irregular Multivariate Time-Series Classification
  - Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification
  - FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification
  - One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification
  - PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks
  - SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data
  - GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care
  - DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification
  - Fault Diagnosis of Irregular Sequences by Adjoint Learning in Continuous-Time Model Space
  - Beyond Observations: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning
  - Can we generate portable representations for clinical time series data using LLMs?
  - QuITE: Query-based Irregular Time-series Embedding
  - Generative Modeling of Irregular Time Series via SDE-Induced Continuous-Discrete Variational Inference
  - MTM: A Multi-Scale Token Mixing Transformer for Irregular Multivariate Time Series Classification
  - MedMamba: Multi-View State Space Models with Adaptive Graph Learning for Medical Time Series Classification
  - MedSpaformer: A Transferable Transformer with Multi-Granularity Token Sparsification for Medical Time Series Classification
  - MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling
  - StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Stars
  - Rethinking Large Language Models for Irregular Time Series Classification in Critical Care
  - Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing
  - Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction
  - Status-Aware Self-Supervised Forecasting for Irregular Clinical Time Series
  - LLM4EHR: Aligning Clinical Time Series with Medical Event Sequences via Large Language Models
  - Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series
  - VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / EHR event-stream / ICU time-series classification 的顶会或顶会相关论文，重点核对 ICLR 2026 OpenReview、ICML 2026、AAAI 2026、KDD 2025/2026、NeurIPS 2025 官方页面、OpenReview、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 VITAL/Mind the Missing 这类暂无顶会录用信息的 arXiv 预印本、FORMED/UniShape 这类分类强但不原生处理 irregular sampling 的工作、MIRA/Time-IMM 这类更偏 forecasting 或 benchmark-only 且分类主任务较弱的候选。本次保留全新工作 2 篇：PULSE 直接评估 ICU time-series classification 的跨中心泛化与 LLM/传统模型差异；Time-Conditioned Foreseeing (TCF) 是 ICML 2026 Poster，面向带不规则时间戳的 EHR 事件流预训练，并在多项下游临床预测/分类任务上验证。

## 1. PULSE: Benchmarking Large Language Models for ICU Time Series Classification

- 会议/状态：ICLR 2026 OpenReview submission
- 作者：Jan Berner, Sophia F. Ehlers, Miklovana Tuci, Tim Hahn, Catherine R. Jutzeler, Lakmal Meegahapola
- OpenReview：https://openreview.net/forum?id=e5OrurYW07
- 代码：https://github.com/lakmalbuddikalucky/pulse-benchmark
- 关键词：ICU time series classification, LLM benchmark, mortality prediction, sepsis prediction, acute kidney injury, cross-domain shift, HiRID, MIMIC-IV, eICU

### 场景、任务与核心难点

PULSE 面向重症监护场景中的高风险时序分类任务，统一评估 HiRID、MIMIC-IV 和 eICU 三个 ICU 数据源上的 mortality、sepsis 与 acute kidney injury 预测。该类任务的难点不仅是输入多变量、缺失多、观测频率随病情和流程变化，而且还包括跨医院部署时的分布偏移：不同 ICU 的变量定义、采样频率、治疗流程、告警阈值和记录习惯都会改变同一个临床终点在数据中的可见形态。

论文的核心问题不是提出一个新的 irregular encoder，而是建立一个可复现实验场来回答：当 ICU 时序分类从同院同分布切换到跨院、少标注或零标注设置时，传统机器学习、深度时序模型与 instruction-following LLM 各自的边界在哪里。PULSE 比较 17 类模型，包含 RandomForest、XGBoost、LightGBM、CNN、LSTM、GRU、InceptionTime 以及 Llama、Mistral、GPT-4o、Gemini、Claude、Grok 等 LLM。结果显示，within-domain 设置下 LightGBM 仍能达到很强的 AUROC，而跨域测试时传统模型和深度模型退化明显，LLM 的 zero-shot / few-shot prompting 与 hybrid reasoning workflow 则表现出更强的 day-zero 可用性。

### 审稿人视角：价值与不足

最有价值的贡献是把近期“LLM 能否处理 ICU 时序分类”的讨论从单点方法比较推进到多中心、多终点、多模型族的基准审计。对审稿人而言，PULSE 的价值在于它没有只报告一个 LLM prompt 的成功案例，而是把强传统基线放进同一框架，并明确指出在标准同分布训练中 LightGBM 这类模型仍然很难被取代；LLM 的优势主要出现在跨域、少数据和新机构冷启动场景。这种结论比“LLM 全面优于时序模型”更可信，也更贴近临床部署。

不足在于，PULSE 仍主要是 benchmark / evaluation 工作，对不规则采样机制本身的可控分解有限。为了跨数据源可比，许多 ICU benchmark 往往会经历窗口化、聚合、变量标准化或任务定义对齐；这些处理提升了实验可复现性，但也可能把原始异步事件流中的采样政策细节压缩掉。论文强调 distribution shift，却还没有系统拆出 shift 中有多少来自 patient mix、变量 schema、采样频率、医院流程或治疗策略。LLM 在跨域中的稳健性也可能来自强先验和文本化提示，而不一定说明它真正学会了 sampling-policy-invariant 的时序状态。

### 对 Sampling-Policy Shift 的启发

PULSE 对我们的问题有直接横向启发：sampling-policy shift 应该被放在跨中心 benchmark 中显式评估，而不是只在单数据集内随机 mask 或改变缺失率。HiRID、MIMIC-IV 和 eICU 的跨域组合天然提供了不同采样政策、不同护理流程和不同变量定义下的环境划分；如果一个模型在同院表现强但跨院 AUROC/AUPRC 大幅下降，就说明它可能依赖了训练医院的观测流程 shortcut。

纵向深化上，可以在 PULSE 框架中加入 policy-aware evaluation 层：为每个样本统计变量采样频率、联测模式、delta-t 分布、value-pending 情况和告警后密集观测窗口，并报告 state-only、policy-only 与 full-model 三组性能。进一步可以把 LLM 的 prompt / reasoning workflow 改造成双通道输出：一部分总结稳定病程状态，一部分显式总结观测政策和数据质量；分类决策只允许依赖前者，后者用于校准和偏移诊断。这样能把 PULSE 从“谁在跨域上更稳”推进到“为什么跨域稳定，以及稳定性是否来自正确隔离采样政策”。

## 2. Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time

- 简称：TCF / TCF-PFM
- 会议：ICML 2026 Poster
- 作者：Bong Gyun Kang, Junyong Ahn, Hyeongrok Han, Sungroh Yoon
- 官方页：https://icml.cc/virtual/2026/poster/64928
- OpenReview：https://openreview.net/forum?id=Z8Hu7CJfZy
- 代码：https://github.com/Pusheen-cat/TCF_PFM
- 关键词：EHR foundation model, irregularly sampled timestamps, calendrical time, temporal generative pretraining, downstream clinical prediction, AUPRC

### 场景、任务与核心难点

TCF 面向 EHR 事件流中的长程临床预测：患者轨迹由化验、生命体征、用药、诊疗事件等 token 组成，每个 token 都带有不规则时间戳。与普通文本不同，EHR 的数值范围有强临床语义，例如异常高低值比常规值更重要；时间也不只是 token 顺序，而同时包含绝对日历时间、住院进程中的相对间隔、事件之间的等待时间和多尺度未来窗口。下游评估覆盖 MIMIC-III 等数据上的 IHM、decompensation、LOS、phenotyping、oliguria/anuria、vasopressor use 等临床预测/分类任务。

论文解决的核心难点是：如何让 EHR foundation model 原生理解“何时发生”和“未来哪个时间点会发生什么”，而不是照搬 NLP 的 next-token prediction。作者提出三部分设计：Pathology-Focused Binning 用密度/病理意义导向的方式量化数值，避免把临床关键异常值淹没在常见区间中；Dual-Calendar RoPE 同时编码绝对时间与相对时间间隔；Time-Conditioned Foreseeing objective 则在预测下一事件时间的同时，给定未来时间条件去预测未来事件。这样预训练目标更接近医生做治疗规划和风险预判的方式，而不只是按 token 顺序续写病历。

### 审稿人视角：价值与不足

最有价值的技术思想是把 irregular EHR 的“时间条件性”放进 foundation model 的预训练目标，而不仅是把时间戳当作附加 embedding。Pathology-Focused Binning 处理数值语义，Dual-Calendar RoPE 处理日历与间隔，TCF objective 处理多时间窗 foreseeing，三者共同把 EHR 从普通离散文本序列拉回到连续、不规则、临床决策相关的时序对象。对审稿人而言，这比单纯扩大 EHR Transformer 参数量更有说服力，因为它针对 EHR 的关键结构错配给出了具体机制，并在多项下游任务上报告了 AUPRC 提升。

不足在于，TCF 仍然是生成式 EHR 预训练框架，采样政策与真实病程的分离还不充分。EHR 中“未来哪个事件会发生”往往同时由病人状态、医生怀疑、医院流程、检查可及性和记录习惯决定；因此 Time-Conditioned Foreseeing 学到的未来事件分布可能混合 patient state 与 care process。若某家医院更倾向于在高风险患者身上提前开某些化验或治疗，模型可能把这种策略当作稳定病程模式。论文展示了多任务预测收益和时间一致生成，但还需要跨医院、跨采样协议、反事实时间戳扰动下的系统评估，才能证明其时间表示不是 policy shortcut。

### 对 Sampling-Policy Shift 的启发

TCF 对 sampling-policy shift 的横向启发是：采样政策偏移不仅改变“观测是否出现”，还会改变模型预训练时要预测的未来事件时间分布。若预训练目标要求模型预测未来某个时间窗会出现哪些化验、诊疗或观测事件，那么它天然会学习医院如何观察病人；这既是信息来源，也是潜在 shortcut。因此，我们可以把 TCF 式 future-event likelihood 分解为 state likelihood 与 policy likelihood：前者描述真实病程会导致什么临床状态，后者描述医院会在何时记录或测量什么。

纵向深化上，可以设计 policy-robust TCF：Dual-Calendar RoPE 保留真实时间结构，但在 representation 中显式分出 state time embedding 与 policy calendar embedding；foreseeing objective 分成 future-state prediction 和 future-observation-process prediction 两个头。对同一患者潜在轨迹施加不同反事实采样策略时，要求 state foreseeing、下游 logits 和关键病理 bin 表示保持一致，同时允许 policy head 预测不同的观测密度、事件可见性和检查触发时间。这样能把 TCF 的时间条件预训练从“更好预测 EHR 未来事件”推进到“区分未来病程与未来观测政策”，更贴近我们要解决的 Sampling-Policy Shift。
