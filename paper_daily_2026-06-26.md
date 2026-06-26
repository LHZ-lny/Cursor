# Paper Daily - 2026-06-26

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / clinical irregular time series classification 的顶会或顶级方向论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、ICASSP 2026、OpenReview、ICML virtual site、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除 Hi-Patch 这类 ICML 2025 时间窗口偏旧条目、偏 forecasting 或普通规则时序分类的条目。本次保留全新工作 2 篇：StarEmbed 是 ICML 2026 正会 benchmark，虽然领域是天文光变曲线而非医疗 IMTS，但其多波段、异步、异方差和跨观测策略问题对 sampling-policy shift 很有横向价值；Rethinking LLMs 是 ICASSP 2026 会议论文，直接评估 LLM 在 ICU 不规则时序分类中的有效性与局限。

## 1. StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Stars

- 会议：ICML 2026 Poster
- 作者：Weijian Li, Hong-Yu Chen, Nabeel Rehemtulla, Ved Shah, Dongho Kim, Dennis Wu, Qinjie Lin, Adam Miller, Han Liu
- 官方页：https://icml.cc/virtual/2026/poster/63677
- 论文：https://arxiv.org/html/2510.06200
- 项目页：https://hibb-bb.github.io/star-embed.github.io/
- 代码：https://github.com/skai-institute/StarEmbed
- 关键词：irregular astronomical time series, light-curve classification, foundation model benchmark, multi-band observations, heteroskedasticity, OOD detection

### 场景、任务与核心难点

StarEmbed 面向天文时域观测中的变量星光变曲线分析。输入来自 Zwicky Transient Facility 的多波段观测：同一恒星在不同滤波波段上的亮度记录并不同步，采样间隔跨越多个数量级，还伴随天气、昼夜、观测几何和仪器条件导致的异方差测量误差。论文构建约 4 万条专家标注的多波段光变曲线，覆盖 7 类天体，并评估无监督聚类、监督分类和 OOD 源检测。

核心难点在于，现有通用 time-series foundation models 通常预训练在规则采样、低噪声或单变量商业/交通/能源数据上，而天文光变曲线是极端 out-of-domain 的不规则科学时序。论文不是提出新的 irregular encoder，而是建立一个标准化 benchmark，检验 Moirai、Chronos、Chronos-Bolt、Time-MoE 等通用 TSFM 以及 Astromer 这类领域模型的 zero-shot embedding 能否支撑变量星分类与异常源发现。结果显示，手工特征在监督分类上仍很强，但 Chronos 系列在聚类和 OOD 检测上表现突出，说明通用基础表征对不规则科学时序具备一定迁移性。

### 审稿人视角：价值与不足

最有价值的贡献是把“不规则采样分类”从医疗/传感器 benchmark 横向扩展到一个真实、规模大、采样机制复杂且有强科学约束的领域。StarEmbed 的意义不只是给出一个新数据集，而是提供固定 split、专家标签、公开 embedding 与多任务评估，使研究者可以检验基础模型在从未见过的采样制度下是否仍能形成可分表征。它也提醒我们，classification accuracy 不是唯一目标；对未来大规模巡天而言，OOD detection 和 embedding quality 同样关键。

不足在于，它主要是 benchmark/评测论文，针对 irregular sampling 的模型机制创新有限。监督分类中手工特征仍优于多数 TSFM，说明通用 foundation embedding 尚未充分捕捉天文领域的周期、相位折叠和异方差误差结构。论文还没有把观测策略本身作为可控变量系统评估，例如不同巡天 cadence、天气缺测、波段调度或训练/测试 split 中采样密度差异如何影响分类边界。换言之，它证明了 TSFM 的跨域潜力，但还未回答哪些表示对观测策略稳定、哪些只是适配了 ZTF 的特定观测政策。

### 对 Sampling-Policy Shift 的启发

StarEmbed 对 sampling-policy shift 的横向启发是：采样政策偏移不只存在于 ICU 或传感器网络，也天然存在于天文巡天。观测 cadence、昼夜窗口、天气缺测、波段选择和测量不确定性共同决定一条光变曲线的可见形态；同一类变量星在 ZTF、LSST 或其他巡天中的采样政策变化，可能改变周期特征可恢复性、相位覆盖度和 OOD 分数。

纵向深化上，可以把 StarEmbed 扩展为 policy-shift benchmark：在同一底层光变曲线或物理模板上生成多种观测 cadence，要求 state embedding 保持天体类别语义稳定，同时允许 policy embedding 解释采样间隔、波段覆盖和不确定性分布。对我们的问题，这提示可以把 sampling-policy shift 量化为“任务相关频率/相位信息的可恢复性变化”，而不只是 mask ratio 或 delta-t 的统计偏移。若一个模型在不同观测政策下分类 logits 稳定但 OOD/policy score 能正确反映采样质量，就更接近真正可迁移的不规则时序分类器。

## 2. Rethinking Large Language Models for Irregular Time Series Classification in Critical Care

- 会议：ICASSP 2026
- 作者：Feixiang Zheng, Yu Wu, Cecilia Mascolo, Ting Dang
- 论文：https://arxiv.org/html/2601.16516
- 会议记录：https://findanexpert.unimelb.edu.au/scholarlywork/2306561-rethinking-large-language-models-for-irregular-time-series-classification-in-critical-care
- 代码：https://github.com/mHealthUnimelb/LLMTS
- 关键词：irregular ICU time series classification, LLM for time series, encoder design, multimodal alignment, mortality prediction, few-shot learning

### 场景、任务与核心难点

这篇工作面向 ICU 不规则时序分类，主要任务是基于 PhysioNet 2012 和 MIMIC-III 等重症监护记录进行院内死亡风险预测，同时用半合成不规则 ECG 检验缺失率变化下的鲁棒性。ICU 数据的典型难点是变量多、缺失率高、采样异步且受临床流程触发：生命体征、化验、用药和记录频率都可能随病情和医院协议变化。

论文要回答的问题很直接：当前 LLM-based time-series 方法在规则数据上看似强大，但是否真的适合不规则 ICU 分类？作者系统比较 Time-LLM、S2IP、CALF、FSCA 等 LLM 方法，以及 MOMENT、UniTS、mTAND、Warpformer 等自监督/监督 baseline，并拆解两个关键组件：time-series encoder 与 multimodal alignment strategy。实验发现，显式处理不规则性的 mTAND encoder 对性能影响远大于 alignment 策略；单纯把规则时序 LLM 框架迁移到 ICU irregular data 会明显退化；LLM 方案通常带来 10 倍量级训练开销，却只得到有限甚至不稳定收益。

### 审稿人视角：价值与不足

最有价值的地方是，它没有继续假设“LLM 一定能靠语义能力解决时序不规则性”，而是做了针对 ICU irregular classification 的组件级审计。结论非常有用：encoder 是否尊重真实时间戳、缺失和异步观测，比后端 LLM alignment 的花哨程度更关键。对审稿人而言，这类负结果/边界结果很重要，因为它约束了近期把 LLM 套到所有时序任务上的趋势，也为后续方法设计提供了清晰优先级。

不足在于，这篇工作仍以经验比较为主，缺少对采样机制的因果分解。它证明了不规则感知 encoder 重要，也证明 LLM few-shot 并不自动占优，但还没有系统区分“状态驱动的 informative sampling”和“医院政策驱动的 sampling shortcut”。半合成 ECG 通过随机 drop 模拟不规则性，能测 missing ratio robustness，却不能充分代表 ICU 中由医生决策、检查协议和病情触发共同形成的策略性采样。论文也主要在死亡风险分类上评估，尚未覆盖跨医院采样政策迁移或反事实采样协议。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：大模型或语言对齐不是解决采样策略偏移的捷径。若前端 encoder 已经把 irregularity 处理成错误的 patch、错误的局部上下文或规则时间假设，后端 LLM 再强也只能在污染表征上做推理。因此，我们在设计 policy-robust 模型时，应优先保证输入层和 encoder 能显式表达观测时间、变量级 mask、delta-t、测量不确定性与环境/政策标签。

纵向深化上，可以借鉴论文的组件审计范式，把采样策略偏移拆成 encoder sensitivity 与 alignment sensitivity 两层评估：先固定分类头，测试不同采样政策下 state encoder 的表征稳定性；再测试 policy-aware alignment 是否把采样模式仅用于校准和不确定性，而不是直接作为类别证据。对 LLM 路线而言，更合理的方向不是把 ICU 序列粗暴转成文本，而是构建 state encoder + policy encoder + semantic alignment 的三分支结构：state branch 学跨策略稳定病程，policy branch 学测量流程和缺失机制，LLM 或语义模块只在两者解耦后做可解释摘要与决策支持。
