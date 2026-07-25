# Paper Daily - 2026-07-25

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multimodal time series / medical time series classification 的顶会论文，重点核对 NeurIPS 2025、ICML 2026、ICLR 2026、AAAI 2026、KDD 2025/2026、OpenReview、NeurIPS proceedings、ICML virtual site、AAAI Proceedings、arXiv 与论文/代码页。
- 已排除全部黑名单论文；同时排除 TimeCHEAT、KEDGN、GNeuralFlow 这类时间窗口偏旧条目，排除 TRIAGE 这类当前仅有 arXiv/preprint 且未确认顶会接收的条目，排除 MIRA 这类 NeurIPS 2025 正会但目前评估明确以 forecasting 为主、尚不支持 classification 的条目，以及 Time-CoT、TimeSliver、MIX、ShapeX 等普通规则时序分类或解释工作。严格直接命中“全新正会 + 不规则采样 + 分类主任务”的候选已很少；本次保留全新工作 1 篇：Time-IMM 是 NeurIPS 2025 Datasets and Benchmarks Track 正会论文，虽然配套库当前以 forecasting 为主，但它系统定义 cause-driven irregularity，并覆盖 healthcare 中的 trigger-based sampling，对构建 sampling-policy shift 下的分类评测基座非常有价值。

## 1. Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series

- 会议：NeurIPS 2025 Datasets and Benchmarks Track Poster
- 作者：Ching Chang, Jeehyun Hwang, Yidan Shi, Haixin Wang, Wei Wang, Wen-Chih Peng, Tien-Fu Chen
- 官方页：https://neurips.cc/virtual/2025/poster/121380
- OpenReview：https://openreview.net/forum?id=yeqrrn51TL
- 论文：https://proceedings.neurips.cc/paper_files/paper/2025/file/4199594d3c15736df2bf5274fa3155f4-Paper-Datasets_and_Benchmarks_Track.pdf
- 代码：https://github.com/blacksnail789521/Time-IMM 和 https://github.com/blacksnail789521/IMM-TSF
- 关键词：irregular multimodal multivariate time series, cause-driven irregularity, asynchronous textual annotations, trigger-based sampling, constraint-based sampling, artifact-based missingness, benchmark

### 场景、任务与核心难点

Time-IMM 面向真实世界中的不规则、多模态、多变量时序。它覆盖 healthcare、climate、finance、network telemetry、social sensing 等场景：数值时间序列本身存在非均匀采样、异步模态、缺失与不同采样频率，同时每条序列还可能伴随临床 notes、传感器日志、事件描述或新闻文本等异步文本信息。论文当前的配套库 IMM-TSF 以 forecasting 为主任务，而不是直接提供序列级分类 benchmark；但它明确指出 anomaly detection、classification、retrieval 是后续自然扩展，因此对异步时序分类的价值主要在数据机制与评测设计层面。

核心难点在于，现有 benchmark 通常把 irregularity 简化成随机缺失或固定 mask ratio，无法反映真实系统中“为什么被观测”的差异。Time-IMM 把不规则性拆成 9 类 cause-driven irregularity，并归纳为 trigger-based、constraint-based 和 artifact-based 三大机制。例如医疗记录中的 clinician-recorded observations 更接近病情和流程触发，金融数据受交易窗口约束，传感器数据可能来自设备延迟或采集故障。这个划分把异步采样从表面统计问题提升为观测机制问题：同样是缺失或异步，背后的策略语义可能完全不同。

### 审稿人视角：价值与不足

最有价值的贡献是给不规则多模态时序建立了“由原因定义的不规则性”评测语言。对审稿人而言，这比再报告一个平均误差或平均分类准确率更基础：如果 benchmark 不区分 trigger、constraint、artifact，模型到底是在学习真实状态、资源约束、流程习惯还是采集故障就很难判断。Time-IMM 还把异步文本作为正式模态纳入，而不是只处理数值矩阵，这更接近 EHR、运维日志和社会事件流中的真实输入形态。

不足也很清楚：论文当前主实验和库设计偏 forecasting，尚未形成成熟的 irregular time series classification protocol。它提供 cause-driven irregularity taxonomy，但没有为每类机制配套分类标签、跨策略 split、反事实采样协议或 policy-only probe。部分文本模态依赖自动摘要/语义过滤，也可能引入额外生成偏差。换言之，它是很好的 benchmark 基座，但还不是直接检验“采样策略变化下分类器是否稳健”的完整答案。

### 对 Sampling-Policy Shift 的启发

Time-IMM 对 Sampling-Policy Shift 的横向启发非常直接：采样策略偏移不应只用 missing rate、delta-t 分布或变量共现率描述，而应先区分观测机制属于 trigger、constraint 还是 artifact。医疗场景中的医生触发化验、设备报警后高频测量和资源约束下的延迟记录，分别会诱导不同类型的 shortcut；如果把它们混在一个 mask embedding 中，分类模型很容易把医院流程当成病理状态。

纵向深化上，可以把 Time-IMM 扩展为 policy-shift classification benchmark：在每个数据集中增加序列级或事件级标签，按 irregularity cause、机构/设备、采样密度分位数和文本可用性构造环境划分；同时为同一潜在序列生成 trigger-based、constraint-based、artifact-based 的反事实采样视图。训练目标上，可以设置 state encoder、policy encoder 和 multimodal fusion 三个分支：state encoder 需要在不同采样机制下保持分类 logits 稳定，policy encoder 专门识别采样机制和观测质量，fusion 模块则只允许策略信息用于校准与不确定性，而不能直接成为类别证据。这样能把 Time-IMM 的 cause-driven irregularity taxonomy 纵向推进为针对 Sampling-Policy Shift 的可操作评测协议。
