# Paper Daily - 2026-06-22

## 追加更新 - 2026-06-22 23:03 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：围绕近 3-7 个月内 irregular sampled / asynchronous / irregular multivariate time-series classification 的顶会论文，重点核对 NeurIPS 2025、AAAI 2026、ICLR 2026、ICML 2026、KDD 2025/2026 官方页面、OpenReview、AAAI Proceedings、ICML/NeurIPS virtual site 与 arXiv/代码页。
- 已排除全部黑名单论文；SITS 因为是 ICML 2025 workshop 条目而非正会排除，CoGRUODE 因为是 ICLR 2023 withdrawn submission 排除；严格 direct-hit 的正会分类论文大多已被历史日报覆盖。本次保留 2 篇未在黑名单中的全新 NeurIPS 2025 工作：DiPro 是稀疏异步临床多模态预测/分类的直接近邻；Time-IMM 以 forecasting benchmark 为主，但其 cause-driven irregularity 数据集对构造异步分类与 Sampling-Policy Shift 评测具有基础设施价值。

## 14. Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment

- 简称：DiPro
- 会议：NeurIPS 2025 Spotlight
- 作者：Chen Liu, Wenfang Yao, Kejing Yin, William K. Cheung, Jing Qin
- OpenReview：https://openreview.net/forum?id=2afhRWVb6p
- 官方页：https://neurips.cc/virtual/2025/poster/120126
- 论文：https://arxiv.org/html/2510.11112
- 代码：https://github.com/Chenliu-svg/DiPro
- 关键词：longitudinal multimodal fusion, sparse irregular CXR, asynchronous EHR, disease progression identification, ICU prediction

### 场景、任务与核心难点

DiPro 面向纵向多模态临床预测：同一 ICU 患者既有连续或高频的 EHR 时序，也有稀疏、非规则时间点采集的胸片序列。下游任务包括疾病进展识别、住院死亡风险预测和 ICU length-of-stay 分类/预测。这与典型 irregular time series classification 的区别在于，它不是单一多变量表格时序，而是稀疏影像快照与异步 EHR 信号的融合分类问题。

论文解决的核心难点有两层。第一，连续胸片中大量静态解剖结构会淹没真正与疾病进展相关的动态病理变化；第二，CXR 是稀疏事件式采集，EHR 则是异步、多频率的临床时序，两者很难在统一时间网格上无损对齐。DiPro 因此先做 region-aware spatiotemporal disentanglement，将静态 anatomy 与动态 pathology progression 分离；再用 progression-aware enhancement 通过时间反转约束强化进展方向；最后用 multiscale multimodal fusion 在局部 interval-level 和全局 sequence-level 同步 CXR 与 EHR。

### 审稿人视角：价值与不足

最有价值的思想是把“异步临床时序分类”从单纯补齐时间戳推进到跨模态语义对齐。DiPro 没有把稀疏 CXR 简单插值成连续影像序列，而是承认影像模态本身是事件式、低频、强冗余的；其静态/动态解耦让模型优先关注病理变化而非个体解剖常量，local-global alignment 又让 EHR 的连续病程信息与影像快照之间形成多尺度桥接。这对医疗场景非常有意义，因为真实诊断往往正是由“稀疏检查 + 连续监护”的异步组合构成。

不足在于，DiPro 对采样机制的建模仍是隐式的。CXR 何时拍摄、EHR 哪些变量被记录，往往由医生怀疑、医院流程、设备可用性和患者严重程度共同决定；因此多尺度对齐模块可能学到的不只是病理进展，也包括“什么情况下会被安排影像检查”的策略性信号。论文在 MIMIC 数据上展示了疾病进展和 ICU 任务的收益，但尚未系统评估跨医院、跨拍片协议、跨监测频率下，动态病理表征和融合注意力是否保持稳定。

### 对 Sampling-Policy Shift 的启发

DiPro 对我们的问题提供了一个横向扩展方向：Sampling-Policy Shift 不仅发生在数值时序的 mask 和 delta-t 上，也发生在多模态检查触发机制上。可以把 CXR 拍摄、实验室检查、生命体征记录都视为由 policy 生成的 observation events；模型需要区分“状态变化导致检查出现”和“制度流程导致检查出现”。

纵向深化上，可以借鉴 DiPro 的 disentanglement 思路，把表征拆成 state-progression component 与 policy-trigger component。前者描述跨采样策略稳定的疾病进展，进入分类主路径；后者解释为什么某个模态在某个时间被采样，用于偏移诊断和不确定性校准。local-global alignment 也可扩展为 policy-counterfactual alignment：对同一潜在病程模拟不同检查调度，要求 state progression 表征和 logits 稳定，同时允许 trigger/policy 表征变化。这样能把异步多模态融合从“对齐观测”推进到“对齐跨策略不变的病程语义”。

## 15. Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series

- 会议：NeurIPS 2025 Datasets and Benchmarks Track
- 作者：Ching Chang, Jeehyun Hwang, Yidan Shi, Haixin Wang, Wei Wang, Wen-Chih Peng, Tien-Fu Chen
- 官方页：https://neurips.cc/virtual/2025/poster/121380
- 论文：https://proceedings.neurips.cc/paper_files/paper/2025/hash/4199594d3c15736df2bf5274fa3155f4-Abstract-Datasets_and_Benchmarks_Track.html
- OpenReview：https://openreview.net/forum?id=yeqrrn51TL
- 代码/数据：https://github.com/blacksnail789521/Time-IMM
- 关键词：irregular multimodal multivariate time series, cause-driven irregularity, trigger-based sampling, constraint-based sampling, artifact-based missingness, asynchronous multimodal benchmark

### 场景、任务与核心难点

Time-IMM 不是一篇以分类精度提升为目标的模型论文，而是一个面向真实不规则多模态多变量时序的数据集与 benchmark。它覆盖 healthcare、climate、finance、repository health、traffic/network、student life 等场景，强调现实数据的不规则性并非同一种缺失噪声，而是来自不同成因：事件触发、资源/制度约束、系统故障或多源异步。

论文当前配套的 IMM-TSF 主要服务 irregular multimodal time-series forecasting，但作者明确指出同类真实数据也需要 anomaly detection、classification 和 retrieval 等下游能力。对异步时序信号分类而言，Time-IMM 的核心难点贡献在于：它把不规则性按 cause-driven taxonomy 标注和组织，而不是只给出 mask ratio、采样频率或插值前后的误差。这使研究者能够区分“因为状态变化而采样”“因为运营窗口限制而采样”“因为系统故障而缺失”等机制。

### 审稿人视角：价值与不足

最有价值的思想是将 irregularity 从数据表面统计推进到生成原因层面。现有不规则时序分类 benchmark 往往只告诉我们哪些点缺失、时间间隔多长，却不告诉我们缺失或异步的来源。Time-IMM 把 irregularity 分成 trigger-based、constraint-based 和 artifact-based 三大类、九种子类型，并为每条时序配套文本模态和异步融合工具，这为研究“采样机制本身是否稳定”提供了比传统 benchmark 更清晰的实验基座。

不足也很明确：当前正式 benchmark 重点是 forecasting，而不是分类；IMM-TSF 的模型、指标和实验设计还没有直接覆盖异步分类的 label leakage、class imbalance 或 cross-policy generalization。多模态文本描述有助于解释不规则性来源，但也可能引入 LLM 生成摘要的语义偏差。另一个限制是，cause-driven 标签多在数据集层面或机制层面组织，未必已经细到每个样本、每个变量、每次观测事件的 policy annotation。

### 对 Sampling-Policy Shift 的启发

Time-IMM 对 Sampling-Policy Shift 的启发非常直接：我们的 benchmark 不应只模拟随机 mask 或改变采样率，而应显式构造不同 sampling policy causes。比如 trigger-based policy 对应“病情恶化才测”、constraint-based policy 对应“只有固定窗口/资源充足时测”、artifact-based policy 对应“设备延迟或数据管道故障”。这三类偏移对分类器的污染方式不同，应该分别评估。

纵向深化上，可以在 Time-IMM 的 taxonomy 之上构建 classification-specific protocol：为同一潜在状态轨迹生成多种观测政策视图，报告 in-policy、cross-policy 和 counterfactual-policy 指标；同时记录变量级观测原因、模态级延迟、采样窗口和缺失触发条件。模型层面可设计 policy-cause encoder，把 trigger/constraint/artifact 表征作为独立支路，用于偏移诊断和校准，而分类主支路通过跨 policy-cause consistency 学习稳定状态表征。这样可以把 Sampling-Policy Shift 从“经验上的采样分布变了”转化为“可标注、可反事实、可分层评估的观测机制变化”。
