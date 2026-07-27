# Paper Daily - 2026-07-27

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / informative missingness 的顶会或顶会相关论文，重点核对 ICLR 2026 OpenReview、ICML 2026、AAAI 2026、ICASSP 2026、ACL 2026、NeurIPS 2025 官方或论文页面与 arXiv。
- 已排除全部黑名单论文；同时排除已被历史日报标记过的 withdrawn 候选、偏 forecasting/generation/imputation 的条目，以及与分类任务关系较弱的 benchmark-only 工作。本次保留全新工作 2 篇：STAR-Set 直接面向异步临床时间序列的 ICU 预测/分类式任务，且在摘要中明确指出 sampling-policy shortcuts；VP-GNN 是 ICASSP 2026 不规则临床时序图建模论文，任务覆盖院内死亡预测与早期脓毒症检测。

## 1. Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series

- 简称：STAR-Set Transformer
- 会议/状态：ICLR 2026 OpenReview submission / Research track；arXiv 2026-03
- OpenReview：https://openreview.net/forum?id=AxXNor3Kd2
- arXiv：https://arxiv.org/abs/2603.06605
- 关键词：asynchronous clinical time series, point-set tokenization, temporal attention bias, variable-type affinity, ICU prediction, sampling-policy shortcuts

### 场景、任务与核心难点

STAR-Set 面向 ICU/EHR 中异步、多变量、不规则采样的临床时间序列预测任务，实验覆盖 CPR、mortality 和 vasopressor use 等 ICU outcome。输入不是规则网格上的完整矩阵，而是由变量类型、时间戳和值组成的观测事件集合；这种 point-set tokenization 可以避免插值和填补，但会丢掉规则网格天然暴露的两类结构：同一变量随时间演化的列结构，以及相近时间内不同变量之间的行结构。

论文解决的核心难点是：如何在保留事件集合灵活性的同时，把这些结构先验重新注入 Transformer。作者在 attention logits 中加入两类轻量 soft bias：一类是 temporal locality penalty，用可学习时间尺度鼓励时间上相近的事件互相关注；另一类是 variable-type affinity，用可学习变量兼容矩阵恢复变量间或同变量内的结构偏好。这样模型不需要把 EHR 强行离散化到固定时间网格，也不需要完全依赖普通 self-attention 自己从稀疏事件中恢复临床结构。

### 审稿人视角：价值与不足

最有价值的思想是把“事件集合表示”和“网格结构归纳偏置”之间的矛盾处理得很简洁。许多 EHR 模型要么使用 grid，保留 time-by-variable 结构但引入 imputation/mask shortcut；要么使用 set token，避免填补却牺牲变量轨迹和局部共现。STAR-Set 用参数量很小的 attention bias 在两者之间折中，且 learned timescale 和 variable affinity 能提供一定解释性，帮助审稿人检查模型究竟依赖哪些时间邻域和变量关系。

不足在于，它仍主要在同一数据来源和任务划分下验证 supervised ICU prediction。虽然论文明确意识到 observation patterns 同时反映 physiology 和 care process，且可能产生 sampling-policy shortcuts，但现有实验还没有把医院、科室、监测协议或主动测量策略作为显式环境变量。temporal bias 和 variable affinity 可能恢复了有用结构，也可能恢复了训练医院特定的联测习惯、化验频率和护理流程。

### 对 Sampling-Policy Shift 的启发

STAR-Set 对我们的横向启发是：sampling-policy shift 可以被视为 attention bias 所刻画的结构先验发生偏移。不同采样策略不仅改变 mask ratio 或 delta-t 分布，也会改变哪些变量在局部时间窗内共现、同一变量的有效时间尺度以及变量兼容矩阵的稳定性。因此，learned timescale、variable affinity 和 attention-bias contribution 可以成为采样策略偏移的诊断指标。

纵向深化上，可以把 STAR-Set 改造成 policy-aware bias Transformer：一组 state bias 学习跨策略稳定的生理时间尺度和变量关系，进入分类主路径；另一组 policy bias 学习医院流程、记录习惯和资源约束诱导的局部共现，只用于偏移诊断或校准。对同一潜在病程构造不同采样策略视图时，可约束 state bias、state representation 和 logits 保持一致，同时允许 policy bias 区分不同观测流程。这样能把“恢复网格先验”推进到“区分哪些结构先验可跨采样政策迁移”。

## 2. VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series

- 会议：ICASSP 2026
- 作者：Yurong He, Boya Zhang, Longfei Liu, Guosheng Cui, Dan Wu
- DOI：https://doi.org/10.1109/ICASSP55912.2026.11461796
- 官方记录：https://cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=18058
- 代码：https://github.com/bursonz/VP-GNN
- 关键词：irregular clinical time series, variable-wise graph, patch-wise graph, selective message passing, in-hospital mortality prediction, early sepsis detection

### 场景、任务与核心难点

VP-GNN 面向不规则临床时间序列中的风险预测和疾病状态评估，实验使用 PhysioNet 2012 与 Sepsis 2019 等 EHR benchmark，任务包括院内死亡预测和早期脓毒症检测。此类数据具有稀疏观测、变量异步、采样间隔不均以及变量依赖异质等典型特点；如果直接重采样到统一网格，会扭曲真实观测时间，也会把缺失和观测密度的语义混在补齐值中。

论文的核心做法是用一个统一图框架同时处理 variable-wise 与 patch-wise 两层结构。第一阶段通过 selective message passing 捕捉变量之间的异步依赖，避免所有变量在所有时刻做无差别交互；第二阶段用 hierarchical Patch-GNN 聚合多尺度时间片段，建模局部到全局的临床动态。这样模型既能学习“哪些变量之间在异步观测下仍然相关”，也能学习“哪些时间片段组合成可判别的风险模式”。

### 审稿人视角：价值与不足

最有价值的技术思想是把临床 IMTS 的结构拆成变量图和时间 patch 图两种互补视角。相比只在变量维做静态图，VP-GNN 通过 patch-wise 层补上多尺度时间演化；相比只做时间 patch 或 Transformer token mixing，variable-wise message passing 又让变量依赖更显式。对死亡预测和早期脓毒症检测这类任务而言，这种“变量依赖 + 多尺度片段”的结构归纳偏置比单纯扩大模型容量更有针对性。

不足在于，图结构仍可能吸收采样政策。selective message passing 选择哪些变量交互，Patch-GNN 选择哪些时间片段聚合，都会受到训练数据中观测频率、告警触发、化验流程和数据集采样协议影响。论文证明了在 P12/P19 benchmark 上性能提升，但还没有系统回答跨医院或跨测量策略时，变量边、patch 重要性和层级聚合路径是否保持稳定。

### 对 Sampling-Policy Shift 的启发

VP-GNN 对 sampling-policy shift 的横向启发是：策略偏移可以沿着“变量交互图”和“时间 patch 图”两条路径进入分类器。某家医院频繁联测的变量可能在 variable-wise graph 上形成强边；某种告警后密集采样的窗口可能在 patch-wise graph 上成为高权重片段。若这些结构随医院流程改变而改变，模型就可能依赖采样政策 shortcut。

纵向深化上，可以把 VP-GNN 扩展为 state-policy 双图框架：state graph 学习跨策略稳定的变量耦合和病程片段，policy graph 学习观测流程诱导的变量联测、采样密度和 patch 可见性。训练时对同一患者轨迹施加不同反事实采样策略，约束 state graph 表征和分类 logits 稳定，同时允许 policy graph 解释采样流程差异。还可以把 graph edge stability、patch selection frequency 和 cross-policy graph distance 作为评估指标，用来判断模型是否真正解决了非规则采样下的采样策略偏移。
