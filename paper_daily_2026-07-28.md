# Paper Daily - 2026-07-28

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / non-stationary irregular sequence classification 的顶会或顶会相关论文，重点核对 NeurIPS 2025、ICLR 2026、ICML 2026、AAAI 2026、ICASSP 2026、IEEE BigData 2025 官方/论文页、OpenReview、NeurIPS proceedings、arXiv 与代码仓库。
- 已排除全部黑名单论文；同时按实质去重排除与 SuperMAN/GMAN 同源的 `Interpretable Graph Learning on Irregular Clinical Time Series`，排除已标记 withdrawn 的 MedFuse、偏 forecasting/generation/imputation/RL 决策的候选，以及普通规则时序分类工作。本次保留全新工作 2 篇：STaRFormer 是 NeurIPS 2025 正会论文，虽然主问题是通用 sequential representation learning，但实验明确覆盖 P19/P12/PAM 等不规则采样分类；WaveGNN 是 IEEE BigData 2025 会议论文，直接面向不规则临床多变量时序分类，补充了图结构可解释性与采样策略偏移的视角。

## 1. STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data

- 会议：NeurIPS 2025 Poster
- 作者：Maximilian Forstenhausler, Daniel Kulzer, Christos Anagnostopoulos, Shameem Puthiya Parambath, Natascha Weber
- 官方页：https://neurips.cc/virtual/2025/poster/116860
- 论文：https://papers.neurips.cc/paper_files/paper/2025/file/1eb7e71099cbbeba47afbbeb0804e820-Paper-Conference.pdf
- 项目页：https://star-former.github.io/
- 代码：https://github.com/STaR-Former/starformer
- 关键词：irregularly sampled time series, non-stationary sequential data, semi-supervised contrastive learning, dynamic attention-based regional masking, P19/P12/PAM classification

### 场景、任务与核心难点

STaRFormer 面向更广义的 sequential data 表征学习，原始动机来自智能设备和车辆附近的用户意图预测，但论文把评估扩展到非平稳、异步/不规则采样、长短序列和多任务场景。其中与我们最相关的是 irregularly sampled time series classification：作者在 P19、P12 和 PAM 上比较 Transformer、GRU-D、SeFT、mTAND、IP-Net、DGM2-O、Raindrop、ViTST 等基线，任务覆盖脓毒症早期预测、ICU 死亡/住院结局和人体活动分类。

这类任务的核心难点不只是观测时间不均匀，而是“任务相关片段”与“采样/非平稳噪声”经常混在一起。普通自监督时序表征往往随机 mask 或做任务无关增强，容易强化与最终分类边界无关的局部重建能力；完全监督训练又容易在小样本、类别不平衡或采样模式变化时过拟合。STaRFormer 因此提出 Dynamic Attention-based Regional Masking (DAReM)，根据注意力/任务信号动态选择区域进行扰动，并结合半监督对比学习，使 representation learning 更贴近下游分类目标，而不是只追求通用重构或全局平滑。

### 审稿人视角：价值与不足

最有价值的思想是把“数据增强/遮蔽策略”从静态随机机制改成 task-informed 机制。对不规则分类而言，哪些时间段被 mask、哪些区域被增强，会直接影响模型学习到的是状态模式还是采样伪迹；DAReM 至少让这种选择与判别目标发生联系。论文在 56 个不同数据集上评估，并在 P19/P12/PAM irregular benchmarks 上取得稳定提升，说明它不是只针对单一医疗数据集调参的专用模型，而是一种可迁移的序列表征训练范式。

不足在于，STaRFormer 并不是专门为 asynchronous multivariate clinical time series 设计的结构化模型。它证明 dynamic masking 和半监督对比学习对不规则采样数据有帮助，但没有显式建模变量级时间戳、观测政策、医院流程、informative missingness 或跨机构 sampling policy。DAReM 依赖模型注意力来决定重要区域，而注意力本身可能已经受到训练环境采样密度、标注流程和传感器可见性的影响；如果某类样本在训练策略下更容易被密集观测，task-informed masking 可能会强化这种 shortcut。

### 对 Sampling-Policy Shift 的启发

STaRFormer 对 Sampling-Policy Shift 的横向启发是：采样策略偏移不仅改变输入分布，也会改变“应当遮蔽/增强哪些区域”的训练分布。若一个增强策略总是围绕训练环境中的高频观测区、告警后密集窗口或易被标注片段展开，模型学到的 task-informed representation 仍可能绑定采样政策。因此，DAReM 的区域选择概率、被遮蔽窗口的变量组成、增强后 logits 变化，可以作为 sampling-policy shortcut 的诊断信号。

纵向深化上，可以把 STaRFormer 改造成 policy-aware regional masking：state masking branch 选择跨策略稳定的状态关键区域，用于分类一致性和半监督对比；policy masking branch 选择由观测流程、设备可见性或医院协议诱导的区域，用于策略识别和偏移诊断。对同一潜在轨迹构造不同采样策略视图时，要求 state-aware masked representation 与 logits 保持一致，同时允许 policy-aware masking map 变化。这样能把“任务知情增强”推进到“策略不变增强”，避免把采样政策下的易观测区域误当作稳定类别证据。

## 2. WaveGNN: Integrating Graph Neural Networks and Transformers for Decay-Aware Classification of Irregular Clinical Time-Series

- 会议：IEEE BigData 2025；Best Student Paper Runner-up
- 作者：Arash Hajisafi, Maria Despoina Siampou, Bita Azarijoo, Zhen Xiong, Cyrus Shahabi
- 论文：https://arxiv.org/abs/2412.10621
- DOI：https://doi.org/10.1109/BigData66926.2025.11401906
- 代码：https://github.com/USC-InfoLab/WaveGNN
- 关键词：irregular clinical time series, decay-aware Transformer, dynamic sample-specific graph, no imputation, P12/P19/MIMIC-III/PAM classification

### 场景、任务与核心难点

WaveGNN 面向临床多变量时序分类，输入来自 P12、P19、MIMIC-III 和 PAM 等 benchmark：每个传感器/变量有自己的观测时间，观测间隔不均，变量之间常常不同步，并且存在大量缺失。任务包括 ICU 结局预测、脓毒症/表型分类和活动识别等，目标是在不插值、不重采样到统一网格的情况下，直接从不规则多变量轨迹预测类别。

论文把难点拆成两类：intra-series irregularity 与 inter-series discrepancy。前者要求模型知道同一变量的旧观测何时过期、不同时间间隔如何影响当前状态；后者要求模型在变量不同步时仍能学习心率、血压、体温、化验指标等跨变量依赖。WaveGNN 用带相对时间编码和可学习 decay 的 Transformer 编码每个变量内部动态，再用动态 GNN 为每个样本构造一张稀疏变量图，融合短期局部相似性和全局可学习变量关系，最后通过图级表征做分类。

### 审稿人视角：价值与不足

最有价值的技术思想是把“每个变量内部的时间衰减”和“变量之间的可解释图关系”放进同一个端到端分类器。许多不规则时序方法要么主要处理单变量时间间隔，要么依赖插值后再做跨通道 attention；WaveGNN 保留原始不规则观测，并输出每个样本一张相对稳定、稀疏、可解释的传感器交互图。对临床任务而言，这比只给出 attention heatmap 更容易检查模型是否学到了合理的生理关系。

不足在于，WaveGNN 的动态图仍可能吸收采样政策。短期局部相似性、缺失 mask、最近观测的 decay 权重，以及全局变量嵌入都会受到医院化验流程、设备记录频率和告警触发机制影响。论文在不同 benchmark 与模拟缺失率下展示了鲁棒性，但这些实验主要是同一数据生成/划分机制内的缺失扰动；它还没有系统评估跨医院、跨科室、跨监测协议或主动采样策略变化后，learned graph 和 decay 参数是否保持语义稳定。

### 对 Sampling-Policy Shift 的启发

WaveGNN 对 Sampling-Policy Shift 的横向启发是：策略偏移可以同时体现在变量内 decay 曲线和变量间图结构上。不同采样政策会改变“多久未测”代表的临床语义，也会改变哪些变量经常在相邻时间窗内共同出现；因此，decay rate、short-term edge、global edge、graph sparsity 和 edge stability 都可以成为采样策略偏移的可观测诊断量。

纵向深化上，可以把 WaveGNN 扩展为 state-policy 双图模型：state graph 学习跨策略稳定的生理变量关系和真实动态衰减，进入分类主路径；policy graph 学习由记录制度、成本约束、告警触发和传感器可用性造成的观测关系，只用于不确定性校准或偏移解释。训练时对同一潜在患者轨迹生成不同反事实采样策略，约束 state graph 表征和分类 logits 稳定，同时允许 policy graph 与 decay residual 变化。这样能把 WaveGNN 的“可解释临床图”推进到“可检验哪些图边可跨采样政策迁移”。
