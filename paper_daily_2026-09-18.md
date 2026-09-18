# Paper Daily - 2026-09-18

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`；同时用标题检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors`、`OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data`、`DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification`、`Continuum Dropout for Neural Differential Equations`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings`、`A novel approach to classification of ECG arrhythmia types with latent ODEs`，以及自动化记忆中记录的 `LERD`、`WIPSNet`、`Context-Aware Neural SDEs`、`RAxSS` 等已总结工作。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time-series classification / clinical irregular time series / informative sampling / sampling-policy shift 的顶会、顶会 workshop、OpenReview、ICLR/ICML/AAAI/NeurIPS 官方页面、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 `MedFuse` 这类 ICLR 2026 withdrawn submission，`SITS` 这类 ICML 2025 workshop 且窗口偏旧条目，`ReIMTS`、`APN`、`ASTGI` 这类近期顶会但主任务为 forecasting 的条目，`Time-IMM` 这类 NeurIPS D&B 价值很高但当前 benchmark 主任务为 forecasting 的条目，以及 `PhASER` 这类分类与 domain generalization 强但不直接处理 irregular sampling 的工作。严格的“近月顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 1 篇未在黑名单中的直接相关工作：`TimeCHEAT` 是 AAAI 顶会论文，覆盖 irregularly sampled multivariate time series 的 classification / forecasting / interpolation 三类任务，且对局部通道依赖与全局通道独立性的拆分可为 Sampling-Policy Shift 下的变量联测策略建模提供新视角。

## 1. TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis

- 方法名：TimeCHEAT / Channel Harmony ISMTS Transformer
- 会议：AAAI 2025 Technical Track，Proceedings of the AAAI Conference on Artificial Intelligence 39(18):18861-18869
- 作者：Jiexi Liu, Meng Cao, Songcan Chen
- AAAI 页面：https://ojs.aaai.org/index.php/AAAI/article/view/34076
- 论文：https://arxiv.org/html/2412.12886
- DOI：https://doi.org/10.1609/aaai.v39i18.34076
- 关键词：irregularly sampled multivariate time series, channel-dependent / channel-independent strategy, bipartite graph embedding, classification, interpolation, forecasting

### 场景、任务与核心难点

TimeCHEAT 面向 irregularly sampled multivariate time series (ISMTS) 的通用分析任务，覆盖医疗、气象、交通和人类活动等场景。最贴近分类的实验包括 P19、P12 和 PAM：P19 有 38,803 名患者、34 个传感器、约 94.9% missing ratio；P12 是 ICU 前 48 小时 36 个传感器的二分类预测，missing ratio 约 88.4%；PAM 则是 17 个传感器、8 类活动识别，missing ratio 约 60.0%。这些数据的共同难点是：同一变量内部采样间隔不均，不同变量之间异步观测，且稀疏通道很难仅靠自身历史形成稳定表示。

论文抓住的核心问题不是再设计一个单纯插值器，而是重新审视多变量时序中的 channel strategy。现有方法常在 channel-independent (CI) 与 channel-dependent (CD) 之间二选一：CI 能保留每个通道自己的采样节奏，但在稀疏通道上信息不足；CD 能利用跨通道信息，却容易过平滑并抹掉通道个性。TimeCHEAT 的策略是局部 CD、全局 CI：先把 ISMTS 切成 sub-series patches，在 patch 内用通道-时间二部图学习 irregular-to-regular 的边权和固定长度 embedding，聚合邻近观测和局部跨通道信息；再在 patch 级别用 CI Transformer 保留每个通道的长期个性化注意力。论文报告 TimeCHEAT 在 P19 上超过主要基线，在 P12 上接近最优且计算/空间成本低于 ViTST，在 PAM 八分类上比既有方法有约 0.7% accuracy 和 0.9% precision 提升。

### 审稿人视角：价值与不足

最有价值的技术思想是把“跨通道依赖应该在什么尺度使用”显式化。很多 IMTS 分类方法要么把所有变量混成一个大 token 空间，要么完全按变量独立建模，再用后端 attention 或 RNN 聚合；TimeCHEAT 指出这两种极端都不适合稀疏异步数据。局部 CD 让稀疏通道在短窗口内借用相关变量的可见信息，全局 CI 则避免模型把所有通道压成同一种动态模式。二部图式 time embedding 也有价值：它把 reference time embedding 的学习转化为边权预测，少依赖“时间越近越相关”这类固定假设，因此比简单 decay 或固定插值更灵活。

不足在于，TimeCHEAT 仍然把观测模式作为提升表示质量的结构资源，而没有进一步区分哪些局部跨通道关系来自稳定生理/物理机制，哪些来自环境特定采样政策。P12/P19/PAM 的随机 train/valid/test split 主要评估同分布泛化；如果换医院、换传感器调度、换变量联测协议，局部 CD 二部图中学到的强边可能从“生理耦合”变成“某机构常一起测”。此外，patch 切分和 reference points 的选择仍会影响哪些时间尺度被局部 CD 吸收；当关键事件在某些采样政策下被压缩、延迟或缺失时，局部聚合可能放大 policy shortcut。

### 对 Sampling-Policy Shift 的启发

TimeCHEAT 对 Sampling-Policy Shift 的横向启发是：采样政策偏移不只改变 delta-t 和 mask ratio，也会改变“局部跨通道信息能否互相借用”。医院 A 可能在怀疑感染时同时测 lactate、WBC 和体温，医院 B 可能只在高风险患者上补测 lactate；在这种情况下，普通 CD 模型很容易把联测关系当成类别证据，普通 CI 模型又可能丢掉真正有用的变量互补。TimeCHEAT 的局部 CD / 全局 CI 分层提供了一个很自然的审计位置：局部二部图边权、patch 内通道聚合强度、以及全局每通道 attention 是否随环境改变，都可以作为 sampling-policy shift 的诊断指标。

纵向深化上，可以把 TimeCHEAT 改造成 state-policy channel harmony。局部图分成两套边：state edges 捕捉跨采样政策稳定的生理/物理耦合，policy edges 捕捉医院流程、设备调度、变量联测和告警后密集观测造成的共现关系；分类主路径只使用 state edges 与全局 CI 表示，policy edges 进入偏移告警、校准或拒识头。训练时对同一潜在轨迹施加多种反事实采样策略，约束 state-edge distribution、channel-wise representations 和分类 logits 保持稳定，同时允许 policy-edge distribution 预测采样策略 ID、变量联测模式和观测密度。这样可以把 TimeCHEAT 的“通道策略和谐”推进到“状态通道关系与采样政策通道关系解耦”的非规则采样分类框架。
