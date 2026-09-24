# Paper Daily - 2026-09-24

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`；同时用标题检索补读兼容入口 `paper_daily.md`，并参考自动化记忆中的 2026-08 至 2026-09 补充黑名单。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`Status-Aware Self-Supervised Forecasting for Irregular Clinical Time Series`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`、`Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models` 与 `MedFuse: Multiplicative Embedding Fusion for Irregular Clinical Time Series` 等全部已总结论文。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous time series / irregular clinical time series classification / event-type prediction / state-space model / sampling-policy shift 的顶会、顶会 workshop、OpenReview、ICML/ICLR/NeurIPS/CIKM/AAAI 页面、arXiv、ACM/Elsevier 论文记录与作者页面。
- 已排除全部黑名单论文；同时排除 `PYRREGULAR`、`ATENet`、`STAR-Set`、`EHR-SPC`、`ORA`、`SLAN`、`DBGL`、`MedFuse`、`Structured Linear CDEs` 等重复命中。本次严格的“近月顶会正会 + 直接 IMTS 分类”新增命中仍很稀缺，因此保留 2 篇未在黑名单中的新跟踪对象：`ReDiTT` 是 ICLR 2026 TSALM Workshop Poster，任务是异步事件流的 next-event type/time 与 long-horizon event forecasting，虽不是整条序列标签分类，但直接建模 asynchronous marked event sequence；`DynaMamba` 是 2026 Journal of Biomedical Informatics 论文，不是顶会论文，但题目与任务最直接命中 irregular clinical time series classification，且在顶会直接命中被历史黑名单耗尽后，对 Mamba/SSM 路线下的采样策略偏移有参考价值。

## 1. ReDiTT: Retrieval Augmented Conditional Diffusion Transformers for Asynchronous Time Series

- 会议/状态：ICLR 2026 Workshop on Time Series in the Age of Large Models (TSALM) Poster；arXiv 2026-07
- 作者：Saiyue Lyu, Ruizhi Deng, Thibaut Durand, Zhitian Zhang
- OpenReview：https://openreview.net/forum?id=686miT6KoQ
- 论文：https://arxiv.org/abs/2607.12391
- 代码：https://github.com/BorealisAI/ReDiTT
- 关键词：asynchronous time series, temporal point process, event-type prediction, conditional diffusion transformer, retrieval augmentation, long-horizon event forecasting

### 场景、任务与核心难点

ReDiTT 面向 asynchronous time series，也就是连续时间事件序列。每个样本不是规则网格上的多变量矩阵，而是一串带 inter-event time 与 event type 的 marked events；典型场景包括医疗监测、用户行为、金融事件和动作序列。论文的主要任务是 next-event prediction 与 long-horizon forecasting：给定历史事件，预测下一次事件的发生时间和事件类型，或生成未来多个事件。严格说它不是传统“整条时序打一个类别标签”的分类器，但 event type prediction 本身是异步事件流上的局部分类问题，且它对非规则采样下“未来会观测到什么事件、何时观测到”的建模与临床风险分类前端高度相关。

核心难点在于，异步事件序列的未来不是单一路径：同一段历史可能对应多种合理的未来事件组合，长程预测还会出现 autoregressive error accumulation。普通 temporal point process 模型依赖强参数化 intensity 或逐步自回归，难以保持全局轨迹结构；普通 diffusion / flow matching 模型虽然能整体生成未来，但缺少强条件信号，容易漂移到训练集中的平均模式。ReDiTT 的做法是在 VAE latent event-token 空间中建立 memory bank，对当前前缀检索 top-k 相似历史轨迹，并把这些 reference tokens 通过 cross-attention 注入 conditional diffusion Transformer。训练和推理都在 latent space 做 retrieval-conditioned flow matching，使模型在生成未来时有具体的相似轨迹作为非参数结构先验。

### 审稿人视角：价值与不足

最有价值的思想是把 retrieval-augmented generation 引入 asynchronous event streams，而不是只依赖参数化序列模型记住所有事件动力学。对审稿人而言，ReDiTT 的技术价值有两点：第一，检索发生在与生成目标一致的 latent event-token 空间，避免手工设计混合连续时间和离散 mark 的相似度；第二，reference 不是粗粒度全局向量，而是序列形状的 token memory，通过 cross-attention 逐位置指导 denoising，因此能为长程异步预测提供轨迹级结构约束。它在 Amazon、Retweet、StackOverflow、Taobao、Taxi、Breakfast、Multithumos 等 7 个数据集上报告 next-event 与 long-horizon 预测提升，说明这个思路不局限于单一医疗数据。

不足在于，它的主任务仍是未来事件生成/预测，而非 ICU mortality、sepsis、phenotyping 这类终局标签分类；医疗场景也只是应用动机之一，实验并未直接覆盖 EHR/ICU irregular clinical classification。更关键的是，retrieval memory bank 可能放大训练环境中的观测政策：如果相似轨迹主要由医院、设备、科室或平台采样规则决定，而不是真实 patient state / user state 决定，那么 cross-attention 可能把 policy-neighbor 当成 state-neighbor。论文强调 prefix-only retrieval 避免未来泄漏，但还没有系统评估跨采样政策、跨机构或反事实 observation process 下检索邻居是否仍代表稳定状态。

### 对 Sampling-Policy Shift 的启发

ReDiTT 对 Sampling-Policy Shift 的横向启发是：采样策略本身可以通过 retrieval 暴露出来。若仅用 event times / event types 的前缀就能检索到同医院、同设备、同流程的样本，并显著改善后续预测，说明 observation policy 已经形成可检索的轨迹指纹。对我们的问题，可以把 retrieval-neighbor composition 作为偏移诊断：同一 patient-state 表征在不同采样策略下是否检索到不同 policy cluster？检索 reference 对分类 logits 的贡献是否随医院/科室/采样密度改变？

纵向深化上，可以设计 policy-audited retrieval IMTS classifier：state memory bank 存储跨采样策略稳定的病程邻居，policy memory bank 存储观测流程邻居。分类器只能通过 state references 强化风险判断，而 policy references 进入校准、拒识和偏移告警。训练时对同一潜在轨迹构造多种 observation policy 视图，约束 state retrieval、latent event representation 和分类 logits 稳定，同时允许 policy retrieval 识别采样制度。这样能把 ReDiTT 的“相似轨迹指导异步生成”推进到“检索时显式区分病程相似与采样政策相似”的非规则采样分类框架。

## 2. DynaMamba: Multi-scale Dynamic Interacting Mamba Network for Irregular Clinical Time Series Classification

- 会议/状态：Journal of Biomedical Informatics, Volume 178, Article 105027, 2026；非顶会论文
- 作者：Hao Chen, Junjie Zhang, Xiaowei Yan, Zhuo Li, Shengye Lu, Buzhou Tang
- DOI：https://doi.org/10.1016/j.jbi.2026.105027
- 论文记录：https://researchr.org/publication/ChenZYLLT26
- 关键词：irregular clinical time series classification, Mamba, state-space models, multi-scale dynamics, dynamic interaction, ICU mortality / sepsis benchmarks

### 场景、任务与核心难点

DynaMamba 直接面向 irregular clinical time series classification。公开索引显示其关联 PhysioNet 2012、PhysioNet 2019 sepsis prediction、MIMIC-III 等典型 ICU/临床 benchmark，任务可理解为院内死亡、脓毒症或临床风险分类。输入难点是典型 IMTS：多变量生命体征和化验异步出现，变量缺失率高，时间间隔不均，且不同变量的更新频率和临床时效性差异很大。

这篇工作的核心问题是：Mamba/SSM 类线性复杂度序列模型能否替代 Transformer/RNN，成为不规则临床时序分类的高效主干。公开可检索细节有限，但题目中的 multi-scale dynamic interacting Mamba 指向三个关键设计动机：多尺度 temporal aggregation 捕捉短期告警与长期病程；dynamic interaction 让变量或时间片段之间的依赖随当前观测内容变化；Mamba/SSM 主干以线性复杂度处理长序列，避免 Transformer 在长 ICU 轨迹和高频监测数据上的二次注意力成本。相较于 MedMamba 这类更偏规则生理信号的医疗分类模型，DynaMamba 的题目明确把目标放在 irregular clinical time series classification 上，因此值得作为非顶会但高相关的新跟踪对象。

### 审稿人视角：价值与不足

最有价值的思想是把 selective state-space modeling 带入临床 IMTS 的多尺度交互建模。Mamba 的选择性状态更新机制天然适合“有些观测需要长期保留、有些观测只是短暂噪声”的临床场景；如果再结合多尺度窗口和动态变量交互，就可能在不显式全局 attention 的情况下同时表示近期异常、长期趋势和跨变量联动。对审稿人而言，这条路线的潜在价值在于效率与结构归纳偏置：它可能比 Transformer 更适合长时间 ICU 轨迹，也比传统 GRU-D 更能表达内容依赖的状态更新。

不足首先是证据透明度：当前公开搜索结果主要给出题名、作者、DOI、期刊和相关数据集线索，缺少完整 abstract、架构图、消融与具体性能指标，因此不能像顶会论文那样充分审查其贡献边界。其次，Mamba/SSM 的长程记忆能力也可能放大 sampling-policy shortcut：某个化验长期未测、某类变量在告警后被密集测量、某个医院常规联测变量的模式，都可能被 selective state 更新当作高价值临床状态保存下来。若缺少跨医院、跨采样协议和 policy-only probe，无法判断 DynaMamba 学到的是病程动力学，还是训练机构的 observation process。

### 对 Sampling-Policy Shift 的启发

DynaMamba 对 Sampling-Policy Shift 的横向启发是：采样策略偏移会通过 state-space memory 的“写入、保留、遗忘”机制进入分类器。与 Transformer attention 主要暴露为注意力权重不同，Mamba 的风险在于 policy signal 被压入隐藏状态并沿时间传播；一旦某类采样模式在训练环境中与标签相关，选择性状态更新可能持续保留这种 policy trace，使跨机构泛化更难诊断。

纵向深化上，可以把 DynaMamba 式模型改造成 state-policy dual SSM。state SSM 只接收经反事实采样增强后仍稳定的观测值和病程特征，负责分类主路径；policy SSM 接收 delta-t、mask、变量联测、采样密度、value-pending 和医院/科室上下文，负责预测观测制度与不确定性。训练时对同一潜在病程生成不同采样策略，约束 state hidden state、分类 logits 和关键多尺度摘要保持一致，同时要求 policy hidden state 可区分采样制度。评估时报告 hidden-state policy predictability、cross-policy retention drift、policy-only AUPRC 和 state-logit counterfactual consistency。这样能把 Mamba 的高效长程记忆能力用于病程建模，而不是无约束地记住采样政策。
