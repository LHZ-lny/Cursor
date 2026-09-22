# Paper Daily - 2026-09-22

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`；同时分段读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`，以及自动化记忆中记录的 `Auditing Black-Box Trends: Structural Inductive Bias Facilitates Causal Interpretability in Clinical Time Series`、`Pretraining EHR Foundation Models with Patient-Aware Sampling` 等近期已总结工作。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / continuous-time sequence model / sampling-policy shift 的顶会、顶会 workshop、NeurIPS Proceedings、ICLR/ICML/AAAI 官方页面、OpenReview、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 `MissTSM`（已在历史记忆中覆盖）、`SUMMIT/SCANE`（ICLR 2024 workshop，时间窗口偏旧）、`PhASER`（TMLR 2025，非原生不规则采样主线）、`WIPSNet`（已由历史记忆标记覆盖）、`DBGL/STAR-Set/SLAN/TimeCHEAT/INTERVenE` 等已总结条目。本次保留 2 篇未在黑名单中的新跟踪对象：`Structured Linear CDEs` 是 NeurIPS 2025 正会论文，虽不是专门医疗 IMTS 方法，但直接继承 NCDE 对不规则采样的连续时间优势，并在多变量时序分类上把 Log-NCDE 级性能和并行效率结合；`MedFuse` 是 arXiv 2511.09247 / ICLR 2026 withdrawn submission，严格说不是已录用顶会论文，但它直接面向 irregular clinical time series classification，且在直接命中已被黑名单耗尽后，对 EVAT 式事件 token 表示和 Sampling-Policy Shift 有新增参考价值。

## 1. Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models

- 简称：SLiCE
- 会议：NeurIPS 2025
- 作者：Benjamin Walker, Lingyi Yang, Nicola Muca Cirone, Cristopher Salvi, Terry Lyons
- 论文：https://proceedings.neurips.cc/paper_files/paper/2025/file/f89ce18433cbae97838bd4690454e7c3-Paper-Conference.pdf
- 代码：https://github.com/Benjamin-Walker/structured-linear-cdes
- 关键词：controlled differential equations, parallel-in-time sequence models, irregular sampling robustness, multivariate time-series classification, structured state-transition matrices

### 场景、任务与核心难点

这篇工作面向长序列建模中的连续时间序列分类和状态跟踪问题。它不是只为 ICU/EHR 设计的 IMTS 分类器，而是从 Linear Neural CDE 的角度重新审视 Mamba、DeltaNet、线性 RNN 和 Log-NCDE 这类序列模型：如果输入是一条不规则观测提升得到的连续控制路径，模型需要既有足够表达力捕捉隐藏状态与输入路径的乘性交互，又要能像现代并行序列模型一样高效训练。

核心难点在于，NCDE/Log-NCDE 对不规则采样天然友好，但非线性向量场和 Log-ODE 训练成本高；而许多并行 SSM 或线性注意力模型虽然快，却使用对角或过度受限的状态转移结构，理论上无法表达某些状态跟踪函数。SLiCE 提出 Structured Linear Controlled Differential Equations，用 block-diagonal、sparse、Walsh-Hadamard、DPLR 等结构化输入依赖状态转移矩阵，证明其中多类结构可以达到 dense matrix 的最大概率表达力，同时把递推计算写成 associative parallel scan。实验上，block-diagonal SLiCE 在 6 个 UEA 多变量时序分类数据集上匹配 Log-NCDE 表现，并将每步训练时间平均降低约 20 倍。

### 审稿人视角：价值与不足

最有价值的思想是把“连续时间模型的表达力”和“并行序列模型的效率”放在同一个数学框架下讨论。对审稿人而言，这比单纯提出一个更快 CDE implementation 更扎实：论文说明哪些状态转移矩阵结构会损害路径函数表达力，哪些结构能在保持近似 dense 表达力的同时显著降低成本。对于不规则采样分类，这意味着我们可能不必在 Log-NCDE 的高成本和 SSM 的表达力限制之间二选一，而可以用结构化线性 CDE 获得可扩展的连续时间 encoder。

不足在于，SLiCE 的重点是序列模型表达力、状态跟踪和并行效率，而不是专门分析 EHR/传感器中的 MNAR 采样机制。UEA 多变量分类实验能够说明其作为连续时间分类主干的实用性，但尚不能证明它能区分 patient state 与 observation policy。由于 SLiCE 仍需要从离散观测构造控制路径，如果训练环境中的观测频率、变量联测和采样触发规则与标签相关，结构化 CDE 可能更高效地吸收这些策略性轨迹差异，而不一定更稳健。

### 对 Sampling-Policy Shift 的启发

SLiCE 对 Sampling-Policy Shift 的横向启发是：采样策略偏移不仅影响模型准确率，也影响我们能否用可扩展方式对反事实采样策略做训练和审计。若每次策略增强都要跑昂贵的 Log-NCDE，policy-invariant 训练很难扩展；SLiCE 的 parallel-in-time 结构使多策略视图、cross-policy posterior alignment 和大规模采样增强更可行。

纵向深化上，可以把 SLiCE 改造成 state-policy structured CDE：state CDE 使用跨策略共享的结构化矩阵来编码底层连续病程，policy CDE 使用单独的结构化子空间来编码观测时间、变量可见性和采样密度。训练时对同一潜在轨迹生成多种 observation policy，约束 state hidden path、分类 logits 和关键 readout 稳定，同时允许 policy hidden path 预测采样制度。这样能把 SLiCE 的“高效最大表达连续时间主干”推进到“可扩展地训练和审计采样策略不变 CDE 分类器”。

## 2. MedFuse: Multiplicative Embedding Fusion for Irregular Clinical Time Series

- 会议/状态：ICLR 2026 withdrawn submission；arXiv 2025-11 预印本
- 作者：Yi-Hsien Hsieh, Ta-Jung Chien, Chun-Kai Huang, Shao-Hua Sun, Che Lin
- OpenReview：https://openreview.net/forum?id=YPZ9Z6lm4y
- 论文：https://arxiv.org/abs/2511.09247
- 关键词：irregular clinical time series, EVAT, multiplicative embedding fusion, imputation-free EHR modeling, ICU mortality classification, chronic disease prediction

### 场景、任务与核心难点

MedFuse 面向 EHR 中不规则、异步、高缺失的临床时序分类。输入由 `(feature identity, value, timestamp)` 三元组构成，只对真实观测生成 token，不对缺失值做显式插补；实验覆盖 PhysioNet 2012 ICU mortality、MIMIC-III ICU mortality 和长期 HCC onset risk 三类任务。核心问题不是重新设计一个复杂的连续时间动力学模型，而是问：在 EVAT/事件 token 路线中，数值观测值应该如何和变量身份融合，才能表达临床上“同一个变量不同数值区间语义完全不同”的现象。

论文指出，许多 EHR tokenization 方法把 feature embedding、value embedding 和 time embedding 加和或拼接；这种 additive fusion 对不同数值的非线性、feature-specific 语义表达有限。MedFuse 提出 MuFuse：用观测值生成的 value embedding 对 feature identity embedding 做逐维或广播式 Hadamard 乘法调制，然后再叠加时间编码，交给 Transformer encoder 分类。作者还把 SUMMIT/SCANE 解释为 MuFuse 的一个特例，并报告 MedFuse 在 MI3、P12、HCC 三个数据集上获得最高 AUPRC，同时通过 P12-MI3 共有变量实验展示 feature embedding 的跨数据集迁移潜力。

### 审稿人视角：价值与不足

最有价值的思想是把 irregular EHR token 的“值-变量融合方式”提升为核心建模问题。很多 IMTS 工作关注时间戳、mask 或连续路径，却默认单个观测 token 的数值语义已经被充分表达；MedFuse 提醒我们，对临床变量而言，数值不是普通标量附加项，而是会条件化变量身份语义的门控信号。乘法调制比加法更能表达“高 creatinine 与低 creatinine 是同一变量但临床语义不同”的非线性差异，且仍保持 imputation-free 和标准 Transformer 兼容。

不足首先来自论文状态：OpenReview 明确标注为 ICLR 2026 withdrawn submission，因此不能按已录用顶会结论看待，只适合作为高相关预印本跟踪。方法上，它仍把每个被观测到的数值 token 当作可靠证据，对“为什么这个变量在这个时间被测到”建模不足。若某些变量在训练医院中只在高风险患者上被测，MuFuse 会更强地学习该变量值与风险之间的复杂交互，但不一定区分这是稳定病理机制还是医院测量政策。此外，2 小时/90 天 summarization 会部分抹平原始异步事件流，使其对细粒度 sampling-policy shift 的验证仍不充分。

### 对 Sampling-Policy Shift 的启发

MedFuse 对 Sampling-Policy Shift 的横向启发是：策略偏移不仅体现在“哪些 token 出现”，还会体现在“出现 token 后，value 与 feature 的融合语义是否可迁移”。同一个 lactate 值在常规筛查医院和只对疑似危重患者下单 lactate 的医院中，可能对应不同的先验风险；若乘法融合无条件进入分类边界，模型会把医院策略条件化的数值语义当成稳定医学语义。

纵向深化上，可以把 MuFuse 拆成 state-conditioned fusion 与 policy-conditioned fusion。state fusion 学习跨医院稳定的 value-feature 临床语义，进入主分类路径；policy fusion 学习某变量为何被测、采样窗口如何构造、该值是否来自告警后密集观测，仅用于校准和偏移告警。训练时可对同一患者状态模拟不同测量政策，并约束 state-fused token representation 与 logits 稳定，同时允许 policy-fused residual 预测医院/科室/采样策略。这样能把 EVAT 的 imputation-free 优势与 Sampling-Policy Shift 下的观测政策解耦结合起来。
