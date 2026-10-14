# Paper Daily - 2026-10-14

## 检索与去重记录

- 已强制读取根目录下全部已发现的日期版 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`、`paper_daily_2026-09-24.md`、`paper_daily_2026-09-25.md`、`paper_daily_2026-09-26.md`、`paper_daily_2026-09-27.md`、`paper_daily_2026-09-29.md`、`paper_daily_2026-10-13.md`；同时用标题索引读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充黑名单。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`、`Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models`、`ReDiTT: Retrieval Augmented Conditional Diffusion Transformers for Asynchronous Time Series`、`ReTAMamba: Reliability-Aware Temporal Aggregation with Mamba for Irregular Clinical Time Series Prediction`、`TANDEM: Temporal Attention-guided Neural Differential Equations for Missingness in Time Series Classification`、`Domain Generalization under Sampling Pattern Shifts in Irregular Time Series` 与 `Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness` 等全部已总结论文。
- 检索范围：近月围绕 irregular sampled / asynchronous / irregular clinical time series classification / sampling pattern shift / informative missingness / irregular observation process 的 OpenReview、NeurIPS/ICLR/ICML/KDD 论文页面、arXiv、PyPI/代码页与论文索引。
- 已排除全部黑名单论文；同时排除 `DBGL`、`ATENet`、`SLAN`、`ReTAMamba`、`Rethinking LLMs`、`STAR-Set`、`PRISM`、`MILM`、`PhASER`、`ReIMTS`、`GRUwE` 等已总结、此前已排除或主任务明显偏 forecasting / next-event prediction 的候选。本次保留 2 篇未在黑名单中的新跟踪对象：`A Time-Aware Bag-of-Receptive-Fields for Interpretable Irregular Time Series Classification` 是 2026-09-30 新预印本，直接面向 irregular time series classification，补足可解释、非插补、确定性分类器路线；`ChronoFlow: Hierarchical Flow Matching for Irregular Time Series Generation` 是 2026-09-27 新预印本，主任务是原生不规则时序生成而非分类，但它显式分层生成 observation counts、observation times、co-observation patterns 与 values，对构造 Sampling-Policy Shift 的反事实采样策略和诊断基准非常直接。

## 1. A Time-Aware Bag-of-Receptive-Fields for Interpretable Irregular Time Series Classification

- 方法名：i-borf / Irregular Bag-Of-Receptive-Fields
- 会议/状态：arXiv 2026-09-30 预印本，尚未确认顶会录用
- 作者：Francesco Spinnato
- 论文：https://arxiv.org/abs/2609.39268
- 代码/实现：https://pypi.org/project/fast-borf/0.2.1/，https://github.com/fspinna/borf
- 关键词：irregular time series classification, interpretable classifier, time-weighted normalization, sliding-window weighted variance, PYRREGULAR benchmark, no imputation

### 场景、任务与核心难点

i-borf 面向非均匀采样、缺失观测和变长序列上的不规则时序分类。典型场景包括医疗监测、移动轨迹、环境传感和其他难以强行对齐到规则网格的任务：每条序列的观测间隔不同，局部窗口内的观测点数量和时间跨度也不同，模型既要输出类别，又要能解释哪些时间片段和局部形态支撑了分类。

核心难点在于，很多不规则时序分类方法要么先插补/重采样，牺牲原始时间结构；要么使用复杂神经网络，虽然能吸收时间戳、mask 和 delta-t，却很难给出稳定可审计的模式解释。BORF 原本是快速、确定性、可解释的规则时序分类变换：在多尺度 receptive fields 上提取符号化局部模式，再形成 bag-of-words 特征。i-borf 将这一思路扩展到 irregular setting：局部均值、方差和归一化不再按观测索引等权计算，而是按每个观测对应的时间间隔进行 time-delta weighting，使窗口特征反映真实时间支持而不只是“第几个点”。为了不牺牲效率，论文推导了滑动窗口加权标准差的递推更新，保留 BORF 的线性复杂度，并在 PYRREGULAR repository 的不规则分类数据集上与 SOTA irregular classifiers 对比。

### 审稿人视角：价值与不足

最有价值的思想是把“解释性”和“原生不规则采样”放在同一个分类器中，而不是默认这二者只能二选一。很多 IMTS 方法通过 Transformer、CDE、ODE 或 Mamba 获得性能，但解释通常退化为 attention heatmap 或事后 saliency；i-borf 则让局部 receptive-field word 本身成为可回溯的分类证据。time-weighted normalization 也很朴素但关键：在不规则序列中，一个稀疏窗口和一个密集窗口即使包含相同数量的观测点，也可能代表完全不同的时间覆盖和状态持续性；按 delta-t 加权能避免把观测顺序误当成真实时间密度。

不足首先是论文当前仍是 arXiv 预印本，尚不能按顶会已验证结论使用。其次，BORF 类方法的表达力来自固定符号化、多尺度窗口和后端浅层分类器，面对高维异步 EHR、文本/影像混合模态或复杂跨变量交互时，可能不如端到端深度模型灵活。更关键的是，time-delta weighting 虽然尊重真实时间支持，但并不会自动区分“某状态持续很久”与“医院长时间没有测量”。如果观测间隔本身由医生下单、设备节能或医院流程决定，i-borf 仍可能把时间支持统计变成 sampling-policy shortcut，只是这个 shortcut 比深度模型更容易被解释和审计。

### 对 Sampling-Policy Shift 的启发

i-borf 对 Sampling-Policy Shift 的横向启发是：策略偏移需要可解释的局部模式审计，而不仅是端到端鲁棒指标。time-weighted receptive fields 可以把分类证据定位到“某个数值形态在多长真实时间内被支持”，这比普通索引窗口更适合检查采样政策变化后模型是否仍关注同一段病程。若反事实改变采样策略后，i-borf 的高权重 word 从生理形态转向长间隔、密集采样或特定缺测结构，就能直接暴露 policy leakage。

纵向深化上，可以把 i-borf 改造成 state-policy 双词袋分类器。state BORF 只统计跨采样政策稳定的数值形态和趋势 receptive fields；policy BORF 单独统计 delta-t、观测密度、缺测持续时间和变量可见性 receptive fields，用于偏移告警、拒识和校准。训练时对同一潜在轨迹生成多种 observation policy，要求 state word distribution 与 logits 保持一致，同时允许 policy word distribution 预测采样制度。这样能把 i-borf 的可解释性进一步转化为 Sampling-Policy Shift 下的“局部证据可审计不变性”。

## 2. ChronoFlow: Hierarchical Flow Matching for Irregular Time Series Generation

- 会议/状态：arXiv 2026-09-27 预印本，尚未确认顶会录用
- 作者：Changhun Kim, Sunguk Jang, Jeongjun Lee, Juhwan Choi, Sangchul Hahn, Grigorios Chrysos, Eunho Yang, Juho Lee
- 论文：https://searcharxiv.com/abs/2609.33276
- 关键词：irregular time series generation, hierarchical flow matching, observation counts, observation times, co-observation patterns, value generation, sampling structure metrics

### 场景、任务与核心难点

ChronoFlow 面向原生 irregular time series generation。它不是直接做分类的模型，而是生成完整不规则序列：一条样本不仅包含观测值，还包含观测次数、每个特征的观测频率、每次观测的时间、哪些特征在同一时间被共同观测，以及在这些 sampling patterns 条件下的数值。应用场景包括医疗 EHR、传感器网络和其他 observation process 本身很重要的时序数据。

核心难点在于，现有时序生成模型大多假设规则时间网格，或在给定采样结构后只生成 values；这会忽略不规则数据最关键的一半：数据为什么在这些时间、以这些变量组合被观察到。ChronoFlow 用 hierarchical flow matching 将复杂联合生成拆成由粗到细的统计层级：先生成 observation counts 和 feature-wise frequencies，再联合生成 observation times 与 feature co-observation patterns，最后在已实现的观测结构条件下生成 values。论文还提出覆盖 sample realism、sampling structure、value fidelity、temporal dependency 和 cross-feature dependency 的评估指标，并通过 controlled corruptions 验证这些指标能识别不同类型的生成退化。

### 审稿人视角：价值与不足

最有价值的思想是把 irregular time series 的生成对象从“数值轨迹”扩展为“观测制度 + 数值轨迹”的联合对象。对审稿人来说，这一点很重要：如果生成模型只会在固定 mask 上补值，它无法模拟跨医院、跨设备或跨采样策略的真实部署变化；ChronoFlow 的层级分解则显式承认 observation counts、feature frequencies、timestamps、co-observation graph 和 values 之间存在依赖关系。这也比简单随机 mask 更接近现实临床采样：哪些化验被联测、何时密集观测、某些变量多久出现一次，本来就是 care process 的一部分。

不足也很明确。首先，它是生成建模预印本，不是 irregular time-series classifier；目前应把它视作分类鲁棒性研究的工具候选，而不是直接替代分类器。其次，层级生成虽然能学习训练数据中的观测结构，但如果训练数据只来自单一医院或单一设备，它也可能忠实复制该环境的 sampling policy bias。最后，生成 fidelity 不等于反事实正确性：模型能生成真实感很强的采样模式，并不说明它知道哪些观测由 patient state 触发、哪些由医院流程触发；若没有因果约束或环境标签，state-policy 分解仍可能纠缠。

### 对 Sampling-Policy Shift 的启发

ChronoFlow 对 Sampling-Policy Shift 的横向启发非常直接：我们需要能生成和操控 observation process 的工具，而不是只在已有数据上被动评估。它的 observation-count / frequency / time / co-observation / value 分层可以改造成反事实采样策略模拟器：固定潜在 patient-state trajectory 和 label mechanism，替换不同医院的观测次数、变量联测图、告警后密集采样和检查触发规则，再测试分类器 logits、校准和解释是否稳定。

纵向深化上，可以设计 policy-controlled ChronoFlow + classifier 闭环。生成器分为 state generator 与 policy generator：state generator 产生连续病程和真实数值，policy generator 产生观测时间、变量可见性和 co-observation patterns。分类器训练时接收同一 state 下的多种 policy views，约束 state representation、分类 logits 和 state rationale 保持一致；policy head 则学习识别观测制度并输出不确定性。评估时报告 cross-policy logit drift、policy-only classifier AUPRC、co-observation graph shift 和 generated-policy coverage。这样能把 ChronoFlow 从“更真实的不规则时序生成”推进为“可控构造 Sampling-Policy Shift 并训练策略不变分类器”的实验底座。
