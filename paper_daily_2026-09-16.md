# Paper Daily - 2026-09-16

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`；同时读取兼容入口 `paper_daily.md` 的末尾索引，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Context-Aware Neural SDEs for Robust Irregular Time-Series Classification`、`RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification`、`DeNOTS: Stable Deep Neural ODEs for Time Series`、`Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification`、`WIPSNet: Deep Learning for Paediatric Wheeze Detection from Impedance Pneumography`、`Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors`、`OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data`、`DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification`、`Continuum Dropout for Neural Differential Equations` 与 `Towards Self-Supervised Foundation Models for Critical Care Time Series` 等全部已总结标题。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time-series classification / clinical irregular time series / informative sampling / continuous positional encoding 的顶会、顶会 workshop、NeurIPS Proceedings、OpenReview、ICML/ICLR/AAAI virtual 页面、arXiv 与论文项目页。
- 已排除全部黑名单论文；同时排除 `GRUwE / Still Competitive` 这类主要任务是 next-observation 与 next-event prediction、`Oscillators Are All You Need` 这类尚未确认顶会录用且偏通用建模的预印本、`A Statistical Approach for Modeling Irregular Multivariate Time Series with Missing Observations` 这类期刊/非顶会来源、以及已由历史记录覆盖的 `QuITE`、`Random Controlled Differential Equations`、`DeNOTS`、`STAR-Set`、`EHR-SPC`、`MILM`、`RAxSS` 等候选。本次保留 2 篇全新工作：`Rotary Masked Autoencoders are Versatile Learners` 是 NeurIPS 2025 正会论文，直接包含 irregular multivariate time-series classification；`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series` 是 NeurIPS 2025 TS4H Poster，直接面向不规则采样临床时序的院内死亡二分类。

## 1. Rotary Masked Autoencoders are Versatile Learners

- 简称：RoMAE
- 会议：NeurIPS 2025
- 作者：Uros Zivanovic, Serafina Di Gioia, Andre Scaffidi, Martin de los Rios, Gabriella Contardo, Roberto Trotta
- NeurIPS 页面：https://proceedings.neurips.cc/paper_files/paper/2025/hash/c2626ef6cdaaa6a18927832820079e1d-Abstract-Conference.html
- 论文：https://arxiv.org/abs/2505.20535
- 关键词：irregular multivariate time-series classification, rotary positional embedding, continuous position, masked autoencoder, light-curve classification, multimodal representation learning

### 场景、任务与核心难点

RoMAE 面向“位置本身不再是规则整数索引”的多模态表示学习，其中最相关的任务是 irregular multivariate time-series classification。论文的代表性实验是 DESC ELAsTiCC Challenge：约 180 万条模拟天文光变曲线、36 个天体类别、6 个不规则采样波段；每条样本不仅时间点不均匀，不同波段也可以异步出现。作者还在 UEA 多变量时序数据上通过随机丢弃 30% 观测构造不规则分类设置，并额外覆盖不规则插值、图像和音频任务。

核心难点是：标准 Transformer/MAE 的位置编码默认处理规则离散位置，面对真实时间戳、通道索引和多维连续坐标时，需要专门架构、ODE 模块或复杂插值才能使用。这样会带来额外计算开销，也削弱了复用成熟 Transformer/MAE 生态的能力。RoMAE 的做法是把 RoPE 推广到连续位置，并用 Axial RoPE 同时编码时间、通道或空间维度；输入仍走标准 MAE 的 mask-pretrain + fine-tune 路线，使模型在不强行补点、不改造主干结构的情况下处理不规则多变量序列。结果上，RoMAE-tiny 在 ELAsTiCC 上 F-score 达到 0.8029，明显高于专门为该任务设计的 ATAT baseline 0.6270；在若干 UEA 不规则化分类任务上也能与 TST、mTAN、S5、ContiFormer 等模型竞争。

### 审稿人视角：价值与不足

最有价值的思想是把“处理不规则采样”从专门时序架构问题，转化为连续位置编码与通用自监督预训练问题。对审稿人而言，这个贡献很干净：RoMAE 并不依赖复杂的临床特征工程或任务特定插值，而是证明标准 Transformer 组件只要具备连续坐标感知能力，就能在不规则多变量分类上达到强性能。另一个值得重视的点是论文没有只展示结果，还分析了 RoPE 的相对位置性质在连续位置和 learned input embeddings 下可能被破坏；这对后续使用 RoPE 处理不规则时间戳很有警示意义。

不足在于，RoMAE 的主要不规则分类证据来自天文光变曲线与人工丢点后的 UEA 数据，并不是 ICU/EHR 中由医生下单、设备触发和工作流共同决定的 MNAR 异步观测。ELAsTiCC 中的观测 cadence、alert mask、波段可见性和测量误差具有强观测策略属性；RoMAE 证明连续 RoPE 可以吸收这些位置结构，但没有系统区分哪些位置/通道模式代表天体物理状态，哪些只是特定巡天策略或筛选流程。再者，MAE 的重构目标可能鼓励模型恢复被 mask 的位置和值，若采样位置本身与类别强相关，预训练表征仍可能学习到 policy shortcut。

### 对 Sampling-Policy Shift 的启发

RoMAE 对 Sampling-Policy Shift 的横向启发是：时间戳、通道索引和采样坐标不能只被当作“更精确的位置编码”，还应被视为 observation policy 的载体。连续 RoPE/axial RoPE 可以很好地表达不规则位置，但表达能力越强，越需要约束模型不要把医院采样节律、设备 duty cycle、巡天 cadence 或 alert mask 当成稳定类别证据。我们可以把 RoMAE 式连续位置编码拆成 state-position channel 与 policy-position channel：前者描述跨策略稳定的病程/物理相位，后者描述采样制度、通道可见性和观测质量。

纵向深化上，可以基于 RoMAE 设计 policy-audited MAE 预训练：对同一潜在连续轨迹生成多种反事实采样视图，要求 state tokens、分类 logits 和语义解释在不同采样政策下保持一致；同时训练 policy tokens 去预测采样密度、通道缺失、alert mask 或医院/设备环境。还可以借鉴论文对位置重构的分析，加入 position-leakage probe：若仅凭 positional embeddings 或 mask pattern 就能高精度预测类别，则说明分类器仍依赖采样政策。这样能把 RoMAE 的“通用连续位置 MAE”推进到“显式审计和约束采样位置泄漏”的非规则时序分类预训练框架。

## 2. No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series

- 方法名：SLAN / Switch LSTM Aggregate Network
- 会议/状态：NeurIPS 2025 Workshop on Learning from Time-Series for Health (TS4H) Poster；arXiv 2023/2025 更新
- 作者：Rohit Agarwal, Aman Sinha, Ayan Vishwakarma, Xavier Coubez, Marianne Clausel, Mathieu Constant, Dilip K. Prasad, Alexander Horsch
- OpenReview：https://openreview.net/forum?id=o3E5pkVOlS
- 论文：https://arxiv.org/abs/2309.08698
- 关键词：irregularly sampled time series, non-imputation, switch layer, sensor-specific LSTM, informative missingness, in-hospital mortality classification

### 场景、任务与核心难点

SLAN 面向临床 irregularly sampled time series 的院内死亡预测。输入来自 MIMIC-III 与 PhysioNet 2012：多变量传感器/临床变量在不一致时间点被观测，某些变量长时间缺失，且不同患者的观测顺序和观测变量集合都不同。下游任务是二分类的 in-hospital mortality prediction，评价指标包括 AUROC 与 AUPRC。

核心难点是：很多不规则时序方法先把数据插补成规则矩阵，再交给 RNN、Transformer 或分类器；但插补必然假设某种 missing mechanism，可能引入人工值和分布偏差。SLAN 反过来不补值，而是为每个 sensor/变量配置一个 LSTM block，并用 switch layer 决定当前时间点哪些变量的 LSTM 被激活。被观测变量更新自己的局部状态；激活 LSTM 的 long-term memory 被聚合为全局 summary，再反馈给所有变量，形成“变量局部记忆 + 跨变量全局状态”的动态结构。论文报告 SLAN 在 MIMIC-III 上达到 51.12 AUPRC / 85.63 AUROC，在 PhysioNet 2012 上达到 55.20 AUPRC / 86.42 AUROC，均优于 GRU-D、IP-Nets、SeFT、Raindrop、CoFormer、IVP-VAE 等 imputation 与 non-imputation baseline；在额外随机丢弃 25%、50%、75% 观测时也保持相对优势。

### 审稿人视角：价值与不足

最有价值的思想是用“结构开关”而不是“数值填补”来响应不规则观测。SLAN 的 switch layer 很直接地承认：某个变量没有被测时，不应伪造一个输入去更新对应状态；只有真实观测到达时，局部 LSTM 才更新。这比把 mask/delta-t 作为附加特征更强，因为模型结构本身随观测集合变化。全局 summary state 也避免了完全变量隔离，使某个化验或生命体征的出现可以影响其他变量的后续解释。作为 workshop 工作，它的设计并不复杂，但很适合临床在线部署：观测按事件到达，模型按变量开关增量更新。

不足在于，SLAN 虽然避免了 imputation bias，却仍显式利用“哪些变量被测、何时被测”这个采样模式。论文把 informative missingness 视为性能来源，但没有进一步区分 state-driven informative missingness 与 policy-driven missingness。若训练医院中某些变量只在高风险病人中测量，switch 激活频率和全局 summary 很可能把这套医院政策学成死亡风险证据；换到检查协议不同的医院，优势可能变成 shortcut。另一个限制是，sensor-specific LSTM 随变量数增长会增加模块数量，面对高维 EHR event vocab、文本事件或跨数据集变量 schema 时，需要额外的参数共享或变量语义建模。

### 对 Sampling-Policy Shift 的启发

SLAN 对 Sampling-Policy Shift 的横向启发非常直接：采样政策可以被建模为一组随时间开关的观测动作，而不是被压缩成静态 missingness ratio。现实 ICU 中，“开关何时打开”往往由患者状态、医生怀疑、科室流程、资源约束和设备配置共同决定。SLAN 的 switch layer 可以扩展为 policy gate：state gate 解释真实病程触发的观测，policy gate 解释医院制度或设备策略触发的观测；分类头主要依赖 state-updated representation，policy gate 则用于校准和偏移告警。

纵向深化上，可以构造 counterfactual-switch training：固定同一潜在患者状态轨迹，模拟不同医院/设备采样策略下的 switch 序列，约束局部状态、全局 summary 和死亡风险 logits 保持稳定，同时允许 policy representation 预测采样策略 ID、变量联测模式和观测密度。评估时可加入 switch-only classifier、cross-policy switch frequency shift、global-summary policy predictability 等指标，检测模型是否只凭“哪些开关被打开”就能预测标签。这样能把 SLAN 的非插补式在线建模推进到“将采样动作本身解耦为状态证据与政策证据”的 Sampling-Policy Shift 解决路径。
