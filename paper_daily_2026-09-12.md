# Paper Daily - 2026-09-12

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`；同时读取兼容入口 `paper_daily.md` 的标题索引，并参考自动化记忆中尚未单独成日期文件的新增标题。
- 本次黑名单论文标题已覆盖既有 50+ 篇历史总结，包括 `Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Observations: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`Status-Aware Self-Supervised Forecasting for Irregular Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Informative Irregularity as a Diagnostic for Model Robustness`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models` 与 `Learning Clinical Representations Under Systematic Distribution Shift` 等全部已总结标题；新候选均已逐项排除黑名单重名。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular physiological time series classification / clinical event dynamics / sampling policy shift 的顶会、顶会 workshop、OpenReview、ICML virtual site、AAAI Proceedings 与 arXiv 论文页面。
- 已排除全部黑名单论文；同时排除 `iTimER / Beyond Observations`（已在历史日报覆盖）、`MedFuse`（ICLR 2026 withdrawn submission）、`Integrating Sequence and Image Modeling in Irregular Medical Time Series Through Self-Supervised Learning`（AAAI-25，时间窗口偏旧）、`CLIR-Bench`（不规则临床时序 QA benchmark，不是分类主任务）和 `Learning Coupled Continuous-Time Latent Dynamics from Irregular Events`（ICML 2026 Spotlight，但主实验是 next-event prediction、trajectory generation 和 sequential recommendation，不是分类主任务）。本次保留 2 篇未在黑名单中的新工作：`LERD` 是 ICML 2026 正会 Poster，面向神经退行性疾病 EEG 分类并显式建模潜在事件时间与跨通道关系；`WIPSNet` 是 ICML 2026 Structured Data for Health 相关工作，面向夜间阻抗肺功能信号的儿科喘鸣夜级分类，虽不是传统 EHR-IMTS benchmark，但对长程、不规则生理信号的多尺度时间-频率表征和采样策略偏移很有横向价值。

## 1. LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification

- 会议：ICML 2026 Poster
- 作者：Yicheng Feng, Hairong Chen, Ziyu Jia, Samir Bhatt, Hengguan Huang
- 官方页：https://icml.cc/virtual/2026/poster/65698
- 论文：https://arxiv.org/abs/2602.18195
- 代码：https://github.com/hairongChenDavid/LERD
- 关键词：neurodegenerative classification, multichannel EEG, latent neural events, continuous-time event inference, event-relational graph, stochastic inter-event intervals

### 场景、任务与核心难点

LERD 面向阿尔茨海默病等神经退行性疾病的 EEG 分类。输入是多通道脑电信号，输出是临床分组或疾病状态判断；与很多只给出黑盒分类结果的 EEG 深度模型不同，这篇论文把诊断差异解释为潜在神经事件的发生时间、频率变化和跨通道协调关系改变。它不要求专家预先标注事件时间或通道交互，而是直接从观测 EEG 中端到端恢复 event-driven latent dynamics。

核心难点在于，神经退行性疾病改变的不是单个时间点的幅值，而是长期、跨通道、事件驱动的动力学结构。EEG 节律减慢、事件间隔分布变化和脑区协同异常都可能与疾病相关，但它们通常以连续时间、不规则事件和跨通道滞后关系的形式出现。LERD 用 Event Posterior Differential Equation (EPDE) 生成每个通道的连续时间潜在表示和下一事件期望时间，再用 Mean-Evolving Lognormal Process (MELP) 对正值、灵活且可多峰的事件间隔建模；同时加入 differentiable leaky-integrate-and-fire prior，将泄漏、 refractory、rate constraint 和合理频率范围作为神经生理先验；最后通过 event-relational graph (ERG) 把跨通道 event lag 映射为有向边权，解释哪些通道事件领先、滞后或同步。

### 审稿人视角：价值与不足

最有价值的技术思想是把“分类器是否准确”进一步拆成“哪些潜在事件、哪些事件间隔、哪些跨通道时滞支持了分类”。这比普通 CNN/RNN/Transformer EEG 分类更可审计：EPDE 负责连续时间状态，MELP 负责随机事件间隔，dLIF prior 约束生理合理性，ERG 负责关系解释，整体在一个变分目标下联合学习。对审稿人而言，这种结构不仅提高分类性能，还能输出 rate、timing 和 graph summaries，用于检查模型是否捕捉到神经退行性疾病中可信的节律和连接改变。

不足在于，LERD 的“irregularity”主要来自模型推断的潜在事件时间，而不是 EHR/传感器数据中显式记录的非规则采样时间戳。若原始 EEG 在采集设备上是较规则的高频信号，那么它与典型 irregular sampled multivariate time series classification 仍有距离。另一个风险是事件可识别性：没有专家事件标注时，模型恢复的 latent events 可能受预处理、分段长度、滤波设置、受试者运动伪影和 cohort 差异影响。ERG 的稳定性虽有理论分析，但跨设备、跨中心、跨采样率和不同 EEG montage 下是否保持同样的临床语义，还需要更系统验证。

### 对 Sampling-Policy Shift 的启发

LERD 对 Sampling-Policy Shift 的横向启发是：不要只把非规则采样看成观测缺口，也可以把它建模为“可解释事件时间结构”。在 ICU/EHR 中，某项化验何时出现、某个变量何时突然被密集测量，本质上也可以视为事件；这些事件既可能反映患者状态变化，也可能反映医院采样政策和医生注意力。LERD 的 EPDE + stochastic interval process 提供了一个模板：用连续时间状态描述 patient dynamics，用事件间隔分布描述 observation dynamics，再通过关系图解释变量间或通道间的时滞耦合。

纵向深化上，可以设计 policy-aware LERD for IMTS：把潜在事件拆成 state events 与 policy events。state events 表示跨采样策略稳定的病理转折，例如感染恶化、循环衰竭或呼吸状态改变；policy events 表示由医院流程触发的观测行为，例如入院常规套餐、告警后加测、夜班记录稀疏和 value-pending。训练时对同一潜在轨迹施加不同反事实采样策略，约束 state-event posterior、分类 logits 和 state ERG 保持一致，同时允许 policy-event interval 和 policy ERG 改变。这样能把 LERD 的“事件-关系可解释分类”推进到“区分病程事件与采样政策事件”的非规则时序分类框架。

## 2. WIPSNet: Deep Learning for Paediatric Wheeze Detection from Impedance Pneumography

- 方法名：Wheeze Impedance Pneumography Scalogram Network (WIPSNet)
- 会议/状态：ICML 2026 Structured Data for Health 相关 virtual 记录
- 作者：Felix Oury, Harley Day, Karina Mayoral, Ville-Pekka Seppa, Sejal Saglani, Reiko J. Tanaka
- 官方页：https://icml.cc/virtual/2026/71761
- 关键词：paediatric wheeze classification, impedance pneumography, long irregular physiological time series, continuous wavelet transform, 3D SE-ResNet, night-level clinical diagnosis

### 场景、任务与核心难点

WIPSNet 面向儿童夜间喘鸣检测。输入是 overnight impedance pneumography (IP) 生理信号，任务是在夜级别判断是否存在 wheeze；传统临床读数 Expiratory Variability Index (EVI) 将整夜记录压缩成单个标量，只得到约 0.63 AUC。论文提出将长程 IP 信号转换为 stacked continuous wavelet transform scalograms，再用 3D SE-ResNet 建模多尺度时间-频率体数据；在 15 名患者、60 个夜晚、281 小时记录上，WIPSNet 达到 0.873 +/- 0.019 AUC，并优于 EVI、Mamba 和两类现代 sleep-staging 架构。

核心难点在于，夜间呼吸异常不是一个固定长度、规则出现的局部片段。喘鸣事件可能短暂、稀疏、被体位和睡眠阶段调制，并夹杂运动伪影、传感器贴附质量变化和夜间记录质量波动。若把整夜信号压成单一指数，会丢掉事件发生时段、频率结构和上下文；若只看很短窗口，又容易把噪声或偶发呼吸模式误判为疾病证据。WIPSNet 的关键处理是把连续小波 scalogram 叠成带有时间上下文深度的体表示，并发现性能随 temporal context 增加到约 32 分钟而提升，说明夜间喘鸣分类需要跨局部事件和中尺度上下文共同判断。

### 审稿人视角：价值与不足

最有价值的思想是把长程生理信号分类从“单指标摘要”推进到“可学习的多尺度时间-频率证据聚合”。对于儿科呼吸监测这类真实临床任务，标签通常在夜级或患者级，而判别证据分散在长时间记录中的少量片段。WIPSNet 用 CWT scalogram 保留频率和时间局部性，再用 3D 卷积聚合多个相邻 scalogram，相当于把短时呼吸形态、频率模式和较长上下文整合成一个分类证据体；这比简单套用通用序列模型或单个 handcrafted index 更贴近任务结构。

不足在于，样本规模很小，只有 15 名患者和 60 个夜晚，虽然总小时数较长，但受试者层面的泛化证据仍有限。论文摘要称其处理 long, irregular physiological time series，但公开信息中对不规则采样机制、缺失段、传感器掉线和跨设备采样率差异的显式建模还不充分；当前贡献更像是面向特定 IP 信号的强表征方案，而不是通用 IMTS 分类架构。另一个审稿风险是 evaluation split：如果同一患者多个夜晚同时出现在训练和测试中，模型可能利用个体特征或设备佩戴习惯，而不完全是喘鸣病理模式；未来需要 patient-level split、跨设备外部验证和采样质量分层分析。

### 对 Sampling-Policy Shift 的启发

WIPSNet 对 Sampling-Policy Shift 的横向启发是：采样策略偏移不只发生在 EHR 化验下单，也发生在连续生理监测的“有效可观测片段”层面。夜间 IP 的信号质量、佩戴松动、睡眠阶段覆盖、伪影剔除规则和窗口选择策略都会改变模型看到的 scalogram 证据。若训练数据中某些患者或某些严重程度对应更完整、更高质量或更特定睡眠阶段的记录，分类器可能学习到 recording policy shortcut，而不是真正的喘鸣动力学。

纵向深化上，可以把 WIPSNet 扩展为 policy-aware time-frequency classifier：state branch 从 scalogram 中学习跨设备和跨采样质量稳定的呼吸病理特征，policy branch 预测信号质量、缺失块、窗口覆盖、睡眠阶段分布和传感器采样/过滤设置。训练时对同一夜间信号构造不同窗口抽样、不同伪影删除、不同采样率和不同缺失块的反事实视图，要求 state logits 和关键时间-频率激活保持稳定，同时允许 policy branch 对记录条件变化敏感。对我们解决非规则采样下的 Sampling-Policy Shift，这提示应把“时间-频率可恢复性”和“有效观测上下文长度”纳入策略偏移指标，而不只统计 mask ratio 或平均 delta-t。
