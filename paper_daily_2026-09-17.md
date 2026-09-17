# Paper Daily - 2026-09-17

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`；同时分段读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Context-Aware Neural SDEs for Robust Irregular Time-Series Classification`、`RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series` 等全部已总结标题。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / heterogeneous sensor time series / physiological time-series classification / wearable ECG sampling-rate robustness / sampling policy shift 的顶会、顶会 workshop、OpenReview、ICML virtual、NeurIPS virtual、arXiv 与代码页。
- 已排除全部黑名单论文；同时继续排除 `MedFuse` 这类 ICLR 2026 withdrawn submission、`Mantis` / `Time-CoT` 这类强分类但未直接处理不规则采样的通用 MTS 工作、`CoCLD` / `MIRA` / `RAF` 等偏 next-event、generation 或 forecasting 的候选。本次保留 2 篇未在黑名单中的新跟踪对象：`Giving Sensors a Voice` 是 ICML 2026 正会论文，虽不是传统 ICU IMTS benchmark，但其 channel-aware JEPA、异构传感器语义和下游分类对跨采样/跨传感器策略迁移有直接启发；`A novel approach to classification of ECG arrhythmia types with latent ODEs` 是 NeurIPS 2025 TS4H workshop poster，直接讨论 wearable ECG 在低频/潜在不规则采样下的分类鲁棒性。

## 1. Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings

- 方法名：CHARM / Channel-Aware Representation Model
- 会议/状态：ICML 2026；arXiv 2026-05
- 作者：Utsav Dutta, Gerardo Pastrana, Sina Khoshfetrat Pakazad, Henrik Ohlsson
- 论文：https://arxiv.org/abs/2605.31580
- 关键词：heterogeneous multivariate sensor time series, channel-aware representation, JEPA, semantic channel descriptions, downstream classification, cross-dataset generalization

### 场景、任务与核心难点

这篇工作面向异构多变量传感器时序的通用表征学习，覆盖工业、能源、健康和金融等场景中的分类、异常检测与预测任务。它并不是传统 ICU EHR 中“每个变量任意时间戳”的 IMTS 专用模型，而是处理另一个相邻难点：不同部署环境的传感器集合、通道含义、噪声水平和时间动态不同，模型若只把通道当作匿名维度，很难跨设备或跨系统迁移到新的分类任务。

核心难点在于，普通时间序列模型通常假设固定长度、固定通道顺序和均匀结构输入；但真实传感器系统中，同一物理过程可能由不同仪器、不同通道命名、不同采样质量和不同组合方式记录。CHARM 将通道级文本描述引入时序 encoder：一方面用 description-aware temporal convolution 根据通道语义调节卷积感受野和核；另一方面用 inter-channel attention gating 与 temporal-offset attention 建模不同通道之间带时间滞后的关系，并保持通道顺序等变。训练上采用 JEPA 式 latent prediction，避免强迫模型重构噪声化原始值。论文报告在 UEA 分类任务上，冻结 embedding + SVM 平均准确率约 79.6%，微调后约 80.9%；在未见过的 17 传感器 UCI Hydraulic Systems valve classification 上，冻结 embedding + 线性 SVM 达到 99.8% accuracy。

### 审稿人视角：价值与不足

最有价值的思想是把“通道是什么”作为一等公民放进时间序列表征，而不是只把它看成维度编号。对异构传感器或医疗设备数据而言，通道身份、单位、测量对象和潜在滞后关系往往决定了跨数据集迁移能力；CHARM 用文本描述作为 channel addressing mechanism，使模型能在不同通道配置中找到可复用的结构。JEPA latent prediction 也是重要选择：相比 MAE 重构原始信号，它更倾向于保留稳定过程语义、过滤传感器噪声和低层 artifact，这对少标签分类和跨部署迁移很有吸引力。

不足在于，它对 irregular sampling 的处理仍是间接的。论文更关注异构通道、传感器语义和跨数据集泛化，模型主体仍以多变量时间网格表示为基础；如果面对 ICU/EHR 中变量级异步时间戳、value-pending、医生下单式测量和 MNAR missingness，需要额外的事件化输入层或连续时间位置机制。另一个风险是，通道描述和 learned inter-channel gates 可能把部署策略也编码进去：某些传感器是否存在、以何种名称出现、是否与其他通道共采样，可能反映机构/设备策略而不只是系统状态。

### 对 Sampling-Policy Shift 的启发

这篇论文对 Sampling-Policy Shift 的横向启发是：采样策略偏移常常伴随“通道语义与仪器配置偏移”。在医疗 IMTS 中，不同医院不仅测量频率不同，也可能把相同生理量拆成不同项目、采用不同单位、或只在特定流程中记录某些变量。CHARM 的 channel description + gating 机制可以迁移为 policy-aware variable semantics：用变量文本、单位、科室、设备和采样上下文描述来帮助模型识别哪些通道关系是生理稳定的，哪些关系可能只是某个观测政策下的联测习惯。

纵向深化上，可以把 CHARM 拆成 state-channel encoder 与 policy-channel encoder。state branch 使用通道语义学习跨策略稳定的变量关系和时间滞后，进入分类主路径；policy branch 则吸收通道可见性、采样密度、变量联测和设备配置，用于偏移告警与校准。训练时对同一潜在轨迹施加不同通道子集、不同采样频率和不同变量命名/单位转换，要求 state embedding 和分类 logits 保持一致，同时允许 policy embedding 识别观测制度。这样能把“传感器有语义”推进到“采样政策也有语义，但不应直接污染分类边界”。

## 2. A novel approach to classification of ECG arrhythmia types with latent ODEs

- 会议/状态：NeurIPS 2025 Workshop on Learning from Time-Series for Health (TS4H) Poster；arXiv 2025-11
- 作者：Angelina Yan, Matt L. Sampson, Peter Melchior
- OpenReview：https://openreview.net/forum?id=vFUaW7US4g
- 论文：https://arxiv.org/abs/2511.16933
- 关键词：wearable ECG, arrhythmia classification, latent ODE, low sampling frequency, sampling-rate robustness, MIT-BIH

### 场景、任务与核心难点

这篇工作面向可穿戴 ECG 的心律失常类型分类。临床 12 导联 ECG 采样频率高、空间信息完整，但通常是短时 spot-check，容易错过间歇性异常；可穿戴单导 ECG 可长期监测，却常因电池、舒适度和边缘硬件限制而降低采样频率，实际信号还可能有噪声、稀疏和局部不规则。任务是在 MIT-BIH Arrhythmia Database 上将心拍分为 AAMI 标准下的 N、V、S、F、Q 等类别，并检验从 360 Hz 下采样到 90 Hz、45 Hz 后分类是否保持稳定。

核心难点是，低频采样会直接损伤 ECG 形态细节，而心律失常分类高度依赖 PQRST 波形的局部形态和时序间隔。作者采用 path-minimized latent ODE 学习连续 ECG 波形的低维 latent vector，再用 GBDT 对 latent vector 分类；测试时对每个心拍采样多个 latent vector 并做多数投票。结果显示，macro AUC-ROC 从 360 Hz 的 0.984 仅降到 90 Hz 的 0.978 和 45 Hz 的 0.976，说明连续时间 latent encoder 能在采样率降低时保留主要分类结构；但 macro F1 从 0.86 降到 0.82，少数类 S、F 受影响更明显。

### 审稿人视角：价值与不足

最有价值的技术思想是把可穿戴 ECG 的低采样率问题转化为连续时间表示学习问题。相比直接在低频离散波形上训练分类器，latent ODE 学到的是能生成/解释完整波形的连续潜在轨迹；分类器使用这个潜在编码，相当于先恢复较稳定的动力学形态，再做类别判别。这个设计非常贴近边缘设备部署：如果 45 Hz 也能维持接近 360 Hz 的 AUC，就可能换来更长续航、更小设备和更长监测窗口。

不足也很清楚：它是 workshop 级别的初步研究，数据只来自 MIT-BIH 的 47 名个体和单导 MLII，并不是真正的大规模可穿戴线上数据。下采样方式是从 360 Hz 规则信号中抽点，不能完全代表真实设备中的非均匀丢包、运动伪影、佩戴中断和厂商自适应采样策略。分类器还依赖 SMOTE 处理类别不平衡，少数类 F 在 45 Hz 下准确率下降明显，说明 latent ODE 并未完全解决稀有事件和低频形态丢失问题。

### 对 Sampling-Policy Shift 的启发

这篇论文对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以具体表现为“同一生理事件在不同采样率下的形态可恢复性差异”。很多 IMTS 研究只改变 missing ratio 或随机 mask，但可穿戴 ECG 场景提醒我们，采样政策会改变频域上可观测的诊断证据：45 Hz 可能保留粗略节律，却损失少数类心拍所需的细节。因此，评估策略偏移时应报告类别级性能，而不仅是整体 AUROC；少数类往往最先暴露采样政策造成的信息损失。

纵向深化上，可以把 latent ODE classifier 扩展为 counterfactual sampling-rate training：固定高频或连续潜在 ECG 轨迹，生成多种设备策略下的观测视图，包括固定低频、事件触发高频、随机丢包和电池约束自适应采样。训练时要求 state latent 与分类 logits 在策略变化下尽量稳定，同时让 policy head 预测当前采样率、丢包模式和不确定性。对医疗 EHR 来说，这一思路可类比为：先学习能解释底层病程的连续状态，再把“何时被测、测得多密、哪些细节丢失”作为独立 policy uncertainty，而不是让分类器把采样频率本身当作疾病证据。
