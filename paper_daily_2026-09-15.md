# Paper Daily - 2026-09-15

## 检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`；同时读取兼容入口 `paper_daily.md` 的标题索引，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors`、`OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data` 与 `DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification` 等全部已总结标题。
- 检索范围：围绕近 3-7 个月内 irregular sampled / asynchronous / sparse clinical time series classification / neural differential equations for irregular observations / critical-care time-series foundation model 的顶会、顶会 workshop、AAAI Proceedings、NeurIPS/OpenReview、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 `TESS` 这类 ICLR 2023 时间窗口过旧候选、`UniShape` 这类分类强但不原生处理 irregular sampling 的工作、`SELDON` 这类主要任务偏连续时间预测/插值而非分类的工作，以及 `RAINCOAT` 这类时间序列 domain adaptation 但非近月 IMTS 分类主线的旧工作。本次保留 2 篇全新工作：`Continuum Dropout` 是 AAAI 2026 正会论文，虽是通用 NDE 正则化方法，但实验直接覆盖不规则、缺失、类别不平衡的时序分类；`Towards Self-Supervised Foundation Models for Critical Care Time Series` 是 NeurIPS 2025 TS4H workshop 工作，直接面向稀疏、不规则 ICU 时序的自监督预训练和跨数据集死亡风险分类迁移。

## 1. Continuum Dropout for Neural Differential Equations

- 会议：AAAI 2026 Technical Track on Machine Learning IV
- 作者：Jinsung Lee, YongKyung Oh, Sungil Kim, Dong-Young Lim
- AAAI 页面：https://ojs.aaai.org/index.php/AAAI/article/view/39442
- 论文：https://arxiv.org/abs/2511.10446
- DOI：https://doi.org/10.1609/aaai.v40i27.39442
- 关键词：neural differential equations, irregular observations, missing values, time-series classification, continuous-time dropout, uncertainty calibration

### 场景、任务与核心难点

这篇工作面向 Neural ODE / Neural CDE / Neural SDE 等连续时间模型在不规则观测数据上的分类与不确定性估计。典型场景包括医疗、生理传感器和语音等连续时间信号：输入可能有不均匀时间戳、缺失观测、噪声和类别不平衡；模型需要在连续潜状态上演化，而不是沿固定离散层堆叠计算。论文的时序分类实验覆盖 Speech Commands 和 PhysioNet Sepsis，并在 PhysioNet Sepsis 中比较是否使用 Observation Intensity (OI) 的设置，直接触及不规则观测强度与标签相关的问题。

核心难点是：传统 dropout 是离散层上的 Bernoulli mask，不能自然作用在连续时间动力系统中。若把普通 dropout 硬塞进 ODE/CDE/SDE 的向量场或隐状态，可能破坏连续轨迹的一致性，带来不稳定训练或错误的不确定性估计；但完全不用 dropout，又会让 NDE 在小样本、高噪声或强缺失场景中过拟合。Continuum Dropout 将 dropout 的开/关机制建模为 alternating renewal process：每个潜在维度在连续时间中随机处于 active evolution 或 inactive paused 状态，从而给连续动力学一个时间一致的随机正则化过程。测试时通过 Monte Carlo 采样获得更校准的预测概率。

### 审稿人视角：价值与不足

最有价值的思想是把“正则化”从离散网络层推广到连续时间随机过程，而不是只调 solver、改插值或换向量场。对不规则时序分类而言，这很重要：许多方法关注如何构造 control path 或 latent dynamics，却忽视了连续时间模型在高缺失、小样本和类别不平衡下同样会过拟合 observation pattern。Continuum Dropout 的 alternating-renewal 形式使随机失活与时间轴对齐，因而比朴素 dropout 更符合 NDE 的语义；它还能自然给出 Monte Carlo uncertainty，对临床风险预测这类需要校准的任务很有价值。论文报告该方法在多类 NDE backbone 上普遍提高 PhysioNet Sepsis AUROC，且多数提升具有统计显著性，这说明它不是只服务于某个特定架构的技巧。

不足在于，Continuum Dropout 主要解决连续时间模型的泛化和校准问题，并没有显式分解 patient state 与 sampling policy。PhysioNet Sepsis 中的 OI 设定说明观测强度本身可能含有预测信息，但论文并未系统讨论当 OI 的语义跨医院或跨测量制度改变时，dropout 后的模型是否仍然稳健。换句话说，它能降低过拟合和过度自信，却不能保证模型不再利用采样频率、缺失间隔或变量可见性中的 policy shortcut。另一个限制是，dropout rate / renewal process 的选择可能改变连续动力学的有效时间尺度，需要更细的消融来判断正则化是否抹掉了真正罕见但关键的临床事件。

### 对 Sampling-Policy Shift 的启发

Continuum Dropout 对 Sampling-Policy Shift 的横向启发是：策略偏移下的不确定性本身应成为诊断对象。若一个不规则时序分类器在训练采样政策下很自信，但在换医院、换测量频率或换告警触发规则后不确定性没有上升，就说明模型缺少 policy-shift awareness。Continuum Dropout 的连续时间 Monte Carlo 机制可用于估计“同一潜在状态在不同观测策略下”的 epistemic / policy uncertainty，帮助区分状态证据不足与采样政策不匹配。

纵向深化上，可以把 Continuum Dropout 改造成 policy-conditioned dropout：state dimensions 使用跨策略共享的 renewal process，policy-sensitive dimensions 的 active/inactive rate 则由观测密度、变量联测、delta-t 分布或医院/设备环境调节。训练时对同一潜在轨迹施加不同反事实采样策略，约束 state representation 和分类 logits 稳定，同时要求 policy uncertainty 能随着采样制度偏移而升高。这样能把“连续时间正则化”推进到“连续时间策略偏移校准”：不只是让 NDE 更不容易过拟合，还让它知道什么时候自己的分类依据可能来自不可迁移的采样政策。

## 2. Towards Self-Supervised Foundation Models for Critical Care Time Series

- 会议/状态：NeurIPS 2025 Workshop on Learning from Time-Series for Health (TS4H) Poster；arXiv 2025-09
- 作者：Katja Naasunnguaq Jagd, Rachael DeVries, Ole Winther
- OpenReview：https://openreview.net/forum?id=w1hriS6A90
- 论文：https://arxiv.org/abs/2509.19885
- 代码：https://github.com/Katja-Jagd/YAIB
- 关键词：critical care time series, self-supervised foundation model, Bi-Axial Transformer, sparse irregular ICU data, transfer learning, mortality classification

### 场景、任务与核心难点

这篇工作面向 ICU critical-care time series 的自监督基础模型预训练。输入来自 MIMIC-III、MIMIC-IV 与 eICU 等 ICU 数据集，由生命体征、化验和静态人口学变量组成；每位患者的多变量观测在时间和变量维度上都高度稀疏且不规则。下游任务是将预训练模型迁移到未见过的数据集上做死亡风险分类，尤其关注标注样本少于 5,000 的低资源场景。

核心难点在于，ICU 时序基础模型很难像文本或图像那样直接依赖海量同质语料。不同医院的数据 schema、采样频率、缺失模式和患者结构差异明显；如果只在一个数据集上从零训练监督模型，跨数据集迁移弱，且低标注场景下容易过拟合。作者基于 Bi-Axial Transformer (BAT) 构建 early-stage critical-care foundation model：BAT 同时沿时间轴和特征轴做 axial attention，并在输入中显式包含观测值、missingness indicator、feature identity embedding 与连续时间位置信息。自监督阶段采用动态 observation / forecasting window 采样，并用 masked MSE 只在未来窗口真实观测位置上计算损失，避免为了预训练而强行填补未观测值。随后在 held-out ICU 数据集上微调死亡风险分类头。

### 审稿人视角：价值与不足

最有价值的贡献是把 ICU 不规则时序的 foundation-model 训练放进一个可复现、多数据集、跨域迁移的实验框架，而不是只在单个 benchmark 上调一个更强分类器。YAIB 框架、leave-one-dataset-out 预训练/微调设置和 head-only fine-tuning 对真实部署很有意义：它们直接检验模型是否学到了可迁移的 critical-care representation，而不是只记住某个医院的统计规律。BAT 的双轴 attention 也很适合 ICU 数据，因为临床判断往往同时依赖“同一变量随时间变化”和“同一时间邻域内多变量之间的关系”。在小样本微调中预训练模型优于从零训练 baseline，说明不规则 ICU 预训练确实可能缓解标注稀缺。

不足在于，这仍是 workshop 级别的 early-stage foundation model，模型规模、任务覆盖和偏移诊断都还比较初步。更关键的是，自监督 forecasting objective 会学习未来会被观测到的值，但“未来被观测到什么”本身由病情和医院采样政策共同决定。动态窗口采样提高了训练多样性，却不等于消除了跨医院 measurement policy 差异；如果 eICU 与 MIMIC 的测量流程、化验触发规则或记录粒度不同，预训练表征可能仍混合 patient state 与 site-specific observation process。论文展示了跨数据集死亡风险迁移，但还没有报告 policy-only classifier、采样密度分层、变量联测稳定性或反事实采样协议下的系统评估。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：预训练阶段本身必须纳入策略偏移问题。若 foundation model 在 pooled ICU 数据上通过 masked forecasting 学习，它不仅学习生理轨迹，也学习各医院“何时测、测什么、哪些值会出现在未来窗口”。因此，预训练数据的跨院多样性是必要但不充分的；还需要显式构造 policy descriptors，例如采样密度、变量可见性、联测图、long-gap pattern、routine rhythm deviation 和 value-pending 结构，并检查预训练 embedding 中这些 policy descriptors 是否可被轻易预测。

纵向深化上，可以把 BAT/YAIB 预训练扩展为 state-policy 双目标 foundation model：state objective 预测跨采样策略稳定的病程状态或风险相关 latent target，policy objective 单独预测未来 observation process，例如哪些变量会被测、何时会被测、观测密度如何变化。对同一患者轨迹生成不同反事实采样视图时，要求 state embedding、head-only fine-tuning logits 和死亡风险排序保持一致，同时允许 policy embedding 区分医院协议和采样制度。这样能把 ICU foundation model 从“跨数据集预训练更好迁移”推进到“预训练时显式隔离采样政策，分类时主要依赖稳定病程状态”。
