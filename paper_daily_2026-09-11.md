# Paper Daily - 2026-09-11

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`；同时读取兼容入口 `paper_daily.md` 的标题索引，并参考自动化记忆中 2026-08-03 至 2026-08-25 的新增标题。
- 本次黑名单论文标题已覆盖既有 50+ 篇历史总结，包括 `Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness` 与 `Shared-Benchmark Regime Decomposition for Nonstationary Clinical Decisions` 等全部已总结标题；新候选均已逐项排除黑名单重名。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / EHR event-stream foundation model / measurement-policy shift / clinical workflow distribution shift 的顶会、顶会 workshop、OpenReview、ICML virtual page 与 arXiv 论文页面。
- 已排除全部黑名单论文。严格的“顶会正会 + 直接 IMTS 分类”新增命中已基本被历史日报覆盖；本次保留 2 篇未在黑名单中的新跟踪对象：`ORA` 有 ICML 2026 virtual 记录，直接把不规则 EHR 事件建模为 marked time-to-event 预训练目标，并在多个临床分类任务上验证；`Learning Clinical Representations Under Systematic Distribution Shift` 是 2026-03 arXiv 预印本，尚未确认顶会录用，但它直接把 measurement policy、documentation practice 与 institutional workflow shift 作为表征解耦目标，对 Sampling-Policy Shift 的研究问题高度贴近。

## 1. One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models

- 简称：ORA
- 会议/状态：ICML 2026 virtual 记录；arXiv 2026-02
- 作者：Zilin Jing, Vincent Jeanselme, Yuta Kobayashi, Simon Lee, Chao Pang, Aparajita Kashyap, Yanwei Li, Xinzhuo Jiang, Shalmali Joshi
- 官方页：https://icml.cc/virtual/2026/71742
- 论文：https://arxiv.org/abs/2602.00541
- OpenReview PDF：https://openreview.net/pdf?id=WnxEI2t9be
- 关键词：irregularly sampled EHR, marked time-to-event, EHR foundation model, event timing, continuous measurements, downstream clinical classification

### 场景、任务与核心难点

ORA 面向结构化 EHR foundation model 的预训练。EHR 轨迹不是普通文本序列：一次化验、用药或生命体征记录既有事件类型，也有发生时间，还可能携带连续数值或剂量；这些事件天然不规则采样，并且不同医院、科室和任务中的事件密度、测量变量和时间间隔差异很大。下游评估覆盖 MIMIC-IV 与 CUMC 等数据上的多个临床分类、回归和 time-to-event 任务，其中分类任务包括常见的风险预测和临床结局预测。

论文解决的核心难点是：现有 EHR FM 多借鉴 NLP 的 next-token prediction，但“预测下一个 code”并不能完整刻画 EHR 的生成结构。对临床序列而言，何时发生某事件、事件对应的数值是多少、异常值如何改变后续事件风险，都是模型应学习的对象。ORA 因此把 EHR 视为 marked point process，提出 marked time-to-event pretraining objective，同时建模事件时间和 associated measurements；该 objective 可接入 Transformer 与 Mamba 等不同 backbone，并在多类下游任务上相对 next-token prediction 和忽略连续测量的预训练损失取得更好的泛化表现。

### 审稿人视角：价值与不足

最有价值的思想是把 EHR 预训练的目标函数从“语言式离散 token 续写”推进到“对不规则临床事件生成过程的联合似然建模”。这比单纯扩大模型参数或改变 tokenizer 更本质：它承认 EHR 中的时间间隔和连续测量值本身就是临床语义的一部分。对审稿人而言，ORA 的架构无关性也很重要，因为同一个损失在 Transformer 与 Mamba backbone 上都带来收益，说明贡献主要来自 pretraining objective 与数据结构的匹配，而不是某个特定网络的偶然优势。

不足在于，ORA 仍可能把 patient state 与 observation/care process 混在同一个 marked event likelihood 中。EHR 事件何时出现不仅由真实病程决定，也由医生下单、医院流程、床位资源、检查返回延迟和记录习惯决定；如果目标函数鼓励模型预测完整事件时间和测量值，它也会学习“这个医院如何观察病人”。论文强调跨数据集和多任务泛化，但还需要更细的 cross-policy、cross-site 和反事实采样评估，来证明 ORA 表示中的时间事件结构不是依赖训练机构的 workflow shortcut。

### 对 Sampling-Policy Shift 的启发

ORA 对 Sampling-Policy Shift 的横向启发是：采样政策偏移应进入预训练目标设计，而不仅是下游分类阶段的鲁棒性补丁。marked time-to-event loss 可以被拆成 state-event likelihood 与 policy-observation likelihood：前者刻画真实病程会产生什么临床状态，后者刻画医院会在何时记录、测量或干预。若只最大化合并似然，模型会自然利用 informative sampling；若显式拆分，则可以保留采样信息用于校准和偏移诊断，同时限制其直接污染分类边界。

纵向深化上，可以设计 policy-factorized ORA：同一条潜在患者轨迹生成多个反事实 observation policy 视图，例如固定节律采样、告警后密集采样、成本约束变量选择和医院特定联测规则。训练时要求 state representation、future-state likelihood 和分类 logits 在这些策略视图下保持一致；另一个 policy head 则预测观测事件时间、变量可见性和 value-pending 结构。这样能把 ORA 的“更结构化 EHR 预训练”推进到“显式分离病程事件与采样政策事件”的 foundation model。

## 2. Learning Clinical Representations Under Systematic Distribution Shift

- 会议/状态：arXiv 2026-03 预印本，尚未确认顶会录用
- 作者：Yuanyun Zhang, Shi Li
- 论文：https://arxiv.org/abs/2603.07348
- 关键词：clinical representation learning, systematic distribution shift, measurement policy, documentation practice, institutional workflow, adversarial environment regularization, invariant risk penalty

### 场景、任务与核心难点

这篇工作面向多机构临床预测中的系统性分布偏移。临床模型往往在一个医院或少数医院训练，却要部署到测量政策、文书规范、传感器采样率、检查流程和患者结构都不同的新机构；这会导致表征同时编码真实生理状态和 practice-specific artifacts。论文的任务覆盖多个 longitudinal EHR prediction 场景，并把医院作为环境变量来评估跨机构泛化、校准和 OOD 表现。

核心难点在于，传统自监督或监督表征学习会奖励一切能预测标签的信号，包括医院特定的观测流程。某个化验是否被测、记录是否密集、某类文书何时出现，在训练医院中可能与风险强相关，但换到另一家医院后语义会改变。作者因此提出 practice-invariant representation learning：将临床观测看成 latent physiologic factors 与 environment-dependent processes 的混合结果，在监督风险最小化之外加入 adversarial environment regularization 和 invariant risk penalties，压制 embedding 中可预测医院环境的信息，同时保持对临床结局的判别能力。论文报告该策略在跨机构预测中带来约 2-3 个 AUROC 点的 OOD 提升，并改善校准。

### 审稿人视角：价值与不足

最有价值的思想是把“医院实践差异”明确作为表征学习中的干扰因素，而不是把它笼统归入 covariate shift。对审稿人来说，这一点非常关键：在医疗 AI 中，很多所谓高性能模型利用的是 measurement policy、documentation habit 或 workflow artifact，而不是可迁移的病理机制。将环境对抗和 invariant risk penalty 结合，提供了一个可操作的 state-vs-practice 解耦框架，也比单纯跨医院微调更接近真实部署前的鲁棒性约束。

不足在于，这仍是未确认顶会录用的预印本，应作为高相关研究信号而非成熟正会结论跟踪。方法依赖可用且可信的环境标签，例如医院、科室或采样制度；如果环境划分过粗，adversarial objective 可能压掉有用的群体差异，若划分过细又会导致不稳定。另一个风险是 invariance 假设本身：并非所有医院差异都是 spurious，有些流程差异可能反映真实治疗路径或病例组合，需要通过因果和临床知识判断哪些 policy signal 应被剥离、哪些应进入校准。

### 对 Sampling-Policy Shift 的启发

这篇论文对 Sampling-Policy Shift 的启发最直接：我们的核心问题可以被表述为 practice-invariant representation learning 在非规则采样时序分类中的特例。采样政策、测量频率、变量联测、value-pending 和文书密度都可以视为 environment-dependent process；模型应学习跨政策稳定的 patient-state representation，同时保留 policy representation 用于偏移检测和不确定性校准。

纵向深化上，可以把该框架具体化到 IMTS：先为每条样本构造 policy descriptors，如 delta-t 分布、变量可见性、采样密度、routine-rhythm deviation、告警后密集观测和医院/科室 ID；再训练 state encoder + policy encoder。分类头主要依赖 state encoder，并通过环境对抗、IRM penalty 和反事实采样一致性限制其使用 policy encoder 的 shortcut。评估时报告 policy-only classifier 的可预测性、cross-policy AUROC/AUPRC drop、state embedding 的环境可分性和 counterfactual-policy consistency。这样可以把抽象的 systematic distribution shift 方法落到“非规则采样下采样策略偏移”的可测、可优化目标上。
