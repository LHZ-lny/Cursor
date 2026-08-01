# Paper Daily - 2026-08-01

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / informative missingness / multimodal clinical time-series 的顶会或顶会相关论文，重点核对 ACL 2026 Findings、ICLR 2026 OpenReview、ICML 2026、AAAI 2026、NeurIPS 2025 TS4H、ICASSP 2026 与 arXiv。
- 已排除全部黑名单论文；同时排除 RAxSS 这类 NeurIPS 2025 workshop 且更偏 variable-length dense iEEG 的条目、CISM 这类目前仅为 arXiv 预印本的候选，以及偏 forecasting / generation / imputation 或普通规则时序分类的结果。本次保留全新工作 1 篇：OPL-MT-MNAR 是 ACL 2026 Findings 收录论文，虽来自医疗 NLP / clinical decision-making 方向，但直接建模结构化临床时序与文本记录的 informative missingness，并把动态患者状态、结局分类和离线治疗策略学习联通，对 Sampling-Policy Shift 有非常直接的纵深价值。

## 1. Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness

- 方法名：OPL-MT-MNAR
- 会议：Findings of ACL 2026
- 作者：Zihan Liang, Ziwen Pan, Ruoxuan Xiong
- ACL Anthology：https://aclanthology.org/2026.findings-acl.1313/
- arXiv：https://arxiv.org/abs/2604.21235
- 代码：https://github.com/CausalMLResearch/OPL-MT-MNAR
- 关键词：multimodal clinical time-series, informative missingness, MNAR, Bayesian filtering, outcome prediction, offline treatment policy learning, ICU sepsis

### 场景、任务与核心难点

这篇工作面向 ICU 脓毒症护理中的多模态临床时间序列表征学习。输入同时包含结构化生命体征/化验记录和临床文本记录：前者是不规则采样的数值测量，后者是不规则出现的护理记录、医嘱或文本更新。下游任务包括 post-72-hour mortality prediction 这类患者结局分类，以及离线治疗策略学习。实验覆盖 MIMIC-III、MIMIC-IV 和 eICU，强调不同数据源与不同记录制度下的泛化。

核心难点在于，临床观测不是随机缺失。生命体征、化验和文本记录的出现频率往往由患者潜在病情、医生决策、科室流程和文档习惯共同决定；高危患者可能被更密集地抽血和记录护理文本，低危患者则可能更稀疏。论文将这种 observation process 明确视为 informative missingness / MNAR，而不是简单的噪声或需要填补的空白。方法上，OPL-MT-MNAR 用 MNAR-aware multimodal encoder 捕捉结构化数据与文本数据各自的观测模式；再用 Bayesian filtering / variational latent state 维护随时间更新的患者 belief state；最后把这个 posterior patient state 同时用于结局预测和离线治疗策略优化。结果显示，模型在 MIMIC-III 上达到 post-72-hour mortality AUROC 0.886，并在三套 ICU 数据上提升 offline policy learning 的 FQE。

### 审稿人视角：价值与不足

最有价值的思想是，它把“缺失/记录行为是否有信息”从辅助特征层面提升到动态状态学习和策略学习层面。许多 IMTS 分类模型会把 mask、delta-t 或 observation count 拼到输入里，但仍然把最终目标限定为分类性能；OPL-MT-MNAR 进一步指出，观测过程本身是临床决策闭环的一部分：它既反映患者状态，也影响后续治疗动作和记录行为。把 MNAR-aware encoder、action-conditioned latent dynamics 和 outcome/policy downstream heads 放在同一框架下，使模型不只是预测风险，也试图学习能支持序贯决策的动态患者状态。

另一点价值是对多模态 observation process 的拆分。结构化化验、生命体征和临床文本并不遵循同一种采样机制：生命体征更接近 routine monitoring，化验更受医生下单触发，文本更依赖文档流程和护理负荷。论文为文本设计 documentation process factor，并让它参与跨模态融合，这比把所有 modality 的 missingness 当成同质 mask 更接近真实 ICU 数据生成机制。

不足也很明确。首先，论文的目标包含离线治疗策略学习，和纯粹的 irregular sampled time series classification 并不完全重合；其分类结果服务于 learned patient state 的验证，而不是专门为 IMTS classifier 设计最优结构。其次，方法显式利用 MNAR 与记录密度，这在同院同流程下非常有效，但也最容易吸收医院政策 shortcut：某家医院中“频繁记录护理文本”可能代表高危，另一家医院中可能只代表文档规范不同。论文虽然跨 MIMIC/eICU 做了实验，但还没有把采样政策作为可控环境变量做反事实实验，也没有明确分离 patient-state MNAR 与 hospital-policy MNAR。对审稿人来说，最大的风险是 learned state 同时包含病程状态和制度状态，进而在部署到新医院或新记录政策时产生错误可迁移性。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发非常直接：采样政策不只决定“哪些数值缺失”，也决定“哪些文本会出现、何时出现、记录密度如何变化”。因此，我们研究非规则采样分类时，不应只建模数值变量的 mask/delta-t，还应把多模态记录行为作为 policy context。结构化测量、临床文本、医嘱和护理记录可以共同构成一个 observation-policy trace，用来解释为什么同一潜在状态在不同医院中呈现出完全不同的可见轨迹。

纵向深化上，可以把 OPL-MT-MNAR 改造成显式 state-policy decomposition：state branch 用观测值、文本语义和稳定临床事件学习跨策略不变的病程状态；policy branch 用 observation counts、missing rates、windowed frequencies、documentation density 和 modality availability 学习医院/科室/记录流程；分类头主要依赖 state branch，而 policy branch 只用于校准、偏移诊断或不确定性估计。训练时可对同一潜在病程构造不同反事实采样/记录策略，约束 state posterior、mortality logits 和关键临床语义保持一致，同时允许 policy posterior 区分不同观测制度。

更进一步，这篇论文提示我们把 Sampling-Policy Shift 从静态 representation invariance 推进到动态 POMDP / filtering 层面：采样政策是 observation function 的一部分，而不是输入噪声。若能在连续时间或离散决策步中同时学习 patient state transition 与 observation policy transition，就可以评估模型到底是在追踪病情演化，还是在追踪医院如何观察病人。对我们的研究而言，一个强基准可以是：固定患者状态轨迹和治疗动作，替换测量/文本记录策略，要求 state belief 和分类结论稳定，但 policy belief、校准置信度和缺失解释随策略改变而变化。
