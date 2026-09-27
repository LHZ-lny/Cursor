# Paper Daily - 2026-09-27

## 检索与去重记录

- 已强制读取根目录下全部已发现的日期版 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`、`paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`、`paper_daily_2026-09-24.md`、`paper_daily_2026-09-25.md`、`paper_daily_2026-09-26.md`；同时用标题级检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充黑名单。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`、`INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction`、`Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models`、`MedFuse: Multiplicative Embedding Fusion for Irregular Clinical Time Series`、`ReDiTT: Retrieval Augmented Conditional Diffusion Transformers for Asynchronous Time Series`、`DynaMamba: Multi-scale Dynamic Interacting Mamba Network for Irregular Clinical Time Series Classification`、`Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring` 与 `Temporally Detailed Hypergraph Neural ODE for Disease Progression Modeling` 等全部已总结论文。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / ICU mortality prediction / reliability-aware missingness / sampling-policy shift 的 ICLR、ICML、NeurIPS、AAAI、OpenReview、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 `Chain-of-Influence`（ICLR 2026 desk-rejected submission，且 MIMIC-IV 部分先聚合到 1 小时网格再保留 mask/time-gap，非原生 IMTS 分类主线）、`ReIMTS`（ICLR 2026 但主任务为 forecasting）、`From Leads to Latents / LAMAE`（ECG 多标签诊断，非不规则采样主线）、`The hidden risks of temporal resampling in clinical reinforcement learning`（与 sampling-policy shift 高相关但任务是 clinical RL 而非分类）等候选。严格的“近月顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 1 篇未在黑名单中的高相关新工作：`ReTAMamba` 是 2026-05 arXiv 预印本，尚未确认顶会录用，但它直接面向 MIMIC-IV、eICU、PhysioNet 2012 上的 irregular clinical time-series in-hospital mortality prediction，并把 missingness、elapsed time、变量特异可靠性衰减、多尺度聚合与 Mamba 线性序列建模合在一个框架中，对 Sampling-Policy Shift 下的“信息新鲜度 vs 观测政策痕迹”有新增价值。

## 1. ReTAMamba: Reliability-Aware Temporal Aggregation with Mamba for Irregular Clinical Time Series Prediction

- 会议/状态：arXiv 2026-05 预印本，尚未确认顶会录用
- 作者：Jinwoong Kim, Sangjin Park
- 论文：https://arxiv.org/abs/2605.16380
- 关键词：irregular clinical time series, in-hospital mortality prediction, Mamba, reliability-aware aggregation, informative missingness, multi-scale token routing

### 场景、任务与核心难点

ReTAMamba 面向 ICU 不规则临床时序上的早期院内死亡风险预测。实验统一使用 MIMIC-IV、eICU 与 PhysioNet 2012，并取 ICU 入院后前 48 小时的多变量生命体征和化验记录作为输入。该任务的典型难点是：不同变量观测频率差异很大，某些变量长时间缺失，delta-t 与 mask 本身反映医生测量决策，且短期生命体征异常和长期病程趋势都可能影响死亡风险。

论文解决的核心问题不是简单把所有观测重采样成固定网格，而是如何在保留观测结构的同时，判断“过去观测现在还多可靠”。ReTAMamba 先把输入重构为 time-variable token sequence：每个时间-变量单元都有 value、mask、变量身份、时间位置和 elapsed-time/staleness 表示。随后 Reliability Gate 为每个变量学习正的衰减率，观测值本身可靠性为 1，缺失占位值的可靠性则随距离上一次观测的时间按变量特异速率指数衰减。模型再在 60/120/240 分钟等多尺度时间桶中做 reliability-weighted aggregation，加入统计摘要，通过 Chronological Weaving 把不同尺度的 summary token 重新按真实时间顺序组织，最后用 budgeted token router 控制序列长度并交给 Mamba 编码。

实验结果显示，ReTAMamba 在三个数据集上都取得最好 AUROC/AUPRC：MIMIC-IV 上约 0.861/0.548，eICU 上约 0.793/0.384，PhysioNet 2012 上约 0.847/0.520。消融显示去掉 Reliability Gate、统计增强、Chronological Weaving 或 token router 都会降低表现；作者还报告 eICU 中 heart rate、respiratory rate、blood pressure、temperature 等动态 vital 的学习衰减率高于更慢变化的 lab 变量，说明模型确实学到“不同临床变量的新鲜度半衰期不同”。

### 审稿人视角：价值与不足

最有价值的技术思想是把 missingness/time-gap 从静态辅助特征推进到“连续可靠性权重”。许多 IMTS 模型把 mask 和 delta-t 拼进输入后交给 Transformer/RNN，自行学习哪些缺失重要；ReTAMamba 则显式要求模型用变量特异的衰减函数判断 forward-filled 或占位信息还能信多久。这个设计贴近 ICU 语义：心率和血压过几小时就可能过期，而某些 lab 的临床含义保留更久。Chronological Weaving 也有价值，因为它避免多尺度摘要各自并行而失去真实时间顺序，使短期恶化和长期趋势可以在同一时间轴上被 Mamba 线性复杂度处理。

不足在于，论文目前是预印本，尚不能按顶会正会结论看待；数据集也主要是标准 ICU mortality benchmark，缺少跨医院采样政策、跨科室测量协议或反事实 observation policy 的系统测试。更关键的是，Reliability Gate 学到的“新鲜度”可能同时包含生理时效性和医院测量政策：某变量长时间未测，可能代表该信息仍稳定，也可能代表医生认为患者风险较低或该医院不常规复测。模型在 benchmark 上利用这种结构提高 AUPRC 是合理的，但若部署到测量频率、变量联测和告警后采样策略不同的医院，学到的衰减率和 token routing 偏好未必稳定。

### 对 Sampling-Policy Shift 的启发

ReTAMamba 对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以被具体化为“信息可靠性函数”的偏移。不同医院对同一变量的复测间隔不同，意味着相同的 elapsed time 在不同环境下可能有不同语义；如果模型把训练医院的衰减率当作普遍生理规律，就会把 observation policy 误认为 patient state。因此，评估非规则采样分类器时，不应只比较 mask ratio 或 delta-t 分布，还应比较 learned reliability decay、multi-scale token selection、routing budget usage 和 staleness distribution 是否跨环境稳定。

纵向深化上，可以把 ReTAMamba 改造成 state-policy dual reliability model。state reliability 学习跨采样政策稳定的生理时效性，例如不同变量真实信息半衰期；policy reliability 学习医院流程、告警后密集采样、变量联测和资源约束造成的观测新鲜度差异。分类主路径只使用 state reliability 加权后的 token，policy reliability 则用于偏移告警、不确定性校准或拒识。训练时可对同一潜在病程生成不同采样策略视图，约束 state decay、state token selection 和死亡风险 logits 保持一致，同时允许 policy decay 和 router 分布识别采样制度。这样能把 ReTAMamba 的“可靠性感知聚合”推进到“区分生理信息过期与观测政策过期”的 Sampling-Policy Shift 解决框架。
