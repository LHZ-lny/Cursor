# Paper Daily - latest

> 最新日期版原始记录见 `paper_daily_2026-06-12.md`。本文件作为自动化提交流程的兼容入口，保留本次新增论文摘要。

## 追加更新 - 2026-06-12 23:00 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`。
- 本次黑名单论文标题：
  - Adaptive Time Encoding for Irregular Multivariate Time-Series Classification
  - Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification
- 检索范围：近 3-7 个月内围绕 irregular sampled / irregular multivariate time-series classification 的顶会论文，重点核对 AAAI 2026、ICLR 2026、ICML 2026 官方页面与论文页。
- 已排除黑名单论文，并仅保留全新工作 2 篇。

## 1. FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification

- 会议：AAAI 2026 Technical Track on Machine Learning VI
- 作者：YongKyung Oh, Dong-Young Lim, Sungil Kim
- 链接：https://ojs.aaai.org/index.php/AAAI/article/view/39643
- DOI：https://doi.org/10.1609/aaai.v40i29.39643
- 关键词：irregularly-sampled time series classification, Neural CDE, control path, invertible neural flow, data-adaptive manifold

### 场景、任务与核心难点

这篇工作面向稀疏、不规则采样时序的分类任务，典型应用包括医疗监测、传感器数据和真实世界事件序列。它关注的核心难点是：连续时间模型虽然天然适合处理非均匀观测，但 Neural CDE 等方法通常需要先把离散观测构造成一条 control path；如果使用线性插值、样条或其他固定插值方案，就会把很强的几何假设硬编码进模型。在高缺失、采样间隔变化大或观测点分布不均时，这条人工路径可能偏离真实数据流形，进而影响下游分类。

FlowPath 的做法是让模型学习 control path 的几何形状。它用 invertible neural flow 构造连续、数据自适应且信息保持的流形路径，而不是简单连接观测点。这样，分类器看到的不只是“观测值随时间如何变化”，还包括“离散观测应如何被嵌入到连续时间轨迹中”这一层可学习结构。

### 审稿人视角：价值与不足

最有价值的技术思想是把不规则采样分类中的“插值路径选择”提升为核心建模对象。很多连续时间模型把插值当作前处理或默认组件，FlowPath 则指出 path geometry 本身决定了模型如何理解稀疏观测，并用可逆流约束避免 unconstrained learnable path 带来的信息丢失或轨迹退化。这一点对高缺失场景尤其重要，因为固定插值会在观测空窗中引入虚假的平滑性。

不足是论文主要围绕控制路径几何与分类精度展开，尚未充分区分“真实状态动力学导致的观测几何”与“采样政策导致的观测几何”。如果训练环境中某些类别被更频繁测量或在特定时间段被重点监测，FlowPath 学到的数据自适应流形可能同时吸收状态动力学与采样调度规则。可逆性保证信息不丢失，但不保证保留的信息都是跨环境稳定的因果信号。

### 对 Sampling-Policy Shift 的启发

FlowPath 对 sampling-policy shift 的横向启发在于：偏移不只发生在 mask 或时间戳分布上，也会改变由观测点诱导出的连续路径几何。因此，我们可以把“策略不变路径几何”作为一个新的研究对象：对同一潜在轨迹模拟不同采样策略，要求 learned path 在分类相关的几何量上保持稳定，而允许采样密度、局部路径弯曲或不确定性显式进入策略支路。

纵向深化上，可以在可逆流路径中加入 policy-conditioned / policy-adversarial decomposition：一部分流形坐标解释真实连续状态，一部分解释采样政策残差，并通过跨策略一致性约束限制分类头使用后者。FlowPath 的可逆结构还适合做反事实采样实验：固定观测值生成机制，替换采样时间策略，比较 path representation 和 logits 的变化，从而直接测量模型是否依赖 spurious sampling geometry。

## 2. One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification

- 简称：GSNF
- 会议：ICML 2026 Poster
- 作者：Mengzhou Gao, Kaiwei Wang, Pengfei Jiao
- 链接：https://icml.cc/virtual/2026/poster/64769
- 论文：https://arxiv.org/html/2605.10179
- 关键词：irregular multivariate time series classification, neural flows, graph-structured interactions, trajectory-level self-supervision, invertibility

### 场景、任务与核心难点

GSNF 面向不规则多变量时序分类，实验覆盖 PhysioNet12、P12、P19、MIMIC-IV、eICU 等医疗/ICU 类数据集。这类任务中，各变量异步观测、缺失率高、类别常常不平衡，而且变量之间存在强交互，例如生命体征与实验室指标之间的联动。

论文解决的核心难点是：Neural Flows 可以用 one-step mapping 高效建模连续时间轨迹，避免 ODE solver 的迭代成本；但 one-step 形式也带来问题，即变量间交互只被施加一次，缺少像 solver-based Graph ODE 那样在多步演化中反复修正交互的机会。GSNF 因此把图结构直接嵌入 flow dynamics，并设计两类辅助轨迹自监督：interaction-aware trajectory generation 通过重初始化暴露图诱导交互，reverse-time trajectory generation 利用可逆性约束前向/反向一致性。

### 审稿人视角：价值与不足

最有价值的地方是把“高效 one-step continuous-time modeling”和“多变量交互学习”结合起来。相比只在初始状态或 attention 层面学习变量关系，GSNF 让交互图参与轨迹演化本身；ITG 与 RTG 也不是普通的数据增强，而是针对 one-step flow 缺少迭代细化这一结构性短板设计的 trajectory-level supervision。消融结果显示去掉 graph 或去掉辅助轨迹监督都会明显降低性能，说明该设计不是装饰性模块。

不足在于，GSNF 默认学习到的变量交互图主要由训练数据中的观测值、mask 和局部片段统计支持，但这些统计可能混合了病理机制、测量流程和机构采样习惯。尤其在 ICU 数据中，某些化验项目是否被测、何时被测，本身受到医生决策和医院协议影响。论文证明了图交互对 benchmark 分类有效，却没有系统评估当采样政策变化时，学习到的 hub variables 和强边是否仍然稳定。

### 对 Sampling-Policy Shift 的启发

GSNF 对我们的启发非常直接：sampling-policy shift 可能不仅改变单变量缺失模式，还会改变模型估计出的变量交互图。例如某一医院频繁联测 lactate 与 WBC，另一医院只在危重患者中联测，那么图上的强边可能代表采样流程，而不一定代表稳定生理耦合。

因此，可以把 GSNF 的图结构扩展为 policy-aware graph：将边分解为 invariant physiological edges 与 policy-induced edges，前者进入分类主路径，后者用于解释观测机制或作为不确定性信号。ITG 的重初始化思想也可用于策略反事实：在不同采样策略下重初始化同一潜在轨迹，要求核心交互边和分类表示保持一致；RTG 的 forward-backward consistency 则可扩展为 policy-cycle consistency，即从策略 A 的观测路径映射到策略 B 再映回 A，约束状态表征不被采样调度破坏。

## 追加更新 - 2026-06-14 23:02 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily*.md`：发现并读取 `paper_daily.md`、`paper_daily_2026-06-12.md`。
- 本次黑名单论文标题：
  - Adaptive Time Encoding for Irregular Multivariate Time-Series Classification
  - Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification
  - FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification
  - One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification
  - SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data
  - GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification 的顶会论文，重点核对 ICLR 2026、AAAI 2026、ICML 2026 官方页面、OpenReview 与论文页。
- 已排除黑名单论文；同时排除偏 forecasting、普通规则时序分类、ICML 2025 旧论文或 workshop 条目。本次仅保留全新工作 1 篇。

## 5. PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks

- 会议：ICLR 2026 Poster
- 作者：Francesco Spinnato, Cristiano Landi
- 链接：https://openreview.net/forum?id=qetBM8nLkf
- 官方页：https://iclr.cc/virtual/2026/poster/10007224
- 关键词：irregular time series, classification benchmarks, naturally irregular datasets, unified array format, reproducible evaluation

### 场景、任务与核心难点

PYRREGULAR 面向不规则时序分类的标准化评测，而不是提出单一新模型。它覆盖 mobility、healthcare、environmental science 等场景中常见的三类不规则性：采样频率不一致、观测时长不同，以及真实缺失/部分观测。论文指出，现有 ITS/IMTS 分类研究常把问题拆散处理：有的只关注人为注入 missingness 的规则数据，有的只在单个医学 benchmark 上比较，有的代码格式和输入约定彼此不兼容，导致新模型之间很难公平复现和横向比较。

这篇工作的核心难点是把“真实不规则性”变成可复用、可比较、可扩展的 benchmark 基础设施。作者构建了一个统一框架和标准化数据仓库，包含 34 个 naturally irregular datasets，并在其上评测 12 个原生支持不规则时序分类的分类器。其 common array format 试图在 xarray 的灵活性与 sparse COO 表示的内存效率之间取得平衡，使不同来源、不同长度、不同采样密度的数据可以进入统一实验管线。

### 审稿人视角：价值与不足

最有价值的贡献是把不规则时序分类从“各论文自带数据处理脚本和局部 benchmark”的状态推进到可复现实验平台。对审稿人而言，这类工作的重要性不在于单次 accuracy 提升，而在于它能降低方法比较中的隐性变量：数据切分、缺失定义、时间轴编码、输入格式、baseline 适配等。论文还明确区分 naturally irregular data 与 artificially induced missingness，这一点对我们判断方法是否真的解决异步/不规则采样问题非常关键。

不足也比较清楚：它本质是 benchmark/framework 论文，技术建模创新有限，不能直接解决采样策略偏移下的稳健学习问题。当前覆盖任务主要是 classification，虽然作者说明框架可扩展到 regression/forecasting，但系统评测仍集中在分类。baseline 也受限于“已有代码能否原生支持该任务”，因此 SSM、latent ODE、foundation-model style 方法和更强的 policy-aware 方法尚未被系统纳入。此外，即使数据是 naturally irregular，也不等于 benchmark 已经显式标注或控制了采样政策来源；不同数据集之间的采样机制差异仍可能和领域、标签、机构流程混在一起。

### 对 Sampling-Policy Shift 的启发

PYRREGULAR 对我们的问题具有基础设施层面的横向启发：研究 sampling-policy shift 不能只依赖某个私有数据集或手工 mask 掉规则序列，而需要一个能表达多种真实不规则性的统一数据层。它的 naturally irregular dataset 选择标准和统一 array format 可以作为构建 policy-shift benchmark 的起点：在同一数据接口下显式记录观测时间、变量级 mask、观测窗口、采样频率与缺失结构，再额外加入 policy/environment 元数据或可控的反事实采样策略。

纵向深化上，可以在 PYRREGULAR 的 benchmark 之上增加“策略维度”的评测协议：例如按医院、设备、时间段、采样密度分位数或观测触发规则构造环境划分；对同一底层序列生成多种采样策略增强；报告 in-policy、cross-policy 与 counterfactual-policy 三类指标。这样可以把当前 irregular time series classification benchmark 从“谁在平均不规则性上准确率更高”推进到“谁在采样政策改变后仍保持稳定”。对于我们的 Sampling-Policy Shift 研究，PYRREGULAR 更像是评测基座：它提醒我们先把数据格式、自然不规则性和 baseline 比较标准化，再在其上定义 policy-invariant representation、policy-aware calibration 与 policy-causal diagnostics。

## 追加更新 - 2026-06-13 23:01 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily*.md`：发现并读取 `paper_daily.md`、`paper_daily_2026-06-12.md`。
- 本次黑名单论文标题：
  - Adaptive Time Encoding for Irregular Multivariate Time-Series Classification
  - Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification
  - FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification
  - One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification
- 检索范围：近 3-7 个月内围绕 irregular sampled / irregular multivariate time series classification / asynchronous clinical time series classification 的顶会论文，重点核对 ICLR 2026、AAAI 2026、ICML 2026 官方页面、OpenReview 与论文页。
- 已排除黑名单论文，并仅保留全新工作 2 篇。

## 3. SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data

- 全称：Super Mixing Additive Networks
- 会议：ICLR 2026 Poster
- 作者：Andrea Zerio, Maya Bechler-Speicher, Maor Huri, Marie Vestergaard, Tine Jess, Ran Gilad-Bachrach, Samir Bhatt, Aleksejs Sazonovs
- 链接：https://openreview.net/forum?id=1MVeSLvfxU
- 论文：https://arxiv.org/abs/2505.19193
- 关键词：temporally sparse heterogeneous data, irregular asynchronous signals, implicit graphs, interpretable graph learning, additive networks

### 场景、任务与核心难点

SuperMAN 面向由多种稀疏异步信号组成的预测/分类任务，典型场景包括例行血检构成的疾病风险预测、ICU 住院时长预测，以及事件日志或传播链上的分类。其输入不是整齐的时间网格，而是多个信号类型在不同时间、不同频率下产生的碎片化观测；如果强行重采样或插值，会损失单个检测项目的原始时间结构，也容易把缺失模式误当作稳定状态。

论文的核心难点是如何在不做信息损失式对齐的情况下，同时获得足够表达力和可解释性。SuperMAN 将每条稀疏时间轨迹建成隐式图，再把多条轨迹作为图集合处理；在 Graph Neural Additive Networks 的思想上扩展出 univariate、multivariate 与 subset-level 的建模路径，使模型既能在单个信号/节点层面解释，又能在有领域先验时把相关信号组合起来提升表达能力。

### 审稿人视角：价值与不足

最有价值的思想是把“异步时序分类”从序列补齐问题转化为“稀疏异质信号集合上的可解释图学习”。这种建模避免了规则网格化带来的时间戳扭曲，并且把可解释性设计成结构属性：节点级、图级、子集级重要性可以直接服务于医疗等高风险场景，而不是依赖事后解释器。它对现实数据尤其友好，因为真实系统中的变量通常天然分组，例如血液指标、炎症指标、用药事件或日志事件簇。

不足在于，SuperMAN 的解释对象仍然来自训练数据中观测到的信号与时间结构。如果某些检测项目的出现频率本身由医院流程、医生怀疑或保险策略驱动，那么节点/图重要性可能解释的是采样政策，而不是稳定的病理机制。论文强调 interpretable-by-design 和高风险任务表现，但对跨机构采样政策变化下解释是否保持语义稳定，还缺少系统检验。

### 对 Sampling-Policy Shift 的启发

SuperMAN 对 sampling-policy shift 的横向启发是：我们可以把采样策略偏移看作“隐式图集合分布”的偏移，而不只是 mask ratio 或 delta-t 的边缘分布变化。若某些变量在策略 A 中经常共同出现、在策略 B 中被拆开测量，那么图集合的组成、边权与子集重要性都会发生偏移。由此可以设计 policy-invariant graph-set objective：在同一潜在病程的不同采样策略增强下，约束关键子图和分类表征保持一致，同时允许采样诱导节点进入单独的 policy explanation 分支。

纵向深化上，SuperMAN 的 subset-level trade-off 提示我们可以显式定义“状态子集”和“策略子集”。前者承载跨环境稳定的生理或系统状态，进入分类主路径；后者解释观测为何出现、为何稀疏或为何成组出现，用于不确定性估计和偏移诊断。这样既不简单丢弃 informative missingness，也不让策略性 missingness 直接污染分类边界。

## 4. GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care

- 全称：Graph Attention-based Relational Learning for Intensive Care
- 会议：ICLR 2026 Poster
- 作者：Ruirui Wang, Yanke Li, Manuel Günther, Diego Paez-Granados
- 链接：https://openreview.net/forum?id=4ZAwmIaA9y
- 官方页：https://iclr.cc/virtual/2026/poster/10011543
- 关键词：irregularly sampled ICU time series, graph attention, exponential-decay encoder, time-lagged summary graphs, interpretable classification

### 场景、任务与核心难点

GARLIC 面向 ICU 多变量时序的临床结局预测，包括死亡风险、脓毒症或其他高风险事件分类。ICU 数据同时具有异步采样、高缺失、变量异质和强临床解释需求：生命体征可能连续记录，化验项目可能按病情触发，某些变量之间的时间滞后关系比同一时间点的相关性更重要。

论文解决的核心难点是如何在不规则采样下同时建模缺失、变量关系和可解释预测。GARLIC 使用可学习的 exponential-decay encoder 处理不规则缺失与旧观测衰减；用 time-lagged summary graphs 捕捉传感器/临床变量之间的滞后依赖；再通过 cross-dimensional sequential attention 融合全局模式。为了避免辅助重构目标和最终分类目标互相干扰，作者设计 alternating decoupled optimization，使插补/重构与分类训练更稳定。

### 审稿人视角：价值与不足

最有价值的技术思想是把临床可解释性放进模型主干，而不是作为后处理：时间步注意力、信号重要性和图边权都由端到端训练得到，能同时对应 observation-level、signal-level 和 edge-level 的解释。相比单纯的 decay imputation 或全局 attention，time-lagged graph 更贴近 ICU 场景中的因果滞后和变量联动，例如炎症指标、血流动力学和治疗响应之间的延迟关系。

不足是它仍然高度依赖 ICU benchmark 中的观测机制。exponential decay、time-lagged graphs 和 attention 权重会共同吸收“哪些变量被测、多久被测一次、哪些变量被联测”这些策略信息。若不同医院的化验协议、报警阈值或治疗路径发生变化，GARLIC 学到的高权重边可能代表临床流程共现，而不一定代表可迁移的生理关系。论文用 feature-removal 验证了解释 fidelity，但这不等同于验证解释在 sampling-policy shift 下的稳定性。

### 对 Sampling-Policy Shift 的启发

GARLIC 给我们的直接启发是：采样策略偏移会沿着三条路径进入分类器，分别是 decay encoder 的时间衰减、time-lagged graph 的边结构，以及 cross-dimensional attention 的变量融合。因此，研究 sampling-policy shift 时不能只评估最终 representation，还应分别检查衰减参数、滞后边权和注意力分布在不同策略下的变化。

纵向上，可以把 GARLIC 改造成 policy-robust relational learner：在图边上区分 physiology edges 与 protocol edges，并对前者施加跨策略一致性，对后者保留为策略诊断信号；在 decay encoder 中加入策略条件化校准，使“多久未测”不被自动解释为“状态稳定或异常”；在优化上引入 decoupled adversarial objective，让重构分支可以利用采样政策提高插补质量，但分类分支被约束为主要依赖跨策略稳定的关系表征。这样能够把 GARLIC 的解释性从“解释当前模型为何预测”推进到“解释预测依据是否会随采样政策改变”。

## 追加更新 - 2026-06-15 23:05 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily*.md`：发现并读取 `paper_daily.md`、`paper_daily_2026-06-12.md`。
- 本次黑名单论文标题：
  - Adaptive Time Encoding for Irregular Multivariate Time-Series Classification
  - Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification
  - FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification
  - One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification
  - PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks
  - SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data
  - GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular medical time series classification / irregular sequence diagnosis 的顶会论文，重点核对 ICLR 2026、AAAI 2026 官方页、OpenReview 与 arXiv/论文页。
- 已排除黑名单论文；同时排除偏 forecasting、普通规则 MedTS、workshop 条目、AAAI abstract reprint / journal-track 摘要或时间较旧的工作。本次仅保留全新工作 2 篇。

## 6. DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification

- 会议：ICLR 2026 Conference Paper
- 作者：Jian Chen, Xiaoyan Yuan, Yuxuan Hu, Jinfeng Xu, Yipeng Du, Xiangyu Zhao, Wei Wang, Edith C. H. Ngai
- OpenReview：https://openreview.net/forum?id=hqdkzm70E6
- 论文：https://arxiv.org/abs/2604.11842
- 关键词：irregular medical time series, asynchronous observations, patient-variable bipartite graph, node-specific temporal decay, clinical classification

### 场景、任务与核心难点

DBGL 面向不规则医疗时序分类，实验覆盖 P19、PhysioNet、MIMIC-III、P12 等临床数据集。输入中的变量采样频率高度异质，例如生命体征、化验指标和监护事件往往在不同时间点出现；同一患者内部存在变量级异步观测、长短不一的时间间隔和大量缺失。任务不是把这些记录补齐成规则网格后再分类，而是在保留真实观测时间和缺失结构的前提下学习患者状态表征。

论文针对的核心难点有两层：第一，常见插值、padding 或统一时间网格会扭曲原始采样不规则性，把观测间隔和 missingness pattern 中的信息打散；第二，不同临床变量的“过期速度”不同，血压、心率、乳酸、白细胞计数等指标对当前状态的有效时间窗并不一致。DBGL 因此构造 patient-variable bipartite graph，将患者节点与变量节点之间的边显式绑定真实观测时间和采样间隔，并设计 node-specific temporal decay encoding，让不同变量以可学习的速度随时间衰减。

### 审稿人视角：价值与不足

最有价值的技术思想是把不规则采样分类中的两个常被混合的问题拆开处理：用二部图承载“哪些变量在何时被观测”的结构，用变量特异的 decay encoding 承载“这次观测对当前状态还剩多少有效信息”。这比把 delta-t 简单拼接到 RNN/Transformer 输入里更结构化，也比全局共享 decay 更符合临床变量的异质性。二部图还避免了强行构造完整的 time-by-variable 矩阵，使模型更自然地处理异步观测。

不足在于，DBGL 明确把真实采样图样作为有效信号利用，但还没有充分区分“状态驱动的 informative observation”与“制度/流程驱动的 policy observation”。在 ICU 或 EHR 中，某变量何时被测往往由医生怀疑、医院协议、设备报警或保险流程触发；二部图的边和 node-specific decay 可能同时吸收病理机制与采样政策。论文报告了同分布 benchmark 上的分类提升，但对跨医院、跨设备或跨采样流程下图结构和 decay 参数是否稳定，仍缺少系统评估。

### 对 Sampling-Policy Shift 的启发

DBGL 对 sampling-policy shift 的横向启发非常直接：采样策略可以被表示成 patient-variable-time 三元图的分布变化，而不仅是缺失率或时间间隔的边缘变化。我们可以借鉴其二部图建模方式，把观测边分解为 state-informative edges 与 policy-induced edges：前者进入分类主路径，后者进入策略识别、偏移诊断或不确定性估计分支。

纵向深化上，node-specific decay 可以扩展成 policy-calibrated decay。也就是说，同一变量的观测有效期不应只由时间间隔决定，还应由采样策略环境决定：若某医院只有在病情恶化时才测某项指标，那么“长时间未测”的语义与常规监测医院完全不同。可设计跨策略一致性约束：对同一潜在病程施加不同采样策略增强，要求患者状态表征和分类 logits 稳定，同时允许二部图边密度、变量 decay 残差和策略分支输出变化。这样能把 DBGL 从“利用不规则采样信息”推进到“控制哪些采样信息可进入分类边界”。

## 7. Fault Diagnosis of Irregular Sequences by Adjoint Learning in Continuous-Time Model Space

- 会议：AAAI 2026 Technical Track
- 作者：Xiren Zhou, Chuyang Wei, Ao Chen, Shikang Liu, Xiangyu Wang, Huanhuan Chen
- 链接：https://ojs.aaai.org/index.php/AAAI/article/view/40141
- DOI：https://doi.org/10.1609/aaai.v40i34.40141
- 关键词：irregular sequences, fault diagnosis, continuous-time reservoir computing, model-space learning, adjoint ESN, limited labels

### 场景、任务与核心难点

这篇工作面向不规则序列上的故障诊断，本质上是工业系统、传感器系统或设备运行序列中的异常/故障类别识别。现实场景中，序列可能来自非均匀采样、传感器中断、事件触发式采集或不同工况环境；同时故障样本有限，训练数据不足使端到端大模型更容易过拟合局部观测图样。

论文解决的核心难点是：如何在不依赖固定时间步和大量标注的情况下，把不规则序列映射为稳定、可区分类别的表示。作者提出 continuous-time model space 的思路：先用 Continuous-Time Reservoir Computing Network (CT-Res) 拟合每条序列的连续时间动态，把原始序列从 data space 映射到由拟合模型参数/状态描述的 model space；再用 adjoint learning 引入与 CT-Res 共享结构和参数的离散时间 adjoint ESN，在拟合动态与类别判别之间联合优化，从而绕开直接 ODE solver 训练的高成本。

### 审稿人视角：价值与不足

最有价值的思想是把分类对象从“不规则观测点序列”转换为“能解释该序列动态的连续时间模型”。这种 model-space learning 对小样本和不规则采样尤其有吸引力：如果拟合模型真正抓住系统动力学，那么不同采样密度下的观测序列可被压缩到更稳定的动态描述中，分类器比较的是动态机制而非原始采样痕迹。adjoint ESN 的设计也有工程价值，它试图在连续时间表达能力和训练效率之间取得平衡。

不足是 model space 的稳定性依赖于拟合模型是否能把采样机制与系统动力学分开。如果传感器采样本身由故障状态、维修策略或运行环境触发，CT-Res 拟合到的动态参数可能仍然混入采样策略差异。论文强调 limited training data 与 varying underlying environments，但主要从故障诊断 benchmark 和效率角度验证方法，并未把 sampling-policy shift 作为显式实验变量，例如改变采样触发规则、传感器缺失机制或工况下的观测频率后再评估 model-space 表征是否保持类别语义。

### 对 Sampling-Policy Shift 的启发

这篇工作对我们的问题提供了纵向深化方向：与其只在 representation 层做 policy-invariant learning，也可以把“每条序列对应的连续时间生成/拟合模型”作为中间对象。若能学习一个将观测值动力学与采样策略分离的 model space，那么 sampling-policy shift 下的分类器就可以主要依赖系统状态动力学参数，而不是依赖观测时间戳的表面模式。

横向应用上，可以把 CT-Res / model-space learning 与反事实采样结合：固定同一底层连续轨迹，生成多种采样策略下的观测序列，要求它们映射到相近的 state-dynamics model coordinates；同时保留单独的 policy coordinates 解释为什么某些时间点被观测。adjoint learning 的联合优化也可扩展为三目标：动态拟合准确、分类可分、策略不变。这样既避免完全丢弃 informative missingness，也能显式惩罚“只要换采样政策，模型空间坐标就漂移”的不稳健解。

## 追加更新 - 2026-06-16 23:00 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification 的顶会论文，重点核对 AAAI 2026、ICLR 2026、ICML 2026 官方页面、OpenReview、AAAI Proceedings 与 arXiv 页面。
- 已排除黑名单论文；同时排除 SITS 这类 ICML 2025 workshop/时间较旧条目、STAR-Set 这类 ICLR 2026 workshop 条目、偏 forecasting/causal discovery/规则时序分类的工作。本次保留全新工作 2 篇。

## 8. Beyond Observations: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning

- 简称：iTimER
- 会议：AAAI 2026 Technical Track on Machine Learning V
- 作者：Jiexi Liu, Meng Cao, Songcan Chen
- 链接：https://ojs.aaai.org/index.php/AAAI/article/view/39545
- 论文：https://arxiv.org/abs/2511.06854
- DOI：https://doi.org/10.1609/aaai.v40i28.39545
- 关键词：irregularly sampled time series, self-supervised pretraining, reconstruction error distribution, pseudo-observations, Wasserstein alignment, classification/interpolation/forecasting

### 场景、任务与核心难点

iTimER 面向不规则采样时序的通用表示学习，并在分类、插值和预测三类下游任务上验证，其中分类实验覆盖医疗和人体活动等典型 ISTS 数据。它处理的场景是：观测时间非均匀、存在自然缺失，且未观测时间点不只是空白位置，而可能反映数据结构、模型不确定性和采样机制共同作用后的信息缺口。

论文解决的核心难点是，现有方法通常只围绕已观测值建模：要么用观测值插补未观测点，要么直接学习连续时间潜变量；但训练过程中产生的 reconstruction error 往往只被当作要最小化的损失，而没有被作为可利用的学习信号。iTimER 反过来建模观测点上的重构误差分布，并将从该分布采样得到的误差与最近观测值 mixup，生成未观测时间戳处的 pseudo-observations；随后用 Wasserstein 距离对齐观测区与伪观测区的误差分布，并用 contrastive learning 增强表示的判别性。

### 审稿人视角：价值与不足

最有价值的思想是把“模型在哪里重构不好”从失败信号转化为自监督信号。对于不规则采样数据，未观测点处没有监督目标，直接插值容易注入过强平滑假设；iTimER 用重构误差分布生成噪声感知的 pseudo-observations，相当于让模型在不确定区域也获得可训练的、分布对齐的目标。Wasserstein alignment 也比简单加入噪声更稳健，因为它显式约束伪观测区域的误差统计不要偏离观测区域太远。

不足在于，重构误差本身未必只反映状态动力学，也可能反映采样政策。若训练环境中某类患者、某些活动阶段或某种设备状态被更稀疏地观测，重构误差分布会吸收这种 policy-induced uncertainty。iTimER 将误差分布作为 informative signal，有助于同分布 ISTS 表示学习，但在跨医院、跨设备或主动采样规则改变时，伪观测机制可能把训练环境中的采样偏差带入表示空间。论文覆盖 classification、interpolation、forecasting，但没有把 sampling-policy shift 作为显式评估维度。

### 对 Sampling-Policy Shift 的启发

iTimER 对我们的横向启发是：采样策略偏移可以通过 reconstruction-error distribution 被显式观测和度量。与其只比较 mask ratio 或 delta-t 分布，我们可以比较不同策略环境下的重构误差分布、伪观测分布和表示漂移；若同一潜在轨迹在策略 A 与策略 B 下产生系统性不同的误差分布，就说明模型仍在依赖策略特定的不确定性结构。

纵向深化上，可以把 iTimER 改造成 policy-aware error modeling：将重构误差分解为 state uncertainty 与 policy uncertainty 两部分。前者用于生成与真实动力学相关的 pseudo-observations，后者进入策略诊断或不确定性校准分支，但不直接服务分类头。还可以在 Wasserstein alignment 中加入跨策略对齐项：对同一序列模拟不同采样策略，要求 state-error distribution 和分类 logits 稳定，同时允许 policy-error distribution 变化。这样能保留 informative missingness 的价值，又减少把采样政策误差当成类别证据的风险。

## 9. Can we generate portable representations for clinical time series data using LLMs?

- 方法名：Record2Vec
- 会议：ICLR 2026 Poster
- 作者：Zongliang Ji, Yifei Sun, Andre Carlos Kajdacsy-Balla Amaral, Anna Goldenberg, Rahul G. Krishnan
- 链接：https://iclr.cc/virtual/2026/poster/10007323
- OpenReview：https://openreview.net/forum?id=pXw0uRTSKT
- 论文：https://arxiv.org/abs/2603.23987
- 关键词：irregular ICU time series, portable patient embeddings, LLM summaries, cross-hospital transfer, clinical classification, distribution shift

### 场景、任务与核心难点

这篇工作面向 ICU 临床时序的跨机构预测与分类。典型输入是 MIMIC-IV、HiRID、PPICU 等医院/队列中的不规则 ICU 记录：变量异步采样、观测密度受医院流程和病情触发影响，且同一模型从一个医院部署到另一个医院时常出现性能下降。论文关心的不只是单一 benchmark 上的分类准确率，而是能否得到 portable patient embeddings，使下游 predictor 在新医院上少量甚至无需重新训练也能工作。

Record2Vec 的核心做法是 summarize-then-embed：先用冻结 LLM 将不规则 ICU 时序窗口转成结构化自然语言摘要，再用冻结文本嵌入模型生成固定长度 patient vector，随后接标准预测/分类模型。这个设计绕开了必须统一时间网格、手工插补或为每个医院适配时序架构的问题；它把不规则观测、趋势、异常值和临床上下文压缩成一种跨站点更可读、更可迁移的中间表示。

### 审稿人视角：价值与不足

最有价值的思想是把不规则临床时序的跨机构泛化问题提升到“输入表示是否便携”的层面。很多不规则时序模型专注于更精细地利用时间戳、mask 和变量关系，但这些特征往往也是最容易携带医院协议偏差的部分。Record2Vec 用语言摘要作为信息转换层，有机会把原始采样流程中的低层格式差异、单位差异和局部采样细节压缩为更语义化的 patient state description；实验中它在 MIMIC-IV、HiRID、PPICU 之间的迁移相对性能下降更小，这一点对真实部署非常有价值。

不足也很明显：它的性能和可靠性高度依赖摘要模板、LLM 对临床时序的归纳能力以及摘要是否遗漏细粒度时间信息。自然语言摘要可能过度平滑短期动态，弱化高频变化或关键观测间隔；如果 LLM 学到的临床常识与某些医院流程不匹配，也可能引入新的语义偏差。此外，方法并没有显式学习采样机制，只是通过摘要层间接降低站点特定 artifact；在需要精确解释“为什么某个变量未测、多久未测意味着什么”的任务中，语言压缩可能不够可控。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发非常强：处理策略偏移不一定只能在原始时间戳/mask 空间中做不变性约束，也可以引入一个“语义中介层”，把观测轨迹翻译成更接近临床状态、弱化采样格式差异的描述。对我们而言，可以设计 state summary branch 与 policy summary branch：前者只描述跨策略稳定的状态趋势和异常，后者描述观测频率、未测模式、联测模式等策略信息；分类头主要依赖 state summary，policy summary 用于校准和偏移诊断。

纵向深化上，Record2Vec 提示我们可以把采样策略偏移评估从单纯的 representation distance 扩展到 summary fidelity：同一潜在病程在不同采样策略下生成的摘要是否保持相同的临床语义？若摘要因为采样稀疏而改变诊断倾向，说明策略信息仍污染了状态表示。进一步，可以用反事实采样增强训练一个 policy-invariant summarizer：给定同一连续状态轨迹的多种观测调度，要求生成的 state summary 和下游 logits 一致，同时显式输出 policy summary 来解释观测过程差异。这样既吸收 LLM 表示的跨机构可迁移优势，又补上其对采样机制控制不足的问题。

## 追加更新 - 2026-06-19 23:02 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification 的顶会论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026 官方页面、OpenReview 与 arXiv/论文页。
- 已排除全部黑名单论文；同时排除 ICLR 2026 workshop 条目、偏 forecasting 或普通规则时序分类的工作。本次保留全新工作 2 篇。

## 10. QuITE: Query-based Irregular Time-series Embedding

- 会议：ICML 2026 Poster
- 作者：Junghoon Lim
- 官方页：https://icml.cc/virtual/2026/poster/64962
- 论文：https://arxiv.org/abs/2605.28166
- 代码：https://github.com/Meaningfull9502/QuITE
- 关键词：irregular multivariate time series, query tokens, plug-and-play embedding, backbone-agnostic, classification/forecasting

### 场景、任务与核心难点

QuITE 面向不规则多变量时序建模，覆盖分类和预测等下游任务。典型场景是医疗监测、传感器网络或工业系统中，各变量在不同时间点被异步观测，观测间隔不均匀，且研究者希望复用已有的 MTS backbone，而不是为每个不规则数据集重新设计专用架构。

论文指出，现有方法大体分为两类：一类是架构型方法，直接为 IMTS 设计专门模型，但会限制成熟 MTS 模型的复用；另一类是数据型方法，把 IMTS 插值或补齐成规则网格，但会注入人工值并扭曲真实动态。QuITE 将问题定位到输入嵌入层：传统 embedding 默认输入是均匀采样的规则序列，因此标准 backbone 无法直接消费不规则观测。它用一组 learnable query tokens 通过单层 self-attention 聚合原始不规则观测，生成与现有 backbone 兼容的固定维 latent representations，从而不需要插值、补齐或改动主干架构。

### 审稿人视角：价值与不足

最有价值的技术思想是把“不规则时序建模”从主干网络设计问题降维成可插拔输入表示问题。相比为 IMTS 重新设计复杂 continuous-time 或 graph architecture，QuITE 的 query-based embedding 更像一个适配层：它保留原始观测时间和值，通过注意力把不规则事件压缩到 backbone 可处理的表示空间。这种模块化设计对实际研究很有吸引力，因为它能系统性复用已有 Transformer、SSM、TCN 等 MTS backbone，并用同一接口比较不同主干在 IMTS 上的能力。论文报告其在多数据集、多 backbone 下能稳定提升，分类任务也有平均相对收益。

不足在于，query tokens 聚合观测时仍可能把采样模式当成判别信号吸收进输入表示。单层注意力的简洁性带来可插拔优势，但也可能缺少显式机制区分“由真实状态触发的 informative observation”和“由采样政策或机构流程触发的 observation”。如果训练环境中某类样本被更频繁或更晚期地采样，query embedding 可能学到策略特定的观测密度、时间分布或变量共现，而这些信号在跨医院、跨设备或主动采样策略变化时未必稳定。

### 对 Sampling-Policy Shift 的启发

QuITE 对我们的问题有直接横向应用价值：可以把 query tokens 设计成 state queries 与 policy queries 两组。state queries 只聚合跨策略稳定的轨迹信息，进入分类主路径；policy queries 则专门吸收观测频率、时间间隔、变量共现和 mask pattern，用于策略识别、偏移诊断或不确定性校准。这样既保留 query-based embedding 的 backbone-agnostic 优势，又避免所有采样信息无差别地进入分类边界。

纵向深化上，QuITE 很适合作为反事实采样一致性框架的前端：对同一底层连续轨迹生成多种采样策略视图，要求 state query representations 和 logits 保持一致，同时允许 policy query representations 区分不同采样策略。还可以在 query-to-observation attention 上加入策略不变性诊断，检查模型在策略改变后是否仍关注同一类状态事件，而不是转向采样密度或联测模式。相比直接约束整个 encoder，约束 query embedding 层更轻量，也更容易与现有 MTS backbone 组合。

## 11. Generative Modeling of Irregular Time Series via SDE-Induced Continuous-Discrete Variational Inference

- 简称：SDEVI
- 会议：ICML 2026 Poster
- 作者：Zexin Yuan, Qinliang Su, Junxi Xiao
- 官方页：https://icml.cc/virtual/2026/poster/64934
- 关键词：irregular time series, sparse asynchronous observations, stochastic differential equations, continuous-discrete variational inference, classification/interpolation/extrapolation

### 场景、任务与核心难点

SDEVI 面向稀疏、异步、不规则观测的连续时间系统建模，并在 healthcare、physics、climate、IoT 等 benchmark 上验证插值、外推、回归和分类任务。它关注的场景不是单纯把不规则序列补齐后分类，而是假设观测背后存在连续时间随机动力学，离散观测只是这一过程在不规则时间点上的投影。

核心难点在于，连续-离散状态空间模型通常要对潜在连续路径做 path-based variational inference，这会带来高计算成本，也常受限于较强的后验假设。SDEVI 改为直接在离散观测的 joint distribution 上做变分推断，同时保证该分布与底层 SDE 连续过程一致。方法使用由 linear time-varying SDE 诱导的 variational posterior 作为可扩展推断骨架，并进一步引入 non-linear-SDE-induced variational inference 与 complex-domain generalization，以表达更复杂的真实动力学。

### 审稿人视角：价值与不足

最有价值的思想是把不规则观测的概率建模从“补一条潜在路径再推断”转向“直接推断离散观测联合分布，同时保持连续动力学一致性”。这在理论和工程上都很重要：理论上，它避免把不规则采样简化成缺失网格问题，而是尊重连续时间生成机制；工程上，它绕开昂贵或受限的 path posterior 近似，为大规模 healthcare/IoT 数据上的分类和插值提供更可扩展的概率框架。与只优化判别表示的方法相比，SDEVI 还提供了对观测不确定性和连续动力学一致性的显式建模入口。

不足是，SDE 一致性主要约束状态动力学，并不自动解决采样机制的偏移问题。若观测时间本身由策略触发，例如重症患者更高频监测、工业设备在报警后密集采样，离散观测 joint distribution 会同时反映状态演化和采样政策。SDEVI 能更优雅地建模不规则观测，但如果没有显式的 observation process / sampling policy model，它仍可能把策略诱导的观测分布变化解释为状态动力学差异。官方摘要显示其覆盖分类任务，但 sampling-policy shift 不是主评估维度。

### 对 Sampling-Policy Shift 的启发

SDEVI 对 Sampling-Policy Shift 的纵向启发是：可以把我们的目标从判别式 representation invariance 推进到生成式 factorization。也就是说，用一个连续状态 SDE 描述真实系统动力学，再用一个单独的 sampling-policy process 描述何时、哪些变量被观测；分类器只依赖前者的 policy-invariant state posterior，而后者用于解释观测机制和校准不确定性。

横向应用上，可以借鉴 SDEVI 的 continuous-discrete inference，把反事实采样策略纳入训练目标：固定同一潜在 SDE 轨迹，改变 observation process，要求离散观测后验中的 state component 稳定，而 policy component 能正确区分采样策略。还可以用跨策略 posterior alignment 检查模型是否把“观测更密集”错误解释成“状态更异常”。这比只在最终 logits 上做一致性更深一层，因为它直接约束连续状态后验和离散观测联合分布之间的分解方式。

## 追加更新 - 2026-06-21 23:00 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：围绕近 3-7 个月内 irregular sampled / asynchronous / irregular multivariate time-series classification 的顶会论文，重点核对 NeurIPS 2025、AAAI 2026、ICLR 2026、ICML 2026、KDD 2025/2026、WWW 2026 页面、OpenReview、AAAI Proceedings、ICML virtual site 与 arXiv/ACM 论文页。
- 已排除全部黑名单论文；严格窗口内的正会 direct hits 基本已被历史日报覆盖。本次保留未在黑名单中的全新工作 2 篇：MTM 直接命中 irregular multivariate time series classification；MedMamba 虽不以 irregular sampling 为标题主轴，但作为 ICML 2026 医疗多通道时序分类工作，其 missing-channel robustness、subject-independent split 和 sample-adaptive graph learning 对采样策略偏移有直接横向价值。

## 12. MTM: A Multi-Scale Token Mixing Transformer for Irregular Multivariate Time Series Classification

- 会议：KDD 2025 Research Track
- 作者：Shuhan Zhong, Weipeng Zhuo, Sizhe Song, Guanyao Li, Zhongyi Yu, S.-H. Gary Chan
- 论文：https://www.cse.ust.hk/~gchan/papers/KDD25_MTM.pdf
- arXiv：https://arxiv.org/html/2509.17809
- DOI：https://doi.org/10.1145/3711896.3737058
- 关键词：irregular multivariate time series classification, channel-wise asynchrony, masked concat pooling, token mixing, imputation-free modeling

### 场景、任务与核心难点

MTM 面向 irregular multivariate time series 的序列级分类，典型数据包括医疗 ICU、生理活动识别和其他多通道传感器记录。论文把 IMTS 的关键难点从“时间间隔不均匀”进一步推进到“channel-wise asynchrony”：不同变量在同一时间点很少同步出现，导致常见 Transformer 的 channel-wise attention 实际只能在少量同步观测上工作，很多潜在相关变量从未在同一 attention context 中交互。

作者提出两个互补机制。第一，masked concat pooling 沿时间维逐层下采样，把相邻时间段内的稀疏观测合并到更粗时间尺度，从而缓解通道不同步，让 channel-wise attention 有更多机会看到跨变量关系。第二，token mixing 通过专门的 CLS/attention 机制从每个通道挑选重要 token，并把这些 token 与其他通道的非同步位置混合，随后再恢复原始 missing pattern，使模型增强跨通道信息流但不直接做插值填补。

### 审稿人视角：价值与不足

最有价值的思想是准确抓住了 IMTS 分类中被低估的瓶颈：不是所有不规则性都能靠 delta-t、mask embedding 或连续时间建模解决；当变量几乎不同时观测时，跨通道模块可能名义存在、实际失效。MTM 的 multi-scale pooling 和 token mixing 都围绕“如何让异步通道发生有效信息交换”设计，且保持 imputation-free，这比先补齐再建模更少引入人工同步假设。

不足在于，多尺度下采样虽然能缓解异步，但也可能压平短时间内的事件顺序和触发式采样细节。token mixing 选择的 pivotal tokens 也可能偏向训练环境中的高频观测变量或策略性联测模式：如果某类患者、某类设备或某家医院更常测某些通道，模型可能把“被选中的 token”当成类别证据，而不是稳定状态证据。论文主要在同分布 IMTS benchmark 上证明分类收益，对跨采样策略、跨机构或主动采样规则变化下的 token selection stability 还缺少系统评估。

### 对 Sampling-Policy Shift 的启发

MTM 对我们的问题有很直接的横向启发：采样策略偏移会改变 channel-wise synchrony distribution，即哪些变量有机会被同一时间窗口聚合、哪些通道的 token 更容易成为 pivotal token。因此，评估 sampling-policy shift 时不应只看 mask ratio 或 delta-t 分布，也应统计跨通道同步率、下采样后共现图和 token-mixing selection frequency 的变化。

纵向深化上，可以把 MTM 改造成 policy-aware token mixing：将 token 分为 state-pivotal 与 policy-pivotal 两类，前者进入分类主路径，后者用于解释观测调度、联测习惯和不确定性。对同一潜在轨迹生成不同采样策略视图时，可以约束 state-pivotal token 的语义和 logits 稳定，同时允许 policy-pivotal token 变化。masked concat pooling 也可加入跨策略一致性诊断：如果换一个采样策略后，模型必须依赖完全不同的粗尺度同步关系才能分类，就说明当前表示仍然暴露在 sampling-policy shortcut 下。

## 13. MedMamba: Multi-View State Space Models with Adaptive Graph Learning for Medical Time Series Classification

- 会议：ICML 2026 Poster
- 作者：Da Zhang, Bingyu Li, Zhiyuan Zhao, Hongyuan Zhang, Junyu Gao, Xuelong Li
- 官方页：https://icml.cc/virtual/2026/poster/61414
- 论文：https://arxiv.org/abs/2605.24961
- 关键词：medical time series classification, state space models, nonstationarity, adaptive graph learning, missing-channel robustness, subject-independent generalization

### 场景、任务与核心难点

MedMamba 面向医疗多通道时序分类，实验覆盖 EEG/ECG 等真实医学数据，并同时报告 subject-dependent 与 subject-independent 评估。它解决的主要场景不是典型 ICU 异步化验表，而是连续或分段生理信号中的疾病/状态分类；不过论文特别关注医疗时序部署时常见的非平稳性、跨通道依赖、主体差异和缺失通道鲁棒性，这些因素与非规则采样和采样策略偏移高度相邻。

方法由三部分组成：multi-scale convolutional embedding 捕捉局部形态；tri-branch differential state space encoder 同时处理原始、时间差分和频域视图，用于抑制 baseline drift 等非平稳伪差；spatial graph Mamba 则为每个样本学习稀疏、近似无环的有向通道依赖图，避免依赖预定义生理图结构。论文还在测试时模拟 missing channels，显示自适应图学习比固定图在通道缺失下更稳健。

### 审稿人视角：价值与不足

最有价值的技术思想是把 SSM 的长序列效率与医疗时序的结构归纳偏置结合起来。TDSSE 的 raw/difference/frequency 三视图不是普通特征拼接，而是在分类主干中显式对抗 baseline drift；SGM 的 sample-conditioned graph 也比固定图更符合医疗信号中个体差异和状态依赖的通道关系。subject-independent 评估值得肯定，因为它比随机 sample split 更接近真实部署中的跨患者泛化。

不足在于，它对 irregular sampling / asynchronous observation 的建模并不是显式目标。输入仍主要被表述为固定长度多通道序列，missing-channel 实验更像传感器失效或通道级遮蔽，而不是变量级非均匀时间戳、事件触发采样和临床流程驱动的 missingness。自适应图虽然能提升缺失通道鲁棒性，但如果通道缺失或信号质量本身由采样政策决定，图结构仍可能吸收策略性 artifacts。论文强调跨主体和 drift robustness，但还没有评估跨设备采样频率、医院采集协议或主动测量策略变化下图依赖是否稳定。

### 对 Sampling-Policy Shift 的启发

MedMamba 对 sampling-policy shift 的横向启发在于：采样策略偏移常常与非平稳性、主体差异和通道可用性共同出现，而不是孤立的 mask 变化。TDSSE 提示我们可以把策略偏移拆成 raw-state drift、difference-level event drift 和 frequency-domain sampling distortion 三种可诊断视图；不同视图下的 representation shift 可以帮助判断模型到底被状态变化还是采样政策变化驱动。

纵向深化上，SGM 可以扩展成 policy-robust adaptive graph：同一类别在不同采样策略下应共享核心 state graph，而 policy-induced graph edges 只进入策略诊断或置信度校准分支。missing-channel robustness 实验也可升级为反事实采样策略实验：不是随机遮蔽通道，而是按医院流程、告警触发或成本约束模拟变量级观测策略，检查 learned graph、SSM hidden state 和 logits 是否稳定。这样能把 MedMamba 的“缺失通道稳健性”进一步推进到“采样政策变化下的图结构不变性”。

## 追加更新 - 2026-06-23 23:00 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification 的顶会或强相关会议论文，重点核对 NeurIPS 2025、AAAI 2026、ICLR 2026、ICML 2026、KDD 2025/2026、WWW 2026、ICASSP 2026、OpenReview、官方 virtual/proceedings 页面与 arXiv。
- 已排除全部黑名单论文；同时排除 forecasting-only、普通规则时序分类、ICLR workshop 条目以及尚未核到正式会议收录的 arXiv-only 候选。本次仅保留全新工作 1 篇。

## 14. Rethinking Large Language Models for Irregular Time Series Classification in Critical Care

- 会议：ICASSP 2026
- 作者：Feixiang Zheng, Yu Wu, Cecilia Mascolo, Ting Dang
- arXiv：https://arxiv.org/abs/2601.16516
- 代码：https://github.com/mHealthUnimelb/LLMTS
- 关键词：irregular ICU time series classification, large language models, time-series encoder, multimodal alignment, critical care, computational trade-off

### 场景、任务与核心难点

这篇工作面向 ICU critical care 场景中的不规则时序分类，核心任务是基于高缺失、异步采样的 ICU 多变量记录进行死亡风险等临床结局预测。与普通规则时序分类不同，ICU 记录中的变量观测频率强烈不均衡：生命体征、化验项目、治疗事件和护理记录往往由病情、医院流程和医生决策共同触发。因此模型面对的不只是数值序列，还包括缺失模式、采样时间、变量是否被观测以及不同模态如何与 LLM 对齐的问题。

论文要解决的核心难点是：近年 LLM-based time series modeling 很热，但它们在 irregular ICU time series classification 上到底靠什么起作用并不清楚。作者把系统拆成两个关键部件来评估：一是 time-series encoder，负责把不规则时序变成高层时间表征；二是 multimodal alignment strategy，负责把时间表征与语言模型可消费的语义空间对齐。通过在两个 ICU benchmark 上比较 Time-LLM、S2IP、CALF、FSCA 等 LLM-based 方法和强 supervised / self-supervised baseline，论文试图回答 LLM 是否真正提升了不规则临床时序分类，还是主要受益于前端 encoder。

### 审稿人视角：价值与不足

最有价值的地方不是提出又一个复杂 LLM 架构，而是对“LLM 是否适合不规则 ICU 时序分类”做了结构化拆解和负责任的实证校验。结果显示，显式建模 irregularity 的 encoder 比 vanilla Transformer 有明显提升，平均 AUPRC 增益可达约 12.8%；相比之下，不同 multimodal alignment 策略的增益更有限。这个结论对审稿人很重要：它说明当前 LLM pipeline 的关键瓶颈仍在不规则时序表征，而不是简单把序列 embedding 接到大语言模型上。

不足也很明确。第一，论文主要是 empirical study，方法创新更多体现在评测框架和组件归因，而不是提出新的 sampling-policy robust model。第二，它虽然指出 LLM-based 方法训练成本至少约 10 倍于强不规则时序监督模型，却只在有限 ICU benchmark 和 mortality-style 任务上验证，尚不足以覆盖更广泛的 EHR endpoint、跨医院部署和多模态事件流。第三，论文评估了 encoder 与 alignment 的作用，但没有进一步把 irregular sampling 拆解为 state-driven informative sampling 与 hospital-policy-driven sampling，因此还不能判断模型究竟利用了稳定临床信号，还是利用了机构流程 shortcut。

### 对 Sampling-Policy Shift 的启发

这篇工作对我们研究 sampling-policy shift 的横向启发是：在 LLM-for-time-series 系统中，采样策略偏移最可能先污染 time-series encoder，而不是最后的语言模型推理层。也就是说，如果前端 encoder 已经把“某变量为什么被测、多久没被测、哪些变量被联测”压缩成类别相关 embedding，后续再复杂的 LLM alignment 也很难自动剥离 policy shortcut。因此，构建 policy-robust LLM pipeline 时，应优先在 encoder 层加入采样策略分解、反事实采样一致性或 policy-adversarial 约束。

纵向深化上，可以借鉴这篇论文的组件化评测方式，把我们的 Sampling-Policy Shift 方法拆成三层诊断：第一层评估 irregularity-aware encoder 在不同采样政策下的 state representation 是否稳定；第二层评估 alignment / summarization 是否放大或削弱采样策略信息；第三层评估最终分类头是否依赖 policy-sensitive tokens。更进一步，可以设计 value-redaction、time-redaction 和 policy-redaction 三组实验：只保留采样时间与变量身份、只保留观测值、或替换采样策略，分别测量模型性能和 representation drift。这样可以把“LLM 是否有用”的问题转化为“哪一层在吸收采样政策，哪一层能学习跨策略稳定状态”的可诊断框架。
