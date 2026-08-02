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

## 追加更新 - 2026-06-25 23:00 UTC

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / multimodal irregular / medical time series classification 的顶会或顶会相关论文，重点核对 AAAI 2026、ICLR 2026、ICML 2026、KDD 2025/2026、WWW 2026、OpenReview、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除偏 forecasting、causal discovery、普通规则时序分类或仅为 workshop 且已有更强正会替代的条目。本次保留全新工作 2 篇：MedSpaformer 是 AAAI 2026 正会医疗时序分类工作，虽不以 irregular sampling 为标题主轴，但其可变长度/通道维度、跨数据集迁移和 token sparsification 与采样策略偏移高度相关；MILM 直接面向 multimodal irregular time series classification 与 informative sampling，是近月对 sampling pattern 可预测性的最直接建模之一。

## 14. MedSpaformer: A Transferable Transformer with Multi-Granularity Token Sparsification for Medical Time Series Classification

- 会议：AAAI 2026 Technical Track
- 作者：Jiexia Ye, Weiqi Zhang, Ziyue Li, Jia Li, Fugee Tsung
- 链接：https://ojs.aaai.org/index.php/AAAI/article/view/40001
- 论文：https://arxiv.org/html/2503.15578v3
- DOI：https://doi.org/10.1609/aaai.v40i33.40001
- 关键词：medical time series classification, transferable transformer, token sparsification, multi-granularity cross-channel encoding, variable lengths/channels, cross-dataset transfer

### 场景、任务与核心难点

MedSpaformer 面向医疗时序分类中的跨数据集、少标注和异质输入问题。典型场景包括不同医院、不同设备或不同诊断任务下的 ECG/EEG/临床多通道时序：序列长度、通道数、标签空间和任务定义并不一致，直接训练一个任务专用模型会导致泛化差、迁移成本高，也难以在少样本或零样本诊断中复用已有知识。

论文解决的核心难点不是传统意义上“每个变量带任意时间戳”的 irregular EHR 建模，而是医疗时序在部署时常见的结构异质性与冗余观测：多通道信号中有大量非关键 token，不同粒度的局部形态和跨通道关系都可能影响分类，但它们在不同数据集中的出现频率、长度和通道配置并不稳定。MedSpaformer 因此提出 sparse token-based dual-attention mechanism，在全局上下文建模时动态聚焦 informative tokens；同时用 multi-granularity cross-channel encoding 捕捉不同时间粒度下的通道内/通道间依赖，并通过 adaptive label encoder 缓解跨数据集标签空间不一致。

### 审稿人视角：价值与不足

最有价值的思想是把医疗时序分类中的“可迁移性”与“稀疏 token 选择”放在同一个框架内处理。许多医疗时序模型在单一 benchmark 上很强，但换数据集、换通道定义或换标签空间后需要重新设计输入层和分类头；MedSpaformer 的多粒度 token sparsification 让模型在不同长度、不同通道维度下仍能抽取任务相关片段，adaptive label encoder 则把标签语义显式纳入迁移过程。这对真实医疗部署比单纯提升同分布 accuracy 更有价值。

不足在于，它对非规则采样机制的建模仍是间接的。论文强调 variable lengths 与 channel dimensions，但没有像 IMTS/EHR 模型那样显式处理变量级异步时间戳、事件触发采样和 missingness pattern。token sparsification 也可能引入新的 shortcut：如果某些通道或片段在训练医院中因为采样流程更频繁出现，模型可能把“被选中的 token”与疾病标签绑定，而不是学习稳定的生理状态。跨数据集迁移实验能部分说明鲁棒性，但还不足以证明在采样政策改变时 token selection 仍保持语义稳定。

### 对 Sampling-Policy Shift 的启发

MedSpaformer 对 sampling-policy shift 的横向启发在于：采样策略偏移可以表现为 token 预算和 token 重要性分布的偏移。不同医院或设备可能让某些时间段、通道或诊断片段更容易被记录，sparsification 模块若不受约束，就会把这种可见性差异放大为分类证据。因此，我们可以把 token selection frequency、multi-granularity attention map 和跨通道 token 共现率作为策略偏移诊断指标。

纵向深化上，可以把 token sparsification 改造成 policy-aware selection：一组 state tokens 承载跨策略稳定的生理/系统状态，进入分类主路径；另一组 policy tokens 解释观测密度、通道可用性和数据集协议差异，只用于校准或偏移诊断。训练时可对同一潜在轨迹施加不同采样策略增强，约束 state-token representation 与 logits 稳定，同时允许 policy-token 分布变化。这样能把 MedSpaformer 的跨数据集迁移能力进一步推进到“跨采样政策迁移”。

## 15. MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling

- 全称：Multimodal Irregular time series Language Model
- 会议/状态：WWW / ACM Web Conference 2026 方向；arXiv 2026-05 版本
- 作者：Hsing-Huan Chung, Shijun Li, Yoav Wald, Xing Han, Suchi Saria, Joydeep Ghosh
- 论文：https://arxiv.org/html/2605.13711v1
- 关键词：multimodal irregular time series, informative sampling, LLM, XML triplets, value-redacted training, EHR classification

### 场景、任务与核心难点

MILM 面向 multimodal irregular time series classification，尤其是 EHR 中数值化验、生命体征和临床文本共同组成的异步记录。与纯数值 IMTS 不同，MITS 的观测不仅有值和时间戳，还包含模态差异：某些时刻可能只有文本记录，某些时刻有化验值，某些检查的“被测量”本身就携带医生怀疑、治疗流程或病情变化的信息。任务通常是 ICU 或住院风险预测/分类，例如死亡风险、再入院或临床事件预测。

论文抓住的核心难点是 informative sampling：在不规则医疗记录中，什么时候观测、观测哪个通道，往往和标签高度相关。传统方法要么把 sampling pattern 当作缺失掩码辅助特征，要么在插补中隐式吸收；MILM 则直接让 LLM 学习这种结构。它把 MITS 序列化为按时间排序的 XML triplets，用两阶段 fine-tuning：第一阶段在 value-redacted MITS 上训练，仅凭时间和通道采样模式预测；第二阶段再加入观测值，让模型联合利用 sampling pattern 与 values。论文还设计 value-pending evaluation，模拟预测时某些值尚未返回但其时间/通道信息已知的临床场景。

### 审稿人视角：价值与不足

最有价值的思想是把 sampling pattern 从“可能有用的 side information”提升为主监督对象。第一阶段 value-redacted training 是一个很清晰的诊断实验：如果模型在没有数值的情况下仍能预测，说明采样行为本身具有强标签信息；第二阶段再融合数值，使模型不会只依赖观测值而忽略采样机制。XML triplet serialization 也提供了一个简单统一的多模态接口，让 LLM 可以同时消费时间、通道、值和文本，而不必为每类模态单独设计复杂融合模块。

不足同样来自它的优势：显式学习 informative sampling 很容易变成显式学习 sampling-policy shortcut。如果训练医院中某些检查只在高风险患者中触发，MILM 会合理地利用这种模式；但换到另一家医院、另一套检查协议或另一个成本约束后，同样的未测/已测模式可能有完全不同的语义。论文已经意识到跨医院 sampling distribution shift 是未来方向，但当前主要证明 sampling pattern 有预测力，并未充分回答哪些 sampling pattern 是可迁移的状态信号、哪些只是环境特定政策信号。此外，LLM 序列化会带来上下文长度、数值精度和可解释性控制问题。

### 对 Sampling-Policy Shift 的启发

MILM 对我们的 Sampling-Policy Shift 研究几乎是直接横向启发：在非规则采样下，不能简单问“是否应该使用 missingness”，而要问“哪些 missingness 来自状态，哪些来自政策”。value-redacted training 可以被改造成策略泄漏检测器：在不同环境中只用采样时间和通道训练一个 policy-only classifier，如果它能强预测标签，就说明当前任务存在强 sampling-policy shortcut。

纵向深化上，可以把 MILM 的两阶段训练改为三分解目标：state branch 从观测值和文本中学习跨策略稳定状态，policy branch 从时间/通道/缺失中学习采样政策，classification head 通过不变性或对抗约束限制对 policy branch 的直接依赖。对同一病程生成不同采样策略视图时，要求 state summary 和 logits 一致，同时允许 policy summary 区分医院协议、检查触发规则或 value-pending 机制。这样既承认 informative sampling 的存在，又避免模型把训练环境中的采样政策当成可迁移病理证据。

## 追加更新 - 2026-06-26 23:02 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / clinical irregular time series classification 的顶会或顶级方向论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、ICASSP 2026、OpenReview、ICML virtual site、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除 Hi-Patch 这类 ICML 2025 时间窗口偏旧条目、偏 forecasting 或普通规则时序分类的条目。本次保留全新工作 2 篇：StarEmbed 是 ICML 2026 正会 benchmark，虽然领域是天文光变曲线而非医疗 IMTS，但其多波段、异步、异方差和跨观测策略问题对 sampling-policy shift 很有横向价值；Rethinking LLMs 是 ICASSP 2026 会议论文，直接评估 LLM 在 ICU 不规则时序分类中的有效性与局限。

## 16. StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Stars

- 会议：ICML 2026 Poster
- 作者：Weijian Li, Hong-Yu Chen, Nabeel Rehemtulla, Ved Shah, Dongho Kim, Dennis Wu, Qinjie Lin, Adam Miller, Han Liu
- 官方页：https://icml.cc/virtual/2026/poster/63677
- 论文：https://arxiv.org/html/2510.06200
- 项目页：https://hibb-bb.github.io/star-embed.github.io/
- 代码：https://github.com/skai-institute/StarEmbed
- 关键词：irregular astronomical time series, light-curve classification, foundation model benchmark, multi-band observations, heteroskedasticity, OOD detection

### 场景、任务与核心难点

StarEmbed 面向天文时域观测中的变量星光变曲线分析。输入来自 Zwicky Transient Facility 的多波段观测：同一恒星在不同滤波波段上的亮度记录并不同步，采样间隔跨越多个数量级，还伴随天气、昼夜、观测几何和仪器条件导致的异方差测量误差。论文构建约 4 万条专家标注的多波段光变曲线，覆盖 7 类天体，并评估无监督聚类、监督分类和 OOD 源检测。

核心难点在于，现有通用 time-series foundation models 通常预训练在规则采样、低噪声或单变量商业/交通/能源数据上，而天文光变曲线是极端 out-of-domain 的不规则科学时序。论文不是提出新的 irregular encoder，而是建立一个标准化 benchmark，检验 Moirai、Chronos、Chronos-Bolt、Time-MoE 等通用 TSFM 以及 Astromer 这类领域模型的 zero-shot embedding 能否支撑变量星分类与异常源发现。结果显示，手工特征在监督分类上仍很强，但 Chronos 系列在聚类和 OOD 检测上表现突出，说明通用基础表征对不规则科学时序具备一定迁移性。

### 审稿人视角：价值与不足

最有价值的贡献是把“不规则采样分类”从医疗/传感器 benchmark 横向扩展到一个真实、规模大、采样机制复杂且有强科学约束的领域。StarEmbed 的意义不只是给出一个新数据集，而是提供固定 split、专家标签、公开 embedding 与多任务评估，使研究者可以检验基础模型在从未见过的采样制度下是否仍能形成可分表征。它也提醒我们，classification accuracy 不是唯一目标；对未来大规模巡天而言，OOD detection 和 embedding quality 同样关键。

不足在于，它主要是 benchmark/评测论文，针对 irregular sampling 的模型机制创新有限。监督分类中手工特征仍优于多数 TSFM，说明通用 foundation embedding 尚未充分捕捉天文领域的周期、相位折叠和异方差误差结构。论文还没有把观测策略本身作为可控变量系统评估，例如不同巡天 cadence、天气缺测、波段调度或训练/测试 split 中采样密度差异如何影响分类边界。换言之，它证明了 TSFM 的跨域潜力，但还未回答哪些表示对观测策略稳定、哪些只是适配了 ZTF 的特定观测政策。

### 对 Sampling-Policy Shift 的启发

StarEmbed 对 sampling-policy shift 的横向启发是：采样政策偏移不只存在于 ICU 或传感器网络，也天然存在于天文巡天。观测 cadence、昼夜窗口、天气缺测、波段选择和测量不确定性共同决定一条光变曲线的可见形态；同一类变量星在 ZTF、LSST 或其他巡天中的采样政策变化，可能改变周期特征可恢复性、相位覆盖度和 OOD 分数。

纵向深化上，可以把 StarEmbed 扩展为 policy-shift benchmark：在同一底层光变曲线或物理模板上生成多种观测 cadence，要求 state embedding 保持天体类别语义稳定，同时允许 policy embedding 解释采样间隔、波段覆盖和不确定性分布。对我们的问题，这提示可以把 sampling-policy shift 量化为“任务相关频率/相位信息的可恢复性变化”，而不只是 mask ratio 或 delta-t 的统计偏移。若一个模型在不同观测政策下分类 logits 稳定但 OOD/policy score 能正确反映采样质量，就更接近真正可迁移的不规则时序分类器。

## 17. Rethinking Large Language Models for Irregular Time Series Classification in Critical Care

- 会议：ICASSP 2026
- 作者：Feixiang Zheng, Yu Wu, Cecilia Mascolo, Ting Dang
- 论文：https://arxiv.org/html/2601.16516
- 会议记录：https://findanexpert.unimelb.edu.au/scholarlywork/2306561-rethinking-large-language-models-for-irregular-time-series-classification-in-critical-care
- 代码：https://github.com/mHealthUnimelb/LLMTS
- 关键词：irregular ICU time series classification, LLM for time series, encoder design, multimodal alignment, mortality prediction, few-shot learning

### 场景、任务与核心难点

这篇工作面向 ICU 不规则时序分类，主要任务是基于 PhysioNet 2012 和 MIMIC-III 等重症监护记录进行院内死亡风险预测，同时用半合成不规则 ECG 检验缺失率变化下的鲁棒性。ICU 数据的典型难点是变量多、缺失率高、采样异步且受临床流程触发：生命体征、化验、用药和记录频率都可能随病情和医院协议变化。

论文要回答的问题很直接：当前 LLM-based time-series 方法在规则数据上看似强大，但是否真的适合不规则 ICU 分类？作者系统比较 Time-LLM、S2IP、CALF、FSCA 等 LLM 方法，以及 MOMENT、UniTS、mTAND、Warpformer 等自监督/监督 baseline，并拆解两个关键组件：time-series encoder 与 multimodal alignment strategy。实验发现，显式处理不规则性的 mTAND encoder 对性能影响远大于 alignment 策略；单纯把规则时序 LLM 框架迁移到 ICU irregular data 会明显退化；LLM 方案通常带来 10 倍量级训练开销，却只得到有限甚至不稳定收益。

### 审稿人视角：价值与不足

最有价值的地方是，它没有继续假设“LLM 一定能靠语义能力解决时序不规则性”，而是做了针对 ICU irregular classification 的组件级审计。结论非常有用：encoder 是否尊重真实时间戳、缺失和异步观测，比后端 LLM alignment 的花哨程度更关键。对审稿人而言，这类负结果/边界结果很重要，因为它约束了近期把 LLM 套到所有时序任务上的趋势，也为后续方法设计提供了清晰优先级。

不足在于，这篇工作仍以经验比较为主，缺少对采样机制的因果分解。它证明了不规则感知 encoder 重要，也证明 LLM few-shot 并不自动占优，但还没有系统区分“状态驱动的 informative sampling”和“医院政策驱动的 sampling shortcut”。半合成 ECG 通过随机 drop 模拟不规则性，能测 missing ratio robustness，却不能充分代表 ICU 中由医生决策、检查协议和病情触发共同形成的策略性采样。论文也主要在死亡风险分类上评估，尚未覆盖跨医院采样政策迁移或反事实采样协议。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：大模型或语言对齐不是解决采样策略偏移的捷径。若前端 encoder 已经把 irregularity 处理成错误的 patch、错误的局部上下文或规则时间假设，后端 LLM 再强也只能在污染表征上做推理。因此，我们在设计 policy-robust 模型时，应优先保证输入层和 encoder 能显式表达观测时间、变量级 mask、delta-t、测量不确定性与环境/政策标签。

纵向深化上，可以借鉴论文的组件审计范式，把采样策略偏移拆成 encoder sensitivity 与 alignment sensitivity 两层评估：先固定分类头，测试不同采样政策下 state encoder 的表征稳定性；再测试 policy-aware alignment 是否把采样模式仅用于校准和不确定性，而不是直接作为类别证据。对 LLM 路线而言，更合理的方向不是把 ICU 序列粗暴转成文本，而是构建 state encoder + policy encoder + semantic alignment 的三分支结构：state branch 学跨策略稳定病程，policy branch 学测量流程和缺失机制，LLM 或语义模块只在两者解耦后做可解释摘要与决策支持。

## 追加更新 - 2026-07-13 23:01 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / continuous-time irregular sequence classification 的顶会论文，重点核对 ICML 2026、ICLR 2026、AAAI 2026、NeurIPS 2025、KDD 2025/2026 官方页、OpenReview、ACM DOI 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除偏 forecasting、普通规则时序分类、ICML/KDD workshop 条目，以及 KDD 2025 中时间窗口偏旧但相关的 ISTS-PLM/HCIB 候选。本次保留全新工作 1 篇：Efficient Neural CDE via Attentive Kernel Smoothing 是 ICML 2026 正会论文，虽然标题不直接写 classification，但论文问题设定和实验明确覆盖监督分类，并聚焦不规则观测下 Neural CDE 的控制路径构造与求解效率。

## 18. Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing

- 简称：MV-CDE / MVC-CDE
- 会议：ICML 2026 Poster
- 作者：Egor Serov, Ilya Kuleshov, Alexey Zaytsev
- 官方页：https://icml.cc/virtual/2026/poster/62701
- 论文：https://arxiv.org/html/2602.02157
- 关键词：irregularly sampled time series, Neural CDE, kernel smoothing, Gaussian Process smoothing, multi-view attention, classification efficiency

### 场景、任务与核心难点

这篇工作面向不规则观测序列上的监督学习，实验主要评估多变量时序分类，并以 CharacterTrajectories、SpokenArabicDigits、UWaveGestureLibrary 等 UEA/UCR 分类数据为 benchmark。它关注的不是传统“如何补齐缺失值”，而是 Neural CDE 在处理不规则采样时一个更底层的问题：离散观测必须先被提升为连续 control path，而常用线性/三次样条插值会强行穿过每个观测点，把噪声、高频抖动和采样不均匀性一起变成非常粗糙的驱动路径。

这种粗糙路径会让自适应 ODE solver 为了控制局部误差而频繁缩小步长，导致 Number of Function Evaluations 和推理时间显著上升。作者因此用 Kernel / Gaussian Process smoothing 替代精确插值，显式控制 control path 的 regularity；为了避免过度平滑丢失判别性细节，再引入 learnable queries 的 Multi-View CDE 和卷积版 MVC-CDE，让多个平滑视图分别捕捉不同时间尺度或局部模式。最终目标是在保持甚至提升分类准确率的同时，显著降低 Neural CDE 的求解成本。

### 审稿人视角：价值与不足

最有价值的技术思想是把 Neural CDE 的效率瓶颈从“solver 本身慢”进一步定位到“control path 几何太粗糙”。很多连续时间模型默认把插值视为无害前处理，然后再优化 vector field 或 solver tolerance；这篇论文指出，只要驱动路径继承了噪声和非均匀采样带来的高频变化，solver 就会被迫沿着复杂几何前进。用平滑路径降低几何复杂度，再用多视图注意力补回高频信息，是一个很清晰的 accuracy-efficiency trade-off 设计。

不足在于，论文主要把 irregularity 看作数值求解和路径光滑性问题，还没有显式区分“真实状态变化导致的高频事件”与“采样政策/传感器策略导致的观测粗糙”。如果某些类别在训练环境中被更密集采样，或者某些变量只在告警后出现，平滑路径可能把策略诱导的观测密度差异压缩成看似稳定的低频趋势；多视图 attention 也可能学习到与采样政策相关的局部可见性模式。论文证明 MVC-CDE 在分类准确率和 NFE 上更优，但尚未系统测试换采样策略、换观测触发规则后，平滑视图和 attention head 是否保持语义稳定。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：采样策略偏移会改变 control path roughness，从而同时影响表示学习和数值求解成本。也就是说，策略偏移不只体现在 mask ratio、delta-t 分布或变量共现图上，还可能体现在 solver 需要多少步、哪些时间段产生高曲率路径、哪些平滑带宽最有效。NFE、路径曲率、GP smoothing bandwidth、multi-view attention 分布都可以成为诊断采样政策偏移的辅助指标。

纵向深化上，可以把 MVC-CDE 改造成 policy-aware path smoothing：一组 state views 用于保留跨策略稳定的真实动力学，一组 policy views 用于解释观测调度、噪声水平和采样密度变化。对同一潜在轨迹生成不同采样策略视图时，可约束 state control path、分类 logits 和关键 attention head 保持一致，同时允许 policy view 的粗糙度和不确定性变化。这样既保留平滑路径带来的可扩展性，又避免把“某种策略下更容易求解/更平滑”的路径误当成可迁移类别证据。

## 追加更新 - 2026-07-19 23:01 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / sparse healthcare time series classification / event detection 的顶会论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、KDD 2025/2026、OpenReview、ICML virtual site、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除 MedFuse 这类 withdrawn submission、STAR-Set 这类 ICLR workshop 条目，以及 LLapDiff、ASTGI、APN、ReIMTS、MOSES、MN-Diff 等偏 forecasting/generation/imputation 而非分类主任务的候选。本次保留全新工作 1 篇：Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction 是 ICLR 2026 正会 Poster，虽然任务形式是事件检测而非整条序列分类，但它明确联合定位事件边界与分类事件类型，且面向临床稀疏事件，在异步/非规则医疗信号分类与 sampling-policy shift 问题上有较强横向价值。

## 19. Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction

- 会议：ICLR 2026 Poster
- 作者：Beomjun Bark, Yun Kwan Kim
- OpenReview：https://openreview.net/forum?id=DulnZ7Dv82
- 官方页：https://iclr.cc/virtual/2026/poster/10010733
- 代码：https://github.com/hbumjj/CDI-TS-Event-Detection
- 关键词：sparse healthcare time-series, event detection, event type classification, boundary localization, adaptive gating, context-detail interaction

### 场景、任务与核心难点

这篇工作面向医疗时序中的稀疏临床事件检测，任务不是只判断一整条序列的类别，而是同时定位事件起止边界并分类事件类型。论文评估的场景包括心律失常检测、情绪识别和活动监测等 healthcare time-series：真正有诊断价值的片段在长序列中占比极低，事件边界模糊，类别分布稀疏，临床上又要求模型给出可操作的时间位置，而不是只输出一个全局风险分数。

核心难点在于，DETR 类检测框架在图像目标检测中能通过 query 匹配定位对象，但直接迁移到医疗时序时会遇到极端事件稀疏：大部分时间窗口是背景，局部高频细节容易被全局上下文淹没；如果始终启用细粒度检测分支，又会在大量无事件区域引入噪声和计算浪费。作者因此提出 coarse-to-fine 框架，由 global context explorer 先建模长程背景和事件可能性，local detail inspector 负责精细边界与事件形态，再用 Adaptive Gating Module (AGM) 作为上下文-细节交互开关。AGM 利用 transformed labels，把事件是否存在、事件位置和原始类别标签转成多视角监督，使模型只在事件可能出现时强化局部细节提取，从而提升极稀疏事件的检测和分类能力。

### 审稿人视角：价值与不足

最有价值的技术思想是把“稀疏事件分类”显式拆成全局筛查与局部精查两种计算模式，并用标签变换驱动的 gate 学习二者何时交互。相比在所有时间点平均施加同样的注意力或检测 query，AGM 更符合医疗监测流程：先判断是否存在可疑片段，再在可疑区域进行边界级和类型级判别。对审稿人而言，这个设计的优势不只是指标提升，还在于它把稀疏性从数据缺陷转化为模型结构先验，使 rare-event learning 不再完全依赖 loss reweighting 或更多负样本采样。

不足在于，论文主要处理“事件在时间轴上稀疏”的问题，并没有把观测过程本身的非规则采样、传感器缺失或医院测量政策作为显式变量。心律失常、活动或情绪数据中的稀疏事件不一定等同于 ICU/EHR 中由医生决策触发的异步化验；如果事件片段更容易被设备高频记录、人工标注或特定监测策略覆盖，AGM 学到的 gate 可能同时反映真实临床事件和采样/标注流程。论文证明了 sparse event detection 的有效性，但还缺少跨设备、跨医院、跨采样频率或跨告警触发规则下 gate 稳定性的系统评估。

### 对 Sampling-Policy Shift 的启发

这篇工作对 Sampling-Policy Shift 的横向启发是：采样策略偏移可以被看作一种“门控触发分布”的偏移。现实医疗系统中，医生或设备并不是均匀观察病人，而是先由粗粒度风险、报警阈值或资源约束触发更密集的局部测量；这与论文中的 global context explorer 触发 local detail inspector 在结构上非常相似。因此，我们可以把采样政策建模为一个 policy gate：它决定哪些时间段、哪些变量会进入高分辨率观测，而分类模型需要区分 gate 是由真实状态驱动，还是由环境/医院流程驱动。

纵向深化上，可以把 AGM 扩展为 state-policy 双门控框架。state gate 负责捕捉跨采样策略稳定的真实事件或病程变化，进入分类主路径；policy gate 负责解释为何某些时间段被密集观测、为何某些变量被联测或为何某些事件更容易被标注，只用于偏移诊断和不确定性校准。训练时可对同一潜在病程施加不同采样策略增强，约束 state gate、事件类型 logits 和关键边界表示保持稳定，同时允许 policy gate 随观测策略改变。这样能把稀疏事件检测中的 context-detail interaction 推进到非规则采样下的策略不变事件分类。

## 追加更新 - 2026-07-26 23:02 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / ICU event-stream prediction / EHR time-series representation 的顶会或顶会 workshop 论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、KDD 2025/2026、OpenReview、ICLR virtual site 与 arXiv 页面。
- 已排除全部黑名单论文；同时排除 MedFuse 这类先前已标记为 withdrawn 的候选、ReTAMamba 这类暂无顶会来源的近期 arXiv 预印本，以及偏 forecasting/generation/imputation、普通规则时序或与异步分类关系较弱的条目。本次保留全新工作 2 篇：EHR-SPC 是 ICLR 2026 TSALM Workshop Poster，直接面向不规则 EHR event streams 的下游 ICU 预测表征；LLM4EHR 是 ICLR 2026 OpenReview submission / 2026-07 arXiv 新预印本，尚未确认正会录用，但其临床事件序列与时序对齐思想对非规则采样分类和 sampling-policy shift 有较强跟踪价值。

## 20. Status-Aware Self-Supervised Forecasting for Irregular Clinical Time Series

- 方法名：EHR-SPC / EHR Set Predictive Coding
- 会议：ICLR 2026 TSALM Workshop Poster
- 作者：Kwanhyung Lee, Joohyung Lee, Jong-Heon Kim, Sangchul Hahn, Eunho Yang
- 官方页：https://iclr.cc/virtual/2026/10013833
- OpenReview：https://openreview.net/forum?id=lx98lmQ14i
- 关键词：irregular EHR event streams, self-supervised learning, ICU prediction, future status forecasting, query-based set prediction, masked event modeling

### 场景、任务与核心难点

这篇工作面向 ICU/EHR 中天然不规则的临床事件流：每条患者轨迹由观测值、时间戳和变量类型三元组组成，不同化验、生命体征、干预记录在不同时间异步出现。下游任务是面向未来临床状态的预测，例如 CPR、死亡风险或其他 ICU outcome；训练难点来自标签稀缺、类别不平衡，以及传统 SSL 方法通常先把事件流离散化到固定网格再做预测，导致原始事件集合结构、真实时间间隔和变量级异步性被抹平。

EHR-SPC 的核心做法是直接在 irregular event sets 上做自监督预训练。模型先把短事件窗口聚合为 status token，再用过去 status context 去预测未来 status representations；未来事件集合大小和组成是不固定的，因此作者借鉴 DETR 风格的 learnable queries，用 query-based Transformer decoder 生成未来 status tokens，并用 EMA momentum status encoder 提供稳定目标。同时，EHR-MAE 辅助目标对过去事件进行遮蔽重构，增强局部观测鲁棒性。整体目标不是补齐一个规则时间网格，而是学习一个与“未来临床状态预测”对齐的事件级表征。

### 审稿人视角：价值与不足

最有价值的技术思想是把不规则 EHR 预训练目标从“恢复被遮蔽的局部值”推进到“预测未来状态表征”。这更贴近临床分类/预后任务的实际需求：医生关心的是患者未来风险状态，而不仅是某个缺失化验值。query-based set prediction 也很适合 EHR，因为未来窗口中会出现哪些变量、出现多少事件本身不固定，强行做逐 token 对齐会引入虚假的顺序和数量假设。

不足在于，这仍是一篇 workshop 短文，实验和偏移分析深度有限。更重要的是，未来 status target 本身可能混合真实病程和医院采样/记录政策：如果某些未来事件集合是由告警、医生怀疑或科室流程触发的，那么模型预测到的“future status”可能部分是 protocol status，而不完全是 patient state。论文展示了对 ICU prediction 的下游收益，但尚未系统评估跨医院、跨记录制度或反事实采样策略下 status token 的语义稳定性。

### 对 Sampling-Policy Shift 的启发

EHR-SPC 对我们的横向启发是：采样策略偏移可以被提升到“未来状态预测目标是否稳定”的层面来分析。若同一潜在病程在不同采样政策下产生不同的 future event set，那么普通 SSL 会鼓励模型学习策略特定的未来观测集合；这解释了为什么一些 EHR 表征在同院同策略有效、跨院后退化。

纵向深化上，可以把 EHR-SPC 改造成 state-policy 双状态预测框架：state status token 预测跨采样策略稳定的病程状态，policy status token 预测未来会被记录哪些变量、何时记录、记录密度如何。训练时对同一患者轨迹构造多种反事实采样视图，约束 state status 和分类 logits 保持一致，同时允许 policy status 区分医院流程、告警触发和资源约束。这样能把“未来状态自监督”从单纯提升表征质量推进到显式约束采样政策不变性。

## 21. LLM4EHR: Aligning Clinical Time Series with Medical Event Sequences via Large Language Models

- 会议/状态：ICLR 2026 OpenReview submission；arXiv 2026-07 预印本（尚未确认正会录用）
- 作者：Jingteng Li, Alexander Capstick, Louise Rigny, Iona Biggart, Neil J. Sebire, Payam Barnaghi
- OpenReview：https://openreview.net/forum?id=pym3JRajmW
- arXiv：https://arxiv.org/html/2607.15447v1
- 关键词：clinical time series, EHR event sequences, LLM alignment, contrastive learning, ICU outcome prediction, transferable embeddings

### 场景、任务与核心难点

LLM4EHR 面向 ICU 临床预测中的多模态 EHR 表征学习。输入不只包括连续或半连续的 clinical time series，也包括药物、医嘱、操作、诊断等 itemized medical event sequences；这些事件天然带时间戳且不规则出现，和生命体征/监护时序之间存在共享但并不显式对齐的时间结构。下游任务包括 mortality、phenotyping、decompensation 和 remaining length-of-stay 等临床预测/分类任务，并强调少样本和跨队列适配。

核心难点在于，已有临床基础模型常把 EHR 事件和 TS 观测分开建模，或者只做粗粒度拼接，未充分利用二者在时间上的互补关系。LLM4EHR 使用 domain-adapted LLM 编码 EHR event sequence，用 Transformer TS encoder 编码临床时序，再通过 time- and semantic-aware regularized contrastive objective 对齐两种表示。这样得到的 TS embedding 不只是从数值轨迹中学习，还被事件语义和临床流程上下文条件化，从而提升多个下游任务和 k-shot 跨队列适配表现。

### 审稿人视角：价值与不足

最有价值的思想是把不规则临床事件序列当作时序表征的语义锚点，而不是只把它们作为额外协变量。对审稿人来说，contrastive alignment 的价值在于它给出了一个比较清晰的中间目标：临床 TS 表征应与同一患者同一时间上下文中的医学事件语义一致。相比直接把所有 EHR token 丢进 LLM，LLM4EHR 保留了专门的 TS encoder，同时利用 LLM 对事件文本、代码和临床语义的表达能力，具有更好的模块化和迁移潜力。

不足也同样明显。首先，当前状态是 submission / 新预印本，尚不能按已录用正会论文看待。其次，医学事件序列本身高度受医院政策影响：哪些药物被开、哪些检查被下单、何时记录护理事件，往往反映病情和流程的混合结果。如果 contrastive objective 不区分 patient-state semantics 与 hospital-policy semantics，TS embedding 可能被迫对齐到策略特定事件模式，跨医院时反而带来新的 shortcut。论文强调 transferable embeddings，但仍需要更强的 cross-policy、cross-site 和 counterfactual sampling 评估来证明对采样政策偏移的稳健性。

### 对 Sampling-Policy Shift 的启发

LLM4EHR 对 Sampling-Policy Shift 的横向启发是：事件序列可以作为采样政策的显式观测窗口。EHR 中的检查、用药、转科、护理记录不仅描述患者状态，也描述医院如何观察和干预患者；它们可以帮助解释为什么某些时段被密集采样、为什么某些变量被联测、为什么某些值尚未返回。因此，处理非规则采样分类时，不应只把 event sequence 当作增强分类性能的语义特征，还应把它作为 policy context 来建模。

纵向深化上，可以把 LLM4EHR 的对齐目标改造成三分支：state TS encoder 学跨策略稳定的病程表征；policy/event encoder 学采样、医嘱和记录流程；semantic alignment 只约束 state branch 与跨环境稳定的临床语义对齐，而通过对抗或条件不变性限制分类头直接利用 policy branch。对同一病程生成不同采样/记录策略时，要求 state embedding 与 logits 一致，同时允许 policy embedding 改变。这样能把 LLM 语义对齐从“增强 EHR 表征”推进到“识别并隔离采样政策语义”。

## 追加更新 - 2026-07-27 23:00 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`；同时读取兼容入口 `paper_daily.md`，纳入其中所有历史追加标题以扩大黑名单。
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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / informative missingness 的顶会或顶会相关论文，重点核对 ICLR 2026 OpenReview、ICML 2026、AAAI 2026、ICASSP 2026、ACL 2026、NeurIPS 2025 官方或论文页面与 arXiv。
- 已排除全部黑名单论文；同时排除已被历史日报标记过的 withdrawn 候选、偏 forecasting/generation/imputation 的条目，以及与分类任务关系较弱的 benchmark-only 工作。本次保留全新工作 2 篇：STAR-Set 直接面向异步临床时间序列的 ICU 预测/分类式任务，且在摘要中明确指出 sampling-policy shortcuts；VP-GNN 是 ICASSP 2026 不规则临床时序图建模论文，任务覆盖院内死亡预测与早期脓毒症检测。

## 22. Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series

- 简称：STAR-Set Transformer
- 会议/状态：ICLR 2026 OpenReview submission / Research track；arXiv 2026-03
- OpenReview：https://openreview.net/forum?id=AxXNor3Kd2
- arXiv：https://arxiv.org/abs/2603.06605
- 关键词：asynchronous clinical time series, point-set tokenization, temporal attention bias, variable-type affinity, ICU prediction, sampling-policy shortcuts

### 场景、任务与核心难点

STAR-Set 面向 ICU/EHR 中异步、多变量、不规则采样的临床时间序列预测任务，实验覆盖 CPR、mortality 和 vasopressor use 等 ICU outcome。输入不是规则网格上的完整矩阵，而是由变量类型、时间戳和值组成的观测事件集合；这种 point-set tokenization 可以避免插值和填补，但会丢掉规则网格天然暴露的两类结构：同一变量随时间演化的列结构，以及相近时间内不同变量之间的行结构。

论文解决的核心难点是：如何在保留事件集合灵活性的同时，把这些结构先验重新注入 Transformer。作者在 attention logits 中加入两类轻量 soft bias：一类是 temporal locality penalty，用可学习时间尺度鼓励时间上相近的事件互相关注；另一类是 variable-type affinity，用可学习变量兼容矩阵恢复变量间或同变量内的结构偏好。这样模型不需要把 EHR 强行离散化到固定时间网格，也不需要完全依赖普通 self-attention 自己从稀疏事件中恢复临床结构。

### 审稿人视角：价值与不足

最有价值的思想是把“事件集合表示”和“网格结构归纳偏置”之间的矛盾处理得很简洁。许多 EHR 模型要么使用 grid，保留 time-by-variable 结构但引入 imputation/mask shortcut；要么使用 set token，避免填补却牺牲变量轨迹和局部共现。STAR-Set 用参数量很小的 attention bias 在两者之间折中，且 learned timescale 和 variable affinity 能提供一定解释性，帮助审稿人检查模型究竟依赖哪些时间邻域和变量关系。

不足在于，它仍主要在同一数据来源和任务划分下验证 supervised ICU prediction。虽然论文明确意识到 observation patterns 同时反映 physiology 和 care process，且可能产生 sampling-policy shortcuts，但现有实验还没有把医院、科室、监测协议或主动测量策略作为显式环境变量。temporal bias 和 variable affinity 可能恢复了有用结构，也可能恢复了训练医院特定的联测习惯、化验频率和护理流程。

### 对 Sampling-Policy Shift 的启发

STAR-Set 对我们的横向启发是：sampling-policy shift 可以被视为 attention bias 所刻画的结构先验发生偏移。不同采样策略不仅改变 mask ratio 或 delta-t 分布，也会改变哪些变量在局部时间窗内共现、同一变量的有效时间尺度以及变量兼容矩阵的稳定性。因此，learned timescale、variable affinity 和 attention-bias contribution 可以成为采样策略偏移的诊断指标。

纵向深化上，可以把 STAR-Set 改造成 policy-aware bias Transformer：一组 state bias 学习跨策略稳定的生理时间尺度和变量关系，进入分类主路径；另一组 policy bias 学习医院流程、记录习惯和资源约束诱导的局部共现，只用于偏移诊断或校准。对同一潜在病程构造不同采样策略视图时，可约束 state bias、state representation 和 logits 保持一致，同时允许 policy bias 区分不同观测流程。这样能把“恢复网格先验”推进到“区分哪些结构先验可跨采样政策迁移”。

## 23. VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series

- 会议：ICASSP 2026
- 作者：Yurong He, Boya Zhang, Longfei Liu, Guosheng Cui, Dan Wu
- DOI：https://doi.org/10.1109/ICASSP55912.2026.11461796
- 官方记录：https://cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=18058
- 代码：https://github.com/bursonz/VP-GNN
- 关键词：irregular clinical time series, variable-wise graph, patch-wise graph, selective message passing, in-hospital mortality prediction, early sepsis detection

### 场景、任务与核心难点

VP-GNN 面向不规则临床时间序列中的风险预测和疾病状态评估，实验使用 PhysioNet 2012 与 Sepsis 2019 等 EHR benchmark，任务包括院内死亡预测和早期脓毒症检测。此类数据具有稀疏观测、变量异步、采样间隔不均以及变量依赖异质等典型特点；如果直接重采样到统一网格，会扭曲真实观测时间，也会把缺失和观测密度的语义混在补齐值中。

论文的核心做法是用一个统一图框架同时处理 variable-wise 与 patch-wise 两层结构。第一阶段通过 selective message passing 捕捉变量之间的异步依赖，避免所有变量在所有时刻做无差别交互；第二阶段用 hierarchical Patch-GNN 聚合多尺度时间片段，建模局部到全局的临床动态。这样模型既能学习“哪些变量之间在异步观测下仍然相关”，也能学习“哪些时间片段组合成可判别的风险模式”。

### 审稿人视角：价值与不足

最有价值的技术思想是把临床 IMTS 的结构拆成变量图和时间 patch 图两种互补视角。相比只在变量维做静态图，VP-GNN 通过 patch-wise 层补上多尺度时间演化；相比只做时间 patch 或 Transformer token mixing，variable-wise message passing 又让变量依赖更显式。对死亡预测和早期脓毒症检测这类任务而言，这种“变量依赖 + 多尺度片段”的结构归纳偏置比单纯扩大模型容量更有针对性。

不足在于，图结构仍可能吸收采样政策。selective message passing 选择哪些变量交互，Patch-GNN 选择哪些时间片段聚合，都会受到训练数据中观测频率、告警触发、化验流程和数据集采样协议影响。论文证明了在 P12/P19 benchmark 上性能提升，但还没有系统回答跨医院或跨测量策略时，变量边、patch 重要性和层级聚合路径是否保持稳定。

### 对 Sampling-Policy Shift 的启发

VP-GNN 对 sampling-policy shift 的横向启发是：策略偏移可以沿着“变量交互图”和“时间 patch 图”两条路径进入分类器。某家医院频繁联测的变量可能在 variable-wise graph 上形成强边；某种告警后密集采样的窗口可能在 patch-wise graph 上成为高权重片段。若这些结构随医院流程改变而改变，模型就可能依赖采样政策 shortcut。

纵向深化上，可以把 VP-GNN 扩展为 state-policy 双图框架：state graph 学习跨策略稳定的变量耦合和病程片段，policy graph 学习观测流程诱导的变量联测、采样密度和 patch 可见性。训练时对同一患者轨迹施加不同反事实采样策略，约束 state graph 表征和分类 logits 稳定，同时允许 policy graph 解释采样流程差异。还可以把 graph edge stability、patch selection frequency 和 cross-policy graph distance 作为评估指标，用来判断模型是否真正解决了非规则采样下的采样策略偏移。

## 追加更新 - 2026-08-02 23:00 UTC

### 本次检索与去重记录

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
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / EHR event-stream / ICU time-series classification 的顶会或顶会相关论文，重点核对 ICLR 2026 OpenReview、ICML 2026、AAAI 2026、KDD 2025/2026、NeurIPS 2025 官方页面、OpenReview、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 VITAL/Mind the Missing 这类暂无顶会录用信息的 arXiv 预印本、FORMED/UniShape 这类分类强但不原生处理 irregular sampling 的工作、MIRA/Time-IMM 这类更偏 forecasting 或 benchmark-only 且分类主任务较弱的候选。本次保留全新工作 2 篇：PULSE 直接评估 ICU time-series classification 的跨中心泛化与 LLM/传统模型差异；Time-Conditioned Foreseeing (TCF) 是 ICML 2026 Poster，面向带不规则时间戳的 EHR 事件流预训练，并在多项下游临床预测/分类任务上验证。

## 24. PULSE: Benchmarking Large Language Models for ICU Time Series Classification

- 会议/状态：ICLR 2026 OpenReview submission
- 作者：Jan Berner, Sophia F. Ehlers, Miklovana Tuci, Tim Hahn, Catherine R. Jutzeler, Lakmal Meegahapola
- OpenReview：https://openreview.net/forum?id=e5OrurYW07
- 代码：https://github.com/lakmalbuddikalucky/pulse-benchmark
- 关键词：ICU time series classification, LLM benchmark, mortality prediction, sepsis prediction, acute kidney injury, cross-domain shift, HiRID, MIMIC-IV, eICU

### 场景、任务与核心难点

PULSE 面向重症监护场景中的高风险时序分类任务，统一评估 HiRID、MIMIC-IV 和 eICU 三个 ICU 数据源上的 mortality、sepsis 与 acute kidney injury 预测。该类任务的难点不仅是输入多变量、缺失多、观测频率随病情和流程变化，而且还包括跨医院部署时的分布偏移：不同 ICU 的变量定义、采样频率、治疗流程、告警阈值和记录习惯都会改变同一个临床终点在数据中的可见形态。

论文的核心问题不是提出一个新的 irregular encoder，而是建立一个可复现实验场来回答：当 ICU 时序分类从同院同分布切换到跨院、少标注或零标注设置时，传统机器学习、深度时序模型与 instruction-following LLM 各自的边界在哪里。PULSE 比较 17 类模型，包含 RandomForest、XGBoost、LightGBM、CNN、LSTM、GRU、InceptionTime 以及 Llama、Mistral、GPT-4o、Gemini、Claude、Grok 等 LLM。结果显示，within-domain 设置下 LightGBM 仍能达到很强的 AUROC，而跨域测试时传统模型和深度模型退化明显，LLM 的 zero-shot / few-shot prompting 与 hybrid reasoning workflow 则表现出更强的 day-zero 可用性。

### 审稿人视角：价值与不足

最有价值的贡献是把近期“LLM 能否处理 ICU 时序分类”的讨论从单点方法比较推进到多中心、多终点、多模型族的基准审计。对审稿人而言，PULSE 的价值在于它没有只报告一个 LLM prompt 的成功案例，而是把强传统基线放进同一框架，并明确指出在标准同分布训练中 LightGBM 这类模型仍然很难被取代；LLM 的优势主要出现在跨域、少数据和新机构冷启动场景。这种结论比“LLM 全面优于时序模型”更可信，也更贴近临床部署。

不足在于，PULSE 仍主要是 benchmark / evaluation 工作，对不规则采样机制本身的可控分解有限。为了跨数据源可比，许多 ICU benchmark 往往会经历窗口化、聚合、变量标准化或任务定义对齐；这些处理提升了实验可复现性，但也可能把原始异步事件流中的采样政策细节压缩掉。论文强调 distribution shift，却还没有系统拆出 shift 中有多少来自 patient mix、变量 schema、采样频率、医院流程或治疗策略。LLM 在跨域中的稳健性也可能来自强先验和文本化提示，而不一定说明它真正学会了 sampling-policy-invariant 的时序状态。

### 对 Sampling-Policy Shift 的启发

PULSE 对我们的问题有直接横向启发：sampling-policy shift 应该被放在跨中心 benchmark 中显式评估，而不是只在单数据集内随机 mask 或改变缺失率。HiRID、MIMIC-IV 和 eICU 的跨域组合天然提供了不同采样政策、不同护理流程和不同变量定义下的环境划分；如果一个模型在同院表现强但跨院 AUROC/AUPRC 大幅下降，就说明它可能依赖了训练医院的观测流程 shortcut。

纵向深化上，可以在 PULSE 框架中加入 policy-aware evaluation 层：为每个样本统计变量采样频率、联测模式、delta-t 分布、value-pending 情况和告警后密集观测窗口，并报告 state-only、policy-only 与 full-model 三组性能。进一步可以把 LLM 的 prompt / reasoning workflow 改造成双通道输出：一部分总结稳定病程状态，一部分显式总结观测政策和数据质量；分类决策只允许依赖前者，后者用于校准和偏移诊断。这样能把 PULSE 从“谁在跨域上更稳”推进到“为什么跨域稳定，以及稳定性是否来自正确隔离采样政策”。

## 25. Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time

- 简称：TCF / TCF-PFM
- 会议：ICML 2026 Poster
- 作者：Bong Gyun Kang, Junyong Ahn, Hyeongrok Han, Sungroh Yoon
- 官方页：https://icml.cc/virtual/2026/poster/64928
- OpenReview：https://openreview.net/forum?id=Z8Hu7CJfZy
- 代码：https://github.com/Pusheen-cat/TCF_PFM
- 关键词：EHR foundation model, irregularly sampled timestamps, calendrical time, temporal generative pretraining, downstream clinical prediction, AUPRC

### 场景、任务与核心难点

TCF 面向 EHR 事件流中的长程临床预测：患者轨迹由化验、生命体征、用药、诊疗事件等 token 组成，每个 token 都带有不规则时间戳。与普通文本不同，EHR 的数值范围有强临床语义，例如异常高低值比常规值更重要；时间也不只是 token 顺序，而同时包含绝对日历时间、住院进程中的相对间隔、事件之间的等待时间和多尺度未来窗口。下游评估覆盖 MIMIC-III 等数据上的 IHM、decompensation、LOS、phenotyping、oliguria/anuria、vasopressor use 等临床预测/分类任务。

论文解决的核心难点是：如何让 EHR foundation model 原生理解“何时发生”和“未来哪个时间点会发生什么”，而不是照搬 NLP 的 next-token prediction。作者提出三部分设计：Pathology-Focused Binning 用密度/病理意义导向的方式量化数值，避免把临床关键异常值淹没在常见区间中；Dual-Calendar RoPE 同时编码绝对时间与相对时间间隔；Time-Conditioned Foreseeing objective 则在预测下一事件时间的同时，给定未来时间条件去预测未来事件。这样预训练目标更接近医生做治疗规划和风险预判的方式，而不只是按 token 顺序续写病历。

### 审稿人视角：价值与不足

最有价值的技术思想是把 irregular EHR 的“时间条件性”放进 foundation model 的预训练目标，而不仅是把时间戳当作附加 embedding。Pathology-Focused Binning 处理数值语义，Dual-Calendar RoPE 处理日历与间隔，TCF objective 处理多时间窗 foreseeing，三者共同把 EHR 从普通离散文本序列拉回到连续、不规则、临床决策相关的时序对象。对审稿人而言，这比单纯扩大 EHR Transformer 参数量更有说服力，因为它针对 EHR 的关键结构错配给出了具体机制，并在多项下游任务上报告了 AUPRC 提升。

不足在于，TCF 仍然是生成式 EHR 预训练框架，采样政策与真实病程的分离还不充分。EHR 中“未来哪个事件会发生”往往同时由病人状态、医生怀疑、医院流程、检查可及性和记录习惯决定；因此 Time-Conditioned Foreseeing 学到的未来事件分布可能混合 patient state 与 care process。若某家医院更倾向于在高风险患者身上提前开某些化验或治疗，模型可能把这种策略当作稳定病程模式。论文展示了多任务预测收益和时间一致生成，但还需要跨医院、跨采样协议、反事实时间戳扰动下的系统评估，才能证明其时间表示不是 policy shortcut。

### 对 Sampling-Policy Shift 的启发

TCF 对 sampling-policy shift 的横向启发是：采样政策偏移不仅改变“观测是否出现”，还会改变模型预训练时要预测的未来事件时间分布。若预训练目标要求模型预测未来某个时间窗会出现哪些化验、诊疗或观测事件，那么它天然会学习医院如何观察病人；这既是信息来源，也是潜在 shortcut。因此，我们可以把 TCF 式 future-event likelihood 分解为 state likelihood 与 policy likelihood：前者描述真实病程会导致什么临床状态，后者描述医院会在何时记录或测量什么。

纵向深化上，可以设计 policy-robust TCF：Dual-Calendar RoPE 保留真实时间结构，但在 representation 中显式分出 state time embedding 与 policy calendar embedding；foreseeing objective 分成 future-state prediction 和 future-observation-process prediction 两个头。对同一患者潜在轨迹施加不同反事实采样策略时，要求 state foreseeing、下游 logits 和关键病理 bin 表示保持一致，同时允许 policy head 预测不同的观测密度、事件可见性和检查触发时间。这样能把 TCF 的时间条件预训练从“更好预测 EHR 未来事件”推进到“区分未来病程与未来观测政策”，更贴近我们要解决的 Sampling-Policy Shift。
