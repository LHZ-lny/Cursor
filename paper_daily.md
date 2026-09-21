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

## 追加更新 - 2026-08-22 23:00 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`；同时读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-08-21 的新增标题，纳入黑名单。
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
  - PULSE: Benchmarking Large Language Models for ICU Time Series Classification
  - Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time
  - Context-Aware Neural SDEs for Robust Irregular Time-Series Classification
  - Context-Informed Sequence Classification: A Multimodal Approach to Vehicle Diagnostics
  - Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness
  - RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification
  - NeurOCNN: A Neural-Operator-Based Model for Physiological Time Series
  - Cached Foundation Model Summaries for Memory-Efficient Clinical Time Series Inference
  - DeNOTS: Stable Deep Neural ODEs for Time Series
  - Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time
  - PathwayLLM: Explainable Clinical Trajectory Modeling with Structured Pathways for Sepsis Prediction
  - ArcTimeSDE: Aligning Compute with Information Via ARC Length Time in Neural SDEs
  - MoRGen: Mixture-of-Resolutions Generative Forecasting for Irregularly Sampled Medical Time-Series Data
  - STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data
  - Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment
  - Repurposing Foundation Model for Generalizable Medical Time Series Classification
  - Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series
  - Investigating a Model-Agnostic and Imputation-Free Approach for Irregularly-Sampled Multivariate Time-Series Modeling
  - TreeText-CTS: Compact, Source-Traceable Tree-Path Evidence for Irregular Clinical Time-Series Prediction
  - Random Controlled Differential Equations
  - An Automated Data Engineering Pipeline for Time Series Classification Via Text Embeddings
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / clinical irregular time series classification 的顶会论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、NeurIPS 2025、KDD 2025/2026、OpenReview、ICLR/ICML virtual site、AAAI Proceedings 与 arXiv 页面。
- 已排除全部黑名单论文；同时继续排除历史日报已明确标记的 MedFuse withdrawn submission、VITAL/Mind the Missing 这类暂无顶会录用信息的预印本、TranSCANE/MUSE-Net 等时间窗口或 venue 不匹配候选，以及 ReDiTT/Time-IMM 等偏 forecasting/generation 的工作。本次保留全新工作 1 篇：CauKer 是 ICLR 2026 Oral，严格来说不是 IMTS 专用模型，但它面向 time series classification foundation models，提出可控、因果一致的合成预训练数据生成机制；在直接 IMTS 顶会命中几乎已被历史日报覆盖的情况下，它对构造 sampling-policy shift 的反事实预训练与评测最有新增价值。

## 26. CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data

- 会议：ICLR 2026 Oral
- 作者：Shifeng Xie, Vasilii Feofanov, Jianfeng Zhang, Themis Palpanas, Ievgen Redko
- OpenReview：https://openreview.net/forum?id=xBW2FIfswU
- 官方页：https://iclr.cc/virtual/2026/poster/10006662
- 论文：https://arxiv.org/abs/2508.02879
- 代码：https://github.com/ShifengXIE/CauKer
- 关键词：time series classification, foundation model pretraining, synthetic data, Gaussian Process kernel composition, Structural Causal Models, scaling laws

### 场景、任务与核心难点

CauKer 面向 time series classification foundation models 的预训练问题。近两年 TSFM 的主流路线常依赖大规模真实时序语料，但分类任务的数据来源高度碎片化：UCR/UEA、医疗、生理、工业和科学数据之间的采样频率、变量数、噪声、类别语义和长度分布差异极大。若直接堆叠真实数据做预训练，成本高、版权和隐私约束重，而且论文观察到真实数据上的 scaling law 很不规则：数据量或模型规模变大并不稳定带来更好的 zero-shot 分类能力。

论文的核心做法是用合成数据替代大规模真实预训练语料。CauKer 先通过 Gaussian Process kernel composition 组合趋势、周期、平滑性、非平稳性等时间结构，再用 Structural Causal Models 将多个生成过程按因果图传播和耦合，从而生成具有现实形态、非线性交互和因果一致性的分类时序。作者用这些合成序列预训练不同架构的分类 TSFM，并在多个真实分类 benchmark 上做 zero-shot / transfer 评估；结果显示，合成数据不仅能达到或接近真实数据预训练效果，还呈现更清晰的数据规模与模型规模 scaling law。

### 审稿人视角：价值与不足

最有价值的思想是把 time series classification 的基础模型瓶颈从“再收集更多真实序列”转向“能否设计足够结构化、可控、覆盖广泛机制的合成世界”。这对审稿人很有吸引力，因为它不是单纯的数据增强，而是把 GP kernel 的时间形态组合能力和 SCM 的变量生成依赖结合起来，让预训练数据具备可解释的生成因素。它也说明分类 TSFM 的泛化能力可能更依赖训练分布覆盖到足够多的形态机制，而不一定依赖真实数据本身。

不足在于，CauKer 主要面向规则或标准化后的通用时序分类 benchmark，还没有把不规则采样、变量异步、informative missingness 或医院观测策略显式作为生成因子。GP kernel composition 能产生丰富的状态轨迹，SCM 能产生变量间依赖，但当前 observation process 更像隐含或固定的采样层；如果目标是 ICU/EHR 这类非规则采样分类，真实难点恰恰在于“哪些值被看见、何时被看见、为什么被看见”本身与标签和环境耦合。因此，这篇论文证明了可控合成预训练的潜力，但尚未证明其合成世界覆盖了 sampling-policy shift 下最关键的观测机制变化。

### 对 Sampling-Policy Shift 的启发

CauKer 对 Sampling-Policy Shift 的横向启发是：我们可以把采样策略偏移从被动 benchmark 现象，升级为可控合成预训练和评测维度。现有许多 IMTS 方法只在真实数据或随机 mask 下测试鲁棒性，难以知道模型失败来自状态动力学改变、噪声改变，还是采样政策改变。CauKer 的 GP + SCM 框架提供了一个生成底座：先生成同一潜在连续状态和类别机制，再叠加多套 observation policy，例如固定间隔采样、阈值触发采样、告警后密集采样、成本约束下变量选择、医院特定联测规则等。

纵向深化上，可以设计 policy-aware CauKer：把生成过程拆成 state generator、label generator 和 sampling-policy generator 三层。预训练时要求模型在不同 observation policy 下保持 state representation 和分类 logits 一致，同时显式预测 policy metadata 或 sampling trace，用来做偏移诊断。评测时则报告 in-policy、cross-policy、counterfactual-policy 三类指标，并检验模型是否在只给时间戳/mask 的 policy-only 设置下泄漏标签。这样能把 CauKer 的“合成数据可替代真实预训练语料”推进到“合成采样政策可系统训练和审计 policy-invariant irregular time-series classifiers”。

## 追加更新 - 2026-08-23 23:02 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`；同时读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-08-21 的新增标题，纳入黑名单。
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
  - PULSE: Benchmarking Large Language Models for ICU Time Series Classification
  - Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time
  - Context-Aware Neural SDEs for Robust Irregular Time-Series Classification
  - Context-Informed Sequence Classification: A Multimodal Approach to Vehicle Diagnostics
  - Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness
  - RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification
  - NeurOCNN: A Neural-Operator-Based Model for Physiological Time Series
  - Cached Foundation Model Summaries for Memory-Efficient Clinical Time Series Inference
  - DeNOTS: Stable Deep Neural ODEs for Time Series
  - Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time
  - PathwayLLM: Explainable Clinical Trajectory Modeling with Structured Pathways for Sepsis Prediction
  - ArcTimeSDE: Aligning Compute with Information Via ARC Length Time in Neural SDEs
  - MoRGen: Mixture-of-Resolutions Generative Forecasting for Irregularly Sampled Medical Time-Series Data
  - STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data
  - Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment
  - Repurposing Foundation Model for Generalizable Medical Time Series Classification
  - Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series
  - Investigating a Model-Agnostic and Imputation-Free Approach for Irregularly-Sampled Multivariate Time-Series Modeling
  - TreeText-CTS: Compact, Source-Traceable Tree-Path Evidence for Irregular Clinical Time-Series Prediction
  - Random Controlled Differential Equations
  - An Automated Data Engineering Pipeline for Time Series Classification Via Text Embeddings
  - CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregularly sampled medical time series / ICU time-series classification / informative sampling / wearable irregular health time series 的顶会、顶会 workshop、OpenReview 与 arXiv 页面，重点核对 ICLR 2026、ICML 2026、AAAI 2026、NeurIPS 2025 TS4H、ICLR 2026 TSALM、OpenReview、arXiv 与项目页。
- 已排除全部黑名单论文。严格的“顶会正会 + 直接 IMTS 分类”新增命中基本已被历史日报覆盖；本次保留 2 篇未在黑名单中的新跟踪对象：JETS 是 NeurIPS 2025 TS4H workshop 工作，直接使用 irregular triplet wearable time series 做诊断预测；TRIAGE 是 2026-06 新预印本，尚未确认顶会录用，但明确面向 irregularly sampled medical time series 的临床风险分类、校准和解释，对 sampling-policy shift 的可解释诊断价值较高。

## 27. JETS: A Self-Supervised Joint Embedding Time Series Foundation Model for Behavioral Data in Healthcare

- 会议/状态：NeurIPS 2025 Workshop on Learning from Time-Series for Health (TS4H)
- 作者：Erik Xie, Raquel Rodriguez Martinez, Wyatt Chang, Brandon Ballinger
- OpenReview：https://openreview.net/forum?id=i4epRiMy8z
- 参考解读：https://www.empirical.health/blog/wearable-foundation-models
- 关键词：irregular wearable time series, behavioral health data, JEPA, triplet tokenization, self-supervised foundation model, diagnostic prediction

### 场景、任务与核心难点

JETS 面向可穿戴设备产生的真实世界健康行为时间序列。输入不是 ICU 化验表，而是 Apple Watch、Samsung Galaxy、Fitbit 等设备上的 63 通道行为/生理指标，例如血氧、静息心率、睡眠阶段、HRV 等；这些数据以 `(timestamp, value, metric type)` triplets 表示，而不是强行对齐到规则网格。下游任务包括 hypertension、sick sinus syndrome、ME/CFS、atrial flutter 等个体级诊断预测，以及 HbA1c、glucose、HDL、hs-CRP 等 biomarker 预测。

核心难点在于，消费级可穿戴数据天然稀疏、碎片化且强噪声：用户摘表、充电、设备型号、系统采样策略、传感器质量和日常行为都会改变哪些指标被记录、何时记录、持续多久。传统 MAE 式重构容易把大量传感器噪声和缺测 artifact 当作目标；规则化网格又会把真实的 wearing / charging / device policy 差异抹平成插补值。JETS 采用 JEPA 风格的 joint-embedding predictive architecture：一个 encoder 看完整序列，另一个 encoder 看随机 30% token，predictor 在 latent space 中预测完整视图表示；模型关注可迁移的行为-生理结构，而不是逐点重构原始噪声。

### 审稿人视角：价值与不足

最有价值的思想是把 wearable health foundation model 从“规则窗口上的传感器重构”推进到“非规则 triplet 序列上的 latent predictive learning”。这对审稿人有吸引力，因为它同时处理了三个现实部署问题：输入异构、观测稀疏、标签稀缺。JEPA 的 latent-space prediction 比 raw reconstruction 更适合可穿戴数据，因为很多缺口和短期尖峰并不是稳定健康状态，而是设备和用户行为造成的观测噪声；让模型预测高层表示有助于忽略不可迁移细节。

不足在于，JETS 目前是 workshop 级别工作，公开细节和系统消融相对有限；它的“不规则性”主要来自 wearable usage 与设备采样，而不是 ICU/EHR 中由医生决策触发的化验和治疗记录。更重要的是，wearable observation policy 本身高度带有健康和社会行为信息：病人越不舒服可能越少佩戴设备，某些设备或地区的采样配置不同，睡眠/运动指标的可见性也受用户合规性影响。若没有跨设备、跨用户群、跨佩戴策略的环境划分，模型学到的 embedding 可能仍混合了 health state 与 device/user policy。

### 对 Sampling-Policy Shift 的启发

JETS 对 Sampling-Policy Shift 的横向启发是：采样政策不一定来自医院，也可能来自用户-设备交互。可穿戴数据中的 missingness、metric availability、charging gaps、sensor duty cycle 和 firmware-level aggregation 都是 observation policy 的组成部分；这些政策变化会影响诊断分类器看到的行为轨迹，甚至比真实生理状态更容易被模型捕捉。

纵向深化上，可以把 JETS 改造成 state-policy 双嵌入的 wearable foundation model。state branch 用 JEPA 预测跨佩戴策略稳定的健康状态表示，policy branch 则预测设备型号、佩戴/充电模式、采样密度和通道可用性。训练时对同一连续行为轨迹模拟不同设备采样策略、不同缺测块和不同 metric subsets，约束 state embedding 与诊断 logits 稳定，同时允许 policy embedding 改变。这样能把 JETS 的 triplet-token JEPA 推进到“可穿戴采样政策变化下仍保持诊断语义稳定”的不规则健康时序分类器。

## 28. TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs

- 会议/状态：arXiv 2026-06 预印本，尚未确认顶会录用
- 作者：Hyeongwon Jang, Gyouk Chu, Changhun Kim, Joonhyung Park, Hangyul Yoon, Eunho Yang
- 论文：https://arxiv.org/abs/2606.09030
- 代码：https://github.com/HyeongWon-Jang/TRIAGE
- 关键词：irregularly sampled medical time series, clinical risk prediction, LLM reasoning, calibration, explainability, risk polarization

### 场景、任务与核心难点

TRIAGE 面向基于 EHR 的临床早期预警系统。输入是 irregularly sampled medical time series：生命体征、化验、病程记录等临床观测以不规则时间出现，模型需要输出可用于 triage 的连续风险分数，并给出临床人员可以核查的解释。与只追求 AUROC/AUPRC 的分类器不同，这类系统还要求校准良好、风险可比较、解释可追溯，因为过度自信的二分类判断会直接影响分诊优先级。

论文指出 LLM 用于临床时序风险预测时容易出现 risk polarization：模型把本应是连续概率的问题压成过度自信的阳性/阴性结论，导致校准差、跨患者风险不可比。TRIAGE 的核心做法是 dialectical reasoning：不让 LLM 只为单一结果找理由，而是为 competing clinical outcomes 分别生成 outcome-specific rationales，再由这些相互竞争的理由导出连续风险分数。训练包括 Dialectical Reasoning Supervision 和 Self-Refinement；论文在 3 个 ISMTS benchmark 上报告平均 AUPRC 提升 3.3%、calibration error 降低 81%，并用 LLM-as-a-judge 评估 rationale 质量。

### 审稿人视角：价值与不足

最有价值的思想是把 irregular clinical time-series classification 的 LLM 路线从“把序列文本化后直接问标签”推进到“风险估计必须同时比较正反临床证据”。这很重要，因为临床 triage 本质上是连续风险排序和证据权衡，而不是简单二分类。Dialectical rationale 也提供了一个可以审查的中间层：模型为什么认为 sepsis / mortality / deterioration 更可能，为什么相反结果仍有一定概率，这比单一链式解释更接近医生讨论风险的方式。

不足在于，TRIAGE 尚未确认顶会录用，当前应作为高相关新预印本跟踪，而不是已验证的正会工作。技术上，它仍依赖 LLM 对数值时序的序列化理解；若输入摘要或 tokenization 已经混入采样政策 shortcut，dialectical reasoning 可能只是把这些 shortcut 解释得更流畅。校准提升主要说明输出概率更可用，但不等于解释是因果正确的；跨医院、跨采样协议、value-pending、变量联测规则改变时，rationale 是否仍指向稳定病程证据，还需要系统实验。

### 对 Sampling-Policy Shift 的启发

TRIAGE 对 Sampling-Policy Shift 的横向启发是：解释层可以成为采样政策泄漏的诊断窗口。若模型在 rationale 中频繁把“某项检查被下单”“近期观测更密集”“某变量长时间未测”作为直接风险证据，我们就能更明确地区分它是在利用真实病程，还是在利用医院观察流程。相比只看 logits，rationale 更容易暴露 policy shortcut。

纵向深化上，可以设计 policy-audited dialectical reasoning：要求 LLM 分别输出 state rationale 与 policy rationale。state rationale 只能引用跨采样策略稳定的病情变化、数值趋势和临床机制；policy rationale 则解释观测为何出现、为何缺失、为何某些变量联测，只用于校准和偏移警报。对同一患者潜在轨迹生成不同反事实采样策略时，要求 state rationale、连续风险分数和分类排序保持稳定，同时允许 policy rationale 改变。这样能把 TRIAGE 的“可解释风险校准”进一步变成 sampling-policy shift 下的可审计分类决策。

## 追加更新 - 2026-08-24 23:00 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`；同时读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-08-21 的新增标题。
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
  - PULSE: Benchmarking Large Language Models for ICU Time Series Classification
  - Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time
  - Context-Aware Neural SDEs for Robust Irregular Time-Series Classification
  - Context-Informed Sequence Classification: A Multimodal Approach to Vehicle Diagnostics
  - Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness
  - RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification
  - NeurOCNN: A Neural-Operator-Based Model for Physiological Time Series
  - Cached Foundation Model Summaries for Memory-Efficient Clinical Time Series Inference
  - DeNOTS: Stable Deep Neural ODEs for Time Series
  - Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time
  - PathwayLLM: Explainable Clinical Trajectory Modeling with Structured Pathways for Sepsis Prediction
  - ArcTimeSDE: Aligning Compute with Information Via ARC Length Time in Neural SDEs
  - MoRGen: Mixture-of-Resolutions Generative Forecasting for Irregularly Sampled Medical Time-Series Data
  - STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data
  - Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment
  - Repurposing Foundation Model for Generalizable Medical Time Series Classification
  - Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series
  - Investigating a Model-Agnostic and Imputation-Free Approach for Irregularly-Sampled Multivariate Time-Series Modeling
  - TreeText-CTS: Compact, Source-Traceable Tree-Path Evidence for Irregular Clinical Time-Series Prediction
  - Random Controlled Differential Equations
  - An Automated Data Engineering Pipeline for Time Series Classification Via Text Embeddings
  - CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data
  - JETS: A Self-Supervised Joint Embedding Time Series Foundation Model for Behavioral Data in Healthcare
  - TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / EHR event-stream clinical prediction 的顶会或顶会相关论文，重点核对 AAAI 2026、ICML 2026、ICLR 2026、ICASSP 2026、KDD 2025/2026、OpenReview、AAAI Proceedings、ICML/ICLR virtual site、IEEE/ICASSP 页面、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 ViTST/SEFT/KEDGN 等时间窗口过旧条目、Hi-Patch/GRUwE/ASTGI/ReIMTS 等偏 forecasting 或 event prediction 而非分类主任务条目、RC-GRF/Transformer autoencoder 等暂无顶会来源的预印本，以及 TreeText-CTS/TRIAGE 等已由自动化记忆标记过的条目。本次保留全新工作 2 篇：MATA-Former & SIICU 是 AAAI 2026 相关 ICU 风险预测工作，直接处理 text-intensive irregular clinical time series；Cross-Representation Benchmarking 是 ICASSP 2026 论文，系统比较 EHR 多变量时序、事件流和文本事件流表征，对非规则采样下的表示选择与 sampling-policy shift 评估有直接方法论价值。

## 29. MATA-Former & SIICU: Semantic Aware Temporal Alignment for High-Fidelity ICU Risk Prediction

- 方法名：Medical-semantics Aware Time-ALiBi Transformer (MATA-Former)
- 会议/状态：AAAI 2026 相关记录；arXiv 2026-04 版本
- 作者：Zhichong Zheng, Xiaohang Nie, Xueqi Wang, Yuanjin Zhao, Haitao Zhang, Yichao Tang
- 论文：https://arxiv.org/abs/2604.01727
- 佐证记录：https://www.csauthors.net/yichao-tang/
- 关键词：irregular clinical time series, ICU risk prediction, semantic-aware temporal alignment, text-intensive EHR, soft labeling, event-wise risk modeling

### 场景、任务与核心难点

MATA-Former 面向 ICU 临床早期预警和多风险预测。输入不是单纯规则化生命体征矩阵，而是由结构化 vitals/labs 与非结构化护理记录、影像报告、操作记录等文本密集事件交织而成的 ICU 轨迹。论文强调 SIICU 数据具有高度异质和稀疏特征：每位患者事件数重尾分布，时间间隔跨越秒、分钟、小时到天，既有自动监护产生的高频结构化数据，也有人工干预和临床叙事产生的 sporadic events。

核心难点在于，ICU 风险并不总是随物理时间距离单调衰减。某些事件的有效期很短，例如急性生命体征变化；某些诊疗、用药或文本事件的病理影响可能持续很久，并且其重要性取决于当前要预测的风险类型。常规 time-aware attention 或 ALiBi 类方法通常把时间距离作为统一衰减项，容易低估远期但语义上仍有效的证据。MATA-Former 因此用医学事件语义来动态参数化时间注意力，使注意力窗口由 risk query 与 clinical event semantics 决定，而不是只由时间戳距离决定。论文还提出 Plateau-Gaussian Soft Labeling，将粗粒度二分类标签转为连续多时间窗风险轨迹，以更细地监督风险从潜伏、发展到恢复的过程。

### 审稿人视角：价值与不足

最有价值的思想是把“时间对齐”从几何距离问题提升为“医学语义有效期”问题。在很多 irregular EHR 模型中，时间戳只是 positional encoding 或 decay feature；MATA-Former 则明确指出，同样过去 12 小时的事件，在不同风险预测任务中可能有完全不同的相关性。用 event semantics 生成 query-dependent temporal bias，可以把临床知识和连续时间建模结合起来，尤其适合 text-intensive ICU 场景中远期病程记录、用药和护理事件的长效影响。SIICU 数据集的人机协同细粒度标注也有价值，因为它试图避免只用 discharge-level ICD 或粗二分类标签带来的 look-ahead bias 和监督稀疏。

不足在于，MATA-Former 将医学语义和时间偏置绑定得更紧，但还没有充分分解“病理语义”与“记录/采样政策语义”。ICU 文本事件、护理记录和操作记录本身高度受医院流程、科室习惯、告警阈值和人力资源影响；某条记录的出现可能说明病情，也可能说明某个班次更勤于记录、某类患者进入特定诊疗流程或某院区有更细的文书规范。MATA-Former 的语义注意力可能更好地捕获这些事件，但不保证捕获的是跨机构稳定病理机制。另一个风险是 SIICU 的 LLM 预标注 + 人工校验流程虽提高了细粒度标签密度，但其标签体系和事件语义是否能跨医院迁移仍需要外部验证。

### 对 Sampling-Policy Shift 的启发

MATA-Former 对 sampling-policy shift 的横向启发是：采样策略偏移不仅改变观测时间和缺失模式，还会改变“哪些语义事件被写入记录、何时写入、以什么粒度写入”。如果一个模型把文本事件语义直接作为时间注意力的调度器，那么采样/记录政策变化会通过 temporal bias 进入分类边界。我们可以借鉴其 semantic-aware temporal bias，但需要把 bias 分解为 state-validity bias 与 policy-documentation bias：前者表示某类病理证据的真实有效期，后者表示医院记录流程和采样习惯。

纵向深化上，可以设计 policy-aware MATA：对同一潜在病程构造不同记录密度、不同 note timing、不同 lab-ordering policy 的反事实视图，要求 state temporal bias、风险轨迹和 logits 保持一致，同时允许 policy bias 解释文书频率、检查触发和事件可见性差异。MATA-Former 的 PSL 也可用于评估策略偏移：如果换采样/记录政策后，连续风险曲线的形状大幅改变，但临床状态没有改变，就说明模型仍把 observation process 当成 disease process。

## 30. Cross-Representation Benchmarking in Time-Series Electronic Health Records for Clinical Outcome Prediction

- 会议：ICASSP 2026
- 作者：Tianyi Chen, Mingcheng Zhu, Zhiyao Luo, Tingting Zhu
- 官方记录：https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=16742
- 论文：https://arxiv.org/abs/2510.09159
- 代码：https://github.com/BrandonC8310/EHR-Cross-Representation-Benchmarks
- 关键词：EHR clinical outcome prediction, multivariate time series, event streams, textual event streams, missingness-based feature pruning, representation benchmark

### 场景、任务与核心难点

这篇工作面向 EHR 临床结局预测中的表示选择问题。它比较三类输入范式：将 EHR 聚合成规则多变量时序矩阵、保留时间戳的事件流，以及把事件流转成可供 LLM 使用的文本事件流。任务覆盖 MIMIC-IV ICU mortality、ICU phenotyping，以及 EHRSHOT 中的 30-day readmission 和 1-year pancreatic cancer 等临床预测/分类任务。

核心难点不是提出一个更复杂的单模型，而是回答一个更基础的问题：在非规则 EHR 中，究竟应把数据整理成哪种表示，才能在公平评估下得到可靠结论？传统 ICU benchmark 常把事件按小时或天聚合、前后向填补、再做 population-median imputation；这种做法便于复用 LSTM/Transformer/MLP，却会扭曲原始时间信息，并把 missingness 和 sampling density 压成难以解释的网格 artifact。事件流表示保留了不规则时间和原始事件顺序，文本事件流进一步给 LLM 提供可读语义，但三者过去常在不同数据切分、不同任务定义和不同预处理下比较，难以判断性能差异来自模型能力还是 representation/curation choices。

### 审稿人视角：价值与不足

最有价值的贡献是把 EHR 表示选择做成统一 benchmark，而不是默认某一种表示天然最优。论文在相同 cohorting、labeling、split 和 metric 下比较多变量时序模型、CLMBR/计数类事件流模型，以及 8-20B LLM 文本流模型，并进一步分析 missingness-based feature pruning。结果显示 event stream models 整体表现最强，CLMBR 在 few-shot 下样本效率高，而稀疏特征剪枝对 ICU 与纵向任务的影响不同：ICU 中剪掉高度缺失特征常能简化模型且损失较小，纵向任务中稀疏特征反而可能关键。

不足是，benchmark 仍然偏经验比较，并没有显式建模采样政策因果机制。多变量时序表示被规则化到固定网格，event stream 表示保留原始时间，但论文尚未把医院、科室、设备、记录流程或医生下单策略作为环境变量进行 cross-policy split。文本事件流中的 LLM 表现也可能受 prompt、上下文长度和事件 verbalization 影响；如果文本化规则把采样密度、缺失或记录习惯转成显著语言线索，LLM 可能学习到 policy shortcut。换言之，它能告诉我们不同表示在统一协议下谁更强，但还不能完全解释强弱差异中有多少来自 state information、有多少来自 observation policy。

### 对 Sampling-Policy Shift 的启发

这篇工作对 sampling-policy shift 的横向启发非常直接：研究策略偏移前，必须先把 representation choice 当作实验变量。相同 EHR 原始数据可以被转换成网格时序、事件流或文本事件流；不同转换会保留或抹平不同类型的采样政策信息。例如网格化和 imputation 可能把策略信息压缩为填充值和 mask，事件流显式保留事件可见性，文本流可能把缺失和检查行为转成语义描述。若不控制表示选择，所谓 policy robustness 可能只是某种表示恰好隐藏或暴露了 policy shortcut。

纵向深化上，可以在该 benchmark 框架上增加 policy-aware evaluation：为每个表示同时构造 state-only、policy-only 和 full-input 三组模型，并按医院/科室/采样密度/联测模式/缺失率分层报告跨环境性能。还可以比较同一反事实采样策略在三种表示中的传播路径：网格表示中表现为 mask 和 imputed value 变化，事件流中表现为 token sequence 变化，文本流中表现为叙述证据变化。这样能帮助我们判断 Sampling-Policy Shift 的解决方案到底应发生在数据表示层、encoder 层，还是分类/校准层。

## 追加更新 - 2026-08-25 23:01 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`；同时读取兼容入口 `paper_daily.md` 的标题索引，并参考自动化记忆中 2026-08-03 至 2026-08-21 的新增标题。
- 本次黑名单论文标题已覆盖既有 50 篇历史总结，包括 `Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`MATA-Former & SIICU: Semantic Aware Temporal Alignment for High-Fidelity ICU Risk Prediction` 与 `Cross-Representation Benchmarking in Time-Series Electronic Health Records for Clinical Outcome Prediction` 等全部已总结标题；新候选均已逐项排除黑名单重名。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / informative irregularity / clinical workflow metadata / sampling policy shift 的顶会或顶会 workshop 论文，重点核对 ICLR 2026、ICML 2026、AAAI 2026、ICASSP 2026、NeurIPS 2025、OpenReview、ICML virtual site、AAAI Proceedings、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 TimEE、SGN、FORMED 等不以 irregular sampling 为核心或已被历史日报覆盖的工作，以及 MIRA/ReDiTT/Time-IMM 等偏 forecasting/generation 的条目。严格的“顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 2 篇未在黑名单中的 ICML 2026 Structured Data for Health workshop 新工作：`Informative Irregularity` 直接把临床观测时间与工作流元数据用于预测鲁棒性诊断；`SBRD` 虽更偏临床决策/动作预测而非传统时序分类，但其 shared benchmark + switching regime 分解对 Sampling-Policy Shift 的状态-政策拆分非常有启发。

## 31. Informative Irregularity as a Diagnostic for Model Robustness

- 会议/状态：ICML 2026 Workshop on Structured Data for Health Poster
- 作者：Tamara Krafft, Bernhard Bauer
- 官方页：https://icml.cc/virtual/2026/71781
- 关键词：irregularly sampled EHR, structural workflow metadata, robustness stratification, clinical observation timing, MIMIC-IV-ED, AUPRC

### 场景、任务与核心难点

这篇工作面向急诊/临床 EHR 中的不规则观测建模。输入不是理想化的规则生命体征矩阵，而是 MIMIC-IV-ED 中 3,497 次就诊的临床观测序列；论文关注的核心任务是临床预测模型在不同 irregularity regime 下的鲁棒性，并检验“何时被测量、是否偏离本地常规节律、工作流结构如何变化”本身是否能补充生命体征数值。

核心难点在于，临床观测时间不是随机噪声。患者病情恶化、医生关注度、科室工作流、检查排队和本地 routine rhythm 都会改变观测时间；这些信号对 acuity 有预测价值，但也可能是医院特定采样政策的产物。论文因此抽取 structural workflow metadata，并按 distinct irregularity regimes 做分层评估。结果显示，加入结构元数据相对 native baseline 带来 +4.1% AUPRC；特征重要性分析显示 workflow metadata 与生命体征 top features 的平均相关绝对值仅约 0.041，说明其提供了相对独立的诊断信号；当采样不遵循本地 routine rhythm 时，结构元数据收益进一步达到 +8.0% AUPRC。

### 审稿人视角：价值与不足

最有价值的思想是把 irregularity 从“需要被模型忍受的缺陷”转化为“可以被量化、分层和审计的工作流信号”。很多 IMTS 论文只报告随机 missing ratio 或同分布测试集性能，而这篇论文要求模型回答更细的问题：在常规节律、非例行采样、局部工作流异常等不同 regime 下，性能为何变化？这种 stratification 对审稿人很有吸引力，因为它能把平均 AUPRC 背后的 failure mode 暴露出来，也能避免把采样信息简单归为有用或有害。

不足在于，它是 workshop 工作，贡献更偏诊断框架和特征分析，而不是一个完整的新分类架构。实验数据集中在 MIMIC-IV-ED 的单一场景，structural workflow metadata 的定义也可能依赖本地记录系统；若换医院、换急诊流程或换变量 schema，这些 metadata 是否可迁移仍需验证。此外，AUPRC 提升证明采样/工作流信息有预测力，但不等于它是跨环境稳定的病理信号；它也可能是高价值但高风险的 policy shortcut。

### 对 Sampling-Policy Shift 的启发

这篇论文对 Sampling-Policy Shift 的横向启发非常直接：采样政策偏移应该被显式变成评估分层，而不仅是隐藏在整体测试集中的分布变化。我们可以借鉴其 workflow metadata 思路，为每条不规则序列构造 policy descriptors，例如 routine-rhythm deviation、变量联测强度、告警后采样密度、long-gap pattern、夜班/日班观测差异和 value-pending 状态，然后分别报告 state-only、policy-only 与 full-model 的分类表现。

纵向深化上，可以设计 policy-diagnostic IMTS classifier：主干只用观测值和稳定时间结构学习 patient-state representation；并行 policy head 预测 workflow/irregularity regime，用于偏移告警和校准，而不直接进入最终分类边界。训练时对同一潜在轨迹施加不同采样策略增强，要求 state logits 稳定，同时允许 policy descriptors 改变。这样能把“informative irregularity 可提升 AUPRC”推进到“哪些 irregularity 是可迁移病情信号，哪些只是当前医院采样政策”的可审计区分。

## 32. Shared-Benchmark Regime Decomposition for Nonstationary Clinical Decisions

- 简称：SBRD
- 会议/状态：ICML 2026 Workshop on Structured Data for Health Poster
- 作者：Yuting Yan, Haozhou Gao, Xinye Chen, Yinghao Fu, Shuang Li
- 官方页：https://icml.cc/virtual/2026/71707
- 关键词：irregularly sampled physiological signals, clinical decision sequences, nonstationarity, latent regimes, continuous-time latent dynamics, ICU sepsis treatment

### 场景、任务与核心难点

SBRD 面向结构化临床决策序列，包括 tabular EHR 与不规则采样 physiological signals。论文的目标不是传统“给整条时间序列打疾病类别标签”，而是预测 ICU sepsis treatment 与 CKD-MBD management 中的临床动作，并解释为什么相似 measured states 在不同时间或不同约束下会导致不同决策。这可视为一种 policy-sensitive sequential classification：模型需要把患者当前 belief state、可选动作价值和临床流程约束同时纳入判断。

核心难点是部分可观测和非平稳性。相同的生理观测值并不总是对应相同治疗动作，因为 ICU 资源、协议压力、治疗复杂度、阶段性约束和医生行为 regime 会改变动作分布。SBRD 提出 shared-benchmark regime decomposition：用 continuous-time latent dynamics layer 从结构化健康记录中构造 belief states 和 benchmark action values，再用 sparse regime layer 恢复 persistent、可解释的 constraint-regime deviations，也就是相对共享 benchmark 的策略性 wedge。实验显示该分解能提升 held-out action prediction，并给出 clinically meaningful regime profiles。

### 审稿人视角：价值与不足

最有价值的思想是把“稳定状态价值”与“环境/约束 regime 偏差”拆开。对 sampling-policy shift 而言，这比单纯提高分类 accuracy 更关键：很多模型失败不是因为无法表示患者状态，而是把某一环境下的观察/治疗政策当成了状态本身。SBRD 的 shared benchmark 相当于寻找跨 regime 可共享的临床决策基线，sparse regime layer 则把协议、资源或时间阶段造成的偏差显式化。这个结构为“状态机制 vs 政策机制”的分解提供了可操作模板。

不足也很明确：它不是标准 irregular time series classification 正会论文，而是 ICML workshop 中偏临床决策建模的工作；任务中心是 action prediction 和决策 regime 解释，不是 P12/P19/PAM 这类 IMTS 分类 benchmark。latent regime 的可识别性依赖建模假设，若未观测混杂同时影响病情和决策，benchmark value 与 regime wedge 仍可能纠缠。论文也尚未系统评估换医院采样协议、换测量频率或反事实 observation policy 后，belief state 与 regime layer 是否保持预期分工。

### 对 Sampling-Policy Shift 的启发

SBRD 对 Sampling-Policy Shift 的纵向启发是：采样策略本身可以被视为一种 switching constraint regime。临床系统并不是被动记录患者，而是在资源、协议和风险判断下主动选择“何时测、测什么、何时治疗”；因此，不规则采样分类器也应分解为 shared state benchmark 与 policy-regime deviation。前者承载跨采样政策稳定的病程表征，后者解释为什么当前环境产生特定观测密度、变量可见性或治疗动作。

横向应用上，可以把 SBRD 的 regime decomposition 移植到 IMTS 分类：先用连续时间 latent dynamics 从不规则观测中构造 patient belief state，再用稀疏 policy-regime layer 捕捉医院/科室/时间段/采样密度诱导的偏差。分类头只允许依赖 shared state 与经校准的 regime uncertainty，而不是直接把 regime wedge 当作类别证据。评估时报告 cross-regime classification drop、policy-only predictability、regime-wedge stability 和 counterfactual-policy consistency，判断模型是否真正抵抗了 Sampling-Policy Shift。

## 追加更新 - 2026-09-11 23:02 UTC

### 本次检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：发现并读取 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`；同时读取兼容入口 `paper_daily.md` 的标题索引，并参考自动化记忆中 2026-08-03 至 2026-08-25 的新增标题。
- 本次黑名单论文标题已覆盖既有 50+ 篇历史总结，包括 `Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness` 与 `Shared-Benchmark Regime Decomposition for Nonstationary Clinical Decisions` 等全部已总结标题；新候选均已逐项排除黑名单重名。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / EHR event-stream foundation model / measurement-policy shift / clinical workflow distribution shift 的顶会、顶会 workshop、OpenReview、ICML virtual page 与 arXiv 论文页面。
- 已排除全部黑名单论文。严格的“顶会正会 + 直接 IMTS 分类”新增命中已基本被历史日报覆盖；本次保留 2 篇未在黑名单中的新跟踪对象：`ORA` 有 ICML 2026 virtual 记录，直接把不规则 EHR 事件建模为 marked time-to-event 预训练目标，并在多个临床分类任务上验证；`Learning Clinical Representations Under Systematic Distribution Shift` 是 2026-03 arXiv 预印本，尚未确认顶会录用，但它直接把 measurement policy、documentation practice 与 institutional workflow shift 作为表征解耦目标，对 Sampling-Policy Shift 的研究问题高度贴近。

## 33. One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models

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

## 34. Learning Clinical Representations Under Systematic Distribution Shift

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

## 追加更新 - 2026-09-13 23:01 UTC

### 本次检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`；同时用标题检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-09-12 的新增标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness`、`Shared-Benchmark Regime Decomposition for Nonstationary Clinical Decisions`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification` 与 `WIPSNet: Deep Learning for Paediatric Wheeze Detection from Impedance Pneumography`。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / ICU time-series domain adaptation / medical time-series language model 的顶会、顶会 workshop、OpenReview、ICML virtual 页面、arXiv 与论文项目页。
- 已排除全部黑名单论文；同时排除 MedFuse 这类历史日报已判断为 withdrawn/submission 风险的候选、Under-Cali/MoRGen 等偏 forecasting 或 generation 的候选、PRIME 这类期刊而非顶会候选，以及 HEARTS 这类 benchmark 价值高但与非规则采样分类联系较弱的候选。严格的“顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 2 篇全新工作：`INPUTADAPTER` 是 ICML 2026 Foundation Models for Structured Data workshop 工作，直接面向跨医院 ICU 时序分类器在目标域退化的问题；`OpenTSLM` 是 ICML 2026 主会 Poster，虽不以 irregular sampling 为标题主轴，但其面向多变量医疗时序、可变长度/频率信号、分类与自然语言推理的原生时序-语言融合，对非规则采样分类的可解释接口和采样策略偏移诊断有新增价值。

## 35. Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors

- 方法名：INPUTADAPTER
- 会议/状态：ICML 2026 Workshop on Foundation Models for Structured Data
- 作者：Omer Gotfrid, Dvir Aran, Barak Gahtan, Alex Bronstein
- 官方页：https://icml.cc/virtual/2026/71728
- Workshop accepted list：https://icml-structured-fm-workshop.github.io/accepted-papers/
- 关键词：ICU time-series classification, domain adaptation, frozen predictor, MIMIC-IV, eICU, HiRID, input-space transformation, cross-hospital shift

### 场景、任务与核心难点

INPUTADAPTER 面向真实医疗部署中很常见但经常被 IMTS benchmark 弱化的问题：一个已经验证、监管备案或嵌入硬件的软件医疗器械模型，在源医院训练后不能随意改权重，但目标医院的 ICU 时序数据分布已经变化。论文评估 MIMIC-IV 到 eICU、HiRID 的跨医院迁移，覆盖 5 个 ICU classification tasks，并同时在 AdaTime 时间序列适配基准上比较。这里的核心任务不是从零训练一个更强 encoder，而是在下游 predictor 完全冻结的情况下，让目标域输入被转换成源域模型能够理解的形式。

核心难点在于，跨医院退化往往不是单一 covariate shift：变量量纲、采样频率、时间窗口统计、缺失模式、告警后测量密度、检查流程和患者结构都可能同时变化。常规 domain adaptation 假设 encoder 可训练，alignment loss 可以反向传播到表征层；但在 SaMD 迁移、嵌入式固件或已验证临床模型场景中，预测器权重被锁定，目标域标签可能有限，且监管上更希望“适配数据”而不是“改模型”。INPUTADAPTER 因此学习一个 input-space adapter，把目标域时间窗口变换成 frozen source predictor 可消费的输入；可选地，它还在源 encoder latent space 中按时间步检索 k 个近邻源样本，并通过 cross-attention 融合上下文。论文报告其在 MIMIC-IV->eICU/HiRID ICU 任务上优于或追平多类 frozen-backbone、end-to-end DA 与 test-time adaptation baseline，其中 eICU AKI 任务 AUCPR 提升最大达到 +14.1 点。

### 审稿人视角：价值与不足

最有价值的思想是把临床时序迁移从“改模型参数”转成“学习一个可审计的输入域变换”。这在医疗 AI 中很现实：很多模型一旦验证和上线，重新训练主干的成本、审计负担和风险都很高。INPUTADAPTER 关注 frozen predictor 这个被低估但重要的部署约束，并通过输入变换 + 源域实例检索，把目标域数据尽量映射回已知模型的工作区间。对审稿人而言，它的贡献不只在指标提升，也在于提出了一种更贴合监管和硬件约束的 adaptation regime。

不足是，INPUTADAPTER 仍主要把跨医院差异作为可学习的输入分布变换来处理，尚未充分因果分解 patient state shift 与 observation policy shift。若目标医院的采样政策本身不同，例如某些化验只在更危重患者中下单、某些变量在 eICU/HiRID 中记录粒度不同，adapter 可能把目标域的政策痕迹“翻译”成源域中同样带标签相关性的痕迹，而不一定恢复真正稳定的病理状态。另一个风险是可解释性：输入空间变换若改变时间窗口统计、缺失模式或数值尺度，临床上需要知道它是在校正设备/流程差异，还是在抹掉目标医院中真实有意义的护理差异。

### 对 Sampling-Policy Shift 的启发

INPUTADAPTER 对 Sampling-Policy Shift 的横向启发非常直接：采样政策变化可以被看作 frozen classifier 前的输入空间错配。现实中我们可能无法重训已部署的 ICU 分类器，但可以学习一个 policy-aware adapter，把不同医院、设备或科室采样策略下的输入映射到分类器训练时的观测协议附近。这提示我们不仅要做 policy-invariant encoder，也可以研究 policy-normalizing input adapter，尤其适合 legacy model 或监管锁定模型的迁移。

纵向深化上，可以把 INPUTADAPTER 拆成 state adapter 与 policy adapter 两层。state adapter 只校正跨域稳定的数值尺度、时间对齐和变量表示，使 patient-state evidence 被 frozen predictor 正确读取；policy adapter 则显式建模目标域的采样密度、变量联测、value-pending 和告警后测量模式，仅用于校准、偏移告警或拒识。训练时可加入反事实采样增强：同一潜在病程经过不同 observation policy 后，adapter 输出给 frozen predictor 的 state-relevant logits 应保持稳定，而 policy head 应能区分采样制度。这样能把“适配输入”推进到“适配采样政策但不把政策 shortcut 注入分类边界”。

## 36. OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data

- 会议：ICML 2026 Poster
- 作者：Patrick Langer, Thomas Kaar, Max Rosenblattl, Maxwell A. Xu, Winnie Chow, Martin Maritsch, Robert Jakob, Ning Wang, Juncheng Liu, Aradhana Verma, Brian Han, Daniel Kim, Henry Chubb, Scott Ceresnak, Aydin Zahedivash, Alexander Sandhu, Fatima Rodriguez, Daniel McDuff, Elgar Fleisch, Oliver Aalami, Filipe Barata, Paul Schmiedmayer
- 官方页：https://icml.cc/virtual/2026/poster/65261
- 论文：https://arxiv.org/abs/2510.02410
- 代码：https://github.com/OpenTSLM/OpenTSLM
- 关键词：medical time-series language model, multivariate signals, text-time-series reasoning, sleep staging, human activity recognition, ECG question answering, variable length and frequency

### 场景、任务与核心难点

OpenTSLM 面向医疗与可穿戴场景中的多变量时序信号和文本联合推理。论文关注的不是 EHR 化验事件流，而是更接近原始传感器的 ECG、EEG/sleep、activity 等多变量信号；下游任务包括 Sleep-CoT 的睡眠阶段分类、HAR-CoT 的人体活动识别以及 ECG-QA-CoT 的心电问答/解释。它要解决的核心问题是：通用 LLM 很擅长处理文本、图像和音频，但通常不能原生消费长序列、多通道、不同长度和不同频率的医学时间序列；把信号转成文本或图片会丢失精细时序结构，也会带来上下文长度和格式偏差。

论文提出两个 TSLM 架构。OpenTSLM-SoftPrompt 将时序 encoder 产生的可学习 token 接入 LLM token 序列，参数效率高，适合较短序列；OpenTSLM-Flamingo 则用 Perceiver-style resampler 和 gated cross-attention，把时序作为独立模态显式接入 LLM，因而在多变量、长序列输入上内存扩展更稳定。实验显示 OpenTSLM 在 sleep staging 和 HAR 上明显优于文本化/图像化 baselines，sleep staging F1 达到约 69.88%，HAR F1 约 65.44%-67.64%，并在 ECG-QA 中获得心脏科专家对解释质量的正面评估。

### 审稿人视角：价值与不足

最有价值的思想是坚持“时间序列应该作为原生模态进入语言模型”，而不是被粗暴序列化为文本或转成图像。对医疗时序而言，波形形态、局部节律、多通道同步关系和长程上下文往往同时决定分类标签；OpenTSLM-Flamingo 的 cross-attention 路线把 LLM 的语言推理能力和专门的时序 encoder 解耦，给出了比 prompt-only 更可扩展的接口。它还通过 CoT 数据集把分类、解释和问答放在同一框架中，有助于把“模型预测了什么”推进到“模型如何解释信号证据”。

不足在于，它不是专门针对 irregular EHR/IMTS 的方法，主要实验数据更接近连续传感器或规则窗口化信号；论文虽然强调多条时间序列可具有不同长度和频率，但对 missingness、异步变量观测、医生下单式采样政策和跨医院 observation process 的处理并不充分。另一个问题是，CoT 解释可能提升可读性，却不自动保证因果正确：如果输入格式、窗口选择或采样频率本身与类别相关，LLM 可能生成看似合理但实际依赖采样 shortcut 的解释。

### 对 Sampling-Policy Shift 的启发

OpenTSLM 对 Sampling-Policy Shift 的横向启发是：采样策略偏移最终也会进入“解释接口”。当我们把不规则采样信号交给 TSLM 或 LLM 辅助诊断时，模型不仅输出类别，还会生成自然语言理由；如果采样密度、变量可见性或传感器频率发生变化，解释文本可能把 policy artifact 包装成临床证据。因此，解决 Sampling-Policy Shift 不能只看 AUROC/AUPRC，还要审计 explanation 是否把“被测了什么、何时被测、采样多密”当作疾病证据。

纵向深化上，可以把 OpenTSLM 改造成 policy-audited TSLM：前端时序 encoder 分出 state tokens 和 policy tokens，前者进入分类与医学解释，后者进入数据质量、采样制度和不确定性说明。LLM 生成时被要求分别输出 state rationale 与 policy rationale，并在反事实采样视图下保持 state rationale 和诊断 logits 稳定，同时允许 policy rationale 描述不同采样协议。这样能把 OpenTSLM 的“原生时序-语言融合”推进到“可解释地隔离采样政策偏移”的临床时序分类助手。

## 追加更新 - 2026-09-14 23:01 UTC

### 本次检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`；同时用标题检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-09-13 的新增标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`Fault Diagnosis of Irregular Sequences by Adjoint Learning in Continuous-Time Model Space`、`Beyond Observations: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning`、`Can we generate portable representations for clinical time series data using LLMs?`、`QuITE: Query-Based Irregular Time Series Embedding`、`Generative Modeling of Irregular Time Series via SDE-Induced Continuous-Discrete Variational Inference`、`MTM: A Multi-Scale Token Mixing Transformer for Irregular Multivariate Time Series Classification`、`MedMamba: Multi-View State Space Models with Adaptive Graph Learning for Medical Time Series Classification`、`MedSpaformer: A Transferable Transformer with Multi-Granularity Token Sparsification for Medical Time Series Classification`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`StarEmbed: Benchmarking Time Series Foundation Models on Astronomical Observations of Variable Stars`、`Rethinking Large Language Models for Irregular Time Series Classification in Critical Care`、`Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing`、`Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction`、`Status-Aware Self-Supervised Forecasting for Irregular Clinical Time Series`、`LLM4EHR: Aligning Clinical Time Series with Medical Event Sequences via Large Language Models`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Context-Aware Neural SDEs for Robust Irregular Time-Series Classification`、`Context-Informed Sequence Classification: A Multimodal Approach to Vehicle Diagnostics`、`Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness`、`RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification`、`NeurOCNN: A Neural-Operator-Based Model for Physiological Time Series`、`Cached Foundation Model Summaries for Memory-Efficient Clinical Time Series Inference`、`DeNOTS: Stable Deep Neural ODEs for Time Series`、`Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time`、`PathwayLLM: Explainable Clinical Trajectory Modeling with Structured Pathways for Sepsis Prediction`、`ArcTimeSDE: Aligning Compute with Information Via ARC Length Time in Neural SDEs`、`MoRGen: Mixture-of-Resolutions Generative Forecasting for Irregularly Sampled Medical Time-Series Data`、`STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data`、`Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment`、`Repurposing Foundation Model for Generalizable Medical Time Series Classification`、`Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series`、`Investigating a Model-Agnostic and Imputation-Free Approach for Irregularly-Sampled Multivariate Time-Series Modeling`、`TreeText-CTS: Compact, Source-Traceable Tree-Path Evidence for Irregular Clinical Time-Series Prediction`、`Random Controlled Differential Equations`、`An Automated Data Engineering Pipeline for Time Series Classification Via Text Embeddings`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`JETS: A Self-Supervised Joint Embedding Time Series Foundation Model for Behavioral Data in Healthcare`、`TRIAGE: Dialectical Reasoning for Explainable Risk Prediction on Irregularly Sampled Medical Time Series with LLMs`、`MATA-Former & SIICU: Semantic Aware Temporal Alignment for High-Fidelity ICU Risk Prediction`、`Cross-Representation Benchmarking in Time-Series Electronic Health Records for Clinical Outcome Prediction`、`Informative Irregularity as a Diagnostic for Model Robustness`、`Shared-Benchmark Regime Decomposition for Nonstationary Clinical Decisions`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification`、`WIPSNet: Deep Learning for Paediatric Wheeze Detection from Impedance Pneumography`、`Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors` 与 `OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data`。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time series classification / functional time-series classification / clinical time-series sampling-policy shift 的顶会、顶会 workshop、OpenReview、ICLR/ICML virtual 页面、AAAI Proceedings、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 LITT 这类偏 survival/timing prediction 且暂无顶会录用的候选、Under-Cali/GRUwE/APN 等偏 forecasting 或 next-event prediction 的候选、T1 等偏 imputation 的候选，以及 TiViT/TsLLM/Mantis 等更偏通用规则时序或不直接建模不规则采样的候选。本次保留全新工作 1 篇：`DeepFRC` 是 ICLR 2026 Poster，明确以 categorical classification 为任务，并在形式化设定中处理 irregular and potentially misaligned sampling points；虽然它来自 functional data analysis 而非 ICU IMTS 主流 benchmark，但其“可微时间重参数化 + 分类判别目标联合训练”对 Sampling-Policy Shift 下的时间对齐和相位偏移建模有直接启发。

## 37. DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification

- 会议：ICLR 2026 Poster
- 作者：Siyuan Jiang, Yihan Hu, Wenjie Li, Pengcheng Zeng
- 官方页：https://iclr.cc/virtual/2026/poster/10011414
- OpenReview：https://openreview.net/forum?id=5vdw8Qmrre
- 论文：https://arxiv.org/abs/2501.18116
- 代码：https://github.com/Drivergo-93589/DeepFRC
- 关键词：functional time-series classification, irregular sampling, temporal misalignment, diffeomorphic warping, Fourier spectral representation, class-aware contrastive learning

### 场景、任务与核心难点

DeepFRC 面向 functional data / trajectory classification：每个样本是一条底层连续轨迹在若干时间点上的观测，并带有离散类别标签。论文形式化地把输入写成 `{(x_i(t_i), y_i)}`，其中 `t_i` 可以是不规则且跨样本错位的采样点；典型应用包括生物医学曲线、运动轨迹和传感器行为序列。这里的核心任务不是预测未来值，而是在时间相位不一致、观测点不对齐、局部形态被拉伸或压缩的情况下，仍然识别轨迹所属类别。

核心难点是 phase variability：同一类别的关键形态可能在不同个体上提前、滞后或以不同速度发生。如果先做独立 registration，再训练分类器，alignment 目标可能并不服务于类别判别；如果直接把错位曲线交给分类器，模型又可能把相位差当成类别差异。DeepFRC 因此把 registration 与 classification 合并到一个端到端模型中：用 neural deformation operator 学习可微同胚时间扭曲函数 `gamma: t -> t_tilde`，将轨迹对齐到更有判别性的相对时间轴；再用 Fourier basis 得到平滑谱表示，并用 class-aware contrastive loss 同时拉近同类对齐后的曲线、拉开异类曲线。论文还给出 registration approximation 与 generalization bound，把对齐误差和分类泛化联系起来。

### 审稿人视角：价值与不足

最有价值的思想是把“时间对齐是否正确”直接纳入分类目标，而不是把它当作分类前的独立预处理。对很多不规则采样场景而言，问题不只是缺哪些点，而是同一语义事件在不同样本的物理时间轴上并不齐；DeepFRC 的 diffeomorphic warping 提供了一种可解释、可约束的相对时间重参数化。Fourier spectral representation 则避免对齐后曲线过度依赖单个观测点，class-aware contrastive loss 让 alignment 更关注类别结构而非纯几何相似。作为审稿人，我认为它的贡献在于把 FDA 中成熟的 registration 思想与深度分类目标结合，并且尝试用理论结果说明为什么更好的 registration 会改善分类。

不足在于，它仍主要处理单条或低维功能曲线的时间相位错位，并不等同于 ICU/EHR 中多变量异步、MNAR missingness 和医生下单式采样政策。可微 warping 可能把真实的发病时间差、护理流程差或设备采样差统一“对齐”掉；若这些时间差本身具有临床意义，过强的 registration 会损失判别信号。另一方面，如果训练环境中特定类别总是被更早、更密集或更规律地采样，class-aware alignment 也可能把采样政策学成类别对齐模板。论文报告了对噪声、缺失和数据规模变化的鲁棒性，但还没有系统评估跨医院、跨设备 cadence、跨主动测量策略时 learned warping 是否稳定。

### 对 Sampling-Policy Shift 的启发

DeepFRC 对 Sampling-Policy Shift 的横向启发是：采样政策偏移有时表现为“时间轴语义”改变，而不只是 mask ratio 或 delta-t 分布改变。例如不同医院可能在症状出现后多久下单化验，不同设备可能在事件前后以不同 cadence 记录；同一病理事件在观测时间轴上的位置和密度都会漂移。DeepFRC 提醒我们，可以把物理时间 `t` 与相对病程时间 `tau` 分开建模：分类主路径应尽量依赖跨政策稳定的病程相位，而 policy path 则解释为什么某些事件在物理时间上提前、滞后或被密集观测。

纵向深化上，可以把 DeepFRC 改造成 policy-aware registration classifier：state warping 学习跨采样政策稳定的 latent disease clock，policy warping 学习医院流程、设备 cadence、告警后密集采样和随访间隔造成的观测时间扭曲。训练时对同一潜在轨迹施加多种反事实采样策略，约束 state-warped representation 与分类 logits 保持一致，同时允许 policy-warping 参数、观测密度和不确定性随策略改变。评估上可报告 cross-policy warp distance、state-clock consistency、policy-only classification leakage，以及在固定病程但替换观测策略时 logits 是否稳定。这样能把 DeepFRC 的“分类驱动对齐”推进到“区分病程相位与采样政策相位”的非规则时序分类框架。

## 追加更新 - 2026-09-15 23:01 UTC

### 本次检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`；同时读取兼容入口 `paper_daily.md` 的标题索引，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors`、`OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data` 与 `DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification` 等全部已总结标题。
- 检索范围：围绕近 3-7 个月内 irregular sampled / asynchronous / sparse clinical time series classification / neural differential equations for irregular observations / critical-care time-series foundation model 的顶会、顶会 workshop、AAAI Proceedings、NeurIPS/OpenReview、arXiv 与论文页面。
- 已排除全部黑名单论文；同时排除 `TESS` 这类 ICLR 2023 时间窗口过旧候选、`UniShape` 这类分类强但不原生处理 irregular sampling 的工作、`SELDON` 这类主要任务偏连续时间预测/插值而非分类的工作，以及 `RAINCOAT` 这类时间序列 domain adaptation 但非近月 IMTS 分类主线的旧工作。本次保留 2 篇全新工作：`Continuum Dropout` 是 AAAI 2026 正会论文，虽是通用 NDE 正则化方法，但实验直接覆盖不规则、缺失、类别不平衡的时序分类；`Towards Self-Supervised Foundation Models for Critical Care Time Series` 是 NeurIPS 2025 TS4H workshop 工作，直接面向稀疏、不规则 ICU 时序的自监督预训练和跨数据集死亡风险分类迁移。

## 38. Continuum Dropout for Neural Differential Equations

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

## 39. Towards Self-Supervised Foundation Models for Critical Care Time Series

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

## 追加更新 - 2026-09-16 23:01 UTC

### 本次检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`；同时读取兼容入口 `paper_daily.md` 的末尾索引，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Context-Aware Neural SDEs for Robust Irregular Time-Series Classification`、`RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification`、`DeNOTS: Stable Deep Neural ODEs for Time Series`、`Contimask: Explaining Irregular Time Series via Perturbations in Continuous Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification`、`WIPSNet: Deep Learning for Paediatric Wheeze Detection from Impedance Pneumography`、`Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors`、`OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data`、`DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification`、`Continuum Dropout for Neural Differential Equations` 与 `Towards Self-Supervised Foundation Models for Critical Care Time Series` 等全部已总结标题。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time-series classification / clinical irregular time series / informative sampling / continuous positional encoding 的顶会、顶会 workshop、NeurIPS Proceedings、OpenReview、ICML/ICLR/AAAI virtual 页面、arXiv 与论文项目页。
- 已排除全部黑名单论文；同时排除 `GRUwE / Still Competitive` 这类主要任务是 next-observation 与 next-event prediction、`Oscillators Are All You Need` 这类尚未确认顶会录用且偏通用建模的预印本、`A Statistical Approach for Modeling Irregular Multivariate Time Series with Missing Observations` 这类期刊/非顶会来源、以及已由历史记录覆盖的 `QuITE`、`Random Controlled Differential Equations`、`DeNOTS`、`STAR-Set`、`EHR-SPC`、`MILM`、`RAxSS` 等候选。本次保留 2 篇全新工作：`Rotary Masked Autoencoders are Versatile Learners` 是 NeurIPS 2025 正会论文，直接包含 irregular multivariate time-series classification；`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series` 是 NeurIPS 2025 TS4H Poster，直接面向不规则采样临床时序的院内死亡二分类。

## 40. Rotary Masked Autoencoders are Versatile Learners

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

## 41. No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series

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

## 追加更新 - 2026-09-17 23:00 UTC

### 本次检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`；同时分段读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`Context-Aware Neural SDEs for Robust Irregular Time-Series Classification`、`RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series` 等全部已总结标题。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / heterogeneous sensor time series / physiological time-series classification / wearable ECG sampling-rate robustness / sampling policy shift 的顶会、顶会 workshop、OpenReview、ICML virtual、NeurIPS virtual、arXiv 与代码页。
- 已排除全部黑名单论文；同时继续排除 `MedFuse` 这类 ICLR 2026 withdrawn submission、`Mantis` / `Time-CoT` 这类强分类但未直接处理不规则采样的通用 MTS 工作、`CoCLD` / `MIRA` / `RAF` 等偏 next-event、generation 或 forecasting 的候选。本次保留 2 篇未在黑名单中的新跟踪对象：`Giving Sensors a Voice` 是 ICML 2026 正会论文，虽不是传统 ICU IMTS benchmark，但其 channel-aware JEPA、异构传感器语义和下游分类对跨采样/跨传感器策略迁移有直接启发；`A novel approach to classification of ECG arrhythmia types with latent ODEs` 是 NeurIPS 2025 TS4H workshop poster，直接讨论 wearable ECG 在低频/潜在不规则采样下的分类鲁棒性。

## 42. Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings

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

## 43. A novel approach to classification of ECG arrhythmia types with latent ODEs

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

## 追加更新 - 2026-09-18 23:00 UTC

### 本次检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`；同时用标题检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中的补充标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks`、`SuperMAN: Interpretable and Expressive Networks over Temporally Sparse Heterogeneous Data`、`GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care`、`DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`Informative Irregularity as a Diagnostic for Model Robustness`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Learning Clinical Representations Under Systematic Distribution Shift`、`Adapt the Data, Not the Model: Input-Space Adaptation for Frozen Time-Series Predictors`、`OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data`、`DeepFRC: An End-to-End Deep Learning Model for Functional Registration and Classification`、`Continuum Dropout for Neural Differential Equations`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`Giving Sensors a Voice: Multimodal JEPA for Semantic Time-Series Embeddings`、`A novel approach to classification of ECG arrhythmia types with latent ODEs`，以及自动化记忆中记录的 `LERD`、`WIPSNet`、`Context-Aware Neural SDEs`、`RAxSS` 等已总结工作。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular multivariate time-series classification / clinical irregular time series / informative sampling / sampling-policy shift 的顶会、顶会 workshop、OpenReview、ICLR/ICML/AAAI/NeurIPS 官方页面、arXiv 与代码页。
- 已排除全部黑名单论文；同时排除 `MedFuse` 这类 ICLR 2026 withdrawn submission，`SITS` 这类 ICML 2025 workshop 且窗口偏旧条目，`ReIMTS`、`APN`、`ASTGI` 这类近期顶会但主任务为 forecasting 的条目，`Time-IMM` 这类 NeurIPS D&B 价值很高但当前 benchmark 主任务为 forecasting 的条目，以及 `PhASER` 这类分类与 domain generalization 强但不直接处理 irregular sampling 的工作。严格的“近月顶会正会 + 直接 IMTS 分类”新增命中已基本被历史记录覆盖；本次保留 1 篇未在黑名单中的直接相关工作：`TimeCHEAT` 是 AAAI 顶会论文，覆盖 irregularly sampled multivariate time series 的 classification / forecasting / interpolation 三类任务，且对局部通道依赖与全局通道独立性的拆分可为 Sampling-Policy Shift 下的变量联测策略建模提供新视角。

## 44. TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis

- 方法名：TimeCHEAT / Channel Harmony ISMTS Transformer
- 会议：AAAI 2025 Technical Track，Proceedings of the AAAI Conference on Artificial Intelligence 39(18):18861-18869
- 作者：Jiexi Liu, Meng Cao, Songcan Chen
- AAAI 页面：https://ojs.aaai.org/index.php/AAAI/article/view/34076
- 论文：https://arxiv.org/html/2412.12886
- DOI：https://doi.org/10.1609/aaai.v39i18.34076
- 关键词：irregularly sampled multivariate time series, channel-dependent / channel-independent strategy, bipartite graph embedding, classification, interpolation, forecasting

### 场景、任务与核心难点

TimeCHEAT 面向 irregularly sampled multivariate time series (ISMTS) 的通用分析任务，覆盖医疗、气象、交通和人类活动等场景。最贴近分类的实验包括 P19、P12 和 PAM：P19 有 38,803 名患者、34 个传感器、约 94.9% missing ratio；P12 是 ICU 前 48 小时 36 个传感器的二分类预测，missing ratio 约 88.4%；PAM 则是 17 个传感器、8 类活动识别，missing ratio 约 60.0%。这些数据的共同难点是：同一变量内部采样间隔不均，不同变量之间异步观测，且稀疏通道很难仅靠自身历史形成稳定表示。

论文抓住的核心问题不是再设计一个单纯插值器，而是重新审视多变量时序中的 channel strategy。现有方法常在 channel-independent (CI) 与 channel-dependent (CD) 之间二选一：CI 能保留每个通道自己的采样节奏，但在稀疏通道上信息不足；CD 能利用跨通道信息，却容易过平滑并抹掉通道个性。TimeCHEAT 的策略是局部 CD、全局 CI：先把 ISMTS 切成 sub-series patches，在 patch 内用通道-时间二部图学习 irregular-to-regular 的边权和固定长度 embedding，聚合邻近观测和局部跨通道信息；再在 patch 级别用 CI Transformer 保留每个通道的长期个性化注意力。论文报告 TimeCHEAT 在 P19 上超过主要基线，在 P12 上接近最优且计算/空间成本低于 ViTST，在 PAM 八分类上比既有方法有约 0.7% accuracy 和 0.9% precision 提升。

### 审稿人视角：价值与不足

最有价值的技术思想是把“跨通道依赖应该在什么尺度使用”显式化。很多 IMTS 分类方法要么把所有变量混成一个大 token 空间，要么完全按变量独立建模，再用后端 attention 或 RNN 聚合；TimeCHEAT 指出这两种极端都不适合稀疏异步数据。局部 CD 让稀疏通道在短窗口内借用相关变量的可见信息，全局 CI 则避免模型把所有通道压成同一种动态模式。二部图式 time embedding 也有价值：它把 reference time embedding 的学习转化为边权预测，少依赖“时间越近越相关”这类固定假设，因此比简单 decay 或固定插值更灵活。

不足在于，TimeCHEAT 仍然把观测模式作为提升表示质量的结构资源，而没有进一步区分哪些局部跨通道关系来自稳定生理/物理机制，哪些来自环境特定采样政策。P12/P19/PAM 的随机 train/valid/test split 主要评估同分布泛化；如果换医院、换传感器调度、换变量联测协议，局部 CD 二部图中学到的强边可能从“生理耦合”变成“某机构常一起测”。此外，patch 切分和 reference points 的选择仍会影响哪些时间尺度被局部 CD 吸收；当关键事件在某些采样政策下被压缩、延迟或缺失时，局部聚合可能放大 policy shortcut。

### 对 Sampling-Policy Shift 的启发

TimeCHEAT 对 Sampling-Policy Shift 的横向启发是：采样政策偏移不只改变 delta-t 和 mask ratio，也会改变“局部跨通道信息能否互相借用”。医院 A 可能在怀疑感染时同时测 lactate、WBC 和体温，医院 B 可能只在高风险患者上补测 lactate；在这种情况下，普通 CD 模型很容易把联测关系当成类别证据，普通 CI 模型又可能丢掉真正有用的变量互补。TimeCHEAT 的局部 CD / 全局 CI 分层提供了一个很自然的审计位置：局部二部图边权、patch 内通道聚合强度、以及全局每通道 attention 是否随环境改变，都可以作为 sampling-policy shift 的诊断指标。

纵向深化上，可以把 TimeCHEAT 改造成 state-policy channel harmony。局部图分成两套边：state edges 捕捉跨采样政策稳定的生理/物理耦合，policy edges 捕捉医院流程、设备调度、变量联测和告警后密集观测造成的共现关系；分类主路径只使用 state edges 与全局 CI 表示，policy edges 进入偏移告警、校准或拒识头。训练时对同一潜在轨迹施加多种反事实采样策略，约束 state-edge distribution、channel-wise representations 和分类 logits 保持稳定，同时允许 policy-edge distribution 预测采样策略 ID、变量联测模式和观测密度。这样可以把 TimeCHEAT 的“通道策略和谐”推进到“状态通道关系与采样政策通道关系解耦”的非规则采样分类框架。

## 追加更新 - 2026-09-21 23:02 UTC

### 本次检索与去重记录

- 已强制读取根目录下全部已发现的 `paper_daily_*.md`：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`、`paper_daily_2026-08-22.md`、`paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`、`paper_daily_2026-08-25.md`、`paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`；同时用标题检索读取兼容入口 `paper_daily.md`，并参考自动化记忆中 2026-08-03 至 2026-09-20 的新增标题。
- 本次黑名单覆盖历史已总结标题，包括但不限于：`Adaptive Time Encoding for Irregular Multivariate Time-Series Classification`、`FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification`、`One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification`、`Beyond Observations: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning`、`QuITE: Query-Based Irregular Time Series Embedding`、`MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling`、`Structure-Aware Set Transformers: Temporal and Variable-Type Attention Biases for Asynchronous Clinical Time Series`、`VP-GNN: A Unified Graph Framework for Variable-Wise and Patch-Wise Modeling of Irregular Clinical Time Series`、`PULSE: Benchmarking Large Language Models for ICU Time Series Classification`、`Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time`、`CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data`、`One Loss to Rule Them All: Marked Time-to-Event for Structured EHR Foundation Models`、`Rotary Masked Autoencoders are Versatile Learners`、`No Imputation Needed: A Switch Approach to Irregularly Sampled Time Series`、`TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis`，以及自动化记忆中记录的 `Auditing Black-Box Trends: Structural Inductive Bias Facilitates Causal Interpretability in Clinical Time Series`、`Pretraining EHR Foundation Models with Patient-Aware Sampling` 等近期已总结工作。
- 检索范围：近 3-7 个月内围绕 irregular sampled / asynchronous / irregular clinical time series classification / ICU event-stream prediction / sampling-policy shift 的顶会、顶会 workshop、OpenReview、ICLR/ICML/AAAI/NeurIPS/KDD 官方页面、arXiv 与代码页。
- 已排除全部黑名单论文。严格的“近月顶会正会 + 直接 IMTS 分类”新增命中在当前检索中基本已被历史日报覆盖；本次仅保留 1 篇未在黑名单中的高相关新工作：`INTERVenE` 是 arXiv 2026-08 预印本，尚未确认顶会录用，但它直接面向稀疏、不规则 ICU EHR 事件流上的短期并发症预测/风险分类，并把原始点观测转为可解释的知识型时间抽象区间，对 Sampling-Policy Shift 下的“观测流程语义化”有新增价值。

## 45. INTERVenE: Temporal-Abstraction-Interval Based Transformers for Short-Horizon Medical Event Prediction

- 会议/状态：arXiv 2026-08 预印本，尚未确认顶会录用
- 作者：Shahar Oded, Yuval Shahar
- 论文：https://arxiv.org/abs/2608.29901
- 关键词：irregular ICU EHR, temporal abstraction intervals, knowledge-based temporal abstraction, short-horizon complication prediction, interpretable clinical event prediction

### 场景、任务与核心难点

INTERVenE 面向 ICU 中稀疏、不规则、异构 EHR 事件流的短期临床并发症预测。论文聚焦 MIMIC-IV 中 57,078 次 diabetes-related ICU admissions：模型观察入院前 48 小时的实验室测量、用药、干预和静态上下文，预测 48-336 小时内是否发生 death、kidney complication、hyperglycemia、severe hyperglycemia、hypoglycemia、severe hypoglycemia 等 6 个目标事件，并同时处理 length-of-stay 相关目标。这里的任务形式不是传统单一二分类 benchmark，而是面向多个未来并发症的短期风险分类/时间到事件预测。

核心难点在于，原始 ICU EHR 不是规则时间网格：测量、用药和干预以病人特异的时间到达，既有点事件，也有持续状态和趋势；同一个数值在不同持续时间、上下文和临床阶段下语义不同。直接把原始 triplet 输入 GRU-D、STraTS 或普通 Transformer，虽然能处理时间戳和缺失，但解释通常仍停留在变量或 token 层，很难回答“哪个临床状态、趋势或上下文区间导致风险升高”。INTERVenE 因此先用 Knowledge-Based Temporal Abstraction (KBTA) 把原始点观测转换为有医学命名的区间 token，例如状态、趋势、事件和上下文，再构建两类 Transformer：INTERVenE-Enc 做 single-pass joint risk / time-to-event prediction；INTERVenE-Ar 以自回归方式生成未来 abstraction trajectory，并在每一步读出风险曲线。实验中 INTERVenE-Enc 在 held-out admissions 上达到 support-weighted AUPRC 0.672、AUROC 0.901，相比最强神经基线 AUPRC 提升约 0.041。

### 审稿人视角：价值与不足

最有价值的思想是把“不规则临床时序如何被解释”前移到表示层：先将原始观测转成临床知识可命名的 interval vocabulary，再让 Transformer 在这些区间 token 上学习风险。这样做的贡献不只是提升 AUPRC，而是让归因天然落到 named clinical concepts 上，避免事后解释在 raw value / bin index 层绕一圈。INTERVenE-Ar 的自回归风险曲线也很有意义，因为它不仅输出 admission-level 风险，还尝试说明风险在未来 abstraction trajectory 中何时、随哪些事件上升。

不足也比较明显。首先，它目前是 arXiv 预印本，尚不能按已录用顶会论文看待。其次，KBTA 的优势依赖人工医学知识库和 abstraction rule，迁移到其他病种、其他医院或更弱结构化的 EHR 时成本不低。更关键的是，KBTA interval 本身可能把医院流程和采样政策语义固化进 token：某些 lab state、trend 或 intervention interval 的出现，既可能反映病程，也可能反映下单习惯、监测频率、护理协议或资源约束。论文证明了 interval abstraction 是可解释 substrate，但还没有系统评估跨医院、跨测量政策或反事实采样策略下，这些 named intervals 是否仍代表稳定病理机制。

### 对 Sampling-Policy Shift 的启发

INTERVenE 对 Sampling-Policy Shift 的横向启发是：采样政策偏移不一定只表现为 mask、delta-t 或变量共现图变化，也可能表现为“被抽象出的临床区间词汇”发生偏移。若一个医院更频繁监测血糖，KBTA 可能生成更多 glucose state / trend intervals；若另一个医院只在危重时密集测量，同样的 interval token 就可能携带不同风险语义。因此，interval abstraction 可以作为可解释接口，但必须审计哪些区间是 patient-state interval，哪些区间其实是 observation-policy interval。

纵向深化上，可以把 INTERVenE 改造成 policy-aware temporal-abstraction Transformer：state abstraction branch 只保留跨采样策略稳定的病程状态、趋势和临床上下文，进入风险分类主路径；policy abstraction branch 则解释哪些 interval 是由测量频率、检查触发、护理记录习惯或 value-pending 机制产生，用于校准和偏移告警。训练时可对同一潜在病程生成不同采样策略下的 KBTA interval streams，约束 state intervals、risk logits 和 state rationale 保持稳定，同时允许 policy intervals 和不确定性随观测政策改变。这样能把 INTERVenE 的“可解释区间 token”推进到“区分病程语义与采样政策语义”的非规则时序分类框架。
