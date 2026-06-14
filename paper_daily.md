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
