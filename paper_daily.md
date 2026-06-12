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
