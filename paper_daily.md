# Daily Paper Review - 2026-06-11

检索主题：异步/不规则采样时序分类（irregular / irregularly-sampled time series classification）  
检索范围：arXiv、AAAI Proceedings，并通过 Web Search/Google Scholar 关键词检索交叉确认（AAAI、NeurIPS、ICML、ICLR、KDD；近三至七个月内优先）。  
今日精选：2 篇，均为 AAAI 2026 已发表论文，发布时间为 2026-03-14，任务与“不规则采样时序分类”直接相关。

---

## 1. FlowPath: Learning Data-Driven Manifolds with Invertible Flows for Robust Irregularly-sampled Time Series Classification

- 会议：AAAI 2026
- 作者：YongKyung Oh, Dong-Young Lim, Sungil Kim
- 来源：
  - AAAI Proceedings: https://ojs.aaai.org/index.php/AAAI/article/view/39643
  - arXiv: https://arxiv.org/abs/2511.10841
- DOI: https://doi.org/10.1609/aaai.v40i29.39643
- 代码： https://github.com/yongkyung-oh/FlowPath

### 原文摘要

Modeling continuous-time dynamics from sparse and irregularly-sampled time series remains a fundamental challenge. Neural controlled differential equations provide a principled framework for such tasks, yet their performance is highly sensitive to the choice of control path constructed from discrete observations. Existing methods commonly employ fixed interpolation schemes, which impose simplistic geometric assumptions that often misrepresent the underlying data manifold, particularly under high missingness. We propose FlowPath, a novel approach that learns the geometry of the control path via an invertible neural flow. Rather than merely connecting observations, FlowPath constructs a continuous and data-adaptive manifold, guided by invertibility constraints that enforce information-preserving and well-behaved transformations. This inductive bias distinguishes FlowPath from prior unconstrained learnable path models. Empirical evaluations on 18 benchmark datasets and a real-world case study demonstrate that FlowPath consistently achieves statistically significant improvements in classification accuracy over baselines using fixed interpolants or non-invertible architectures. These results highlight the importance of modeling not only the dynamics along the path but also the geometry of the path itself, offering a robust and generalizable solution for learning from irregular time series.

### 摘要中文翻译

从稀疏且不规则采样的时间序列中建模连续时间动态，仍然是一个基础性挑战。神经受控微分方程为这类任务提供了原则性的框架，但其性能高度依赖于如何从离散观测中构造控制路径。现有方法通常采用固定插值方案，这会引入过于简单的几何假设，尤其在高缺失率场景下容易误表示真实数据流形。本文提出 FlowPath，一种通过可逆神经流学习控制路径几何结构的新方法。FlowPath 不只是简单连接观测点，而是在可逆性约束引导下构造连续且数据自适应的流形，使变换保持信息并具有良好行为。这种归纳偏置使 FlowPath 区别于以往无约束的可学习路径模型。作者在 18 个基准数据集和一个真实案例研究上进行实验，结果表明 FlowPath 相比固定插值或不可逆结构的基线，在分类准确率上取得了统计显著提升。这些结果强调：对于不规则时间序列学习，不仅要建模路径上的动态，也要建模路径自身的几何结构，从而获得稳健且泛化性强的解决方案。

### 专家讲解

这篇论文面向医疗、传感器、人类活动识别等“观测时间不均匀、变量存在缺失、采样间隔不固定”的序列分类任务。它选择的技术路线是连续时间建模，核心骨架是 Neural CDE；但论文没有把重点放在更复杂的向量场或分类头上，而是重新审视 Neural CDE 中最关键、也最容易被默认处理的步骤：如何把离散观测点构造成连续控制路径。

传统 Neural CDE 通常使用线性插值、三次样条等固定插值方式。问题在于，这些插值隐含了非常强的几何假设：观测点之间的真实轨迹被认为可以由某种固定规则连接。当缺失严重或采样极不均匀时，这个假设会扭曲潜在流形，导致模型沿着“错误路径”积分。FlowPath 的突破点是把控制路径本身变成可学习对象，并用可逆神经流约束它。也就是说，它不是任意用 MLP 生成一条路径，而是要求路径变换近似微分同胚，尽量保持概率质量和类别可分结构，避免不可逆映射带来的折叠、撕裂和不稳定。

因此，这篇论文实现的能力可以概括为：在高缺失、不规则观测下，学习一个更加符合数据几何的连续时间控制路径，让下游 Neural CDE 在更可靠的路径上演化，从而提升分类准确率和鲁棒性。

### 审稿人视角：价值、不足与启发

我认为最有价值的思想是：把“不规则时序分类”的瓶颈从“如何设计更强的时序编码器”转向“输入路径几何是否可信”。这是一种很好的问题重构。很多工作把插值当作预处理或工程细节，但 FlowPath 证明控制路径本身就是模型假设的一部分，且会直接影响连续时间模型的可识别性和稳健性。可逆流的引入也不是简单堆模块，而是给可学习路径补上了结构约束，使灵活性和稳定性之间取得更好的平衡。

可能的不足是：可逆流和 Neural CDE 的组合会带来额外训练与推理成本，对长序列、高维多变量序列以及实时部署场景可能不够轻量；此外，论文主要证明路径几何约束有效，但不同领域中“真实路径几何”是否一定能由这类流模型充分表达，仍需要更多真实业务数据验证。另一个值得继续研究的问题是，FlowPath 主要提升输入路径构造，但对类别不平衡、观测机制偏置、标签噪声等现实因素处理较少。

对我们工作的横向启发是：如果我们面对的是医疗监测、工业传感、交易行为、用户事件流等异步采样数据，不应只在缺失值补全或时间编码上做文章，还可以把“观测点之间的连接方式”作为可学习模块。纵向深化上，可以将 FlowPath 的路径几何思想与领域先验结合，例如把物理约束、医学检查流程、设备采样机制或事件触发机制融入可逆路径学习；也可以进一步研究更轻量的可逆路径模块，用于边缘设备或在线分类。

---

## 2. Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification

- 会议：AAAI 2026
- 作者：Xin Qin, Mengna Liu, Wenjie Wang, Shuxin Li, Tianjiao Li, Xiufeng Liu, Xu Cheng
- 来源：
  - AAAI Proceedings: https://ojs.aaai.org/index.php/AAAI/article/view/39682
  - AAAI PDF: https://ojs.aaai.org/index.php/AAAI/article/view/39682/43643
- DOI: https://doi.org/10.1609/aaai.v40i29.39682
- 注：本次检索未发现该论文的 arXiv 版本，采用 AAAI 正式页面与 PDF 摘要。

### 原文摘要

Irregular time series (IRTS) are prevalent in real-world applications, where uneven sampling and missing data pose fundamental challenges to deep learning-based feature modeling. Although existing methods attempt to retain timestamp information, they often overlook the structured patterns embedded within the missingness itself, and tend to perform poorly when confronted with class imbalance exacerbated by data incompleteness. Specifically, temporal irregularity hinders the modeling of long-range dependencies and local patterns, while sparse observations limit representational capacity, disproportionately impairing minority classes and leading to severe classification bias. To address these deeply coupled challenges, we propose SPECTRA (Structured Pattern and Enriched Context-aware Temporal Representation Architecture), a unified framework for robust IRTS classification. SPECTRA introduces a frequency-guided observation encoder that reconstructs temporal dependencies in a stable manner, mitigating spectral distortion and information corruption. Complementarily, a missingness pattern encoder explicitly captures the dynamic evolution of missing data and leverages it as a discriminative signal. In addition, a prototype-constrained classification paradigm directly optimizes the geometric structure of the feature space, enhancing intra-class compactness and alleviating generalization bottlenecks caused by class imbalance. Extensive experiments on three public IRTS datasets—P12, P19, and PAM—demonstrate the superior performance of SPECTRA under both missing and imbalanced conditions.

### 摘要中文翻译

不规则时间序列广泛存在于真实应用中，其中不均匀采样和缺失数据给基于深度学习的特征建模带来了根本挑战。虽然现有方法试图保留时间戳信息，但它们往往忽略缺失本身所蕴含的结构化模式；当数据不完整进一步加剧类别不平衡时，这些方法表现较差。具体而言，时间不规则性阻碍了长程依赖和局部模式建模，而稀疏观测限制了表示能力，并对少数类造成不成比例的损害，导致严重的分类偏置。为了解决这些深度耦合的挑战，作者提出 SPECTRA，即结构化模式与增强上下文感知的时间表示架构，用于稳健的不规则时间序列分类。SPECTRA 引入频率引导的观测编码器，以稳定方式重构时间依赖，缓解频谱失真和信息损坏；同时，缺失模式编码器显式捕获缺失数据的动态演化，并将其作为判别信号。此外，原型约束的分类范式直接优化特征空间几何结构，增强类内紧凑性，并缓解类别不平衡带来的泛化瓶颈。作者在 P12、P19 和 PAM 三个公开不规则时间序列数据集上进行了大量实验，证明 SPECTRA 在缺失和不平衡条件下具有优越表现。

### 专家讲解

这篇论文面向的是临床事件预测、可穿戴监测等高风险分类场景：样本不仅是不规则采样和缺失的，而且类别分布往往极不均衡。少数类通常对应更关键但更稀少的事件，例如死亡风险、异常状态、疾病发作或特殊活动模式。论文认为，真实数据中的“缺失”与“类别不平衡”不是两个可以独立处理的问题，而是存在信息论意义上的耦合：在 MAR 或 MNAR 等现实观测机制下，缺失模式本身可能包含类别信息，而且这种信息对少数类尤其重要。

技术路线上，SPECTRA 采用观测值与缺失掩码的双流建模。第一条路线是 Missingness-Aware Frequency Filtering，通过频域增强和动态感受野机制处理不规则采样造成的频谱泄漏、混叠和局部信息断裂；第二条路线是 Missingness Pattern Encoder，把缺失掩码从“待修复的麻烦”转化为“可学习的判别信号”，建模连续缺失、间歇缺失、突发缺失等模式；第三部分是 Category-Guided Feature Refinement，用类别原型约束优化特征空间，使同类样本更紧凑、异类更可分，从而缓解少数类表示坍塌。

论文突破的核心难点是：在不规则采样、缺失严重、类别不平衡同时存在时，传统补全式或时间编码式方法容易把缺失当作噪声抹掉，反而丢失与类别相关的观测机制信息。SPECTRA 的能力在于联合利用观测值、缺失结构、频域稳定性和类别原型几何，使模型在缺失和不平衡条件下仍能保持更稳健的分类边界。

### 审稿人视角：价值、不足与启发

我认为最有价值的思想是“Missing-Imbalance Coupling”这个问题定义。它提醒我们：缺失并不总是纯粹的数据质量问题，也可能是数据生成机制的一部分。尤其在医疗、工业和行为数据中，某些变量为什么没被观测、多久没被观测、缺失是否连续，本身可能反映了病情、设备状态、采样策略或用户行为。因此，把缺失模式显式建模为判别信息，比单纯追求补全精度更接近分类任务目标。

不足方面，SPECTRA 的模块较多，包含频域编码、动态卷积、缺失模式编码和原型约束，整体方法复杂度和可解释成本都不低。论文的理论叙事较强，但在实际应用中，缺失模式是否稳定携带类别信息依赖具体数据采集机制；如果缺失主要来自随机系统故障或跨机构采样差异，模型可能学习到不可泛化的观测偏置。此外，频域方法通常默认一定的周期或频谱结构，对强事件驱动、非平稳、短序列场景的适配仍值得验证。

对我们工作的横向启发是：在任何异步时序分类系统中，都应把缺失掩码、时间间隔、观测频率等“采样行为特征”作为一等公民，而不是仅仅用于补值或 mask attention。纵向深化上，可以进一步研究缺失机制的因果建模：区分“由真实状态导致的有信息缺失”和“由系统流程导致的无关缺失”，避免模型把采样偏差误当作类别规律。还可以将 SPECTRA 的类别原型思想与开放集识别、少样本异常检测、主动采样策略结合，让模型不仅分类，还能判断哪些变量在未来最值得观测。

