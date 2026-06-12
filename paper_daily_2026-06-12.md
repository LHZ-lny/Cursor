# Paper Daily - 2026-06-12

## 检索与去重记录

- 已强制读取根目录下 `paper_daily_*.md`：本次未发现历史日报文件。
- 黑名单论文标题：空。
- 检索范围：围绕近 3-7 个月内的 irregular sampled / irregular multivariate time-series classification 顶会论文，重点查看 NeurIPS 2025、AAAI 2026、ICLR 2026、ICML 2026 等来源。
- 本次仅保留全新工作 2 篇。

## 1. Adaptive Time Encoding for Irregular Multivariate Time-Series Classification

- 会议：NeurIPS 2025 Poster
- 作者：Sangho Lee, Kyeongseo Min, Youngdoo Son, Hyungrok Do
- 链接：https://nips.cc/virtual/2025/loc/san-diego/poster/116324
- 关键词：irregular sampling, adaptive time encoding, learnable reference points, temporal/inter-variable consistency regularization

### 场景、任务与核心难点

这篇工作面向不规则采样的多变量时序分类。典型场景包括医疗监测、传感器网络和可穿戴设备：不同变量不在同一时间戳被观测，观测间隔不均匀，单条样本内各变量的观测次数也可能不同。核心难点不是简单的缺值填补，而是如何在时间轴错位、变量异步和观测密度变化的条件下，提取对分类真正有用的动态模式。

作者提出 adaptive time encoding：不再手工指定固定参考时间点或先粗暴重采样到规则网格，而是在模型中学习一组 reference points，并在这些点上生成潜在表征。模型进一步引入 temporal consistency 与 inter-variable consistency regularization，使学习到的表征同时尊重时间连续性和变量间依赖。

### 审稿人视角：价值与不足

最有价值的思想是把“不规则时间戳如何对齐到分类表征”转化为可学习的时间编码问题。learnable reference points 相当于让模型自己选择一组对任务最有判别力的时间锚点，比固定网格、手工插值或单纯把 delta-t 拼到输入中更灵活；一致性正则也使其不只是记住某些采样位置，而是显式压住时间邻域和变量间关系的表征漂移。

不足在于，它主要优化的是同分布 benchmark 上的不规则采样分类性能。若训练集中的采样策略本身与标签高度相关，例如某类病人被更频繁监测，learnable reference points 和 missingness-aware 表征可能吸收这种策略性信号。这样在测试环境换医院、换传感器调度规则或换主动采样策略时，模型可能把采样政策当成病理或状态模式，存在 sampling-policy leakage 的风险。

### 对 Sampling-Policy Shift 的启发

这篇工作可横向启发我们把“采样策略”从原始时间序列中显式拆出来：一条支路学习基于观测值的连续时间表征，另一条支路学习时间戳/缺失模式，再通过不变性约束控制二者对分类决策的贡献。learnable reference points 也可被改造成 policy-invariant anchors：在不同采样策略增强、mask 重采样或环境分组下，要求同一真实轨迹投影到相近的参考点表征。纵向上，可以在 ATENet 的一致性正则之外加入跨采样策略一致性，例如对同一序列模拟不同观测调度，约束分类 logits 或中间 reference-point representation 保持稳定。

## 2. Beyond Missing Data Imputation: Information-Theoretic Coupling of Missingness and Class Imbalance for Optimal Irregular Time Series Classification

- 简称：SPECTRA
- 会议：AAAI 2026 Technical Track
- 作者：Xin Qin, Mengna Liu, Wenjie Wang, Shuxin Li, Tianjiao Li, Xiufeng Liu, Xu Cheng
- 链接：https://ojs.aaai.org/index.php/AAAI/article/view/39682
- 关键词：irregular time series classification, missingness pattern encoder, frequency-guided observation encoder, class imbalance, prototype-constrained classification

### 场景、任务与核心难点

这篇工作处理不规则时序分类中的两个耦合问题：不均匀采样/缺失导致可观测信息不足，类别不平衡又使少数类更容易被稀疏观测进一步伤害。论文在 P12、P19、PAM 等公开 IRTS 数据集上验证，任务以医疗/生理时序分类为代表。

其核心难点是：缺失并非纯噪声。采样频率、缺失间隔和缺失模式可能反映了系统行为、病人状态或数据采集约束；但如果只做 imputation，会丢掉这部分结构；如果直接利用缺失模式，又可能放大类别偏差和策略依赖。SPECTRA 因此同时引入 frequency-guided observation encoder、missingness pattern encoder 和 prototype-constrained classification，用频域稳定表征修复时间依赖，用缺失模式编码捕捉 missingness dynamics，并用原型约束改善类内紧凑性与类间结构。

### 审稿人视角：价值与不足

最有价值的地方是明确把 missingness 与 class imbalance 作为耦合机制建模，而不是把缺失仅视为预处理问题。frequency-guided observation encoder 关注不规则观测对时序频谱与长期依赖的破坏；missingness pattern encoder 则承认缺失模式本身具有判别信息；prototype-constrained classifier 进一步把表征几何结构纳入优化目标，针对少数类泛化弱的问题给出直接约束。

主要不足是方法的优势也可能成为风险：如果缺失模式来自训练环境中的采样政策，而非稳定的目标生成机制，那么显式利用 missingness pattern 可能在跨机构、跨设备或主动采样策略变化时失效。论文强调 missingness-class imbalance coupling，但对“采样政策改变后哪些 missingness 信号仍应保留、哪些应被剥离”的因果判别还不充分。其评估也主要围绕公开 benchmark，尚不足以证明在真实 sampling-policy shift 下的稳健性。

### 对 Sampling-Policy Shift 的启发

SPECTRA 对我们的问题非常直接：采样策略偏移不应被简单当作缺失率变化，而应被建模为影响观测图样、类别可分性和表征几何的结构性因素。可以借鉴它的 missingness pattern encoder，但加入环境/策略对抗或条件不变性约束，区分“由真实状态驱动的 informative missingness”和“由采样政策驱动的 spurious missingness”。prototype 思路也可扩展为 policy-aware prototypes：同一类别在不同采样策略下共享核心类别原型，同时允许策略残差存在但不进入最终分类边界。frequency-guided encoder 则提示我们，在非规则采样下评估策略偏移时，不只看缺失比例，也要看采样间隔分布对频域可恢复性的影响。
