# Title: Sklar Shift Breaker：采样边缘剥离的 Copula 状态分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `*summary*.md`、`*work*.md`：当前工作区仍未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md`：其中记录了多轮任务均未发现 `my_work_summary.md`，并列出了当前工作区未检出的历史提案机制。
- 已读取近期论文记录：`paper_daily.md`。
- 已读取当前工作区内全部历史提案：
  - `ideas/Idea_Proposal_2026-06-12.md`
  - `ideas/Idea_Proposal_2026-06-13.md`
  - `ideas/Idea_Proposal_2026-06-14.md`
  - `ideas/Idea_Proposal_2026-06-16.md`
  - `ideas/Idea_Proposal_2026-06-19.md`
  - `ideas/Idea_Proposal_2026-06-21.md`
  - `ideas/Idea_Proposal_2026-06-22.md`
  - `ideas/Idea_Proposal_2026-06-23.md`
  - `ideas/Idea_Proposal_2026-06-25.md`
  - `ideas/Idea_Proposal_2026-06-26.md`
  - `ideas/Idea_Proposal_2026-07-12.md`
  - `ideas/Idea_Proposal_2026-07-13.md`
  - `ideas/Idea_Proposal_2026-07-14.md`
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`
  - `Idea_Proposal_2026-06-24.md`
  - `Idea_Proposal_2026-06-27.md`
  - `Idea_Proposal_2026-07-15.md`
  - `Idea_Proposal_2026-07-16.md`
  - `Idea_Proposal_2026-07-17.md`
  - `Idea_Proposal_2026-07-18.md`

### 历史核心机制黑名单

为避免与历史提案和 `paper_daily.md` 中已有机制发生思维重合，本提案明确避开以下方向作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer。
8. 分类梯度与采样 score 的零空间正交化。
9. hazard-driven counterfactual resampling。
10. 多个 `do(policy)` 视图的 risk variance 约束。
11. 生理流算子与采样算子的交换子手术。
12. value graph / policy graph 的交换或分离损失。
13. policy residual sink 作为采样残差收纳槽。
14. additive evidence market、protocol tax、token-level evidence budget 或边际证据审计。
15. 后验商动力学、模型空间 posterior quotient、采样似然因子相除、干预积分分类。
16. reconstruction error cartography、ANOVA-style state/policy error projection、VQ semantic clauses、policy-sensitive acquisition clauses、HSIC redaction checksum。
17. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler、policy coverage loss。
18. 采样测度上的 Radon-Nikodym density ratio、doubly robust target-measure correction、influence-bound regularization。
19. previsible martingale query、continuous-discrete standardized innovation、optional-stopping moment control、stopping recipe collator、predictability barrier。
20. soft excursion topology、censored persistence interval likelihood、censor envelope、fragmentation sobriety。
21. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
22. policy-only negative film、dual-exposure value/shadow encoders、latent shadow eraser/stencil、shadow nullification high-entropy loss、value retention buckets。
23. latent packet codeword、learnable parity-check、syndrome locator、syndrome-guided packet repair、codeword closure。
24. conditional knockoff calendar、calendar adapter、knockoff group statistics、soft knockoff-FDR firewall、swap symmetry calibration。
25. measurement-Jacobian/Fisher-style observability witness、counterfactual observability probe bank、low-quantile state-coordinate gate、observability-routed classification。
26. subjective-logic / Dirichlet evidential classification、policy-induced ignorance/vacuity mass、class-wise evidence discount、policy stress audit bank。
27. observation-set policy lattice、meet/join visibility masks、信息偏序单调性、次模边际契约、quality-order loss、shortcut curvature penalty。
28. trace-instrumented kernel CDE、solver-trace mediator、reference trace bank、front-door trace-standardized readout、trace do-slope penalty。
29. RKHS cubature weights、kernel moment exactness、ridge KKT cubature solver、control-variate cubature constraints、cubature jackknife leverage。
30. measurement-action bisimulation、canonical measurement action battery、potential observation response head、action Bellman closure、response-distance Lipschitz contract。
31. policy-word signature、mixed value-policy iterated integrals、shuffle-algebra counterterm renormalization、policy-word edit bank。
32. policy-thermodynamic free energy、policy entropy meter、temperature ladder、heat-capacity penalty、Maxwell-style relation。
33. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing、MedMamba 的 frequency/adaptive graph branch、MedSpaformer 的普通 token sparsification、MILM 的 value-redacted classifier、StarEmbed 的直接 foundation embedding、LLMTS 的普通 LLM alignment 或 MVC-CDE 的普通多视图平滑路径作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做对抗、不做一致性、不做后验除法、不做随机平滑、不做停时鞅、不做拓扑、不做 gauge、不做纠错码、不做 knockoff、不做观测性门控、不做 evidential uncertainty、不做信息格、不做 solver trace、不做 RKHS cubature、不做 measurement-action bisimulation、不做 signature renormalization 或 thermodynamic annealing；而是用 Sklar 定理把不规则事件的联合分布拆成“采样/测量边缘分布”和“状态依赖 Copula”。分类器只消费被边缘 CDF 剥离后的 copula state code，从架构上让训练医院的采样边缘、时间间隔边缘、测量质量边缘难以直接进入类别边界。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

采样策略偏移经常首先改变的是 **边缘分布**：

- 某医院更频繁地测某个变量，使该变量的 `delta_t` 边缘分布变化；
- 某设备在夜间测量质量下降，使 measurement noise / quality 的边缘分布变化；
- 某个实验室 panel 在训练环境中同步出现，使变量可见性的边缘频率变化；
- 某个巡天或 ICU 流程改变 cadence，使观测时间的边缘分布变化。

但真正对分类更稳定的信号往往不是这些边缘本身，而是剥离边缘后仍存在的依赖结构：

```text
同一病程下：乳酸异常与血压下降的秩依赖关系；
同一设备状态下：多通道传感器变化的尾部共动关系；
同一变量星类别下：不同 band 光变幅度的相位秩相关。
```

近期 `paper_daily.md` 给出三个关键启发：

1. **SDEVI** 强调不规则观测应在离散观测联合分布上做连续-离散推断，而不是先补齐网格。本提案沿着“联合分布建模”继续推进，但不做 SDE 后验分解，而是用 Sklar factorization 分离边缘和 copula。
2. **StarEmbed** 强调真实不规则科学时序中 cadence、band coverage 与异方差测量误差会共同改变可恢复信息。本提案把这些变化视为采样/测量边缘，而不是直接把它们送入分类。
3. **Rethinking LLMs for Irregular Time Series Classification in Critical Care** 指出，前端 encoder 是否尊重时间戳、缺失和异步比后端 LLM alignment 更关键。因此本提案把去偏动作放在 encoder 的概率积分变换层，而不是依赖大模型或语言摘要。

**Sklar Shift Breaker (SSB)** 的核心目标是：

> 让模型先学习每个变量、时间间隔和测量质量在当前采样政策下的边缘 CDF，再把观测事件变换成 policy-marginal-free 的 copula rank tokens；分类器只能基于这些 rank tokens 的依赖结构做决策，而采样边缘由单独的 marginal heads 解释和审计。

这与当前“采样解耦/反事实干预”框架的关系是：

- value process 不再直接输出 pooled hidden state，而是输出观测值的条件边缘 CDF；
- sampling process 不再输出策略编码、危险率、密度比或残差槽，而是输出时间间隔、变量可见性、质量噪声的边缘 CDF；
- counterfactual intervention 不再生成一致性视图、风险视图或认证视图，而是生成 **marginal stress recipes**，专门训练边缘 CDF 对采样政策变化敏感；
- classifier 只读取 copula state code，不读取 raw gap、raw mask、policy id、marginal likelihood 或 recipe code。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从 event embedding 改为 Marginal-Stripped Copula Encoder

给定不规则事件流：

```text
e_i = (x_i, t_i, d_i, q_i)
```

其中 `x_i` 是观测值，`t_i` 是时间戳，`d_i` 是变量 id，`q_i` 是测量质量或噪声 proxy。SSB 分三步编码。

#### A. Value Marginal CDF

对每个事件，使用历史摘要、变量 id 和质量 proxy 预测观测值的条件 CDF：

```text
u_i^x = F_x(x_i | history_before_i, d_i, q_i)
```

随后做可微概率积分变换：

```text
r_i^x = Phi^{-1}(clip(u_i^x))
```

这里的 `r_i^x` 不是 OS-MQ 的标准化创新：它不构造 previsible query，不使用 optional stopping moment，不把停时 recipe 作为约束；它只是 Sklar 分解中的边缘剥离坐标。

#### B. Sampling / Quality Marginal CDF

sampling branch 只解释边缘：

```text
u_i^g = F_gap(delta_t_i | d_i, local_context)
u_i^q = F_quality(q_i | d_i, device/site-free context)
```

这些边缘 CDF 用于训练和诊断，但不进入分类头。它们回答的是“这类 gap / quality 在当前采样政策下有多典型”，不是“这个类别是什么”。

#### C. Copula State Code

将 value rank token、变量 id 和少量质量权重送入 copula encoder：

```text
c_i = Embed(d_i) + RankProj(r_i^x) + QualityWeight(q_i)
z_copula = CopulaSSM({c_i})
```

分类器只消费 `z_copula`。因此，如果训练环境中某个类别被更频繁测量，raw `delta_t` 的边缘变化会被 marginal CDF 解释掉；只有剥离边缘后仍稳定存在的 value-rank dependence 才能推动分类。

### 2.2 改 Loss：从去偏正则转向 Sklar Factorization Discipline

总目标：

```text
L = L_cls
  + lambda_val * L_value_pit
  + lambda_sam * L_sampling_pit
  + lambda_cop * L_copula_tail
  + lambda_stress * L_marginal_stress
```

#### A. 分类损失 `L_cls`

分类器只使用 copula state code：

```text
L_cls = CE(Classifier(z_copula), y)
```

不把 sampling marginal code、gap rank、quality rank 或 policy recipe 拼入分类器。

#### B. Value PIT Calibration `L_value_pit`

为了确保 value marginal CDF 真正剥离变量尺度、噪声尺度与站点边缘，对 `u_i^x` 做概率积分变换校准：

```text
L_value_pit =
  sum_bins (mean(u^x_bin) - 0.5)^2
  + sum_bins (var(u^x_bin) - 1/12)^2
```

bin 可按变量、质量分位数和粗时间窗划分。若边缘 CDF 校准成功，`u_i^x` 在条件边缘下近似均匀，分类器被迫依赖多个 rank token 之间的 copula 依赖，而不是单个变量的原始数值尺度。

#### C. Sampling PIT Calibration `L_sampling_pit`

对时间间隔和测量质量边缘做同样校准：

```text
L_sampling_pit = PITUniform(u^g) + PITUniform(u^q)
```

这不是 density ratio，也不是采样概率估计。它不试图重加权风险，只要求 sampling branch 能把政策边缘解释成可校准的 CDF 坐标。

#### D. Copula Tail Dependence Loss `L_copula_tail`

采样政策偏移下最容易保留下来的状态信号往往是 rank-tail dependence：例如多个变量同时处于异常分位、某些 band 同时进入尾部亮度变化、传感器多通道共同进入高风险秩区间。

SSB 为 copula encoder 加一个低秩 tail-dependence head：

```text
lambda_tail = P(r_j > tau | r_i > tau, z_copula)
```

训练时用 batch 内同变量组或临床/物理变量组的 tail co-exceedance 做自监督：

```text
L_copula_tail = BCE(tail_head(z_copula), observed_tail_pair)
```

它不同于 graph edge 学习：不构造变量图、不分离 policy graph / state graph、不使用 graph attention；只在 rank space 中学习尾部共动结构。

#### E. Marginal Stress Loss `L_marginal_stress`

counterfactual intervention 只用于训练边缘 CDF，而不是训练分类一致性。`MarginalStressCollator` 生成以下 recipes：

- `gap_stretch`：拉长某些变量的采样间隔；
- `quality_degrade`：提高测量噪声或异方差；
- `coverage_skew`：改变变量可见性边缘；
- `cadence_bucket_shift`：改变早/中/晚窗口的采样边缘。

对每个 recipe，只要求 sampling marginal heads 能识别新的边缘坐标：

```text
L_marginal_stress = PITUniform(F_gap(delta_t^recipe)) + PITUniform(F_quality(q^recipe))
```

分类头不看 recipe，也不被要求在 recipe 之间保持 logits 一致。这样反事实模块的作用是“把采样边缘交给边缘模型解释”，而不是“把所有反事实视图压成同一个表示”。

### 2.3 改 Dataloader：返回 Marginal Stress Bank

新增 `CopulaMarginalStressCollator`。每个 batch 返回：

1. 原始事件：`event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_std`。
2. 边缘 stress recipes：
   - `gap_stretch_visibility` / `gap_stretch_time`；
   - `quality_degrade_std`；
   - `coverage_skew_visibility`；
   - `cadence_bucket_time`。
3. tail pair supervision：
   - 在 rank space 中构造 `tail_pair_index`；
   - 只记录变量组或 band 组是否共同超过高分位阈值。
4. marginal diagnostics：
   - value PIT histogram；
   - gap PIT histogram；
   - quality PIT histogram；
   - copula tail-dependence score。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `ValueMarginalCDF + CopulaSSM`。
- 现有 sampling branch 改为 `SamplingMarginalCDF`，只学习 gap / visibility / quality 的边缘分布。
- 现有 counterfactual intervention 改为 `MarginalStressBank`，训练 sampling marginal heads 对政策边缘变化敏感。
- 推理阶段：
  - 快速模式：只计算 `u_i^x` 和 `z_copula`，输出分类；
  - 诊断模式：同时输出 value/gap/quality PIT 偏离、tail dependence 与 marginal-shift score；
  - 部署告警：若 gap/quality PIT 严重偏离训练均匀性，但 copula classifier 仍高置信，可标记“边缘采样政策已偏移，分类依赖需要审计”。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def normal_icdf(u: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    u = u.clamp(eps, 1.0 - eps)
    dist = torch.distributions.Normal(
        torch.zeros((), device=u.device, dtype=u.dtype),
        torch.ones((), device=u.device, dtype=u.dtype),
    )
    return dist.icdf(u)


def pit_uniform_loss(u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Match probability-integral-transform samples to Uniform(0, 1)."""
    u = u.clamp(1e-4, 1.0 - 1e-4)
    mean = masked_mean(u, mask, dim=1)
    second = masked_mean(u.pow(2), mask, dim=1)
    # Uniform(0, 1): E[U] = 1/2, E[U^2] = 1/3.
    return (mean - 0.5).pow(2).mean() + (second - 1.0 / 3.0).pow(2).mean()


class CausalHistoryStem(nn.Module):
    """Build history states; state at index i only summarizes events before i."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, batch: dict) -> torch.Tensor:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id.clamp_min(0))
        event_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(meas_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * mask.unsqueeze(-1)
        seq_h, _ = self.rnn(event_h)
        pre_h = torch.zeros_like(seq_h)
        pre_h[:, 1:] = seq_h[:, :-1]
        return pre_h


class LogisticCDFHead(nn.Module):
    """Predict a conditional logistic CDF for scalar event attributes."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, 2),
        )

    def forward(self, feature: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(feature)
        loc = raw[..., 0]
        scale = F.softplus(raw[..., 1]) + 1e-3
        u = torch.sigmoid((target - loc) / scale)
        nll = -(
            torch.log(u.clamp_min(1e-6))
            + torch.log((1.0 - u).clamp_min(1e-6))
            - torch.log(scale)
        )
        return u, loc, nll


class MarginalCDFBlock(nn.Module):
    """Estimate value, gap and quality marginal CDFs."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.value_cdf = LogisticCDFHead(2 * hidden_dim + 1)
        self.gap_cdf = LogisticCDFHead(hidden_dim + 2)
        self.quality_cdf = LogisticCDFHead(hidden_dim + 2)

    def forward(self, batch: dict, history: torch.Tensor) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id.clamp_min(0))

        value_feature = torch.cat([history, var_h, torch.log1p(meas_std).unsqueeze(-1)], dim=-1)
        u_value, _, value_nll = self.value_cdf(value_feature, value)

        marginal_feature = torch.cat(
            [var_h, torch.log1p(time).unsqueeze(-1), mask.unsqueeze(-1)],
            dim=-1,
        )
        u_gap, _, gap_nll = self.gap_cdf(marginal_feature, torch.log1p(delta_t))
        u_quality, _, quality_nll = self.quality_cdf(marginal_feature, torch.log1p(meas_std))

        return {
            "u_value": u_value,
            "u_gap": u_gap,
            "u_quality": u_quality,
            "value_nll": value_nll * mask,
            "gap_nll": gap_nll * mask,
            "quality_nll": quality_nll * mask,
        }


class CopulaStateEncoder(nn.Module):
    """Encode dependence among marginal-stripped value rank tokens."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.rank_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, batch: dict, u_value: torch.Tensor) -> dict:
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(u_value))

        rank_value = normal_icdf(u_value)
        quality_weight = 1.0 / (1.0 + torch.log1p(meas_std))
        rank_x = torch.stack([rank_value, quality_weight], dim=-1)
        token = self.var_embed(var_id.clamp_min(0)) + self.rank_proj(rank_x)
        token = token * mask.unsqueeze(-1)
        seq_h, _ = self.rnn(token)
        state = masked_mean(seq_h, mask, dim=1)
        return {"copula_state": state, "rank_value": rank_value, "token": token}


class TailDependenceHead(nn.Module):
    """Predict rank-space tail co-exceedance without constructing a graph."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, copula_state: torch.Tensor) -> torch.Tensor:
        return self.net(copula_state).squeeze(-1)


class SklarShiftBreaker(nn.Module):
    """Sampling-policy robust classifier via marginal-stripped copula states."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.history = CausalHistoryStem(num_vars, hidden_dim)
        self.marginal = MarginalCDFBlock(num_vars, hidden_dim)
        self.copula = CopulaStateEncoder(num_vars, hidden_dim)
        self.tail = TailDependenceHead(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict) -> dict:
        history = self.history(batch)
        marg = self.marginal(batch, history)
        cop = self.copula(batch, marg["u_value"])
        logits = self.classifier(cop["copula_state"])
        return {**marg, **cop, "logits": logits, "tail_logit": self.tail(cop["copula_state"])}

    def tail_target(self, rank_value: torch.Tensor, mask: torch.Tensor, threshold: float = 1.25) -> torch.Tensor:
        high_tail = ((rank_value > threshold).to(rank_value.dtype) * mask).sum(dim=1)
        visible = mask.sum(dim=1).clamp_min(1.0)
        return (high_tail / visible > 0.15).to(rank_value.dtype)

    def training_loss(
        self,
        batch: dict,
        lambda_val: float = 0.15,
        lambda_sam: float = 0.15,
        lambda_cop: float = 0.10,
        lambda_stress: float = 0.10,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]
        mask = batch["event_mask"]

        cls_loss = F.cross_entropy(out["logits"], labels)
        value_pit = pit_uniform_loss(out["u_value"], mask)
        sampling_pit = pit_uniform_loss(out["u_gap"], mask) + pit_uniform_loss(out["u_quality"], mask)

        tail_target = self.tail_target(out["rank_value"].detach(), mask)
        tail_loss = F.binary_cross_entropy_with_logits(out["tail_logit"], tail_target)

        stress_losses = []
        for stress_batch in batch.get("marginal_stress_bank", []):
            with torch.no_grad():
                stress_history = self.history(stress_batch)
            stress_marg = self.marginal(stress_batch, stress_history)
            stress_mask = stress_batch["event_mask"]
            stress_losses.append(pit_uniform_loss(stress_marg["u_gap"], stress_mask))
            stress_losses.append(pit_uniform_loss(stress_marg["u_quality"], stress_mask))
        if stress_losses:
            stress_loss = torch.stack(stress_losses).mean()
        else:
            stress_loss = torch.zeros((), device=cls_loss.device, dtype=cls_loss.dtype)

        total = (
            cls_loss
            + lambda_val * value_pit
            + lambda_sam * sampling_pit
            + lambda_cop * tail_loss
            + lambda_stress * stress_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "value_pit_loss": value_pit.detach(),
            "sampling_pit_loss": sampling_pit.detach(),
            "copula_tail_loss": tail_loss.detach(),
            "marginal_stress_loss": stress_loss.detach(),
        }


@torch.no_grad()
def build_marginal_stress_bank(batch: dict) -> list[dict]:
    """Create sampling-margin stress views for marginal heads only."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    meas_std = batch.get("measurement_std", torch.zeros_like(value))
    bsz, num_events = time.shape
    device = time.device

    out = []

    def clone(time_new, mask_new, std_new):
        item = dict(batch)
        item["event_value"] = value * mask_new
        item["event_time"] = time_new
        item["event_var_id"] = var_id
        item["event_mask"] = mask_new
        item["measurement_std"] = std_new
        return item

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    # 1. Gap stretch: later events become more sparse.
    late = (time_norm > 0.66).to(time.dtype)
    stretched_time = time * (1.0 + 0.35 * late)
    out.append(clone(stretched_time, mask, meas_std))

    # 2. Quality degradation: value is kept, measurement uncertainty shifts.
    degraded_std = meas_std + 0.25 * value.detach().abs().mean(dim=1, keepdim=True)
    out.append(clone(time, mask, degraded_std))

    # 3. Coverage skew: odd variables become less visible.
    odd_var = (var_id % 2 == 1).to(mask.dtype)
    keep_even = mask * (1.0 - odd_var)
    out.append(clone(time, keep_even, meas_std))

    # 4. Cadence bucket shift: early routine sampling only.
    early_middle = (time_norm <= 0.66).to(mask.dtype) * mask
    out.append(clone(time, early_middle, meas_std))

    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `gap-marginal shift`：训练环境高风险样本被密集复测，测试环境采用稀疏 routine cadence。
   - `quality-marginal shift`：测试环境测量噪声或异方差更强，借鉴 StarEmbed 的 measurement uncertainty。
   - `coverage-marginal shift`：变量组或 band 覆盖率系统性改变。
   - `cadence-bucket shift`：早/中/晚时间窗的采样边缘分布改变。

2. **对比方法**
   - 普通 irregular encoder。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - SDEVI-style continuous-discrete classifier without copula stripping。
   - StarEmbed / foundation embedding + OOD score baseline。
   - LLMTS 风格 encoder/alignment baseline。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN、RKHS Cubature、Measurement-Action Bisimulation、Policy-Word Signature、Thermodynamic Annealer。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - value PIT calibration error。
   - gap / quality PIT shift score。
   - copula tail-dependence stability：跨采样边缘变化时 tail score 是否稳定。
   - marginal leakage probe：只用 gap / quality PIT histogram 预测标签的能力，越低越好。

4. **消融实验**
   - 去掉 value PIT calibration，验证原始数值边缘是否重新成为 shortcut。
   - 去掉 sampling PIT calibration，验证 gap / quality 边缘偏移是否污染 copula state。
   - 去掉 tail-dependence head，检查 rank-tail 共动是否是关键稳定信号。
   - 让 classifier 同时读取 `u_gap/u_quality`，验证采样边缘进入分类头会降低 worst-policy accuracy。
   - 将 marginal stress recipes 替换成随机 mask，验证收益来自边缘压力而非普通增强。

## 5. 预期创新性

1. **从采样去偏转向 Sklar 边缘剥离**：不估计采样概率、不做密度比、不做对抗，而是把采样策略最容易改变的边缘分布从联合分布中剥离出去。
2. **从 raw event representation 转向 copula state code**：分类器依赖 rank-space dependence，而不是依赖 raw delta-t、mask density、measurement quality 或单变量尺度。
3. **从反事实一致性转向边缘压力校准**：counterfactual intervention 只训练 marginal CDF heads 解释政策边缘，不要求不同政策视图的 logits 或 representation 一致。
4. **吸收 SDEVI 的联合分布视角但不做 SDE 后验**：SSB 同样尊重离散不规则观测的联合分布，但用 Sklar factorization 替代路径后验或采样似然除法。
5. **吸收 StarEmbed 的异方差/cadence 启发**：measurement noise、band coverage 和 cadence 被纳入边缘 CDF 诊断，而不是作为直接分类特征。
6. **吸收 LLMTS 的 encoder-first 审计结论**：鲁棒性来自前端概率积分变换和 copula encoder，而不是后端 LLM alignment。

## 6. 一句话投稿卖点

**Sklar Shift Breaker 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样边缘分布改变污染联合事件分布”的问题，并通过 value/gap/quality marginal CDF、概率积分变换与 copula state encoder，让分类器依赖剥离采样边缘后的 rank-dependence 结构，而不是依赖训练医院、设备或巡天策略制造的时间间隔、覆盖率和测量质量边缘捷径。**
