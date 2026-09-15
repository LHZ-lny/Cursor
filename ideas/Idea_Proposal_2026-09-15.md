# Title: Do-Schwarzian Warp Conditioner：面向采样策略偏移的反事实曲率配准调理器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 因任务明确引用该文件，已执行远端主分支核对：`origin/main` 顶层同样未发现 `my_work_summary.md` 或 `summary / work / 总结` 类替代文件。
- 已读取 `paper_daily.md` 的开头、最新末段与关键词检索结果，重点纳入最新 `paper_daily_2026-09-14` 记录中的：
  - **DeepFRC**：端到端 functional registration + classification、可微同胚时间扭曲、分类驱动配准。
  - 同时记住 DeepFRC 自身的 Fourier spectral representation 与 class-aware contrastive loss，本提案不复用这两个机制作为主创新。
- 已读取当前工作区内全部历史 proposal 文件：
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
  - `ideas/Idea_Proposal_2026-07-28.md`
  - `ideas/Idea_Proposal_2026-07-30.md`
  - `ideas/Idea_Proposal_2026-08-05.md`
  - `ideas/Idea_Proposal_2026-08-06.md`
  - `ideas/Idea_Proposal_2026-08-08.md`
  - `ideas/Idea_Proposal_2026-08-09.md`
  - `ideas/Idea_Proposal_2026-08-22.md`
  - `ideas/Idea_Proposal_2026-08-23.md`
  - `ideas/Idea_Proposal_2026-08-24.md`
  - `ideas/Idea_Proposal_2026-08-25.md`
  - `ideas/Idea_Proposal_2026-08-26.md`
  - `ideas/Idea_Proposal_2026-09-12.md`
  - `ideas/Idea_Proposal_2026-09-13.md`
  - `ideas/Idea_Proposal_2026-09-14.md`
- 已读取自动化记忆 `MEMORIES.md`，并逐项读取其中记录但当前工作区未完整落盘的历史 proposal 摘要，包括 `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`、`idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`、`idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-05.md`、`idea_2026-08-06.md`、`idea_2026-08-07.md`、`idea_2026-08-08.md`、`idea_2026-08-09.md`、`idea_2026-08-10.md`、`idea_2026-08-11.md`、`idea_2026-08-21.md`、`idea_2026-08-22.md`、`idea_2026-08-23.md`、`idea_2026-08-24.md`、`idea_2026-08-25.md`、`idea_2026-09-12.md`、`idea_2026-09-13.md` 与 `idea_2026-09-14.md`。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下机制作为主创新：

1. learnable reference points / adaptive time encoding、频域掩码修复、prototype 约束、missingness pattern 直接分类、简单 policy adversarial / IRM。
2. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
3. 生理流-采样算子交换子、value/policy graph 分离、policy residual sink。
4. protocol tax / additive evidence market、posterior quotient、density ratio / doubly robust、policy-simplex randomized smoothing。
5. reconstruction error cartography、VQ clauses、optional-stopping martingale、censored topology、policy gauge / horizontal transport、syndrome code、knockoff calendar。
6. observability witness、evidential vacuity、information lattice、solver trace front-door、conformal sleeves、IV/control-function、Borda jury。
7. Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock、feasible hull、IRT-DIF、RG fixed point。
8. bitemporal curtain、clinical tomography、matched risk-set likelihood、Gaussian privacy cloak、CauKer orthogonal synthetic forge、JEPA 双辩裁判。
9. cross-representation PID prism、Noether semantic action、fast/slow meta workflow adapter、test-time workflow adaptation。
10. ORA collider factorization、explain-away responsibility gate、workflow-swap path-specific seal。
11. frozen source predictor input adapter、Source Atlas、policy-antipodal source retrieval、policy-neighborhood monopoly penalty。
12. DeepFRC 原始 Fourier spectral representation、class-aware contrastive alignment、把同类曲线简单拉近/异类曲线简单拉远。

本提案选择新的正交切入点：**不再把采样偏移做成概率估计、证据定价、证明、保形、纠错、隐私发布、合成反例、图/后验/信息分解，也不学习病程偏序时钟或 gauge frame；而是把采样政策偏移视为对观测时间轴施加的局部高曲率同胚扭曲。模型允许低曲率的生物相位配准，但把高 Schwarzian 曲率的采样日历扭曲收纳到 policy sidecar。分类器只读取经过最小曲率调理后的 value curve，而不读取 warp 曲率、采样摘要或 policy recipe。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-09-14` 中的 **DeepFRC** 提醒我们：在 functional / trajectory classification 中，时间错位本身就是核心难点。关键形态可能提前、滞后、压缩或拉伸；若不做 registration，分类器可能把相位差当成类别差异。这个思想对非规则采样时间序列非常重要，因为 ICU / 设备 / 科学观测中的采样政策经常改变观测事件在时间轴上的分布。

但直接把 DeepFRC 搬到 sampling-policy shift 上会有风险：

- 若训练医院总是在告警后密集复测，普通分类驱动 registration 可能把这种复测 burst 学成某个类别的标准相位模板；
- 若某中心把 panel 同步记录，另一个中心拆成异步事件，配准器可能把 workflow batching 当成病理事件对齐；
- 若 value-pending 或文书延迟改变了事件出现顺序，分类驱动 warp 可能用高弯曲时间扭曲把记录流程“对齐”成疾病进展；
- Fourier spectral representation 与 class-aware contrastive loss 对相位对齐有效，但历史黑名单已经排除了频域主机制和跨视图对比主机制。

本轮的关键问题不是“要不要做时间配准”，而是：

> 允许模型对真实病理相位做平滑配准，但不允许模型用高度局部、由采样政策制造的日历曲率来提升类别边际。

**Do-Schwarzian Warp Conditioner (DSWC)** 的核心直觉：

1. 每个样本学习一个单调同胚 warp `gamma(t)`，把事实观测时间拉回一个 canonical registration coordinate。
2. 低曲率、全局平滑的 warp 被视为可能的生物相位差，可进入 value curve canonicalizer。
3. 局部高曲率、burst/panel/pending/routine 触发的 warp 被视为采样政策扭曲，由 `WarpCurvatureSidecar` 解释，但不进入分类器。
4. 用 Schwarzian derivative

```text
S(gamma)(t) = gamma'''(t) / gamma'(t) - 1.5 * (gamma''(t) / gamma'(t))^2
```

度量时间 warp 的局部非线性弯曲。它对平移、缩放和简单 projective reparameterization 相对不敏感，却能捕捉密集复测、panel 合并、记录延迟带来的局部时间加速度。
5. 现有反事实采样模块不生成 logits consistency pair，也不做平滑/保形/证明/隐私；它只生成 **warp curvature probes**，训练 sidecar 解释哪些时间弯曲来自采样流程。

这样 DSWC 与当前“采样解耦/反事实干预”框架自然结合：

- value process 产生可配准的 event value curve；
- sampling process 产生采样坐标摘要，只用于预测高 Schwarzian 曲率 sidecar；
- counterfactual intervention 生成 routine/alarm/panel/pending 等曲率探针；
- classifier 读取被调理后的 canonical value curve，不读取 policy sidecar。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通时间编码改为 Minimal-Schwarzian Warp Canonicalizer

DSWC 不学习 reference points，也不构造 disease-progress poset clock。它学习的是每条样本的 **单调注册函数**：

```text
gamma: [0, 1] -> [0, 1], gamma'(t) > 0
```

具体包含三层：

1. **Value Functional Stem**
   - 输入事件 `(value, time, variable, quality)`。
   - 用变量 embedding + value projection 得到 event state。
   - 不使用 Fourier spectrum，不做 class-aware contrastive。

2. **Monotone Warp Net**
   - 从 value stem 与少量时间统计中输出正增量 `delta_gamma = softplus(raw)`。
   - 通过 cumulative sum 得到单调 warp。
   - warp 被分成：

```text
gamma_state  = low-curvature registered calendar
kappa_policy = high-Schwarzian curvature sidecar
```

3. **Warp-Conditioned Canonical Curve**
   - 将事件拉回 canonical grid `tau_j`：

```text
weight_ij = exp(-|gamma_state(t_i) - tau_j| / temp)
curve_j   = sum_i weight_ij * value_event_i
```

   - 分类器只读取 `curve_j` 的 value-derived representation。
   - `kappa_policy`、sampling summary、warp recipe 和 center id 都不进入分类头。

### 2.2 改 Sampling Branch：Warp Curvature Sidecar

sampling branch 不再输出 hazard、density ratio、DIF、privacy noise、atlas key 或 workflow parent。它只从观测坐标中预测一个高曲率模板：

```text
policy_curve = Sidecar(times, var_id, mask, pending, panel)
```

该模板的目标是解释 `|S(gamma)|` 中由采样政策引起的局部曲率峰：

- `routine_round`：时间吸附到固定查房窗口，通常产生分段低频弯曲；
- `alarm_dense`：告警后短窗口密集复测，通常产生局部高正曲率；
- `panel_pack/split`：同步和拆单产生变量簇附近的曲率尖峰；
- `pending_latency`：记录时间与值可用时间错位，产生延迟型曲率。

sidecar 的输出只用于训练和诊断，不进入分类器。

### 2.3 改 Loss：从对齐准确率转向 Schwarzian Warp Discipline

总目标：

```text
L = L_cls
  + lambda_sch * L_state_schwarzian_budget
  + lambda_side * L_policy_curvature_sidecar
  + lambda_min * L_minimal_warp
  + lambda_shape * L_value_shape_preservation
```

#### A. Canonical Curve Classification `L_cls`

分类器只读被低曲率 warp 调理后的 value curve：

```text
L_cls = CE(Classifier(CanonicalCurve(gamma_state, values)), y)
```

这不是 DeepFRC 的 class-aware contrastive，也不是多视图 logits consistency。只有事实 canonical curve 被用于分类。

#### B. State Schwarzian Budget `L_state_schwarzian_budget`

真实病理相位配准应该相对平滑；局部高曲率更可能来自采样政策。对进入分类器的 `gamma_state` 施加预算：

```text
L_state_sch = mean relu(|S(gamma_state)| - b_state)^2
```

这不是 adaptive time encoding，也不是 Noether policy work。它不看类别 margin 对 policy transform 的方向导数，只限制分类路径中使用的时间 warp 不得依赖局部高弯曲日历。

#### C. Policy Curvature Sidecar `L_policy_curvature_sidecar`

反事实采样模块生成若干 warp probes，计算 probe 与事实 warp 的 Schwarzian 差异：

```text
target_kappa_r = stopgrad(|S(gamma_probe_r)| - |S(gamma_factual)|)_+
L_side = SmoothL1(Sidecar(probe_policy_summary_r), target_kappa_r)
```

这让采样支路承担“解释曲率从哪里来”的责任，但 sidecar 不给分类头任何类别证据。与 proof / DIF / RG / privacy / collider 不同，它只建模时间扭曲曲率。

#### D. Minimal Warp `L_minimal_warp`

防止 warp net 为了分类任意扭曲时间轴：

```text
L_minimal =
  mean |log gamma'(t)| 
  + mean |delta log gamma'(t)|
```

它鼓励最小必要配准：能用平滑低曲率 warp 解释的相位差允许保留，必须依赖局部尖锐 warp 的采样捷径被压到 sidecar。

#### E. Value Shape Preservation `L_value_shape_preservation`

配准不能改写观测值语义。将 canonical curve 再采样回原观测点，要求恢复 observed value：

```text
L_shape = SmoothL1(InterpolateBack(curve, gamma_state(t_i)), value_i)
```

这与 reconstruction-error cartography 不同：它不是用误差分布生成伪观测或做 Wasserstein 对齐，只是防止时间 warp 把 value morphology 扭坏。

### 2.4 改 Dataloader：返回 Warp Curvature Probe Bank

新增 `SchwarzianWarpProbeCollator`，每个 batch 返回：

1. 原始事件：`event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_quality`。
2. `policy_probe_bank`：
   - `routine_round_probe`：时间吸附到固定查房窗口；
   - `alarm_dense_probe`：早期稀疏、晚期或异常附近密集；
   - `panel_split_probe`：同步 panel 拆成异步事件；
   - `pending_latency_probe`：值可见性延迟；
   - `budget_sparse_probe`：变量预算导致局部低覆盖。
3. `probe_policy_summary`：每个 probe 的采样坐标摘要。
4. `curvature_recipe_id`：仅用于 sidecar 诊断，不进入分类器。

这些 probe 不是 positive pair，不做 contrastive，不要求 representation / logits / risk 一致。它们只用于估计“若采样政策改变，时间 warp 的 Schwarzian 曲率会在哪里出现”。

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def normalize_time(event_time: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    return (event_time / horizon).clamp(0.0, 1.0)


def finite_schwarzian(gamma: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Finite-difference Schwarzian derivative on a uniform grid.

    gamma: [B, M], monotone values in [0, 1].
    returns: [B, M], padded to match gamma length.
    """

    d1 = gamma[:, 1:] - gamma[:, :-1]
    d1 = F.pad(d1, (1, 0), mode="replicate").clamp_min(eps)

    d2 = d1[:, 1:] - d1[:, :-1]
    d2 = F.pad(d2, (1, 0), mode="replicate")

    d3 = d2[:, 1:] - d2[:, :-1]
    d3 = F.pad(d3, (1, 0), mode="replicate")

    return d3 / d1 - 1.5 * (d2 / d1).pow(2)


class EventValueStem(nn.Module):
    """Value-derived event encoder; sampling summaries stay outside logits."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        time_norm = normalize_time(time, mask)
        delta_t = torch.zeros_like(time_norm)
        delta_t[:, 1:] = (time_norm[:, 1:] - time_norm[:, :-1]).clamp_min(0.0)

        x = torch.cat(
            [
                self.var_embed(var_id.clamp_min(0)),
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(x) * mask.unsqueeze(-1)
        seq_h, _ = self.context(event_h)
        event_state = self.out(seq_h) * mask.unsqueeze(-1)
        pooled = masked_mean(event_state, mask, dim=1)
        return {"event_state": event_state, "pooled_state": pooled, "time_norm": time_norm}


class MonotoneWarpNet(nn.Module):
    """Predict a sample-level monotone diffeomorphic registration warp."""

    def __init__(self, hidden_dim: int, grid_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, grid_size),
        )

    def forward(self, pooled_state: torch.Tensor) -> dict:
        raw = self.net(pooled_state)
        increment = F.softplus(raw) + 1e-4
        gamma = torch.cumsum(increment, dim=-1)
        gamma = gamma / gamma[:, -1:].clamp_min(1e-6)
        gamma = torch.cat([torch.zeros_like(gamma[:, :1]), gamma[:, :-1]], dim=1)
        schwarzian = finite_schwarzian(gamma)
        return {"gamma": gamma, "log_increment": increment.log(), "schwarzian": schwarzian}


def interpolate_gamma_at_events(gamma: torch.Tensor, time_norm: torch.Tensor) -> torch.Tensor:
    """Linear interpolation from uniform gamma grid to event times."""

    grid_size = gamma.size(1)
    pos = (time_norm * (grid_size - 1)).clamp(0, grid_size - 1 - 1e-6)
    left = pos.floor().long()
    right = (left + 1).clamp(max=grid_size - 1)
    frac = (pos - left.to(pos.dtype)).clamp(0.0, 1.0)
    batch = torch.arange(gamma.size(0), device=gamma.device)[:, None]
    return gamma[batch, left] * (1.0 - frac) + gamma[batch, right] * frac


class WarpCanonicalCurve(nn.Module):
    """Pull irregular events back to a canonical low-curvature registration grid."""

    def __init__(self, num_vars: int, hidden_dim: int, grid_size: int = 64, temperature: float = 0.04):
        super().__init__()
        self.num_vars = num_vars
        self.grid_size = grid_size
        self.temperature = temperature
        self.curve_proj = nn.Sequential(
            nn.Linear(num_vars + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, batch: dict, event_state: torch.Tensor, gamma_event: torch.Tensor) -> dict:
        value = batch["event_value"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        tau = torch.linspace(0.0, 1.0, self.grid_size, device=value.device, dtype=value.dtype)

        dist = (gamma_event[:, :, None] - tau[None, None]).abs()
        weight = torch.softmax(-dist / self.temperature, dim=1) * mask[:, :, None]
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

        var_onehot = F.one_hot(var_id, self.num_vars).to(value.dtype)
        value_by_var = var_onehot * value.unsqueeze(-1)
        curve_value = torch.einsum("bnm,bnv->bmv", weight, value_by_var)
        curve_state = torch.einsum("bnm,bnh->bmh", weight, event_state)

        curve_h = self.curve_proj(torch.cat([curve_value, curve_state], dim=-1))
        seq_h, _ = self.temporal(curve_h)
        canonical_state = self.out(seq_h).mean(dim=1)
        return {"canonical_state": canonical_state, "curve_value": curve_value, "event_to_grid_weight": weight}


class WarpCurvatureSidecar(nn.Module):
    """Explain high-Schwarzian warp curvature from observation coordinates only."""

    def __init__(self, num_vars: int, hidden_dim: int, grid_size: int = 64):
        super().__init__()
        self.num_vars = num_vars
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(num_vars + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, grid_size),
            nn.Softplus(),
        )

    def summarize_policy(self, batch: dict) -> torch.Tensor:
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        pending = batch.get("value_pending", torch.zeros_like(mask))

        time_norm = normalize_time(time, mask)
        delta_t = torch.zeros_like(time_norm)
        delta_t[:, 1:] = (time_norm[:, 1:] - time_norm[:, :-1]).clamp_min(0.0)

        var_rate = F.one_hot(var_id, self.num_vars).to(time.dtype) * mask.unsqueeze(-1)
        var_rate = var_rate.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        close = (delta_t <= masked_mean(delta_t, mask, dim=1).unsqueeze(-1).clamp_min(1e-6)).to(time.dtype)
        var_change = torch.zeros_like(mask)
        var_change[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(time.dtype)

        stats = torch.cat(
            [
                mask.mean(dim=1, keepdim=True),
                masked_mean((time_norm <= 0.33).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean(((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean((time_norm > 0.66).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean(torch.log1p(delta_t), mask, dim=1).unsqueeze(-1),
                masked_mean(close, mask, dim=1).unsqueeze(-1),
                masked_mean(close * var_change, mask, dim=1).unsqueeze(-1),
                masked_mean(pending, mask, dim=1).unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.cat([var_rate, stats], dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net(self.summarize_policy(batch))


class DoSchwarzianWarpConditioner(nn.Module):
    """Sampling-policy robust classifier using low-curvature warp-conditioned curves."""

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        num_classes: int,
        grid_size: int = 64,
        state_schwarzian_budget: float = 0.12,
    ):
        super().__init__()
        self.value_stem = EventValueStem(num_vars, hidden_dim)
        self.warp = MonotoneWarpNet(hidden_dim, grid_size)
        self.curve = WarpCanonicalCurve(num_vars, hidden_dim, grid_size)
        self.sidecar = WarpCurvatureSidecar(num_vars, hidden_dim, grid_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.state_schwarzian_budget = state_schwarzian_budget

    def encode(self, batch: dict) -> dict:
        value = self.value_stem(batch)
        warp = self.warp(value["pooled_state"])

        # Detach sidecar curvature from classification: it is diagnostic only.
        policy_curvature = self.sidecar(batch)
        high_curve = F.relu(warp["schwarzian"].abs() - policy_curvature.detach())
        gamma_event = interpolate_gamma_at_events(warp["gamma"], value["time_norm"])
        curve = self.curve(batch, value["event_state"], gamma_event)
        logits = self.classifier(curve["canonical_state"])
        return {**value, **warp, **curve, "policy_curvature": policy_curvature, "high_curve": high_curve, "logits": logits}

    def state_schwarzian_loss(self, out: dict) -> torch.Tensor:
        excess = F.relu(out["schwarzian"].abs() - self.state_schwarzian_budget)
        return excess.pow(2).mean()

    def minimal_warp_loss(self, out: dict) -> torch.Tensor:
        log_inc = out["log_increment"]
        slope_size = log_inc.abs().mean()
        slope_tv = (log_inc[:, 1:] - log_inc[:, :-1]).abs().mean()
        return slope_size + 0.5 * slope_tv

    def value_shape_loss(self, out: dict, batch: dict) -> torch.Tensor:
        # Reconstruct observed event values from the canonical curve by transposed soft assignment.
        var_id = batch["event_var_id"].clamp(0, out["curve_value"].size(-1) - 1)
        event_curve = torch.einsum("bnm,bmv->bnv", out["event_to_grid_weight"], out["curve_value"])
        pred_value = event_curve.gather(-1, var_id.unsqueeze(-1)).squeeze(-1)
        raw = F.smooth_l1_loss(pred_value, batch["event_value"], reduction="none")
        return (raw * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

    def sidecar_loss(self, batch: dict, factual: dict) -> torch.Tensor:
        probes = batch.get("policy_probe_bank", [])
        if not probes:
            return torch.zeros((), device=factual["logits"].device)

        losses = []
        factual_curve = factual["schwarzian"].abs().detach()
        for probe in probes:
            with torch.no_grad():
                probe_value = self.value_stem(probe)
                probe_warp = self.warp(probe_value["pooled_state"])
                target = F.relu(probe_warp["schwarzian"].abs() - factual_curve)
            pred = self.sidecar(probe)
            losses.append(F.smooth_l1_loss(pred, target))
        return torch.stack(losses).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_sch: float = 0.25,
        lambda_side: float = 0.25,
        lambda_min: float = 0.05,
        lambda_shape: float = 0.10,
    ) -> dict:
        labels = batch["labels"]
        out = self.encode(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)
        sch_loss = self.state_schwarzian_loss(out)
        side_loss = self.sidecar_loss(batch, out)
        min_loss = self.minimal_warp_loss(out)
        shape_loss = self.value_shape_loss(out, batch)

        total = (
            cls_loss
            + lambda_sch * sch_loss
            + lambda_side * side_loss
            + lambda_min * min_loss
            + lambda_shape * shape_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "state_schwarzian_budget_loss": sch_loss.detach(),
            "policy_curvature_sidecar_loss": side_loss.detach(),
            "minimal_warp_loss": min_loss.detach(),
            "value_shape_preservation_loss": shape_loss.detach(),
            "mean_policy_curvature": out["policy_curvature"].mean().detach(),
        }
```

## 4. Warp Probe Collator 草稿

```python
import torch


@torch.no_grad()
def build_schwarzian_warp_probes(batch: dict) -> dict:
    """Create policy warp probes for curvature-sidecar training.

    These probes are not contrastive positives and are not used for logits
    consistency. They only reveal which time-warp curvature is sampling-policy
    induced.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, pending=None):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        if pending is not None:
            out["value_pending"] = pending
        out.pop("policy_probe_bank", None)
        return out

    probes = []

    # 1. Routine-round probe: timestamps attach to coarse clinical rounds.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    probes.append(clone_with(value * mask, rounded_time, var_id, mask))

    # 2. Alarm-dense probe: early observations are thinned, late region stays dense.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_mask = torch.where(late > 0, mask, mask * alternating)
    probes.append(clone_with(value * alarm_mask, time, var_id, alarm_mask))

    # 3. Panel-split probe: near-synchronous cross-variable observations are separated.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    split_jitter = 0.04 * horizon * close * changed_var
    probes.append(clone_with(value * mask, time + split_jitter, var_id, mask))

    # 4. Pending-latency probe: event remains recorded, value evidence is delayed/low quality.
    pending = mask
    pending_value = torch.zeros_like(value)
    probes.append(clone_with(pending_value, time, var_id, mask, pending=pending))

    # 5. Variable-budget probe: selected variable groups become sparse.
    odd_var = (var_id % 2 == 1).to(mask.dtype)
    budget_mask = mask * (1.0 - 0.5 * odd_var)
    probes.append(clone_with(value * budget_mask, time, var_id, budget_mask))

    out = dict(batch)
    out["policy_probe_bank"] = probes
    return out
```

## 5. 实验切入点

1. **Policy shift 构造**
   - `DeepFRC-style phase shift`：底层病理形态相同但出现时间提前/滞后，验证低 Schwarzian 生物配准是否提升分类。
   - `alarm-warp shift`：训练中心告警后密集复测，测试中心改为固定查房，检查高曲率 sidecar 是否吸收该 shortcut。
   - `panel-split warp shift`：同步 panel 与异步拆单互换，检查配准器是否避免把 panel batching 当病理相位模板。
   - `pending-latency warp shift`：记录时间与值可用时间错位，检查 pending 是否制造虚假 warp curvature。
   - `cross-center registration shift`：MIMIC-IV / eICU / HiRID 风格采样 cadence 下比较 canonical curve 的稳定性。

2. **对比方法**
   - DeepFRC 原始 registration + classification。
   - DeepFRC 去掉 Fourier/contrastive 后的普通 monotone warp classifier。
   - 普通 irregular Transformer / CDE / STAR-Set / VP-GNN。
   - INPUTADAPTER / OpenTSLM-style semantic anchor baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DSPP、DCPD、DIPF、DRG-SFF、DPPC、DCOFF、DD-JEPA、DPSP、DNSA、DMWI、DCST、DAAA 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy worst AUROC / AUPRC。
   - state warp Schwarzian energy：进入分类器的 `gamma_state` 曲率能量，越低越少依赖局部采样日历。
   - policy curvature capture：sidecar 对 routine/alarm/panel/pending probe 曲率峰的解释度。
   - registration utility under true phase shift：真实生物相位错位下，低曲率配准是否保留性能。
   - warp shortcut score：错误预测是否伴随高 `|S(gamma)|` 且 sidecar 未能吸收。
   - value shape error：配准后再采样回观测点是否保留原始 value morphology。

4. **消融实验**
   - 去掉 `L_state_schwarzian_budget`，检查 warp 是否使用局部高曲率采样捷径。
   - 去掉 `WarpCurvatureSidecar`，检查高曲率是否被迫进入分类路径。
   - 去掉 `L_minimal_warp`，检查 warp 是否过度弯曲以适配训练标签。
   - 去掉 `L_value_shape_preservation`，检查 canonical curve 是否扭坏观测值语义。
   - 将 policy probes 替换成随机 mask，验证收益来自结构化时间曲率探针，而不是普通增强。
   - 直接把 Schwarzian curvature 拼入分类器作为反例，验证院内性能可能升高但跨政策退化。

## 6. 预期创新性

1. **从时间配准转向曲率调理**：吸收 DeepFRC 的 diffeomorphic registration，但不使用 Fourier/contrastive 主机制；关键创新是把高 Schwarzian warp curvature 解释为采样政策扭曲并从分类路径中隔离。
2. **从日历时间 / 病程时钟转向注册曲率**：DSWC 不学习 disease-progress poset、order ideal 或 RG scale fixed point；它只区分低曲率可接受配准和高曲率 policy warp。
3. **从反事实一致性转向曲率探针**：counterfactual sampling 不要求 logits、risk、representation 或 proof 一致，只训练 sidecar 解释采样政策会在哪里制造 warp curvature。
4. **从采样去偏转向保留合法相位差**：相比简单去掉时间或压平采样信息，DSWC 允许真实生物 phase variability 被平滑 warp 校正，同时阻止 routine/alarm/panel/pending 的局部高曲率日历捷径。
5. **与采样解耦/反事实干预框架低侵入兼容**：value process 变成 canonical curve classifier；sampling process 变成 curvature sidecar；反事实干预变成 warp probe bank。
6. **部署诊断直接可解释**：当模型失败时，可报告哪个时间窗、变量 panel 或 pending 区域产生异常 Schwarzian 曲率，判断预测是否依赖采样流程而非病理形态。

## 7. 一句话投稿卖点

**DSWC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“观测日历在时间配准中注入局部高 Schwarzian 曲率”的问题，通过 minimal-Schwarzian monotone warp、warp-conditioned canonical value curve、policy curvature sidecar 与反事实曲率 probe bank，让模型保留 DeepFRC 式合法生物相位配准能力，同时避免把医院 routine/alarm、panel split、pending latency 或跨中心 cadence 制造的高曲率时间扭曲误当成可迁移类别证据。**
