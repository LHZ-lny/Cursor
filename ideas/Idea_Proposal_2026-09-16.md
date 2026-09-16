# Title: Do-Renewal Exposure Meter：面向采样策略偏移的连续重生暴露计量分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代总结文件。
- 已读取当前工作区内全部历史 proposal 文件，覆盖 `ideas/Idea_Proposal_2026-06-12.md` 至 `ideas/Idea_Proposal_2026-09-14.md` 的 27 份落盘提案。
- 已读取自动化记忆 `MEMORIES.md` 及未完整落盘的历史 proposal 摘要，额外覆盖 `idea_2026-07-24.md`、`2026-07-25.md`、`2026-07-26.md`、`2026-07-27.md`、`2026-07-29.md`、`2026-07-31.md`、`2026-08-01.md`、`2026-08-04.md`、`2026-08-07.md`、`2026-08-10.md`、`2026-08-11.md`、`2026-08-21.md` 与 `idea_2026-09-15.md` 等机制摘要。
- 已读取最新 `paper_daily_2026-09-15.md`，并用 `paper_daily.md` 标题索引核验近期记录；重点纳入：
  - **Continuum Dropout for Neural Differential Equations**：用 alternating renewal process 在连续时间中执行 dropout，避免离散 dropout 破坏 NDE 轨迹，并提供 Monte Carlo 校准。
  - **Towards Self-Supervised Foundation Models for Critical Care Time Series**：用 Bi-Axial Transformer、动态 observation / forecasting window 和跨 ICU 数据集自监督预训练支持低资源死亡风险迁移。

### 历史核心机制黑名单

为避免思维重合，本轮明确避开以下历史主机制：

1. learnable reference points / adaptive time encoding、频域掩码修复、prototype 约束、missingness pattern 直接分类、简单 policy adversarial / IRM。
2. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
3. 生理流-采样算子交换子、value/policy graph 分离、policy residual sink。
4. protocol tax / additive evidence market、posterior quotient、density ratio / doubly robust、policy-simplex randomized smoothing。
5. reconstruction error cartography、VQ clauses、optional-stopping martingale、censored topology、policy gauge、syndrome code、knockoff calendar。
6. observability witness、evidential vacuity、information lattice、solver trace front-door、conformal sleeves、IV/control-function、Borda jury。
7. Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock、feasible hull、IRT-DIF、RG fixed point。
8. bitemporal curtain、clinical tomography、matched risk-set likelihood、Gaussian privacy cloak、CauKer orthogonal synthetic forge、JEPA 双辩裁判。
9. cross-representation PID prism、Noether semantic action、fast/slow meta workflow adapter、test-time workflow adaptation。
10. ORA collider factorization、explain-away responsibility gate、workflow-swap path-specific seal。
11. source atlas / policy-antipodal retrieval、frozen predictor input-adapter contract。
12. DeepFRC-inspired monotone registration、minimal-Schwarzian warp canonicalizer、warp-curvature sidecar。

本提案选择新的正交切入点：**不把采样政策估成 hazard，不要求多视图 logits 一致，不做随机平滑认证，也不把采样流程封印、检索、隐私化或元适配；而是把分类证据本身改写为连续时间 renewal reward rate。采样政策可以改变观测事件数量和密度，但不能通过“多测几次”线性放大分类证据，因为每一份 reward 都必须除以模型内部 alternating-renewal clock 产生的活跃暴露量。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列中的 sampling-policy shortcut 很多时候不是某个单独 mask bit，而是 **暴露量错觉**：

- ICU 训练中心在报警后密集复测，模型把更多 late-window token 累加成更高死亡风险；
- 可穿戴设备某个固件在夜间降低 duty cycle，模型把长间隔误解成低活动或稳定状态；
- 跨医院数据集中，同一病程在一个中心被记录成细粒度 burst，在另一个中心被记录成粗粒度 routine round，普通 pooling / attention 会让事件数、局部密度和窗口长度直接影响 logit 大小。

最新 `paper_daily_2026-09-15.md` 中的 Continuum Dropout 给了一个新工具：dropout 不必是离散层上的 Bernoulli mask，而可以是连续时间 alternating renewal process。它的启发不是“再做一次不确定性估计”，而是：**模型内部可以拥有一只与外部采样日历不同步的随机活跃时钟**。如果分类证据只在这只内部时钟活跃时累计，并且最终按活跃暴露量归一化，那么外部采样政策改变事件密度时，不能再简单地把 token 数量转化为类别分数。

同时，ICU foundation model 工作说明预训练阶段应利用动态 observation / forecasting window，但普通 masked forecasting 会同时学习“未来会被测到什么”和“未来病理状态是什么”。本提案只保留它的动态窗口思想：用窗口内的观测值预测 **病理值摘要 / reward-rate target**，而不预测未来观测流程、next mark 或采样密度。

**Do-Renewal Exposure Meter (DREM)** 的核心直觉是：

> 把不规则时序分类从“累加观测证据”改成“估计单位内部活跃暴露下的病理 reward rate”。真正的病理状态会改变 reward rate；医院/设备采样政策主要改变外部事件暴露量。只要 reward 和 exposure 同时经过同一只 renewal clock，采样密集 burst 就不能凭事件数直接放大分类 margin。

这与当前“采样解耦/反事实干预”框架自然结合：

- value process 产生 event reward；
- sampling process 不进入分类头，只生成 policy stress 和反事实采样 recipe，用来审计 reward-exposure 计量是否守恒；
- counterfactual intervention 不做一致性、不做 risk variance、不做 certified smoothing，而是生成不同外部采样暴露下的 renewal-balance 残差；
- classifier 只读取 renewal-normalized reward rate。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从事件累加改为 Renewal Exposure Meter

给定事件流 `e_i = (x_i, t_i, d_i, q_i)`，先用 value encoder 生成病理 reward token：

```text
r_i = RewardStem(value_i, variable_i, pathology_bin_i, delta_t_i, quality_i)
```

同时，模型内部采样 `M` 条 alternating-renewal clocks：

```text
G_m(t) in {0, 1}
active duration ~ Exp(lambda_on)
inactive duration ~ Exp(lambda_off)
```

这里的 `G_m(t)` 是 **模型内部暴露时钟**，不是外部 observation process 的 hazard，也不是 policy simplex smoothing。它只决定哪些时间片段允许 reward 被计入分类证据。

对每条 clock，计算：

```text
reward_sum_m   = sum_i G_m(t_i) * r_i * delta_t_i
exposure_sum_m = sum_i G_m(t_i) * e_i^+ * delta_t_i
z_m            = reward_sum_m / (exposure_sum_m + eps)
z_rate         = mean_m z_m
logits         = Classifier(z_rate)
```

其中 `e_i^+ = softplus(ExposureHead(r_i))` 是 value-derived exposure unit。直觉上，采样政策若增加了相似事件，只会同时增加 reward numerator 和 exposure denominator；只有观测值语义真的改变时，单位暴露 reward rate 才会改变。

关键区别：

- 不估计外部采样危险率；
- 不把 policy summary、mask pattern、workflow metadata 输入分类器；
- 不要求不同反事实视图 logits 一致；
- 不做 Monte Carlo prediction smoothing 或认证半径；
- 不输出 evidential uncertainty / conformal set / privacy noise。

### 2.2 改 Loss：从不变性约束转向 Renewal-Reward Balance

总目标：

```text
L = L_cls
  + lambda_bal * L_renewal_balance
  + lambda_cf  * L_counterfactual_exposure_balance
  + lambda_ssl * L_window_pathology_pretrain
  + lambda_exp * L_exposure_duty_calibration
```

#### A. Renewal-Rate Classification `L_cls`

事实观测下只用 `z_rate` 分类：

```text
L_cls = CE(Classifier(z_rate), y)
```

如果训练医院有密集复测，普通 attention 可能把更多 token 变成更大 logit；DREM 的 logit 来自 reward rate，因此天然抑制事件数捷径。

#### B. Renewal Balance Loss `L_renewal_balance`

根据 renewal-reward theorem，同一条轨迹在多条内部 renewal clock 下，`reward_sum - rate * exposure_sum` 的 clock 条件均值应接近 0：

```text
B_m = reward_sum_m - stopgrad(z_rate) * exposure_sum_m
L_renewal_balance = || mean_m B_m ||_2^2
```

这不是多视图一致性：不同 clocks 的 `z_m` 可以不同，甚至某些 clock 暂时看不到关键事件。该损失只要求“单位暴露 reward rate”是一个稳定可估计量，而不是采样事件数的函数。

#### C. Counterfactual Exposure Balance `L_counterfactual_exposure_balance`

反事实采样模块生成外部 policy views，例如 routine-round、alarm-dense、panel-split、variable-budget、pending-latency。DREM 不比较这些视图的 logits，而是在每个外部政策下重新计算 renewal residual：

```text
B_m^policy = reward_sum_m^policy - stopgrad(z_rate_factual) * exposure_sum_m^policy
L_counterfactual_exposure_balance = mean_policy || mean_m B_m^policy ||_2^2
```

若某个采样政策只是增加重复观测或改变事件密度，残差应被 exposure denominator 吸收；若它真的改变可观测病理值，残差可以显现为诊断信号，但不会通过事件数直接推动分类。

#### D. Dynamic Window Pathology Pretraining `L_window_pathology_pretrain`

借鉴 ICU foundation model 的动态 observation / forecasting window，但目标不预测 future observation process。Dataloader 随机抽取 context window 和 value target window：

```text
z_ctx = DREM(context_window)
target_pathology_hist = histogram(pathology_bins in target_value_window)
L_window_pathology_pretrain = CE(PathologyHead(z_ctx), target_pathology_hist)
```

这让 reward rate 学到跨窗口病理语义，而不是学习哪个变量未来会被测、何时被测或 pending 率。

#### E. Exposure Duty Calibration `L_exposure_duty_calibration`

Continuum Dropout 的 alternating renewal process 有理论 duty cycle：

```text
duty = lambda_off / (lambda_on + lambda_off)
```

DREM 约束采样到的内部活跃暴露与理论 duty 匹配：

```text
L_exposure_duty = SmoothL1(mean_m exposure_sum_m,
                           duty * total_value_exposure)
```

这防止 renewal clock 退化成永远 active 或永远 inactive，也防止模型把采样密集窗口偷偷设成更高内部 duty。

### 2.3 改 Dataloader：返回 Renewal Exposure Audit Batch

新增 `RenewalExposureCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_quality`。
2. `pathology_bin_id` 或软分箱 `bin_prob`，只作为 value 语义锚。
3. `context_window_mask` 与 `target_window_mask`，用于动态窗口病理预训练。
4. `policy_exposure_bank`：
   - `routine_round_exposure`：时间吸附到固定查房窗口；
   - `alarm_dense_exposure`：报警后密集复测或其稀疏反事实；
   - `panel_split_exposure`：同步 panel 拆成异步事件；
   - `variable_budget_exposure`：变量组采样预算变化；
   - `pending_latency_exposure`：事件保留但值质量降低。
5. `renewal_clock_params`：内部 clock 的 `lambda_on / lambda_off` 初值或 batch-level schedule。

这些 views 不是 contrastive positives，不做 logits consistency，不做风险方差，也不用于 policy classifier；它们只审计 reward-exposure 计量是否把外部采样暴露变化吸收掉。

### 2.4 推理阶段

给定测试样本：

1. value encoder 生成 event rewards；
2. 内部 alternating-renewal clock 采样少量轨迹或使用 duty-cycle 均值近似；
3. 输出 `z_rate` 与分类概率；
4. 同时报告：
   - `renewal_balance_residual`：单位暴露计量是否稳定；
   - `effective_active_exposure`：预测实际使用了多少内部活跃暴露；
   - `policy_exposure_alarm`：若 routine / alarm / panel / pending audit 下 residual 异常高，提示该预测可能受采样暴露偏移影响；
   - `reward_per_exposure_map`：哪些变量/时间窗贡献了真正高 reward rate，而非高事件数。

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


def event_delta_time(event_time: torch.Tensor) -> torch.Tensor:
    delta_t = torch.zeros_like(event_time)
    delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
    if event_time.size(1) > 1:
        delta_t[:, 0] = delta_t[:, 1]
    return delta_t


class PathologyRewardStem(nn.Module):
    """Encode value-bearing events into reward and exposure units."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.5, 2.5, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_bins + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reward_head = nn.Linear(hidden_dim, hidden_dim)
        self.exposure_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * mask.unsqueeze(-1)

        delta_t = event_delta_time(time)
        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon

        event_x = torch.cat(
            [
                self.var_embed(var_id),
                bin_prob,
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * mask.unsqueeze(-1)
        reward = self.reward_head(event_h) * mask.unsqueeze(-1)
        exposure_unit = (self.exposure_head(event_h).squeeze(-1) + 1e-4) * mask
        return {
            "event_h": event_h,
            "reward": reward,
            "exposure_unit": exposure_unit,
            "bin_prob": bin_prob,
            "delta_t": delta_t,
            "event_mask": mask,
        }


class AlternatingRenewalClock(nn.Module):
    """Internal continuous-time active/inactive clock for exposure accounting."""

    def __init__(self, init_on_rate: float = 1.5, init_off_rate: float = 1.0):
        super().__init__()
        self.log_on_rate = nn.Parameter(torch.log(torch.tensor(init_on_rate)))
        self.log_off_rate = nn.Parameter(torch.log(torch.tensor(init_off_rate)))

    @property
    def on_rate(self) -> torch.Tensor:
        return F.softplus(self.log_on_rate) + 1e-4

    @property
    def off_rate(self) -> torch.Tensor:
        return F.softplus(self.log_off_rate) + 1e-4

    def duty_cycle(self) -> torch.Tensor:
        # Fraction of time expected to be active.
        return self.off_rate / (self.on_rate + self.off_rate)

    def forward(self, event_time: torch.Tensor, event_mask: torch.Tensor, num_clocks: int) -> torch.Tensor:
        """Sample active gates at event times.

        This loop is intentionally explicit in the draft: production code can
        vectorize renewal jumps, but the semantics are easier to audit here.
        """

        bsz, num_events = event_time.shape
        device = event_time.device
        active = torch.zeros(bsz, num_clocks, num_events, device=device, dtype=event_time.dtype)

        duty = self.duty_cycle().detach().clamp(0.05, 0.95)
        state = torch.bernoulli(torch.full((bsz, num_clocks), duty, device=device, dtype=event_time.dtype))
        next_switch = torch.empty_like(state).exponential_(1.0)
        next_switch = next_switch / torch.where(state > 0, self.on_rate.detach(), self.off_rate.detach())

        last_t = torch.zeros(bsz, num_clocks, device=device, dtype=event_time.dtype)
        for idx in range(num_events):
            t = event_time[:, idx : idx + 1].expand(-1, num_clocks)
            valid = event_mask[:, idx : idx + 1].expand(-1, num_clocks) > 0
            elapsed = (t - last_t).clamp_min(0.0)
            next_switch = next_switch - elapsed

            switched = (next_switch <= 0.0) & valid
            if switched.any():
                state = torch.where(switched, 1.0 - state, state)
                rate = torch.where(state > 0, self.on_rate.detach(), self.off_rate.detach())
                fresh = torch.empty_like(next_switch).exponential_(1.0) / rate
                next_switch = torch.where(switched, fresh, next_switch)

            active[:, :, idx] = state * valid.to(event_time.dtype)
            last_t = torch.where(valid, t, last_t)
        return active


class RenewalExposureMeter(nn.Module):
    """Convert event rewards into renewal-normalized reward rates."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int, num_classes: int, num_clocks: int = 8):
        super().__init__()
        self.stem = PathologyRewardStem(num_vars, num_bins, hidden_dim)
        self.clock = AlternatingRenewalClock()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.pathology_head = nn.Linear(hidden_dim, num_bins)
        self.num_clocks = num_clocks
        self.num_bins = num_bins

    def meter(self, batch: dict, num_clocks: int | None = None) -> dict:
        stem = self.stem(batch)
        clocks = num_clocks or self.num_clocks
        active = self.clock(batch["event_time"], stem["event_mask"], clocks)
        dt = stem["delta_t"] * stem["event_mask"]

        reward_weight = active.unsqueeze(-1) * dt[:, None, :, None]
        reward_sum = (reward_weight * stem["reward"][:, None]).sum(dim=2)

        exposure_weight = active * dt[:, None, :] * stem["exposure_unit"][:, None]
        exposure_sum = exposure_weight.sum(dim=2).clamp_min(1e-4)

        rate_by_clock = reward_sum / exposure_sum.unsqueeze(-1)
        reward_rate = rate_by_clock.mean(dim=1)
        logits = self.classifier(reward_rate)

        total_exposure = (dt * stem["exposure_unit"]).sum(dim=1).clamp_min(1e-4)
        return {
            **stem,
            "active": active,
            "reward_sum": reward_sum,
            "exposure_sum": exposure_sum,
            "rate_by_clock": rate_by_clock,
            "reward_rate": reward_rate,
            "logits": logits,
            "total_exposure": total_exposure,
        }

    def renewal_balance_loss(self, out: dict) -> torch.Tensor:
        rate = out["reward_rate"].detach()
        residual = out["reward_sum"] - out["exposure_sum"].unsqueeze(-1) * rate[:, None, :]
        return residual.mean(dim=1).pow(2).mean()

    def exposure_duty_loss(self, out: dict) -> torch.Tensor:
        expected = self.clock.duty_cycle() * out["total_exposure"]
        observed = out["exposure_sum"].mean(dim=1)
        return F.smooth_l1_loss(observed, expected.detach())

    def window_pathology_loss(self, batch: dict, out: dict) -> torch.Tensor:
        if "target_window_mask" not in batch:
            return torch.zeros((), device=out["logits"].device)

        target_mask = batch["target_window_mask"] * out["event_mask"]
        target_hist = (out["bin_prob"] * target_mask.unsqueeze(-1)).sum(dim=1)
        target = target_hist.argmax(dim=-1).clamp(0, self.num_bins - 1)
        pred = self.pathology_head(out["reward_rate"])
        return F.cross_entropy(pred, target)

    def counterfactual_exposure_balance_loss(self, batch: dict, factual: dict) -> torch.Tensor:
        views = batch.get("policy_exposure_bank", [])
        if not views:
            return torch.zeros((), device=factual["logits"].device)

        factual_rate = factual["reward_rate"].detach()
        losses = []
        for view in views:
            cf = self.meter(view)
            residual = cf["reward_sum"] - cf["exposure_sum"].unsqueeze(-1) * factual_rate[:, None, :]
            losses.append(residual.mean(dim=1).pow(2).mean())
        return torch.stack(losses).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_bal: float = 0.25,
        lambda_cf: float = 0.30,
        lambda_ssl: float = 0.15,
        lambda_exp: float = 0.08,
    ) -> dict:
        out = self.meter(batch)
        labels = batch["labels"]
        cls_loss = F.cross_entropy(out["logits"], labels)
        bal_loss = self.renewal_balance_loss(out)
        cf_loss = self.counterfactual_exposure_balance_loss(batch, out)
        ssl_loss = self.window_pathology_loss(batch, out)
        exp_loss = self.exposure_duty_loss(out)

        total = cls_loss + lambda_bal * bal_loss + lambda_cf * cf_loss + lambda_ssl * ssl_loss + lambda_exp * exp_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "renewal_balance_loss": bal_loss.detach(),
            "counterfactual_exposure_balance_loss": cf_loss.detach(),
            "window_pathology_pretrain_loss": ssl_loss.detach(),
            "exposure_duty_loss": exp_loss.detach(),
            "mean_active_exposure": out["exposure_sum"].mean().detach(),
        }
```

## 4. Renewal Exposure Collator 草稿

```python
import torch


@torch.no_grad()
def build_renewal_exposure_batch(batch: dict) -> dict:
    """Build dynamic windows and policy exposure audits for DREM.

    The policy views are not contrastive positives and are not used for logits
    consistency. They only test whether reward per internal exposure is stable.
    """

    out = dict(batch)
    out.update(build_dynamic_windows(batch))
    out["policy_exposure_bank"] = build_policy_exposure_views(batch)
    return out


@torch.no_grad()
def build_dynamic_windows(batch: dict, target_ratio: float = 0.30) -> dict:
    mask = batch["event_mask"]
    rand = torch.rand_like(mask)
    target = ((rand < target_ratio).to(mask.dtype) * mask).clamp(0.0, 1.0)
    context = (mask - target).clamp_min(0.0)

    empty = context.sum(dim=1) == 0
    if empty.any():
        first = mask[empty].argmax(dim=1)
        context[empty, first] = 1.0
        target[empty, first] = 0.0
    return {"context_window_mask": context, "target_window_mask": target}


@torch.no_grad()
def build_policy_exposure_views(batch: dict) -> list[dict]:
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    quality = batch.get("measurement_quality", torch.ones_like(value))
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, new_quality):
        view = dict(batch)
        view["event_value"] = new_value
        view["event_time"] = new_time
        view["event_var_id"] = new_var
        view["event_mask"] = new_mask
        view["measurement_quality"] = new_quality
        view.pop("policy_exposure_bank", None)
        return view

    views = []

    # 1. Routine-round exposure: external calendar changes, values stay factual.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    views.append(clone_with(value * mask, rounded_time, var_id, mask, quality))

    # 2. Alarm-dense / sparse exposure: thin early routine events, keep late events.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_mask = torch.where(late > 0, mask, mask * alternating)
    views.append(clone_with(value * alarm_mask, time, var_id, alarm_mask, quality * alarm_mask))

    # 3. Panel split exposure: reduce near-synchronous cross-variable events.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_mask = mask * (1.0 - 0.5 * close * changed_var)
    views.append(clone_with(value * panel_mask, time, var_id, panel_mask, quality * panel_mask))

    # 4. Variable-budget exposure: keep only a small budget per variable.
    budget_mask = torch.zeros_like(mask)
    for var in torch.unique(var_id[mask > 0]).tolist():
        hit = ((var_id == int(var)) & (mask > 0)).to(mask.dtype)
        rank = hit.cumsum(dim=1)
        budget_mask = torch.maximum(budget_mask, (rank <= 2).to(mask.dtype) * hit)
    views.append(clone_with(value * budget_mask, time, var_id, budget_mask, quality * budget_mask))

    # 5. Pending-latency exposure: administration visible, value quality lower.
    pending_quality = quality * 0.20
    views.append(clone_with(value * mask, time, var_id, mask, pending_quality))
    return views
```

## 5. 实验切入点

1. **Policy shift 构造**
   - `exposure-density reversal`：训练中 alarm-dense 与高风险相关，测试中同样 dense follow-up 出现在普通患者。
   - `duty-cycle shift`：可穿戴或 ICU 设备的 active / inactive 采样节律改变。
   - `panel-exposure shift`：训练中心同步 panel 带来事件数暴增，测试中心拆成异步事件。
   - `variable-budget shift`：测试中心对某些变量只保留有限观测。
   - `pending-latency shift`：保留施测事件但值质量下降或延迟返回。

2. **对比方法**
   - 普通 irregular Transformer / GRU / CDE / BAT。
   - Continuum Dropout 原始 NDE 正则化。
   - ICU self-supervised foundation model / YAIB-style BAT pretraining。
   - mask dropout / random missing augmentation。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DSPP、DCPD、DIPF、DRG-SFF、DPPC、DCOFF、DD-JEPA、DPSP、DNSA、DMWI、DCST、DAAA、Schwarzian Warp 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy worst AUROC / AUPRC。
   - exposure amplification score：同一病理值被重复观测时 logit 是否异常线性上升。
   - renewal balance residual：`reward_sum - rate * exposure_sum` 在 policy views 下的均值偏移。
   - effective active exposure：预测依赖的内部活跃暴露量是否合理。
   - reward-per-exposure stability：不同采样密度下单位暴露 reward rate 是否稳定。

4. **消融实验**
   - 去掉 renewal exposure denominator，退化为普通 reward pooling，检查 dense sampling shortcut 是否回归。
   - 去掉 `L_counterfactual_exposure_balance`，检查外部 policy views 下 residual 是否偏移。
   - 用普通 Bernoulli dropout 替代 alternating-renewal clock，验证连续时间 active / inactive 过程的重要性。
   - 让 policy summary 进入 classifier 作为反例，验证院内性能可能上升但跨政策退化。
   - 将动态窗口预训练目标改成 next-mark / future-observed-value，验证预测 observation process 会重新吸收采样政策。
   - 扫描 renewal clock 数量，评估推理开销与 reward-rate 稳定性的 trade-off。

## 6. 预期创新性

1. **从采样去偏转向暴露计量**：历史方案多在表示、后验、图、证明、校准或检索层修补采样偏移；DREM 直接改变证据累计单位，让分类器读取单位内部活跃暴露下的 reward rate。
2. **从 Continuum Dropout 正则化转向 renewal reward accounting**：吸收 alternating-renewal dropout 的连续时间语义，但不把它当普通不确定性工具，而是让它成为病理证据的内部暴露分母。
3. **从 ICU self-supervised forecasting 转向 pathology-only window target**：保留动态 observation / forecasting window 的预训练优势，但不预测未来 observation process，避免 foundation model 在预训练阶段学习医院采样流程。
4. **从多视图一致转向 renewal-balance residual**：反事实采样 view 只用于检查 reward-exposure 平衡，不要求 logits、representation 或风险相同。
5. **部署诊断直接指向事件数捷径**：若某个预测主要来自采样密集 burst，exposure amplification score 和 renewal residual 会升高；若来自真实病理值，reward-per-exposure map 会在不同政策下保持稳定。

## 7. 一句话投稿卖点

**DREM 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“外部采样政策改变了观测暴露量，从而错误放大分类证据”的问题，通过 alternating-renewal internal clocks、reward-per-exposure readout、renewal-balance residual 与 pathology-only dynamic window pretraining，让分类器依赖单位内部活跃暴露下的病理 reward rate，而不是依赖训练医院或设备制造的高频复测、panel 同步、变量预算或 pending-latency 事件数捷径。**
