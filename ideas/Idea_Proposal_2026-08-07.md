# Title: Do-Facet Feasibility Hull：面向采样策略偏移的反事实病理可行域凸包分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已读取自动化记忆 `MEMORIES.md`，其中包含 2026-06-12 至 2026-08-06 的历史机制黑名单。
- 已读取近期历史提案原文与索引：
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
- 已读取近期论文记录 `paper_daily.md` 与 `paper_daily_2026-08-02.md`，重点纳入：
  - **PULSE**：HiRID / MIMIC-IV / eICU 跨中心 ICU 分类基准，强调真实 sampling-policy shift 来自医院流程、变量定义、告警阈值与记录习惯。
  - **TCF**：Pathology-Focused Binning、Dual-Calendar RoPE、Time-Conditioned Foreseeing，强调 EHR 事件的病理数值语义与未来时间条件预测。

### 历史核心机制黑名单

本轮避免把以下机制作为主创新：自适应时间编码、频域掩码、missingness classifier、原型分类、简单策略对抗、危险率 score 零空间、图交换子、证据市场、后验商、误差地图、随机平滑认证、采样测度密度比、停时鞅、审查拓扑、gauge transport、policy-only negative film、syndrome code、knockoff-FDR、observability witness、evidential vacuity、信息格次模边际、solver-trace 前门、measurement-action bisimulation、signature renormalization、热力学自由能、Sklar copula stripping、triage queue、Sinkhorn canonicalization、MDL episode transducer、causal sheaf、trigger hysteresis、control barrier、regret escrow、principal stratum、conformal sleeves、IV residualization、policy jury、Krylov annihilator、Nystrom volume basis、tropical support routes、fixed viva question bank、temporal sequent proof、disease-progress poset clock。

本提案选择新的正交切入点：**不把鲁棒性建成表示一致、集合校准、证明规则、排序投票、病程偏序或策略图分解，而是把每条不规则观测序列翻译成一个 latent clinical state 的“病理可行域凸包”。采样政策只决定哪些半空间约束被观测到；分类器读取的是整个可行域上的保守类别边际，而不是某个观测日历或某条事件路径。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

PULSE 提醒我们：跨中心退化往往不是因为模型完全看不到病理信号，而是因为不同 ICU 用不同方式“切片”同一病程。某中心早测乳酸，另一中心晚测；某中心同步 panel，另一中心异步拆开；某中心告警后密集记录，另一中心按固定查房窗口记录。若分类器直接消费 token 序列、日历时间、未来事件分布或 co-observation pattern，就容易把这些中心政策当成类别证据。

TCF 的 Pathology-Focused Binning 很适合解决另一个问题：EHR 数值不是普通连续变量，异常区间、临界区间和变量特异阈值本身带有病理语义。但 TCF 的 future-event foreseeing 仍可能混合 patient state 与 hospital observation process：未来某个化验出现，可能是病程发展，也可能是医院 A 的下单习惯。

**Do-Facet Feasibility Hull (DFFH)** 的核心判断是：

> 每个被观测到的病理 bin 不应被直接当成分类 token，而应被看作对潜在临床状态 `z` 的一个软约束：`a_i^T z <= b_i` 或 `a_i^T z >= b_i`。不同采样政策会改变约束数量、约束紧度和约束可见性，但如果类别判断是真正由病理状态支持的，那么在所有可行状态上都应有足够的类别边际。

换言之，DFFH 不问“这个医院有没有测某项变量”，也不问“不同采样视图 logits 是否一致”。它问：

1. 当前观测值允许哪些潜在病理状态仍然可行？
2. 在这个可行域中的最坏状态上，真实类别是否仍能击败竞争类别？
3. 反事实采样删改后，新增或消失的是病理约束，还是只由日历流程伪造的约束？

这与当前“采样解耦/反事实干预”框架自然匹配：

- value process 产生病理 bin，并把 bin 翻译成 clinical-state half-space facets；
- sampling process 只决定哪些 facets 被看到、哪些 facets 因 value-pending / panel-split / routine-round 变得松弛；
- counterfactual intervention 生成 policy-specific facet subsets；
- classifier 从可行域的 Chebyshev center、半径和 worst-case margin 做决策。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从事件表示改为 Pathology Facet Lift

现有 irregular encoder 可保留作为 value backbone，但在分类读出前加入 **Facet Lift**：

1. **Pathology-Focused Facet Generator**
   - 吸收 TCF 的 Pathology-Focused Binning，把每个观测值 `(x_i, v_i, t_i)` 映射到变量特异病理 bin。
   - 每个 bin 不再生成一个 token embedding，而是生成潜在临床状态空间中的半空间约束：

```text
normal_i = f_a(variable_i, pathology_bin_i, value_strength_i)
bound_i  = f_b(variable_i, pathology_bin_i)
slack_i  = f_s(measurement_quality_i, value_pending_i)
facet_i: normal_i^T z <= bound_i + slack_i
```

2. **Facet Confidence**
   - confidence 来自测量质量、值是否返回、bin posterior entropy。
   - `event_time` 只用于估计 policy stress 和反事实 facet dropout，不直接进入分类主路径。

3. **Soft Feasibility Hull Projector**
   - 用 unrolled projected optimization 求一个可微 Chebyshev-like center `c`：

```text
c = argmin_z mean_i confidence_i * relu(normal_i^T z - bound_i - slack_i)^2 + alpha * ||z||^2
radius = average positive slack to active facets
```

   - `c` 表示当前观测约束共同支持的临床状态中心。
   - `radius` 表示由于采样稀疏、value-pending 或互相矛盾观测导致的可行域不确定性。

### 2.2 改 Loss：从一致性/证明/保形转向 Feasible-Hull Robust Margin

总目标：

```text
L = L_hull_cls
  + lambda_margin * L_worst_case_margin
  + lambda_facet  * L_calendar_facet_sobriety
  + lambda_radius * L_radius_honesty
  + lambda_bin    * L_tcf_bin_grounding
```

#### A. Hull Classification Loss `L_hull_cls`

分类器不直接读事件序列，而读可行域中心与半径：

```text
logits = W c + b - radius * ||W||_2
L_hull_cls = CE(logits, y)
```

`radius * ||W||_2` 是保守惩罚：若采样政策导致可行域很宽，模型不能给出虚假的高置信预测。

#### B. Worst-Case Margin Loss `L_worst_case_margin`

对真实类 `y` 和竞争类 `k`，要求整个可行域上都有类别边际：

```text
m_yk = (w_y - w_k)^T c + (b_y - b_k)
robust_m_yk = m_yk - radius * ||w_y - w_k||_2
L_margin = mean_k relu(gamma - robust_m_yk)^2
```

这不是 conformal prediction set：DFFH 不校准覆盖率，也不输出预测集合；它在 latent clinical state 可行域上做几何鲁棒分类。

#### C. Calendar Facet Sobriety `L_calendar_facet_sobriety`

反事实采样模块生成 `value_pending`、`routine_round`、`panel_debatch`、`alarm_burst` 等 policy views。DFFH 不要求这些 views 的 logits 一致，而只审查：

> 纯日历变化不能创造更紧的病理半空间。

若某个 view 只改变时间、panel 分组或 value-pending，而没有新增真实 pathology value，则可行域不应变得更窄：

```text
tightness = mean_i confidence_i / (slack_i + eps)
L_calendar_facet =
  mean_cf relu(tightness_cf - tightness_factual - eps)^2
```

这能阻止模型把“某医院联测 panel”翻译成一个虚假的高置信病理 facet。

#### D. Radius Honesty `L_radius_honesty`

当反事实采样删除关键值、制造 value-pending 或显著减少变量覆盖时，可行域半径应变大，而不是由模型偷偷维持小半径：

```text
coverage_drop = max(0, coverage_factual - coverage_cf)
L_radius = relu(radius_factual + eta * coverage_drop - radius_cf)^2
```

这与 evidential vacuity 不同：没有 Dirichlet 主观逻辑，也不把类别不确定性建成证据质量；半径是 latent clinical state 可行域的几何宽度。

#### E. TCF Bin Grounding `L_tcf_bin_grounding`

保留 TCF 的病理分箱语义，但把 foreseeing 改成 facet grounding：

- 对已观测值，facet generator 必须能重构变量特异 pathology bin；
- 对未来时间 query，TCF-style foreseeing 只生成“可能新增哪些病理约束”，不直接生成分类证据。

```text
L_bin = CE(predicted_bin_from_facet, pathology_bin)
```

### 2.3 改 Dataloader：返回 Counterfactual Facet Views

新增 `FacetHullCollator`，每个 batch 返回：

1. 原始事件：`event_value`、`event_time`、`event_var_id`、`event_mask`。
2. TCF-style `pathology_bin_id` 或 soft bin distribution。
3. `measurement_quality` 与 `value_pending_mask`。
4. `facet_view_bank`：
   - `routine_round`: 只改变日历时间；
   - `panel_debatch`: 拆开同步 panel，但不新增 value；
   - `value_pending`: 观测事件可见但 value/bin 不可用；
   - `alarm_burst_thin`: 告警后密集观测被稀释。
5. `coverage_drop` 与 `calendar_stress`：
   - 只用于 facet sobriety 与 radius honesty；
   - 不进入分类 head。

### 2.4 与 PULSE / TCF 的结合方式

- **来自 PULSE**：把 HiRID / MIMIC-IV / eICU 的跨中心差异转成 facet coverage、facet tightness、hull radius 和 robust margin 的差异，报告 worst-center robust margin。
- **来自 TCF**：使用 Pathology-Focused Binning 提供病理语义，但不把 future event likelihood 直接作为分类证据；未来预测只用于估计可能新增的 pathology facets。
- **与采样解耦/反事实干预结合**：sampling branch 不做对抗、不做一致性、不做 jury、不做 proof audit；它只生成反事实 facet visibility，从而检查可行域是不是被纯日历流程不合理地收紧。

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


class PathologyFacetLift(nn.Module):
    """Translate pathology-focused bins into soft half-space facets."""

    def __init__(self, num_vars: int, num_bins: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.state_dim = state_dim
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_embed = nn.Embedding(num_bins, hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.normal_head = nn.Linear(hidden_dim, state_dim)
        self.bound_head = nn.Linear(hidden_dim, 1)
        self.slack_head = nn.Linear(hidden_dim, 1)
        self.conf_head = nn.Linear(hidden_dim, 1)
        self.bin_reconstruct = nn.Linear(state_dim, num_bins)

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        bin_id = batch["pathology_bin_id"].clamp(0, self.num_bins - 1)
        quality = batch.get("measurement_quality", torch.ones_like(value))
        pending = batch.get("value_pending_mask", torch.zeros_like(value))

        var_h = self.var_embed(var_id)
        bin_h = self.bin_embed(bin_id)
        x = torch.cat(
            [
                var_h,
                bin_h,
                value.unsqueeze(-1),
                quality.unsqueeze(-1),
                pending.unsqueeze(-1),
            ],
            dim=-1,
        )
        h = self.value_proj(x)

        normal = F.normalize(self.normal_head(h), dim=-1)
        bound = self.bound_head(h).squeeze(-1)

        # Pending values and low-quality measurements become loose constraints.
        learned_slack = F.softplus(self.slack_head(h).squeeze(-1))
        slack = learned_slack + 2.0 * pending + (1.0 - quality).clamp_min(0.0)

        confidence = torch.sigmoid(self.conf_head(h).squeeze(-1))
        confidence = confidence * quality * (1.0 - 0.8 * pending) * mask

        return {
            "facet_h": h * mask.unsqueeze(-1),
            "normal": normal * mask.unsqueeze(-1),
            "bound": bound * mask,
            "slack": slack * mask,
            "confidence": confidence,
            "event_mask": mask,
        }


class FeasibleHullProjector(nn.Module):
    """Unrolled differentiable projection to a soft feasible-hull center."""

    def __init__(self, state_dim: int, steps: int = 8, step_size: float = 0.20, l2: float = 1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.steps = steps
        self.step_size = step_size
        self.l2 = l2

    def forward(self, facets: dict) -> dict:
        normal = facets["normal"]
        bound = facets["bound"]
        slack = facets["slack"]
        conf = facets["confidence"]
        batch_size = normal.size(0)
        z = torch.zeros(batch_size, self.state_dim, device=normal.device, dtype=normal.dtype)

        for _ in range(self.steps):
            violation = (normal * z[:, None, :]).sum(dim=-1) - bound - slack
            positive = F.relu(violation)
            grad = (2.0 * conf * positive).unsqueeze(-1) * normal
            grad = grad.sum(dim=1) / conf.sum(dim=1, keepdim=True).clamp_min(1.0)
            z = z - self.step_size * (grad + self.l2 * z)

        violation = (normal * z[:, None, :]).sum(dim=-1) - bound
        active = torch.sigmoid(8.0 * (violation - slack))
        weighted_slack = conf * active * slack
        radius = weighted_slack.sum(dim=1) / (conf * active).sum(dim=1).clamp_min(1.0)
        radius = radius + 0.05 / conf.sum(dim=1).clamp_min(1.0).sqrt()

        tightness = (conf / (slack + 1e-4)).sum(dim=1) / facets["event_mask"].sum(dim=1).clamp_min(1.0)
        return {"center": z, "radius": radius, "tightness": tightness}


class RobustHullClassifier(nn.Module):
    """Classify with conservative logits over a latent feasible hull."""

    def __init__(self, state_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, state_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, center: torch.Tensor, radius: torch.Tensor) -> dict:
        nominal = center @ self.weight.t() + self.bias
        conservative = nominal - radius[:, None] * self.weight.norm(dim=-1)[None, :]
        return {"logits": conservative, "nominal_logits": nominal}

    def worst_case_margin_loss(
        self,
        center: torch.Tensor,
        radius: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.50,
    ) -> torch.Tensor:
        true_w = self.weight[labels]
        true_b = self.bias[labels]
        delta_w = true_w[:, None, :] - self.weight[None, :, :]
        delta_b = true_b[:, None] - self.bias[None, :]
        robust_margin = (delta_w * center[:, None, :]).sum(dim=-1) + delta_b
        robust_margin = robust_margin - radius[:, None] * delta_w.norm(dim=-1)
        competitor = ~F.one_hot(labels, self.weight.size(0)).bool()
        return F.relu(margin - robust_margin[competitor]).pow(2).mean()


class DoFacetFeasibilityHull(nn.Module):
    """Sampling-policy robust classifier via pathology feasible-hull geometry."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        state_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.facet_lift = PathologyFacetLift(num_vars, num_bins, state_dim, hidden_dim)
        self.projector = FeasibleHullProjector(state_dim)
        self.classifier = RobustHullClassifier(state_dim, num_classes)
        self.num_bins = num_bins

    def forward(self, batch: dict) -> dict:
        facets = self.facet_lift(batch)
        hull = self.projector(facets)
        cls = self.classifier(hull["center"], hull["radius"])
        return {**facets, **hull, **cls}

    def tcf_bin_grounding_loss(self, out: dict, batch: dict) -> torch.Tensor:
        target = batch["pathology_bin_id"].clamp(0, self.num_bins - 1)
        logits = self.facet_lift.bin_reconstruct(out["normal"])
        mask = out["event_mask"].bool()
        if not mask.any():
            return torch.zeros((), device=target.device)
        return F.cross_entropy(logits[mask], target[mask])

    def calendar_facet_sobriety_loss(self, batch: dict, factual: dict, eps: float = 0.05) -> torch.Tensor:
        losses = []
        for view in batch.get("facet_view_bank", []):
            cf = self.forward(view)
            losses.append(F.relu(cf["tightness"] - factual["tightness"].detach() - eps).pow(2).mean())
        if not losses:
            return torch.zeros((), device=factual["logits"].device)
        return torch.stack(losses).mean()

    def radius_honesty_loss(self, batch: dict, factual: dict, eta: float = 0.20) -> torch.Tensor:
        losses = []
        factual_coverage = factual["event_mask"].float().mean(dim=1)
        for view in batch.get("facet_view_bank", []):
            cf = self.forward(view)
            cf_coverage = view["event_mask"].float().mean(dim=1)
            coverage_drop = (factual_coverage - cf_coverage).clamp_min(0.0)
            target_radius = factual["radius"].detach() + eta * coverage_drop
            losses.append(F.relu(target_radius - cf["radius"]).pow(2).mean())
        if not losses:
            return torch.zeros((), device=factual["logits"].device)
        return torch.stack(losses).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_margin: float = 0.30,
        lambda_facet: float = 0.25,
        lambda_radius: float = 0.15,
        lambda_bin: float = 0.10,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)
        margin_loss = self.classifier.worst_case_margin_loss(out["center"], out["radius"], labels)
        facet_loss = self.calendar_facet_sobriety_loss(batch, out)
        radius_loss = self.radius_honesty_loss(batch, out)
        bin_loss = self.tcf_bin_grounding_loss(out, batch)

        total = (
            cls_loss
            + lambda_margin * margin_loss
            + lambda_facet * facet_loss
            + lambda_radius * radius_loss
            + lambda_bin * bin_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "worst_case_margin_loss": margin_loss.detach(),
            "calendar_facet_sobriety_loss": facet_loss.detach(),
            "radius_honesty_loss": radius_loss.detach(),
            "tcf_bin_grounding_loss": bin_loss.detach(),
            "mean_hull_radius": out["radius"].mean().detach(),
            "mean_facet_tightness": out["tightness"].mean().detach(),
        }


@torch.no_grad()
def build_counterfactual_facet_views(batch: dict) -> list[dict]:
    """Create policy views that alter facet visibility, not class labels."""
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    quality = batch.get("measurement_quality", torch.ones_like(value))
    pending = batch.get("value_pending_mask", torch.zeros_like(value))
    bin_id = batch["pathology_bin_id"]
    device = value.device
    _, num_events = value.shape

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_mask, new_quality, new_pending):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = var_id
        out["event_mask"] = new_mask
        out["measurement_quality"] = new_quality
        out["value_pending_mask"] = new_pending
        out["pathology_bin_id"] = bin_id
        out.pop("facet_view_bank", None)
        return out

    views = []

    # Routine-round: calendar changes should not create tighter pathology facets.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    views.append(clone_with(value * mask, rounded_time, mask, quality, pending))

    # Panel debatching: near-synchronous cross-variable panels are split without new values.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    debatch_quality = quality * (1.0 - 0.4 * close * changed_var)
    views.append(clone_with(value * mask, time + 0.03 * horizon * close * changed_var, mask, debatch_quality, pending))

    # Value-pending: order exists, but pathology facets should loosen.
    pending_value = torch.zeros_like(value)
    pending_flag = torch.maximum(pending, mask)
    pending_quality = quality * 0.10
    views.append(clone_with(pending_value, time, mask, pending_quality, pending_flag))

    # Alarm-burst thinning: remove alternating late dense events.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    thin_mask = torch.where(late > 0, mask * alternating, mask)
    views.append(clone_with(value * thin_mask, time, thin_mask, quality * thin_mask, pending * thin_mask))

    return views
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cross-center facet coverage shift`：在 HiRID / MIMIC-IV / eICU 风格环境中比较哪些病理 facets 经常可见。
   - `value-pending shift`：化验 order 可见但数值未返回，检查 DFFH 是否自动放宽可行域。
   - `panel-debatch shift`：同步 panel 被拆成异步事件，检查 facet tightness 是否被日历共现伪造。
   - `alarm-burst thinning`：告警后密集观测被稀释，检查 robust margin 是否仍由病理约束支持。

2. **对比方法**
   - 普通 TCF / EHR foundation model。
   - 普通 irregular Transformer、mTAND、STAR-Set、VP-GNN、MTM。
   - PULSE 中强传统模型与 LLM workflow。
   - 历史方案：DHN、PQD、OS-MQ、SCSC、CKCF、C-CRS、DJRT、DSPP、DCPD 等。

3. **核心指标**
   - in-policy accuracy / AUPRC。
   - cross-center worst-policy accuracy / AUPRC。
   - mean hull radius：跨中心可行域是否合理变宽。
   - calendar-fabricated tightness：纯日历 view 是否不合理收紧可行域。
   - robust true-class margin：真实类在可行域 worst-case 下的边际。
   - facet grounding accuracy：TCF 病理 bin 是否被正确翻译成 facets。

4. **消融实验**
   - 去掉 conservative radius penalty，检查高缺失/跨中心下是否过度自信。
   - 去掉 `L_calendar_facet_sobriety`，检查 routine/panel 日历是否制造虚假紧约束。
   - 去掉 `L_radius_honesty`，检查 value-pending 下 hull radius 是否诚实变大。
   - 用普通 event pooling 替代 feasible hull，验证凸几何可行域而非模型容量带来鲁棒性。
   - 将 pathology-focused bins 替换为均匀分箱，验证 TCF 病理语义的作用。

## 5. 预期创新性

1. **从事件 token 转向病理可行域**：每个观测值不再直接成为分类证据，而是成为 latent clinical state 的半空间约束。
2. **从采样一致性转向可行域鲁棒边际**：反事实采样不要求 logits 或 representation 一致，只检查类别判断是否在可行域 worst-case 下仍成立。
3. **从不确定性概率转向凸几何半径**：模型用 hull radius 表示采样不足导致的临床状态可行域宽度，不使用 evidential vacuity 或 conformal set。
4. **从 TCF future event 转向 future facet**：保留 TCF 病理分箱，但避免把未来观测流程直接当类别证据；未来预测只作为可能新增的病理约束。
5. **跨中心解释性强**：DFFH 能报告某个中心为何失败：是缺少关键 facets、纯日历制造了 tightness，还是可行域太宽导致 robust margin 不足。

## 6. 一句话投稿卖点

**DFFH 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“不同医院只观测到同一潜在临床状态可行域的不同半空间切面”的问题，通过 TCF 病理分箱、Pathology Facet Lift、可微 feasible-hull projector 与 worst-case hull margin，让分类器只在整个病理可行域都支持真实类别时给出高置信预测，从而避免把跨中心日历、panel 共现、value-pending 或告警密集采样误当作可迁移类别证据。**
