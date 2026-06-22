# Title: Censored Excursion Topology Capsules：面向采样策略偏移的审查拓扑胶囊分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取 `paper_daily.md` 与 `paper_daily_2026-06-12.md`。
- 已读取当前工作区内全部历史提案：
  - `ideas/Idea_Proposal_2026-06-12.md`
  - `ideas/Idea_Proposal_2026-06-13.md`
  - `ideas/Idea_Proposal_2026-06-14.md`
  - `ideas/Idea_Proposal_2026-06-16.md`
  - `ideas/Idea_Proposal_2026-06-19.md`
  - `ideas/Idea_Proposal_2026-06-21.md`
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案核心机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`

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
20. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token 适配层、MTM 的普通 pivotal token mixing，或 MedMamba 的 frequency/graph branch 作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗，不做多视图一致性、不做随机平滑认证、不做后验相除，也不把采样停时建成鞅；而是把采样政策视为对潜在状态“峰谷/异常区间/跨变量共振结构”的审查 censoring。分类器不消费原始采样节奏，而消费在审查区间内仍可稳定存在的 excursion topology capsules。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列的分类捷径往往来自“哪些点被看见”，而不是“底层状态真的发生了什么”。例如 ICU 中乳酸被频繁复查可能意味着病情严重，也可能只是某家医院的固定 protocol；可穿戴设备晚间低频采样可能来自电量策略，而不是用户状态平稳。若模型直接利用 token 密度、共现窗口或变量缺失，它会把采样流程误当成类别证据。

近期 `paper_daily.md` 中两个最新机制给出关键启发，但也暴露了可继续突破的空白：

- **MTM** 指出 IMTS 的核心瓶颈之一是 channel-wise asynchrony：变量很少同步观测，普通跨通道 attention 或 token mixing 容易受“哪些 token 有机会同窗出现”影响。其风险是 pivotal token selection 会被采样政策牵引。
- **MedMamba** 强调医疗时序中的 nonstationarity、raw/difference 视图和 missing-channel robustness，但其 adaptive graph / frequency branch 并未显式区分状态拓扑与采样流程。

Censored Excursion Topology Capsules (CETC) 的核心直觉是：

> 采样政策可以改变某个异常峰是否被完整看到、某段变量共振是否被截断、某个通道是否缺席；但它不应凭空创造一个真实底层状态的拓扑事件。与其让模型学习“哪些 token 被选中”，不如让模型学习“哪些峰谷区间、持续异常和跨变量共振在审查后仍有可验证的拓扑存在区间”。

这与当前“采样解耦/反事实干预”框架的结合方式非常自然：

- value process 负责把观测值转为 raw / differential excursion 信号；
- sampling process 不输出 hazard、density ratio、policy residual 或 stop recipe，而是输出 **censoring envelope**：某个反事实采样策略最多可能遮蔽多少拓扑寿命、幅度与持续质量；
- counterfactual intervention 不要求不同采样视图 representation 相等，而是生成被审查后的观测子序列，用于构造拓扑胶囊的上下界；
- classifier 只能使用落在审查区间内的持久拓扑胶囊，因此难以直接利用采样密度、时间窗共现或变量联测捷径。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从 token mixing 改为 Soft Excursion Topology Capsules

CETC 把每条不规则多变量序列展开为事件流 `(t_i, d_i, x_i)`，再为每个变量、每个 value scale、每种极性构造可微的 excursion capsule。

1. **Raw / Differential Excursion Bank**
   - raw excursion：某变量值超过若干幅度阈值后形成的高值或低值区间。
   - differential excursion：相邻同变量观测的变化量形成的上升/下降突变区间。
   - 这里吸收 MedMamba 的 raw/difference 视角，但不使用 frequency branch，也不学习 adaptive graph。

2. **Soft Topological Statistics**
   - 对每个 `(variable, scale, polarity)` 计算：
     - soft birth time：该 excursion 第一次稳定出现的时间；
     - soft death time：该 excursion 最后一次稳定出现的时间；
     - persistence span：`death - birth`；
     - excursion mass：持续异常的软面积；
     - fragmentation：该 excursion 被采样断裂成多少碎片；
     - persistence quality：`mass / (1 + fragmentation)`。
   - 这些量不是 learnable time reference points，也不是 token attention；它们描述的是状态峰谷集合的拓扑形态。

3. **Censor-Aware Capsule Projection**
   - sampling branch 只给出审查 envelope，例如某个反事实策略可能遮蔽多少时间跨度、多少变量观测、多少峰值幅度。
   - 胶囊通过 `censor_confidence` 门控进入分类头：

```text
capsule = [span, mass, quality, amplitude, fragmentation]
robust_capsule = capsule * sigmoid(CensorProjector(capsule, censor_envelope))
logits = Classifier(robust_capsule)
```

关键差异：分类头不接收 mask pattern、采样策略 id、policy residual、hazard、density ratio、stop rule 或 protocol tax；它只接收“在当前采样审查下仍可被解释为真实状态拓扑”的胶囊。

### 2.2 改 Loss：从跨视图一致转向 Censored Persistence Interval Likelihood

总目标：

```text
L = L_cls
  + lambda_int  * L_censored_interval
  + lambda_frag * L_fragmentation_sobriety
  + lambda_env  * L_envelope_calibration
```

#### A. 分类损失 `L_cls`

只使用事实观测下的 robust topology capsules：

```text
L_cls = CE(Classifier(robust_capsule_factual), y)
```

#### B. 审查区间损失 `L_censored_interval`

对同一条事实序列，反事实采样模块生成若干被审查的观测子序列。若反事实策略只删除或遮蔽观测，那么在被审查视图中看到的拓扑寿命、质量和幅度应是潜在事实拓扑的下界；而 sampling branch 给出的 censoring envelope 给出上界松弛。

```text
lower_r = capsule(do_censor_r(x))
upper_r = lower_r + envelope_r
L_censored_interval =
  sum_r relu(lower_r - capsule_factual)^2
       + relu(capsule_factual - upper_r)^2
```

这不是 representation consistency：我们不要求不同反事实视图的 capsule 相同，也不要求 logits 相同。我们只要求事实胶囊落入“被审查观测能够支持的拓扑区间”。如果模型依赖某个只在训练采样政策中出现的短暂 token，该 token 在反事实审查下会给出很窄或不稳定的区间，无法支撑稳定分类。

#### C. Fragmentation Sobriety `L_fragmentation_sobriety`

采样密度越高，短碎片 excursion 越容易被制造出来。为了避免分类器利用这种 sampling-induced fragmentation，CETC 对高碎片、低质量胶囊施加清醒度约束：

```text
L_fragmentation_sobriety =
  mean relu(fragmentation - rho * mass - margin)^2
```

它不惩罚真实持续异常；只有“碎片很多但拓扑质量很低”的胶囊会被压制。

#### D. Envelope Calibration `L_envelope_calibration`

sampling branch 的 envelope 不能任意放大，否则区间损失会失去约束。使用反事实删除量的可计算上界监督它：

```text
observed_gap_r = topology_gap(capsule_factual, capsule_censored_r)
L_envelope_calibration =
  pinball_loss(envelope_r, stopgrad(observed_gap_r), quantile=q)
```

这不是 density ratio，也不是 doubly robust correction；它只校准“当前反事实采样最多审查掉多少拓扑证据”。

### 2.3 改 Dataloader：返回 Censor Recipe Bank，而不是一致性增强或停时 recipe

新增 `CensorTopologyCollator`，每个 batch 返回：

1. `event_value`：按时间排序的观测值。
2. `event_time`：连续时间戳。
3. `event_var_id`：变量 id。
4. `event_mask`：事件 padding mask。
5. `censor_recipe_bank`：一组只删除/遮蔽观测的审查 recipe，例如：
   - `channel_budget_censor`：某些变量只保留前 `k` 个观测；
   - `late_window_blackout`：晚期窗口被设备策略遮蔽；
   - `asynchrony_gap_censor`：删除会制造跨变量同窗的观测；
   - `burst_thinning_censor`：把高频复测 burst 稀疏化。
6. `censor_envelope_target`：由删除的时间跨度、幅度上界和变量覆盖率计算出的拓扑证据损失上界。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `ExcursionSignalStem`：输出 raw value 与同变量 differential value。
- 现有 sampling branch 改为 `CensorEnvelopeEstimator`：估计反事实采样会遮蔽的拓扑寿命、质量和幅度上界。
- 现有 counterfactual intervention 保留，但从“生成 policy views 用于一致性/风险/平滑/停时矩”改为“生成 censor recipe bank 用于拓扑区间似然”。
- 推理阶段无需知道测试策略标签；只从事实观测事件构造 topology capsules，并输出：
  - 分类概率；
  - 每个变量/阈值/极性的 persistence quality；
  - censor confidence；
  - topology fragmentation score，作为采样捷径诊断。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_sum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim)


def normalize_time(event_time: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    end_time = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    return event_time / end_time


class SoftExcursionTopology(nn.Module):
    """Differentiable excursion topology capsules for irregular event streams."""

    def __init__(
        self,
        num_vars: int,
        num_levels: int,
        sharpness: float = 12.0,
        time_temperature: float = 8.0,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.num_levels = num_levels
        self.sharpness = sharpness
        self.time_temperature = time_temperature
        self.raw_threshold = nn.Parameter(torch.linspace(0.3, 1.5, num_levels).repeat(num_vars, 1))

    def _capsules_for_signal(
        self,
        signal: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        # signal/event_time/event_var_id/event_mask/visibility: [B, N]
        time_norm = normalize_time(event_time, event_mask)
        delta_t = torch.zeros_like(event_time)
        delta_t[:, 1:] = (time_norm[:, 1:] - time_norm[:, :-1]).clamp_min(0.0)
        active_mask = event_mask * visibility

        capsules = []
        for var_idx in range(self.num_vars):
            var_mask = ((event_var_id == var_idx).to(signal.dtype) * active_mask).clamp(0.0, 1.0)
            threshold = F.softplus(self.raw_threshold[var_idx]).view(1, 1, self.num_levels)

            centered = signal.unsqueeze(-1)
            high = torch.sigmoid(self.sharpness * (centered - threshold))
            low = torch.sigmoid(self.sharpness * (-centered - threshold))
            active = torch.stack([high, low], dim=-1) * var_mask[:, :, None, None]

            log_active = (active + 1e-6).log()
            first_weight = torch.softmax(log_active - self.time_temperature * time_norm[:, :, None, None], dim=1)
            last_weight = torch.softmax(log_active + self.time_temperature * time_norm[:, :, None, None], dim=1)

            birth = (first_weight * time_norm[:, :, None, None]).sum(dim=1)
            death = (last_weight * time_norm[:, :, None, None]).sum(dim=1)
            span = (death - birth).clamp_min(0.0)

            mass = (active * delta_t[:, :, None, None]).sum(dim=1)
            jump = (active[:, 1:] - active[:, :-1]).abs().sum(dim=1)
            amplitude = (active * signal.abs()[:, :, None, None]).amax(dim=1)
            quality = mass / (1.0 + jump)

            # [B, L, 2, 5]
            var_capsule = torch.stack([span, mass, quality, amplitude, jump], dim=-1)
            capsules.append(var_capsule.flatten(start_dim=1))

        return torch.cat(capsules, dim=-1)

    def forward(
        self,
        event_value: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
        visibility: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visibility is None:
            visibility = torch.ones_like(event_mask)

        raw_capsule = self._capsules_for_signal(
            signal=event_value,
            event_time=event_time,
            event_var_id=event_var_id,
            event_mask=event_mask,
            visibility=visibility,
        )

        diff_value = torch.zeros_like(event_value)
        diff_value[:, 1:] = event_value[:, 1:] - event_value[:, :-1]
        same_var = (event_var_id[:, 1:] == event_var_id[:, :-1]).to(event_value.dtype)
        diff_value[:, 1:] = diff_value[:, 1:] * same_var
        diff_capsule = self._capsules_for_signal(
            signal=diff_value,
            event_time=event_time,
            event_var_id=event_var_id,
            event_mask=event_mask,
            visibility=visibility,
        )

        return torch.cat([raw_capsule, diff_capsule], dim=-1)


class CensorEnvelopeEstimator(nn.Module):
    """Estimate how much topology a sampling intervention could censor."""

    def __init__(self, capsule_dim: int, recipe_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(recipe_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, capsule_dim),
            nn.Softplus(),
        )

    def forward(
        self,
        recipe: torch.Tensor,
        event_time: torch.Tensor,
        event_mask: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        # recipe: [B, R], visibility/event_mask: [B, N]
        kept = (visibility * event_mask).sum(dim=1, keepdim=True)
        total = event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        keep_rate = kept / total

        time_norm = normalize_time(event_time, event_mask)
        observed_span = (time_norm * visibility * event_mask).amax(dim=1, keepdim=True)
        removed_rate = 1.0 - keep_rate
        max_gap = (1.0 - observed_span).clamp_min(0.0)

        stats = torch.cat([keep_rate, removed_rate, observed_span, max_gap], dim=-1)
        return self.net(torch.cat([recipe, stats], dim=-1))


def censored_interval_loss(
    factual_capsule: torch.Tensor,
    censored_capsule: torch.Tensor,
    envelope: torch.Tensor,
) -> torch.Tensor:
    lower_violation = F.relu(censored_capsule - factual_capsule)
    upper_violation = F.relu(factual_capsule - censored_capsule - envelope)
    return lower_violation.pow(2).mean() + upper_violation.pow(2).mean()


def fragmentation_sobriety_loss(
    capsule: torch.Tensor,
    feature_width: int = 5,
    rho: float = 4.0,
    margin: float = 0.05,
) -> torch.Tensor:
    # Layout per capsule unit: [span, mass, quality, amplitude, jump].
    view = capsule.view(capsule.size(0), -1, feature_width)
    mass = view[..., 1]
    jump = view[..., 4]
    return F.relu(jump - rho * mass - margin).pow(2).mean()


def envelope_pinball_loss(
    envelope: torch.Tensor,
    target_gap: torch.Tensor,
    quantile: float = 0.8,
) -> torch.Tensor:
    error = target_gap.detach() - envelope
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


class CensoredExcursionTopologyClassifier(nn.Module):
    """Classify irregular time series using censor-robust topology capsules."""

    def __init__(
        self,
        num_vars: int,
        num_levels: int,
        recipe_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.topology = SoftExcursionTopology(num_vars=num_vars, num_levels=num_levels)
        capsule_dim = 2 * num_vars * num_levels * 2 * 5
        self.envelope = CensorEnvelopeEstimator(capsule_dim, recipe_dim, hidden_dim)
        self.censor_gate = nn.Sequential(
            nn.Linear(2 * capsule_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, capsule_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(capsule_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_capsule(self, batch: dict, visibility: torch.Tensor | None = None) -> torch.Tensor:
        return self.topology(
            event_value=batch["event_value"],
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
            visibility=visibility,
        )

    def forward(self, batch: dict) -> dict:
        factual_capsule = self.encode_capsule(batch)
        zero_recipe = torch.zeros(
            factual_capsule.size(0),
            batch["censor_recipe_bank"].size(-1),
            device=factual_capsule.device,
            dtype=factual_capsule.dtype,
        )
        factual_envelope = self.envelope(
            recipe=zero_recipe,
            event_time=batch["event_time"],
            event_mask=batch["event_mask"],
            visibility=batch["event_mask"],
        )
        gate = self.censor_gate(torch.cat([factual_capsule, factual_envelope], dim=-1))
        robust_capsule = factual_capsule * gate
        logits = self.classifier(robust_capsule)
        return {
            "logits": logits,
            "factual_capsule": factual_capsule,
            "robust_capsule": robust_capsule,
            "factual_envelope": factual_envelope,
            "censor_gate": gate,
        }

    def training_loss(
        self,
        batch: dict,
        lambda_int: float = 0.25,
        lambda_frag: float = 0.05,
        lambda_env: float = 0.1,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]

        cls_loss = F.cross_entropy(out["logits"], labels)
        interval_losses = []
        envelope_losses = []

        for recipe, visibility, target_gap in zip(
            batch["censor_recipe_bank"].unbind(dim=1),
            batch["censor_visibility_bank"].unbind(dim=1),
            batch["censor_envelope_target"].unbind(dim=1),
        ):
            censored_capsule = self.encode_capsule(batch, visibility=visibility)
            envelope = self.envelope(
                recipe=recipe,
                event_time=batch["event_time"],
                event_mask=batch["event_mask"],
                visibility=visibility,
            )
            interval_losses.append(
                censored_interval_loss(
                    factual_capsule=out["factual_capsule"],
                    censored_capsule=censored_capsule,
                    envelope=envelope,
                )
            )
            envelope_losses.append(envelope_pinball_loss(envelope, target_gap))

        interval_loss = torch.stack(interval_losses).mean()
        env_loss = torch.stack(envelope_losses).mean()
        frag_loss = fragmentation_sobriety_loss(out["factual_capsule"])

        total = cls_loss + lambda_int * interval_loss + lambda_frag * frag_loss + lambda_env * env_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "censored_interval_loss": interval_loss.detach(),
            "fragmentation_sobriety_loss": frag_loss.detach(),
            "envelope_calibration_loss": env_loss.detach(),
        }


@torch.no_grad()
def build_censor_recipe_bank(
    event_time: torch.Tensor,
    event_var_id: torch.Tensor,
    event_value: torch.Tensor,
    event_mask: torch.Tensor,
    num_vars: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create censor recipes, visibility masks and topology-envelope targets."""

    bsz, num_events = event_time.shape
    device = event_time.device
    recipes = []
    visibility = []
    targets = []

    # Recipe 1: channel budget censor, keeping at most two events per variable.
    budget_visible = torch.zeros_like(event_mask)
    for var_idx in range(num_vars):
        var_event = ((event_var_id == var_idx) & (event_mask > 0)).to(event_mask.dtype)
        rank = var_event.cumsum(dim=1)
        budget_visible = torch.where(
            (var_event > 0) & (rank <= 2),
            torch.ones_like(budget_visible),
            budget_visible,
        )
    recipes.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(bsz, -1))
    visibility.append(budget_visible)

    # Recipe 2: late window blackout.
    horizon = event_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
    late_visible = ((event_time / horizon) < 0.67).to(event_mask.dtype) * event_mask
    recipes.append(torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(bsz, -1))
    visibility.append(late_visible)

    # Recipe 3: burst thinning, keeping alternating observations.
    alternating = ((torch.arange(num_events, device=device)[None, :] % 2) == 0).to(event_mask.dtype)
    burst_visible = alternating.expand(bsz, -1) * event_mask
    recipes.append(torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).expand(bsz, -1))
    visibility.append(burst_visible)

    # Recipe 4: asynchrony gap censor, removing events too close to previous event.
    gap = torch.zeros_like(event_time)
    gap[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
    median_gap = gap.masked_fill(event_mask == 0, 0.0).mean(dim=1, keepdim=True)
    async_visible = ((gap >= median_gap) | (torch.arange(num_events, device=device)[None, :] == 0)).to(event_mask.dtype)
    async_visible = async_visible * event_mask
    recipes.append(torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(bsz, -1))
    visibility.append(async_visible)

    full_abs = (event_value.abs() * event_mask).amax(dim=1, keepdim=True)
    for vis in visibility:
        removed = (event_mask - vis).clamp_min(0.0)
        removed_rate = removed.sum(dim=1, keepdim=True) / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        removed_amp = (event_value.abs() * removed).amax(dim=1, keepdim=True) / full_abs.clamp_min(1e-6)
        # Broadcastable target; the model-specific capsule dimension is expanded later by training code.
        target = torch.maximum(removed_rate, removed_amp)
        targets.append(target)

    recipe_bank = torch.stack(recipes, dim=1)
    visibility_bank = torch.stack(visibility, dim=1)
    target_bank = torch.stack(targets, dim=1)
    return recipe_bank, visibility_bank, target_bank
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `burst protocol shift`：训练医院报警后密集复测，测试医院只保留稀疏复查。
   - `late blackout shift`：设备或流程导致晚期窗口缺失。
   - `channel budget shift`：某些变量只保留前几次观测。
   - `asynchrony shift`：删除会制造跨变量同窗的观测，测试 MTM 类 token mixing 是否依赖同步机会。

2. **对比方法**
   - 普通 mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - DHN hazard/null-space baseline。
   - CGS commutator graph surgery baseline。
   - PT-AEM evidence market baseline。
   - PQD posterior quotient baseline。
   - DS-CS certified smoothing baseline。
   - OS-MQ optional-stopping martingale baseline。
   - MTM 普通 token mixing 与 MedMamba raw/difference/frequency/graph 主干。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - topology interval violation rate。
   - mean censor confidence。
   - fragmentation reliance score：错误样本是否更依赖高碎片低质量胶囊。
   - policy-shift topology gap：不同采样策略下 capsule 是否超出审查 envelope。

4. **消融实验**
   - 去掉 `L_censored_interval`，验证模型是否回到采样特定 token shortcut。
   - 去掉 differential excursion bank，验证突变拓扑是否对事件触发采样有贡献。
   - 去掉 `L_fragmentation_sobriety`，检查高频采样 burst 是否被误用。
   - 将 censor envelope 替换为常数，验证收益来自审查上界校准。
   - 将 topology capsules 替换为普通 pooled tokens，验证不是参数量或简单聚合带来的提升。

## 5. 预期创新性

1. **从采样策略建模转向拓扑审查建模**：不问“为什么被采样”，而问“当前观测在被采样政策审查后仍能支持哪些底层状态拓扑事件”。
2. **从 token selection 转向 excursion persistence**：吸收 MTM 对 channel asynchrony 的洞察，但不选择 pivotal tokens；分类证据来自峰谷区间、持续质量和碎片度。
3. **从 raw/difference 多视图转向拓扑胶囊**：吸收 MedMamba 的 raw/differential 非平稳诊断，但不使用 frequency branch 或 adaptive graph。
4. **从一致性增强转向区间似然**：不同采样视图不必一致；被审查视图只提供 latent topology 的下界与上界 envelope。
5. **解释性直接对应采样偏移**：模型能报告某个预测依赖的是持久异常区间，还是高碎片、低质量、易被采样政策制造的伪拓扑。

## 6. 一句话投稿卖点

**CETC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“状态拓扑被采样政策审查”的问题，并通过 soft excursion topology capsules 与 censored persistence interval likelihood，让分类器依赖跨审查策略仍可验证的峰谷/突变/持续异常结构，而不是依赖训练环境中特定的采样密度、同步窗口或变量联测捷径。**
