# Title: Do-Measure Doubly Robust Risk Rewriting：采样测度变换的反事实双稳健风险重写

## 0. 强制读取记录与思维黑名单

### 已读取材料

- `paper_daily.md`
- `paper_daily_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-13.md`
- `ideas/Idea_Proposal_2026-06-14.md`
- `ideas/Idea_Proposal_2026-06-16.md`
- `ideas/Idea_Proposal_2026-06-19.md`
- 自动化记忆中记录但当前工作区未检出的 `ideas/Idea_Proposal_2026-06-17.md` 核心机制。

### 未检出但已确认搜索

- 当前仓库未检出 `my_work_summary.md`。
- 当前仓库也未检出 `*summary*.md`、`*Summary*.md`、`*work*.md` 或中文 `*总结*.md` 形式的可替代工作总结。

### 历史核心机制黑名单

为保证本轮提案与历史想法显著正交，本提案明确避开以下机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder。
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
17. policy simplex randomized smoothing、logit-normal / Dirichlet 平滑核、certified policy radius、policy coverage loss。
18. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的 state-query / policy-query 分裂作为主机制。

本提案选择一个新的正交切入点：

> **不再试图删除、压缩、征税、相除、平滑或解释采样信息，而是把不规则观测看成一个随机“采样测度”。采样策略偏移就是观测测度从训练政策 `mu` 变到部署或反事实政策 `nu`。我们直接把训练风险通过 Radon-Nikodym 密度比和双稳健 influence correction 重写到目标采样测度下。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列中的每条样本并不只是一个 padded matrix，而是一个带观测值的离散随机测度：

```text
mu_x = sum_i delta_(t_i, d_i, value_i)
```

训练医院、设备或采集协议改变时，最先漂移的往往不是底层状态 `x(t)`，而是这套观测测度：哪些变量进入测度、何时进入测度、同一时间附近哪些变量共同进入测度。过去提案主要在 representation 或模型空间上处理这种偏移：做零空间、交换子、证据税、后验商、误差语义图或随机平滑。但这些方法都在回答“分类器应该怎样不使用采样信息”。本提案换一个问题：

> 如果未来部署时希望在目标采样政策 `do(nu)` 下评估分类风险，为什么还要用训练采样政策 `mu` 下的经验风险训练分类器？

近期 `paper_daily.md` 中两篇最新工作提供了直接启发：

- **QuITE** 说明不规则观测可以先通过 query-like embedding 变成 backbone-agnostic 的固定表示；但 query attention 本身可能吸收采样密度和变量共现捷径。本提案借用“可插拔观测函数”的思想，但不把 query 分成 state/policy，也不做跨视图一致性；我们把 query 看成对观测测度的有限 test functions，用于估计目标政策下的测度矩。
- **SDEVI** 强调不规则观测应尊重 continuous-discrete 生成关系，而不是先补齐路径；本提案不构造完整 latent path，也不做后验商，而是估计目标观测测度下的有限充分矩，并用双稳健校正保证：只要密度比或测度矩模型有一个可靠，目标风险估计就不容易被训练采样政策带偏。
- **PYRREGULAR** 提醒我们需要统一 event-array 接口与跨策略评测；本提案天然以 `(time, variable, value, mask)` 的稀疏事件表为输入，可以作为 policy-shift benchmark 的风险层插件。

当前“采样解耦/反事实干预”框架已经有两个可复用资产：

1. value process 可以把观测事件编码成状态相关的 event functions；
2. counterfactual intervention 可以生成目标采样政策 `do(nu)` 的 recipe 或少量模拟事件。

**Do-Measure Doubly Robust Risk Rewriting (DM-DRR)** 将 sampling branch 的角色改成“测度变换器”：

- 它估计训练测度 `mu` 到目标测度 `nu` 的 Radon-Nikodym 密度比 `rho = dnu / dmu`；
- 它不把 `rho` 输入分类头，也不训练策略对抗器；
- 分类器实际看到的是目标采样测度下的双稳健校正矩 `z_dr`；
- 训练损失直接近似 `R_nu(f) = E_{do(nu)} CE(f(x), y)`，而不是训练政策 `mu` 下的普通 CE。

这样做能解决 sampling-policy shift 的核心原因是：当训练集的观测测度偏向某类病人或某类流程时，普通 ERM 优化的是 `mu` 风险；DM-DRR 把风险重写到用户指定的目标政策 `nu`，让模型学到“在部署采样方式下仍然成立”的分类边界。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：返回目标采样测度 recipe，而不是多视图一致性增强

新增 `DoMeasureCollator`。每个 batch 返回：

1. **factual event measure**
   - `event_value`: `[B, N]`
   - `event_time`: `[B, N]`
   - `event_var_id`: `[B, N]`
   - `event_mask`: `[B, N]`

2. **target measure recipe `nu_recipe`**
   - 由当前 counterfactual intervention 模块产生，描述希望训练风险对齐到的部署采样政策；
   - 例如 routine panel、低预算设备、晚期随访、报警后复测等；
   - 它不作为分类特征进入 classifier，只用于测度比估计和目标矩预测。

3. **ratio calibration events**
   - 少量从 `do(nu_recipe)` 生成的目标政策事件；
   - 只用于训练密度比头区分“这是 factual measure 的事件还是 target measure 的事件”；
   - 不要求 factual view 与 target view 的 logits 一致，也不计算 risk variance。

4. **optional target moment probes**
   - 若反事实模块能生成目标政策下的观测事件，则用于校准目标测度矩；
   - 若不能生成，也可以只依赖双稳健中的模型项和密度比项。

### 2.2 改 Encoder：从 sequence encoder 改为“采样测度 test-function encoder”

保留现有 value encoder 的底座，但把输入事件先映射为一组可积函数：

```text
phi_theta(t_i, d_i, x_i) in R^H
```

这类似 QuITE 的可插拔 embedding，但语义完全不同：

- `phi` 不是 learnable time reference；
- `phi` 不是 state-query / policy-query 分裂；
- `phi` 不接收 missingness pattern 作为分类捷径；
- `phi` 是一组对观测测度求积分的 test functions。

目标政策下的理想表示是：

```text
z_nu = Integral phi_theta(t, d, x) dnu(t, d)
```

但训练时只能看到 `mu` 下的 factual events，因此用双稳健形式估计：

```text
z_dr = m_phi(nu_recipe)
     + sum_i normalized_rho_i * [phi(event_i) - b_phi(event_i, nu_recipe)]
```

其中：

- `m_phi(nu_recipe)` 是模型预测的目标测度矩；
- `rho_i = dnu / dmu (event_i)` 是测度密度比；
- `b_phi` 是 event-level control variate；
- 若 `rho` 估计准确，即使 `m_phi/b_phi` 有偏，校正项也能拉回目标测度；
- 若 `m_phi/b_phi` 估计准确，即使 `rho` 有偏，模型项也能提供稳定目标矩。

这就是“双稳健”的核心创新。

### 2.3 改 Loss：从普通 ERM 改为目标采样测度下的双稳健风险

总目标：

```text
L = L_dr_cls
  + lambda_ratio * L_ratio_calibration
  + lambda_moment * L_measure_moment
  + lambda_if     * L_influence_bound
```

#### A. 双稳健分类损失 `L_dr_cls`

分类器只使用 `z_dr`：

```text
logits = Classifier(z_dr)
L_dr_cls = CE(logits, y)
```

这不是跨策略 logits 一致性。模型不会被要求在 factual policy 和 target policy 下输出同一结果；它只被要求在目标采样测度 `nu` 下的双稳健风险最小。

#### B. 密度比校准 `L_ratio_calibration`

用 factual events 与 `do(nu)` events 训练一个密度比分类头：

```text
s(e, nu) = log p_nu(e) - log p_mu(e)
rho(e) = exp(s(e, nu))
```

`s` 不进入分类器。它只用于把 factual event measure 重权到 target event measure。

#### C. 目标测度矩校准 `L_measure_moment`

若 dataloader 能提供少量目标政策事件，则用它校准 `z_dr`：

```text
target_moment = mean_j phi(e_j under do(nu))
L_measure_moment = SmoothL1(z_dr, stopgrad(target_moment))
```

注意这不是表示一致性：我们不比较 factual representation 与 target representation；我们只检查双稳健估计出的目标测度矩是否正确。

#### D. influence bound `L_influence_bound`

密度比过大时，一个训练政策中特有的事件可能主导目标风险。加入 influence bound：

```text
influence_i = normalized_rho_i * ||phi_i - b_i||
L_influence_bound = mean relu(influence_i - c)^2
```

这不是 evidence tax，也不是 token budget。它不根据“协议风险”惩罚证据，而是控制双稳健估计器的方差，避免少数高权重事件让训练不稳定。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value process：改造成 event test-function encoder `phi_theta`。
- 现有 sampling process：不输出 residual、tax、hazard 或 smoothing policy，而是输出测度密度比 `rho = dnu/dmu`。
- 现有 counterfactual intervention：不再生成一致性视图、risk-variance 视图或随机平滑核；它只生成目标采样测度 recipe 和少量 ratio calibration events。
- 推理阶段：用户指定部署 policy recipe `nu_recipe`，模型输出 `p(y | do(nu_recipe))`；如果不知道部署政策，可用一个保守 canonical recipe 或报告多个政策下的目标风险。

## 3. Code Draft: PyTorch 核心模块草稿

```python
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DoMeasureConfig:
    hidden_dim: int
    recipe_dim: int
    num_vars: int
    num_classes: int
    max_ratio: float = 20.0
    influence_cap: float = 1.5
    eps: float = 1e-6


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(dtype=x.dtype)
    denom = weight.sum(dim=dim).clamp_min(1.0)
    return (x * weight).sum(dim=dim) / denom


class EventMeasureFunction(nn.Module):
    """Map sparse irregular events to test functions phi(t, d, value)."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        event_value: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        # event_*: [B, N], event_mask: [B, N]
        time_scale = event_time.amax(dim=1, keepdim=True).clamp_min(1.0)
        time_norm = event_time / time_scale
        time_feat = torch.stack([time_norm, torch.log1p(event_time)], dim=-1)

        h = torch.cat(
            [
                self.value_proj(event_value.unsqueeze(-1)),
                self.time_proj(time_feat),
                self.var_embed(event_var_id),
            ],
            dim=-1,
        )
        return self.out(h) * event_mask.unsqueeze(-1).to(dtype=h.dtype)


class MeasureRatioHead(nn.Module):
    """Estimate log dnu/dmu for events under a target sampling recipe."""

    def __init__(self, hidden_dim: int, recipe_dim: int):
        super().__init__()
        self.recipe_proj = nn.Linear(recipe_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def log_ratio(self, event_h: torch.Tensor, recipe: torch.Tensor) -> torch.Tensor:
        recipe_h = self.recipe_proj(recipe)[:, None, :].expand_as(event_h)
        return self.net(torch.cat([event_h.detach(), recipe_h], dim=-1)).squeeze(-1)

    def ratio(self, event_h: torch.Tensor, recipe: torch.Tensor, max_ratio: float) -> torch.Tensor:
        return self.log_ratio(event_h, recipe).exp().clamp(max=max_ratio)


class TargetMomentModel(nn.Module):
    """Model-based target-measure moment and event-level control variate."""

    def __init__(self, hidden_dim: int, recipe_dim: int):
        super().__init__()
        self.recipe_proj = nn.Sequential(
            nn.Linear(recipe_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.event_baseline = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_moment = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        event_h: torch.Tensor,
        event_mask: torch.Tensor,
        recipe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = masked_mean(event_h, event_mask, dim=1)
        recipe_h = self.recipe_proj(recipe)
        context_h = self.context_proj(context)

        target_moment = self.target_moment(torch.cat([context_h, recipe_h], dim=-1))
        recipe_event = recipe_h[:, None, :].expand_as(event_h)
        baseline = self.event_baseline(torch.cat([event_h.detach(), recipe_event], dim=-1))
        return target_moment, baseline


class DoMeasureDRRClassifier(nn.Module):
    """Classify under a target sampling measure using doubly robust risk rewriting."""

    def __init__(self, config: DoMeasureConfig):
        super().__init__()
        self.config = config
        self.event_fn = EventMeasureFunction(config.num_vars, config.hidden_dim)
        self.ratio_head = MeasureRatioHead(config.hidden_dim, config.recipe_dim)
        self.target_model = TargetMomentModel(config.hidden_dim, config.recipe_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def encode_events(self, batch: dict, prefix: str = "event") -> torch.Tensor:
        return self.event_fn(
            event_value=batch[f"{prefix}_value"],
            event_time=batch[f"{prefix}_time"],
            event_var_id=batch[f"{prefix}_var_id"],
            event_mask=batch[f"{prefix}_mask"],
        )

    def doubly_robust_moment(self, batch: dict) -> dict:
        event_h = self.encode_events(batch, prefix="event")
        event_mask = batch["event_mask"]
        recipe = batch["nu_recipe"]

        ratio = self.ratio_head.ratio(event_h, recipe, self.config.max_ratio)
        ratio = ratio * event_mask.to(dtype=ratio.dtype)
        normalized_ratio = ratio / ratio.sum(dim=1, keepdim=True).clamp_min(self.config.eps)

        target_prior, event_baseline = self.target_model(event_h, event_mask, recipe)
        correction = normalized_ratio.unsqueeze(-1) * (event_h - event_baseline)
        z_dr = target_prior + correction.sum(dim=1)

        influence = normalized_ratio.unsqueeze(-1) * (event_h - event_baseline).norm(dim=-1, keepdim=True)
        influence = influence.squeeze(-1) * event_mask.to(dtype=event_h.dtype)

        return {
            "z_dr": z_dr,
            "event_h": event_h,
            "ratio": ratio,
            "normalized_ratio": normalized_ratio,
            "target_prior": target_prior,
            "event_baseline": event_baseline,
            "influence": influence,
        }

    def forward(self, batch: dict) -> dict:
        out = self.doubly_robust_moment(batch)
        out["logits"] = self.classifier(out["z_dr"])
        return out

    def ratio_calibration_loss(self, batch: dict, factual_h: torch.Tensor) -> torch.Tensor:
        """Train log-ratio as target-vs-factual event discriminator.

        With equal class priors, the optimal discriminator logit estimates
        log p_target(event) - log p_factual(event), i.e. log dnu/dmu.
        """

        recipe = batch["nu_recipe"]
        factual_logit = self.ratio_head.log_ratio(factual_h, recipe)
        factual_loss = F.binary_cross_entropy_with_logits(
            factual_logit,
            torch.zeros_like(factual_logit),
            reduction="none",
        )
        factual_loss = (factual_loss * batch["event_mask"]).sum()
        factual_loss = factual_loss / batch["event_mask"].sum().clamp_min(1.0)

        if "target_value" not in batch:
            return factual_loss

        target_h = self.encode_events(batch, prefix="target")
        target_logit = self.ratio_head.log_ratio(target_h, recipe)
        target_loss = F.binary_cross_entropy_with_logits(
            target_logit,
            torch.ones_like(target_logit),
            reduction="none",
        )
        target_loss = (target_loss * batch["target_mask"]).sum()
        target_loss = target_loss / batch["target_mask"].sum().clamp_min(1.0)
        return 0.5 * (factual_loss + target_loss)

    def target_moment_loss(self, batch: dict, z_dr: torch.Tensor) -> torch.Tensor:
        if "target_value" not in batch:
            return z_dr.new_zeros(())

        with torch.no_grad():
            target_h = self.encode_events(batch, prefix="target")
            target_moment = masked_mean(target_h, batch["target_mask"], dim=1)
        return F.smooth_l1_loss(z_dr, target_moment)

    def training_loss(
        self,
        batch: dict,
        lambda_ratio: float = 0.2,
        lambda_moment: float = 0.1,
        lambda_if: float = 0.03,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]

        cls_loss = F.cross_entropy(out["logits"], labels)
        ratio_loss = self.ratio_calibration_loss(batch, out["event_h"])
        moment_loss = self.target_moment_loss(batch, out["z_dr"])

        influence_loss = F.relu(out["influence"] - self.config.influence_cap).pow(2)
        influence_loss = (influence_loss * batch["event_mask"]).sum()
        influence_loss = influence_loss / batch["event_mask"].sum().clamp_min(1.0)

        total = (
            cls_loss
            + lambda_ratio * ratio_loss
            + lambda_moment * moment_loss
            + lambda_if * influence_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "ratio_loss": ratio_loss.detach(),
            "moment_loss": moment_loss.detach(),
            "influence_loss": influence_loss.detach(),
            "mean_ratio": out["ratio"].detach().mean(),
            "max_ratio": out["ratio"].detach().amax(),
        }


@torch.no_grad()
def build_do_measure_recipe(
    batch_size: int,
    recipe_dim: int,
    device: torch.device,
    mode: str = "canonical",
) -> torch.Tensor:
    """Create target sampling-measure recipes for do(nu).

    The recipe specifies deployment sampling measure, not a classifier feature.
    """

    recipe = torch.zeros(batch_size, recipe_dim, device=device)
    if mode == "canonical":
        recipe[:, 0] = 1.0
    elif mode == "low_budget":
        recipe[:, min(1, recipe_dim - 1)] = 1.0
    elif mode == "late_followup":
        recipe[:, min(2, recipe_dim - 1)] = 1.0
    else:
        active = torch.randint(0, recipe_dim, (batch_size,), device=device)
        recipe[torch.arange(batch_size, device=device), active] = 1.0
    return recipe
```

## 4. 实验切入点

1. **Policy-shift 构造**
   - 训练政策 `mu`: 原始数据采样测度。
   - 目标政策 `nu`: routine panel、低预算设备、late-followup、alarm-followup、变量组稀疏等。
   - 每个样本在训练时只优化一个或少量目标 `nu_recipe`，避免退化成随机平滑。

2. **对比方法**
   - 普通 ERM。
   - mask dropout。
   - missingness-aware encoder。
   - policy consistency / contrastive augmentation。
   - hazard/null-space baseline。
   - commutator graph surgery baseline。
   - protocol-tax evidence market baseline。
   - posterior quotient dynamics baseline。
   - do-sampler certified smoothing baseline。
   - QuITE-style plug-in embedding baseline。
   - SDEVI-style continuous-discrete model without measure-risk rewriting。

3. **核心指标**
   - target-policy accuracy: `Acc_{do(nu)}`。
   - worst-target-policy accuracy。
   - target-policy NLL。
   - density-ratio effective sample size:

```text
ESS = (sum_i rho_i)^2 / sum_i rho_i^2
```

   - influence violation rate: 高影响校正项占比。
   - risk rewriting gap: 普通 ERM 风险与 DM-DRR 目标风险的差距。

4. **消融实验**
   - 去掉密度比项，只用 target moment model，验证是否依赖模型外推。
   - 去掉 target moment model，只用 importance weighting，验证双稳健项是否降低方差。
   - 去掉 influence bound，观察高权重训练事件是否造成不稳定。
   - 把 `nu_recipe` 固定为训练政策，验证收益来自目标测度重写而不是额外参数量。
   - 用错误的目标政策 recipe 测试风险重写是否会按预期偏向指定部署采样方式。

## 5. 预期创新性

1. **从表示去偏转向风险测度重写**：不再问“representation 是否包含采样信息”，而是直接把训练目标从 factual sampling measure 改写为 target sampling measure。
2. **从反事实增强转向 Radon-Nikodym 测度变换**：counterfactual intervention 不负责制造一致性视图，而是定义目标政策 `nu` 并校准 `dnu/dmu`。
3. **从单稳健重加权转向双稳健校正**：仅靠 importance weighting 方差大，仅靠目标矩模型容易外推偏；双稳健形式让两者互为保险。
4. **与 QuITE/SDEVI 的正交结合**：吸收 query-like plug-in test functions 和 continuous-discrete observation-measure 思想，但不复用 state/policy query split、后验商、潜在路径重构或 Wasserstein 对齐。
5. **部署语义清晰**：用户可以明确指定“我希望模型在什么采样政策下鲁棒”，模型优化的就是该政策下的目标风险。

## 6. 一句话投稿卖点

**DM-DRR 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“训练观测测度与部署观测测度不一致”的风险估计问题，并用 Radon-Nikodym 密度比与双稳健 influence correction 将 ERM 直接重写到 `do(target policy)` 下，从而在不依赖危险率、交换子、证据税、后验商、误差语义图或随机平滑证书的前提下，为采样策略偏移鲁棒分类提供一个更因果、更可部署的训练目标。**
