# Title: Policy-Censored Rank Envelopes：面向采样策略偏移的反事实覆盖排序分类器

## 0. 强制去重记录与思维黑名单

### 已读取材料

- `paper_daily.md`
- `paper_daily_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-13.md`
- `ideas/Idea_Proposal_2026-06-14.md`
- `ideas/Idea_Proposal_2026-06-16.md`

### 未检出但已确认搜索

- 当前仓库未检出 `my_work_summary.md`。
- 当前仓库也未检出 `*work_summary*.md`、`*summary*.md`、`*工作*.md` 或 `*总结*.md` 命名的 Markdown 工作总结文件。

### 历史核心机制黑名单

为避免与历史提案和 `paper_daily.md` 中已有机制发生思维重合，本提案明确不把以下机制作为主创新：

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
15. posterior quotient、model-space posterior factor division 或 policy likelihood factor 商运算。
16. 单纯把 DBGL/GARLIC 式 decay、二部图、time-lagged graph 或 attention 解释作为主创新。
17. iTimER 式 pseudo-observations、Wasserstein error alignment 或误差分布对比学习。
18. Record2Vec 式直接 summarize-then-embed 作为完整分类器替代方案。

本提案选择一个与上述机制正交的切入点：**不要求反事实采样下的 representation 或 logits 一致，不估计采样危险率，不做对抗、交换子、后验商或协议征税；而是把采样政策偏移看作“类别排序证据被部分审查 censoring 后的不确定排名问题”，用反事实干预学习每个类别 logit 的政策覆盖区间，并让分类决策基于区间下界/上界的鲁棒支配关系。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中最适合激发新方向的两个前沿机制是：

- **iTimER** 提醒我们：重构误差不只是训练失败信号，它能暴露不规则采样下哪些区域更不确定。但 iTimER 将误差用于 pseudo-observation 与分布对齐，可能把 policy-induced uncertainty 继续带入表示。
- **Record2Vec** 提醒我们：跨医院或跨队列迁移时，低层时间戳和 mask 细节往往不如语义化状态摘要便携。但纯摘要方法会过度压缩细粒度动态，也缺少对“摘要是否因采样政策改变而改变”的可控诊断。

采样策略偏移的核心不是所有表示都必须不变，而是：

> 当采样政策从医院 A 切到医院 B、从密集监测切到预算受限、从常规 panel 切到风险触发 panel 时，模型能否知道“当前类别排序有多大一部分是稳定状态证据，有多大一部分只是采样政策造成的排序抖动”？

历史方案大多尝试在表示层、梯度层、图算子层或后验层直接剥离采样因素。**Policy-Censored Rank Envelopes (PCRE)** 采用完全不同的决策层视角：  
模型先给出 value-derived 的中心类别证据 `center_logit`，再由 sampling branch 与 reconstruction residual 给每个类别估计一个非负政策审查半径 `radius`。最终不直接相信单点 logit，而是形成类别证据区间：

```text
lower_c = center_logit_c - radius_c
upper_c = center_logit_c + radius_c
```

分类时，只有当真实类别或预测类别的 **lower bound** 能支配其他类别的 **upper bound**，模型才做高置信判断；否则输出更保守的 policy-censored prediction set 或 abstention signal。这样，反事实采样干预不再用于制造一致性约束，而是用于学习“采样政策最多会让类别排序晃动多少”。  

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 负责产生中心类别证据；
- sampling process 负责估计政策审查半径；
- counterfactual intervention 负责提供 logit displacement 的监督；
- reconstruction residual / semantic summary 只用于校准区间宽度，不作为采样捷径直接进入分类边界。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从单点分类表征改为“状态中心证据 + 政策审查半径”

PCRE 保留现有不规则时间序列 value encoder，但将输出拆成两个互不等价的对象。

#### A. State Evidence Center

`StateEvidenceEncoder` 输入观测值、变量 id、时间戳和基础 mask，输出一个语义状态向量 `z_state` 与中心 logits：

```text
center = ClassHead(z_state)
```

这里可以轻量吸收 Record2Vec 的思想，但不调用 LLM 作为主模型：把不规则事件压缩成一组可训练的 **semantic state predicates**，例如趋势、异常幅度、恢复方向和变量组状态。它们是 value-derived summary，不直接编码“这个变量为什么被测”。

#### B. Policy Censor Radius

`PolicyCensorHead` 输入三类信息：

1. 采样结构摘要：观测密度、变量覆盖率、时间窗覆盖率、co-order 粗统计；
2. value encoder 的重构残差摘要：不同变量/时间窗的 residual quantile；
3. counterfactual policy recipe：表示本次要覆盖的策略扰动类型。

输出每个类别的非负半径：

```text
radius = softplus(Head(policy_features, residual_features, recipe))
```

关键区别：

- 采样特征不直接进入 `center` 分类头；
- residual 不生成 pseudo-observation，也不做 Wasserstein alignment；
- radius 只表示“当前采样政策会让类别证据最多漂移多少”，因此它是决策不确定性而非类别证据。

### 2.2 改 Loss：从一致性/风险方差转向“排序覆盖 + 区间支配”

总目标：

```text
L = L_center_cls
  + lambda_cov  * L_counterfactual_coverage
  + lambda_rank * L_interval_dominance
  + lambda_eff  * L_envelope_efficiency
  + lambda_cal  * L_split_conformal_calibration
```

#### A. 中心分类损失 `L_center_cls`

中心 logits 仍需具备基础判别力：

```text
L_center_cls = CE(center, y)
```

它保证 value process 不是只输出宽区间而放弃分类。

#### B. 反事实覆盖损失 `L_counterfactual_coverage`

对每条样本生成若干 `do(policy=k)` recipe。不同于历史方案，PCRE 不要求反事实下 logits 相同，也不惩罚风险方差；它只观测中心 logit 在策略干预下的位移：

```text
delta_k = abs(center_cf_k - stopgrad(center_factual))
```

半径应该覆盖这些位移的高分位数。用 pinball loss 学习目标覆盖率 `q`：

```text
u_k = radius_k - delta_k
L_counterfactual_coverage = mean(max(q * (-u_k), (q - 1) * (-u_k)))
```

直觉：如果某个类别 logit 在采样政策改变下经常大幅漂移，模型必须承认这个类别证据被政策审查得更严重，给它更宽区间。

#### C. 区间支配损失 `L_interval_dominance`

分类不再比较单点 logits，而比较真实类别下界与最强竞争类别上界：

```text
true_lower = center_y - radius_y
rival_upper = max_{c != y}(center_c + radius_c)
L_interval_dominance = relu(margin + rival_upper - true_lower)
```

这不是 prototype，也不是一致性：它直接优化“即使考虑政策扰动，真实类别排序仍能支配竞争类别”的决策边界。

#### D. 区间效率损失 `L_envelope_efficiency`

为了防止模型把所有半径无限放大，加入效率约束：

```text
L_envelope_efficiency = mean(radius_y + topk_mean(radius_non_y))
```

同时保留覆盖损失，形成“能覆盖但尽量窄”的张力。

#### E. Split Conformal Calibration `L_split_conformal_calibration`

在训练后或训练过程中维护一个 calibration buffer，记录验证环境中的非一致性分数：

```text
score_i = max(0, max_{c != y_i} upper_c - lower_{y_i})
```

推理时，用校准分位数 `q_hat` 扩展半径：

```text
radius_cal = radius + q_hat
```

这样可以把 PCRE 从普通 interval classifier 推进到 policy-shift-aware prediction set：当采样政策未知或明显偏移时，模型至少能给出覆盖更可靠的类别集合，而不是过度自信地输出错误单类。

### 2.3 改 Dataloader：返回“策略覆盖审查账本”，不是增强正样本

新增 `PolicyCensorCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `policy_recipe_bank`：反事实采样 recipe，包括：
   - `routine_to_selective`：常规监测切到风险触发监测；
   - `dense_to_budgeted`：密集采样切到预算受限采样；
   - `early_to_late_window`：早期窗口偏置切到晚期窗口偏置；
   - `panel_to_singleton`：联测 panel 切到单变量测量。
3. `residual_summary`：由 value-only reconstruction 得到的变量级/窗口级 residual quantile。
4. `semantic_predicate_target`：可选，用于训练 value branch 生成趋势、异常、恢复方向等状态摘要。

注意：collator 可以构造反事实 views 来测量 logit displacement，但这些 views 不构成 contrastive pair，不要求 representation 相等，也不用于 risk variance。它们只是监督 radius 的覆盖性。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 保留为中心证据生成器 `StateEvidenceEncoder`。
- 现有 sampling branch 改为 `PolicyCensorHead`，输出 class-wise radius，不输出 policy residual、不做 adversarial、不进入 center head。
- 现有反事实干预模块改为 `CounterfactualDisplacementOracle`：只提供不同 recipe 下 center logit 的位移，用于训练覆盖半径。
- 推理阶段输出三种结果：
  - **robust class**：若 `lower_pred > max upper_rival`，输出单类；
  - **prediction set**：若排序不稳定，输出所有 `upper_c` 仍可能超过最佳 lower 的类别；
  - **policy-censor score**：`mean(radius)` 与 `dominance_gap`，用于判断是否应采集更多观测或触发人工复核。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom.squeeze(dim).unsqueeze(-1)


class StateEvidenceEncoder(nn.Module):
    """Produce value-derived semantic state evidence for irregular events."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int, predicate_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.predicate_head = nn.Linear(hidden_dim, predicate_dim)
        self.class_head = nn.Linear(hidden_dim + predicate_dim, num_classes)
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]

        var_h = self.var_embed(var_id)
        event_x = torch.cat(
            [var_h, value.unsqueeze(-1), torch.log1p(time).unsqueeze(-1)],
            dim=-1,
        )
        event_h = self.event_net(event_x) * mask.unsqueeze(-1)
        seq_h, _ = self.temporal_mixer(event_h)
        pooled = masked_mean(seq_h, mask, dim=1)

        predicates = torch.tanh(self.predicate_head(pooled))
        center = self.class_head(torch.cat([pooled, predicates], dim=-1))
        recon = self.recon_head(seq_h).squeeze(-1)
        residual = (recon - value).abs() * mask
        return {
            "center": center,
            "state": pooled,
            "predicates": predicates,
            "recon": recon,
            "residual": residual,
        }


class PolicyCensorHead(nn.Module):
    """Estimate class-wise policy-censoring radii, not class evidence."""

    def __init__(self, num_vars: int, recipe_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.recipe_proj = nn.Linear(recipe_dim, hidden_dim)
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.policy_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.radius_head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),
        )

    def forward(self, batch: dict, residual: torch.Tensor, recipe: torch.Tensor) -> torch.Tensor:
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        value = batch["event_value"]

        var_h = self.var_embed(var_id)
        recipe_h = self.recipe_proj(recipe)[:, None, :].expand_as(var_h)

        gap = torch.zeros_like(time)
        gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        observed_density = mask.mean(dim=1, keepdim=True).expand_as(time)
        value_abs = value.abs()

        residual_mean = residual.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        residual_centered = residual - residual_mean

        policy_x = torch.cat(
            [
                var_h,
                recipe_h,
                torch.log1p(time).unsqueeze(-1),
                torch.log1p(gap).unsqueeze(-1),
                observed_density.unsqueeze(-1),
                value_abs.unsqueeze(-1),
                residual_centered.unsqueeze(-1),
            ],
            dim=-1,
        )
        token_h = self.policy_net(policy_x) * mask.unsqueeze(-1)
        pooled = masked_mean(token_h, mask, dim=1)
        return self.radius_head(pooled)


def interval_dominance_loss(
    center: torch.Tensor,
    radius: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    lower = center - radius
    upper = center + radius
    true_lower = lower.gather(1, labels[:, None]).squeeze(1)
    rival_upper = upper.masked_fill(
        F.one_hot(labels, center.size(-1)).bool(),
        -1e4,
    ).max(dim=-1).values
    return F.relu(margin + rival_upper - true_lower).mean()


def pinball_coverage_loss(radius: torch.Tensor, displacement: torch.Tensor, quantile: float = 0.9) -> torch.Tensor:
    """Quantile loss so radius covers counterfactual logit displacement."""

    error = displacement.detach() - radius
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


class PolicyCensoredRankEnvelopes(nn.Module):
    def __init__(
        self,
        num_vars: int,
        recipe_dim: int,
        hidden_dim: int,
        num_classes: int,
        predicate_dim: int = 16,
    ):
        super().__init__()
        self.state_encoder = StateEvidenceEncoder(
            num_vars=num_vars,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            predicate_dim=predicate_dim,
        )
        self.censor_head = PolicyCensorHead(
            num_vars=num_vars,
            recipe_dim=recipe_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

    def envelope(self, batch: dict, recipe: torch.Tensor) -> dict:
        state_out = self.state_encoder(batch)
        radius = self.censor_head(batch, residual=state_out["residual"], recipe=recipe)
        center = state_out["center"]
        return {
            **state_out,
            "radius": radius,
            "lower": center - radius,
            "upper": center + radius,
        }

    def prediction_set(self, batch: dict, recipe: torch.Tensor, conformal_q: float = 0.0) -> dict:
        out = self.envelope(batch, recipe)
        lower = out["lower"] - conformal_q
        upper = out["upper"] + conformal_q
        best_lower = lower.max(dim=-1, keepdim=True).values
        pred_set = upper >= best_lower
        robust_pred = lower.argmax(dim=-1)
        dominance_gap = best_lower.squeeze(-1) - upper.masked_fill(
            F.one_hot(robust_pred, upper.size(-1)).bool(),
            -1e4,
        ).max(dim=-1).values
        out.update({
            "prediction_set": pred_set,
            "robust_pred": robust_pred,
            "dominance_gap": dominance_gap,
        })
        return out

    def training_loss(
        self,
        batch: dict,
        lambda_cov: float = 0.4,
        lambda_rank: float = 0.5,
        lambda_eff: float = 0.02,
        lambda_sem: float = 0.05,
        quantile: float = 0.9,
    ) -> dict:
        labels = batch["labels"]
        factual_recipe = batch["policy_recipe"]
        out = self.envelope(batch, factual_recipe)

        center_loss = F.cross_entropy(out["center"], labels)
        rank_loss = interval_dominance_loss(out["center"], out["radius"], labels)
        efficiency_loss = out["radius"].mean()

        coverage_terms = []
        for recipe, cf_batch in zip(
            batch["policy_recipe_bank"].unbind(dim=1),
            batch["counterfactual_batches"],
        ):
            cf_center = self.state_encoder(cf_batch)["center"]
            displacement = (cf_center - out["center"].detach()).abs()
            cf_radius = self.censor_head(batch, residual=out["residual"], recipe=recipe)
            coverage_terms.append(pinball_coverage_loss(cf_radius, displacement, quantile=quantile))

        if coverage_terms:
            coverage_loss = torch.stack(coverage_terms).mean()
        else:
            coverage_loss = torch.zeros((), device=out["center"].device)

        # Optional semantic predicate supervision, used only for value-derived state summaries.
        if "semantic_predicate_target" in batch:
            semantic_loss = F.smooth_l1_loss(
                out["predicates"],
                batch["semantic_predicate_target"],
            )
        else:
            semantic_loss = torch.zeros((), device=out["center"].device)

        total = (
            center_loss
            + lambda_cov * coverage_loss
            + lambda_rank * rank_loss
            + lambda_eff * efficiency_loss
            + lambda_sem * semantic_loss
        )
        return {
            "loss": total,
            "center_loss": center_loss.detach(),
            "coverage_loss": coverage_loss.detach(),
            "rank_loss": rank_loss.detach(),
            "efficiency_loss": efficiency_loss.detach(),
            "semantic_loss": semantic_loss.detach(),
            "mean_radius": out["radius"].mean().detach(),
        }


@torch.no_grad()
def conformal_policy_quantile(model: PolicyCensoredRankEnvelopes, calib_loader, device: torch.device, alpha: float = 0.1) -> torch.Tensor:
    """Estimate split-conformal expansion for policy-censored prediction sets."""

    scores = []
    for batch in calib_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model.envelope(batch, batch["policy_recipe"])
        labels = batch["labels"]
        true_lower = out["lower"].gather(1, labels[:, None]).squeeze(1)
        rival_upper = out["upper"].masked_fill(
            F.one_hot(labels, out["upper"].size(-1)).bool(),
            -1e4,
        ).max(dim=-1).values
        scores.append(F.relu(rival_upper - true_lower))

    scores = torch.cat(scores).sort().values
    index = torch.ceil(torch.tensor((1.0 - alpha) * (scores.numel() + 1))).long() - 1
    index = index.clamp(0, scores.numel() - 1)
    return scores[index]
```

## 4. 预期创新性

1. **从表示鲁棒转向排序覆盖鲁棒**：不再追求反事实视图表征一致，而是学习类别排序在采样政策扰动下的可覆盖区间。
2. **从单点 logit 转向 policy-censored interval logit**：每个类别都有中心证据和政策审查半径，决策基于 lower/upper 的支配关系。
3. **从误差生成伪观测转向误差校准决策半径**：吸收 iTimER “误差有信息”的洞察，但不用 pseudo-observation、Wasserstein alignment 或 contrastive objective。
4. **从 LLM 摘要替代模型转向轻量语义谓词辅助**：吸收 Record2Vec 的便携语义层思想，但保留可训练时序主干和细粒度反事实审查。
5. **自然支持可信部署**：当采样政策偏移过大时，模型输出 prediction set 或 abstention，而不是给出过度自信的错误单类。

## 5. 实验切入点

1. 构造四类 policy shift：
   - routine monitoring 与 selective monitoring 互换；
   - dense sampling 与 budgeted sparse sampling 互换；
   - early-window bias 与 late-window bias 互换；
   - panel co-ordering 与 singleton observation 互换。
2. 对比：
   - missingness-aware encoder；
   - mask dropout / random thinning；
   - policy adversarial baseline；
   - hazard/null-space baseline；
   - commutator graph surgery baseline；
   - protocol-tax evidence market baseline；
   - posterior quotient dynamics baseline；
   - iTimER / Record2Vec 风格启发模型。
3. 新增指标：
   - worst-policy accuracy；
   - set-valued coverage under policy shift；
   - average prediction set size；
   - robust dominance gap；
   - radius-displacement calibration error；
   - abstention benefit under severe policy shift。
4. 消融：
   - 去掉 `L_counterfactual_coverage`，验证半径是否失去政策覆盖语义；
   - 去掉 `L_interval_dominance`，验证区间只会变成普通不确定性估计；
   - 去掉 reconstruction residual features，验证 iTimER 式误差信号对半径校准的价值；
   - 去掉 semantic predicate auxiliary，验证 Record2Vec 式语义中介是否改善跨机构排序稳定性；
   - 把 class-wise radius 改为 scalar radius，验证类别特异政策审查的重要性。

## 6. 一句话投稿卖点

**PCRE 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“类别排序证据被采样政策审查后的覆盖决策问题”，通过反事实 logit displacement 学习 class-wise policy-censor envelopes，使模型在策略偏移下不仅追求准确率，还能给出可校准的鲁棒类别支配、预测集合与拒判信号。**
