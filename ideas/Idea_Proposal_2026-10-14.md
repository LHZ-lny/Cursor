# Title: Do-Counterchoice Utility Balancer：面向采样策略偏移的反选择效用均衡分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前 `/workspace` 未检出该文件；同时搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`，也未发现可替代工作总结文件。
- 已读取自动化持久记忆 `MEMORIES.md` 与最新 `idea_2026-09-30.md`，纳入历史 proposal 摘要和长期黑名单。
- 已读取/检索当前仓库 `ideas/Idea_Proposal_*.md` 的历史标题、黑名单、核心机制段，并完整复核最近多份提案正文，覆盖当前工作区已落盘的 2026-06-12 至 2026-09-30 提案。
- 已读取最新 `paper_daily_2026-10-13.md`，重点吸收：
  - **PRISM / Domain Generalization under Sampling Pattern Shifts in Irregular Time Series**：把 sampling pattern shift 明确作为独立跨域问题，并区分 feature-centric 与 sampling-centric representation。
  - **Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness**：把模态缺失看成 clinical decision / assignment mechanism 导致的 MMNAR，而不是普通 mask 噪声。

### 历史核心机制黑名单

本提案显式避开以下已出现主机制：

1. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
2. 生理流/采样算子交换子、state-policy graph 拆分、policy residual sink。
3. protocol tax、additive evidence market、token 边际审计、证据预算。
4. posterior quotient、Radon-Nikodym density ratio、doubly robust correction、RKHS cubature。
5. reconstruction error cartography、ANOVA projection、VQ clauses、HSIC redaction。
6. optional-stopping martingale、拓扑胶囊、policy gauge、syndrome code、knockoff calendar。
7. observability witness、evidential vacuity、policy lattice、solver trace front-door、conformal sleeve、IV/control-function。
8. Borda jury、Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock。
9. IRT-DIF、RG fixed point、bitemporal curtain、clinical tomography、matched risk-set likelihood、policy privacy cloak。
10. synthetic falsification forge、dialectical JEPA referee、PID semantic prism、Noether semantic action、meta workflow immunization。
11. collider seal、source-atlas antipodal retrieval、Schwarzian warp canonicalizer、renewal exposure meter、palimpsest memory。
12. typed SSA observation program、Robin boundary field solver、Blackwell experiment order、elicitable functional compass。
13. Lyapunov interval observer、BBP / Marchenko-Pastur random-matrix spike sieve、RIP sparse pathology recovery。
14. factorial cumulant thinning algebra、repeat/panel cumulant subtraction、Poisson clutter null。
15. Delta-XAI / SWING 风险跃迁熔断、prediction-change attribution consistency、policy interrupt buffer。
16. B-spline knot-vector framing、Boehm knot insertion/removal、canonical disease control polygon、knot residual buffer。
17. ReTAMamba reliability tensor 原子解混、nonnegative anchor-cone freshness demixer、state/policy freshness atoms。
18. Cauchy contour / meromorphic disease pole / residue firewall。
19. microlocal semantic wavefront、policy conormal directions、directional singular energy、cone margin separation。
20. 单纯 state-policy 双分支、对抗环境分类器、跨视图 logits/representation 一致性、频域掩码对比学习、missingness pattern 直接分类、普通 retrieval-augmented classifier、普通 dual-SSM classifier。

本轮选择新的正交切入点：**不把采样政策建成概率密度、图、谱、样条、复轮廓、波前、程序、证明、记忆、实验序、可靠性锥或高阶计数；而是把每一次观测看成医生/设备/医院流程在一个“可行测量选择集”中的被选动作。采样策略偏移首先改变的是可选项集合与选择偏好，而不是观测值本身。分类器不读取“为什么被选中”的效用痕迹，只读取实际观测值相对未选备选项基线所产生的 value-supported counterchoice surplus。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-10-13.md` 中的两条前沿线索非常适合与现有“采样解耦/反事实干预”框架结合：

- **PRISM** 证明 sampling pattern shift 可以单独造成严重泛化退化，传统 IMTS 模型会把 mask、delta-t、变量可见性、观测密度当成 shortcut。
- **MMNAR causal representation** 强调临床记录是否出现，本质上是 clinical decision / assignment mechanism 的结果：检查、影像、文本和数值变量的可见性同时受患者状态、医生偏好、医院流程和资源约束影响。

历史方案已经大量处理“采样概率”“采样图结构”“观测拓扑”“采样日历”“可靠性权重”“样条 knots”“风险跃迁”“复轮廓”和“语义波前”。本提案换一个尚未占用的对象：

> 非规则观测不是孤立 token，而是一次 **被选中的测量选择**。当乳酸被测时，临床上同时存在很多未选备选项：复测血压、查血气、等待下一轮 routine panel、记录文本、调用影像，或者什么都不测。不同医院的 sampling policy shift 往往不是“观测值分布改变”，而是“在同一状态下偏好选择哪类测量动作”改变。

如果模型只看被选中的观测，它会把“这个变量被选了”“这个时间被选了”“这个模态出现了”误当成类别证据。**Do-Counterchoice Utility Balancer (DCUB)** 的核心主张是：

```text
observed event evidence
    = value-supported clinical surplus
    + chosen-action utility trace
    + choice-set availability artifact
```

分类器只能读取第一项：实际观测值相对同一上下文中未选备选项的语义盈余。选择效用、可选项集合大小、未选项类型、医院偏好温度和模态可用性只进入 policy audit head，不进入主分类头。

这与“采样解耦/反事实干预”框架的结合方式是：

- value process：编码实际测到的值、趋势新息、阈值跨越和跨变量临床语义。
- sampling process：编码 choice set、chosen action、unchosen alternatives、availability、ordering preference。
- counterfactual intervention：不再生成粗暴的多视图一致正样本，而是生成 **counterchoice swaps**：只改变“本来会选哪项测量/哪些备选项可用”，不改变真实病程值。
- classifier：只读取 chosen measurement 相对 counterchoice baseline 的 value surplus；如果只是医院 A 更爱测乳酸、医院 B 更爱测血气，policy utility 可以变，但 state surplus 不应凭选择偏好变大。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Counterchoice Set Bank

新增 `CounterchoiceSetCollator`。每个事实事件不只返回 `(time, variable, value, mask)`，还返回一个局部可行选择集：

```text
event_i = {
    chosen_value_i,
    chosen_time_i,
    chosen_variable_i,
    chosen_modality_i,
    context_i,
    choice_set_i = [a_i^0, a_i^1, ..., a_i^K],
    chosen_index_i,
}

alternative a_i^k = {
    alt_time,
    alt_variable,
    alt_modality,
    alt_availability,
    alt_cost_proxy,
    alt_workflow_tag,
}
```

其中 `a_i^chosen` 是实际发生的测量动作，其他 `a_i^k` 是同一局部上下文中的可行但未选动作。未选项不需要真实值；它们只携带“如果医院流程选择另一种测量，会暴露什么选择偏好”的上下文特征。

反事实视图包括：

1. `counterchoice_swap`
   - 保持真实值和病程上下文不变，只把 chosen action 替换为另一个可行备选项。
   - 用于模拟医院偏好不同：同样病人状态下，A 医院选择测 lactate，B 医院选择测 blood gas。
   - 不能凭 chosen action 变化增加 state surplus。

2. `availability_shift`
   - 增删部分未选备选项，模拟设备可用性、夜班资源、医院 panel 配置变化。
   - 可改变 policy utility entropy，但不应改变 value-supported surplus。

3. `preference_temperature_shift`
   - 改变选择效用温度或 top-k 选择尖锐度，模拟不同机构的 order aggressiveness。
   - 不改变测到的值，不应改变分类证据。

4. `modality_choice_shift`
   - 在数值、文本、影像、护理记录之间替换可选模态，吸收 MMNAR 的临床记录决策视角。
   - 只影响 assignment audit，不直接进入分类头。

5. `semantic_value_release`
   - 保持 choice set 不变，注入真实 value novelty、阈值跨越、趋势反转或跨变量一致恶化。
   - 必须释放 state surplus，避免模型把所有选择相关变化都压成不变。

### 2.2 改 Encoder：Chosen-vs-Counterchoice Surplus Encoder

#### A. Alternative Utility Stem

先对每个可行备选动作编码：

```text
c_i^k = AltStem(alt_time, alt_variable, alt_modality, availability, cost_proxy, workflow_tag)
u_i^k = UtilityHead(context_i, c_i^k)
```

`u_i^k` 只解释“为什么这个动作被选中”，不作为 inverse propensity、不做 density ratio、不进入主分类器。

#### B. Counterchoice Baseline

对未选备选项形成一个局部选择偏好基线：

```text
b_i = sum_{k != chosen} softmax(u_i^k / tau) * c_i^k
```

这个 baseline 表示：在相同上下文下，采样流程本可选择的其他记录动作会带来的政策痕迹。它不含未观测值，因此不会偷看反事实结果。

#### C. Value-Supported Surplus Token

实际测量值通过 value stem 得到 `v_i`，chosen action 上下文为 `c_i^chosen`。DCUB 只把以下盈余交给分类器：

```text
s_i = Gate(value_novelty_i, trend_delta_i, threshold_cross_i)
      * Balance(v_i + c_i^chosen - b_i)
```

直觉：

- 如果一个事件只是因为医院偏好选择了某变量，`c_i^chosen - b_i` 会暴露选择效用痕迹，但缺少 value novelty，gate 不放行。
- 如果同样 choice set 下值真的发生语义新息，`v_i` 会相对 counterchoice baseline 产生 surplus，进入分类头。
- 如果换一个医院偏好导致选择集和 chosen action 变化，policy head 可以记录变化，但主分类 token 不随选择偏好放大。

#### D. Safe Temporal Classifier

分类器只读取 surplus tokens：

```text
safe_sequence = TemporalEncoder({s_i})
logits = Classifier(pool(safe_sequence))
```

主路径不接收 raw mask、delta-t histogram、event count、chosen action id、choice-set size、availability、workflow tag、hospital id、utility logits 或 policy entropy。

### 2.3 改 Loss：Counterchoice Utility Discipline

总目标：

```text
L = L_cls
  + lambda_choice * L_revealed_choice_fit
  + lambda_bal    * L_counterchoice_balance
  + lambda_rel    * L_semantic_surplus_release
  + lambda_ent    * L_counterchoice_entropy_floor
  + lambda_audit  * L_choice_label_audit
```

#### A. Safe Classification `L_cls`

```text
L_cls = CE(Classifier(surplus_tokens), y)
```

#### B. Revealed Choice Fit `L_revealed_choice_fit`

sampling branch 学会解释实际被选动作：

```text
L_choice = CE(utility_logits over choice_set, chosen_index)
```

注意：它不是 inverse-propensity weighting，也不把选择概率用于风险校正；它只把“为何被测”封装到 assignment utility ledger。

#### C. Counterchoice Balance `L_counterchoice_balance`

对 `counterchoice_swap`、`availability_shift`、`preference_temperature_shift`：

```text
L_bal = || summary(surplus_policy_view) - stopgrad(summary(surplus_factual)) ||_1
```

这里约束的不是整段 hidden representation 或 logits，而是 **chosen-minus-counterchoice surplus ledger**。policy-only 的选择偏好变化可以改变 utility logits 和 audit 指标，但不应凭空改变 value surplus。

#### D. Semantic Surplus Release `L_semantic_surplus_release`

对 `semantic_value_release`：

```text
target_i = value_surprise_i + trend_flip_i + threshold_cross_i
L_rel = BCE(release_logit_i, target_i)
      + relu(min_surplus_gain - ||s_i^semantic - s_i^factual||)^2
```

这避免模型过度均衡：真实数值新息必须穿过 counterchoice baseline。

#### E. Counterchoice Entropy Floor `L_counterchoice_entropy_floor`

为防止 baseline 退化成单个伪备选项，要求未选项权重保持最低有效支持：

```text
L_ent = relu(H_min - H(softmax(u_unchosen)))^2
```

这不是 evidence budget 或 protocol tax；它只保证“反选择基线”确实来自多个可行未选动作。

#### F. Choice Label Audit `L_choice_label_audit`

训练一个只读 policy utility statistics 的 stop-gradient probe：

```text
probe_logits = Probe(stopgrad([utility_hist, choice_entropy, availability_stats]))
L_audit = CE(probe_logits, y)
```

该项主要作为报告指标。如果它很强，说明数据中存在严重 assignment-label coupling；主分类器仍无法读取这些 utility statistics。

---

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def masked_entropy(prob: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = prob.clamp_min(eps)
    ent = -(p * p.log() * mask.to(p.dtype)).sum(dim=-1)
    denom = mask.to(p.dtype).sum(dim=-1).clamp_min(1.0).log().clamp_min(1.0)
    return ent / denom


class ValueEventStem(nn.Module):
    """Encode measured values; unchosen alternatives never receive hidden values."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_emb = nn.Embedding(num_vars, hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.LayerNorm(hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        value = batch["event_value"]
        quality = batch.get("measurement_quality", torch.ones_like(value))
        novelty = batch.get("value_novelty", value.abs())
        trend = batch.get("trend_delta", torch.zeros_like(value))
        threshold = batch.get("threshold_cross", torch.zeros_like(value))
        var_id = batch["event_var_id"].clamp_min(0)

        feat = torch.stack(
            [value, torch.tanh(value), quality, novelty, trend + threshold],
            dim=-1,
        )
        return self.out(self.value_proj(feat) + self.var_emb(var_id))


class AlternativeUtilityStem(nn.Module):
    """Encode feasible measurement actions and their revealed-choice utilities."""

    def __init__(self, num_vars: int, num_modalities: int, hidden_dim: int):
        super().__init__()
        self.var_emb = nn.Embedding(num_vars, hidden_dim)
        self.mod_emb = nn.Embedding(num_modalities, hidden_dim)
        self.alt_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        alt_var = batch["choice_alt_var_id"].clamp_min(0)          # [B, T, A]
        alt_mod = batch["choice_alt_modality"].clamp_min(0)        # [B, T, A]
        alt_time = batch["choice_alt_time"]                        # [B, T, A]
        availability = batch["choice_alt_available"].float()       # [B, T, A]
        cost = batch.get("choice_alt_cost", torch.zeros_like(alt_time))
        workflow = batch.get("choice_alt_workflow", torch.zeros_like(alt_time))

        time = batch["event_time"]
        mask = batch["event_mask"].float()
        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        alt_feat = torch.cat(
            [
                self.var_emb(alt_var),
                self.mod_emb(alt_mod),
                alt_time.unsqueeze(-1),
                availability.unsqueeze(-1),
                cost.unsqueeze(-1),
                workflow.unsqueeze(-1),
                (alt_time - time.unsqueeze(-1)).unsqueeze(-1),
            ],
            dim=-1,
        )
        alt_h = self.alt_proj(alt_feat)

        context_var = self.var_emb(batch["event_var_id"].clamp_min(0))
        context_feat = torch.cat(
            [
                context_var,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                batch.get("value_novelty", torch.zeros_like(time)).unsqueeze(-1),
                mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        context_h = self.context_proj(context_feat)
        context_h = context_h.unsqueeze(2).expand_as(alt_h)

        utility = self.utility_head(torch.cat([context_h, alt_h], dim=-1)).squeeze(-1)
        utility = utility.masked_fill(availability <= 0, -1e4)
        return {"alt_h": alt_h, "utility": utility, "availability": availability}


class CounterchoiceSurplusEncoder(nn.Module):
    """Compute chosen-minus-counterchoice surplus tokens for classification."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.balance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.release_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        value_h: torch.Tensor,
        alt_h: torch.Tensor,
        utility: torch.Tensor,
        availability: torch.Tensor,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        bsz, steps, num_alt, hidden_dim = alt_h.shape
        chosen = batch["chosen_alt_index"].clamp(0, num_alt - 1)
        chosen_onehot = F.one_hot(chosen, num_classes=num_alt).to(availability.dtype)
        unchosen_mask = availability * (1.0 - chosen_onehot)

        chosen_h = torch.gather(
            alt_h,
            dim=2,
            index=chosen.view(bsz, steps, 1, 1).expand(-1, -1, 1, hidden_dim),
        ).squeeze(2)

        unchosen_logits = (utility / temperature).masked_fill(unchosen_mask <= 0, -1e4)
        unchosen_weight = torch.softmax(unchosen_logits, dim=-1) * unchosen_mask
        unchosen_weight = unchosen_weight / unchosen_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        baseline = torch.einsum("bta,btah->bth", unchosen_weight, alt_h)

        raw_surplus = value_h + chosen_h - baseline
        balanced = self.balance(raw_surplus)

        novelty = batch.get("value_novelty", batch["event_value"].abs())
        trend = batch.get("trend_delta", torch.zeros_like(novelty))
        threshold = batch.get("threshold_cross", torch.zeros_like(novelty))
        release_input = torch.cat([balanced, novelty.unsqueeze(-1), trend.unsqueeze(-1), threshold.unsqueeze(-1)], dim=-1)
        release_logit = self.release_head(release_input).squeeze(-1)
        release_gate = torch.sigmoid(release_logit)

        mask = batch["event_mask"].float()
        surplus_token = balanced * release_gate.unsqueeze(-1) * mask.unsqueeze(-1)
        return {
            "surplus_token": surplus_token,
            "raw_surplus": raw_surplus,
            "release_logit": release_logit,
            "release_gate": release_gate,
            "counterchoice_weight": unchosen_weight,
            "choice_entropy": masked_entropy(unchosen_weight, unchosen_mask),
            "chosen_utility": (utility * chosen_onehot).sum(dim=-1),
            "utility": utility,
        }


class SafeTemporalClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, surplus_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq, _ = self.gru(surplus_token * mask.unsqueeze(-1).to(surplus_token.dtype))
        pooled = masked_mean(seq, mask, dim=1)
        return self.classifier(pooled)


class DoCounterchoiceUtilityBalancer(nn.Module):
    def __init__(
        self,
        num_vars: int,
        num_modalities: int,
        num_classes: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.value_stem = ValueEventStem(num_vars, hidden_dim)
        self.alt_stem = AlternativeUtilityStem(num_vars, num_modalities, hidden_dim)
        self.surplus = CounterchoiceSurplusEncoder(hidden_dim)
        self.safe_classifier = SafeTemporalClassifier(hidden_dim, num_classes)
        self.choice_label_probe = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, batch: dict[str, torch.Tensor], temperature: float = 1.0) -> dict[str, torch.Tensor]:
        value_h = self.value_stem(batch)
        alt = self.alt_stem(batch)
        out = self.surplus(
            batch=batch,
            value_h=value_h,
            alt_h=alt["alt_h"],
            utility=alt["utility"],
            availability=alt["availability"],
            temperature=temperature,
        )
        logits = self.safe_classifier(out["surplus_token"], batch["event_mask"])

        policy_stats = torch.stack(
            [
                masked_mean(out["chosen_utility"], batch["event_mask"], dim=1),
                masked_mean(out["choice_entropy"], batch["event_mask"], dim=1),
                masked_mean(alt["availability"].sum(dim=-1), batch["event_mask"], dim=1),
                masked_mean(out["release_gate"].detach(), batch["event_mask"], dim=1),
            ],
            dim=-1,
        )
        probe_logits = self.choice_label_probe(policy_stats.detach())
        return {**alt, **out, "logits": logits, "policy_stats": policy_stats, "probe_logits": probe_logits}

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        lambda_choice: float = 0.20,
        lambda_balance: float = 0.35,
        lambda_release: float = 0.30,
        lambda_entropy: float = 0.05,
        lambda_audit: float = 0.02,
        entropy_floor: float = 0.35,
    ) -> dict[str, torch.Tensor]:
        labels = batch["labels"]
        factual = self.encode(batch["factual_view"])
        cls_loss = F.cross_entropy(factual["logits"], labels)

        choice_mask = batch["factual_view"]["event_mask"].bool()
        choice_logits = factual["utility"][choice_mask]
        choice_target = batch["factual_view"]["chosen_alt_index"][choice_mask].long()
        choice_loss = F.cross_entropy(choice_logits, choice_target) if choice_logits.numel() else cls_loss.new_zeros(())

        # Policy-only edits may alter revealed utilities, not the value-supported surplus ledger.
        balance_terms = []
        factual_summary = masked_mean(factual["surplus_token"], batch["factual_view"]["event_mask"], dim=1).detach()
        for view in batch.get("policy_choice_views", []):
            view_out = self.encode(view, temperature=float(view.get("utility_temperature", 1.0)))
            view_summary = masked_mean(view_out["surplus_token"], view["event_mask"], dim=1)
            balance_terms.append(F.smooth_l1_loss(view_summary, factual_summary))
        balance_loss = torch.stack(balance_terms).mean() if balance_terms else cls_loss.new_zeros(())

        semantic_view = batch.get("semantic_value_release_view")
        if semantic_view is not None:
            sem = self.encode(semantic_view)
            target = semantic_view.get("semantic_target", torch.ones_like(sem["release_logit"]))
            release_loss = F.binary_cross_entropy_with_logits(
                sem["release_logit"],
                target.float(),
                reduction="none",
            )
            release_loss = (release_loss * semantic_view["event_mask"]).sum() / semantic_view["event_mask"].sum().clamp_min(1.0)

            factual_norm = factual["surplus_token"].norm(dim=-1).detach()
            sem_norm = sem["surplus_token"].norm(dim=-1)
            gain_target = semantic_view.get("semantic_gain_target", target.float())
            gain_loss = F.relu(gain_target - (sem_norm - factual_norm)).pow(2)
            release_loss = release_loss + (gain_loss * semantic_view["event_mask"]).sum() / semantic_view["event_mask"].sum().clamp_min(1.0)
        else:
            release_loss = cls_loss.new_zeros(())

        entropy_loss = F.relu(entropy_floor - factual["choice_entropy"]).pow(2)
        entropy_loss = (entropy_loss * batch["factual_view"]["event_mask"]).sum() / batch["factual_view"]["event_mask"].sum().clamp_min(1.0)

        # Audit only: the classifier cannot access policy_stats, but this reveals dataset leakage.
        audit_loss = F.cross_entropy(factual["probe_logits"], labels)

        total = (
            cls_loss
            + lambda_choice * choice_loss
            + lambda_balance * balance_loss
            + lambda_release * release_loss
            + lambda_entropy * entropy_loss
            + lambda_audit * audit_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "revealed_choice_fit_loss": choice_loss.detach(),
            "counterchoice_balance_loss": balance_loss.detach(),
            "semantic_surplus_release_loss": release_loss.detach(),
            "counterchoice_entropy_loss": entropy_loss.detach(),
            "choice_label_audit_loss": audit_loss.detach(),
            "mean_choice_entropy": factual["choice_entropy"].mean().detach(),
            "mean_release_gate": factual["release_gate"].mean().detach(),
        }
```

### Collator 草稿

```python
import torch


@torch.no_grad()
def build_counterchoice_set_batch(
    batch: dict[str, torch.Tensor],
    num_vars: int,
    num_modalities: int,
    num_alternatives: int = 6,
) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
    """Create feasible unchosen measurement alternatives and policy-only choice edits."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    device = value.device
    bsz, steps = value.shape

    alt_var = torch.empty(bsz, steps, num_alternatives, device=device, dtype=var_id.dtype)
    alt_mod = torch.empty_like(alt_var)
    alt_time = torch.empty(bsz, steps, num_alternatives, device=device, dtype=time.dtype)
    available = torch.ones(bsz, steps, num_alternatives, device=device, dtype=value.dtype)
    alt_cost = torch.zeros_like(alt_time)
    workflow = torch.zeros_like(alt_time)

    # Slot 0 is the observed action. Other slots are feasible counterchoices.
    alt_var[..., 0] = var_id
    alt_mod[..., 0] = batch.get("event_modality", torch.zeros_like(var_id))
    alt_time[..., 0] = time

    for k in range(1, num_alternatives):
        alt_var[..., k] = (var_id + k) % num_vars
        alt_mod[..., k] = (alt_mod[..., 0] + k) % num_modalities
        jitter = (k - num_alternatives // 2) * 0.01
        alt_time[..., k] = (time + jitter).clamp_min(0.0)
        alt_cost[..., k] = float(k) / max(1, num_alternatives - 1)
        workflow[..., k] = float(k % 3)

    factual = dict(batch)
    factual["choice_alt_var_id"] = alt_var
    factual["choice_alt_modality"] = alt_mod
    factual["choice_alt_time"] = alt_time
    factual["choice_alt_available"] = available * mask.unsqueeze(-1)
    factual["choice_alt_cost"] = alt_cost
    factual["choice_alt_workflow"] = workflow
    factual["chosen_alt_index"] = torch.zeros(bsz, steps, device=device, dtype=torch.long)
    factual["value_novelty"] = batch.get("value_novelty", value.abs())
    factual["trend_delta"] = batch.get("trend_delta", torch.zeros_like(value))
    factual["threshold_cross"] = batch.get("threshold_cross", torch.zeros_like(value))

    def clone_view(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.clone() if torch.is_tensor(v) else v for k, v in src.items()}

    policy_views = []

    # 1. Counterchoice swap: a different feasible action would have been selected.
    swap = clone_view(factual)
    swap["chosen_alt_index"] = torch.randint(1, num_alternatives, (bsz, steps), device=device)
    swap["chosen_alt_index"] = torch.where(mask.bool(), swap["chosen_alt_index"], torch.zeros_like(swap["chosen_alt_index"]))
    policy_views.append(swap)

    # 2. Availability shift: some unchosen options disappear or become available.
    availability_shift = clone_view(factual)
    keep = torch.ones_like(available)
    keep[..., 2::2] = 0.0
    keep[..., 0] = 1.0
    availability_shift["choice_alt_available"] = available * keep * mask.unsqueeze(-1)
    policy_views.append(availability_shift)

    # 3. Preference temperature shift: same choice set, more aggressive utility.
    temp = clone_view(factual)
    temp["utility_temperature"] = 0.45
    policy_views.append(temp)

    # 4. Modality choice shift: available modalities rotate while values stay fixed.
    modality = clone_view(factual)
    modality["choice_alt_modality"] = (modality["choice_alt_modality"] + 1) % num_modalities
    policy_views.append(modality)

    # Semantic release: same choice set, true value novelty is injected.
    semantic = clone_view(factual)
    surprise = semantic["value_novelty"]
    threshold = surprise.quantile(0.75, dim=1, keepdim=True)
    inject = (surprise >= threshold).to(value.dtype) * mask
    semantic["event_value"] = value + 0.5 * inject * torch.sign(value + 1e-3)
    semantic["semantic_target"] = inject
    semantic["semantic_gain_target"] = 0.1 + inject

    return {
        "factual_view": factual,
        "policy_choice_views": policy_views,
        "semantic_value_release_view": semantic,
        "labels": batch["labels"],
    }
```

---

## 4. 实验切入点

1. **跨采样政策 split**
   - source hospital 偏好 high-cost labs，target hospital 偏好 routine vitals；
   - source 有固定 lab panel，target 按症状单项下单；
   - source 倾向文本记录，target 倾向结构化变量；
   - source alarm-aggressive，target resource-conservative；
   - PRISM 风格 HAR-C / IMTS controlled sampling pattern shift。

2. **Counterchoice leakage diagnostics**
   - 只用 `policy_stats = [utility_hist, choice_entropy, availability_stats]` 训练 probe。
   - 若 probe 对标签强，说明 assignment mechanism 本身携带 shortcut；DCUB 主分类器仍不能读取它。

3. **对比方法**
   - PRISM-style feature/sampling representation DG；
   - MMNAR-aware fusion / missingness-conditioned fusion；
   - GRU-D、mTAN、Raindrop、TimeCHEAT、ReTAMamba、TANDEM；
   - 现有采样解耦/反事实干预的普通 state-policy branch。

4. **关键消融**
   - 去掉 counterchoice baseline，只用 chosen event value；
   - 让 classifier 读取 utility logits，验证院内性能可能升高但跨政策退化；
   - 去掉 `L_revealed_choice_fit`，验证无法稳定识别选择痕迹；
   - 去掉 `L_semantic_surplus_release`，检查是否过度均衡导致真实恶化漏检；
   - 未选备选项数 `K` 从 1 到 12 扫描，验证 choice-set 覆盖对鲁棒性的影响。

5. **可解释性输出**
   - 每个样本显示 chosen action 与 top unchosen alternatives；
   - 显示 counterchoice surplus 最大的真实值新息；
   - 显示 policy utility 高但 surplus 低的采样偏好事件；
   - 跨医院展示 choice entropy / availability shift 与主分类 margin 的解耦程度。

---

## 5. 预期创新性

1. **从采样模式特征转向选择集反事实**  
   PRISM 把 sampling-centric representation 拆出来；DCUB 进一步追问“这次观测是在什么可行选择集中被选中的”，把 sampling pattern shift 落到 chosen-vs-unchosen clinical actions。

2. **从 MMNAR mask 条件化转向未选备选项均衡**  
   MMNAR-aware fusion 识别模态是否出现；DCUB 显式构造未出现但可行的 counterchoices，让模型学习实际值相对未选动作的语义盈余，而不是把“出现/未出现”本身当证据。

3. **不是概率校正或密度比**  
   revealed choice utility 只用于构造 policy audit 和 counterchoice baseline，不做 inverse propensity weighting、不估计采样 hazard、不做 Radon-Nikodym 校正。

4. **不是普通双分支或跨视图一致性**  
   分离对象不是隐藏表示，也不是让所有视图 logits 一致；约束对象是每个事件的 `chosen_value + chosen_action - unchosen_baseline` surplus ledger。

5. **可自然扩展到多模态临床记录**  
   数值化验、影像、文本、护理记录都可以作为 choice set 中的 alternative action，因此能把变量级非规则采样和模态级 MMNAR 放进同一框架。

## 6. 一句话投稿卖点

**Do-Counterchoice Utility Balancer 首次把非规则采样策略偏移表述为“临床可行测量选择集中的被选动作偏移”：模型学习为何某项测量被选择，但主分类器只读取实际观测值相对未选备选项基线的 value-supported counterchoice surplus，从而避免把医院下单偏好、模态可用性、panel 配置或选择温度误当成稳定病理证据。**
