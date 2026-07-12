# Title: Policy-Induced Ignorance Evidential Shields：面向采样策略偏移的策略诱导无知证据盾

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已读取自动化记忆：确认 2026-06-13 至 2026-06-27 多轮任务中也未发现 `my_work_summary.md` 或可替代总结文件。
- 已读取 `paper_daily.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案核心机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`
  - `Idea_Proposal_2026-06-24.md`
  - `Idea_Proposal_2026-06-27.md`

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
26. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing、MedMamba 的 frequency/adaptive graph branch、MedSpaformer 的普通 token sparsification、MILM 的 value-redacted classifier、StarEmbed 的直接 foundation embedding 或 LLMTS 的普通 LLM alignment 作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗，不要求多视图一致，不做平滑认证、后验除法、拓扑审查、gauge 投影、纠错码、knockoff 负控或观测性 gating；而是把采样策略偏移转化为主观逻辑中的“策略诱导无知质量”。模型在证据层面承认某些类别概率因为采样政策不可判定，并用 evidential shield 把这部分不可判定性显式保留为 vacuity，而不是把它伪装成高置信分类证据。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列分类中的最大部署风险，不只是测试医院或设备改变了 mask ratio，而是训练好的模型在面对新采样政策时仍然给出过度自信预测。很多历史方法都试图把采样信息切除、投影、折扣、审计或修复；但真实场景中，采样信息有时确实承载状态线索。例如某项化验被触发可能反映医生怀疑，某个波段未观测可能反映天气和测量误差，某段 ICU 记录稀疏也可能意味着病情稳定。问题不是“永远不能用采样信息”，而是：

> 当采样政策改变导致某个类别证据不可判定时，模型是否愿意把概率质量放进“我不知道”，而不是硬塞给某个类别？

近期 `paper_daily.md` 中两个机制给了关键启发：

1. **StarEmbed** 强调真实不规则观测不仅有异步采样，还有异方差测量误差、观测质量和 OOD detection。对 sampling-policy shift 来说，采样政策会改变“某类状态信息是否可恢复”，因此鲁棒模型应该报告可判定性与无知，而不是只报告 softmax 置信度。
2. **Rethinking LLMs for Irregular Time Series Classification in Critical Care** 指出，解决 ICU irregular classification 的关键在 encoder 是否正确处理时间戳、缺失和异步，而不是后端 LLM alignment。由此可见，前端 encoder 需要先产生经过策略审计的证据质量，再交给决策层。

**Policy-Induced Ignorance Evidential Shields (PIIES)** 的核心直觉是：采样分支不再做去偏器、概率估计器、负控生成器或纠错器，而是学习一个 **policy-induced ignorance shield**。它估计在当前观测政策下，每个类别证据有多少属于可判定 belief，多少应退回到主观逻辑中的 uncertainty mass `u`。分类器输出的不再是普通 softmax，而是 Dirichlet / subjective-logic opinion：

```text
opinion = (belief_1, ..., belief_K, uncertainty)
sum_k belief_k + uncertainty = 1
```

当反事实采样干预显示某些类别证据强烈依赖采样政策时，PIIES 不要求所有视图 logits 一致，也不计算风险方差；它只要求模型降低伪证据强度，把这部分质量放进 `uncertainty`。这样既保留 informative sampling 的可能价值，又避免模型在 policy shift 下以高置信方式犯错。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从 pooled logits 改为 Evidence + Ignorance 双输出

保留当前“采样解耦/反事实干预”框架中的 value process 与 sampling process：

1. **Value Evidence Encoder**
   - 输入观测值、变量 id、时间戳和 delta-t。
   - 输出类别证据 `e_value in R_+^K`，表示观测值本身支持每个类别的强度。
   - 它可以接现有 irregular encoder、mTAND、SSM 或轻量 event transformer；关键是输出非负 evidence，而不是直接输出 logits。

2. **Policy Ignorance Shield**
   - 输入只包含采样坐标摘要、测量质量摘要和反事实 policy audit features。
   - 输出两类量：
     - `discount in [0, 1]^K`：每个类别证据中有多少需要被策略不确定性折扣。
     - `ignorance_boost >= 0`：折扣掉的证据注入到主观逻辑 uncertainty mass 的强度。
   - 它不把 policy code 作为分类特征，也不预测环境标签；它只决定“哪些类别证据在当前政策下不可判定”。

3. **Subjective Evidence Composer**
   - 折扣后的 Dirichlet 参数：

```text
e_safe_k = e_value_k * (1 - discount_k)
alpha_k  = e_safe_k + 1
S        = sum_k alpha_k + ignorance_boost
belief_k = e_safe_k / S
uncertainty = K / S + ignorance_boost / S
```

   - 分类训练使用 `alpha` 的 evidential expected log likelihood，而部署时同时输出 `belief` 和 `uncertainty`。

关键差异：历史提案常把 sampling branch 用于去掉、投影、收费、修复或审计采样信息；PIIES 则把采样分支作为 **无知质量分配器**。它不阻止模型利用采样相关证据，但要求模型在证据不可由当前观测政策稳定支持时显式增加 vacuity。

### 2.2 改 Loss：从不变性/纠错/负控转向 Evidential Vacuity Discipline

总目标：

```text
L = L_evidential_cls
  + lambda_vac * L_policy_vacuity
  + lambda_err * L_error_aware_uncertainty
  + lambda_sharp * L_selective_sharpness
```

#### A. Evidential Classification Loss `L_evidential_cls`

使用 Dirichlet evidential learning 的期望交叉熵：

```text
L_evidential_cls = E_{p ~ Dir(alpha)}[-log p_y]
                 = digamma(S) - digamma(alpha_y)
```

它区别于普通 softmax：证据总量越小，uncertainty 越高；模型不能仅靠把 logits 拉大来获得高置信。

#### B. Policy Vacuity Loss `L_policy_vacuity`

当前反事实干预模块生成若干 policy audit recipes，但不要求不同 recipe 的 logits 或表示一致。对每个 recipe，仅比较“同一 value evidence 在该采样政策下是否仍可判定”：

```text
vacuity_target = normalized_policy_stress(e_value, e_value_under_policy_recipe)
L_policy_vacuity = SmoothL1(uncertainty, stopgrad(vacuity_target))
```

其中 `policy_stress` 可以由三类信号构成：

- 类别证据排名是否因采样政策改变而变得不可区分；
- 测量质量或异方差是否恶化，借鉴 StarEmbed 对 heteroskedasticity / OOD 的重视；
- 前端 encoder 对时间戳与 mask 的组件审计是否显示高敏感，借鉴 LLMTS 的 encoder audit 范式。

注意：这不是 consistency loss。若某个反事实政策下信息确实不足，PIIES 不要求它给出相同预测，而是要求事实预测承认该类别证据存在策略诱导无知。

#### C. Error-Aware Uncertainty Loss `L_error_aware_uncertainty`

若训练样本分类错误或预测 margin 很小，模型应提高 uncertainty；若预测正确且 evidence 主要来自 value path，则允许低 uncertainty：

```text
wrong = 1[pred != y]
margin_gap = belief_y - max_{k != y} belief_k
L_error_aware_uncertainty =
  wrong * relu(u_min - uncertainty)^2
  + relu(margin_target - margin_gap)^2 * relu(u_floor - uncertainty)^2
```

这项关注“错误时是否过度自信”，不同于 historical 的 risk variance、certified radius 或 knockoff FDR。

#### D. Selective Sharpness Loss `L_selective_sharpness`

为了避免模型把所有样本都推到高 uncertainty，加入选择性锐度：

```text
value_quality = stopgrad(value_encoder_quality_score)
L_selective_sharpness =
  value_quality * uncertainty^2
  + (1 - value_quality) * relu(u_target - uncertainty)^2
```

其中 `value_quality` 来自纯 value path 的重构残差、测量噪声估计或 encoder self-audit；它不是 iTimER 式伪观测/Wasserstein 对齐，也不是 observability witness。它只作为“当前观测值证据是否足够清楚”的门槛，避免 vacuity 崩塌为全局拒识。

### 2.3 改 Dataloader：返回 Policy Stress Audit Bank，而不是一致性增强样本

新增 `PolicyStressAuditCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `measurement_std` 或 `quality_score`：若数据集有测量误差则直接使用；否则由局部重复观测差异、变量级噪声统计或缺测窗口估计。
3. `policy_audit_recipe_bank`：用于审计可判定性的策略扰动：
   - `quality_degrade`：模拟测量误差或观测质量恶化；
   - `band_coverage_drop`：借鉴 StarEmbed 的多波段/多变量覆盖变化；
   - `encoder_timestamp_scramble`：轻微扰动时间戳，测试 encoder 对采样坐标敏感性；
   - `pending_value_mask`：借鉴 MILM 的 value-pending 场景，只保留“值尚未返回”的观测坐标。
4. `policy_stress_target`：反事实审计得到的 evidence stress 分数，而不是 label、环境 id 或 policy id。

这些 recipe 只用于估计“当前类别证据有多少不可判定”。它们不产生对比正样本，不做 randomized smoothing，不做风险方差，不做 density ratio，也不要求跨 recipe 预测一致。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `ValueEvidenceEncoder`，输出非负类别证据。
- 现有 sampling branch 改为 `PolicyIgnoranceShield`，输出类别级 evidence discount 与 uncertainty boost。
- 现有 counterfactual intervention 改为 `PolicyStressAuditBank`，生成可判定性审计 probes。
- 推理阶段输出：
  - `belief`：可判定的类别支持；
  - `uncertainty`：策略诱导无知；
  - `discount_by_class`：每个类别证据被采样政策折扣的程度；
  - `selective prediction`：当 uncertainty 超阈值时拒识或请求补采样。

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


class EventValueEvidenceEncoder(nn.Module):
    """Encode irregular values into non-negative class evidence."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id)
        event_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(meas_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * event_mask.unsqueeze(-1)
        seq_h, _ = self.rnn(event_h)
        pooled = masked_mean(seq_h, event_mask, dim=1)
        evidence = self.evidence_head(pooled)
        quality = self.quality_head(pooled).squeeze(-1)
        return {"evidence": evidence, "quality": quality, "pooled": pooled}


class PolicyIgnoranceShield(nn.Module):
    """Estimate policy-induced ignorance without feeding policy as class evidence."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int, recipe_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.recipe_proj = nn.Linear(recipe_dim, hidden_dim)
        self.policy_proj = nn.Sequential(
            nn.Linear(num_vars + 7 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.discount_head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid(),
        )
        self.ignorance_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def summarize_policy(self, batch: dict) -> torch.Tensor:
        time = batch["event_time"]
        mask = batch["event_mask"]
        var_id = batch["event_var_id"]
        meas_std = batch.get("measurement_std", torch.zeros_like(time))

        horizon = time.amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        early = (time_norm <= 0.33).to(time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype)
        late = (time_norm > 0.66).to(time.dtype)

        var_obs = F.one_hot(var_id.clamp_min(0), self.num_vars).to(time.dtype) * mask.unsqueeze(-1)
        var_rate = var_obs.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        stats = torch.cat(
            [
                mask.mean(dim=1, keepdim=True),
                (early * mask).mean(dim=1, keepdim=True),
                (middle * mask).mean(dim=1, keepdim=True),
                (late * mask).mean(dim=1, keepdim=True),
                torch.log1p(delta_t).mean(dim=1, keepdim=True),
                meas_std.mean(dim=1, keepdim=True),
                meas_std.amax(dim=1, keepdim=True),
            ],
            dim=-1,
        )
        return torch.cat([var_rate, stats], dim=-1)

    def forward(self, batch: dict, recipe: torch.Tensor) -> dict:
        policy_summary = self.summarize_policy(batch)
        recipe_h = self.recipe_proj(recipe)
        h = self.policy_proj(torch.cat([policy_summary, recipe_h], dim=-1))
        return {
            "discount": self.discount_head(h),
            "ignorance_boost": self.ignorance_head(h).squeeze(-1),
        }


def compose_subjective_opinion(
    evidence: torch.Tensor,
    discount: torch.Tensor,
    ignorance_boost: torch.Tensor,
) -> dict:
    """Convert evidence plus policy discount into subjective-logic opinion."""

    safe_evidence = evidence * (1.0 - discount)
    alpha = safe_evidence + 1.0
    num_classes = evidence.size(-1)
    strength = alpha.sum(dim=-1) + ignorance_boost
    belief = safe_evidence / strength.unsqueeze(-1).clamp_min(1e-6)
    uncertainty = (num_classes + ignorance_boost) / strength.clamp_min(1e-6)
    expected_prob = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return {
        "safe_evidence": safe_evidence,
        "alpha": alpha,
        "belief": belief,
        "uncertainty": uncertainty.clamp(0.0, 1.0),
        "expected_prob": expected_prob,
    }


def evidential_classification_loss(alpha: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    strength = alpha.sum(dim=-1)
    target_alpha = alpha.gather(1, labels[:, None]).squeeze(1)
    return (torch.digamma(strength) - torch.digamma(target_alpha)).mean()


def policy_vacuity_target(
    factual_evidence: torch.Tensor,
    audit_evidence: torch.Tensor,
    audit_quality: torch.Tensor,
) -> torch.Tensor:
    """Estimate how much class evidence becomes undecidable under a policy audit."""

    factual_prob = factual_evidence / factual_evidence.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    audit_prob = audit_evidence / audit_evidence.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    rank_stress = (factual_prob - audit_prob).abs().sum(dim=-1) * 0.5
    quality_stress = (1.0 - audit_quality).clamp(0.0, 1.0)
    return torch.maximum(rank_stress, quality_stress).clamp(0.0, 1.0)


class PolicyInducedIgnoranceEvidentialShield(nn.Module):
    """Sampling-policy robust classifier with explicit policy-induced vacuity."""

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        num_classes: int,
        recipe_dim: int,
    ):
        super().__init__()
        self.value_encoder = EventValueEvidenceEncoder(num_vars, hidden_dim, num_classes)
        self.shield = PolicyIgnoranceShield(num_vars, hidden_dim, num_classes, recipe_dim)
        self.num_classes = num_classes
        self.recipe_dim = recipe_dim

    def forward(self, batch: dict, recipe: torch.Tensor | None = None) -> dict:
        if recipe is None:
            recipe = torch.zeros(
                batch["event_value"].size(0),
                self.recipe_dim,
                device=batch["event_value"].device,
                dtype=batch["event_value"].dtype,
            )
        value = self.value_encoder(batch)
        shield = self.shield(batch, recipe)
        opinion = compose_subjective_opinion(
            evidence=value["evidence"],
            discount=shield["discount"],
            ignorance_boost=shield["ignorance_boost"],
        )
        return {**value, **shield, **opinion}

    def training_loss(
        self,
        batch: dict,
        lambda_vac: float = 0.25,
        lambda_err: float = 0.15,
        lambda_sharp: float = 0.05,
        u_min: float = 0.45,
        margin_target: float = 0.20,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)

        cls_loss = evidential_classification_loss(out["alpha"], labels)

        # Policy audit recipes supervise vacuity, not representation consistency.
        vac_targets = []
        for recipe, audit_batch in zip(
            batch["policy_audit_recipe_bank"].unbind(dim=1),
            batch["policy_audit_batch_bank"],
        ):
            with torch.no_grad():
                audit_value = self.value_encoder(audit_batch)
                target = policy_vacuity_target(
                    factual_evidence=out["evidence"].detach(),
                    audit_evidence=audit_value["evidence"],
                    audit_quality=audit_value["quality"],
                )
            audit_out = self.forward(batch, recipe=recipe)
            vac_targets.append(F.smooth_l1_loss(audit_out["uncertainty"], target))
        vacuity_loss = torch.stack(vac_targets).mean()

        pred = out["belief"].argmax(dim=-1)
        wrong = (pred != labels).to(out["belief"].dtype)
        true_belief = out["belief"].gather(1, labels[:, None]).squeeze(1)
        rival_belief = out["belief"].masked_fill(
            F.one_hot(labels, self.num_classes).bool(),
            -1.0,
        ).max(dim=-1).values
        margin_gap = true_belief - rival_belief
        error_loss = (
            wrong * F.relu(u_min - out["uncertainty"]).pow(2)
            + F.relu(margin_target - margin_gap).pow(2) * F.relu(0.30 - out["uncertainty"]).pow(2)
        ).mean()

        quality = out["quality"].detach()
        sharp_loss = (
            quality * out["uncertainty"].pow(2)
            + (1.0 - quality) * F.relu(0.50 - out["uncertainty"]).pow(2)
        ).mean()

        total = cls_loss + lambda_vac * vacuity_loss + lambda_err * error_loss + lambda_sharp * sharp_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "policy_vacuity_loss": vacuity_loss.detach(),
            "error_uncertainty_loss": error_loss.detach(),
            "selective_sharpness_loss": sharp_loss.detach(),
            "mean_uncertainty": out["uncertainty"].mean().detach(),
        }


@torch.no_grad()
def build_policy_stress_audit_bank(batch: dict, recipe_dim: int = 4) -> tuple[torch.Tensor, list[dict]]:
    """Create policy-stress audit recipes and batches.

    The audit views estimate undecidability; they are not used as consistency pairs.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    meas_std = batch.get("measurement_std", torch.zeros_like(value))
    bsz, num_events = value.shape
    device = value.device

    recipes = []
    audit_batches = []

    def clone_with(value_new, time_new, var_new, mask_new, std_new):
        out = dict(batch)
        out["event_value"] = value_new
        out["event_time"] = time_new
        out["event_var_id"] = var_new
        out["event_mask"] = mask_new
        out["measurement_std"] = std_new
        return out

    # 1. Measurement quality degradation.
    noisy_std = meas_std + 0.25 * value.detach().abs().mean(dim=1, keepdim=True)
    recipes.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(bsz, -1))
    audit_batches.append(clone_with(value, time, var_id, mask, noisy_std))

    # 2. Variable/band coverage drop.
    odd_var = (var_id % 2 == 1).to(mask.dtype)
    coverage_mask = mask * (1.0 - odd_var)
    recipes.append(torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(bsz, -1))
    audit_batches.append(clone_with(value * coverage_mask, time, var_id, coverage_mask, meas_std))

    # 3. Timestamp scramble for encoder-sensitivity audit.
    jitter = 0.05 * time.amax(dim=1, keepdim=True).clamp_min(1e-6) * torch.randn_like(time)
    scrambled_time = (time + jitter * mask).clamp_min(0.0)
    recipes.append(torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).expand(bsz, -1))
    audit_batches.append(clone_with(value, scrambled_time, var_id, mask, meas_std))

    # 4. Value-pending audit: calendar observed, value not yet returned.
    pending_value = torch.zeros_like(value)
    pending_std = meas_std + value.detach().abs()
    recipes.append(torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(bsz, -1))
    audit_batches.append(clone_with(pending_value, time, var_id, mask, pending_std))

    recipe_bank = torch.stack(recipes, dim=1)
    if recipe_dim != 4:
        pad = torch.zeros(bsz, recipe_bank.size(1), max(recipe_dim - 4, 0), device=device)
        recipe_bank = torch.cat([recipe_bank[..., :recipe_dim], pad], dim=-1)[..., :recipe_dim]
    return recipe_bank, audit_batches
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `quality-shift`：训练环境测量误差低，测试环境某些变量或波段异方差增大。
   - `coverage-shift`：多变量/多波段覆盖发生系统性变化，借鉴 StarEmbed 中观测 cadence 与 band coverage 的问题。
   - `timestamp-sensitivity shift`：相同观测值在不同采样时间坐标下出现，测试 encoder 是否过度依赖采样坐标。
   - `value-pending shift`：某些化验值尚未返回但采样事件已出现，借鉴 MILM 的 value-pending evaluation。

2. **对比方法**
   - 普通 softmax irregular encoder。
   - evidential learning 但无 policy ignorance shield。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - MILM-style value-redacted sampling classifier。
   - StarEmbed / foundation embedding + OOD score baseline。
   - LLMTS 风格 encoder/alignment baseline。
   - DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness 等历史方案。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - selective accuracy at coverage：拒识高 uncertainty 样本后的准确率。
   - policy-induced vacuity calibration：高 uncertainty 是否对应高 policy stress。
   - overconfidence under shift：错误样本中低 uncertainty 的比例。
   - class-wise discount profile：哪些类别证据最易受采样政策诱导无知影响。

4. **消融实验**
   - 去掉 `PolicyIgnoranceShield`，验证普通 evidential classifier 是否仍过度自信。
   - 去掉 `L_policy_vacuity`，检查 uncertainty 是否不再响应 policy stress。
   - 去掉 measurement quality 输入，验证 StarEmbed 式异方差信息是否关键。
   - 将 audit recipes 替换成随机 mask，验证收益来自可判定性审计而非普通增强。
   - 固定 uncertainty threshold，评估选择性预测在跨政策部署中的稳定性。

## 5. 预期创新性

1. **从采样去偏转向采样诱导无知建模**：不删除、不投影、不征税、不纠错采样信息，而是在主观逻辑中为策略不可判定证据保留 vacuity mass。
2. **从 softmax 置信度转向 evidential opinion**：分类输出同时包含 belief 与 uncertainty，能直接处理跨采样政策下的过度自信问题。
3. **从反事实一致性转向可判定性审计**：counterfactual intervention 只估计某类证据是否仍可判定，不要求 logits、representation 或风险在多视图间一致。
4. **吸收 StarEmbed 的异方差/OOD 启发**：将观测质量、band coverage 与 policy stress 纳入 uncertainty，而不是直接用 foundation embedding 分类。
5. **吸收 ICU LLM 审计启发**：优先审计前端 encoder 对时间戳、mask 与 value-pending 的敏感性，而不是依赖后端 LLM alignment。
6. **部署友好**：当 policy shift 过强时，模型可以拒识、请求补采样或输出“该类别证据当前不可判定”，比高置信错误更适合医疗和科学观测场景。

## 6. 一句话投稿卖点

**PIIES 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“策略诱导无知质量”的问题，并通过 subjective-logic evidential shield 把不可由当前采样政策稳定支持的类别证据转化为可校准 uncertainty，从而在不依赖危险率、对抗、一致性、后验商、随机平滑、拓扑、gauge、纠错码、knockoff 或观测性门控的前提下，解决跨采样政策部署中的过度自信分类。**
