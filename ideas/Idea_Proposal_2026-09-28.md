# Title: Do-AnchorCone Freshness Demixer：面向采样策略偏移的可靠性锚锥解混分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化持久记忆 `MEMORIES.md` 与最新 `idea_2026-09-27.md`，纳入未落盘或历史较早 proposal 的核心机制摘要。
- 已读取/检索当前仓库 `ideas/Idea_Proposal_*.md` 的历史标题、黑名单与核心机制段，覆盖当前工作区已落盘的 2026-06-12 至 2026-09-27 提案。
- 已读取近期 `paper_daily_2026-09-27.md`，并从 `paper_daily.md`、`paper_daily_2026-09-16.md` 等入口复核近期机制；本轮重点吸收 **ReTAMamba** 的 reliability-aware aggregation、variable-specific freshness decay、multi-scale token routing、Chronological Weaving 与 Mamba 线性序列建模。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下主机制：

1. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
2. 生理流/采样算子交换子、state-policy graph 拆分、policy residual sink。
3. protocol tax、additive evidence market、token 边际审计、证据预算。
4. posterior quotient、Radon-Nikodym density ratio、doubly robust correction、RKHS cubature。
5. reconstruction error cartography、ANOVA projection、VQ clauses、HSIC redaction。
6. optional-stopping martingale、拓扑胶囊、policy gauge、syndrome code、knockoff calendar。
7. observability witness、evidential vacuity、policy lattice、solver trace front-door、conformal sleeve、IV/control-function。
8. Borda jury、Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock。
9. trigger-phase hysteresis、control-barrier certificate、regret escrow、principal stratum compiler、Do-IV purifier。
10. IRT-DIF、RG fixed point、bitemporal curtain、clinical tomography、matched risk-set likelihood、policy privacy cloak。
11. synthetic falsification forge、dialectical JEPA referee、PID semantic prism、Noether semantic action、meta workflow immunization。
12. collider seal、source-atlas antipodal retrieval、Schwarzian warp canonicalizer、renewal exposure meter、palimpsest memory。
13. typed SSA observation program、Robin boundary field solver、Blackwell experiment order、elicitable functional compass。
14. Lyapunov interval observer、BBP / Marchenko-Pastur random-matrix spike sieve、RIP sparse pathology recovery。
15. factorial cumulant thinning algebra、repeat/panel cumulant subtraction、Poisson clutter null。
16. Delta-XAI / SWING 风险跃迁熔断、prediction-change attribution consistency、policy interrupt buffer。
17. B-spline knot-vector framing、Boehm knot insertion/removal、canonical disease control polygon、knot residual buffer。
18. 单纯 state-policy 双分支、对抗环境分类器、跨视图 logits/representation 一致性、频域掩码对比学习、missingness pattern 直接分类、普通 retrieval-augmented classifier、普通 dual-SSM classifier。

本提案选择新的正交切入点：**不把采样政策建成概率、图、谱、程序、证明、记忆、样条 knot、高阶计数或在线风险跃迁；也不只学习一条 state reliability 与一条 policy reliability。相反，它把 ReTAMamba 中每个变量、时间桶、尺度和 token router 的“可靠性权重”看作一个非负混合信号。采样偏移真正污染的是 reliability tensor 的原子组成：有些原子表示生理信息随时间过期，有些原子表示医院工作流、复测制度、panel 打包或 router 预算偏好。Do-AnchorCone Freshness Demixer 用可分离非负锚锥把这些 freshness atoms 解混，分类器只读取落在 state freshness cone 内的 token。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

ReTAMamba 的关键机制是：对每个 time-variable token 估计 reliability，再做多尺度 reliability-weighted aggregation、Chronological Weaving 和 Mamba 编码。这个设计抓住了 ICU 非规则采样中的真实痛点：同样是 6 小时未测，心率、血压、乳酸、白细胞的“过期速度”不同；同样是缺失，占位值是否还能代表当前状态也不同。

但对 sampling-policy shift 来说，单个 Reliability Gate 有一个危险混合：

- 变量的真实生理半衰期会影响 reliability；
- 医院是否 routine 复测、是否告警后加密采样、是否 panel 同步返回，也会影响 reliability；
- multi-scale router 可能把“某中心更常产生 60 分钟密集 token”当成死亡风险证据；
- Chronological Weaving 保护了时间顺序，却也把某个采样制度特有的尺度交织模式完整保留下来。

因此，我们不问“这个 token 现在还可靠不可靠”，而问：

> 这个 reliability 是由哪些 freshness atoms 线性组合出来的？其中哪些 atom 在反事实采样政策下只随工作流变化，哪些 atom 必须有 value surprise、趋势新息或跨变量生理支持才会激活？

Do-AnchorCone Freshness Demixer (DAFD) 的核心直觉是：

1. ReTAMamba 给了我们一个强大的 reliability tensor：`R[var, scale, time, route]`。
2. 采样政策偏移表现为 reliability tensor 的 **非负混合比例变化**，而不是必须改变病理值。
3. 若存在 counterfactual intervention，可以构造“只改 schedule、不改 value”和“只改 value novelty、不改 schedule”的锚视图。
4. 在可分离 NMF / anchor-word identifiability 的思想下，这些锚视图能把 freshness atoms 分成 state cone 与 policy cone。
5. 分类器只读取 state cone reconstruction；policy cone 只用于偏移诊断、router 预算告警和部署不确定性。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 提供 state-anchor：真实数值新息、阈值跨越、趋势反转、跨变量生理一致性。
- sampling process 提供 policy-anchor：elapsed-time 伸缩、panel pack/split、routine bucket snap、router budget 改变、低频 duty cycle。
- counterfactual intervention 不再要求所有视图 logits 一致，也不估计采样概率；它只负责制造可识别的 anchor views，让 reliability tensor 被解混为 state freshness atoms 与 policy freshness atoms。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Freshness Anchor Bank

新增 `FreshnessAnchorCollator`。每个 batch 同时返回事实视图与四类反事实锚视图：

```text
factual_view:
    原始 event stream

policy_anchor_view:
    保持 value、variable、label 不变，只改变 observation schedule

state_anchor_view:
    保持 schedule 不变，只注入 value-supported novelty

router_anchor_view:
    保持 event stream 不变，只改变 multi-scale token budget / scale bucket

mixed_probe_view:
    同时改变 schedule 与部分 value novelty，用于检验解混是否可加
```

具体 recipes：

1. `staleness_dilate`
   - 拉伸 elapsed time / time gap，但不改观测值。
   - 应主要激活 policy freshness atoms。

2. `routine_bucket_snap`
   - 将时间吸附到 60/120/240 分钟桶边界。
   - 用于检测 Chronological Weaving 是否记住中心流程。

3. `panel_pack_split`
   - 把同步 panel 拆成异步返回，或把近邻变量打包为同步返回。
   - value 不变，生理 state 不应变化。

4. `router_budget_swap`
   - 改变每个尺度允许进入 Mamba 的 token 数。
   - 若分类 margin 大幅变动，说明 router 偏好含有 policy shortcut。

5. `semantic_freshness_inject`
   - 在相同 schedule 下注入真实 value surprise、趋势反转或阈值跨越。
   - 应主要激活 state freshness atoms。

### 2.2 改 Encoder：Reliability Tensor -> Anchor Cone Demixer -> Safe Mamba

#### A. ReTAMamba-style Reliability Front-End

保留 ReTAMamba 的优点：构造 time-variable token，计算 staleness、mask、变量 ID、尺度 ID，并得到原始 reliability logit：

```text
z_i = TokenStem(value_i, mask_i, var_i, scale_i, elapsed_i)
r_i = ReliabilityGate(z_i)
```

这里的 `r_i` 不直接用于分类，而是进入解混器。

#### B. Freshness Atom Dictionary

学习一个非负 atom dictionary：

```text
A_state  = [a_1, ..., a_Ks]
A_policy = [b_1, ..., b_Kp]
A        = concat(A_state, A_policy)
```

对每个 token 产生非负混合系数：

```text
alpha_i = softplus(ConeRouter(z_i, r_i))
fresh_i = alpha_i @ A
state_fresh_i  = alpha_state_i  @ A_state
policy_fresh_i = alpha_policy_i @ A_policy
```

与普通双分支不同，DAFD 的分离对象不是 hidden representation，而是 **reliability / freshness decision 的原子混合**。同一个 token embedding 可以存在，但它被多少权重送进 Mamba，必须先经过 anchor cone 解混。

#### C. Safe Reliability Weaving

只用 state cone reconstruction 加权 token：

```text
safe_token_i = sigmoid(w_state^T state_fresh_i) * value_token_i
```

再做多尺度 aggregation、Chronological Weaving 与 Mamba / Transformer 编码：

```text
safe_sequence = ChronologicalWeave(AggregateByScale(safe_token))
logits = Classifier(Mamba(safe_sequence))
```

policy cone 不进入分类头，只输出：

- `policy_freshness_score`
- `router_policy_load`
- `scale_policy_atom_histogram`
- `deployment_shift_alarm`

#### D. Anchor Identifiability Buffers

为了让 atom 有可解释语义，batch 内维护两类 anchor buffer：

```text
state_anchor_buffer:
    value novelty 强、schedule 不变的 token

policy_anchor_buffer:
    schedule novelty 强、value novelty 弱的 token
```

训练时要求 state anchor 的 `alpha_state` 低熵高响应、`alpha_policy` 低响应；policy anchor 反之。这样不是对抗分类器，也不是简单环境识别，而是用反事实锚点建立 freshness atom 的可分离性。

### 2.3 改 Loss：Anchor-Cone Freshness Discipline

总目标：

```text
L = L_cls
  + lambda_rel * L_reliability_reconstruction
  + lambda_anc * L_anchor_purity
  + lambda_pol * L_policy_cone_absorption
  + lambda_sem * L_state_semantic_release
  + lambda_mix * L_mixed_additivity
  + lambda_lkg * L_cone_leakage_audit
```

#### A. Safe Classification `L_cls`

只用 state cone 加权后的 safe tokens 分类：

```text
L_cls = CE(Classifier(SafeMamba(state_weighted_tokens)), y)
```

分类头不接收 raw mask、elapsed-time histogram、event count、router budget、scale occupancy 或 policy cone coefficients。

#### B. Reliability Reconstruction `L_reliability_reconstruction`

解混器必须解释原始 Reliability Gate，而不是随意丢弃信息：

```text
r_hat_i = Readout(alpha_i @ A)
L_rel = SmoothL1(r_hat_i, r_i)
```

这保证 DAFD 仍继承 ReTAMamba 的可靠性建模能力。

#### C. Anchor Purity `L_anchor_purity`

对 state-anchor tokens：

```text
L_state_anchor =
    entropy(alpha_state_norm) - entropy(alpha_policy_norm)
    + ||alpha_policy||_1
```

对 policy-anchor tokens：

```text
L_policy_anchor =
    entropy(alpha_policy_norm) - entropy(alpha_state_norm)
    + ||alpha_state||_1
```

直觉：反事实锚视图提供可识别性。真正的 value novelty 应落到 state atoms；纯 schedule 改动应落到 policy atoms。

#### D. Policy Cone Absorption `L_policy_cone_absorption`

对 `staleness_dilate`、`routine_bucket_snap`、`panel_pack_split`、`router_budget_swap` 这类 policy-only views，允许 policy cone 改变，但要求 state cone 的加权 token 稳定：

```text
L_pol =
    ||state_fresh(policy_view) - stopgrad(state_fresh(factual))||_1
  + relu(||policy_fresh(policy_view) - policy_fresh(factual)||_1_margin - m)
```

这不是 logits 一致性，也不是表示一致性；它只约束 reliability tensor 中的 state freshness atoms 不被 schedule 改动激活。

#### E. State Semantic Release `L_state_semantic_release`

对 `semantic_freshness_inject`，state cone 必须允许变化，否则模型会过度鲁棒：

```text
semantic_gain = value_surprise + trend_flip + threshold_cross
L_sem = relu(min_state_delta - ||state_fresh(semantic_view) - state_fresh(factual)||_1)^2
```

policy cone 不应独占真实语义新息：

```text
L_sem_policy = ||policy_fresh(semantic_view) - policy_fresh(factual)||_1 * semantic_only_mask
```

#### F. Mixed Additivity `L_mixed_additivity`

对同时包含 schedule 与 value novelty 的 mixed view，要求 freshness decomposition 近似可加：

```text
Delta_mixed_state  ~= Delta_state_anchor
Delta_mixed_policy ~= Delta_policy_anchor
```

这让模型在真实部署中面对“告警后复测且数值确实恶化”的复杂情况时，既不会把全部变化归因给政策，也不会让政策变化污染 state freshness。

#### G. Cone Leakage Audit `L_cone_leakage_audit`

训练一个只读 policy cone 的轻量 probe。如果 policy cone 单独能强预测标签，主模型不受影响，但报告数据集中存在强 policy-label coupling；同时用 stop-gradient 约束主分类头不能读取 policy cone：

```text
policy_probe_logits = Probe(stopgrad(policy_cone_stats))
L_lkg = relu(auc_proxy(policy_probe_logits, y) - tau)^2
```

该项是审计项，不是对抗环境分类器；它不反向鼓励隐藏环境信息，只用于发现当前 split 中采样制度是否足以预测标签。

---

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenStem(nn.Module):
    def __init__(self, num_vars: int, num_scales: int, hidden_dim: int):
        super().__init__()
        self.var_emb = nn.Embedding(num_vars, hidden_dim)
        self.scale_emb = nn.Embedding(num_scales, hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        value: torch.Tensor,
        mask: torch.Tensor,
        elapsed: torch.Tensor,
        var_id: torch.Tensor,
        scale_id: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack(
            [
                value,
                mask.to(value.dtype),
                torch.log1p(elapsed.clamp_min(0.0)),
                torch.tanh(value),
            ],
            dim=-1,
        )
        h = self.value_proj(x)
        h = h + self.var_emb(var_id) + self.scale_emb(scale_id)
        return self.norm(h)


class ReliabilityGate(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.net(token).squeeze(-1)


class AnchorConeDemixer(nn.Module):
    """
    Nonnegative freshness atom demixer.

    The classifier only receives state_freshness. Policy freshness is kept for
    diagnostics and deployment-shift alarms.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_state_atoms: int = 8,
        num_policy_atoms: int = 8,
        atom_dim: int = 32,
    ):
        super().__init__()
        self.num_state_atoms = num_state_atoms
        self.num_policy_atoms = num_policy_atoms
        self.num_atoms = num_state_atoms + num_policy_atoms

        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_atoms),
        )
        self.atom_raw = nn.Parameter(torch.randn(self.num_atoms, atom_dim) * 0.02)
        self.rel_readout = nn.Linear(atom_dim, 1)
        self.state_weight = nn.Linear(atom_dim, 1)
        self.policy_weight = nn.Linear(atom_dim, 1)

    @property
    def atoms(self) -> torch.Tensor:
        return F.softplus(self.atom_raw)

    def forward(self, token: torch.Tensor, rel_logit: torch.Tensor) -> dict[str, torch.Tensor]:
        rel_feat = rel_logit.unsqueeze(-1)
        alpha = F.softplus(self.router(torch.cat([token, rel_feat], dim=-1)))
        atom_mix = alpha @ self.atoms

        alpha_state = alpha[..., : self.num_state_atoms]
        alpha_policy = alpha[..., self.num_state_atoms :]
        state_mix = alpha_state @ self.atoms[: self.num_state_atoms]
        policy_mix = alpha_policy @ self.atoms[self.num_state_atoms :]

        rel_hat = self.rel_readout(atom_mix).squeeze(-1)
        state_gate = torch.sigmoid(self.state_weight(state_mix).squeeze(-1))
        policy_gate = torch.sigmoid(self.policy_weight(policy_mix).squeeze(-1))
        return {
            "alpha": alpha,
            "alpha_state": alpha_state,
            "alpha_policy": alpha_policy,
            "state_fresh": state_mix,
            "policy_fresh": policy_mix,
            "rel_hat": rel_hat,
            "state_gate": state_gate,
            "policy_gate": policy_gate,
        }


class SafeSequenceEncoder(nn.Module):
    """A compact stand-in for Mamba/Transformer after chronological weaving."""
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        packed = token * mask.unsqueeze(-1).to(token.dtype)
        out, _ = self.gru(packed)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0).to(token.dtype)
        pooled = (out * mask.unsqueeze(-1).to(token.dtype)).sum(dim=1) / denom
        return self.classifier(pooled)


class AnchorConeFreshnessClassifier(nn.Module):
    def __init__(
        self,
        num_vars: int,
        num_scales: int,
        num_classes: int,
        hidden_dim: int = 128,
        atom_dim: int = 32,
    ):
        super().__init__()
        self.stem = TokenStem(num_vars, num_scales, hidden_dim)
        self.reliability = ReliabilityGate(hidden_dim)
        self.demixer = AnchorConeDemixer(hidden_dim, atom_dim=atom_dim)
        self.safe_encoder = SafeSequenceEncoder(hidden_dim, num_classes)
        self.policy_probe = nn.Sequential(
            nn.Linear(atom_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token = self.stem(
            batch["value"],
            batch["mask"],
            batch["elapsed"],
            batch["var_id"],
            batch["scale_id"],
        )
        rel_logit = self.reliability(token)
        cone = self.demixer(token, rel_logit)

        safe_token = token * cone["state_gate"].unsqueeze(-1) * batch["mask"].unsqueeze(-1)
        logits = self.safe_encoder(safe_token, batch["mask"])

        policy_stats = torch.cat(
            [
                cone["policy_fresh"].mean(dim=1),
                cone["policy_gate"].mean(dim=1, keepdim=True),
                cone["alpha_policy"].sum(dim=-1).mean(dim=1, keepdim=True),
            ],
            dim=-1,
        )
        policy_probe_logits = self.policy_probe(policy_stats.detach())
        return {
            "logits": logits,
            "token": token,
            "rel_logit": rel_logit,
            "safe_token": safe_token,
            "policy_probe_logits": policy_probe_logits,
            **cone,
        }


def normalized_entropy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = x / x.sum(dim=-1, keepdim=True).clamp_min(eps)
    ent = -(p * (p + eps).log()).sum(dim=-1)
    return ent / torch.log(torch.tensor(x.size(-1), device=x.device, dtype=x.dtype))


class AnchorConeFreshnessLoss(nn.Module):
    def __init__(
        self,
        lambda_rel: float = 0.5,
        lambda_anchor: float = 0.2,
        lambda_policy: float = 0.5,
        lambda_semantic: float = 0.3,
        lambda_mixed: float = 0.2,
        lambda_leakage: float = 0.05,
        leakage_tau: float = 0.65,
    ):
        super().__init__()
        self.lambda_rel = lambda_rel
        self.lambda_anchor = lambda_anchor
        self.lambda_policy = lambda_policy
        self.lambda_semantic = lambda_semantic
        self.lambda_mixed = lambda_mixed
        self.lambda_leakage = lambda_leakage
        self.leakage_tau = leakage_tau

    def forward(
        self,
        factual: dict[str, torch.Tensor],
        policy_view: dict[str, torch.Tensor],
        semantic_view: dict[str, torch.Tensor],
        mixed_view: dict[str, torch.Tensor] | None,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        y = batch["label"]
        mask = batch["mask"].to(factual["rel_logit"].dtype)

        loss_cls = F.cross_entropy(factual["logits"], y)
        loss_rel = F.smooth_l1_loss(factual["rel_hat"] * mask, factual["rel_logit"].detach() * mask)

        state_anchor = batch.get("state_anchor_mask", torch.zeros_like(mask))
        policy_anchor = batch.get("policy_anchor_mask", torch.zeros_like(mask))

        ent_state = normalized_entropy(factual["alpha_state"])
        ent_policy = normalized_entropy(factual["alpha_policy"])
        state_mass = factual["alpha_state"].sum(dim=-1)
        policy_mass = factual["alpha_policy"].sum(dim=-1)

        loss_state_anchor = (
            (ent_state - ent_policy + policy_mass) * state_anchor
        ).sum() / state_anchor.sum().clamp_min(1.0)
        loss_policy_anchor = (
            (ent_policy - ent_state + state_mass) * policy_anchor
        ).sum() / policy_anchor.sum().clamp_min(1.0)
        loss_anchor = loss_state_anchor + loss_policy_anchor

        loss_policy_absorb = F.l1_loss(
            policy_view["state_fresh"] * mask.unsqueeze(-1),
            factual["state_fresh"].detach() * mask.unsqueeze(-1),
        )

        semantic_strength = batch.get("semantic_strength", torch.ones_like(mask))
        state_delta = (semantic_view["state_fresh"] - factual["state_fresh"].detach()).abs().mean(dim=-1)
        policy_delta_sem = (semantic_view["policy_fresh"] - factual["policy_fresh"].detach()).abs().mean(dim=-1)
        loss_semantic = (
            F.relu(semantic_strength - state_delta) + 0.1 * policy_delta_sem
        ).mul(mask).sum() / mask.sum().clamp_min(1.0)

        if mixed_view is None:
            loss_mixed = factual["rel_logit"].new_tensor(0.0)
        else:
            delta_state_expected = semantic_view["state_fresh"] - factual["state_fresh"]
            delta_policy_expected = policy_view["policy_fresh"] - factual["policy_fresh"]
            loss_mixed = F.l1_loss(
                mixed_view["state_fresh"] - factual["state_fresh"],
                delta_state_expected.detach(),
            ) + F.l1_loss(
                mixed_view["policy_fresh"] - factual["policy_fresh"],
                delta_policy_expected.detach(),
            )

        loss_leakage = F.cross_entropy(factual["policy_probe_logits"], y)
        total = (
            loss_cls
            + self.lambda_rel * loss_rel
            + self.lambda_anchor * loss_anchor
            + self.lambda_policy * loss_policy_absorb
            + self.lambda_semantic * loss_semantic
            + self.lambda_mixed * loss_mixed
            + self.lambda_leakage * loss_leakage
        )
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_rel": loss_rel,
            "loss_anchor": loss_anchor,
            "loss_policy_absorb": loss_policy_absorb,
            "loss_semantic": loss_semantic,
            "loss_mixed": loss_mixed,
            "loss_leakage_probe": loss_leakage,
        }
```

### Collator 草稿

```python
def make_freshness_anchor_view(batch: dict, recipe: str) -> dict:
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    value, elapsed, mask = out["value"], out["elapsed"], out["mask"]

    out["state_anchor_mask"] = torch.zeros_like(mask)
    out["policy_anchor_mask"] = torch.zeros_like(mask)
    out["semantic_strength"] = torch.zeros_like(mask)

    if recipe == "staleness_dilate":
        out["elapsed"] = elapsed * 1.8
        out["policy_anchor_mask"] = mask

    elif recipe == "routine_bucket_snap":
        bucket = torch.round(out["time"] * 24.0) / 24.0
        out["time"] = torch.where(mask.bool(), bucket, out["time"])
        out["elapsed"] = torch.diff(
            F.pad(out["time"], (1, 0), value=0.0),
            dim=1,
        ).abs()
        out["policy_anchor_mask"] = mask

    elif recipe == "panel_pack_split":
        jitter = 0.01 * ((out["var_id"] % 3).to(out["time"].dtype) - 1.0)
        out["time"] = (out["time"] + jitter * mask).clamp(0.0, 1.0)
        out["policy_anchor_mask"] = mask

    elif recipe == "router_budget_swap":
        out["scale_id"] = (out["scale_id"] + 1) % int(out["num_scales"])
        out["policy_anchor_mask"] = mask

    elif recipe == "semantic_freshness_inject":
        surprise = out.get("value_surprise", torch.zeros_like(value))
        inject = (surprise > surprise.quantile(0.75, dim=1, keepdim=True)).to(value.dtype) * mask
        out["value"] = value + 0.5 * inject * value.sign().clamp(min=0.0).add(0.1)
        out["state_anchor_mask"] = inject
        out["semantic_strength"] = inject

    else:
        raise ValueError(f"unknown freshness anchor recipe: {recipe}")

    return out
```

---

## 4. 实验切入点

1. **跨采样政策鲁棒性**
   - MIMIC-IV、eICU、PhysioNet 2012、P12/P19。
   - 构造 routine bucket、alarm burst、panel pack/split、low-duty-cycle、router-budget shift。
   - 报告 in-policy、cross-policy、counterfactual-policy AUROC/AUPRC 与 calibration。

2. **Freshness atom 可解释性**
   - 展示 state atoms 对 value surprise、trend flip、threshold crossing 的响应。
   - 展示 policy atoms 对 elapsed dilation、panel split、scale budget shift 的响应。
   - 对比 ReTAMamba 原始 Reliability Gate 的变量衰减率是否被拆成生理半衰期与工作流半衰期。

3. **Ablation**
   - 去掉 anchor purity，只做普通 reliability gate。
   - 去掉 policy cone absorption，让 state cone 接触 schedule 改动。
   - 让分类头同时读取 state/policy cone。
   - 将 anchor cone demixer 替换为普通 dual MLP branch，验证可分离锚锥是否带来额外稳健性。

4. **Router-shift diagnostics**
   - 改变每个尺度 token budget，检查主分类性能与 `router_policy_load`。
   - 若原始 ReTAMamba 在 budget shift 下 logits 漂移大，而 DAFD 只增加 policy freshness alarm，则说明模型没有把 router 偏好当作类别证据。

5. **Mixed-view stress test**
   - 同时模拟“告警后复测”和“真实恶化”。
   - 期望 policy cone 解释复测密度，state cone 解释真实数值新息；两者的 mixed additivity residual 应低。

---

## 5. 预期创新性

1. **从 reliability decay 转向 reliability atom unmixing**  
   ReTAMamba 学变量特异可靠性衰减；DAFD 进一步追问可靠性由哪些 freshness atoms 组成，避免把医院采样制度学成生理半衰期。

2. **从普通双分支转向可识别锚锥**  
   历史方案已禁止单纯 state-policy 双分支。DAFD 的分离发生在 reliability tensor 的非负原子混合上，并由反事实 anchor views 提供可分离性约束。

3. **从视图一致性转向 freshness 原子归因**  
   不要求反事实视图 logits 或表示一致；只要求 policy-only 改动由 policy freshness atoms 吸收，semantic novelty 由 state freshness atoms 释放。

4. **从时间新鲜度到 router 新鲜度**  
   采样偏移不仅改变 elapsed time，也改变多尺度 token routing 与 Chronological Weaving 的输入组成。DAFD 显式审计 router budget 与 scale occupancy 的 policy atom load。

5. **低侵入兼容强 backbone**  
   可直接插入 ReTAMamba、Mamba、Transformer、SLiCE 或任意 irregular encoder 前，只替换 reliability / token weighting 层，不重写整体时序主干。

## 6. 一句话投稿卖点

**Do-AnchorCone Freshness Demixer 首次把非规则采样分类中的 reliability gate 视为可分离非负 freshness atom mixture，通过反事实 value-anchor 与 policy-anchor 视图把生理信息过期和采样制度过期解混，让 Mamba/Transformer 分类器只读取 state freshness cone，从而避免在跨医院采样政策变化时把 elapsed-time、panel 打包和 router 预算偏好误当成稳定病理证据。**
