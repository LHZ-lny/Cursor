# Title: Do-Tropical Support Routes：面向采样策略偏移的反事实热带支持路径分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*work*.md` 与相关 Markdown 总结文件：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md`，并纳入其中记录的历史机制黑名单。
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
  - `ideas/Idea_Proposal_2026-07-28.md`
  - `ideas/Idea_Proposal_2026-07-30.md`
- 已读取持久记忆中的最新未检出提案摘要：
  - `idea_2026-07-31.md`：Krylov Policy Mode Annihilator。
  - `idea_2026-08-01.md`：Do-Volume Nystrom Basis。
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-07-27.md`

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮永久避开以下核心机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、采样 score 零空间、hazard-driven resampling、do-risk variance。
8. 生理流-采样算子交换子、图交换子、policy residual sink。
9. additive evidence market、protocol tax、token evidence budget、边际证据审计。
10. 模型空间 posterior quotient、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、ANOVA projection、VQ semantic clauses、HSIC redaction。
12. policy-simplex randomized smoothing、certified policy radius、logit-normal/Dirichlet do-sampler。
13. 采样测度 density ratio、doubly robust target-measure correction、influence-bound。
14. optional-stopping martingale queries、standardized innovation、stopping recipe moment control。
15. soft excursion topology、censored persistence interval、censor envelope、fragmentation sobriety。
16. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
17. policy-only negative film、shadow eraser/stencil、shadow nullification。
18. latent packet codeword、parity-check、syndrome locator、packet repair decoder。
19. conditional knockoff calendar、soft knockoff-FDR、swap symmetry。
20. observability witness、counterfactual observability probe、observability-routed classification。
21. subjective-logic / Dirichlet evidential shield、policy-induced vacuity、class-wise evidence discount。
22. observation-set policy lattice、meet/join masks、monotone/submodular margin、shortcut curvature。
23. solver-trace front-door、NFE/roughness trace mediator、reference trace bank。
24. measurement-action bisimulation、policy-word signature renormalization、thermodynamic free-energy、Sklar copula stripping、triage queue debt、Sinkhorn detail canonicalization、MDL episode transducer、causal sheaf gluing、trigger hysteresis、control barrier certificates、regret escrow、principal-stratum status compiler。
25. counterfactual conformal risk sleeves、policy-conditional nonconformity set、leave-policy-out conformal calibration。
26. counterfactual sampling instruments、first-stage structural equation、control-function residualized structure readout、weak-instrument guard。
27. counterfactual policy jury、Borda / pairwise-majority rank tribunal、no-dictator / structural bribery loss。
28. counterfactual low-rank policy displacement operators、Krylov policy subspace、annihilating polynomial、policy-mode energy diagnostics。
29. determinantal / Nystrom event basis、state-volume logdet、policy-overload logdet sobriety、duplicate amplification volume diagnostics。
30. 单纯把 STAR-Set 的 temporal/variable attention bias、VP-GNN 的 variable/patch graph、EHR-SPC 的 status token 或 LLM4EHR 的事件语义对齐拆成 state/policy 双分支并做一致性约束。

本提案选择新的正交切入点：**不做图拆分、不做 rank 投票、不做保形、不做 IV、不做 Krylov 消模、不做 Nystrom 选基；而是把每个类别判定写成若干条“支持路径”，并用热带半环中的 soft-min / soft-max 读出要求这些路径在多种反事实采样政策下仍存在。采样政策可以改变某些路径的局部强度，但最终类别分数由跨政策最弱环节决定，训练医院特有的单点高峰证据无法通过热带瓶颈。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-07-27.md` 中的两个前沿机制暴露了 sampling-policy shortcut 的结构入口：

- **STAR-Set Transformer** 用 temporal locality bias 与 variable-type affinity 恢复事件集合中的时间邻域和变量兼容结构。它的风险是：局部时间尺度与变量共现可能来自医院流程，而不是稳定病程。
- **VP-GNN** 用 variable-wise graph 与 patch-wise graph 同时建模变量依赖和多尺度片段。它的风险是：某个高权重变量边或 patch 可能只在训练采样政策下可见。

历史方案已经尝试过把这些结构拆成 state/policy 双图、作为 IV 残差、用保形集合校准、让政策陪审团投票、用 Krylov 多项式消除政策模态，或用 Nystrom/logdet 选出状态体积基。本轮换一个更像“判别机制重写”的角度：

> 一个可靠类别结论不应该依赖某个采样政策下突然出现的单个高置信 token、单条边或单个 patch；它应该能由一条或多条跨政策仍然连贯的支持路径支撑。若某条路径在 routine-round、alarm-dense、panel-split、patch-budget 等反事实政策中有任一关键环节断裂，这条路径就不应主导最终分类。

**Do-Tropical Support Routes (DTSR)** 将类别分数从普通 `sum / mean / attention pooling / graph pooling` 改成热带半环式读出：

```text
route_score = softmin(同一路径上的多个必要支持槽)
class_score_under_policy = softmax/logsumexp(该类可替代支持路径)
final_class_score = softmin(多个反事实采样政策下的 class_score)
```

这里的 `softmin` 不是保形阈值、不是投票、不是一致性约束，而是“最弱环节”瓶颈：一条类别证据路径只有在所有必要支持槽都存在时才强；一个类别只有在多种可部署采样政策下仍有至少一条支持路径存活时才强。

这能解决采样偏移的原因是：

1. **压制单政策高峰**：训练医院中特有的联测、晚期密集复测或 patch 可见性，可能让普通 attention / graph pooling 得到极高 logit；但它在其他政策视图下路径断裂，经过跨政策 soft-min 后无法支配分类。
2. **保留真实状态证据**：真实病程通常不是一个孤立 token，而会在多个变量、时间片段或状态语义槽中形成可替代支持路径；DTSR 允许不同政策下使用不同局部事件，只要求路径级语义仍可连通。
3. **兼容采样解耦/反事实干预**：sampling branch 不进入分类头，只生成反事实政策视图；value encoder 负责产生 event support atoms；counterfactual intervention 不用于对比、不用于校准、不用于投票，而是用于测试支持路径是否跨政策存活。
4. **与近期 paper 机制自然结合**：STAR-Set 的 temporal/variable bias 和 VP-GNN 的 variable/patch结构不再直接给最终 logit 加分，而是成为 support route 的候选槽来源；EHR-SPC 的 status token 和 LLM4EHR 的事件语义可作为高层支持槽，但必须通过热带瓶颈。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从单一 pooled representation 改为 Support Atom Field

DTSR 可以包裹 STAR-Set、VP-GNN、EHR-SPC 式事件 encoder 或普通 irregular Transformer。基础 encoder 输出事件级或 patch 级 support atoms：

```text
a_i = EventSupportEncoder(value_i, time_i, var_i, delta_t_i)
```

每个 atom 只由观测值、变量、时间差与可选测量质量构造；采样政策描述不拼入分类 logit。若底座是 STAR-Set / VP-GNN，可以把以下结构作为 atom 的辅助上下文：

- temporal-bias-aware local event state；
- variable-affinity neighborhood state；
- variable-wise message state；
- patch-wise summary state；
- EHR-SPC future status token；
- LLM4EHR event semantic anchor。

但这些结构不再直接 pooling 成 logits，而是进入类别支持路径。

### 2.2 改 Classifier：Tropical Support Route Readout

对每个类别 `c` 学习 `Q` 条可替代支持路径，每条路径有 `S` 个必要支持槽：

```text
route(c, q) = [slot_{c,q,1}, ..., slot_{c,q,S}]
```

每个 slot 是一个查询向量，用来从 support atom field 中寻找对应证据：

```text
slot_support_{c,q,s} = Attention(slot_{c,q,s}, {a_i})
```

一条路径的强度由该路径的最弱槽决定：

```text
route_score_{c,q} = softmin_s slot_support_{c,q,s}
```

同一类别可以有多条可替代路径，类别在某个采样政策视图下的分数是路径 softmax：

```text
class_score_c^r = softmax_q route_score_{c,q}^r
```

最后，在反事实采样政策集合 `r = 1..R` 上取热带瓶颈：

```text
final_score_c = softmin_r class_score_c^r
```

这三个层次分别对应：

1. **slot soft-min**：阻止单个 token 高峰支撑整条路径。
2. **route soft-max**：允许不同病程模式有可替代解释。
3. **policy soft-min**：阻止单个采样政策独占最终证据。

### 2.3 改 Loss：从一致性/投票/校准转向 Tropical Route Survival

总目标：

```text
L = L_tropical_cls
  + lambda_surv * L_true_route_survival
  + lambda_frag * L_policy_fragility
  + lambda_slot * L_slot_coverage
  + lambda_div  * L_route_diversity
```

#### A. Tropical Classification Loss `L_tropical_cls`

用跨政策热带瓶颈分数做最终分类：

```text
L_tropical_cls = CE(final_score, y)
```

它不是 average ensemble，也不是 Borda 投票；每个类别分数取决于该类别在最不利反事实采样政策下是否仍有支持路径。

#### B. True Route Survival Loss `L_true_route_survival`

要求真实类别至少存在一条路径在所有政策下都不低于竞争类别 margin：

```text
survival_yq = softmin_r route_score_{y,q}^r
best_survival_y = softmax_q survival_yq
L_survival = relu(margin - best_survival_y + max_{k != y} final_score_k)^2
```

这不是 logits 一致性：不同政策下可以激活不同 atom；约束对象是“是否仍有一条完整支持路径能存活”。

#### C. Policy Fragility Loss `L_policy_fragility`

若某条路径在某个政策视图极强、在其他政策视图极弱，却被最终真实类大量使用，说明它可能是 policy shortcut：

```text
fragility_{c,q} = max_r route_score_{c,q}^r - min_r route_score_{c,q}^r
L_fragility = weighted_mean_q relu(fragility_{y,q} - fragility_cap)^2
```

权重只来自真实类路径 posterior，不要求所有路径都稳定，也不要求所有类别路径稳定；它只抑制“最终依赖的路径”变成单政策高峰。

#### D. Slot Coverage Loss `L_slot_coverage`

每条被使用的路径应从多个变量/时间片段取证，而不是所有 slot 盯住同一个高频观测：

```text
L_slot_coverage = mean cosine(attention_slot_s, attention_slot_s')^2
```

这与 Nystrom/logdet 选基不同：它不最大化体积、不选择事件基，只防止同一路径内部的必要槽塌缩成同一个采样可见 token。

#### E. Route Diversity Loss `L_route_diversity`

同一类别的多条替代路径应覆盖不同临床/状态模式：

```text
L_route_diversity = mean cosine(route_query_{c,q}, route_query_{c,q'})^2
```

它不同于 prototype classifier：route query 不是类别原型，不直接约束样本靠近原型；它只是定义可替代支持路径的槽结构。

### 2.4 改 Dataloader：返回 Tropical Policy View Bank

新增 `TropicalPolicyRouteCollator`，每个 batch 返回事实视图和若干反事实政策视图：

1. `routine_round_view`
   - 时间戳吸附到固定查房式粗网格。
   - 检验模型是否依赖训练医院特有时间间隔。

2. `panel_split_view`
   - 把近同步变量联测拆成更异步的事件。
   - 检验 STAR-Set variable affinity 或 VP-GNN variable edge 是否依赖 panel 共现。

3. `patch_budget_view`
   - 每个粗时间 patch 只保留少量代表事件。
   - 检验 VP-GNN patch-wise graph 或 sparse event detector 是否依赖局部高密度。

4. `alarm_dense_thin_view`
   - 对晚期或告警后 dense burst 做稀疏化。
   - 检验模型是否把复测密度当成风险类别证据。

5. `status_anchor_view`
   - 保留 EHR-SPC / LLM4EHR 风格的高层 status/event anchors，但扰动低层采样日历。
   - 检验状态语义是否能支撑跨政策 route survival。

这些视图不构成正负对比样本，不要求 logits 一致，不做投票或保形校准；它们只是热带瓶颈读出所需要的政策维度。

### 2.5 推理阶段

给定测试样本：

1. 快速模式：使用事实视图和少量标准政策视图，输出 `final_score`。
2. 诊断模式：输出每个类别的：
   - `route_survival`：哪条路径跨政策存活；
   - `weakest_policy`：哪种采样视图使真实类路径最弱；
   - `weakest_slot`：路径中哪个变量/时间/status 槽是最弱环节；
   - `policy_fragility`：预测是否依赖单政策高峰。

当真实预测的 `weakest_slot` 长期来自 panel 共现、late dense patch 或单变量高频复测时，说明仍存在 sampling-policy shortcut；当路径弱点分散且 final margin 稳定时，说明模型更接近可迁移状态证据。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmin(x: torch.Tensor, mask: torch.Tensor | None, dim: int, temperature: float) -> torch.Tensor:
    """Differentiable soft minimum."""

    if mask is not None:
        x = x.masked_fill(mask == 0, 1e4)
    return -temperature * torch.logsumexp(-x / temperature, dim=dim)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


class EventSupportEncoder(nn.Module):
    """Lift irregular events into value-driven support atoms."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]
        measurement_std = batch.get("measurement_std", torch.zeros_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id.clamp_min(0))
        event_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(measurement_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        atom = self.event_proj(event_x) * event_mask.unsqueeze(-1)
        atom, _ = self.context(atom)
        atom = atom * event_mask.unsqueeze(-1)
        return {"support_atoms": atom, "event_mask": event_mask}


class TropicalRouteReadout(nn.Module):
    """Class scores from policy-stable tropical support routes."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        num_routes: int = 6,
        num_slots: int = 4,
        slot_temperature: float = 0.10,
        policy_temperature: float = 0.15,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_routes = num_routes
        self.num_slots = num_slots
        self.slot_temperature = slot_temperature
        self.policy_temperature = policy_temperature

        self.slot_query = nn.Parameter(torch.randn(num_classes, num_routes, num_slots, hidden_dim) * 0.02)
        self.slot_value = nn.Linear(hidden_dim, hidden_dim)
        self.slot_score = nn.Linear(hidden_dim, 1)

    def score_one_policy(self, atoms: torch.Tensor, event_mask: torch.Tensor) -> dict:
        # atoms: [B, N, H]
        bsz, num_events, hidden_dim = atoms.shape
        query = F.normalize(self.slot_query, dim=-1)
        atoms_norm = F.normalize(atoms, dim=-1)

        # [B, C, Q, S, N]
        attn_logits = torch.einsum("bnh,cqsh->bcqsn", atoms_norm, query)
        attn_logits = attn_logits.masked_fill(event_mask[:, None, None, None, :] == 0, -1e4)
        attn = torch.softmax(attn_logits, dim=-1)

        slot_h = torch.einsum("bcqsn,bnh->bcqsh", attn, self.slot_value(atoms))
        slot_support = self.slot_score(torch.tanh(slot_h)).squeeze(-1)  # [B, C, Q, S]

        route_score = masked_softmin(
            slot_support,
            mask=None,
            dim=-1,
            temperature=self.slot_temperature,
        )  # [B, C, Q]

        class_score = torch.logsumexp(route_score, dim=-1)  # [B, C]
        return {
            "slot_support": slot_support,
            "route_score": route_score,
            "class_score": class_score,
            "slot_attention": attn,
        }

    def forward(self, atoms_by_policy: list[torch.Tensor], masks_by_policy: list[torch.Tensor]) -> dict:
        policy_out = [
            self.score_one_policy(atoms, mask)
            for atoms, mask in zip(atoms_by_policy, masks_by_policy)
        ]
        route_scores = torch.stack([out["route_score"] for out in policy_out], dim=1)  # [B, R, C, Q]
        class_scores = torch.stack([out["class_score"] for out in policy_out], dim=1)  # [B, R, C]

        final_score = masked_softmin(
            class_scores,
            mask=None,
            dim=1,
            temperature=self.policy_temperature,
        )  # [B, C]

        return {
            "final_score": final_score,
            "route_scores": route_scores,
            "class_scores_by_policy": class_scores,
            "policy_outputs": policy_out,
        }


class DoTropicalSupportRoutes(nn.Module):
    """Wrap an irregular encoder with counterfactual tropical route survival."""

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        num_classes: int,
        num_routes: int = 6,
        num_slots: int = 4,
    ):
        super().__init__()
        self.encoder = EventSupportEncoder(num_vars, hidden_dim)
        self.readout = TropicalRouteReadout(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_routes=num_routes,
            num_slots=num_slots,
        )
        self.num_classes = num_classes

    def encode_policy_bank(self, batch: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        policy_bank = batch.get("tropical_policy_bank", [batch])
        atoms_by_policy = []
        masks_by_policy = []
        for view in policy_bank:
            enc = self.encoder(view)
            atoms_by_policy.append(enc["support_atoms"])
            masks_by_policy.append(enc["event_mask"])
        return atoms_by_policy, masks_by_policy

    def forward(self, batch: dict) -> dict:
        atoms_by_policy, masks_by_policy = self.encode_policy_bank(batch)
        return self.readout(atoms_by_policy, masks_by_policy)

    def training_loss(
        self,
        batch: dict,
        lambda_surv: float = 0.30,
        lambda_frag: float = 0.15,
        lambda_slot: float = 0.04,
        lambda_div: float = 0.02,
        survival_margin: float = 0.50,
        fragility_cap: float = 1.25,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        final_score = out["final_score"]
        route_scores = out["route_scores"]  # [B, R, C, Q]

        cls_loss = F.cross_entropy(final_score, labels)

        true_routes = route_scores.gather(
            2,
            labels[:, None, None, None].expand(-1, route_scores.size(1), 1, route_scores.size(3)),
        ).squeeze(2)  # [B, R, Q]

        route_survival = masked_softmin(
            true_routes,
            mask=None,
            dim=1,
            temperature=self.readout.policy_temperature,
        )  # [B, Q]
        best_true_survival = torch.logsumexp(route_survival, dim=-1)
        rival_score = final_score.masked_fill(F.one_hot(labels, self.num_classes).bool(), -1e4).max(dim=-1).values
        survival_loss = F.relu(survival_margin + rival_score - best_true_survival).pow(2).mean()

        route_weight = torch.softmax(route_survival.detach(), dim=-1)
        fragility = true_routes.max(dim=1).values - true_routes.min(dim=1).values
        fragility_loss = (F.relu(fragility - fragility_cap).pow(2) * route_weight).sum(dim=-1).mean()

        # Slot coverage: slots used by the same route should not collapse to the same event.
        slot_losses = []
        for policy_out in out["policy_outputs"]:
            attn = policy_out["slot_attention"]  # [B, C, Q, S, N]
            true_attn = attn.gather(
                1,
                labels[:, None, None, None, None].expand(-1, 1, attn.size(2), attn.size(3), attn.size(4)),
            ).squeeze(1)  # [B, Q, S, N]
            sim = torch.einsum("bqsn,bqtn->bqst", true_attn, true_attn)
            eye = torch.eye(sim.size(-1), device=sim.device, dtype=torch.bool)
            slot_losses.append(sim.masked_fill(eye[None, None], 0.0).pow(2).mean())
        slot_loss = torch.stack(slot_losses).mean()

        query = F.normalize(self.readout.slot_query, dim=-1)
        route_repr = query.mean(dim=2)  # [C, Q, H]
        route_sim = torch.einsum("cqh,ckh->cqk", route_repr, route_repr)
        route_eye = torch.eye(route_sim.size(-1), device=route_sim.device, dtype=torch.bool)
        diversity_loss = route_sim.masked_fill(route_eye[None], 0.0).pow(2).mean()

        total = (
            cls_loss
            + lambda_surv * survival_loss
            + lambda_frag * fragility_loss
            + lambda_slot * slot_loss
            + lambda_div * diversity_loss
        )
        return {
            "loss": total,
            "tropical_cls_loss": cls_loss.detach(),
            "route_survival_loss": survival_loss.detach(),
            "policy_fragility_loss": fragility_loss.detach(),
            "slot_coverage_loss": slot_loss.detach(),
            "route_diversity_loss": diversity_loss.detach(),
            "mean_route_fragility": fragility.mean().detach(),
        }


@torch.no_grad()
def build_tropical_policy_bank(batch: dict) -> list[dict]:
    """Create counterfactual policy views for tropical route survival."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        return out

    views = [batch]

    # 1. Routine-round view: snap to coarse rounds while preserving values.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    views.append(clone_with(value * mask, rounded_time, var_id, mask))

    # 2. Panel-split view: weaken near-synchronous cross-variable co-observation.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    split_mask = mask * (1.0 - 0.5 * close * changed_var)
    views.append(clone_with(value * split_mask, time, var_id, split_mask))

    # 3. Patch-budget view: keep only a small budget in each coarse temporal patch.
    patch_keep = torch.zeros_like(mask)
    for start, end in [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]:
        in_patch = ((time_norm >= start) & (time_norm < end)).to(mask.dtype) * mask
        rank = in_patch.cumsum(dim=1)
        patch_keep = torch.maximum(patch_keep, (rank <= 2).to(mask.dtype) * in_patch)
    views.append(clone_with(value * patch_keep, time, var_id, patch_keep))

    # 4. Alarm-dense thinning: thin early routine events while keeping late events.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_mask = torch.where(late > 0, mask, mask * alternating)
    views.append(clone_with(value * alarm_mask, time, var_id, alarm_mask))

    return views
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `temporal-bias shift`：改变 STAR-Set temporal locality 支持的时间尺度。
   - `variable-affinity shift`：训练环境变量联测，测试环境拆成异步测量。
   - `patch-budget shift`：VP-GNN 中高权重 patch 在测试政策下被压缩或不可见。
   - `status-calendar shift`：EHR-SPC / LLM4EHR 的高层 status/event anchor 保持，但低层采样日历改变。

2. **对比方法**
   - STAR-Set / VP-GNN 原模型。
   - 普通 logits 平均或 worst-policy DRO。
   - Counterfactual Conformal Risk Sleeves、Do-Jury Rank Tribunal。
   - Do-IV Structural Purifier、Krylov Policy Mode Annihilator、Do-Volume Nystrom Basis。
   - 历史 DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - tropical route survival margin。
   - policy fragility：最终依赖路径的 `max_policy - min_policy` 差距。
   - weakest-slot attribution：最弱槽集中在哪些变量/时间/status anchor。
   - shortcut peak ratio：错误预测中是否存在高单政策路径峰。

4. **消融实验**
   - 用 mean policy score 替代 policy soft-min，验证热带瓶颈不是普通 ensemble。
   - 用 route softmax 替代 slot soft-min，验证最弱环节约束的必要性。
   - 去掉 `L_policy_fragility`，观察单政策路径高峰是否重新出现。
   - 去掉 slot coverage，检查所有 slot 是否塌缩到同一高频观测。
   - 将 policy bank 替换为随机 mask，验证收益来自结构化采样政策而非普通增强。

## 5. 预期创新性

1. **从表示不变转向支持路径存活**：不要求多策略 logits 或 representation 一致，只要求真实类别存在跨政策仍完整的支持路径。
2. **从 pooling/attention 转向热带半环读出**：使用 slot soft-min、route soft-max、policy soft-min 三层半环结构，阻止单个采样政策下的局部高峰直接变成类别证据。
3. **从图拆分转向路径瓶颈**：吸收 STAR-Set 和 VP-GNN 的结构信号，但不拆 state/policy 图、不做 IV、不做 graph distance；只问支持路径是否跨政策可行。
4. **从投票裁决转向最弱环节鲁棒性**：不同于 Do-Jury 的社会选择排序，DTSR 不让政策视图投票，而是让每个类别必须通过最不利政策下的路径瓶颈。
5. **从选基/体积转向可解释路线**：不同于 DVNB 的 Nystrom/logdet 事件基，DTSR 不选择事件子集；它输出“哪条状态支持路径、哪个槽、哪种政策”决定了预测稳定性。
6. **与采样解耦低侵入兼容**：value encoder 只需输出 support atoms；sampling branch 只需生成政策视图；反事实模块只改变路径存活测试，不需要估计 hazard、density ratio、posterior factor、conformal threshold、Krylov operator 或 logdet volume。

## 6. 一句话投稿卖点

**Do-Tropical Support Routes 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“类别支持路径在反事实采样政策下的热带最弱环节存活问题”，通过 slot-level soft-min、route-level soft-max 与 policy-level soft-min，阻止 STAR-Set/VP-GNN 式 temporal bias、variable affinity 和 patch visibility 的单政策高峰支配分类，同时保留跨政策仍可连通的真实状态证据。**
