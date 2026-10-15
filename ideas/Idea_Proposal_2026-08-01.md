# Title: Do-Volume Nyström Basis：面向采样策略偏移的反事实体积基去冗余分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已读取自动化记忆 `MEMORIES.md` 以及 `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`、`idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`
  - `Idea_Proposal_2026-06-24.md`
  - `Idea_Proposal_2026-06-27.md`
  - `Idea_Proposal_2026-07-15.md`
  - `Idea_Proposal_2026-07-16.md`
  - `Idea_Proposal_2026-07-17.md`
  - `Idea_Proposal_2026-07-18.md`
  - `Idea_Proposal_2026-07-19.md`
  - `Idea_Proposal_2026-07-20.md`
  - `Idea_Proposal_2026-07-21.md`
  - `Idea_Proposal_2026-07-22.md`
  - `Idea_Proposal_2026-07-23.md`
  - `Idea_Proposal_2026-07-24.md`
  - `Idea_Proposal_2026-07-25.md`
  - `Idea_Proposal_2026-07-26.md`
  - `Idea_Proposal_2026-07-27.md`
  - `Idea_Proposal_2026-07-29.md`
  - `Idea_Proposal_2026-07-31.md`
- 已读取近期论文记录：`paper_daily.md`。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮永久避开以下核心机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、采样 score 零空间、hazard-driven resampling、do-risk variance。
8. 生理流算子与采样算子交换子、value/policy graph commutator、policy residual sink。
9. additive evidence market、protocol tax、token-level evidence budget、边际证据审计。
10. 模型空间 posterior quotient、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、ANOVA projection、VQ semantic clauses、HSIC redaction。
12. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler。
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
27. differentiable Borda / pairwise-majority policy jury、no-dictator juror leverage、structural bribery loss。
28. counterfactual low-rank policy displacement operators、Krylov policy subspace、annihilating polynomial over policy modes。
29. 单纯把 STAR-Set 的 temporal/variable attention bias 或 VP-GNN 的 variable/patch graph 拆成 state-policy 双分支并做一致性约束。

本提案选择新的正交切入点：**不删除采样信息、不做对抗、不做一致性、不做保形/认证/IV/排序裁决，也不把 STAR-Set 或 VP-GNN 的结构拆成 state-policy 双图；而是把非规则事件序列看成一个带政策冗余的点云，用 determinantal volume / Nyström basis 只保留能增加“状态几何体积”的少量事件基。采样政策制造的密集复测、panel 共现和 patch 可见性通常只增加重复点或政策坐标体积，不能显著增加条件状态体积，因此不会在分类读出中获得线性放大。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 中最近两个结构信号非常关键：

- **STAR-Set Transformer** 通过 temporal locality bias 与 variable-type affinity 把异步事件集合中的局部时间邻域和变量兼容结构重新注入 attention。
- **VP-GNN** 通过 variable-wise graph 与 patch-wise graph 表示不规则临床时序中的变量依赖和多尺度片段依赖。

这两类工作都提醒我们：sampling-policy shift 往往不是单个 mask ratio 变化，而是事件集合的**几何形状**变化：

- 报警后密集复测会在晚期 patch 产生一簇高度相似的重复点；
- 某家医院把 `lactate + WBC + CRP` 同步 panel 下单，会在变量亲和结构中产生高共现团；
- 固定查房式采样会让同一变量在规则局部邻域内重复出现；
- 测试医院拆分 panel 后，原来 attention bias 或 patch graph 中的强结构突然消失。

普通 Transformer / GNN / token sparsification 往往把“有多少 token 被看见”“哪些 token 共同出现”直接变成可学习权重。这样在训练医院中，策略制造的重复观测会被多次计入分类证据；换医院后，同样的病程状态因为观测日历不同而给出不同 logit。

**Do-Volume Nyström Basis (DVNB)** 的核心直觉是：

> 真正有状态信息的新观测，会在 value-driven latent space 中增加新的几何方向；采样政策制造的密集复测、联测 panel 或 patch 可见性，常常只是在已有方向附近堆叠冗余点。分类器不应该按 token 数量累加证据，而应该按事件集合张成的“状态体积”读取证据。

因此，DVNB 不问“某个 token 是否重要”，也不问“哪个 policy view 的 logits 更稳定”。它把事件表征组织成一个 **反事实体积基**：

```text
event cloud -> state kernel K_state
            -> policy-structure kernel K_policy
            -> conditional state volume basis
            -> Nyström readout
```

当采样政策只增加冗余观测时，`log det(K_state)` 很快饱和；当采样政策制造新的 temporal/variable/patch 结构但没有新增状态方向时，`K_policy` 体积会增加，但条件状态体积不会同步增加。最终分类头只消费由少量体积基重构出的状态表示，避免被政策诱导的 token multiplicity 和结构重复放大。

这与当前“采样解耦/反事实干预”框架的结合方式很直接：

- value process 仍然编码观测值、时间和变量，输出 state event embeddings；
- sampling branch 不进入分类头，只输出 policy descriptors，用于衡量某个事件基是否主要扩张了政策结构体积；
- counterfactual intervention 不用于一致性、风险方差、保形、IV 或 jury，而是生成 panel merge/split、late burst、patch budget 等结构压力视图，用来训练体积基在不同政策下保持“状态体积优先、政策体积不过载”。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从 attention pooling 改为 Conditional Volume Event Basis

给定事件流：

```text
E = {(x_i, t_i, d_i)}_{i=1}^N
```

DVNB 先生成两组嵌入：

1. **State Event Embedding `z_i`**
   - 输入观测值、变量 id、归一化时间、局部 delta-t。
   - 表示该事件带来的 value-driven 状态方向。
   - 不把 policy id、环境标签或 missingness pattern 直接拼入分类头。

2. **Policy Structure Descriptor `p_i`**
   - 只描述观测坐标和结构：时间窗、局部事件密度、变量 id、panel/patch id、相邻变量是否共现。
   - 对应 STAR-Set 的 temporal locality / variable affinity 与 VP-GNN 的 patch visibility。
   - 不作为分类特征读出，只用于体积基选择时估计“这个事件是否主要扩张了政策结构空间”。

构造两个 Gram kernel：

```text
K_state[i, j]  = <z_i, z_j>
K_policy[i, j] = <p_i, p_j>
```

然后学习 soft basis weights `w_i`。选择原则不是普通 attention score，而是：

```text
maximize  log det(I + Z_w Z_w^T)
minimize  policy-volume overload
```

其中 `Z_w = diag(sqrt(w)) Z`。直觉：

- 若一批 token 只是密集复测的重复点，它们彼此高度相似，log-det 边际增益很小；
- 若一批 token 只扩张了 policy descriptor，例如 panel 同窗共现或 late-burst patch，而没有扩张 state latent directions，则 policy-volume overload 会升高；
- 若一个 token 代表新的状态变化方向，即使它来自稀疏采样，也会显著增加 state log-volume。

最后用 basis weights 做 Nyström-style readout：

```text
h_basis = sum_i w_i z_i
logits = Classifier([h_basis, volume_stats])
```

这里的 `volume_stats` 只包括 state volume、conditional volume ratio、basis effective rank 等诊断量，不包括 policy id 或原始 mask pattern。

### 2.2 改 Loss：从一致性/校准转向 Determinantal Volume Sobriety

总目标：

```text
L = L_cls
  + lambda_vol * L_state_volume
  + lambda_pol * L_policy_overload
  + lambda_cf  * L_counterfactual_volume_floor
  + lambda_ent * L_basis_entropy
```

#### A. 分类损失 `L_cls`

用体积基读出做标准分类：

```text
L_cls = CE(logits_volume, y)
```

分类头看不到 sampling policy descriptor 的逐 token 表征，只能看基于 state event embeddings 的 Nyström 汇总。

#### B. State Volume Loss `L_state_volume`

鼓励被选事件基张成足够大的状态体积：

```text
V_state = log det(I + Z_w Z_w^T / sigma_state^2)
L_state_volume = - mean(V_state)
```

它不同于 MedSpaformer 式 token sparsification：不是挑“最显著 token”，而是挑能共同张成互补状态方向的事件基。

#### C. Policy Overload Loss `L_policy_overload`

若选中的基主要扩张 policy descriptor 体积，而 state volume 没有同步增加，则说明分类器可能在购买采样结构而不是状态信息：

```text
V_policy = log det(I + P_w P_w^T / sigma_policy^2)
overload = V_policy / (V_state + eps)
L_policy_overload = relu(overload - rho_max)^2
```

这不是 graph 分解、policy adversarial 或 protocol tax。`p_i` 不进入分类头，也不作为成本参与 token 买卖；它只是 determinantal sobriety 的结构诊断坐标。

#### D. Counterfactual Volume Floor `L_counterfactual_volume_floor`

当前反事实采样模块生成多种 policy stress views，例如：

- `panel_merge`：把异步变量压成联测窗口；
- `panel_split`：把同步 panel 拆成异步事件；
- `late_burst`：模拟报警后密集复测；
- `patch_budget`：每个 patch 只保留少量事件；
- `routine_round`：把时间戳压到固定查房轮次。

DVNB 不要求这些 views 的 logits 一致，也不做 conformal / smoothing / IV。它只要求每个合法采样视图都能找到足够的 state-volume basis：

```text
L_counterfactual_volume_floor =
  mean_r relu(v_min - V_state(do(policy_r, x)))^2
```

这避免模型只在训练采样日历下找到体积基，而在 panel split 或 patch budget 改变后退化为政策结构读出。

#### E. Basis Entropy `L_basis_entropy`

若 soft basis 权重塌缩到单点，体积基退化为普通 attention；若完全均匀，又会重新按 token 数量累加。加入轻量有效基数约束：

```text
N_eff = 1 / sum_i w_i^2
L_basis_entropy = relu(n_min - N_eff)^2 + relu(N_eff - n_max)^2
```

它不是 evidence budget，也不是 token tax；它只保证 determinantal basis 既不塌缩也不变成普通平均池化。

### 2.3 改 Dataloader：返回 Volume Stress Views

新增 `VolumeStressCollator`，每个 batch 返回：

1. `factual_batch`：原始事件流。
2. `policy_volume_view_bank`：仅用于体积基训练的反事实结构视图：
   - `late_burst_thicken`
   - `panel_merge`
   - `panel_split`
   - `patch_budget_thin`
   - `routine_round_time`
3. `policy_descriptor`：时间窗、变量 id、局部密度、panel/patch 结构。
4. `event_mask`：padding mask。

关键差异：

- 不生成对比学习正样本。
- 不要求多视图 logits 或 representation 一致。
- 不估计 hazard、density ratio、posterior factor、instrument、conformal score 或 solver trace。
- 不把 policy descriptor 输入分类头。
- 反事实 view 只用于检查：在采样政策变化后，模型是否仍能从 value events 中找到足够的非冗余状态体积。

### 2.4 推理阶段

部署时只需事实观测：

1. 计算 state embeddings `Z` 与 policy descriptors `P`。
2. 选择 soft determinantal basis `w`。
3. 输出分类 logits。
4. 额外输出诊断：
   - `state_volume`：当前样本可恢复的状态几何体积；
   - `policy_overload`：选中 basis 是否主要扩张政策结构；
   - `basis_effective_rank`：预测是否由少数事件支配；
   - `volume_leverage_by_event`：哪些事件真正增加了状态体积。

当 `policy_overload` 高而 `state_volume` 低时，系统可提示“当前预测可能依赖采样结构冗余”，建议补测能最大增加 state log-det 的变量或时间窗。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(score: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    score = score.masked_fill(mask == 0, -1e4)
    weight = torch.softmax(score, dim=dim)
    return weight * mask.to(weight.dtype)


def stable_logdet_from_features(features: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Compute log det(I + X X^T) for weighted event features.

    Args:
        features: [B, N, H].
    Returns:
        logdet: [B].
    """
    bsz, _, hidden_dim = features.shape
    gram = torch.bmm(features.transpose(1, 2), features)
    eye = torch.eye(hidden_dim, device=features.device, dtype=features.dtype)[None]
    sign, logabsdet = torch.linalg.slogdet(eye + gram + eps * eye)
    return torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))


def effective_basis_size(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 / weights.pow(2).sum(dim=1).clamp_min(eps)


class EventStateLift(nn.Module):
    """Lift irregular events into value-driven state embeddings."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp_min(0)
        mask = batch["event_mask"]

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_h = self.var_embed(var_id)
        event_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(event_x) * mask.unsqueeze(-1)


class PolicyDescriptorLift(nn.Module):
    """Encode observation-structure descriptors for volume diagnostics only."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp_min(0)
        mask = batch["event_mask"]

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        local_density = 1.0 / (1.0 + delta_t)
        early = (time_norm <= 0.33).to(time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype)
        late = (time_norm > 0.66).to(time.dtype)

        changed_var = torch.zeros_like(mask)
        changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
        close_gap = (delta_t <= (delta_t * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)).to(mask.dtype)
        local_coobs = changed_var * close_gap

        patch_id = torch.floor(time_norm * 4.0).clamp(0, 3) / 3.0
        var_onehot = F.one_hot(var_id, self.num_vars).to(time.dtype)
        desc = torch.cat(
            [
                var_onehot,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                local_density.unsqueeze(-1),
                early.unsqueeze(-1),
                middle.unsqueeze(-1),
                late.unsqueeze(-1),
                local_coobs.unsqueeze(-1),
                patch_id.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(desc) * mask.unsqueeze(-1)


class VolumeBasisSelector(nn.Module):
    """Soft determinantal basis selector for event clouds."""

    def __init__(self, hidden_dim: int, policy_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim + policy_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_z: torch.Tensor, policy_p: torch.Tensor, mask: torch.Tensor) -> dict:
        state_norm = state_z.pow(2).sum(dim=-1, keepdim=True)
        policy_norm = policy_p.pow(2).sum(dim=-1, keepdim=True)
        local_ratio = state_norm / (policy_norm + 1e-4)
        raw = self.score(torch.cat([state_z, policy_p, state_norm, policy_norm, local_ratio], dim=-1)).squeeze(-1)
        weights = masked_softmax(raw, mask, dim=1)

        sqrt_w = weights.clamp_min(0.0).sqrt().unsqueeze(-1)
        weighted_state = state_z * sqrt_w
        weighted_policy = policy_p * sqrt_w

        state_volume = stable_logdet_from_features(weighted_state)
        policy_volume = stable_logdet_from_features(weighted_policy)
        overload = policy_volume / state_volume.clamp_min(1e-4)

        pooled = (weights.unsqueeze(-1) * state_z).sum(dim=1)
        return {
            "weights": weights,
            "pooled": pooled,
            "state_volume": state_volume,
            "policy_volume": policy_volume,
            "policy_overload": overload,
            "basis_eff_size": effective_basis_size(weights),
        }


class DoVolumeNystromBasisClassifier(nn.Module):
    """Policy-shift robust classifier using counterfactual volume bases."""

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        policy_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.state_lift = EventStateLift(num_vars=num_vars, hidden_dim=hidden_dim)
        self.policy_lift = PolicyDescriptorLift(num_vars=num_vars, hidden_dim=policy_dim)
        self.selector = VolumeBasisSelector(hidden_dim=hidden_dim, policy_dim=policy_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_once(self, batch: dict) -> dict:
        state_z = self.state_lift(batch)
        policy_p = self.policy_lift(batch)
        basis = self.selector(state_z, policy_p, batch["event_mask"])
        stats = torch.stack(
            [
                basis["state_volume"],
                basis["policy_volume"],
                basis["policy_overload"],
                basis["basis_eff_size"],
            ],
            dim=-1,
        )
        logits = self.classifier(torch.cat([basis["pooled"], stats], dim=-1))
        return {**basis, "state_z": state_z, "policy_p": policy_p, "logits": logits}

    def forward(self, batch: dict) -> dict:
        return self.encode_once(batch)

    def training_loss(
        self,
        batch: dict,
        lambda_vol: float = 0.08,
        lambda_pol: float = 0.20,
        lambda_cf: float = 0.15,
        lambda_ent: float = 0.03,
        overload_cap: float = 0.75,
        min_cf_volume: float = 1.5,
        n_eff_min: float = 2.0,
        n_eff_max: float = 12.0,
    ) -> dict:
        labels = batch["labels"]
        out = self.encode_once(batch)

        cls_loss = F.cross_entropy(out["logits"], labels)

        state_volume_loss = -out["state_volume"].mean()
        policy_overload_loss = F.relu(out["policy_overload"] - overload_cap).pow(2).mean()

        cf_losses = []
        for cf_batch in batch.get("policy_volume_view_bank", []):
            cf_out = self.encode_once(cf_batch)
            cf_losses.append(F.relu(min_cf_volume - cf_out["state_volume"]).pow(2).mean())
        if cf_losses:
            cf_volume_loss = torch.stack(cf_losses).mean()
        else:
            cf_volume_loss = torch.zeros((), device=labels.device)

        n_eff = out["basis_eff_size"]
        basis_entropy_loss = (
            F.relu(n_eff_min - n_eff).pow(2)
            + F.relu(n_eff - n_eff_max).pow(2)
        ).mean()

        total = (
            cls_loss
            + lambda_vol * state_volume_loss
            + lambda_pol * policy_overload_loss
            + lambda_cf * cf_volume_loss
            + lambda_ent * basis_entropy_loss
        )

        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "state_volume_loss": state_volume_loss.detach(),
            "policy_overload_loss": policy_overload_loss.detach(),
            "cf_volume_floor_loss": cf_volume_loss.detach(),
            "basis_entropy_loss": basis_entropy_loss.detach(),
            "mean_state_volume": out["state_volume"].mean().detach(),
            "mean_policy_overload": out["policy_overload"].mean().detach(),
            "mean_basis_eff_size": out["basis_eff_size"].mean().detach(),
        }


@torch.no_grad()
def build_policy_volume_view_bank(batch: dict) -> list[dict]:
    """Create counterfactual policy views for volume-basis training only."""
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon
    views = []

    def clone_with(new_value, new_time, new_var, new_mask):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        return out

    # 1. Late-burst thickening: duplicate the effect of dense alarm follow-up by
    # retaining late events and thinning early routine events.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    late_burst_mask = torch.where(late > 0, mask, mask * alternating)
    views.append(clone_with(value * late_burst_mask, time, var_id, late_burst_mask))

    # 2. Panel split: weaken local cross-variable co-observation.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_split_mask = mask * (1.0 - 0.5 * close * changed_var)
    views.append(clone_with(value * panel_split_mask, time, var_id, panel_split_mask))

    # 3. Patch budget thinning: retain a small number of observations per coarse patch.
    patch_keep = torch.zeros_like(mask)
    for start, end in [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]:
        in_patch = ((time_norm >= start) & (time_norm < end)).to(mask.dtype) * mask
        rank = in_patch.cumsum(dim=1)
        patch_keep = torch.maximum(patch_keep, (rank <= 2).to(mask.dtype) * in_patch)
    views.append(clone_with(value * patch_keep, time, var_id, patch_keep))

    # 4. Routine-round time: snap timestamps to coarse rounds without changing values.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    views.append(clone_with(value * mask, rounded_time, var_id, mask))

    return views
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `late-burst shift`：训练环境报警后密集复测，测试环境只保留固定复查。
   - `panel merge/split shift`：训练环境同步 panel，测试环境拆成异步变量。
   - `patch-budget shift`：某些高权重 patch 在测试策略下被压缩。
   - `routine-round shift`：从事件触发采样换到固定查房时间。

2. **对比方法**
   - STAR-Set / VP-GNN 原模型。
   - 普通 attention pooling / Set Transformer。
   - MedSpaformer-style token sparsification。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、D-IVSP、DJRT、KPMA 等。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - state volume under shift：反事实采样后状态体积是否仍可恢复。
   - policy overload ratio：选中基是否主要扩张政策结构。
   - basis effective size：预测是否退化为单 token 或普通平均。
   - duplicate amplification score：密集复测增加 token 数时 logits 是否异常放大。
   - volume leverage localization：哪些变量/时间窗真正增加 state log-det。

4. **消融实验**
   - 去掉 `L_policy_overload`，检查模型是否重新利用 panel / patch 政策结构。
   - 去掉 `L_state_volume`，验证是否退化为普通 attention pooling。
   - 去掉 `L_counterfactual_volume_floor`，检查在 unseen policy views 下体积基是否塌缩。
   - 将 policy descriptors 替换为随机噪声，验证 STAR-Set/VP-GNN 式结构诊断是否必要。
   - 用普通 top-k token selection 替代 volume basis，验证收益来自 determinantal de-duplication 而非少 token 计算。

## 5. 预期创新性

1. **从 token 重要性转向事件集合体积**：不是选择最显著 token，而是选择能张成互补状态方向的 Nyström basis，天然抑制密集复测与重复 panel 的数量放大。
2. **从结构分解转向条件体积清醒度**：吸收 STAR-Set 的 temporal/variable bias 和 VP-GNN 的 patch/variable graph 作为 policy descriptor，但不拆 state/policy graph，也不让它们进入分类头。
3. **从反事实一致性转向体积可恢复性**：counterfactual intervention 只检查不同采样政策下是否仍能恢复足够 state volume，不要求 logits 或 representation 相同。
4. **从 sparsification 转向 determinantal de-duplication**：与 MedSpaformer 式 token sparsification 不同，DVNB 的核心是 log-det marginal volume；重复 token 即使很多，也不会线性增加分类证据。
5. **部署诊断直接**：state volume、policy overload、basis leverage 可以告诉用户当前预测是在利用新的状态方向，还是被采样政策制造的结构冗余支撑。

## 6. 一句话投稿卖点

**Do-Volume Nyström Basis 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样政策改变事件点云中的冗余体积，而非真实状态体积”的问题，并通过 determinantal volume basis、policy-overload sobriety 与 counterfactual volume floor，让分类器按可恢复状态几何方向读出证据，而不是按医院协议、设备 cadence 或 panel/patch 可见性制造的 token 数量和结构重复读出证据。**
