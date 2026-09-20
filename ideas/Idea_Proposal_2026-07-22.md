# Title: Policy-Artifact MDL Episode Transducer：面向采样策略偏移的策略产物最短描述事件转导器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md` 与 `*Summary*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md`：纳入其中记录的 2026-06-12 至 2026-07-21 历史提案机制黑名单。
- 已读取近期论文记录：`paper_daily.md`。
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

### 历史核心机制黑名单

为避免与历史提案发生思维重合，本提案明确避开以下方向作为主创新：

1. 采样危险率、采样 score 零空间、hazard-driven counterfactual resampling、do-risk variance。
2. 生理流与采样算子的交换子、图交换、policy residual sink。
3. additive evidence market、protocol tax、token evidence budget、边际证据审计。
4. posterior quotient、采样似然因子相除、模型空间 posterior、干预积分分类。
5. reconstruction error cartography、VQ semantic clauses、HSIC redaction、policy-sensitive acquisition clauses。
6. randomized smoothing、certified policy radius、policy simplex kernel。
7. Radon-Nikodym density ratio、doubly robust target-measure correction、influence bound。
8. optional-stopping martingale query、标准化 innovation、stopping recipe moment control。
9. excursion topology capsules、censored persistence interval、fragmentation sobriety。
10. policy gauge、horizontal transport、chart span、vertical blindness。
11. policy-only negative film、shadow eraser/stencil、shadow nullification。
12. parity-check code、syndrome locator、packet repair。
13. conditional knockoff calendar、soft knockoff-FDR firewall、swap symmetry。
14. observability witness、measurement-action bisimulation、canonical action battery。
15. evidential shield、Dirichlet uncertainty、policy-induced ignorance/vacuity。
16. policy lattice、meet/join visibility、submodular margins、quality-order loss。
17. solver trace front-door、NFE/roughness trace standardization。
18. RKHS cubature weights、kernel moment exactness、cubature KKT solver。
19. policy-word signature、shuffle algebra counterterms、thermodynamic free energy、Sklar copula、triage queue debt、Sinkhorn detail-to-anchor canonicalization。
20. 直接复用 FlowPath、GSNF、SuperMAN、GARLIC、DBGL、iTimER、Record2Vec、QuITE、SDEVI、MTM、MedMamba、MedSpaformer、MILM、StarEmbed、LLMTS、MVC-CDE 或 CDI-TS 的原始机制作为主创新。

本提案选择新的正交切入点：

> **不估计采样概率，不做对抗，不做一致性，不做后验除法，不做证据/不确定性/拓扑/图/gauge/纠错/knockoff/solver trace；而是把采样政策造成的额外观测、重复复测、panel 合并/拆分、值等待与局部高分辨率细节看作“策略产物”。模型必须先用一个最短描述长度（MDL）事件转导器，把原始不规则事件流压缩成少量 label-relevant episode；分类器只读取压缩后的 episode program，而不读取可由策略产物解释的冗余细节。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中的前沿机制给了三个关键启发：

1. **MILM** 证明 sampling pattern 本身具有很强预测力，但这也意味着模型容易把医院流程、value-pending 或检查触发规则学成类别捷径。
2. **Rethinking LLMs for Irregular Time Series Classification in Critical Care** 指出，真正关键的是前端 encoder 是否尊重不规则事件结构，而不是把序列粗暴交给后端大模型。
3. **Enhancing Sparse Event Detection via Context-Detail Interaction** 提醒我们：医疗时序中真正有判别力的 often 是稀疏 episode 或局部事件，而不是每个采样 token 都同等重要；但普通 context-detail gate 仍可能把“哪里被高分辨率观测”学成采样策略捷径。

当前“采样解耦/反事实干预”框架已经能生成不同采样策略下的观测流。历史方案大多把这些干预用于去偏、审计、对齐、平滑、积分、认证或不确定性建模。本提案换一个角度：

```text
同一底层病程 / 状态轨迹
  -> 策略 A: routine sparse observations
  -> 策略 B: alarm-triggered dense follow-up
  -> 策略 C: panel merge / panel split
```

这些不同策略会改变原始 token 数量、局部密度、panel 同步结构和值返回状态，但它们常常共享同一个更短的状态叙事：

```text
episode_1: baseline stable
episode_2: acute transition
episode_3: partial recovery / deterioration
```

如果一个分类器必须依赖“报警后多出来的 12 个复测 token”才能预测，那么它学到的是策略产物；如果它能把这些复测 token 压缩为同一个 acute transition episode，再进行分类，就更可能跨采样政策稳定。

**Policy-Artifact MDL Episode Transducer (PA-MDL)** 的核心目标是：

> 把反事实采样干预从“生成增强视图”改造成“生成策略产物 production rules”。模型用可微 MDL 转导器解释哪些原始事件应被保留为状态 episode，哪些应被作为 sampling artifact 以 insert/delete/repeat/merge/pending 等 production 压缩掉。最终分类只基于最短 episode program。

这能解决采样偏移问题，因为不同医院或设备的采样政策首先改变的是描述长度：高频复测、panel 联测和值等待会让 raw event stream 变长、更碎、更同步；但若这些只是策略产物，它们应被低成本压缩，而不是提升分类 logit。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从事件流编码改为 Episode Program Compressor

输入仍是标准不规则事件：

```text
(event_value, event_time, event_var_id, event_mask, measurement_std)
```

PA-MDL 前端包括三层：

1. **Event Lift**
   - 编码观测值、变量 id、相对时间间隔和测量质量。
   - 不把 policy id、environment id 或纯 missingness pattern 直接送入分类头。

2. **Episode Compressor**
   - 从事件流中产生少量连续 episode states `E = {e_1, ..., e_M}`。
   - episode 数量远小于事件数，承担“病程叙事骨架”的角色。
   - episode assignment 由 value dynamics 与局部变化驱动，而不是 learnable reference time points。

3. **Policy-Artifact Production Cost**
   - 采样分支只看采样坐标、变量可见性、panel/pending/quality 信息，输出每个事件的 production cost：

```text
op in {
  keep_state_event,      # 保留为状态 episode 证据
  insert_protocol_token, # 策略额外插入/复测
  delete_censored_token, # 策略遮蔽导致缺失
  repeat_followup,       # 报警后密集重复
  merge_panel,           # panel 同步合并
  pending_value          # 值尚未返回但坐标已出现
}
```

这些 production cost 不进入分类 logits；它们只用于计算 raw events 到 episode program 的最短描述长度。

### 2.2 改 Classifier：从 pooled representation 改为 Label Automaton Readout

普通分类器直接读事件池化：

```text
logits = Classifier(pool(event_states))
```

PA-MDL 改成：

```text
episode_program = MDLTransducer(raw_events, production_cost)
logits = LabelAutomaton(episode_program)
```

`LabelAutomaton` 不是原型分类器，也不是 VQ semantic compiler。它是一个连续状态自动机：每个类别对应一组可学习 transition matrices，episode state 只负责驱动状态转移。它关心“episode 叙事顺序”是否匹配某类病程，而不是原始采样 token 如何堆叠。

### 2.3 改 Loss：从一致性/对抗转向 MDL Compression Contracts

总目标：

```text
L = L_cls
  + lambda_mdl  * L_description_length
  + lambda_prod * L_production_supervision
  + lambda_art  * L_artifact_absorption
  + lambda_use  * L_episode_usage
```

#### A. Classification Loss `L_cls`

只使用 episode program 的 label automaton logits：

```text
L_cls = CE(LabelAutomaton(E), y)
```

原始事件池化 logits 不参与训练主目标，避免模型绕过压缩器直接读取采样密度。

#### B. Description Length Loss `L_description_length`

可微动态规划计算 raw event stream 到 episode program 的最短编码长度：

```text
DL = min_path sum_i production_cost(op_i)
             + match_cost(event_i, episode_j)
             + code_cost(episode_j)
```

若某个观测是高频复测、panel 同步或 pending-value 策略产物，它应被低成本解释为 artifact production；若某个观测包含不可压缩的状态变化，则应保留为 episode 证据。

这不是 token sparsification：不是简单选少数 token；它要求未被选中的 token 能被明确的策略 production rule 解释。

#### C. Production Supervision `L_production_supervision`

当前反事实干预模块生成已知采样 production recipes：

- `dense_followup_insert`：报警后增加复测事件；
- `panel_merge_split`：变量 panel 同步或拆开；
- `routine_budget_delete`：固定预算导致某些事件不可见；
- `value_pending`：只有观测坐标出现，值尚未返回；
- `detail_resolution_change`：局部细节被高分辨率或低分辨率采样。

这些 recipe 给出 event-level operation target：

```text
L_production_supervision = CE(op_logits, op_target)
```

它不要求反事实视图 logits 一致，也不让 operation target 进入分类器；它只教转导器识别“这个 token 是状态 episode 还是策略产物”。

#### D. Artifact Absorption Loss `L_artifact_absorption`

对由反事实采样明确插入的 protocol artifacts，要求它们在 episode program 中的保留概率低：

```text
L_artifact_absorption =
  mean artifact_mask * keep_state_probability^2
```

这与 policy-shadow / stencil 不同：这里没有 shadow film，也没有负片编码器；artifact 只通过 MDL production 被吸收，不产生单独的分类支路。

#### E. Episode Usage Loss `L_episode_usage`

避免压缩器把所有事件都删除，或把每个事件都变成 episode：

```text
L_episode_usage =
  relu(min_usage - episode_usage)^2
  + relu(episode_usage - max_usage)^2
```

它保证模型保留足够的状态叙事，同时压缩明显策略产物。

### 2.4 改 Dataloader：返回 Policy-Artifact Production Bank

新增 `PolicyArtifactProductionCollator`。每个 batch 返回：

1. 原始事实事件：`event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_std`。
2. `artifact_recipe_bank`：
   - `dense_followup_insert`
   - `panel_merge_split`
   - `routine_budget_delete`
   - `value_pending`
   - `detail_resolution_change`
3. `op_target_bank`：每个事件对应的 production label。
4. `artifact_mask_bank`：哪些事件是由策略 recipe 显式制造的产物。
5. `episode_budget_hint`：由序列长度、变量数和观测窗口粗略给出的 episode 数量上限。

反事实模块不再生成 consistency pair、risk view、policy simplex samples、knockoff calendar、lattice meet/join、solver trace bank 或 Sinkhorn exposure edits；它只生成可解释的策略 production rules。

### 2.5 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `EventLift + EpisodeCompressor`。
- 现有 sampling branch 改为 `PolicyArtifactCostNet`，只输出 production cost 和 operation logits。
- 现有 counterfactual intervention 改为 `PolicyArtifactProductionBank`，生成可监督的策略产物。
- 分类头改为 `LabelAutomatonReadout`，只读压缩后的 episode program。
- 推理阶段：
  - 输出分类概率；
  - 输出 episode program；
  - 输出 raw token 的 artifact absorption map；
  - 输出 description length ratio：`DL(raw -> episode) / raw_length`，作为采样策略复杂度诊断。

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


def softmin(values: torch.Tensor, temperature: float = 0.08) -> torch.Tensor:
    return -temperature * torch.logsumexp(-values / temperature, dim=-1)


class EventLift(nn.Module):
    """Lift irregular observations into value-driven event states."""

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
        meas_std = batch.get("measurement_std", torch.zeros_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon

        x = torch.cat(
            [
                self.var_embed(var_id),
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(meas_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(x) * mask.unsqueeze(-1)


class EpisodeCompressor(nn.Module):
    """Compress event states into a short continuous episode program."""

    def __init__(self, hidden_dim: int, max_episodes: int):
        super().__init__()
        self.max_episodes = max_episodes
        self.boundary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.assign = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_episodes),
        )
        self.episode_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, event_h: torch.Tensor, event_time: torch.Tensor, event_mask: torch.Tensor) -> dict:
        # Boundary signal is value-driven; no learned reference timestamps are used.
        boundary_prob = torch.sigmoid(self.boundary(event_h)).squeeze(-1) * event_mask
        progress = boundary_prob.cumsum(dim=1)
        progress = progress / progress.amax(dim=1, keepdim=True).clamp_min(1.0)

        assign_logits = self.assign(torch.cat([event_h, progress.unsqueeze(-1)], dim=-1))
        assign_logits = assign_logits.masked_fill(event_mask.unsqueeze(-1) == 0, -1e4)
        assign_prob = torch.softmax(assign_logits, dim=-1) * event_mask.unsqueeze(-1)

        denom = assign_prob.sum(dim=1).clamp_min(1.0)
        episode = torch.einsum("bnm,bnh->bmh", assign_prob, event_h) / denom.unsqueeze(-1)
        episode = self.episode_proj(episode)
        episode_usage = denom / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return {
            "episode": episode,
            "assign_prob": assign_prob,
            "boundary_prob": boundary_prob,
            "episode_usage": episode_usage,
        }


class PolicyArtifactCostNet(nn.Module):
    """Estimate production costs from sampling coordinates only."""

    def __init__(self, num_vars: int, hidden_dim: int, num_ops: int):
        super().__init__()
        self.num_vars = num_vars
        self.coord_net = nn.Sequential(
            nn.Linear(num_vars + 7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.op_head = nn.Linear(hidden_dim, num_ops)
        self.cost_head = nn.Sequential(nn.Linear(hidden_dim, num_ops), nn.Softplus())

    def forward(self, batch: dict) -> dict:
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp_min(0)
        mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(time))
        pending = batch.get("value_pending_mask", torch.zeros_like(time))
        panel = batch.get("panel_mask", torch.zeros_like(time))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        local_density = 1.0 / (1.0 + delta_t)
        early = (time_norm <= 0.33).to(time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype)
        late = (time_norm > 0.66).to(time.dtype)

        var_onehot = F.one_hot(var_id, self.num_vars).to(time.dtype)
        coord = torch.cat(
            [
                var_onehot,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                local_density.unsqueeze(-1),
                early.unsqueeze(-1) + 0.5 * middle.unsqueeze(-1),
                late.unsqueeze(-1),
                panel.unsqueeze(-1),
                pending.unsqueeze(-1),
                torch.log1p(meas_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        h = self.coord_net(coord) * mask.unsqueeze(-1)
        op_logits = self.op_head(h)
        op_cost = self.cost_head(h) + 1e-4
        return {"op_logits": op_logits, "op_cost": op_cost}


def soft_edit_description_length(
    event_h: torch.Tensor,
    episode: torch.Tensor,
    op_cost: torch.Tensor,
    event_mask: torch.Tensor,
    keep_op: int = 0,
    insert_op: int = 1,
    delete_op: int = 2,
    temperature: float = 0.08,
) -> torch.Tensor:
    """Differentiable MDL alignment from raw events to episode program.

    dp[i, j] softly minimizes the code length for first i events and first j episodes.
    keep/match consumes one event and one episode; insert/delete absorb strategy artifacts.
    """

    bsz, num_events, hidden_dim = event_h.shape
    num_episodes = episode.size(1)
    big = torch.full((bsz,), 1e4, device=event_h.device, dtype=event_h.dtype)
    dp = [[None for _ in range(num_episodes + 1)] for _ in range(num_events + 1)]
    dp[0][0] = torch.zeros(bsz, device=event_h.device, dtype=event_h.dtype)
    for i in range(1, num_events + 1):
        active = event_mask[:, i - 1]
        dp[i][0] = dp[i - 1][0] + active * op_cost[:, i - 1, insert_op]
    for j in range(1, num_episodes + 1):
        dp[0][j] = dp[0][j - 1] + 0.05

    for i in range(1, num_events + 1):
        active = event_mask[:, i - 1]
        for j in range(1, num_episodes + 1):
            match = (event_h[:, i - 1] - episode[:, j - 1]).pow(2).mean(dim=-1)
            keep = dp[i - 1][j - 1] + active * (match + op_cost[:, i - 1, keep_op])
            insert = dp[i - 1][j] + active * op_cost[:, i - 1, insert_op]
            delete = dp[i][j - 1] + 0.05 + op_cost[:, i - 1, delete_op] * 0.0
            stacked = torch.stack([keep, insert, delete], dim=-1)
            dp[i][j] = torch.where(active > 0, softmin(stacked, temperature), dp[i - 1][j])

    return dp[num_events][num_episodes] / event_mask.sum(dim=1).clamp_min(1.0)


class LabelAutomatonReadout(nn.Module):
    """Continuous label automata over compressed episode programs."""

    def __init__(self, hidden_dim: int, num_classes: int, automaton_states: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.automaton_states = automaton_states
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes * automaton_states * automaton_states),
        )
        self.start = nn.Parameter(torch.zeros(num_classes, automaton_states))
        self.accept = nn.Parameter(torch.randn(num_classes, automaton_states) * 0.02)

    def forward(self, episode: torch.Tensor, usage: torch.Tensor) -> torch.Tensor:
        bsz, num_episodes, _ = episode.shape
        state = torch.softmax(self.start, dim=-1)[None].expand(bsz, -1, -1)
        trans_raw = self.transition(episode)
        trans = trans_raw.view(
            bsz,
            num_episodes,
            self.num_classes,
            self.automaton_states,
            self.automaton_states,
        )
        trans = torch.softmax(trans, dim=-1)
        active = (usage > 0.01).to(episode.dtype)
        for idx in range(num_episodes):
            next_state = torch.einsum("bcs,bcst->bct", state, trans[:, idx])
            state = active[:, idx, None, None] * next_state + (1.0 - active[:, idx, None, None]) * state
        return torch.einsum("bcs,cs->bc", state, self.accept)


class PolicyArtifactMDLEpisodeTransducer(nn.Module):
    """Sampling-policy robust classifier through MDL episode transduction."""

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        max_episodes: int,
        num_classes: int,
        num_ops: int = 6,
    ):
        super().__init__()
        self.event_lift = EventLift(num_vars, hidden_dim)
        self.compressor = EpisodeCompressor(hidden_dim, max_episodes)
        self.cost_net = PolicyArtifactCostNet(num_vars, hidden_dim, num_ops)
        self.readout = LabelAutomatonReadout(hidden_dim, num_classes)

    def forward(self, batch: dict) -> dict:
        event_h = self.event_lift(batch)
        comp = self.compressor(event_h, batch["event_time"], batch["event_mask"])
        cost = self.cost_net(batch)
        logits = self.readout(comp["episode"], comp["episode_usage"])
        dl = soft_edit_description_length(
            event_h=event_h,
            episode=comp["episode"],
            op_cost=cost["op_cost"],
            event_mask=batch["event_mask"],
        )
        return {"event_h": event_h, "logits": logits, "description_length": dl, **comp, **cost}

    def training_loss(
        self,
        batch: dict,
        lambda_mdl: float = 0.20,
        lambda_prod: float = 0.15,
        lambda_art: float = 0.10,
        lambda_use: float = 0.05,
        min_usage: float = 0.05,
        max_usage: float = 0.65,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]
        cls_loss = F.cross_entropy(out["logits"], labels)
        mdl_loss = out["description_length"].mean()

        op_target = batch["op_target"]
        prod_raw = F.cross_entropy(
            out["op_logits"].transpose(1, 2),
            op_target,
            reduction="none",
        )
        prod_loss = (prod_raw * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

        keep_prob = torch.softmax(out["op_logits"], dim=-1)[..., 0]
        artifact_mask = batch.get("artifact_mask", torch.zeros_like(batch["event_mask"]))
        artifact_loss = (artifact_mask * keep_prob.pow(2)).sum() / artifact_mask.sum().clamp_min(1.0)

        usage = out["episode_usage"]
        usage_loss = (
            F.relu(min_usage - usage).pow(2)
            + F.relu(usage - max_usage).pow(2)
        ).mean()

        total = (
            cls_loss
            + lambda_mdl * mdl_loss
            + lambda_prod * prod_loss
            + lambda_art * artifact_loss
            + lambda_use * usage_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "description_length_loss": mdl_loss.detach(),
            "production_loss": prod_loss.detach(),
            "artifact_absorption_loss": artifact_loss.detach(),
            "episode_usage_loss": usage_loss.detach(),
            "mean_description_length": out["description_length"].mean().detach(),
        }


@torch.no_grad()
def build_policy_artifact_targets(batch: dict, num_ops: int = 6) -> dict:
    """Sketch production labels for policy-artifact supervision."""

    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    value = batch["event_value"]
    meas_std = batch.get("measurement_std", torch.zeros_like(value))
    bsz, num_events = time.shape
    device = time.device

    op_target = torch.zeros(bsz, num_events, device=device, dtype=torch.long)
    artifact = torch.zeros_like(mask)

    delta_t = torch.zeros_like(time)
    delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    small_gap = delta_t <= delta_t.masked_fill(mask == 0, 0.0).mean(dim=1, keepdim=True).clamp_min(1e-6)
    same_var_repeat = torch.zeros_like(mask, dtype=torch.bool)
    same_var_repeat[:, 1:] = var_id[:, 1:] == var_id[:, :-1]

    # repeat_followup
    repeat = small_gap & same_var_repeat & (mask > 0)
    op_target[repeat] = 3
    artifact[repeat] = 1.0

    # merge_panel approximation: multiple variables share very close timestamps.
    close_time = torch.zeros_like(mask, dtype=torch.bool)
    close_time[:, 1:] = (time[:, 1:] - time[:, :-1]).abs() <= 1e-4
    panel = close_time & (mask > 0)
    op_target[panel] = 4
    artifact[panel] = 1.0

    # pending / low-quality values.
    pending = (meas_std > meas_std.masked_fill(mask == 0, 0.0).mean(dim=1, keepdim=True) + 1e-6) & (mask > 0)
    op_target[pending] = 5
    artifact[pending] = 1.0

    out = dict(batch)
    out["op_target"] = op_target
    out["artifact_mask"] = artifact
    out["panel_mask"] = panel.to(mask.dtype)
    out["value_pending_mask"] = pending.to(mask.dtype)
    return out
```

## 4. 实验切入点

### 4.1 Policy shift 构造

1. **dense-followup shift**
   - 训练环境中高风险样本报警后密集复测；
   - 测试环境改为延迟复测或固定查房采样。

2. **panel merge/split shift**
   - 训练环境中 lab panel 同步下单；
   - 测试环境中同一变量组被异步拆分。

3. **value-pending shift**
   - 借鉴 MILM 的 value-pending 场景：观测事件已出现但数值尚未返回；
   - 检查模型是否把“检查被下单”直接当成类别证据。

4. **detail-resolution shift**
   - 借鉴 CDI-TS 的 context/detail 思想：某些局部片段在训练环境高分辨率采样，在测试环境低分辨率采样；
   - 评估 episode program 是否保持稳定。

### 4.2 对比方法

- 普通 irregular encoder。
- mask dropout / random missing augmentation。
- missingness-aware encoder。
- policy adversarial baseline。
- MILM-style value-redacted sampling classifier。
- CDI-style context-detail gate baseline。
- MedSpaformer / MTM token selection baseline。
- 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN、RKHS Cubature、Measurement-Action Bisimulation、Policy-Word Signature、Thermodynamic、Sklar、Queue Debt、Sinkhorn Detail Canonicalizer。

### 4.3 核心指标

- in-policy accuracy。
- worst-policy accuracy。
- description length ratio：压缩后描述长度相对 raw event 数。
- artifact keep rate：已知策略产物被误保留为 state event 的比例。
- episode stability under policy productions：不同采样 production 下 episode 数量和顺序是否稳定。
- pending-value leakage：只出现采样坐标、值未返回时的错误置信度。
- compression-failure calibration：描述长度异常高时模型是否降低置信或触发补采样建议。

### 4.4 消融实验

- 去掉 `L_description_length`，验证模型是否回到 raw token shortcut。
- 去掉 `L_production_supervision`，验证转导器是否无法识别 panel/repeat/pending artifacts。
- 去掉 `L_artifact_absorption`，检查策略插入 token 是否污染 episode program。
- 将 production labels 替换为随机 mask，验证收益来自策略产物语义而非普通稀疏化。
- 将 `LabelAutomatonReadout` 替换为 pooled episode classifier，验证 episode 顺序建模是否必要。
- 扫描 `max_episodes`，检验短叙事压缩和分类信息保留之间的 trade-off。

## 5. 预期创新性

1. **从采样去偏转向策略产物压缩**：不估计、不删除、不对抗、不平滑采样机制，而是要求 raw event stream 能被短 episode program 加策略 production rules 解释。
2. **从 token selection 转向 MDL transduction**：不是挑选少数 token，而是为未进入分类的 token 给出可解释 production code。
3. **从 context-detail gate 转向 episode description**：吸收 sparse event detection 中“局部细节有价值”的启发，但不让 gate 直接决定分类；细节必须降低 episode program 的描述成本才会被保留。
4. **从 value-pending exploitation 转向 pending artifact absorption**：承认“检查被下单”有信息，但若数值未返回或跨院语义不稳定，它应优先被编码为 production artifact，而不是类别证据。
5. **与采样解耦/反事实干预框架低侵入兼容**：value process 产生 episode，sampling process 产生 production cost，counterfactual intervention 产生 operation supervision，分类器只读压缩 program。
6. **部署解释性强**：可以直接报告“该预测依赖哪些 episode、哪些 raw events 被压缩为策略产物、当前样本是否过度复杂到可能发生采样政策偏移”。

## 6. 一句话投稿卖点

**PA-MDL 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“不同采样政策向同一底层病程插入、删除、重复、合并或延迟返回策略产物，从而拉长原始事件描述”的问题，并通过可微最短描述长度事件转导器把 raw events 压缩为 label-relevant episode program，让分类器依赖跨策略稳定的病程叙事，而不是依赖训练医院或设备制造出的复测、panel、pending-value 与高分辨率细节捷径。**
