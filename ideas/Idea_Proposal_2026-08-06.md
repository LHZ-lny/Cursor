# Title: Do-Clock Poset Distiller：面向采样策略偏移的反事实病程偏序时钟蒸馏器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `**/*summary*.md`、`**/*work*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md` 及其中记录的历史机制黑名单。
- 已读取自动化记忆中的历史 proposal 摘要：`idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`、`idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`、`idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-05.md`。
- 已读取当前工作区内历史 proposal 文件：
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
- 已读取近期论文记录 `paper_daily.md` 与最新日期文件 `paper_daily_2026-08-02.md`，重点纳入：
  - **PULSE**：跨 HiRID / MIMIC-IV / eICU 的 ICU 分类基准，提示采样政策偏移必须放到跨中心部署环境中评估。
  - **TCF**：Pathology-Focused Binning、Dual-Calendar RoPE、Time-Conditioned Foreseeing，提示未来时间条件与病理数值语义可以成为 EHR foundation model 的核心训练对象。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮永久避开以下核心机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、采样 score 零空间、hazard-driven resampling、do-risk variance。
8. 生理流算子与采样算子交换子、policy residual sink。
9. additive evidence market、protocol tax、token-level evidence budget、边际证据审计。
10. 模型空间 posterior quotient、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、ANOVA projection、VQ semantic/acquisition clauses、HSIC redaction。
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
26. counterfactual sampling instruments、first-stage structural equation、control-function residualized structure readout。
27. counterfactual policy jury、Borda / majority rank tribunal、no-dictator / no-structural-bribery loss。
28. Krylov policy subspace、annihilating polynomial、policy-mode annihilation。
29. determinantal / Nystrom state-volume basis、policy-overload logdet sobriety。
30. tropical support routes、soft-min bottleneck、route survival under policy。
31. fixed clinical viva question bank、canonical answer sheet、source-chart-to-answer translator。
32. pathology sequent proof bank、provable / unproved / refuted proof audit、availability sobriety。

本提案选择新的正交切入点：**不把鲁棒性建成证明、校准、纠错、税费、拓扑、gauge、后验或集合契约，而是把采样政策偏移视为“采样日历扭曲了病程进展时钟”。分类器不读原始日历时间，也不读固定问答表；它先从观测值中蒸馏一个可反事实重排的病程偏序时钟，再用这个偏序时钟承载 TCF 式未来时间条件预测与下游分类。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-08-02.md` 中的 PULSE 和 TCF 给当前 AAAI 方向提供了两个互补信号。

第一，**PULSE** 强调真实部署中的 shift 来自跨中心护理流程、变量定义、告警阈值和记录习惯。也就是说，sampling-policy shift 不是简单随机缺失，而是不同医院把同一病程写入数据表时使用了不同的观测日历。若模型把“第 4 小时测了 lactate”或“报警后 30 分钟连续复测”直接当成类别证据，跨中心后就会失效。

第二，**TCF** 的 Time-Conditioned Foreseeing 说明 EHR foundation model 不应只按 token 顺序续写，而要理解“给定未来时间条件，未来临床事件会是什么”。但 TCF 的未来事件分布仍可能混合 patient state 与 care process：未来某项检查出现，可能是病人真的恶化，也可能只是医院 A 的流程提前下单。

历史方案已经覆盖了很多强机制：危险率、后验商、鞅、保形集合、纠错码、knockoff、证明、jury、lattice、solver trace 等。本轮换一个更底层的时间表述：

> 对同一条潜在病程，不同采样政策会改变事件的日历坐标，但不应改变病理进展的相对偏序。先有“病程到达了某个阶段”，再有“医院在某个日历时刻观察到它”。因此，鲁棒分类器应从观测值中恢复一个 **disease-progress poset clock**，让分类和 foreseeing 基于病程偏序，而不是基于采样日历。

**Do-Clock Poset Distiller (DCPD)** 的关键思想是：

1. 用 value-derived pathology atoms 学一个连续病程分数 `s_i`，表示事件在病理进展轴上的位置。
2. 从事件值之间的可预测性中学习 soft precedence relation `P(i -> j)`，表示事件 `i` 是否在病程机制上应先于事件 `j`。
3. 反事实采样干预只改变日历坐标、可见性和 panel batching；模型不要求 logits 一致，而要求 **foreseeing relation** 与 **病程偏序时钟** 不被采样日历重排污染。
4. 分类头读取的是病程偏序摘要，如 order-ideal mass、milestone occupancy 和 future-answerability gain，而不是原始 timestamp、mask density 或 policy signature。

这与当前“采样解耦/反事实干预”框架兼容：

- value process 负责产生 pathology atoms 与病程偏序时钟；
- sampling process 只产生 calendar morph recipes，用于训练时审查“这个偏序是不是只来自采样日历”；
- counterfactual intervention 生成日历重排、panel 合并/拆分、value-pending、routine/alarm 视图；
- classifier 只消费经过偏序时钟蒸馏后的 state-progress signature。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从日历时间编码改为 Disease-Progress Poset Clock

DCPD 可以包裹现有 irregular event encoder，也可以作为 TCF-style EHR foundation model 的前端。

1. **Pathology Atom Lift**
   - 吸收 TCF 的 Pathology-Focused Binning：每个观测事件 `(x_i, t_i, d_i)` 转为变量特异的病理 bin embedding。
   - 时间 `t_i` 只用于构造局部间隔和反事实日历重排，不直接作为分类主证据。

2. **Progress Clock Head**
   - 为每个事件输出一个标量病程时钟 `s_i in [0, 1]`。
   - `s_i` 由观测值、变量、病理 bin 和历史 value context 决定，而不是由日历时间单调决定。
   - 允许同一日历窗口内多个变量落在不同病程阶段，也允许不同医院日历下的相似病理事件映射到相近 `s_i`。

3. **Foreseeability Precedence Matrix**
   - 对事件对 `(i, j)` 计算 soft precedence：

```text
P_ij = sigmoid(MLP([atom_i, atom_j, s_j - s_i]))
```

   - 若 `i` 真正处在 `j` 的病程上游，那么用 `atom_i` 应该能降低对 `atom_j` 的 time-conditioned foreseeing loss。
   - 若 `P_ij` 只因为医院日历中二者常常联测或相邻出现而变大，反事实 calendar morph 会破坏这种关系，foreseeing gain 不会稳定存在。

4. **Order-Ideal Classifier**
   - 对每个类别学习若干 disease milestones，不是固定 viva questions，也不是 temporal sequent proof。
   - 每个 milestone 是病程偏序中的 soft ideal：若上游必要事件已出现，该 milestone 的 occupancy 升高。
   - 分类头读取：

```text
order_ideal_mass = sum_i assignment(i, milestone) * progress_weight(s_i)
precedence_pool  = aggregate(P_ij over value-supported edges)
logits = Classifier([order_ideal_mass, precedence_pool])
```

分类证据来自“病程进展结构是否到达某些阶段”，而不是“采样日历在某些时刻出现了哪些事件”。

### 2.2 改 Loss：从一致性/证明/集合校准转向 Poset-Foreseeing Discipline

总目标：

```text
L = L_cls
  + lambda_fore * L_poset_foreseeing
  + lambda_cal  * L_calendar_shuffle_sobriety
  + lambda_ord  * L_poset_order_laws
  + lambda_clk  * L_clock_noncollapse
```

#### A. 分类损失 `L_cls`

事实观测下，用偏序时钟摘要做分类：

```text
L_cls = CE(logits_poset, y)
```

#### B. Poset Foreseeing Loss `L_poset_foreseeing`

吸收 TCF 的 time-conditioned foreseeing，但把“未来日历时间”改成“未来病程阶段”。模型给定 anchor event `i` 和目标 progress query `q_s`，预测在病程位置 `q_s` 附近应出现的 pathology bin 分布：

```text
p_hat(bin, var | i, q_s) = ForeseeHead(atom_i, q_s)
```

训练目标只在 value-visible target 上计算：

```text
L_poset_foreseeing = CE(p_hat_{i -> q_s}, target_pathology_atom_j)
```

关键区别：TCF 预测“未来日历时间会有什么事件”；DCPD 预测“病程进展到某个相对阶段时会有什么病理状态”。这能减少医院日历流程对 future-event likelihood 的污染。

#### C. Calendar Shuffle Sobriety `L_calendar_shuffle_sobriety`

反事实采样模块生成 calendar morphs：

- `routine_round`: 日历时间规整到查房窗口；
- `alarm_burst_split`: 报警后密集复测被拆分或合并；
- `panel_debatch`: 同步 panel 被拆成异步事件；
- `value_pending`: 采样事件可见但病理 value 不可见。

DCPD 不要求这些视图 logits 一致，也不要求 representation 一致。它只要求“由 value foreseeing 支持的 precedence edge”不应被纯日历重排制造或消灭：

```text
edge_gain_ij = foreseeing_loss_without_i(j) - foreseeing_loss_with_i(j)
L_calendar_shuffle =
  mean relu(P_ij_cf - stopgrad(edge_gain_ij) - eps)^2
```

含义：如果反事实日历视图让两个事件变得相邻或同步，但 `i` 并不能真正帮助预测 `j` 的病理 bin，那么这条 precedence 不能变强。它不同于 proof availability、conformal calibration、jury voting 或 lattice meet/join；它只审查偏序边是否有 value-foreseeing 支撑。

#### D. Poset Order Laws `L_poset_order_laws`

为了让 `P` 真正成为病程偏序，而不是任意 attention map：

```text
L_antisym = mean P_ij * P_ji
L_trans   = mean relu(P_ij + P_jk - 1 - P_ik)^2
L_self    = mean P_ii^2
```

这不是变量图、patch 图或 sheaf gluing；它不解释变量关系，也不做图消息传递。它只是约束病程先后关系具有偏序形状，防止日历相邻性退化为全连接注意力。

#### E. Clock Non-Collapse `L_clock_noncollapse`

防止所有事件的 `s_i` 塌缩到同一位置，或直接复制日历时间：

```text
L_clock_noncollapse =
  relu(min_spread - std(s_i))^2
  + corr(s_i, normalized_calendar_time_i)^2 * stopgrad(calendar_stress)
```

当 calendar stress 高时，惩罚 `s_i` 与原始日历时间过强相关；当 value sequence 本身确有单调进展时，`s_i` 可以保持合理顺序。这样 DCPD 不是删除时间，而是把“日历时间”与“病程时间”区分开。

### 2.3 改 Dataloader：返回 Calendar Morph + Progress Query Bank

新增 `PosetClockCollator`，每个 batch 返回：

1. 原始事件：`event_value`、`event_time`、`event_var_id`、`event_mask`。
2. 病理分箱：`pathology_bin_id` 或 `bin_prob`。
3. `progress_query_bank`：
   - anchor event index `i`；
   - target event index `j`；
   - query progress `q_s`；
   - target pathology bin / variable。
4. `calendar_morph_bank`：
   - routine-round calendar；
   - alarm-burst split/merge；
   - panel debatching；
   - value-pending。
5. `calendar_stress`：
   - 由日历密度、panel 同步率、局部 burst 强度、value-pending 比例估计；
   - 只用于 clock non-collapse 和日历重排审计，不进入分类 head。

### 2.4 与 PULSE / TCF 的结合方式

- **来自 PULSE**：实验不只在单数据集随机 mask，而是在 HiRID / MIMIC-IV / eICU 风格跨中心 policy shift 下检查病程偏序时钟是否稳定。
- **来自 TCF**：保留 Pathology-Focused Binning 与 time-conditioned foreseeing 的价值，但把“calendar-conditioned future event prediction”改成“progress-conditioned pathology foreseeing”。
- **与反事实框架结合**：现有 counterfactual sampling 不再服务 logits consistency、风险方差、proof audit、conformal sleeves 或 jury；它只生成 calendar morphs，用于审查 `s_i` 和 `P_ij` 是否被采样日历伪造。

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


def pairwise_mask(event_mask: torch.Tensor) -> torch.Tensor:
    return event_mask[:, :, None] * event_mask[:, None, :]


class PathologyAtomLift(nn.Module):
    """Lift irregular observations into pathology-aware event atoms."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.0, 2.0, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.atom_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_bins + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        event_mask = batch["event_mask"]

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * event_mask.unsqueeze(-1)

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id)
        atom_x = torch.cat(
            [var_h, bin_prob, value.unsqueeze(-1), torch.log1p(delta_t).unsqueeze(-1)],
            dim=-1,
        )
        atom_h = self.atom_proj(atom_x) * event_mask.unsqueeze(-1)
        return {"atom_h": atom_h, "bin_prob": bin_prob, "event_mask": event_mask}


class ProgressPosetClock(nn.Module):
    """Learn disease-progress positions and value-supported precedence relations."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.clock_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, atom_h: torch.Tensor, event_mask: torch.Tensor) -> dict:
        ctx, _ = self.context(atom_h)
        progress = torch.sigmoid(self.clock_head(ctx).squeeze(-1)) * event_mask

        hi = ctx[:, :, None, :].expand(-1, -1, ctx.size(1), -1)
        hj = ctx[:, None, :, :].expand(-1, ctx.size(1), -1, -1)
        ds = progress[:, None, :] - progress[:, :, None]
        edge_x = torch.cat([hi, hj, hi * hj, (hj - hi).abs(), ds.unsqueeze(-1)], dim=-1)
        precedence = torch.sigmoid(self.edge_mlp(edge_x).squeeze(-1))

        mask2 = pairwise_mask(event_mask)
        eye = torch.eye(atom_h.size(1), device=atom_h.device, dtype=torch.bool)
        precedence = precedence.masked_fill(eye[None], 0.0) * mask2
        return {"context": ctx, "progress": progress, "precedence": precedence}


class ProgressForeseeHead(nn.Module):
    """Predict pathology bins at queried disease-progress positions."""

    def __init__(self, hidden_dim: int, num_vars: int, num_bins: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.query_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.var_head = nn.Linear(hidden_dim, num_vars)
        self.bin_head = nn.Linear(hidden_dim, num_bins)

    def forward(self, anchor_h: torch.Tensor, target_h: torch.Tensor, query_progress: torch.Tensor) -> dict:
        q = self.query_proj(torch.cat([anchor_h, target_h, query_progress.unsqueeze(-1)], dim=-1))
        return {"var_logits": self.var_head(q), "bin_logits": self.bin_head(q)}


class OrderIdealReadout(nn.Module):
    """Classify from disease-progress order-ideal summaries."""

    def __init__(self, hidden_dim: int, num_classes: int, num_milestones: int = 8):
        super().__init__()
        self.milestones = nn.Parameter(torch.linspace(0.05, 0.95, num_milestones))
        self.assign = nn.Linear(hidden_dim, num_milestones)
        self.classifier = nn.Sequential(
            nn.Linear(2 * num_milestones + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, context: torch.Tensor, progress: torch.Tensor, precedence: torch.Tensor, event_mask: torch.Tensor) -> dict:
        assign = torch.softmax(self.assign(context), dim=-1) * event_mask.unsqueeze(-1)
        milestone_gate = torch.sigmoid(12.0 * (progress.unsqueeze(-1) - self.milestones.view(1, 1, -1)))
        ideal_mass = (assign * milestone_gate).sum(dim=1)

        incoming = precedence.sum(dim=1) / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        order_strength = (assign * incoming.unsqueeze(-1)).sum(dim=1)
        pooled_context = masked_mean(context, event_mask, dim=1)

        logits = self.classifier(torch.cat([ideal_mass, order_strength, pooled_context], dim=-1))
        return {"logits": logits, "ideal_mass": ideal_mass, "order_strength": order_strength}


class DoClockPosetDistiller(nn.Module):
    """Sampling-policy robust classifier based on a value-derived disease-progress poset clock."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        hidden_dim: int,
        num_classes: int,
        num_milestones: int = 8,
    ):
        super().__init__()
        self.atom = PathologyAtomLift(num_vars, num_bins, hidden_dim)
        self.clock = ProgressPosetClock(hidden_dim)
        self.foresee = ProgressForeseeHead(2 * hidden_dim, num_vars, num_bins)
        self.readout = OrderIdealReadout(2 * hidden_dim, num_classes, num_milestones)
        self.num_vars = num_vars
        self.num_bins = num_bins

    def forward(self, batch: dict) -> dict:
        atom = self.atom(batch)
        poset = self.clock(atom["atom_h"], atom["event_mask"])
        readout = self.readout(
            context=poset["context"],
            progress=poset["progress"],
            precedence=poset["precedence"],
            event_mask=atom["event_mask"],
        )
        return {**atom, **poset, **readout}

    def poset_order_law_loss(self, precedence: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        mask2 = pairwise_mask(event_mask)
        antisym = (precedence * precedence.transpose(1, 2) * mask2).sum() / mask2.sum().clamp_min(1.0)

        # Soft transitivity over triples: if i precedes j and j precedes k, i should precede k.
        pij = precedence[:, :, :, None]
        pjk = precedence[:, None, :, :]
        pik = precedence[:, :, None, :]
        trans_raw = F.relu(pij + pjk - 1.0 - pik).pow(2)
        mask3 = event_mask[:, :, None, None] * event_mask[:, None, :, None] * event_mask[:, None, None, :]
        trans = (trans_raw * mask3).sum() / mask3.sum().clamp_min(1.0)
        return antisym + trans

    def progress_foreseeing_loss(self, batch: dict, out: dict) -> torch.Tensor:
        anchors = batch["progress_query_anchor"]
        targets = batch["progress_query_target"]
        q_progress = batch["progress_query_value"]
        target_var = batch["progress_query_var"].clamp(0, self.num_vars - 1)
        target_bin = batch["progress_query_bin"].clamp(0, self.num_bins - 1)

        bidx = torch.arange(out["context"].size(0), device=out["context"].device)[:, None]
        anchor_h = out["context"][bidx, anchors]
        target_h = out["context"][bidx, targets]
        pred = self.foresee(anchor_h, target_h, q_progress)

        var_loss = F.cross_entropy(pred["var_logits"].flatten(0, 1), target_var.flatten())
        bin_loss = F.cross_entropy(pred["bin_logits"].flatten(0, 1), target_bin.flatten())
        return var_loss + bin_loss

    def calendar_shuffle_sobriety_loss(self, batch: dict, factual: dict, eps: float = 0.05) -> torch.Tensor:
        if "calendar_morph_bank" not in batch:
            return torch.zeros((), device=factual["logits"].device)

        # Edge support proxy: target progress should be downstream of anchor progress.
        anchors = batch["progress_query_anchor"]
        targets = batch["progress_query_target"]
        bidx = torch.arange(factual["context"].size(0), device=factual["context"].device)[:, None]
        factual_edge = factual["precedence"][bidx, anchors, targets].detach()

        losses = []
        for morph_batch in batch["calendar_morph_bank"]:
            cf = self.forward(morph_batch)
            cf_edge = cf["precedence"][bidx, anchors, targets]
            losses.append(F.relu(cf_edge - factual_edge - eps).pow(2).mean())
        return torch.stack(losses).mean()

    def clock_noncollapse_loss(self, batch: dict, out: dict, min_spread: float = 0.08) -> torch.Tensor:
        mask = out["event_mask"]
        progress = out["progress"]
        mean = masked_mean(progress, mask, dim=1).unsqueeze(-1)
        spread = masked_mean((progress - mean).pow(2), mask, dim=1).sqrt()
        spread_loss = F.relu(min_spread - spread).pow(2).mean()

        time = batch["event_time"]
        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        p_center = progress - masked_mean(progress, mask, dim=1).unsqueeze(-1)
        t_center = time_norm - masked_mean(time_norm, mask, dim=1).unsqueeze(-1)
        corr = masked_mean(p_center * t_center, mask, dim=1)
        corr = corr / torch.sqrt(
            masked_mean(p_center.pow(2), mask, dim=1) * masked_mean(t_center.pow(2), mask, dim=1) + 1e-6
        )
        calendar_stress = batch.get("calendar_stress", torch.ones_like(corr)).detach()
        calendar_copy_loss = (calendar_stress * corr.pow(2)).mean()
        return spread_loss + 0.2 * calendar_copy_loss

    def training_loss(
        self,
        batch: dict,
        lambda_fore: float = 0.25,
        lambda_cal: float = 0.20,
        lambda_ord: float = 0.10,
        lambda_clk: float = 0.05,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)
        fore_loss = self.progress_foreseeing_loss(batch, out)
        shuffle_loss = self.calendar_shuffle_sobriety_loss(batch, out)
        order_loss = self.poset_order_law_loss(out["precedence"], out["event_mask"])
        clock_loss = self.clock_noncollapse_loss(batch, out)

        total = (
            cls_loss
            + lambda_fore * fore_loss
            + lambda_cal * shuffle_loss
            + lambda_ord * order_loss
            + lambda_clk * clock_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "poset_foreseeing_loss": fore_loss.detach(),
            "calendar_shuffle_sobriety_loss": shuffle_loss.detach(),
            "poset_order_law_loss": order_loss.detach(),
            "clock_noncollapse_loss": clock_loss.detach(),
        }


@torch.no_grad()
def build_poset_clock_calendar_morphs(batch: dict) -> list[dict]:
    """Create calendar morphs for poset-clock auditing, not for logits consistency."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = value.shape
    device = value.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        out.pop("calendar_morph_bank", None)
        return out

    morphs = []

    # 1. Routine-round calendar: calendar shifts, values remain factual.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    morphs.append(clone_with(value * mask, rounded_time, var_id, mask))

    # 2. Alarm burst split/merge: thin alternating events in dense late regions.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    burst_mask = torch.where(late > 0, late * alternating, mask) * mask
    morphs.append(clone_with(value * burst_mask, time, var_id, burst_mask))

    # 3. Panel debatching: jitter near-synchronous cross-variable observations.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    close = (gap <= (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    jitter = 0.05 * horizon * close * changed_var
    morphs.append(clone_with(value * mask, time + jitter, var_id, mask))

    # 4. Value-pending: calendar visible but pathology value unavailable.
    pending_value = torch.zeros_like(value)
    pending_mask = mask
    morphs.append(clone_with(pending_value, time, var_id, pending_mask))

    return morphs
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cross-center calendar shift`：借鉴 PULSE，在 HiRID / MIMIC-IV / eICU 风格采样频率、变量 schema 和 routine/alarm 流程间迁移。
   - `foreseeing-calendar shift`：训练中未来事件按中心 A 的检查流程出现，测试中改成中心 B 的流程，检查 DCPD 是否仍保留病程偏序。
   - `panel batching shift`：训练中心同步 panel，测试中心异步 debatching。
   - `value-pending shift`：采样事件已出现但值尚未返回，验证偏序边是否必须由 pathology value 支撑。

2. **对比方法**
   - 普通 TCF / EHR foundation model。
   - 普通 irregular Transformer / mTAND / STAR-Set / VP-GNN。
   - PULSE 中强传统基线与 LLM prompt / hybrid baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、D-IVSP、DJRT、KPMA、DVNB、DSPP 等。

3. **核心指标**
   - in-policy accuracy。
   - cross-center worst-policy accuracy。
   - progress-clock calendar correlation：病程时钟与原始日历时间的相关性，越低越不依赖采样日历。
   - poset edge foreseeing gain：高权重偏序边是否真正降低病理 foreseeing loss。
   - calendar-fabricated edge rate：反事实日历重排后新增但无 value-foreseeing 支撑的边比例。
   - progress-conditioned foreseeing AUPRC：在病程进展查询下预测未来 pathology bins 的质量。

4. **消融实验**
   - 去掉 `L_poset_foreseeing`，检查偏序是否退化为普通 attention。
   - 去掉 `L_calendar_shuffle_sobriety`，检查 panel / routine 日历是否制造伪偏序边。
   - 去掉 `L_poset_order_laws`，检查 precedence matrix 是否出现循环和互指。
   - 去掉 `L_clock_noncollapse`，检查病程时钟是否复制日历时间或全部塌缩。
   - 将 progress query 换回 TCF 式 calendar-time query，验证病程偏序时钟相比日历 foreseeing 的跨中心鲁棒性。

## 5. 预期创新性

1. **从采样日历转向病程偏序时钟**：首次把 sampling-policy shift 表述为“观测日历扭曲了真实病程进展坐标”的问题，分类只读取 value-derived progress poset。
2. **从 TCF 日历 foreseeing 转向 progress-conditioned foreseeing**：保留 TCF 对未来时间条件和病理分箱的优势，但把未来查询从医院日历时间迁移到病程进展阶段。
3. **从反事实一致性转向偏序边真伪审计**：counterfactual sampling 只审查某条 precedence edge 是否由 value-foreseeing 支撑，而不是约束 logits、representation、proof、conformal set 或 risk。
4. **从固定问答/证明规则转向可学习病程顺序结构**：DCPD 不预设 viva question，也不学习 temporal sequent proof；它学习事件之间的 soft disease-progress precedence 与 order-ideal occupancy。
5. **跨中心部署解释性清晰**：如果某个预测依赖采样日历，系统能报告高 calendar-fabricated edge rate；如果预测依赖真实病程进展，则偏序边应在不同中心日历下保持 value-foreseeing 支撑。

## 6. 一句话投稿卖点

**DCPD 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样日历对病程进展时钟的扭曲”，通过 Pathology Atom Lift、Disease-Progress Poset Clock、progress-conditioned foreseeing 与 calendar-morph edge sobriety，让模型从观测值中恢复跨中心稳定的病程偏序，再基于该偏序完成分类，从而避免把医院 routine/alarm 日历、panel batching、value-pending 或未来检查流程误当作可迁移类别证据。**
