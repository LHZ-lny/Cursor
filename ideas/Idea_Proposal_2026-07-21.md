# Title: Context-Anchored Sinkhorn Detail Canonicalizer：面向采样策略偏移的上下文锚定细节规范化器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已读取自动化记忆 `MEMORIES.md`：其中记录了历史多轮任务同样未发现 `my_work_summary.md`，并给出 2026-06-12 至 2026-07-20 的历史机制黑名单。
- 已读取近期论文记录：`paper_daily.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-13.md`。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：2026-06-17、2026-06-20、2026-06-24、2026-06-27、2026-07-15、2026-07-16、2026-07-17、2026-07-18、2026-07-19、2026-07-20。

### 历史核心机制黑名单

为避免思维重合，本提案明确避开以下已提出机制作为主创新：

1. 连续时间 hazard scorer、采样 score 零空间、hazard-driven counterfactual resampling、do-risk variance。
2. 生理流-采样算子交换子、value graph / policy graph 分离、policy residual sink。
3. additive evidence market、protocol tax、边际证据审计。
4. posterior quotient、采样似然因子相除、interventional policy marginalization。
5. reconstruction error cartography、VQ semantic clauses、HSIC redaction。
6. policy-simplex randomized smoothing、certified policy radius。
7. sampling measure density ratio、doubly robust correction、influence-bound regularization。
8. optional-stopping martingale query、standardized innovation、stopping recipe。
9. censored topology capsules、persistence interval、fragmentation sobriety。
10. policy gauge frame、horizontal transport、vertical blindness。
11. policy-only negative film、shadow eraser/stencil。
12. syndrome code、parity-check、packet repair。
13. conditional knockoff calendar、soft knockoff-FDR firewall。
14. observability witness、low-observability gate、probe ordering。
15. evidential shield、Dirichlet/subjective uncertainty、policy-induced vacuity。
16. policy lattice、meet/join information order、submodular margin。
17. solver trace front-door、NFE/roughness trace standardization。
18. RKHS cubature weights、measurement-action bisimulation、policy-word signature、thermodynamic free energy、copula rank stripping。
19. triage queue debt、service discipline bank、debt-neutral detail gate、queue conservation/service replay。
20. 直接复用普通 token sparsification、ordinary query tokens、ordinary context-detail gate、ordinary LLM alignment、ordinary graph decay 或 frequency branch 作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做对抗，不要求多采样视图 logits 一致，不做 token 选择或 gate，不把策略残差作为表征分支；而是把“采样政策改变局部细节曝光量”建模为非平衡最优传输中的质量膨胀/缺损问题。粗粒度上下文先生成固定数量的语义锚点，局部细节事件通过 Unbalanced Sinkhorn 被规范化到这些锚点；由采样政策制造的冗余、重复、panel 拆分或 value-pending 细节不会形成额外分类证据，而会被质量松弛项吸收。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 `paper_daily_2026-07-19.md` 中的 **Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction** 给出一个重要信号：稀疏医疗时序不能只靠全局表示，模型需要知道何时让局部细节参与判别。它的 global context explorer 与 local detail inspector 说明，真正有诊断价值的局部片段往往极少，且需要由上下文决定是否精查。

但直接把这个机制搬到 sampling-policy shift 会有风险：

- 某医院报警后密集复测，局部细节数量突然膨胀，普通 detail inspector 会看到更多 token；
- 某实验室 panel 从同步下单变成异步拆单，同一语义细节被拆成多个事件；
- 某些 value-pending 场景中，采样坐标先出现但数值尚未返回，模型可能把“已下单”当作类别证据；
- 如果用 gate 决定是否看细节，gate 本身可能学习到医院流程触发规则，这一点已被 2026-07-20 的 queue debt / detail gate 方向占用，不能重复。

**Context-Anchored Sinkhorn Detail Canonicalizer (CAS-DC)** 的核心动机是：

> 让模型仍然能利用上下文和局部细节，但局部细节必须先被运输到由上下文定义的固定语义锚点上；采样政策只能改变“有多少观测质量可被运输”，不能因为重复采样、密集复测或 panel 拆分而凭空增加分类证据。

换言之，CAS-DC 把 sampling-policy shift 表述为 **detail exposure mass shift**：

```text
same latent state -> different sampling policy -> different number/layout of detail events
```

普通 attention / gating / sparsification 会把更多 detail events 转成更强 logit；CAS-DC 则用非平衡最优传输将任意数量的细节事件规范化到固定容量的语义锚点：

```text
coarse context -> K semantic anchors
local details  -> unbalanced Sinkhorn transport -> K canonical detail slots
classifier     -> context + canonical slots
```

这样可以吸收最新 context-detail 机制的优点，却避免以下历史路线：

- 不做 context-detail gate；
- 不做 queue/service/debt 建模；
- 不做 token sparsification 或 pivotal token selection；
- 不做跨视图一致性；
- 不估计采样 hazard、density ratio 或 posterior quotient；
- 不把 policy code、mask pattern 或 sampling-only branch 输入分类器。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从 gated detail interaction 改为 Context-Anchored UOT Canonicalization

CAS-DC 包含三层。

#### A. Coarse Context Anchor Generator

输入完整不规则事件流，但只通过低分辨率、低曝光敏感的方式聚合，例如变量级稳健均值、分位数、粗时间窗摘要或轻量 GRU。它输出 `K` 个语义锚点：

```text
A = {a_1, ..., a_K},  a_k in R^H
```

这些锚点表示“在当前粗上下文下，分类可能需要检查的 K 类语义细节位置”，但锚点本身不由采样密度、panel 重复数或 policy id 决定。

#### B. Local Detail Inspector

每个观测事件 `(value, time, variable, delta_t, measurement_std)` 被编码成 detail embedding：

```text
d_i = DetailEncoder(x_i, t_i, var_i, delta_t_i, quality_i)
```

这里吸收稀疏事件检测中 local detail inspector 的思想，但它不输出 gate，也不直接进入分类器。

#### C. Unbalanced Sinkhorn Detail Transport

计算 detail 到 anchor 的运输代价：

```text
C_{i,k} = || W_d d_i - W_a a_k ||_2^2 + time_cost(i, k) + quality_cost(i)
```

然后求解非平衡最优传输：

```text
T* = argmin_T <T, C>
     + eps * KL(T || r c^T)
     + tau_row * KL(T 1 || r)
     + tau_col * KL(T^T 1 || c)
```

其中：

- `r` 是每个 detail event 的可用质量；
- `c` 是每个语义锚点的上下文容量；
- `tau_row/tau_col` 允许事件质量被部分丢弃或锚点容量未完全填满；
- 采样政策造成的重复事件、过密复测、panel split 不会线性增加分类证据，只会增加可被 Sinkhorn 松弛吸收的曝光质量。

规范化后的 slot：

```text
s_k = sum_i T_{i,k} * d_i / (sum_i T_{i,k} + eps)
```

分类器只接收：

```text
Classifier([context_state, s_1, ..., s_K, slot_mass_1, ..., slot_mass_K])
```

注意：`slot_mass` 是规范化后的可恢复细节容量，不是原始 mask ratio 或采样政策特征。

### 2.2 改 Loss：从 gate supervision / consistency 转向 Exposure-Mass Canonicalization

总目标：

```text
L = L_cls
  + lambda_absorb * L_policy_exposure_absorption
  + lambda_cap    * L_context_capacity
  + lambda_bal    * L_slot_noncollapse
  + lambda_type   * L_transformed_detail_type
```

#### A. Classification Loss `L_cls`

只使用 Sinkhorn 规范化后的 slots 分类：

```text
L_cls = CE(Classifier(context, canonical_slots), y)
```

它不是多视图一致性，也不是 smoothed prediction；每条样本只用自身的 canonical detail slots 做分类。

#### B. Policy Exposure Absorption `L_policy_exposure_absorption`

反事实采样模块生成一些 **曝光编辑**，这些编辑只改变细节事件的可见形式，不改变底层观测值语义：

- `burst_duplicate`：把报警后复测事件复制或局部加密；
- `panel_split`：把同步 panel 拆成多个近邻异步事件；
- `pending_stub`：保留时间/变量坐标但遮蔽 value，模拟 value-pending；
- `routine_thin`：按固定查房窗口稀疏化低价值重复点。

Collator 知道哪些事件是 policy-inserted / value-pending / duplicate detail。CAS-DC 不要求编辑前后 logits 一致，而是要求这些 **政策曝光事件** 的语义运输质量低：

```text
L_policy_exposure_absorption =
  mean_{i in policy_inserted} sum_k T_{i,k}
```

直觉：如果一个事件只是采样流程制造出的重复曝光，它不应占用有限语义锚点容量。

#### C. Context Capacity Loss `L_context_capacity`

锚点容量由粗上下文预测：

```text
c = softplus(CapacityHead(context_state))
```

若某条序列因策略复测产生 3 倍 detail events，`sum_k c_k` 不应随 event count 线性增大。训练时用 stop-gradient 的粗上下文统计监督容量，而不是用 mask ratio：

```text
target_capacity = f(value_quantiles, coarse_value_variation, coarse_window_coverage)
L_context_capacity = SmoothL1(sum_k transported_mass_k, target_capacity)
```

这与历史信息格/证据税不同：它不约束 margin 的集合形状，不给 token 定价；它只防止采样曝光量变成分类容量。

#### D. Slot Non-Collapse `L_slot_noncollapse`

防止所有细节都被运到一个 slot，加入轻量均衡项：

```text
L_slot_noncollapse =
  entropy(mean_batch normalized_slot_mass) * (-1)
  + mean_k relu(min_mass - slot_mass_k)^2
```

该项不是 token sparsification；它反而鼓励多个语义锚点都可用，避免单一高频变量支配全部细节。

#### E. Transformed Detail Type `L_transformed_detail_type`

借鉴最新事件检测论文的 transformed labels，但不做 gate。Collator 为细节事件提供轻量类型标签：

```text
0 = factual_value_detail
1 = policy_duplicate
2 = panel_split_child
3 = value_pending_stub
4 = routine_thinned_context
```

CAS-DC 只用这些标签训练 **transport-side auxiliary head**，帮助 UOT 识别哪些 detail mass 不该进入语义 slots：

```text
L_transformed_detail_type = CE(DetailTypeHead(d_i), detail_type_i)
```

这个 head 不参与最终分类，且不输出环境标签或采样政策类别；它只是把 transformed label supervision 用于运输规范化。

### 2.3 改 Dataloader：返回 Exposure Edit Bank，而不是一致性视图

新增 `ExposureEditCollator`，每个 batch 返回：

1. 原始事件：`event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_std`。
2. `event_detail_type`：事实事件类型，默认 `factual_value_detail`。
3. `edited_event_*`：可选曝光编辑后的事件流，用于训练吸收损失和 detail type head。
4. `policy_inserted_mask`：哪些事件是采样政策曝光制造的 duplicate / split / pending stub。
5. `coarse_capacity_target`：由 value quantile、粗时间窗覆盖和局部变化幅度计算的锚点容量目标。

关键区别：

- 不生成对比正样本；
- 不要求事实和编辑视图 logits 一致；
- 不返回 policy id；
- 不估计采样概率；
- 不进行 token 预算、证据税、knockoff、lattice meet/join 或 queue/service 模拟；
- 只告诉 Sinkhorn 规范化器：哪些新增细节质量属于采样曝光，不应占用语义锚点。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `CoarseContextEncoder + LocalDetailEncoder`。
- 现有 sampling branch 改为 `ExposureEditGenerator`：只生成 duplicate / split / pending / thinning 这些可解释曝光编辑和 transformed labels。
- 现有 counterfactual intervention 不再服务于 logits 一致、风险方差、平滑认证或 queue debt，而是服务于 UOT 运输质量监督。
- 推理阶段无需生成反事实视图；只运行 context anchors、detail embeddings、Unbalanced Sinkhorn 和分类头。
- 部署诊断输出：
  - 每个 anchor 的 transported mass；
  - policy exposure absorption rate；
  - value-pending / duplicate 细节是否占用语义 slots；
  - 预测是否由少数过密采样事件支配。

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


class CoarseContextEncoder(nn.Module):
    """Low-exposure-sensitivity context encoder that produces semantic anchors."""

    def __init__(self, num_vars: int, hidden_dim: int, num_anchors: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.anchor_head = nn.Linear(hidden_dim, num_anchors * hidden_dim)
        self.capacity_head = nn.Sequential(
            nn.Linear(hidden_dim, num_anchors),
            nn.Softplus(),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
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
        event_h = self.event_proj(event_x) * mask.unsqueeze(-1)
        seq_h, _ = self.context_rnn(event_h)
        context = masked_mean(seq_h, mask, dim=1)
        anchors = self.anchor_head(context).view(value.size(0), self.num_anchors, -1)
        capacity = self.capacity_head(context) + 1e-3
        return {"context": context, "anchors": anchors, "capacity": capacity}


class LocalDetailEncoder(nn.Module):
    """Encode local details; final classifier never sees raw details directly."""

    def __init__(self, num_vars: int, hidden_dim: int, num_detail_types: int = 5):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.detail_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.type_head = nn.Linear(hidden_dim, num_detail_types)
        self.quality_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        measurement_std = batch.get("measurement_std", torch.zeros_like(value))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_h = self.var_embed(var_id.clamp_min(0))
        detail_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(measurement_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        detail = self.detail_proj(detail_x) * mask.unsqueeze(-1)
        quality = self.quality_head(detail).squeeze(-1) * mask
        return {
            "detail": detail,
            "quality": quality,
            "detail_type_logits": self.type_head(detail),
        }


def unbalanced_sinkhorn(
    cost: torch.Tensor,
    row_mass: torch.Tensor,
    col_mass: torch.Tensor,
    epsilon: float = 0.08,
    tau_row: float = 0.55,
    tau_col: float = 0.55,
    iters: int = 24,
) -> torch.Tensor:
    """A compact unbalanced Sinkhorn solver for detail-to-anchor transport."""

    # cost: [B, N, K], row_mass: [B, N], col_mass: [B, K]
    kernel = torch.exp(-cost / epsilon).clamp_min(1e-8)
    u = torch.ones_like(row_mass)
    v = torch.ones_like(col_mass)
    row_power = tau_row / (tau_row + epsilon)
    col_power = tau_col / (tau_col + epsilon)

    for _ in range(iters):
        kv = torch.einsum("bnk,bk->bn", kernel, v).clamp_min(1e-8)
        u = (row_mass / kv).clamp_min(1e-8).pow(row_power)
        ktu = torch.einsum("bnk,bn->bk", kernel, u).clamp_min(1e-8)
        v = (col_mass / ktu).clamp_min(1e-8).pow(col_power)

    return u.unsqueeze(-1) * kernel * v.unsqueeze(1)


class SinkhornDetailCanonicalizer(nn.Module):
    """Canonicalize irregular local details into fixed context-defined slots."""

    def __init__(self, hidden_dim: int, num_classes: int, num_anchors: int):
        super().__init__()
        self.detail_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.anchor_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.slot_mixer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (num_anchors + 1), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, context: torch.Tensor, anchors: torch.Tensor, capacity: torch.Tensor, detail: torch.Tensor, quality: torch.Tensor, mask: torch.Tensor) -> dict:
        detail_z = F.normalize(self.detail_proj(detail), dim=-1)
        anchor_z = F.normalize(self.anchor_proj(anchors), dim=-1)
        cost = (detail_z[:, :, None, :] - anchor_z[:, None, :, :]).pow(2).sum(dim=-1)

        row_mass = (quality * mask).clamp_min(1e-6)
        col_mass = capacity / capacity.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        col_mass = col_mass * row_mass.sum(dim=-1, keepdim=True).clamp_min(1.0)

        transport = unbalanced_sinkhorn(cost, row_mass=row_mass, col_mass=col_mass)
        slot_mass = transport.sum(dim=1)
        slots = torch.einsum("bnk,bnh->bkh", transport, detail)
        slots = slots / slot_mass.unsqueeze(-1).clamp_min(1e-6)
        slots = self.slot_mixer(torch.cat([slots, torch.log1p(slot_mass).unsqueeze(-1)], dim=-1))

        flat = torch.cat([context, slots.flatten(start_dim=1)], dim=-1)
        logits = self.classifier(flat)
        return {"logits": logits, "transport": transport, "slot_mass": slot_mass, "slots": slots}


class ContextAnchoredSinkhornDetailCanonicalizer(nn.Module):
    """Sampling-policy robust classifier via context-anchored unbalanced transport."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int, num_anchors: int = 6, num_detail_types: int = 5):
        super().__init__()
        self.context = CoarseContextEncoder(num_vars, hidden_dim, num_anchors)
        self.detail = LocalDetailEncoder(num_vars, hidden_dim, num_detail_types)
        self.canonicalizer = SinkhornDetailCanonicalizer(hidden_dim, num_classes, num_anchors)
        self.num_detail_types = num_detail_types

    def forward(self, batch: dict) -> dict:
        ctx = self.context(batch)
        det = self.detail(batch)
        can = self.canonicalizer(
            context=ctx["context"],
            anchors=ctx["anchors"],
            capacity=ctx["capacity"],
            detail=det["detail"],
            quality=det["quality"],
            mask=batch["event_mask"],
        )
        return {**ctx, **det, **can}

    def training_loss(
        self,
        batch: dict,
        lambda_absorb: float = 0.30,
        lambda_cap: float = 0.15,
        lambda_bal: float = 0.04,
        lambda_type: float = 0.10,
        min_slot_mass: float = 0.02,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]

        cls_loss = F.cross_entropy(out["logits"], labels)

        policy_inserted = batch.get("policy_inserted_mask", torch.zeros_like(batch["event_mask"]))
        inserted_mass = (out["transport"].sum(dim=-1) * policy_inserted).sum(dim=1)
        denom = policy_inserted.sum(dim=1).clamp_min(1.0)
        absorb_loss = (inserted_mass / denom).mean()

        target_capacity = batch.get(
            "coarse_capacity_target",
            out["slot_mass"].sum(dim=-1).detach(),
        )
        cap_loss = F.smooth_l1_loss(out["slot_mass"].sum(dim=-1), target_capacity)

        normalized_mass = out["slot_mass"] / out["slot_mass"].sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mean_mass = normalized_mass.mean(dim=0).clamp_min(1e-8)
        balance_loss = -(mean_mass * mean_mass.log()).sum() / torch.log(torch.tensor(float(mean_mass.numel()), device=mean_mass.device))
        balance_loss = -balance_loss + F.relu(min_slot_mass - normalized_mass).pow(2).mean()

        detail_type = batch.get("event_detail_type")
        if detail_type is None:
            type_loss = torch.zeros((), device=out["logits"].device)
        else:
            raw_type = F.cross_entropy(
                out["detail_type_logits"].reshape(-1, self.num_detail_types),
                detail_type.reshape(-1).clamp(0, self.num_detail_types - 1),
                reduction="none",
            ).view_as(batch["event_mask"])
            type_loss = (raw_type * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

        total = cls_loss + lambda_absorb * absorb_loss + lambda_cap * cap_loss + lambda_bal * balance_loss + lambda_type * type_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "policy_exposure_absorption_loss": absorb_loss.detach(),
            "context_capacity_loss": cap_loss.detach(),
            "slot_noncollapse_loss": balance_loss.detach(),
            "transformed_detail_type_loss": type_loss.detach(),
            "mean_slot_mass": out["slot_mass"].mean().detach(),
        }


@torch.no_grad()
def build_exposure_edit_batch(batch: dict) -> dict:
    """Sketch exposure edits for transport-side supervision, not consistency training."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = value.shape
    device = value.device

    event_detail_type = torch.zeros_like(var_id)
    policy_inserted = torch.zeros_like(mask)

    # Mark likely panel-split children: very close neighboring events with different variables.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).abs()
    close = (gap <= gap[mask > 0].median().clamp_min(1e-6)).to(mask.dtype) if (mask > 0).any() else torch.zeros_like(mask)
    diff_var = torch.zeros_like(mask)
    diff_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_child = close * diff_var * mask
    event_detail_type = torch.where(panel_child > 0, torch.full_like(event_detail_type, 2), event_detail_type)

    # Mark alternating dense repeats as policy duplicates.
    alternating = ((torch.arange(num_events, device=device)[None, :] % 2) == 0).to(mask.dtype)
    same_var_prev = torch.zeros_like(mask)
    same_var_prev[:, 1:] = (var_id[:, 1:] == var_id[:, :-1]).to(mask.dtype)
    duplicate = alternating * same_var_prev * mask
    event_detail_type = torch.where(duplicate > 0, torch.full_like(event_detail_type, 1), event_detail_type)
    policy_inserted = torch.maximum(policy_inserted, duplicate)

    # Value-pending stubs: high measurement uncertainty if provided.
    if "measurement_std" in batch:
        std = batch["measurement_std"]
        pending = (std > std[mask > 0].median().clamp_min(1e-6)).to(mask.dtype) * mask if (mask > 0).any() else torch.zeros_like(mask)
        event_detail_type = torch.where(pending > 0, torch.full_like(event_detail_type, 3), event_detail_type)
        policy_inserted = torch.maximum(policy_inserted, pending)

    # Capacity target comes from value variation, not event count.
    centered = value - masked_mean(value, mask, dim=1).unsqueeze(-1)
    coarse_variation = masked_mean(centered.abs(), mask, dim=1)
    coarse_capacity_target = (1.0 + coarse_variation).detach()

    out = dict(batch)
    out["event_detail_type"] = event_detail_type
    out["policy_inserted_mask"] = policy_inserted
    out["coarse_capacity_target"] = coarse_capacity_target
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `burst exposure shift`：训练环境报警后密集复测，测试环境只保留单次复测。
   - `panel split shift`：训练环境同步 panel，测试环境拆成异步事件，或反向。
   - `value-pending shift`：某些化验只有下单时间和变量名，数值延迟返回。
   - `detail exposure inflation`：同一底层轨迹下人为复制局部细节事件，测试模型是否把重复曝光当成更强证据。

2. **对比方法**
   - 普通 irregular Transformer / GRU / mTAND encoder。
   - Adaptive context-detail gate baseline。
   - token sparsification / pivotal token baseline。
   - mask dropout / random missing augmentation。
   - missingness-aware classifier。
   - policy adversarial baseline。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN、Triage-Queue Debt Neutralizer。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - exposure inflation sensitivity：重复细节事件后 logit margin 增幅。
   - policy exposure absorption rate：policy-inserted events 的平均 transported mass。
   - slot mass stability：采样密度变化下语义 slot mass 的变异系数。
   - value-pending leakage：仅靠 pending stubs 是否显著改变预测。

4. **消融实验**
   - 去掉 Unbalanced Sinkhorn，改成普通 attention pooling，检查是否重新依赖细节数量。
   - 去掉 `L_policy_exposure_absorption`，检查 duplicate / panel split 是否占用语义 slot。
   - 去掉 context capacity，检查 slot mass 是否随采样密度线性增加。
   - 将 transformed detail labels 替换为随机标签，验证收益来自曝光类型监督。
   - 扫描 anchor 数量 `K` 与 Sinkhorn 松弛系数，分析容量不足和过度容量的鲁棒性差异。

## 5. 预期创新性

1. **从 gate 触发转向非平衡运输规范化**：吸收最新 context-detail 论文对全局上下文和局部细节交互的启发，但不使用 gate，也不建模 queue/service/debt。
2. **从 token 数量鲁棒转向 detail mass 鲁棒**：采样策略造成的重复、拆分、pending 细节不会线性增加分类证据，而是在固定语义锚点容量下被 Sinkhorn 规范化。
3. **从反事实一致性转向运输侧监督**：counterfactual intervention 只标注哪些细节属于政策曝光，不要求多个采样视图 logits 或 representation 一致。
4. **从 token sparsification 转向 soft canonical slots**：模型不选择少数 pivotal tokens，而是把全部可用细节按成本和容量软运输到语义 slots。
5. **部署诊断直观**：若某个预测依赖医院复测 burst、panel split 或 value-pending stub，transported mass 会直接暴露这种依赖。

## 6. 一句话投稿卖点

**CAS-DC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“局部细节曝光质量的膨胀/缺损”问题，并通过上下文生成语义锚点、非平衡 Sinkhorn 运输和反事实曝光吸收监督，把任意数量和布局的局部观测规范化为固定容量的可分类细节 slots，从而在不依赖 hazard、对抗、一致性、后验商、随机平滑、拓扑、gauge、syndrome、knockoff、evidential、信息格、solver trace 或 queue debt 的前提下，提升跨采样政策鲁棒性。**
