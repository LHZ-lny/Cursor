# Title: Do-Noether Semantic Action：面向采样策略偏移的反事实语义作用量守恒分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`、`*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未检出可读取的工作总结文件。
- 已读取自动化记忆 `MEMORIES.md`，纳入其中记录的未落盘历史机制摘要。
- 已读取当前工作区内全部历史 proposal 文件：
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
  - `ideas/Idea_Proposal_2026-08-06.md`
  - `ideas/Idea_Proposal_2026-08-08.md`
  - `ideas/Idea_Proposal_2026-08-09.md`
  - `ideas/Idea_Proposal_2026-08-22.md`
  - `ideas/Idea_Proposal_2026-08-23.md`
  - `ideas/Idea_Proposal_2026-08-24.md`
  - `ideas/Idea_Proposal_2026-08-25.md`
- 已读取近期论文记录 `paper_daily.md` 的最新相关段落，以及 `paper_daily_2026-08-23.md`、`paper_daily_2026-08-24.md`，重点纳入：
  - **MATA-Former & SIICU**：semantic-aware temporal alignment、医学事件语义有效期、text-intensive ICU event-wise risk modeling。
  - **Cross-Representation Benchmarking**：同一 EHR 可渲染为 grid / event / text 三种表示，representation choice 会改变 sampling-policy shortcut 的暴露路径。

### 历史核心机制黑名单

为避免思维重合，本轮明确避开以下历史主机制：

1. learnable reference points / adaptive time encoding、频域掩码修复、prototype 约束、简单 policy adversarial。
2. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
3. 算子交换子、value/policy graph 分离、policy residual sink。
4. protocol tax / additive evidence market、后验商、density ratio / doubly robust、随机平滑认证。
5. error cartography、VQ clauses、optional-stopping martingale、censored topology、policy gauge、syndrome code、knockoff calendar。
6. evidential vacuity、information lattice、solver trace front-door、conformal sleeves、IV/control-function、Borda jury。
7. Krylov policy annihilator、Nystrom volume、tropical support routes、fixed viva、sequent proof、disease-progress poset clock、IRT-DIF、RG fixed point、Gaussian privacy cloak。
8. CauKer-style orthogonal synthetic forge、outcome-conditioned JEPA debate、policy-only decoy silence。
9. 最新 `Idea_Proposal_2026-08-25.md` 的 cross-representation PID prism、shared semantic ray、unique label quarantine、policy synergy suppression。

本提案选择新的正交切入点：**不做表示一致性、PID 信息分解、投票、证明、隐私发布、合成预训练或后验/图/测度分解；而是把分类证据写成一个语义作用量，并把采样政策变化看成不应对病理类别作功的观测变换。若某个类别判断主要由医院记录频率、panel 同步、文本记录延迟或网格聚合窗口驱动，它会表现为沿 policy transformation 的非零“语义功”；Do-Noether Semantic Action 通过 Noether-style 守恒损失让最终分类只依赖对采样变换近似守恒的医学语义电荷。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 paper daily 给出两个很强但仍有空白的方向。

第一，**MATA-Former** 说明 ICU 风险建模不应只看物理时间间隔，而应看事件语义的有效期：同样过去 12 小时，抗生素、乳酸、护理 note、影像报告对 sepsis / mortality / respiratory failure 的有效时间完全不同。但语义有效期有一个危险副作用：文本记录、护理 note、复查行为本身受医院工作流影响。如果 attention bias 直接被 note timing、documentation density 或 panel ordering 牵引，模型会把“记录为何出现”误当成“病理证据仍有效”。

第二，**Cross-Representation Benchmarking** 说明，同一 EHR 原始轨迹被渲染为 grid、event stream、text stream 后，采样政策捷径会以不同形态显现：grid 中是 imputation artifact，event stream 中是 panel 共现和事件顺序，text stream 中是“复查、待返回、护理记录密集”等语言痕迹。昨天的 DPSP 已经把这个问题做成 PID 棱镜，分解 shared / unique / synergy。本轮不能再沿着信息分解路线走。

**Do-Noether Semantic Action (DNSA)** 的新观点是：

> 真正稳定的病理语义应像守恒电荷：不管同一病程被网格化、事件化、文本化，或被 routine-round / alarm-dense / panel-split / documentation-delay 等采样政策轻微变换，它对类别边界的“语义作用量”不应被观测流程作功。采样政策可以改变可见 token、记录粒度和局部时间坐标，但不能凭空给某个类别注入净语义功。

这与当前“采样解耦/反事实干预”框架自然结合：

- value process 生成医学语义事件场；
- sampling process 不进入分类头，只生成 policy transformation generator；
- counterfactual intervention 不产生一致性 pair、不做 PID、不做 conformal/JEPA/privacy，而是提供有限差分方向，用来测量采样变换是否对类别语义电荷作功；
- classifier 只读取 Noether charge，而不是 raw logits、mask、policy summary 或三表示拼接。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从 pooled representation 改为 Semantic Action Field

DNSA 把每个观测事件编码为一个医学语义作用量单元：

```text
e_i = (value_i, variable_i, semantic_type_i, time_i, quality_i)
h_i = SemanticEventStem(e_i)
```

为了吸收 MATA-Former 的优势，事件语义有效期不直接作为 attention bias 进入分类，而是作为 **Lagrangian gate**：

```text
v_i^c = sigmoid(g_semantic(type_i, class_c, delta_t_i, quality_i))
L_i^c = v_i^c * ell_c(h_i, h_{i-1}, delta_t_i)
```

其中 `L_i^c` 是类别 `c` 的局部语义作用量。它表达“该事件在当前语义有效期内对类别 c 的状态动力学贡献”，但它不读取 documentation density、policy id 或中心标签。

### 2.2 改 Classifier：Noether Charge Readout

分类器不直接池化 `h_i`，而是先计算类别语义电荷：

```text
p_i^c = MomentumHead_c(h_i, h_i - h_{i-1})
Q_c   = sum_i <p_i^c, G_c(type_i)> * delta_t_i
logits = ChargeClassifier(Q)
```

`G_c(type_i)` 是类别相关的语义生成元，例如 sepsis 风险下 inflammation / infection / perfusion 事件的生成方向。`Q_c` 可以理解为“沿病理语义流累计下来的守恒电荷”。如果模型只依赖采样流程，`Q_c` 会在反事实 policy transformation 下剧烈漂移；如果它依赖真实病理语义，`Q_c` 应接近守恒。

### 2.3 改 Loss：从一致性/信息分解转向 Noether Policy-Work Discipline

总目标：

```text
L = L_cls
  + lambda_work * L_policy_work
  + lambda_el   * L_semantic_euler_lagrange
  + lambda_gen  * L_generator_sobriety
  + lambda_val  * L_mata_validity
```

#### A. Charge Classification `L_cls`

事实观测下只用 Noether charge 分类：

```text
L_cls = CE(ChargeClassifier(Q_factual), y)
```

这不是三视图投票，也不是 shared representation 分类；所有表示最终都必须先变成语义电荷。

#### B. Policy Work Loss `L_policy_work`

反事实采样模块生成一组 policy transformation：

- `grid_window_shift`：改变网格聚合窗口；
- `event_panel_split`：拆分或合并 panel 同步事件；
- `text_document_delay`：延迟或粗化文本记录；
- `routine_alarm_morph`：在 routine-round 与 alarm-dense 之间连续变换。

对每个变换 `T_r`，计算 charge 的有限差分：

```text
W_r^c = (Q_c(T_r(x)) - Q_c(x)) / epsilon_r
```

DNSA 不要求 `Q(T_r(x)) == Q(x)`，更不要求 logits 一致。它只要求 **真实类 margin 沿采样变换方向的净作功接近 0**：

```text
margin_y = Q_y - max_{k != y} Q_k
L_policy_work = mean_r (d margin_y / d epsilon_r)^2
```

若某个训练医院的 panel 同步、文本记录密度或网格填补方式能直接推高真实类 margin，就会产生非零 policy work，被该项压制。

#### C. Semantic Euler-Lagrange Loss `L_semantic_euler_lagrange`

为了让 `L_i^c` 不是任意打分器，而像一个语义动力学作用量，对相邻事件施加离散 Euler-Lagrange 残差：

```text
EL_i = dL_i/dh_i - d/dt dL_i/d(delta h_i)
L_semantic_euler_lagrange = ||EL_i||^2
```

直觉：真实病理状态变化应由观测值和语义类型驱动，而不是由采样坐标突然注入能量。该项不是 ODE solver trace，不记录 NFE/roughness；它只约束语义作用量自身的动力学平滑性。

#### D. Generator Sobriety `L_generator_sobriety`

防止语义生成元退化为全零或任意吸收 policy：

```text
L_generator_sobriety =
  ||G_c^T G_c - I||_F^2
  + relu(min_charge_energy - ||Q||)^2
  + relu(||Q|| - max_charge_energy)^2
```

它不是 gauge frame，也不是 horizontal projection；`G_c` 不定义垂直/横向子空间，只定义类别语义作用量的生成方向。

#### E. MATA Validity Grounding `L_mata_validity`

吸收 MATA-Former 的 semantic-aware temporal alignment，但避免复用 DPSP 的 PID 语义棱镜。这里的 validity 只监督 Lagrangian gate：

```text
L_mata_validity = BCE(RiskValidityHead(Q, query_time), soft_event_risk)
```

若有 SIICU / ICU event-wise soft risk labels，可直接使用；否则可用弱标签构造 plateau-Gaussian risk curve。该项只校准语义作用量的时间有效期，不做 representation fusion、shared ray 或 policy-synergy probe。

### 2.4 改 Dataloader：返回 Policy Transformation Tangents

新增 `NoetherActionCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_semantic_type`、`event_mask`、`measurement_quality`。
2. `policy_transform_bank`：同一原始轨迹在 grid / event / text 渲染扰动下的事件 batch。
3. `epsilon_bank`：每个变换的强度，用于有限差分 policy work。
4. `semantic_validity_target`：event-wise 或 curve-wise soft risk label。
5. `transform_descriptor`：只用于生成反事实 view，不进入分类器。

与昨天 DPSP 的区别：

- 不做 grid/event/text 的 PID 分解。
- 不训练 unique label probe 或 policy synergy probe。
- 不要求三种表示 shared latent 一致。
- 不把 representation choice 当作 ensemble 成员。
- 只把多表示/多采样渲染当成估计 policy transformation tangent 的工具。

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


def true_margin(charge_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_score = charge_logits.gather(1, labels[:, None]).squeeze(1)
    rival = charge_logits.masked_fill(
        F.one_hot(labels, charge_logits.size(-1)).bool(),
        -1e4,
    ).max(dim=-1).values
    return true_score - rival


class SemanticActionStem(nn.Module):
    """Lift irregular clinical events into semantic action states."""

    def __init__(self, num_vars: int, num_semantic_types: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.type_embed = nn.Embedding(num_semantic_types, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        sem_type = batch["event_semantic_type"]
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon

        x = torch.cat(
            [
                self.var_embed(var_id.clamp_min(0)),
                self.type_embed(sem_type.clamp_min(0)),
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(x) * mask.unsqueeze(-1)
        ctx, _ = self.context(event_h)
        h = self.out(ctx) * mask.unsqueeze(-1)
        return {"event_state": h, "delta_t": delta_t, "event_mask": mask}


class SemanticLagrangian(nn.Module):
    """Class-wise semantic Lagrangian with MATA-style validity gates."""

    def __init__(self, hidden_dim: int, num_semantic_types: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.type_embed = nn.Embedding(num_semantic_types, hidden_dim)
        self.validity = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.local_energy = nn.Sequential(
            nn.Linear(3 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.momentum = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, sem_type: torch.Tensor, delta_t: torch.Tensor, mask: torch.Tensor) -> dict:
        # h: [B, N, H]
        bsz, num_events, hidden_dim = h.shape
        classes = torch.arange(self.num_classes, device=h.device)
        class_h = self.class_embed(classes)[None, None].expand(bsz, num_events, -1, -1)
        type_h = self.type_embed(sem_type.clamp_min(0))[:, :, None].expand_as(class_h)
        event_h = h[:, :, None, :].expand_as(class_h)

        validity_x = torch.cat(
            [
                type_h,
                class_h,
                torch.log1p(delta_t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_classes, -1),
                mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_classes, -1),
            ],
            dim=-1,
        )
        validity = torch.sigmoid(self.validity(validity_x)).squeeze(-1)

        dh = torch.zeros_like(h)
        dh[:, 1:] = h[:, 1:] - h[:, :-1]
        dh_c = dh[:, :, None, :].expand_as(class_h)
        energy_x = torch.cat(
            [
                event_h,
                dh_c,
                class_h,
                torch.log1p(delta_t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_classes, -1),
            ],
            dim=-1,
        )
        lagrangian = self.local_energy(energy_x).squeeze(-1) * validity * mask.unsqueeze(-1)

        momentum = self.momentum(torch.cat([h, dh], dim=-1)) * mask.unsqueeze(-1)
        return {"lagrangian": lagrangian, "validity": validity, "momentum": momentum, "delta_h": dh}


class NoetherChargeReadout(nn.Module):
    """Compute class charges from semantic momenta and class generators."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.generator = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, momentum: torch.Tensor, delta_t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gen = F.normalize(self.generator, dim=-1)
        local_charge = torch.einsum("bnh,ch->bnc", momentum, gen)
        charge = (local_charge * torch.log1p(delta_t).unsqueeze(-1) * mask.unsqueeze(-1)).sum(dim=1)
        return charge + self.bias

    def sobriety_loss(self, charge: torch.Tensor, min_energy: float = 0.05, max_energy: float = 20.0) -> torch.Tensor:
        gen = F.normalize(self.generator, dim=-1)
        gram = gen @ gen.T
        eye = torch.eye(gen.size(0), device=gen.device, dtype=gen.dtype)
        ortho = (gram - eye).pow(2).mean()
        energy = charge.norm(dim=-1)
        return ortho + F.relu(min_energy - energy).pow(2).mean() + F.relu(energy - max_energy).pow(2).mean()


class DoNoetherSemanticAction(nn.Module):
    """Sampling-policy robust classifier using conserved semantic action charges."""

    def __init__(
        self,
        num_vars: int,
        num_semantic_types: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.stem = SemanticActionStem(num_vars, num_semantic_types, hidden_dim)
        self.lagrangian = SemanticLagrangian(hidden_dim, num_semantic_types, num_classes)
        self.charge = NoetherChargeReadout(hidden_dim, num_classes)
        self.validity_head = nn.Sequential(
            nn.Linear(num_classes + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_classes = num_classes

    def encode_charge(self, batch: dict) -> dict:
        stem = self.stem(batch)
        action = self.lagrangian(
            h=stem["event_state"],
            sem_type=batch["event_semantic_type"],
            delta_t=stem["delta_t"],
            mask=stem["event_mask"],
        )
        charge_logits = self.charge(action["momentum"], stem["delta_t"], stem["event_mask"])
        return {**stem, **action, "charge_logits": charge_logits}

    def euler_lagrange_loss(self, out: dict) -> torch.Tensor:
        # Stable proxy: penalize abrupt class-wise action jumps not explained by hidden-state displacement.
        lag = out["lagrangian"]
        mask = out["event_mask"]
        d_lag = torch.zeros_like(lag)
        d_lag[:, 1:] = lag[:, 1:] - lag[:, :-1]
        speed = out["delta_h"].pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
        residual = d_lag - speed.detach() * d_lag.sign()
        active = mask.unsqueeze(-1)
        return (residual.pow(2) * active).sum() / active.sum().clamp_min(1.0)

    def policy_work_loss(self, batch: dict, factual: dict, labels: torch.Tensor) -> torch.Tensor:
        views = batch.get("policy_transform_bank", [])
        eps_bank = batch.get("epsilon_bank", None)
        if not views:
            return torch.zeros((), device=factual["charge_logits"].device)

        factual_margin = true_margin(factual["charge_logits"], labels).detach()
        losses = []
        for idx, view in enumerate(views):
            cf = self.encode_charge(view)
            cf_margin = true_margin(cf["charge_logits"], labels)
            if eps_bank is None:
                eps = 1.0
            else:
                eps = eps_bank[:, idx].clamp_min(1e-3)
            policy_work = (cf_margin - factual_margin) / eps
            losses.append(policy_work.pow(2).mean())
        return torch.stack(losses).mean()

    def mata_validity_loss(self, out: dict, batch: dict) -> torch.Tensor:
        if "semantic_validity_target" not in batch:
            return torch.zeros((), device=out["charge_logits"].device)
        query_time = batch["validity_query_time"]
        charge = out["charge_logits"][:, None].expand(-1, query_time.size(1), -1)
        x = torch.cat([charge, query_time.unsqueeze(-1)], dim=-1)
        pred = self.validity_head(x).squeeze(-1)
        return F.binary_cross_entropy_with_logits(pred, batch["semantic_validity_target"].to(pred.dtype))

    def training_loss(
        self,
        batch: dict,
        lambda_work: float = 0.35,
        lambda_el: float = 0.08,
        lambda_gen: float = 0.05,
        lambda_val: float = 0.10,
    ) -> dict:
        labels = batch["labels"]
        factual = self.encode_charge(batch)
        cls_loss = F.cross_entropy(factual["charge_logits"], labels)
        work_loss = self.policy_work_loss(batch, factual, labels)
        el_loss = self.euler_lagrange_loss(factual)
        gen_loss = self.charge.sobriety_loss(factual["charge_logits"])
        val_loss = self.mata_validity_loss(factual, batch)

        total = (
            cls_loss
            + lambda_work * work_loss
            + lambda_el * el_loss
            + lambda_gen * gen_loss
            + lambda_val * val_loss
        )
        return {
            "loss": total,
            "charge_cls_loss": cls_loss.detach(),
            "policy_work_loss": work_loss.detach(),
            "semantic_euler_lagrange_loss": el_loss.detach(),
            "generator_sobriety_loss": gen_loss.detach(),
            "mata_validity_loss": val_loss.detach(),
        }
```

## 4. Noether Action Collator 草稿

```python
import torch


@torch.no_grad()
def build_noether_policy_transforms(batch: dict) -> dict:
    """Create policy-transformed event views for finite-difference policy work.

    These views are not contrastive positives and are not used for logits or
    representation consistency. They only estimate whether sampling transforms
    do work on semantic class charges.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    sem_type = batch["event_semantic_type"]
    mask = batch["event_mask"]
    quality = batch.get("measurement_quality", torch.ones_like(value))
    bsz, num_events = value.shape
    device = value.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_type, new_mask, new_quality):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_semantic_type"] = new_type
        out["event_mask"] = new_mask
        out["measurement_quality"] = new_quality
        out.pop("policy_transform_bank", None)
        out.pop("epsilon_bank", None)
        return out

    transforms = []
    eps = []

    # 1. Grid-window shift: snap event time to coarser clinical bins.
    rounded_time = torch.round(time_norm * 8.0) / 8.0 * horizon
    transforms.append(clone_with(value * mask, rounded_time, var_id, sem_type, mask, quality))
    eps.append(torch.full((bsz,), 0.20, device=device))

    # 2. Event panel split: weaken near-synchronous cross-variable observations.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_mask = mask * (1.0 - 0.5 * close * changed_var)
    transforms.append(clone_with(value * panel_mask, time, var_id, sem_type, panel_mask, quality * panel_mask))
    eps.append(torch.full((bsz,), 0.50, device=device))

    # 3. Documentation delay: delay text-like semantic events without altering values.
    text_like = (sem_type > 0).to(time.dtype)
    delayed_time = time + 0.05 * horizon * text_like * mask
    transforms.append(clone_with(value * mask, delayed_time, var_id, sem_type, mask, quality))
    eps.append(torch.full((bsz,), 0.05, device=device))

    # 4. Routine/alarm morph: thin early routine events while retaining late follow-up.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    routine_alarm_mask = torch.where(late > 0, mask, mask * alternating)
    transforms.append(clone_with(value * routine_alarm_mask, time, var_id, sem_type, routine_alarm_mask, quality))
    eps.append(torch.full((bsz,), 0.50, device=device))

    out = dict(batch)
    out["policy_transform_bank"] = transforms
    out["epsilon_bank"] = torch.stack(eps, dim=1)
    return out
```

## 5. 实验切入点

1. **Policy shift 构造**
   - `semantic-documentation shift`：文本/护理记录延迟、粗化或频率改变，测试 MATA-style semantic validity 是否被记录流程污染。
   - `grid-render shift`：不同中心使用不同聚合窗口、imputation 策略和 bin 宽度。
   - `event-panel shift`：训练中心同步 panel，测试中心拆分为异步事件。
   - `routine-vs-alarm morph`：固定查房式采样与报警后密集复测互换。

2. **对比方法**
   - MATA-Former-style semantic temporal bias。
   - Cross-Representation Benchmarking 的 grid / event / text 单表示与拼接 baseline。
   - 普通三表示 average / concat。
   - 历史方案：DPSP、DD-JEPA、DCOFF、DPPC、DRG-SFF、DIPF、DCPD、DSPP、DJRT、C-CRS、ST-FDN、PLSM、PIIES、CKCF、SCSC、PGHT、CETC、OS-MQ、DS-CS、PQD、PT-AEM、CGS、DHN 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy worst AUROC / AUPRC。
   - policy work norm：`||d margin_y / d epsilon_policy||`。
   - charge drift under render shift：grid / event / text 渲染改变后的语义电荷漂移。
   - semantic validity calibration：语义有效期预测与 event-wise soft risk 的匹配度。
   - shortcut work map：错误样本的 policy work 主要来自 grid window、panel split、documentation delay 还是 routine/alarm morph。

4. **消融实验**
   - 去掉 `L_policy_work`，检查类别 margin 是否重新被采样变换作功。
   - 去掉 Noether charge，直接 pooled hidden 分类，验证跨表示/跨采样退化。
   - 去掉 semantic validity gate，验证 MATA 式语义有效期对 text-intensive ICU 的价值。
   - 将 policy transform views 改为随机 mask，验证收益来自结构化 policy tangent 而非普通增强。
   - 让 transform descriptor 进入分类头作为反例，验证采样流程特征会提升院内性能但损害跨政策鲁棒性。

## 6. 预期创新性

1. **从采样去偏转向语义作用量守恒**：不估计采样概率、不做信息分解、不做一致性或投票，而是约束采样变换不能对类别语义电荷作净功。
2. **从 MATA 语义时间偏置转向 Noether 电荷**：保留医学事件语义有效期，但把它放进 class-wise Lagrangian gate，避免 documentation density 直接成为 attention 捷径。
3. **从 cross-representation comparison 转向 policy tangent measurement**：grid / event / text 渲染不用于 ensemble 或 PID，而用于估计不同表示下采样政策对语义电荷的有限差分作功。
4. **从反事实增强转向作用量诊断**：counterfactual intervention 只生成 policy transformation tangent，训练目标不是让视图相同，而是让类别 margin 对采样流程的方向导数为零。
5. **部署解释性强**：当跨中心失败时，DNSA 可以报告是哪一种采样变换对预测做了异常大的语义功，从而定位 grid artifact、panel 流程或文本记录延迟造成的 shortcut。

## 7. 一句话投稿卖点

**DNSA 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“观测流程对类别语义作用量作了不应有的功”的问题，通过 MATA-style semantic Lagrangian、Noether charge readout 与 counterfactual policy-work loss，让分类器只依赖对 grid / event / text 渲染扰动、panel 拆合、documentation delay 和 routine/alarm 采样变换近似守恒的医学语义电荷，从而避免把采样政策制造的记录密度、同步窗口或文本流程误当成可迁移病理证据。**
