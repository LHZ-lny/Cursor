# Title: Do-Palimpsest Homeostatic Memory：面向采样策略偏移的病理记忆稳态写入分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*work*summary*.md`、`**/*summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取当前工作区内全部历史 proposal 的标题、黑名单与正交切入点段落，覆盖：
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
  - `ideas/Idea_Proposal_2026-08-26.md`
  - `ideas/Idea_Proposal_2026-09-12.md`
  - `ideas/Idea_Proposal_2026-09-13.md`
  - `ideas/Idea_Proposal_2026-09-14.md`
  - `ideas/Idea_Proposal_2026-09-16.md`
- 已读取自动化记忆 `MEMORIES.md` 及额外历史摘要，纳入当前工作区未完全落盘或以记忆形式保留的历史机制：`idea_2026-07-24.md`、`2026-07-25.md`、`2026-07-26.md`、`2026-07-27.md`、`2026-07-29.md`、`2026-07-31.md`、`2026-08-01.md`、`2026-08-04.md`、`2026-08-07.md`、`2026-08-10.md`、`2026-08-11.md`、`2026-08-21.md`、`idea_2026-09-15.md` 与 `idea_2026-09-16.md`。
- 已读取近期 `paper_daily.md` 索引和 `paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md` 的最新机制，重点纳入：
  - **Continuum Dropout for Neural Differential Equations**：连续时间随机正则化提示 sampling shift 下应关注时间一致的内部动力学，但本提案不复用 alternating-renewal exposure meter。
  - **Towards Self-Supervised Foundation Models for Critical Care Time Series**：Bi-Axial Transformer 和动态窗口预训练提示 ICU 表示需要同时处理时间轴与变量轴，但本提案不预测未来 observation process，也不复用 reward-rate pretraining。
  - **INPUTADAPTER / OpenTSLM / ORA / Informative Irregularity / SBRD**：提示跨中心部署、原生医疗时序表示、marked event 目标与 workflow shift 都重要，但本提案不做冻结源域检索、语言解释、collider 封印或元适配。

### 历史核心机制黑名单

为避免思维重合，本轮明确避开以下历史主机制：

1. learnable reference points / adaptive time encoding、频域掩码修复、prototype 约束、missingness pattern 直接分类、简单 policy adversarial / IRM。
2. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
3. 生理流-采样算子交换子、value/policy graph 分离、policy residual sink。
4. protocol tax / additive evidence market、posterior quotient、density ratio / doubly robust、policy-simplex randomized smoothing。
5. reconstruction error cartography、VQ clauses、optional-stopping martingale、censored topology、policy gauge、syndrome code、knockoff calendar。
6. observability witness、evidential vacuity、information lattice、solver trace front-door、conformal sleeves、IV/control-function、Borda jury。
7. Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock、feasible hull、IRT-DIF、RG fixed point。
8. bitemporal curtain、clinical tomography、matched risk-set likelihood、Gaussian privacy cloak、CauKer orthogonal synthetic forge、JEPA 双辩裁判。
9. cross-representation PID prism、Noether semantic action、fast/slow meta workflow adapter、test-time workflow adaptation。
10. ORA collider factorization、explain-away responsibility gate、workflow-swap path-specific seal。
11. source atlas / policy-antipodal retrieval、frozen predictor input-adapter contract。
12. DeepFRC-inspired monotone registration、minimal-Schwarzian warp canonicalizer、warp-curvature sidecar。
13. alternating-renewal internal clocks、renewal-normalized reward-rate readout、renewal-balance residual、exposure duty calibration。

本提案选择新的正交切入点：**不估计采样过程，不校准集合，不做检索契约，不做证据定价，不把证据除以暴露量，也不让不同采样视图保持 logits 或 representation 一致；而是把分类器前的序列读出改成一块固定容量的病理记忆板。观测事件只有在带来足够 value novelty 时才允许写入记忆；重复复测、panel 打包、routine round 或 pending 造成的额外事件会快速累积写入压力并进入“重复疲劳”，因此采样政策可以改变事件数量，却不能通过多写几次把同一病理证据无限放大。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样分类中，很多 shortcut 来自 **无限写入权**：

- 训练医院报警后 2 小时内连续复查 lactate，普通 attention / pooling 会给这些重复 token 多次发言机会；
- 另一个中心把同一 panel 拆成 8 个异步事件，模型看到的不是一次临床测量，而是 8 次可被累计的证据；
- 可穿戴设备在某些时段高频采样，模型把采样频率当成状态持续时间；
- ICU foundation model 在动态窗口中预测未来值时，密集记录中心拥有更多被预测 token，表示可能更偏向该中心的记录语法。

现有历史方案已经从概率、几何、后验、信息、证明、校准、隐私、检索、元学习和 renewal 计量等角度处理过这个问题。本轮换一个更接近神经系统记忆和在线部署的视角：

> 稳健分类器不应允许采样政策任意增加“写入病理记忆”的次数。真正新的病理变化应刷新记忆；同一值在短时间内被重复采样，只应增加写入压力和疲劳，而不能线性增加类别 margin。

近期 paper daily 的两个机制给了直接启发：

1. **Continuum Dropout** 提醒我们连续时间模型中正则化不能只是离散 token dropout，需要尊重时间轴上的状态持续性。本提案吸收“连续时间内部动力学”的思想，但不使用 alternating-renewal clock，也不把证据写成 reward / exposure rate；它使用 deterministic leaky pressure dynamics，记录 slot 被频繁写入后的疲劳程度。
2. **Critical-care foundation model / BAT** 提示 ICU 表示需要同时沿时间轴和变量轴建模。Do-Palimpsest 把双轴注意力改造成 **双轴固定容量记忆**：变量轴决定写到哪类病理槽，时间轴决定记忆压力如何衰减和覆盖；分类头读的是最终病理记忆板，而不是所有事件 token 的可变长度集合。

这个机制与当前“采样解耦/反事实干预”框架自然结合：

- value process 产生 event semantic novelty 和 candidate memory content；
- sampling process 不进入分类头，只生成 policy edit recipes 和 write-pressure audit；
- counterfactual intervention 不用于 logits consistency、risk variance、conformal calibration、atlas retrieval 或 renewal balance，而是制造 routine / alarm / panel / pending 等采样情景，审计这些情景是否只增加 memory pressure 而不增加 accepted pathological writes；
- classifier 只读取 homeostatic memory slots。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从全事件注意力改为 Palimpsest Homeostatic Memory

给定不规则事件流：

```text
e_i = (x_i, t_i, v_i, q_i, pending_i)
```

先由 value stem 产生候选写入内容、病理 novelty 与 slot key：

```text
h_i, novelty_i, key_i = ValueNoveltyStem(x_i, v_i, delta_t_i, q_i)
```

然后写入固定容量记忆板：

```text
M in R^{K x H}, P in R^K
```

其中：

- `M_k` 是第 `k` 个病理记忆槽，类似一块可被反复覆盖的 palimpsest；
- `P_k` 是该槽的连续时间写入压力；
- 事件到 slot 的 assignment 只由 value semantics / variable semantics 决定，不由 policy id、center id 或显式采样摘要决定；
- 写入 gate 同时依赖 novelty 与 pressure：

```text
a_{i,k} = softmax(key_i M_k)
g_{i,k} = sigmoid(NoveltyHead(h_i)) * exp(-P_k)
M_k <- (1 - a_{i,k} g_{i,k}) M_k + a_{i,k} g_{i,k} Candidate(h_i)
P_k <- exp(-rho * delta_t_i) P_k + a_{i,k} g_raw_i
```

直觉：

- alarm-dense 重复复测会持续命中同一 slot，使 `P_k` 升高，后续相似事件写入被疲劳抑制；
- 如果观测值真的发生新的病理突变，`novelty_i` 会升高，可以克服部分 pressure 并刷新 slot；
- panel pack / split 改变事件排列，但固定 slot 容量和压力衰减让分类头看到的是“最终病理记忆状态”，不是 token 个数；
- value-pending 可被写成低质量候选，除非其后有真实 value novelty，否则只能增加压力，不能制造高类别证据。

关键差异：

- 不是 evidence market：没有协议税、预算购买或 token price；
- 不是 renewal exposure：不采样内部 active clock，也不做 reward/exposure 分母；
- 不是 topology / code / proof / atlas：不构造拓扑胶囊、纠错码、证明链或源域近邻；
- 不是多视图一致性：反事实视图只审计写入预算，不要求最终 slot 或 logits 相同；
- 不是 adaptive time reference：时间只用于压力衰减，不学习 reference points。

### 2.2 改 Loss：从表示不变转向写入稳态纪律

总目标：

```text
L = L_cls
  + lambda_homeo * L_slot_homeostasis
  + lambda_dup   * L_duplicate_write_fatigue
  + lambda_cf    * L_counterfactual_write_budget
  + lambda_axis  * L_biaxial_memory_balance
  + lambda_nov   * L_value_novelty_anchor
```

#### A. Palimpsest Classification `L_cls`

分类器只读取最终记忆板：

```text
z = Pool(M_final, slot_usage)
L_cls = CE(Classifier(z), y)
```

事件数量、mask 密度、panel 同步度不直接进入分类头。它们只能通过 value novelty 触发记忆写入；重复事件主要增加 pressure。

#### B. Slot Homeostasis `L_slot_homeostasis`

固定容量记忆容易出现两个退化：所有事件写入同一槽，或所有槽都不写。用 slot usage 的熵与目标利用率约束：

```text
usage_k = sum_i a_{i,k} g_{i,k}
L_homeo = KL(normalize(usage) || Uniform(K))
         + relu(mean(usage) - usage_hi)^2
         + relu(usage_lo - mean(usage))^2
```

它的目标不是让不同策略表示相同，而是保持记忆板处于可写、不过载、不空转的稳态。

#### C. Duplicate Write Fatigue `L_duplicate_write_fatigue`

对同一变量的短间隔相似值，dataloader 计算 `duplicate_score_i`；这些事件不应产生大量 accepted write：

```text
L_dup = mean duplicate_score_i * accepted_write_i
```

若值发生实质变化，`duplicate_score_i` 低或 `novelty_target_i` 高，写入仍被允许。这样避免把“多测几次同一个值”变成“多份类别证据”。

#### D. Counterfactual Write Budget `L_counterfactual_write_budget`

反事实采样模块生成 routine-round、alarm-dense、panel-pack、panel-split、variable-budget、pending-latency 等 policy edits。Do-Palimpsest 不比较 logits，而比较“额外事件带来的写入质量”是否被 value novelty 解释：

```text
excess_write = accepted_write_cf - accepted_write_factual
novelty_gain = value_novelty_cf - value_novelty_factual
L_cf = relu(excess_write - c * novelty_gain - tau)^2
```

如果反事实视图只是把同一值复制、拆分、重排或降低质量，`novelty_gain` 近似为 0，额外写入应被抑制；如果视图确实揭示新的病理值，允许写入预算增加。

#### E. Bi-Axial Memory Balance `L_biaxial_memory_balance`

借鉴 BAT 的时间轴 / 变量轴分工，但不做普通双轴 attention。记忆板被拆成：

- variable slots：承载变量特异状态；
- temporal phase slots：承载早/中/晚或病程段的状态；
- cross slots：只在 value novelty 同时支持变量和阶段变化时写入。

约束 cross slots 不能仅由 panel 同步或时间密集度驱动：

```text
L_axis = corr(cross_usage, panel_density)^2
       + corr(cross_usage, local_event_count)^2
       - corr(cross_usage, value_novelty).detach()
```

实现中最后一项可作为监控或弱奖励，防止 cross memory 完全不用。

#### F. Value Novelty Anchor `L_value_novelty_anchor`

为了避免模型把所有重复事件都拒写而漏掉急性恶化，使用 value-only novelty target：

```text
novelty_target_i = normalized_abs_change_same_variable
                 + pathology_bin_transition_i
                 + quality_recovery_i
L_nov = BCE(novelty_i, novelty_target_i)
```

该 target 不包含中心、policy id、未来是否被测或 workflow metadata；它只锚定“这个事件是否带来新的病理值信息”。

### 2.3 改 Dataloader：返回 Palimpsest Audit Batch

新增 `PalimpsestAuditCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_quality`、`value_pending`。
2. `same_var_delta`、`pathology_bin_transition`、`quality_recovery`，用于 value novelty anchor。
3. `duplicate_score`：短间隔、同变量、相似值、低质量变化的重复分数。
4. `panel_density`、`local_event_count`：只用于 memory balance audit，不进入分类头。
5. `policy_edit_bank`：
   - `routine_round_edit`：将时间吸附到固定查房节律；
   - `alarm_dense_edit`：在 late / abnormal 区域插入重复观测；
   - `panel_pack_edit`：把异步变量压到同一时间簇；
   - `panel_split_edit`：把同步 panel 拆成异步事件；
   - `variable_budget_edit`：削减或扩充变量组观测；
   - `pending_latency_edit`：保留施测事件但降低 value quality。

这些 edit 不是 contrastive positives，不做 logits consistency，不做风险方差，也不用于 policy classifier；它们只检查采样政策改变时，额外事件是否被写入压力吸收。

### 2.4 推理阶段

给定测试样本：

1. value stem 逐事件计算 novelty 和候选写入；
2. palimpsest memory 根据压力衰减和写入疲劳在线更新；
3. classifier 读取最终固定容量记忆板；
4. 同时报告：
   - `accepted_write_mass`：本样本最终有多少事件真正写入病理记忆；
   - `fatigue_ratio`：有多少高密度事件被判定为重复疲劳；
   - `slot_pressure_max`：是否存在某个变量或时间段被过度采样；
   - `write_budget_alarm`：policy edits 下 excess write 是否无法被 value novelty 解释；
   - `memory_trace_map`：哪些事件刷新了记忆，哪些事件只是增加压力。

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def event_delta_time(event_time: torch.Tensor) -> torch.Tensor:
    delta_t = torch.zeros_like(event_time)
    delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
    if event_time.size(1) > 1:
        delta_t[:, 0] = delta_t[:, 1]
    return delta_t


def safe_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (x.square().mean().sqrt() * y.square().mean().sqrt() + eps)


class ValueNoveltyStem(nn.Module):
    """Encode events and estimate whether an event deserves a memory write."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.5, 2.5, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_bins + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.key_head = nn.Linear(hidden_dim, hidden_dim)
        self.content_head = nn.Linear(hidden_dim, hidden_dim)
        self.novelty_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))
        pending = batch.get("value_pending", torch.zeros_like(value))

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * mask.unsqueeze(-1)

        delta_t = event_delta_time(time)
        same_delta = batch.get("same_var_delta", torch.zeros_like(value))
        event_x = torch.cat(
            [
                self.var_embed(var_id),
                bin_prob,
                value.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                same_delta.unsqueeze(-1),
                quality.unsqueeze(-1),
                pending.unsqueeze(-1),
            ],
            dim=-1,
        )
        h = self.event_proj(event_x) * mask.unsqueeze(-1)
        novelty = torch.sigmoid(self.novelty_head(h).squeeze(-1)) * mask
        return {
            "event_h": h,
            "slot_key": F.normalize(self.key_head(h), dim=-1),
            "candidate": self.content_head(h),
            "novelty": novelty,
            "delta_t": delta_t,
            "event_mask": mask,
            "bin_prob": bin_prob,
        }


class PalimpsestHomeostaticMemory(nn.Module):
    """Fixed-capacity memory with continuous-time write pressure and fatigue."""

    def __init__(self, hidden_dim: int, num_slots: int = 16, pressure_decay: float = 0.25):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.pressure_decay = pressure_decay
        self.init_memory = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.slot_type = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.write_gain = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.readout_gate = nn.Sequential(nn.Linear(hidden_dim + 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, stem: dict) -> dict:
        key = stem["slot_key"]
        candidate = stem["candidate"]
        novelty = stem["novelty"]
        delta_t = stem["delta_t"]
        mask = stem["event_mask"]
        bsz, num_events, _ = candidate.shape
        device = candidate.device

        memory = self.init_memory.unsqueeze(0).expand(bsz, -1, -1).clone()
        pressure = torch.zeros(bsz, self.num_slots, device=device, dtype=candidate.dtype)
        usage = torch.zeros_like(pressure)
        accepted_trace = []
        raw_trace = []
        assign_trace = []
        pressure_trace = []

        slot_key = F.normalize(self.slot_type, dim=-1)
        for idx in range(num_events):
            valid = mask[:, idx : idx + 1]
            decay = torch.exp(-self.pressure_decay * delta_t[:, idx : idx + 1])
            pressure = pressure * decay

            logits = torch.einsum("bh,kh->bk", key[:, idx], slot_key)
            assign = torch.softmax(logits, dim=-1) * valid
            raw_write = torch.sigmoid(self.write_gain(candidate[:, idx]).squeeze(-1)) * novelty[:, idx] * valid.squeeze(-1)
            fatigue = torch.exp(-pressure).clamp(0.02, 1.0)
            accepted = assign * raw_write.unsqueeze(-1) * fatigue

            write_content = candidate[:, idx].unsqueeze(1)
            memory = memory * (1.0 - accepted.unsqueeze(-1)) + accepted.unsqueeze(-1) * write_content
            pressure = pressure + assign * raw_write.unsqueeze(-1)
            usage = usage + accepted

            accepted_trace.append(accepted.sum(dim=-1))
            raw_trace.append(raw_write)
            assign_trace.append(assign)
            pressure_trace.append(pressure)

        slot_usage = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pressure_stat = torch.stack(pressure_trace, dim=1).amax(dim=1) if pressure_trace else pressure
        read_x = torch.cat(
            [
                memory,
                slot_usage.unsqueeze(-1).expand(-1, -1, 1),
                pressure_stat.unsqueeze(-1).expand(-1, -1, 1),
            ],
            dim=-1,
        )
        read_gate = torch.softmax(self.readout_gate(read_x).squeeze(-1), dim=-1)
        summary = torch.einsum("bk,bkh->bh", read_gate, memory)
        return {
            "memory": memory,
            "summary": summary,
            "usage": usage,
            "slot_usage": slot_usage,
            "read_gate": read_gate,
            "accepted_write": torch.stack(accepted_trace, dim=1),
            "raw_write": torch.stack(raw_trace, dim=1),
            "assignment": torch.stack(assign_trace, dim=1),
            "slot_pressure": pressure_stat,
        }


class DoPalimpsestHomeostaticMemory(nn.Module):
    """Sampling-policy robust classifier via homeostatic pathological memory."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        hidden_dim: int,
        num_classes: int,
        num_slots: int = 16,
    ):
        super().__init__()
        self.stem = ValueNoveltyStem(num_vars=num_vars, num_bins=num_bins, hidden_dim=hidden_dim)
        self.memory = PalimpsestHomeostaticMemory(hidden_dim=hidden_dim, num_slots=num_slots)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_slots = num_slots

    def encode(self, batch: dict) -> dict:
        stem = self.stem(batch)
        mem = self.memory(stem)
        logits = self.classifier(mem["summary"])
        return {**stem, **mem, "logits": logits}

    def slot_homeostasis_loss(self, out: dict, lo: float = 0.10, hi: float = 4.00) -> torch.Tensor:
        usage = out["usage"]
        prob = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        uniform = torch.full_like(prob, 1.0 / self.num_slots)
        entropy_kl = (prob * (prob.clamp_min(1e-6).log() - uniform.log())).sum(dim=-1).mean()
        mean_usage = usage.mean(dim=-1)
        range_loss = F.relu(mean_usage - hi).pow(2).mean() + F.relu(lo - mean_usage).pow(2).mean()
        return entropy_kl + range_loss

    def duplicate_write_fatigue_loss(self, batch: dict, out: dict) -> torch.Tensor:
        duplicate = batch.get("duplicate_score")
        if duplicate is None:
            return torch.zeros((), device=out["logits"].device)
        accepted = out["accepted_write"] * out["event_mask"]
        return (duplicate * accepted).sum() / out["event_mask"].sum().clamp_min(1.0)

    def biaxial_memory_balance_loss(self, batch: dict, out: dict) -> torch.Tensor:
        panel = batch.get("panel_density")
        count = batch.get("local_event_count")
        if panel is None or count is None:
            return torch.zeros((), device=out["logits"].device)
        cross_usage = out["accepted_write"].sum(dim=1)
        return safe_corr(cross_usage, panel.float()).pow(2) + safe_corr(cross_usage, count.float()).pow(2)

    def value_novelty_anchor_loss(self, batch: dict, out: dict) -> torch.Tensor:
        if "novelty_target" in batch:
            target = batch["novelty_target"].float()
        else:
            same_delta = batch.get("same_var_delta", torch.zeros_like(out["novelty"]))
            transition = batch.get("pathology_bin_transition", torch.zeros_like(out["novelty"]))
            quality = batch.get("quality_recovery", torch.zeros_like(out["novelty"]))
            target = (same_delta.abs() + transition + quality).clamp(0.0, 1.0)
        raw = F.binary_cross_entropy(out["novelty"], target, reduction="none")
        return (raw * out["event_mask"]).sum() / out["event_mask"].sum().clamp_min(1.0)

    def counterfactual_write_budget_loss(self, batch: dict, factual: dict, slack: float = 0.05) -> torch.Tensor:
        views = batch.get("policy_edit_bank", [])
        if not views:
            return torch.zeros((), device=factual["logits"].device)

        factual_write = factual["accepted_write"].sum(dim=1).detach()
        factual_novelty = (factual["novelty"] * factual["event_mask"]).sum(dim=1).detach()
        losses = []
        for view in views:
            cf = self.encode(view)
            cf_write = cf["accepted_write"].sum(dim=1)
            cf_novelty = (cf["novelty"] * cf["event_mask"]).sum(dim=1)
            excess_write = cf_write - factual_write
            novelty_gain = (cf_novelty - factual_novelty).clamp_min(0.0)
            losses.append(F.relu(excess_write - novelty_gain - slack).pow(2).mean())
        return torch.stack(losses).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_homeo: float = 0.15,
        lambda_dup: float = 0.25,
        lambda_cf: float = 0.35,
        lambda_axis: float = 0.10,
        lambda_nov: float = 0.15,
    ) -> dict:
        out = self.encode(batch)
        cls_loss = F.cross_entropy(out["logits"], batch["labels"])
        homeo_loss = self.slot_homeostasis_loss(out)
        dup_loss = self.duplicate_write_fatigue_loss(batch, out)
        cf_loss = self.counterfactual_write_budget_loss(batch, out)
        axis_loss = self.biaxial_memory_balance_loss(batch, out)
        novelty_loss = self.value_novelty_anchor_loss(batch, out)
        total = (
            cls_loss
            + lambda_homeo * homeo_loss
            + lambda_dup * dup_loss
            + lambda_cf * cf_loss
            + lambda_axis * axis_loss
            + lambda_nov * novelty_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "slot_homeostasis_loss": homeo_loss.detach(),
            "duplicate_write_fatigue_loss": dup_loss.detach(),
            "counterfactual_write_budget_loss": cf_loss.detach(),
            "biaxial_memory_balance_loss": axis_loss.detach(),
            "value_novelty_anchor_loss": novelty_loss.detach(),
            "accepted_write_mass": out["accepted_write"].sum(dim=1).mean().detach(),
            "max_slot_pressure": out["slot_pressure"].amax(dim=-1).mean().detach(),
        }
```

## 4. Palimpsest Audit Collator 草稿

```python
from __future__ import annotations

import torch


@torch.no_grad()
def build_palimpsest_audit_batch(batch: dict, num_bins: int = 8) -> dict:
    """Attach value-novelty targets and policy edits for write-budget auditing."""

    out = dict(batch)
    out.update(build_value_novelty_targets(batch, num_bins=num_bins))
    out.update(build_sampling_density_audits(batch))
    out["policy_edit_bank"] = build_policy_edit_bank(batch)
    return out


@torch.no_grad()
def build_value_novelty_targets(batch: dict, num_bins: int) -> dict:
    value = batch["event_value"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    quality = batch.get("measurement_quality", torch.ones_like(value))

    same_var = torch.zeros_like(mask)
    same_var[:, 1:] = (var_id[:, 1:] == var_id[:, :-1]).to(mask.dtype)
    same_delta = torch.zeros_like(value)
    same_delta[:, 1:] = (value[:, 1:] - value[:, :-1]).abs() * same_var[:, 1:]

    value_scale = (value * mask).std(dim=1, keepdim=True).clamp_min(1e-3)
    same_var_delta = (same_delta / value_scale).clamp(0.0, 1.0) * mask

    bins = torch.bucketize(
        value,
        torch.linspace(value.min().detach(), value.max().detach(), num_bins - 1, device=value.device),
    ).clamp(0, num_bins - 1)
    transition = torch.zeros_like(mask)
    transition[:, 1:] = ((bins[:, 1:] != bins[:, :-1]) & (same_var[:, 1:] > 0)).to(mask.dtype)

    quality_recovery = torch.zeros_like(quality)
    quality_recovery[:, 1:] = (quality[:, 1:] - quality[:, :-1]).clamp_min(0.0)

    novelty_target = (same_var_delta + transition + quality_recovery).clamp(0.0, 1.0) * mask
    duplicate_score = ((1.0 - novelty_target) * same_var * mask).clamp(0.0, 1.0)
    return {
        "same_var_delta": same_var_delta,
        "pathology_bin_transition": transition,
        "quality_recovery": quality_recovery,
        "novelty_target": novelty_target,
        "duplicate_score": duplicate_score,
    }


@torch.no_grad()
def build_sampling_density_audits(batch: dict) -> dict:
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]

    delta_t = torch.zeros_like(time)
    delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (delta_t * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (delta_t <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_density = (close * changed_var * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    local_event_count = mask.sum(dim=1) / mask.size(1)
    return {"panel_density": panel_density, "local_event_count": local_event_count}


@torch.no_grad()
def build_policy_edit_bank(batch: dict) -> list[dict]:
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    quality = batch.get("measurement_quality", torch.ones_like(value))
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, new_quality):
        view = dict(batch)
        view["event_value"] = new_value
        view["event_time"] = new_time
        view["event_var_id"] = new_var
        view["event_mask"] = new_mask
        view["measurement_quality"] = new_quality
        view.pop("policy_edit_bank", None)
        view.update(build_value_novelty_targets(view, num_bins=8))
        view.update(build_sampling_density_audits(view))
        return view

    views = []

    # Routine-round edit: same values, more regular calendar.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    views.append(clone_with(value * mask, rounded_time, var_id, mask, quality))

    # Alarm-dense edit: repeat late-window observations without adding value novelty.
    late = (time_norm > 0.66).to(mask.dtype)
    repeat_value = torch.where(late > 0, value.roll(shifts=1, dims=1), value)
    views.append(clone_with(repeat_value * mask, time, var_id, mask, quality))

    # Panel-pack edit: compress near-neighbor variables into synchronous clusters.
    packed_time = time.clone()
    packed_time[:, 1:] = torch.where(mask[:, 1:] > 0, time[:, :-1], time[:, 1:])
    views.append(clone_with(value * mask, packed_time, var_id, mask, quality))

    # Panel-split edit: weaken half of cross-variable close observations.
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    split_mask = mask * torch.where((var_id % 2 == 0), torch.ones_like(mask), alternating)
    views.append(clone_with(value * split_mask, time, var_id, split_mask, quality * split_mask))

    # Variable-budget edit: reduce odd-variable exposure.
    budget_mask = mask * (1.0 - 0.5 * (var_id % 2 == 1).to(mask.dtype))
    views.append(clone_with(value * budget_mask, time, var_id, budget_mask, quality * budget_mask))

    # Pending-latency edit: administration visible, value quality downgraded.
    pending_quality = quality * 0.2
    views.append(clone_with(value * mask, time, var_id, mask, pending_quality))
    return views
```

## 5. 实验切入点

1. **Policy shift 构造**
   - `repeat amplification shift`：训练中心中高风险患者有更多重复复测，测试中心把同样复测协议扩展到普通患者。
   - `panel pack / split shift`：同一组变量在训练中同步联测，测试中被异步拆分，或反向操作。
   - `pending latency shift`：保留事件但降低 value quality，检查模型是否只因“已下单”而写入高风险记忆。
   - `variable budget shift`：测试中心削减某类变量观测预算，检查固定容量记忆是否仍保留关键 novelty。
   - `foundation pretraining shift`：先用 BAT / ICU self-supervised backbone 预训练，再接 Do-Palimpsest 读出，比较普通 token pooling 是否更容易被记录密度放大。

2. **对比方法**
   - 普通 irregular Transformer / GRU-D / Neural CDE / Mamba / BAT。
   - Continuum Dropout 原始正则化。
   - Critical-care foundation model 的 head-only fine-tuning。
   - token dropout / random missing augmentation / attention pooling。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DSPP、DCPD、DIPF、DRG-SFF、DCOFF、DD-JEPA、DPSP、DNSA、DMWI、DCST、DAAA、DREM 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy worst AUROC / AUPRC。
   - repeat amplification slope：重复相似事件数量增加时 true-class logit 的斜率。
   - accepted-write / raw-event ratio：高密度样本中真实写入比例是否下降。
   - fatigue calibration：重复复测、panel pack、pending 事件是否对应更高 slot pressure。
   - novelty preservation：真实急性 value transition 是否仍能刷新记忆。
   - write-budget violation AUC：无法由 value novelty 解释的 excess write 是否预测跨政策错误。

4. **消融实验**
   - 去掉 pressure fatigue，退化为普通 slot attention，检查重复复测 shortcut 是否回归。
   - 去掉 `L_counterfactual_write_budget`，检查 policy edits 是否能制造额外 accepted writes。
   - 去掉 value novelty anchor，检查模型是否过度拒写而漏掉急性恶化。
   - 把 `panel_density` / `local_event_count` 直接输入 classifier 作为反例，验证院内性能可能升高但跨政策退化。
   - 用普通 Bernoulli dropout 替代 pressure dynamics，验证固定容量记忆与写入疲劳不是普通正则化。
   - 扫描 slot 数量和 pressure decay，评估记忆容量、信息保留与采样鲁棒性的 trade-off。

## 6. 预期创新性

1. **从采样去偏转向病理记忆写入权控制**：历史方案多在概率、几何、后验、校准、隐私或检索层处理偏移；Do-Palimpsest 直接限制采样政策增加事件数后对分类记忆的写入次数。
2. **从可变长度 token pooling 转向固定容量 palimpsest memory**：分类头不再面对由采样政策决定长度和密度的 token set，而面对固定槽位的最终记忆状态。
3. **从 dropout / renewal clock 转向 deterministic write pressure**：吸收连续时间内部动力学的启发，但不随机失活、不做 reward/exposure rate；slot pressure 是可解释的重复疲劳诊断。
4. **从双轴 attention 转向双轴记忆稳态**：借鉴 BAT 的时间轴与变量轴，但把注意力改成写入容量和压力平衡，减少 panel 同步和事件密度对分类证据的直接放大。
5. **从反事实一致性转向写入预算审计**：反事实采样 view 不要求 logits 或 representation 相同，只检查额外写入是否有额外 value novelty 支持。
6. **部署解释直指采样偏移失败模式**：当预测失败时，可以报告是 repeat amplification、panel packing、pending latency 还是 variable budget 造成了异常写入压力。

## 7. 一句话投稿卖点

**Do-Palimpsest Homeostatic Memory 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样制度夺取了病理记忆的重复写入权”的问题，通过固定容量 palimpsest slots、连续时间写入压力、重复采样疲劳、双轴记忆稳态与 counterfactual write-budget audit，让分类器只读取由 value novelty 刷新的病理记忆，而不是读取训练医院或设备制造的复测次数、panel 拆分、记录密度或 pending 事件数量。**
