# Title: Do-Collider Seal Transformer：面向采样策略偏移的观测碰撞器封印分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取当前工作区内历史 proposal 文件，包括 `ideas/Idea_Proposal_2026-06-12.md`、`2026-06-13.md`、`2026-06-14.md`、`2026-06-16.md`、`2026-06-19.md`、`2026-06-21.md`、`2026-06-22.md`、`2026-06-23.md`、`2026-06-25.md`、`2026-06-26.md`、`2026-07-12.md`、`2026-07-13.md`、`2026-07-14.md`、`2026-07-28.md`、`2026-07-30.md`、`2026-08-05.md`、`2026-08-06.md`、`2026-08-08.md`、`2026-08-09.md`、`2026-08-22.md`、`2026-08-23.md`、`2026-08-24.md`、`2026-08-25.md`、`2026-08-26.md` 与 `2026-09-12.md`。
- 已读取自动化记忆 `MEMORIES.md` 以及其中记录的未完全落盘历史 proposal 摘要，覆盖 `idea_2026-07-24.md` 至 `idea_2026-08-21.md` 的额外机制。
- 已读取近期论文记录 `paper_daily.md`、`paper_daily_2026-08-25.md` 与 `paper_daily_2026-09-11.md`，重点纳入：
  - **ORA / One Loss to Rule Them All**：把结构化 EHR 表述为 marked point process，同时建模事件时间、事件标记和连续测量。
  - **Learning Clinical Representations Under Systematic Distribution Shift**：把 measurement policy、documentation practice 与 institutional workflow shift 视为临床表征中必须处理的系统性偏移。
  - **Informative Irregularity**：routine-rhythm deviation 与 workflow metadata 对预测有价值，但也可能是高风险采样捷径。
  - **SBRD**：shared benchmark 与 regime deviation 的分工提示了状态机制和工作流机制应被分开解释。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下机制作为主创新：

1. learnable reference points / adaptive time encoding、频域掩码修复、prototype 约束、missingness pattern 直接分类、简单 policy adversarial / IRM。
2. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
3. 生理流-采样算子交换子、value/policy graph 分离、policy residual sink。
4. protocol tax / additive evidence market、posterior quotient、density ratio / doubly robust、policy-simplex randomized smoothing。
5. reconstruction error cartography、VQ clauses、optional-stopping martingale、censored topology、policy gauge、syndrome code、knockoff calendar。
6. observability witness、evidential vacuity、information lattice、solver trace front-door、conformal sleeves、IV/control-function、Borda jury。
7. Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock、feasible hull、IRT-DIF、RG fixed point。
8. bitemporal curtain、clinical tomography、matched risk-set likelihood、Gaussian privacy cloak、CauKer orthogonal synthetic forge、JEPA 双辩裁判。
9. cross-representation PID prism、Noether semantic action、fast/slow meta workflow adapter 与 test-time workflow adaptation。

本提案选择新的正交切入点：**不把采样政策删除、征税、平滑、隐私化、校准成集合、投票、证明、纠错、后验相除或元学习适配；而是把每个被记录的 EHR / IMTS 事件视为一个 collider：它同时由潜在病理状态和医院工作流共同导致。普通模型一旦条件化在“事件被观测到”上，就打开了 `workflow -> observed event <- pathology -> label` 的碰撞器偏差通道。Do-Collider Seal Transformer 用 ORA-style marked-event 预测来解释观测事件的生成，同时用可微 explain-away responsibility gate 把“为什么被测/何时被测/是否 pending”归因给工作流支路，把“测到什么病理值”归因给状态支路，最终分类器只读取 collider-sealed state tokens。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-09-11.md` 中的 ORA 给了一个非常强的建模信号：EHR 不规则事件不是普通 token，事件时间、事件标记、连续数值和 pending 状态都应进入一个统一的 marked-event 预训练目标。这个方向很有价值，但对 sampling-policy shift 来说也有一个结构性风险：事件发生在模型输入中这一事实本身，是患者状态和医疗工作流共同作用的结果。

例如：

- 乳酸被频繁复查，既可能因为病人真的更危险，也可能因为医院 A 的 sepsis protocol 更激进；
- 某个 panel 同步出现，既可能反映炎症链条，也可能反映检验科固定套餐；
- value-pending 或文书延迟，既可能与急诊拥堵有关，也可能与高风险患者优先处理有关；
- routine-rhythm deviation 有预测力，但这种预测力可能只在本地 workflow 中成立。

这正是一个典型的 **collider bias**：

```text
latent pathology state ----> observed marked event ----< workflow / sampling policy
          |                              |
          v                              v
        label                  time, mark, pending, density
```

如果模型直接以 observed marked event 为条件学习分类，workflow 与 pathology 会在 collider 上被统计关联起来。训练医院中“某类病人更常被某项检查观察到”的机制会被模型吸收；部署到新医院时，workflow 改变，碰撞器打开出的伪关联也随之改变。

**Do-Collider Seal Transformer (DCST)** 的核心直觉是：

> ORA-style marked event modeling 不应被丢弃，因为它确实能学习 EHR 事件结构；但 marked event likelihood 必须被解释为“病理状态贡献 + 工作流贡献”的 collider 生成，而不能让二者混成分类证据。我们不删除 workflow，也不让 workflow 快速适配分类器；我们让 workflow 在事件生成任务中 explain away 观测坐标，分类器只消费被封印后的 state tokens。

它与当前“采样解耦/反事实干预”框架天然兼容：

- value process 变成 **pathology-state parent**，负责解释观测值、病理分箱和可迁移状态；
- sampling process 变成 **workflow parent**，负责解释事件时间、变量标记、pending、routine deviation 和 panel 共现；
- observed marked event 被显式建模为 collider，而不是普通输入 token；
- counterfactual intervention 不再做 logits consistency、risk variance、证据税、校准集合或元学习 support；它生成 **workflow swap / collider ablation** 场景，用来训练 responsibility gate 是否把采样坐标归因给 workflow，而不是写进 state；
- classifier 只读取 `do(seal collider)` 后的 state summary。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：双父节点 Collider-Sealed Event Transformer

DCST 将每个事件 `e_i = (x_i, t_i, d_i, q_i, pending_i)` 拆成两条父支路。

1. **Pathology Parent Stream `S_theta`**
   - 输入：观测值、变量 id、TCF-style pathology bin、测量质量和局部 value change。
   - 输出：`s_i`，表示该事件中可由病理状态解释的内容。
   - 不输入 routine-rhythm deviation、center id、policy recipe 或 panel metadata。

2. **Workflow Parent Stream `W_phi`**
   - 输入：时间间隔、变量可见性、panel-like 共现、pending、routine deviation、documentation density 等观测坐标。
   - 输出：`w_i`，表示该事件中可由工作流/采样制度解释的内容。
   - 不接收真实 label，也不直接进入分类器。

3. **Collider Responsibility Gate `R_psi`**
   - 对 ORA-style 预测任务产生责任分配：

```text
r_i = sigmoid(R_psi([s_i, w_i, value_surprise_i, workflow_regular_i]))
```

其中 `r_i` 越大，表示下一事件的 gap/mark/pending 更应由 workflow explain away；`1-r_i` 越大，表示该事件确实包含病理值导致的状态信息。

4. **Collider-Sealed State Token**

```text
sealed_i = s_i * (1 - stopgrad(r_i)) + SealMixer(s_i, stopgrad(w_i)) * small_eps
```

`small_eps` 只允许工作流支路提供极弱的坐标校正，不能形成完整分类路径。最终：

```text
h_state = StateBackbone({sealed_i})
logits  = Classifier(h_state)
```

这与 DMWI 的快慢参数元适配不同：DCST 没有 test-time inner loop，也不让 workflow adapter 学分类前调制；它在单次前向中把观测事件作为 collider 解释，并通过 explain-away gate 把 workflow 对 marked-event likelihood 的贡献留在生成支路。

### 2.2 改 Loss：从不变性转向 Collider Explain-Away Discipline

总目标：

```text
L = L_cls
  + lambda_mark * L_collided_marked_event
  + lambda_resp * L_responsibility_explain_away
  + lambda_do   * L_path_specific_seal
  + lambda_val  * L_state_value_sufficiency
  + lambda_wf   * L_workflow_absorption
```

#### A. Sealed Classification `L_cls`

分类器只读取 `sealed_i` 聚合后的状态：

```text
L_cls = CE(Classifier(StateBackbone(sealed_tokens)), y)
```

workflow stream 不直接进入分类器，避免 measurement policy 和 documentation practice 作为快捷路径。

#### B. Collided Marked Event Loss `L_collided_marked_event`

吸收 ORA 的 marked event 预训练思想，但将 gap、mark、pending 的预测写成状态父节点与工作流父节点的 mixture：

```text
p_next_mark = (1 - r) * p_state(mark | s) + r * p_workflow(mark | w)
p_gap       = (1 - r) * p_state(gap  | s) + r * p_workflow(gap  | w)
p_pending   = (1 - r) * p_state(pend | s) + r * p_workflow(pend | w)
```

状态分支也预测 pathology bin / value bucket；工作流分支重点预测 timing、mark availability、pending 和 routine deviation。这样保留 ORA 对事件结构的建模优势，但不会把完整 marked-event likelihood 无差别写进分类状态。

#### C. Responsibility Explain-Away Loss `L_responsibility_explain_away`

用可计算的弱监督构造 responsibility target：

```text
state_target = sigmoid(value_surprise - workflow_regular)
workflow_target = 1 - state_target
L_resp = BCE(r, workflow_target)
```

- `value_surprise`：当前观测值相对同变量历史/病理分箱的异常程度；
- `workflow_regular`：routine rhythm、panel 同步、pending、局部高密度复测等是否能解释该事件为何出现。

直觉：若某事件只是常规 panel 或文书流程，应由 workflow explain away；若它携带难以由工作流解释的病理值突变，应保留给 state。

#### D. Path-Specific Seal Loss `L_path_specific_seal`

反事实采样模块生成 `workflow_swap_bank`：保持观测值序列与 pathology bins，不改变 label，只替换或扰动观测坐标 `t, mark visibility, pending, routine deviation`。DCST 不要求所有 counterfactual logits 一致，而是只惩罚 **workflow 直接路径** 对真实类 margin 的影响：

```text
DE_workflow = margin_y(sealed_state; do(w <- w_swap), stop_state=True)
            - margin_y(sealed_state; do(w <- w_fact), stop_state=True)
L_path_specific_seal = relu(abs(DE_workflow) - eps_de)^2
```

这里 `stop_state=True` 表示冻结 pathology stream，只测量 workflow 通过 collider gate / small coordinate mixer 对分类 margin 的直接效应。若换一个医院工作流就能显著改变真实类 margin，说明 collider 没封住。

#### E. State Value Sufficiency `L_state_value_sufficiency`

为了避免 state stream 被 workflow explain-away 后变空，要求 sealed state 仍能预测病理值摘要：

```text
L_state_value_sufficiency =
  CE(pathology_bin_hat, pathology_bin_summary)
  + SmoothL1(value_trend_hat, value_trend_summary)
```

它不同于 proof、poset、IRT 或 fixed viva；这里只是保证分类状态仍由观测值语义支撑。

#### F. Workflow Absorption `L_workflow_absorption`

workflow stream 必须能承担观测坐标预测任务：

```text
L_workflow_absorption =
  CE(gap_bucket_w, gap_bucket)
  + CE(next_mark_w, next_mark)
  + BCE(pending_w, pending)
  + BCE(routine_dev_w, routine_dev)
```

这不是 policy adversarial，也不是 policy-only decoy classification。workflow 分支被奖励解释 workflow，但其输出被 architecture seal 阻断在分类头之外。

### 2.3 改 Dataloader：返回 Collider Audit Batch

新增 `ColliderSealCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_quality`。
2. `pathology_bin_id` 或 `pathology_bin_summary`。
3. ORA-style targets：
   - `next_gap_bucket`
   - `next_mark_id`
   - `pending_target`
   - `routine_deviation_target`
4. responsibility 弱监督：
   - `value_surprise`
   - `workflow_regular`
5. `workflow_swap_bank`：
   - `routine_round_swap`：时间被规整到查房节律；
   - `alarm_dense_swap`：早期稀疏、晚期密集；
   - `panel_pack_swap` / `panel_split_swap`：改变同步共现；
   - `pending_latency_swap`：保留施测事件但 value availability 延迟；
   - `cross_center_mark_swap`：改变变量覆盖和 next-mark 分布。

这些 swap 不是 contrastive positive，不做 risk variance，不做 smoothing，不做 conformal calibration，不做 fast adaptation。它们只估计 workflow 通过 collider direct path 对分类 margin 的影响。

### 2.4 推理阶段

给定新医院或新设备的不规则序列：

1. Pathology stream 编码观测值与病理分箱；
2. Workflow stream 编码当前采样坐标；
3. responsibility gate 将可由 workflow 解释的观测坐标成分封印在 marked-event 生成支路；
4. classifier 读取 sealed state 输出类别；
5. 同时报告：
   - `mean_workflow_responsibility`：预测中多少事件被判定主要由工作流解释；
   - `direct_path_effect`：workflow swap 对真实类 margin 的直接影响；
   - `collider_leakage_alarm`：当 direct path effect 超阈值时，提示当前采样政策可能打开了 collider shortcut；
   - `state_value_sufficiency`：sealed state 是否仍有足够病理值语义。

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
    return delta_t


def true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_score = logits.gather(1, labels[:, None]).squeeze(1)
    rival = logits.masked_fill(
        F.one_hot(labels, logits.size(-1)).bool(),
        -1e4,
    ).max(dim=-1).values
    return true_score - rival


class PathologyParentStream(nn.Module):
    """State parent of the observed-event collider."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.5, 2.5, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_bins + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden_dim, hidden_dim)
        self.bin_head = nn.Linear(hidden_dim, num_bins)
        self.trend_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * mask.unsqueeze(-1)

        delta_t = event_delta_time(time)
        value_jump = torch.zeros_like(value)
        value_jump[:, 1:] = value[:, 1:] - value[:, :-1]
        same_var = torch.zeros_like(mask)
        same_var[:, 1:] = (var_id[:, 1:] == var_id[:, :-1]).to(mask.dtype)
        value_jump = value_jump * same_var

        x = torch.cat(
            [
                self.var_embed(var_id),
                bin_prob,
                value_jump.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(x) * mask.unsqueeze(-1)
        ctx, _ = self.context(event_h)
        state_event = self.out(ctx) * mask.unsqueeze(-1)
        pooled = masked_mean(state_event, mask, dim=1)
        return {
            "state_event": state_event,
            "state_pooled": pooled,
            "bin_prob": bin_prob,
            "bin_logits": self.bin_head(pooled),
            "trend_hat": self.trend_head(pooled).squeeze(-1),
        }


class WorkflowParentStream(nn.Module):
    """Workflow parent that absorbs timing, marks, pending, and routine deviations."""

    def __init__(self, num_vars: int, hidden_dim: int, num_gap_buckets: int):
        super().__init__()
        self.num_vars = num_vars
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gap_head = nn.Linear(hidden_dim, num_gap_buckets)
        self.mark_head = nn.Linear(hidden_dim, num_vars)
        self.pending_head = nn.Linear(hidden_dim, 1)
        self.routine_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        pending = batch.get("value_pending", torch.zeros_like(mask))
        routine = batch.get("routine_deviation", torch.zeros_like(mask))
        panel = batch.get("panel_indicator", torch.zeros_like(mask))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = event_delta_time(time)
        local_density = 1.0 / (1.0 + delta_t)
        var_change = torch.zeros_like(mask)
        var_change[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)

        early = (time_norm <= 0.33).to(time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype)
        late = (time_norm > 0.66).to(time.dtype)
        workflow_x = torch.cat(
            [
                self.var_embed(var_id),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                local_density.unsqueeze(-1),
                pending.unsqueeze(-1),
                routine.unsqueeze(-1),
                panel.unsqueeze(-1),
                var_change.unsqueeze(-1),
                early.unsqueeze(-1) + 0.5 * middle.unsqueeze(-1) + late.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(workflow_x) * mask.unsqueeze(-1)
        ctx, _ = self.context(event_h)
        workflow_event = self.out(ctx) * mask.unsqueeze(-1)
        pooled = masked_mean(workflow_event, mask, dim=1)
        return {
            "workflow_event": workflow_event,
            "workflow_pooled": pooled,
            "gap_logits_w": self.gap_head(pooled),
            "mark_logits_w": self.mark_head(pooled),
            "pending_logits_w": self.pending_head(pooled).squeeze(-1),
            "routine_logits_w": self.routine_head(pooled).squeeze(-1),
        }


class ColliderResponsibilityGate(nn.Module):
    """Explain-away gate: high responsibility means workflow explains the mark process."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mixer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        state_event: torch.Tensor,
        workflow_event: torch.Tensor,
        value_surprise: torch.Tensor,
        workflow_regular: torch.Tensor,
        event_mask: torch.Tensor,
        eps: float = 0.05,
    ) -> dict:
        gate_x = torch.cat(
            [
                state_event,
                workflow_event,
                value_surprise.unsqueeze(-1),
                workflow_regular.unsqueeze(-1),
            ],
            dim=-1,
        )
        responsibility = torch.sigmoid(self.net(gate_x).squeeze(-1)) * event_mask
        coordinate_correction = self.mixer(torch.cat([state_event, workflow_event.detach()], dim=-1))
        sealed = state_event * (1.0 - responsibility.detach().unsqueeze(-1))
        sealed = sealed + eps * coordinate_correction * event_mask.unsqueeze(-1)
        return {"responsibility": responsibility, "sealed_event": sealed}


class ColliderMarkedHeads(nn.Module):
    """ORA-style marked-event heads with state/workflow mixture responsibilities."""

    def __init__(self, hidden_dim: int, num_vars: int, num_gap_buckets: int):
        super().__init__()
        self.state_gap = nn.Linear(hidden_dim, num_gap_buckets)
        self.state_mark = nn.Linear(hidden_dim, num_vars)
        self.state_pending = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_pooled: torch.Tensor,
        workflow_out: dict,
        mean_resp: torch.Tensor,
    ) -> dict:
        gate = mean_resp.unsqueeze(-1)
        gap_s = self.state_gap(state_pooled)
        mark_s = self.state_mark(state_pooled)
        pending_s = self.state_pending(state_pooled).squeeze(-1)

        gap_logits = (1.0 - gate) * gap_s + gate * workflow_out["gap_logits_w"]
        mark_logits = (1.0 - gate) * mark_s + gate * workflow_out["mark_logits_w"]
        pending_logits = (1.0 - mean_resp) * pending_s + mean_resp * workflow_out["pending_logits_w"]
        return {
            "gap_logits": gap_logits,
            "mark_logits": mark_logits,
            "pending_logits": pending_logits,
            "state_gap_logits": gap_s,
            "state_mark_logits": mark_s,
            "state_pending_logits": pending_s,
        }


class DoColliderSealTransformer(nn.Module):
    """Sampling-policy robust classifier by sealing observed-event collider paths."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        hidden_dim: int,
        num_classes: int,
        num_gap_buckets: int = 8,
    ):
        super().__init__()
        self.pathology = PathologyParentStream(num_vars, num_bins, hidden_dim)
        self.workflow = WorkflowParentStream(num_vars, hidden_dim, num_gap_buckets)
        self.gate = ColliderResponsibilityGate(hidden_dim)
        self.marked = ColliderMarkedHeads(hidden_dim, num_vars, num_gap_buckets)
        self.state_backbone = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.state_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_bins = num_bins

    def encode(self, batch: dict, workflow_override: dict | None = None) -> dict:
        state = self.pathology(batch)
        workflow = self.workflow(batch) if workflow_override is None else workflow_override
        value_surprise = batch.get("value_surprise", torch.zeros_like(batch["event_mask"]))
        workflow_regular = batch.get("workflow_regular", torch.zeros_like(batch["event_mask"]))
        gate = self.gate(
            state["state_event"],
            workflow["workflow_event"],
            value_surprise,
            workflow_regular,
            batch["event_mask"],
        )
        seq, _ = self.state_backbone(gate["sealed_event"])
        pooled = masked_mean(self.state_proj(seq), batch["event_mask"], dim=1)
        logits = self.classifier(pooled)
        mean_resp = masked_mean(gate["responsibility"], batch["event_mask"], dim=1)
        marked = self.marked(state["state_pooled"], workflow, mean_resp)
        return {**state, **workflow, **gate, **marked, "state_summary": pooled, "logits": logits, "mean_resp": mean_resp}

    def marked_event_loss(self, out: dict, batch: dict) -> torch.Tensor:
        gap_loss = F.cross_entropy(out["gap_logits"], batch["next_gap_bucket"])
        mark_loss = F.cross_entropy(out["mark_logits"], batch["next_mark_id"])
        pending_target = batch.get("pending_target", torch.zeros_like(out["pending_logits"]))
        pending_loss = F.binary_cross_entropy_with_logits(out["pending_logits"], pending_target.float())
        return gap_loss + mark_loss + pending_loss

    def workflow_absorption_loss(self, out: dict, batch: dict) -> torch.Tensor:
        gap_loss = F.cross_entropy(out["gap_logits_w"], batch["next_gap_bucket"])
        mark_loss = F.cross_entropy(out["mark_logits_w"], batch["next_mark_id"])
        pending_target = batch.get("pending_target", torch.zeros_like(out["pending_logits_w"]))
        routine_target = batch.get("routine_deviation_target", torch.zeros_like(out["routine_logits_w"]))
        return (
            gap_loss
            + mark_loss
            + F.binary_cross_entropy_with_logits(out["pending_logits_w"], pending_target.float())
            + F.binary_cross_entropy_with_logits(out["routine_logits_w"], routine_target.float())
        )

    def responsibility_loss(self, out: dict, batch: dict) -> torch.Tensor:
        value_surprise = batch.get("value_surprise", torch.zeros_like(batch["event_mask"]))
        workflow_regular = batch.get("workflow_regular", torch.zeros_like(batch["event_mask"]))
        workflow_target = torch.sigmoid(workflow_regular - value_surprise).detach()
        raw = F.binary_cross_entropy(out["responsibility"], workflow_target, reduction="none")
        return (raw * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

    def state_value_sufficiency_loss(self, out: dict, batch: dict) -> torch.Tensor:
        if "pathology_bin_summary" in batch:
            target = batch["pathology_bin_summary"].clamp(0, self.num_bins - 1)
        else:
            target = out["bin_prob"].sum(dim=1).argmax(dim=-1).clamp(0, self.num_bins - 1)
        bin_loss = F.cross_entropy(out["bin_logits"], target)
        if "value_trend_summary" in batch:
            trend_loss = F.smooth_l1_loss(out["trend_hat"], batch["value_trend_summary"].float())
        else:
            trend_loss = torch.zeros((), device=out["logits"].device)
        return bin_loss + trend_loss

    def path_specific_seal_loss(self, batch: dict, factual: dict, eps_de: float = 0.05) -> torch.Tensor:
        swaps = batch.get("workflow_swap_bank", [])
        if not swaps:
            return torch.zeros((), device=factual["logits"].device)
        labels = batch["labels"]
        factual_margin = true_margin(factual["logits"], labels).detach()
        losses = []
        for swap_batch in swaps:
            swap_workflow = self.workflow(swap_batch)
            swapped = self.encode(batch, workflow_override=swap_workflow)
            swapped_margin = true_margin(swapped["logits"], labels)
            direct_effect = (swapped_margin - factual_margin).abs()
            losses.append(F.relu(direct_effect - eps_de).pow(2).mean())
        return torch.stack(losses).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_mark: float = 0.25,
        lambda_resp: float = 0.20,
        lambda_do: float = 0.30,
        lambda_val: float = 0.10,
        lambda_wf: float = 0.15,
    ) -> dict:
        out = self.encode(batch)
        cls_loss = F.cross_entropy(out["logits"], batch["labels"])
        mark_loss = self.marked_event_loss(out, batch)
        resp_loss = self.responsibility_loss(out, batch)
        seal_loss = self.path_specific_seal_loss(batch, out)
        value_loss = self.state_value_sufficiency_loss(out, batch)
        wf_loss = self.workflow_absorption_loss(out, batch)
        total = (
            cls_loss
            + lambda_mark * mark_loss
            + lambda_resp * resp_loss
            + lambda_do * seal_loss
            + lambda_val * value_loss
            + lambda_wf * wf_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "collided_marked_event_loss": mark_loss.detach(),
            "responsibility_explain_away_loss": resp_loss.detach(),
            "path_specific_seal_loss": seal_loss.detach(),
            "state_value_sufficiency_loss": value_loss.detach(),
            "workflow_absorption_loss": wf_loss.detach(),
            "mean_workflow_responsibility": out["mean_resp"].mean().detach(),
        }
```

## 4. ColliderSealCollator 草稿

```python
import torch


@torch.no_grad()
def build_collider_seal_batch(batch: dict, num_vars: int, num_gap_buckets: int = 8) -> dict:
    """Build ORA-style targets plus workflow swaps for collider sealing.

    Workflow swaps are used to estimate direct workflow path effects; they are
    not contrastive positives and are not used for logits consistency.
    """

    out = dict(batch)
    out.update(build_marked_event_targets(batch, num_vars, num_gap_buckets))
    out.update(build_explain_away_targets(batch))
    out["workflow_swap_bank"] = build_workflow_swaps(batch)
    return out


@torch.no_grad()
def build_marked_event_targets(batch: dict, num_vars: int, num_gap_buckets: int) -> dict:
    time = batch["event_time"]
    var_id = batch["event_var_id"].clamp(0, num_vars - 1)
    mask = batch["event_mask"]
    pending = batch.get("value_pending", torch.zeros_like(mask))
    routine = batch.get("routine_deviation", torch.zeros_like(mask))

    delta_t = torch.zeros_like(time)
    delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    last_idx = mask.sum(dim=1).long().sub(1).clamp_min(0)
    row = torch.arange(time.size(0), device=time.device)
    last_gap = delta_t[row, last_idx]
    gap_bucket = torch.bucketize(
        torch.log1p(last_gap),
        torch.linspace(0.0, 3.0, num_gap_buckets - 1, device=time.device),
    ).clamp(0, num_gap_buckets - 1)
    return {
        "next_gap_bucket": gap_bucket.long(),
        "next_mark_id": var_id[row, last_idx].long(),
        "pending_target": (pending * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0),
        "routine_deviation_target": (routine * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0),
    }


@torch.no_grad()
def build_explain_away_targets(batch: dict) -> dict:
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    pending = batch.get("value_pending", torch.zeros_like(mask))

    value_center = (value * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    value_scale = ((value - value_center).pow(2) * mask).sum(dim=1, keepdim=True)
    value_scale = (value_scale / mask.sum(dim=1, keepdim=True).clamp_min(1.0)).sqrt().clamp_min(1e-3)
    value_surprise = ((value - value_center).abs() / value_scale).clamp(0.0, 6.0) / 6.0

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    delta_t = torch.zeros_like(time)
    delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    close = (delta_t <= (delta_t * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_like = close * changed_var * mask
    routine_like = ((time / horizon * 6.0).round() / 6.0 - time / horizon).abs() < 0.03
    workflow_regular = (0.35 * panel_like + 0.35 * routine_like.to(mask.dtype) + 0.30 * pending).clamp(0.0, 1.0)

    return {"value_surprise": value_surprise * mask, "workflow_regular": workflow_regular * mask}


@torch.no_grad()
def build_workflow_swaps(batch: dict) -> list[dict]:
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_workflow(new_time, new_var, new_mask, pending=None, routine=None, panel=None):
        out = dict(batch)
        out["event_value"] = value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        if pending is not None:
            out["value_pending"] = pending
        if routine is not None:
            out["routine_deviation"] = routine
        if panel is not None:
            out["panel_indicator"] = panel
        out.pop("workflow_swap_bank", None)
        return out

    swaps = []

    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    routine_dev = (rounded_time - time).abs() / horizon
    swaps.append(clone_workflow(rounded_time, var_id, mask, routine=routine_dev))

    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_mask = torch.where(late > 0, mask, mask * alternating)
    swaps.append(clone_workflow(time, var_id, alarm_mask))

    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    panel = ((gap <= mean_gap.clamp_min(1e-6)) & (mask > 0)).to(mask.dtype)
    shuffled_var = var_id.roll(shifts=1, dims=0)
    swaps.append(clone_workflow(time, shuffled_var, mask, panel=panel))

    pending = mask
    swaps.append(clone_workflow(time, var_id, mask, pending=pending))

    exposure_mask = mask * (1.0 - 0.4 * (var_id % 2 == 1).to(mask.dtype))
    swaps.append(clone_workflow(time, var_id, exposure_mask))
    return swaps
```

## 5. 实验切入点

1. **Policy shift 构造**
   - `collider routine shift`：训练中心 routine rhythm 与高风险标签相关，测试中心 routine schedule 改变。
   - `ORA mark-policy shift`：训练中 next-mark 分布受医院 panel policy 支配，测试中变量覆盖和联测规则改变。
   - `pending collider shift`：训练中 value-pending 与危重程度相关，测试中 pending 主要由实验室延迟或拥堵造成。
   - `alarm-dense collider shift`：报警后密集复测在训练中心表示高风险，测试中心把同一流程扩展到普通患者。
   - `cross-center workflow swap`：借鉴 PULSE，在 MIMIC-IV / eICU / HiRID 风格的事件密度、变量 schema 和文书流程之间迁移。

2. **对比方法**
   - ORA-style marked time-to-event pretraining 后直接 fine-tune。
   - practice-invariant adversarial / IRM baseline。
   - Informative Irregularity workflow metadata 直接拼接分类。
   - SBRD-style shared benchmark + regime deviation。
   - 普通 irregular Transformer / GRU / CDE / Mamba / STAR-Set / VP-GNN。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、D-IVSP、DJRT、KPMA、DVNB、DSPP、DCPD、DFFH、DIPF、DRG-SFF、DBCC、DKCT、DRNC、DCOFF、DD-JEPA、DPSP、DNSA、DMWI 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy worst AUROC / AUPRC。
   - direct workflow path effect：workflow swap 后真实类 margin 的直接路径变化。
   - collider responsibility calibration：高 routine / pending / panel 事件是否被 gate 归因给 workflow，高 value-surprise 事件是否被归因给 state。
   - workflow absorption accuracy：workflow branch 对 gap、mark、pending、routine deviation 的预测能力。
   - state value sufficiency：sealed state 预测 pathology bins 和 value trend 的能力。
   - collider leakage alarm AUC：direct path effect 是否能预测跨政策错误。

4. **消融实验**
   - 去掉 responsibility gate，直接把 ORA marked-event embedding 输入分类器。
   - 去掉 `L_path_specific_seal`，检查 workflow swap 是否重新改变分类 margin。
   - 去掉 workflow absorption，只保留 state stream，验证 ORA 事件结构是否需要被专门 explain away。
   - 让 workflow pooled vector 直接进入 classifier 作为反例，验证院内性能可能升高但跨政策退化。
   - 将 workflow swaps 替换为随机 mask，验证收益来自结构化 collider 干预而非普通增强。
   - 固定 `r_i=0` 或 `r_i=1`，验证必须在事件级自适应地区分病理值贡献与工作流贡献。

## 6. 预期创新性

1. **从采样去偏转向 collider bias 封印**：历史方案多把 sampling policy 当成 nuisance、证据、集合、图、测验项或尺度；DCST 把 observed marked event 本身建成病理与工作流共同导致的 collider，并阻断条件化事件带来的伪路径。
2. **从 ORA 合并似然转向 ORA collider factorization**：保留 marked time / mark / pending / value 的结构化预训练，但用 responsibility gate 区分哪些 likelihood 由 workflow explain away，哪些由 pathology state 支撑。
3. **从 practice invariance 转向 path-specific direct-effect 控制**：不做简单 environment adversarial 或 IRM，而是在冻结 state path 后测量 workflow swap 对分类 margin 的直接路径效应。
4. **从元学习工作流适配转向单次因果路由**：不同于 DMWI 的 fast adapter 与 test-time inner loop，DCST 在事件级进行 collider responsibility routing，部署时不需要无标签 support adaptation。
5. **与现有反事实框架低侵入兼容**：counterfactual sampler 只需生成 workflow swap / collider ablation；value process、sampling process 与 ORA-style marked targets 都能复用。
6. **诊断直接指向采样偏移失败模式**：当模型失败时，可以报告是 routine rhythm、pending latency、panel mark 还是 cross-center exposure 打开了 workflow-to-label 的 collider shortcut。

## 7. 一句话投稿卖点

**DCST 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“条件化在 observed marked event 上打开了病理状态与工作流之间的 collider bias”的问题，通过 ORA-style marked-event collider factorization、事件级 explain-away responsibility gate 与 workflow-swap path-specific seal loss，让分类器只读取被封印后的 pathology state tokens，而把 routine rhythm、next-mark、panel、pending 和 cross-center workflow 差异留在观测生成支路，从因果图路径层面阻断采样策略捷径。**
