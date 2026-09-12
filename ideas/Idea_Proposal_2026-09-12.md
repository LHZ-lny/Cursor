# Title: Do-Meta Workflow Immunization：面向采样策略偏移的快慢参数免疫分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取 `paper_daily.md` 最新段落，重点纳入：
  - **ORA / Marked Time-to-Event for Structured EHR Foundation Models**：把结构化 EHR 预训练从 next-token 推进到事件时间、事件标记与连续测量的联合建模。
  - **Informative Irregularity as a Diagnostic for Model Robustness**：把 routine-rhythm deviation、workflow metadata 与 irregularity regime 作为鲁棒性分层诊断信号。
  - **SBRD / Shared-Benchmark Regime Decomposition**：把共享状态基线与约束/流程 regime deviation 分开建模。
  - 同时参考近期 **MATA-Former / SIICU**、**Cross-Representation Benchmarking**、**CauKer / PULSE / TCF / JETS / TRIAGE** 等已记录机制，但不复用其历史提案中的主方法。
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
  - `ideas/Idea_Proposal_2026-08-26.md`
- 已读取自动化记忆 `MEMORIES.md`，纳入其中记录但当前工作区未落盘的历史 proposal 摘要与机制黑名单。

### 历史核心机制黑名单

为避免思维重合，本轮明确避开以下历史主机制：

1. learnable reference points / adaptive time encoding、频域掩码修复、prototype 约束、简单 policy adversarial / IRM。
2. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
3. 生理流-采样算子交换子、value/policy graph 分离、policy residual sink。
4. protocol tax / additive evidence market、posterior quotient、density ratio / doubly robust、policy-simplex randomized smoothing。
5. reconstruction error cartography、VQ clauses、optional-stopping martingale、censored topology、policy gauge、syndrome code、knockoff calendar。
6. evidential vacuity、information lattice、solver trace front-door、conformal sleeves、IV/control-function、Borda jury。
7. Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock、IRT-DIF、RG fixed point、Gaussian privacy cloak。
8. bitemporal curtain、clinical tomography、matched risk-set likelihood、CauKer orthogonal synthetic forge、JEPA 双辩裁判。
9. cross-representation PID prism、unique label quarantine、policy synergy suppression、Noether semantic action / policy work。

本提案选择新的正交切入点：**不把采样政策删除、投影、征税、证明、投票、校准、隐私化或做信息分解；而是把采样政策当成“必须快速适配、但不能写入慢速分类知识”的工作流任务。模型被拆成 slow pathology core 与 fast workflow adapter：fast adapter 用无标签的 ORA-style 标记时间/流程预测在内循环中吸收当前医院的采样制度；slow core 在外循环中学习无论 fast adapter 如何适配，都能基于病理值完成分类的免疫表示。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 `paper_daily.md` 暴露了一个此前历史方案没有直接占用的缝隙。

第一，**ORA** 强调 EHR 不应只做 next-token prediction，而应同时建模事件时间、事件标记和连续测量。这说明“什么时候测、测什么、测到什么值”确实是 EHR foundation model 的核心结构。但 ORA 的风险也很明显：如果把事件时间和标记放进统一 likelihood，模型可能同时学习病程状态和医院工作流，最终在下游分类中利用 workflow shortcut。

第二，**Informative Irregularity** 说明 routine rhythm deviation、非例行采样和 workflow metadata 对预测有增益，但这种增益不必然可迁移。它更像一个 warning：采样不规则性既可能是病情信号，也可能是机构流程信号，因此不能简单拼进分类器。

第三，**SBRD** 的 shared benchmark + sparse regime deviation 提示我们：真实临床系统中存在一个相对稳定的 patient-state benchmark，也存在由资源、协议和流程造成的 regime wedge。对于非规则采样分类，采样策略就是一种 workflow regime；关键不是把 regime 信息完全删除，而是让它被局部、快速、可替换的参数吸收，避免沉淀为全局分类规则。

**Do-Meta Workflow Immunization (DMWI)** 的核心直觉是：

> 跨医院采样偏移像“每到一个新中心就要重新学会它的记录语法”。记录语法应该由 fast adapter 在几步无标签内循环中临时适配；真正用于分类的病理语义应该由 slow core 承载，并通过外循环元学习对这些 fast workflow adaptation 免疫。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 升级为 **slow pathology core**，只慢速学习病理状态。
- sampling process 升级为 **fast workflow adapter**，只在内循环学习当前 policy 的事件标记、时间桶、pending 与 routine deviation。
- counterfactual intervention 不生成一致性 pair、不做风险方差、不做平滑/保形/证明/隐私；它生成 **meta-episodes**，让同一类病理状态经历不同 workflow support/query 划分。
- 分类头只读 slow core 经 fast adapter 轻量调制后的状态；训练目标不是让多策略 logits 一样，而是让分类器在“fast adapter 已适配任何可部署工作流”之后仍能正确。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：Slow Pathology Core + Fast Workflow Adapter

DMWI 把模型参数按更新速度而不是按特征类型拆开：

1. **Slow Pathology Core `theta_s`**
   - 输入观测值、变量、相对时间、测量质量和 TCF-style pathology bin。
   - 输出 patient-state representation `h_state`。
   - 只在外循环更新，目标是跨 workflow episode 共享。

2. **Fast Workflow Adapter `phi_f`**
   - 输入采样坐标摘要：变量覆盖、gap 分布、routine-rhythm deviation、panel-like 共现、pending 率、文本/记录密度等。
   - 输出低幅度 FiLM / LoRA 调制量 `(gamma, beta)`，用于让 slow core 的表示适配当前工作流的观测语法。
   - 只在内循环用无标签 workflow objective 快速更新；不直接输出类别 logits，不接收 label。

3. **Workflow-Adapted Classifier**
   - 分类前只允许 fast adapter 以小幅、可约束的方式调制 `h_state`：

```text
z = h_state * (1 + epsilon * tanh(gamma_phi)) + epsilon * beta_phi
logits = Classifier(z)
```

其中 `epsilon` 很小，保证 fast adapter 学到的是“如何解释当前工作流下的观测坐标”，而不是替代 slow core 做分类。

### 2.2 改 Dataloader：Workflow Meta-Episode Collator

新增 `WorkflowEpisodeCollator`，每个 batch 被组织成 meta-episode：

1. `support_policy_batch`
   - 无标签或弱标签，仅包含当前 policy / center / workflow 的观测坐标和值可见性。
   - 用于内循环更新 fast adapter。

2. `query_value_batch`
   - 带标签，包含需要分类的病理值序列。
   - 用于外循环评估 slow core 在该 fast adapter 下是否仍正确。

3. `counterfactual_support_bank`
   - 由现有反事实采样模块生成不同 workflow support：
     - `routine_round_support`
     - `alarm_dense_support`
     - `panel_split_support`
     - `value_pending_support`
     - `cross_center_exposure_support`

4. `workflow_targets`
   - ORA-style 但避免 hazard 化：预测下一事件的 **离散 gap bucket**、下一标记变量 `mark_id`、是否 pending、是否偏离本地 routine rhythm。
   - 这些 target 只训练 fast adapter，不进入分类头。

关键区别：反事实模块不再构造“同一轨迹多视图一致性”；它构造的是“不同工作流语法的 support set”，用于测试 slow classifier 是否会被 fast workflow adaptation 带偏。

### 2.3 改 Loss：从去偏约束转向 Bilevel Immunization

总目标：

```text
L = L_meta_cls
  + lambda_wf   * L_workflow_inner
  + lambda_swap * L_adapter_swap_query
  + lambda_amp  * L_adapter_amplitude
  + lambda_path * L_pathology_semantic_anchor
```

#### A. Inner Workflow Adaptation `L_workflow_inner`

在 support batch 上，仅更新 fast adapter：

```text
phi'_f = phi_f - alpha * grad_phi L_workflow_inner
```

`L_workflow_inner` 包括离散 gap bucket、event mark、pending flag 和 routine deviation 的预测：

```text
L_workflow_inner =
  CE(gap_bucket_hat, gap_bucket)
  + CE(mark_hat, next_mark)
  + BCE(pending_hat, pending)
  + BCE(routine_dev_hat, routine_deviation)
```

这吸收 ORA 对事件时间/标记结构的重视，但不估计连续危险率、不把事件时间 likelihood 当成分类证据。

#### B. Meta Classification `L_meta_cls`

用内循环更新后的 fast adapter 在 query batch 上分类：

```text
L_meta_cls = CE(Classifier(SlowCore(query; phi'_f)), y_query)
```

外循环更新 slow core、classifier 和 adapter 初始化。若 slow core 把某个训练政策 shortcut 学成慢速知识，那么换一个 support policy 后 query loss 会升高；元学习会迫使慢速知识更接近病理状态。

#### C. Adapter Swap Query `L_adapter_swap_query`

对同一 query，分别用来自不同 counterfactual support 的 fast adapter：

```text
phi'_r = Adapt(phi_f, support_policy_r)
L_adapter_swap_query = mean_r CE(Classifier(SlowCore(query; phi'_r)), y_query)
```

这不是 logits consistency：不同 adapter 下 logits 可以不同、置信度可以不同。唯一要求是无论当前中心工作流如何被 fast adapter 吸收，真实标签仍能被 slow pathology state 支撑。

#### D. Adapter Amplitude `L_adapter_amplitude`

为了防止 fast adapter 变成隐形分类器，限制其调制能量：

```text
L_adapter_amplitude =
  mean(||gamma||_2^2 + ||beta||_2^2)
  + relu(||z - h_state|| / ||h_state|| - rho_max)^2
```

这不同于 protocol tax、privacy noise 或 policy residual sink；它只是快慢参数分工中的幅度护栏。

#### E. Pathology Semantic Anchor `L_pathology_semantic_anchor`

借鉴 TCF / MATA 的病理语义，但不走 proof、poset、IRT、Noether 或 PID 路线。slow core 额外预测 pathology bin summary：

```text
L_pathology_semantic_anchor =
  CE(bin_summary_hat, pathology_bin_summary)
```

作用是让 slow core 的慢速知识锚定在观测值病理语义，而不是在工作流节律。

### 2.4 推理阶段

给定一个新医院 / 新设备：

1. 使用最近一段无标签事件流作为 support，内循环快速适配 fast workflow adapter。
2. 冻结 slow pathology core 和 classifier。
3. 对待分类样本用适配后的 adapter 输出 logits。
4. 同时报告：
   - `workflow_adaptation_loss`：新中心工作流是否超出训练支持；
   - `adapter_amplitude`：当前预测是否需要过强工作流调制；
   - `adapter_swap_sensitivity`：用多个标准 support 适配后 query CE 的离散程度。

高 `adapter_amplitude` 或高 `swap_sensitivity` 表示该预测可能依赖工作流语法，建议进入补采样或人工复核。

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def summarize_workflow(batch: dict, num_vars: int) -> torch.Tensor:
    """Observation-coordinate summary used by the fast adapter only."""

    time = batch["event_time"]
    var_id = batch["event_var_id"].clamp(0, num_vars - 1)
    mask = batch["event_mask"]
    pending = batch.get("value_pending", torch.zeros_like(mask))

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon
    delta_t = torch.zeros_like(time)
    delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

    var_rate = F.one_hot(var_id, num_vars).to(time.dtype) * mask.unsqueeze(-1)
    var_rate = var_rate.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    early = masked_mean((time_norm <= 0.33).to(time.dtype), mask, dim=1).unsqueeze(-1)
    middle = masked_mean(((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype), mask, dim=1).unsqueeze(-1)
    late = masked_mean((time_norm > 0.66).to(time.dtype), mask, dim=1).unsqueeze(-1)
    mean_gap = masked_mean(torch.log1p(delta_t), mask, dim=1).unsqueeze(-1)

    close = (delta_t <= masked_mean(delta_t, mask, dim=1).unsqueeze(-1).clamp_min(1e-6)).to(time.dtype)
    var_change = torch.zeros_like(mask)
    var_change[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(time.dtype)
    panel_like = masked_mean(close * var_change, mask, dim=1).unsqueeze(-1)
    pending_rate = masked_mean(pending, mask, dim=1).unsqueeze(-1)
    routine_deviation = batch.get("routine_deviation", torch.zeros_like(mask))
    routine_dev_rate = masked_mean(routine_deviation, mask, dim=1).unsqueeze(-1)

    return torch.cat(
        [var_rate, early, middle, late, mean_gap, panel_like, pending_rate, routine_dev_rate],
        dim=-1,
    )


class SlowPathologyCore(nn.Module):
    """Slow parameters: value/pathology representation shared across workflows."""

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
        self.to_state = nn.Linear(2 * hidden_dim, hidden_dim)
        self.pathology_head = nn.Linear(hidden_dim, num_bins)

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

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        event_x = torch.cat(
            [
                self.var_embed(var_id),
                bin_prob,
                torch.log1p(delta_t).unsqueeze(-1),
                quality.unsqueeze(-1),
                mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * mask.unsqueeze(-1)
        ctx, _ = self.context(event_h)
        pooled = masked_mean(self.to_state(ctx), mask, dim=1)
        return {
            "state": pooled,
            "pathology_logits": self.pathology_head(pooled),
            "bin_prob": bin_prob,
        }


class FastWorkflowAdapter(nn.Module):
    """Fast parameters: learn the local workflow grammar, then lightly modulate state."""

    def __init__(self, num_vars: int, hidden_dim: int, num_gap_buckets: int):
        super().__init__()
        self.num_vars = num_vars
        self.summary_dim = num_vars + 7
        self.net = nn.Sequential(
            nn.Linear(self.summary_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.to_gamma_beta = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.gap_head = nn.Linear(hidden_dim, num_gap_buckets)
        self.mark_head = nn.Linear(hidden_dim, num_vars)
        self.pending_head = nn.Linear(hidden_dim, 1)
        self.routine_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        summary = summarize_workflow(batch, self.num_vars)
        h = self.net(summary)
        gamma, beta = self.to_gamma_beta(h).chunk(2, dim=-1)
        return {
            "gamma": gamma,
            "beta": beta,
            "gap_logits": self.gap_head(h),
            "mark_logits": self.mark_head(h),
            "pending_logits": self.pending_head(h).squeeze(-1),
            "routine_logits": self.routine_head(h).squeeze(-1),
        }


def workflow_inner_loss(adapter_out: dict, support: dict) -> torch.Tensor:
    """Unlabeled/weak workflow objective for fast adaptation."""

    gap_loss = F.cross_entropy(adapter_out["gap_logits"], support["next_gap_bucket"])
    mark_loss = F.cross_entropy(adapter_out["mark_logits"], support["next_mark_id"])
    pending_target = support.get("pending_target", torch.zeros_like(adapter_out["pending_logits"]))
    routine_target = support.get("routine_deviation_target", torch.zeros_like(adapter_out["routine_logits"]))
    pending_loss = F.binary_cross_entropy_with_logits(adapter_out["pending_logits"], pending_target.float())
    routine_loss = F.binary_cross_entropy_with_logits(adapter_out["routine_logits"], routine_target.float())
    return gap_loss + mark_loss + pending_loss + routine_loss


class DoMetaWorkflowImmunization(nn.Module):
    """Bilevel classifier that keeps workflow syntax fast and pathology knowledge slow."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        hidden_dim: int,
        num_classes: int,
        num_gap_buckets: int = 8,
        adapter_eps: float = 0.10,
    ):
        super().__init__()
        self.core = SlowPathologyCore(num_vars, num_bins, hidden_dim)
        self.adapter = FastWorkflowAdapter(num_vars, hidden_dim, num_gap_buckets)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.adapter_eps = adapter_eps
        self.num_bins = num_bins

    def adapt_fast(
        self,
        support: dict,
        steps: int = 2,
        inner_lr: float = 0.05,
    ) -> tuple[OrderedDict[str, torch.Tensor], torch.Tensor]:
        fast_params = OrderedDict((name, param) for name, param in self.adapter.named_parameters())
        losses = []
        for _ in range(steps):
            adapter_out = functional_call(self.adapter, fast_params, (support,))
            loss = workflow_inner_loss(adapter_out, support)
            grads = torch.autograd.grad(loss, tuple(fast_params.values()), create_graph=True)
            fast_params = OrderedDict(
                (name, param - inner_lr * grad)
                for (name, param), grad in zip(fast_params.items(), grads)
            )
            losses.append(loss)
        return fast_params, torch.stack(losses).mean()

    def classify_with_adapter(
        self,
        query: dict,
        fast_params: OrderedDict[str, torch.Tensor] | None = None,
    ) -> dict:
        core_out = self.core(query)
        if fast_params is None:
            adapter_out = self.adapter(query)
        else:
            adapter_out = functional_call(self.adapter, fast_params, (query,))

        gamma = torch.tanh(adapter_out["gamma"])
        beta = torch.tanh(adapter_out["beta"])
        z = core_out["state"] * (1.0 + self.adapter_eps * gamma) + self.adapter_eps * beta
        logits = self.classifier(z)
        amp = (z - core_out["state"]).norm(dim=-1) / core_out["state"].norm(dim=-1).clamp_min(1e-6)
        return {**core_out, **adapter_out, "adapted_state": z, "logits": logits, "adapter_amplitude": amp}

    def pathology_anchor_loss(self, out: dict, query: dict) -> torch.Tensor:
        if "pathology_bin_summary" not in query:
            # Fallback: use dominant observed pathology bin as weak target.
            bin_mass = out["bin_prob"].sum(dim=1)
            target = bin_mass.argmax(dim=-1).clamp(0, self.num_bins - 1)
        else:
            target = query["pathology_bin_summary"].clamp(0, self.num_bins - 1)
        return F.cross_entropy(out["pathology_logits"], target)

    def adapter_amplitude_loss(self, out: dict, rho_max: float = 0.20) -> torch.Tensor:
        raw_amp = out["gamma"].pow(2).mean() + out["beta"].pow(2).mean()
        rel_amp = F.relu(out["adapter_amplitude"] - rho_max).pow(2).mean()
        return 0.01 * raw_amp + rel_amp

    def training_loss(
        self,
        episode: dict,
        lambda_wf: float = 0.20,
        lambda_swap: float = 0.50,
        lambda_amp: float = 0.08,
        lambda_path: float = 0.10,
    ) -> dict:
        support = episode["support_policy_batch"]
        query = episode["query_value_batch"]
        labels = query["labels"]

        fast_params, inner_loss = self.adapt_fast(support)
        out = self.classify_with_adapter(query, fast_params)
        meta_cls = F.cross_entropy(out["logits"], labels)

        swap_losses = []
        for alt_support in episode.get("counterfactual_support_bank", []):
            alt_params, _ = self.adapt_fast(alt_support)
            alt_out = self.classify_with_adapter(query, alt_params)
            swap_losses.append(F.cross_entropy(alt_out["logits"], labels))
        if swap_losses:
            swap_loss = torch.stack(swap_losses).mean()
        else:
            swap_loss = torch.zeros((), device=labels.device)

        amp_loss = self.adapter_amplitude_loss(out)
        path_loss = self.pathology_anchor_loss(out, query)
        total = (
            meta_cls
            + lambda_wf * inner_loss
            + lambda_swap * swap_loss
            + lambda_amp * amp_loss
            + lambda_path * path_loss
        )
        return {
            "loss": total,
            "meta_cls_loss": meta_cls.detach(),
            "workflow_inner_loss": inner_loss.detach(),
            "adapter_swap_query_loss": swap_loss.detach(),
            "adapter_amplitude_loss": amp_loss.detach(),
            "pathology_anchor_loss": path_loss.detach(),
            "mean_adapter_amplitude": out["adapter_amplitude"].mean().detach(),
        }
```

## 4. WorkflowEpisodeCollator 草稿

```python
import torch


@torch.no_grad()
def build_workflow_meta_episode(batch: dict, num_vars: int, num_gap_buckets: int = 8) -> dict:
    """Create support/query workflow episodes for bilevel immunization.

    Counterfactual supports are not contrastive positives and are not used for
    logits consistency. They teach the fast adapter multiple workflow grammars.
    """

    support = dict(batch)
    support.update(build_workflow_targets(batch, num_vars, num_gap_buckets))

    query = dict(batch)
    cf_supports = []
    for cf in build_counterfactual_workflow_supports(batch):
        cf.update(build_workflow_targets(cf, num_vars, num_gap_buckets))
        cf_supports.append(cf)

    return {
        "support_policy_batch": support,
        "query_value_batch": query,
        "counterfactual_support_bank": cf_supports,
    }


@torch.no_grad()
def build_workflow_targets(batch: dict, num_vars: int, num_gap_buckets: int) -> dict:
    time = batch["event_time"]
    var_id = batch["event_var_id"].clamp(0, num_vars - 1)
    mask = batch["event_mask"]
    pending = batch.get("value_pending", torch.zeros_like(mask))
    routine_dev = batch.get("routine_deviation", torch.zeros_like(mask))

    delta_t = torch.zeros_like(time)
    delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    last_gap = masked_last(delta_t, mask)
    gap_bucket = torch.bucketize(
        torch.log1p(last_gap),
        torch.linspace(0.0, 3.0, num_gap_buckets - 1, device=time.device),
    ).clamp(0, num_gap_buckets - 1)

    last_mark = masked_last(var_id.to(time.dtype), mask).long().clamp(0, num_vars - 1)
    pending_target = (pending * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    routine_target = (routine_dev * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return {
        "next_gap_bucket": gap_bucket.long(),
        "next_mark_id": last_mark,
        "pending_target": pending_target,
        "routine_deviation_target": routine_target,
    }


@torch.no_grad()
def masked_last(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    idx = mask.sum(dim=1).long().sub(1).clamp_min(0)
    return x[torch.arange(x.size(0), device=x.device), idx]


@torch.no_grad()
def build_counterfactual_workflow_supports(batch: dict) -> list[dict]:
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, pending=None, routine_dev=None):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        if pending is not None:
            out["value_pending"] = pending
        if routine_dev is not None:
            out["routine_deviation"] = routine_dev
        return out

    supports = []

    # 1. Routine-round support: regularize timestamps into coarse clinical rounds.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    routine_dev = (rounded_time - time).abs() / horizon
    supports.append(clone_with(value * mask, rounded_time, var_id, mask, routine_dev=routine_dev))

    # 2. Alarm-dense support: retain late/alarm-like region, thin early routine events.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_mask = torch.where(late > 0, mask, mask * alternating)
    supports.append(clone_with(value * alarm_mask, time, var_id, alarm_mask))

    # 3. Panel-split support: weaken near-synchronous cross-variable observations.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_mask = mask * (1.0 - 0.5 * close * changed_var)
    supports.append(clone_with(value * panel_mask, time, var_id, panel_mask))

    # 4. Value-pending support: administration remains visible but values are unavailable.
    pending = mask
    supports.append(clone_with(torch.zeros_like(value), time, var_id, mask, pending=pending))

    # 5. Cross-center exposure support: simulate variable-schema coverage change.
    odd_var = (var_id % 2 == 1).to(mask.dtype)
    exposure_mask = mask * (1.0 - 0.4 * odd_var)
    supports.append(clone_with(value * exposure_mask, time, var_id, exposure_mask))

    return supports
```

## 5. 实验切入点

1. **Policy shift 构造**
   - `routine-rhythm shift`：训练中心按固定查房节律采样，测试中心转为事件触发采样。
   - `alarm-dense shift`：训练中告警后密集复测强关联高风险，测试中反转或稀疏化。
   - `panel-split shift`：训练中心同步 panel，测试中心异步拆单。
   - `value-pending shift`：保留施测事件但隐藏部分值，测试模型是否仍把“已下单”当类别证据。
   - `cross-center workflow shift`：借鉴 PULSE，在 MIMIC-IV / eICU / HiRID 风格的变量覆盖和记录流程间迁移。

2. **对比方法**
   - ORA-style marked time/event pretraining 直接 fine-tune。
   - Informative Irregularity workflow metadata 直接拼接分类。
   - SBRD-style regime decomposition 但无 bilevel fast adapter。
   - 普通 test-time adaptation / self-supervised adaptation。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DSPP、DCPD、DIPF、DRG-SFF、DPPC、DCOFF、DD-JEPA、DPSP、DNSA 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy worst AUROC / AUPRC。
   - fast adaptation loss：新中心工作流是否能被少量无标签 support 吸收。
   - adapter amplitude：预测是否依赖过强工作流调制。
   - adapter swap sensitivity：同一 query 在不同 workflow support 适配后，真实标签 CE 的离散程度。
   - slow-state workflow probe AUC：从 slow state 中预测 workflow regime 的能力，越低越好。
   - shortcut relapse after fine-tuning：真实数据微调后，adapter swap sensitivity 是否重新升高。

4. **消融实验**
   - 去掉 bilevel adaptation，直接联合训练 workflow head，验证 fast/slow 更新速度分离是否必要。
   - 去掉 `L_adapter_swap_query`，检查 slow core 是否只适配训练中心 support。
   - 让 workflow summary 直接进入 classifier 作为反例，验证院内性能可能上升但跨政策退化。
   - 固定 fast adapter 不更新，验证鲁棒性不是简单 FiLM 参数量带来的。
   - 把 counterfactual supports 替换为随机 mask supports，验证收益来自结构化工作流语法而非普通增强。
   - 扫描内循环步数与 support 大小，评估测试时无标签适配的计算/鲁棒性 trade-off。

## 6. 预期创新性

1. **从采样去偏转向采样工作流免疫**：不是删除、正交、征税、隐私化或证明采样信息，而是让采样制度成为 fast adapter 的临时语法任务。
2. **从静态分支解耦转向快慢参数解耦**：历史方案多按 state/policy 特征分支拆分；DMWI 按学习速度拆分，slow core 学病理，fast adapter 学工作流。
3. **从 ORA 合并似然转向内循环工作流吸收**：吸收 marked time-to-event 的结构化事件建模，但只用于 fast adapter 无标签适配，不把事件时间/标记 likelihood 直接变成分类证据。
4. **从 workflow metadata 诊断转向 test-time adaptation**：借鉴 Informative Irregularity 的 regime metadata，但不把 metadata 拼入分类器，而是用于适配和告警。
5. **从 SBRD regime wedge 转向 bilevel immunization**：保留 shared benchmark / regime deviation 思想，但用 support-query 元学习检验 slow classifier 是否对不同 workflow adaptation 免疫。
6. **与现有反事实框架低侵入兼容**：counterfactual sampler 只需生成 workflow supports；不需要 hazard、density ratio、posterior quotient、knockoff、conformal、IRT、RG、privacy、PID、JEPA 或 Noether 结构。

## 7. 一句话投稿卖点

**DMWI 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“医院/设备工作流语法应被 fast adapter 快速吸收，而不应沉淀进 slow pathology classifier”的元学习问题，通过 ORA-style workflow support objective、counterfactual workflow meta-episodes、adapter-swap query CE 与幅度护栏，让模型在测试时能无标签适配新采样制度，同时保持慢速病理表示对 routine/alarm、panel、pending 和跨中心 exposure 变化的免疫性。**
