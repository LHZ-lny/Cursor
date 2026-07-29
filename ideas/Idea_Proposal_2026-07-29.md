# Title: Do-IV Structural Purifier：面向采样策略偏移的反事实工具变量结构净化器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*work*.md` 与相关 summary/work 命名：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md` 以及 `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`。
- 已读取近期论文记录：
  - `paper_daily.md`
  - 已检索到 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`；其中最新机制已在 `paper_daily.md` 兼容入口中完整纳入。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：2026-06-17、2026-06-20、2026-06-24、2026-06-27、2026-07-15 至 2026-07-27 的提案摘要。

### 历史核心机制黑名单

为避免与历史提案发生思维重合，本提案明确避开以下机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder、频域掩码修复或 raw/difference/frequency 三视图鲁棒化。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、hazard-driven resampling、分类梯度与采样 score 零空间正交、do-risk variance。
8. 生理流算子与采样算子交换子、图交换子、policy residual sink。
9. additive evidence market、protocol tax、证据预算、边际证据审计。
10. 模型空间 posterior quotient、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、VQ semantic clauses、ANOVA/HSIC redaction。
12. policy-simplex randomized smoothing、certified policy radius、Dirichlet/logit-normal do-sampler。
13. 采样测度 density ratio、doubly robust correction、influence-bound regularization。
14. optional-stopping martingale queries、standardized innovation、停时矩约束。
15. censored excursion topology、censored persistence interval、topology capsules。
16. policy-gauge frame、horizontal transport、chart span supervision。
17. policy-only negative film、shadow eraser/stencil、shadow nullification。
18. latent packet codeword、parity-check、syndrome locator、packet repair decoder。
19. conditional knockoff calendar、soft knockoff-FDR、swap symmetry。
20. observability witness、counterfactual observability probe、observability-routed classification。
21. subjective-logic / Dirichlet evidential shield、policy-induced vacuity、class-wise evidence discount。
22. observation-set policy lattice、meet/join visibility masks、monotone/submodular margin contracts、shortcut curvature。
23. solver trace mediator、front-door trace-standardized readout、NFE/roughness do-slope。
24. measurement-action bisimulation、policy-word signature renormalization、thermodynamic free energy、Sklar copula stripping、triage queue debt、Sinkhorn detail canonicalization、MDL episode transducer、causal sheaf gluing、trigger hysteresis、control barrier certificates、regret escrow、principal strata、conformal risk sleeves。
25. 单纯把 STAR-Set 的 temporal/variable attention bias 或 VP-GNN 的 variable/patch graph 拆成 state-policy 双分支并做一致性约束。

本提案选择新的正交切入点：**不删除采样信息，不把结构图拆成 state/policy 双图，不做一致性、认证、保形、证据不确定性、knockoff 或子模约束；而是把 STAR-Set / VP-GNN 暴露出的 attention bias、变量消息路径、patch 路由等结构量视为“内生结构处理变量”，再把反事实采样 recipe 视为只影响结构可见性的工具变量。模型通过控制函数净化掉结构量中可由采样工具变量解释的部分，分类头只消费 value state 与结构残差中的可识别状态成分。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中的两个前沿机制给了非常清晰的结构信号：

1. **STAR-Set Transformer** 用 temporal locality penalty 与 variable-type affinity 把事件集合重新注入时间邻域和变量兼容先验。它说明 attention bias 本身是一个可观测的结构通道：某变量多久还有效、哪些变量在局部时间窗内常共现，都会改变分类器看到的证据。
2. **VP-GNN** 用 variable-wise graph 与 patch-wise graph 同时建模异步变量依赖和多尺度时间片段。它说明采样政策偏移会沿着“变量消息传递”和“patch 路由”两条结构路径进入决策。

历史方案已经从很多角度处理这些结构偏移：把结构拆成 state/policy 双图、约束一致性、做保形校准、做信息格边际、做 solver trace 中介、做证据不确定性或负控审计。但这些思路都默认我们要么能定义稳定结构，要么能通过约束让结构变稳定。真实部署中的问题更像计量经济学中的“内生处理变量”：

> 模型学到的 attention bias、变量边、patch 路由并不是纯粹生理状态，也不是纯粹采样政策；它们是由潜在病程、医院采样流程、设备预算和观测质量共同生成的内生结构变量。

如果直接把这些结构量用于分类，模型会把训练医院的联测习惯、局部时间尺度或告警后 patch 可见性当作标签证据。若简单丢掉结构量，又会浪费 STAR-Set / VP-GNN 证明有效的结构归纳偏置。**Do-IV Structural Purifier (D-IVSP)** 的核心动机是：

> 不问“这条边是否属于 state graph 或 policy graph”，而问“这条结构证据中有多少可被我们主动施加的反事实采样工具变量解释”。可由工具变量解释的结构变化被视为采样诱导成分；分类头只使用控制函数残差中仍与 value state 协同的部分。

这与当前“采样解耦/反事实干预”框架高度兼容：

- value process 继续编码观测值驱动的病程状态。
- sampling process 不进入分类头，而是产生反事实采样 recipe / instrument。
- counterfactual intervention 不再被用作一致性、平滑、保形、knockoff 或信息格构造；它只服务于 **第一阶段结构方程**，学习采样工具变量如何改变 attention bias、变量图与 patch 路由。
- 最终 readout 不是普通 `Classifier(h, structure)`，而是 `Classifier(h, residualized_structure)`，其中 residualized structure 是控制函数净化后的结构证据。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：输出内生结构处理变量，而不是 state/policy 双图

D-IVSP 可以包裹 STAR-Set、VP-GNN 或任意 irregular Transformer/GNN。基础 encoder 需要额外输出结构诊断向量 `s_obs`：

1. **Value state `h_value`**
   - 来自原有值路径，编码观测值、变量 id、时间戳和局部上下文。
   - 作为分类的主状态表示。

2. **Endogenous structure treatment `s_obs`**
   - 来自 STAR-Set：temporal bias energy、learned timescale usage、variable affinity attention mass。
   - 来自 VP-GNN：variable-wise edge energy、selective message passing gate、patch-wise route entropy、patch depth usage。
   - 这些结构量不被拆成 state graph / policy graph；它们被统一视为内生处理变量。

3. **Counterfactual instrument `z_iv`**
   - 由 sampling branch / collator 生成，只描述外部施加的采样 recipe，例如改变局部时间窗、变量预算、patch 可见性或 panel 同步机会。
   - `z_iv` 不进入分类头。
   - 在因果语义上，`z_iv` 被设计为影响 `s_obs`，但在给定 value state 后不应直接影响标签；这就是工具变量排除限制的近似。

### 2.2 改 Loss：从不变性约束转向 Counterfactual IV Control Function

总目标：

```text
L = L_cls
  + lambda_fs   * L_first_stage
  + lambda_cf   * L_control_function
  + lambda_orth * L_instrument_residual_moment
  + lambda_wk   * L_weak_instrument_guard
```

#### A. 第一阶段结构方程 `L_first_stage`

对同一条患者轨迹施加多个反事实采样工具变量 `z_r`，得到结构量 `s_r`。训练一个第一阶段网络预测：

```text
s_hat_r = g_phi(h_value_stopgrad, z_r)
```

损失为：

```text
L_first_stage = SmoothL1(s_hat_r, stopgrad(s_r))
```

直觉：第一阶段只学习“采样 recipe 如何移动结构量”，例如让 STAR-Set 的 temporal scale 偏向短窗、让 VP-GNN 的 patch 路由更浅、让某些变量边因联测机会增加而增强。

#### B. 控制函数净化 `L_control_function`

从事实结构量中减去可由采样工具变量解释的部分：

```text
u_struct = s_obs - s_hat_obs
```

分类头只消费 `h_value` 与 `u_struct`：

```text
logits = Classifier(h_value, u_struct)
L_cls = CE(logits, y)
```

为了让 `u_struct` 不塌缩成纯噪声，加入控制函数预测项：

```text
L_control_function = CE(Classifier(h_value, u_struct), y)
                   + beta * ||u_struct||_Huber
```

其中 Huber 项只是防止残差爆炸，不是信息瓶颈，也不是证据税。

#### C. 工具变量残差矩 `L_instrument_residual_moment`

若工具变量的可解释部分已经被第一阶段扣除，则分类残差不应再被工具变量线性预测：

```text
e_y = one_hot(y) - softmax(logits)
moment = mean_batch( z_iv * e_y )
L_instrument_residual_moment = ||moment||_F^2
```

这不是 adversarial loss：没有环境判别器，也不反转梯度；它是一个显式的正交矩条件，要求“剩余预测误差不再随外生采样工具变量系统偏移”。它也不是 DHN 的梯度零空间，因为不计算分类梯度与采样 score 的夹角。

#### D. 弱工具变量保护 `L_weak_instrument_guard`

若某个 recipe 完全不改变结构量，那么 IV 识别无效。训练时要求第一阶段具有足够解释力：

```text
first_stage_r2 = 1 - Var(s_r - s_hat_r) / Var(s_r)
L_weak_instrument_guard = relu(r_min - first_stage_r2)^2
```

这项不是鼓励模型依赖采样政策，而是防止使用无效工具变量导致的残差净化幻觉。

### 2.3 改 Dataloader：返回 Policy Instrument Bank

新增 `PolicyInstrumentCollator`，每个 batch 返回：

1. `factual_batch`：原始不规则事件。
2. `instrument_recipe_bank`：外生采样工具变量，例如：
   - `temporal_window_push`：压缩或拉伸局部时间窗，使 STAR-Set timescale 发生变化；
   - `variable_budget_rotate`：改变变量观测预算，使 variable affinity / selective passing 发生变化；
   - `patch_visibility_gate`：改变 VP-GNN patch 可见性与 patch route depth；
   - `panel_sync_toggle`：把异步变量局部同步或把同步 panel 拆开。
3. `cf_batch_bank`：每个 recipe 下的反事实观测事件。
4. `instrument_code_bank`：recipe 的低维 one-hot / continuous code。
5. `first_stage_mask`：标记哪些结构维度应被该工具变量显著移动，用于弱工具诊断。

关键区别：

- 不把 recipe code 输入分类头。
- 不要求多 recipe 的 logits 或 representation 一致。
- 不做 policy smoothing、conformal calibration、knockoff 交换、lattice meet/join、syndrome repair、evidential vacuity 或 solver trace 标准化。
- 只利用反事实采样干预估计“结构量中哪一部分由采样工具变量造成”。

### 2.4 推理阶段

给定测试样本：

1. base encoder 输出 `h_value` 与 `s_obs`。
2. sampling branch 根据当前观测结构输出事实 instrument summary `z_obs`，或使用零 recipe / routine recipe 作为参考。
3. 第一阶段网络预测 `s_hat_obs = g_phi(h_value, z_obs)`。
4. 结构残差：

```text
u_struct = s_obs - s_hat_obs
```

5. 分类头输出：

```text
logits = Classifier(h_value, u_struct)
```

部署诊断：

- `iv_strength`：当前样本附近第一阶段能否解释结构漂移。
- `struct_policy_explained_ratio`：结构量中可由采样工具变量解释的比例。
- `residual_structure_reliance`：分类头最终使用多少 residualized structure。
- `instrument_moment_gap`：测试环境中预测残差是否重新与 instrument 相关。

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


class StructureTraceHead(nn.Module):
    """Compress STAR-Set / VP-GNN diagnostics into endogenous structure treatments."""

    def __init__(self, raw_trace_dim: int, structure_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(raw_trace_dim, structure_dim),
            nn.SiLU(),
            nn.Linear(structure_dim, structure_dim),
        )

    def forward(self, model_out: dict) -> torch.Tensor:
        if "structure_trace" in model_out:
            raw_trace = model_out["structure_trace"]
        else:
            logits = model_out["logits"]
            prob = torch.softmax(logits, dim=-1)
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            top2 = prob.topk(2, dim=-1).values
            margin = top2[:, :1] - top2[:, 1:2]
            raw_trace = torch.cat([entropy, margin, prob.mean(dim=-1, keepdim=True)], dim=-1)
        return self.net(raw_trace)


class InstrumentEncoder(nn.Module):
    """Encode exogenous counterfactual sampling recipes for first-stage equations only."""

    def __init__(self, recipe_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(recipe_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, instrument_code: torch.Tensor) -> torch.Tensor:
        return self.net(instrument_code)


class FirstStageStructureEquation(nn.Module):
    """Predict the policy-movable part of structure diagnostics from IV recipes."""

    def __init__(self, value_dim: int, iv_hidden: int, structure_dim: int, hidden_dim: int):
        super().__init__()
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + iv_hidden, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, structure_dim),
        )

    def forward(self, value_state: torch.Tensor, iv_state: torch.Tensor) -> torch.Tensor:
        value_h = self.value_proj(value_state.detach())
        return self.net(torch.cat([value_h, iv_state], dim=-1))


class ControlFunctionClassifier(nn.Module):
    """Classify from value state plus residualized structure, never from raw IV codes."""

    def __init__(self, value_dim: int, structure_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(value_dim + structure_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, value_state: torch.Tensor, structure_residual: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([value_state, structure_residual], dim=-1))


def instrument_residual_moment_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    instrument_code: torch.Tensor,
) -> torch.Tensor:
    """Orthogonal moment: prediction residual should not be linearly explained by IV codes."""

    prob = torch.softmax(logits, dim=-1)
    target = F.one_hot(labels, logits.size(-1)).to(prob.dtype)
    residual = target - prob

    z = instrument_code - instrument_code.mean(dim=0, keepdim=True)
    e = residual - residual.mean(dim=0, keepdim=True)
    moment = torch.einsum("br,bc->rc", z, e) / max(logits.size(0), 1)
    return moment.pow(2).mean()


def first_stage_strength_loss(
    structure: torch.Tensor,
    predicted: torch.Tensor,
    min_r2: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Guard against weak counterfactual instruments."""

    residual_var = (structure.detach() - predicted).pow(2).mean(dim=0)
    total_var = structure.detach().var(dim=0, unbiased=False).clamp_min(1e-6)
    r2 = 1.0 - residual_var / total_var
    loss = F.relu(min_r2 - r2).pow(2).mean()
    return loss, r2.mean().detach()


class DoIVStructuralPurifier(nn.Module):
    """Wrap an irregular classifier with counterfactual-IV control-function purification."""

    def __init__(
        self,
        base_model: nn.Module,
        value_dim: int,
        raw_trace_dim: int,
        structure_dim: int,
        recipe_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.base_model = base_model
        self.structure_head = StructureTraceHead(raw_trace_dim, structure_dim)
        self.instrument = InstrumentEncoder(recipe_dim, hidden_dim)
        self.first_stage = FirstStageStructureEquation(
            value_dim=value_dim,
            iv_hidden=hidden_dim,
            structure_dim=structure_dim,
            hidden_dim=hidden_dim,
        )
        self.classifier = ControlFunctionClassifier(
            value_dim=value_dim,
            structure_dim=structure_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

    def encode_once(self, batch: dict, instrument_code: torch.Tensor | None = None) -> dict:
        out = self.base_model(batch)
        logits_base = out["logits"]
        value_state = out.get("value_state", logits_base)
        structure_obs = self.structure_head(out)

        if instrument_code is None:
            instrument_code = batch.get(
                "instrument_code",
                torch.zeros(value_state.size(0), batch["instrument_recipe_bank"].size(-1), device=value_state.device),
            )
        iv_state = self.instrument(instrument_code)
        structure_hat = self.first_stage(value_state, iv_state)
        structure_residual = structure_obs - structure_hat.detach()
        logits = self.classifier(value_state, structure_residual)
        return {
            **out,
            "value_state": value_state,
            "structure_obs": structure_obs,
            "structure_hat": structure_hat,
            "structure_residual": structure_residual,
            "logits_iv": logits,
            "instrument_code": instrument_code,
        }

    def training_loss(
        self,
        batch: dict,
        lambda_fs: float = 0.35,
        lambda_cf: float = 0.03,
        lambda_orth: float = 0.20,
        lambda_wk: float = 0.10,
    ) -> dict:
        labels = batch["labels"]
        factual = self.encode_once(batch, batch["instrument_code"])
        cls_loss = F.cross_entropy(factual["logits_iv"], labels)
        control_penalty = F.smooth_l1_loss(
            factual["structure_residual"],
            torch.zeros_like(factual["structure_residual"]),
            reduction="mean",
        )

        first_stage_losses = []
        strength_losses = []
        r2_values = []
        moment_losses = [
            instrument_residual_moment_loss(factual["logits_iv"], labels, batch["instrument_code"])
        ]

        for cf_batch, z_code in zip(batch["cf_batch_bank"], batch["instrument_recipe_bank"].unbind(dim=1)):
            cf_out = self.base_model(cf_batch)
            cf_value_state = cf_out.get("value_state", cf_out["logits"])
            cf_structure = self.structure_head(cf_out)

            iv_state = self.instrument(z_code)
            cf_pred = self.first_stage(cf_value_state, iv_state)
            first_stage_losses.append(F.smooth_l1_loss(cf_pred, cf_structure.detach()))

            wk_loss, r2 = first_stage_strength_loss(cf_structure, cf_pred)
            strength_losses.append(wk_loss)
            r2_values.append(r2)

            # Moment diagnostics under counterfactual instruments. This does not
            # force logits to match factual logits; it only removes residual IV bias.
            cf_residual = cf_structure - cf_pred.detach()
            cf_logits = self.classifier(cf_value_state, cf_residual)
            moment_losses.append(instrument_residual_moment_loss(cf_logits, labels, z_code))

        first_stage_loss = torch.stack(first_stage_losses).mean()
        moment_loss = torch.stack(moment_losses).mean()
        weak_loss = torch.stack(strength_losses).mean()
        mean_r2 = torch.stack(r2_values).mean() if r2_values else torch.zeros((), device=labels.device)

        total = (
            cls_loss
            + lambda_fs * first_stage_loss
            + lambda_cf * control_penalty
            + lambda_orth * moment_loss
            + lambda_wk * weak_loss
        )
        explained_ratio = (
            factual["structure_hat"].detach().pow(2).mean()
            / factual["structure_obs"].detach().pow(2).mean().clamp_min(1e-6)
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "first_stage_loss": first_stage_loss.detach(),
            "control_penalty": control_penalty.detach(),
            "instrument_moment_loss": moment_loss.detach(),
            "weak_instrument_loss": weak_loss.detach(),
            "mean_first_stage_r2": mean_r2.detach(),
            "struct_policy_explained_ratio": explained_ratio.detach(),
        }


@torch.no_grad()
def build_policy_instrument_bank(batch: dict, recipe_dim: int = 4) -> tuple[torch.Tensor, list[dict]]:
    """Create counterfactual sampling instruments and batches.

    The generated recipe codes are first-stage instruments only. They should not
    be concatenated into the final classifier input.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = value.shape
    device = value.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon
    codes = []
    views = []

    def clone_with(new_value, new_time, new_var, new_mask):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        return out

    # 1. Temporal window push: emphasize early and middle windows.
    early_mid = (time_norm <= 0.66).to(mask.dtype) * mask
    codes.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(bsz, -1))
    views.append(clone_with(value * early_mid, time, var_id, early_mid))

    # 2. Variable budget rotate: keep alternating variable groups.
    even_var = (var_id % 2 == 0).to(mask.dtype)
    budget = torch.zeros_like(mask)
    for local_var in torch.unique(var_id[mask > 0]).tolist():
        hit = ((var_id == int(local_var)) & (mask > 0)).to(mask.dtype)
        order = hit.cumsum(dim=1)
        budget = torch.maximum(budget, (order <= 2).to(mask.dtype) * hit)
    rotate_mask = torch.maximum(even_var * mask, budget)
    codes.append(torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(bsz, -1))
    views.append(clone_with(value * rotate_mask, time, var_id, rotate_mask))

    # 3. Patch visibility gate: sparse representatives in coarse temporal patches.
    patch_keep = torch.zeros_like(mask)
    for start, end in [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]:
        in_patch = ((time_norm >= start) & (time_norm < end)).to(mask.dtype) * mask
        patch_rank = in_patch.cumsum(dim=1)
        patch_keep = torch.maximum(patch_keep, (patch_rank <= 2).to(mask.dtype) * in_patch)
    codes.append(torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).expand(bsz, -1))
    views.append(clone_with(value * patch_keep, time, var_id, patch_keep))

    # 4. Panel sync toggle: snap close events to coarse rounds without changing values.
    rounded_time = torch.round(time_norm * 8.0) / 8.0 * horizon
    codes.append(torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(bsz, -1))
    views.append(clone_with(value * mask, rounded_time, var_id, mask))

    recipe_bank = torch.stack(codes, dim=1)
    if recipe_dim != 4:
        pad = torch.zeros(bsz, recipe_bank.size(1), max(recipe_dim - 4, 0), device=device)
        recipe_bank = torch.cat([recipe_bank[..., :recipe_dim], pad], dim=-1)[..., :recipe_dim]
    return recipe_bank, views
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `temporal-bias shift`：训练环境中短局部时间窗可见，测试环境改为稀疏 routine window。
   - `variable-affinity shift`：训练环境中某些变量联测，测试环境中同样变量异步拆开。
   - `patch-route shift`：训练环境中关键 patch 可见且路径浅，测试环境中 patch 被预算限制遮蔽。
   - `panel-sync shift`：同步 panel 与异步检查互换，检查 learned variable affinity 是否依赖流程共现。

2. **对比方法**
   - STAR-Set / VP-GNN 原模型。
   - 直接使用 structure trace 的 classifier。
   - state-policy 双图 / 双 bias 分支一致性 baseline。
   - policy adversarial baseline。
   - mask dropout / random missing augmentation。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、PSSC、C-CRS。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - first-stage IV strength / mean R2。
   - instrument moment gap：预测残差与反事实采样 instrument 的相关矩。
   - structure explained ratio：结构量中可由采样工具变量解释的比例。
   - residual structure reliance：分类头对净化后结构残差的使用强度。
   - policy transfer calibration error：不同采样政策下按 IV strength 分桶的校准误差。

4. **消融实验**
   - 去掉第一阶段，只直接用 `s_obs` 分类，验证结构内生性会导致 policy shortcut。
   - 去掉 instrument residual moment，检查预测残差是否重新与采样 recipe 相关。
   - 去掉 weak-instrument guard，验证无效 recipe 是否造成虚假净化。
   - 将 policy instruments 替换为随机 mask，验证收益来自结构定向工具变量而非普通增强。
   - 只用 `h_value` 不用 `u_struct`，检查净化后的结构残差是否仍提供有效状态信息。

## 5. 预期创新性

1. **从结构分解转向结构内生性识别**：不再预设哪条边、哪个 bias、哪个 patch 属于 state 或 policy，而是把结构量当作内生处理变量，通过反事实工具变量识别可由采样解释的部分。
2. **从一致性/保形/认证转向控制函数净化**：counterfactual intervention 只服务第一阶段结构方程和正交矩条件，不要求表示、logits、风险或预测集在多策略下相同。
3. **自然吸收 STAR-Set 与 VP-GNN**：它们产生的 attention bias、variable graph 和 patch graph 不被丢弃，而是被转化为可被 IV 净化的结构证据。
4. **与采样解耦框架低侵入兼容**：value encoder 保留，sampling branch 改成 instrument generator，分类头只接收 value state 与 residualized structure。
5. **审稿卖点清晰**：D-IVSP 把 sampling-policy shift 下的结构 shortcut 表述为“内生结构变量的工具变量识别问题”，提供 first-stage strength、instrument moment gap 等可检验诊断，而不是只报告经验鲁棒性。

## 6. 一句话投稿卖点

**D-IVSP 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“attention bias、变量图和 patch 路由等结构证据的内生性污染”问题，并通过反事实采样工具变量、第一阶段结构方程与控制函数残差分类，让模型保留 STAR-Set / VP-GNN 的结构归纳偏置，同时剥离可由医院流程、设备预算或 panel 同步策略解释的结构捷径。**
