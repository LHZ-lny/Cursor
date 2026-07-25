# Title: Observer-Control Barrier Certificates：面向采样策略偏移的观测控制屏障证书分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `my_work_summary|work_summary|工作总结|summary`：未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md` 与 `idea_2026-07-24.md`，纳入当前工作区缺失但记忆中存在的历史机制。
- 已读取近期论文入口 `paper_daily.md`。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：
  - `2026-06-17`: Counterfactual Error Cartography Semantic Compiler。
  - `2026-06-20`: Do-Measure Doubly Robust Risk Rewriting。
  - `2026-06-24`: Policy-Shadow Negative Film。
  - `2026-06-27`: Observability-Witness State Routing。
  - `2026-07-15`: RKHS Cubature Debiaser。
  - `2026-07-16`: Measurement-Action Bisimulation Lens。
  - `2026-07-17`: Policy-Word Signature Renormalizer。
  - `2026-07-18`: Policy-Thermodynamic Free-Energy Annealer。
  - `2026-07-19`: Sklar Shift Breaker。
  - `2026-07-20`: Triage-Queue Debt Neutralizer。
  - `2026-07-21`: Context-Anchored Sinkhorn Detail Canonicalizer。
  - `2026-07-22`: Policy-Artifact MDL Episode Transducer。
  - `2026-07-23`: Causal Sheaf-Glue Classifier。
  - `2026-07-24`: Trigger-Phase Hysteresis Neutralizer。

### 历史核心机制黑名单

为保证本提案与既往思路显著正交，本轮明确避开以下机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、hazard-driven counterfactual resampling。
8. 分类梯度与采样 score 的零空间正交化。
9. 多个 `do(policy)` 视图的 risk variance 约束。
10. 生理流算子与采样算子的交换子手术、value graph / policy graph 交换损失、policy residual sink。
11. additive evidence market、protocol tax、token evidence budget、边际证据审计。
12. 后验商动力学、采样似然因子相除、干预积分分类。
13. reconstruction error cartography、ANOVA state/policy error projection、VQ semantic clauses、HSIC redaction。
14. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler。
15. Radon-Nikodym density ratio、doubly robust target-measure correction、influence-bound regularization。
16. previsible martingale query、optional-stopping moment control、predictability barrier。
17. soft excursion topology、censored persistence interval likelihood、censor envelope。
18. policy-gauge frame、horizontal transport、vertical blindness。
19. policy-only negative film、shadow eraser/stencil。
20. parity-check codeword、syndrome locator、packet repair。
21. conditional knockoff calendar、soft knockoff-FDR firewall。
22. observability witness / Fisher-style probe / low-observability routed classification。
23. subjective-logic evidential shield、policy-induced ignorance/vacuity。
24. observation-set information lattice、meet/join monotonicity、submodular margin curvature。
25. solver trace / NFE / roughness front-door neutralization。
26. RKHS cubature weights、kernel moment exactness、control-variate cubature。
27. measurement-action bisimulation、canonical response head、label Lipschitz response metric。
28. policy-word signatures、shuffle-algebra counterterms。
29. thermodynamic free energy、policy temperature / heat capacity。
30. Sklar copula marginal stripping、rank-space tail dependence。
31. triage queue debt recurrence、service discipline neutralization。
32. unbalanced Sinkhorn detail-to-anchor canonicalization。
33. MDL episode transducer / artifact production costs。
34. causal sheaf gluing over observation covers。
35. trigger-threshold phase sweep、phase plateau readout、opening/closing hysteresis loss、boundary quarantine。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗、不做随机平滑认证、不做一致性、不做后验除法、不做停时鞅、不做 gauge 投影、不做拓扑胶囊、不做 knockoff、不做 evidence uncertainty，也不扫描 gate 阈值或 hysteresis；而是把采样策略看作作用在 observer hidden state 上的“观测控制场”，并学习一组 class-wise control-barrier certificates。分类器不仅要在事实观测上正确，还要证明：在由反事实采样模块定义的可部署观测控制集合内，真实类 margin 沿所有采样控制方向都不会被推穿决策边界。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中的 **Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction** 给出一个值得借鉴、但不能直接复用的结构事实：医疗时序模型经常先用全局上下文判断哪里值得细查，再把局部 detail inspector 投入关键片段。现实采样政策也类似一个 observer controller：

- 医生或设备先根据粗风险、报警、资源预算决定是否追加细粒度检查；
- 某些变量是否被联测，取决于 protocol panel、设备能耗和临床流程；
- 同一底层病程在不同医院下，会被不同的 detail exposure policy 观察到；
- 稀疏事件检测中的 context-detail interaction 暴露了“观测控制”这一层，但历史 `2026-07-24` 已经使用了 gate 阈值扫描、相位平台和 hysteresis，本提案必须避开这些机制。

因此，OCBC 不把 gate 当作可扫描阈值，也不对开关路径做滞回面积惩罚；它把 sampling-policy shift 形式化为一个控制系统：

```text
state observer hidden h
sampling/control recipe u
policy-induced latent drift g_phi(h, u)
class safety barrier B_y(h)
```

当训练环境中的采样策略改变时，模型 hidden state 会沿着某些 policy-control directions 漂移。例如：

- `u = dense_followup`: 报警后更密集复测，hidden state 更强调局部 detail；
- `u = panel_expand`: 同步 panel 让多个变量突然共同可见；
- `u = budget_cut`: 设备预算削减，某些变量或晚期窗口缺失；
- `u = detail_suppress`: 局部细节 inspector 被资源策略关闭。

如果分类器把这些 drift 当作类别证据，那么换医院或换设备后，margin 会被采样控制方向推穿边界。OCBC 的核心目标是：

> 允许 encoder 使用 context/detail、时间戳、变量值和反事实采样模块，但要求真实类别的安全 margin 在一族可部署采样控制方向上满足 Hamilton-Jacobi / control-barrier 不穿越条件。

这比“多视图 logits 一致”更弱，也比“删除采样信息”更细：

- 若采样策略真的带来新的状态观测，hidden state 可以变化；
- 若变化方向只是观测流程造成的 policy-control drift，它不能把真实类 margin 从安全区推到错误区；
- 训练后模型可以输出一个 **barrier violation score**，直接说明某次预测是否靠近采样控制可达的危险边界。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通 irregular encoder 改为 Observer-Control State Encoder

OCBC 可以包裹现有的 CDE、event Transformer、mTAND、SSM 或 context-detail encoder。核心是让 encoder 输出一个可微 hidden state `h`，并额外识别“采样控制如何推动 h”：

1. **Value Observer Encoder**
   - 输入事实不规则事件 `(event_value, event_time, event_var_id, event_mask, measurement_std)`。
   - 输出 `h_state`，主要描述观测值驱动的病程或系统状态。
   - 不把 policy id、环境标签或 sampling summary 直接拼入分类头。

2. **Policy Control Field**
   - 输入 `h_state` 和采样控制 recipe `u`。
   - 输出 latent drift `g_phi(h, u)`，表示如果当前样本被另一种观测控制策略读取，observer state 会沿哪个方向移动。
   - 它不是 hazard、不是 density ratio、不是 residual sink、不是 gauge vertical frame；它只提供 barrier loss 所需的控制方向。

3. **Barrier-Margin Classifier**
   - 分类头输出普通 logits，同时输出 class-wise barrier score。
   - 最简单可令 barrier 等于真实类 margin：

```text
B_y(h) = logit_y(h) - max_{k != y} logit_k(h)
```

   - 若 `B_y(h) > 0`，事实预测处在安全区；若存在控制方向 `u` 使 `B_y` 快速下降，则说明模型可能依赖采样策略捷径。

### 2.2 改 Loss：从一致性/对抗转向 Hamilton-Jacobi Barrier Safety

总目标：

```text
L = L_cls
  + lambda_bar * L_hj_barrier
  + lambda_id  * L_control_field_identification
  + lambda_min * L_minimal_barrier_slack
```

#### A. 分类损失 `L_cls`

事实观测直接分类：

```text
L_cls = CE(Classifier(h_state), y)
```

这保证基础准确率，不改变已有分类目标。

#### B. HJ / Control Barrier Loss `L_hj_barrier`

对每个样本和每个反事实采样控制 recipe `u_r`，计算真实类 margin 关于 hidden state 的方向导数：

```text
dot_B_y(h, u_r) = grad_h B_y(h)^T g_phi(h, u_r)
```

control-barrier 条件要求：

```text
dot_B_y(h, u_r) + alpha * B_y(h) >= safety_margin
```

含义是：即使采样策略沿 `u_r` 推动 observer state，真实类安全 margin 也不能以足够快的速度下降并穿越 0。可微损失为：

```text
L_hj_barrier =
  mean_r relu(safety_margin - dot_B_y(h, u_r) - alpha * B_y(h))^2
```

这不是 consistency loss：

- 不要求 `h(do(u_r)) = h(factual)`。
- 不要求 `logits(do(u_r)) = logits(factual)`。
- 不惩罚所有采样控制引起的状态变化。
- 只惩罚“沿采样控制方向会把真实类 margin 推穿边界”的危险动力学。

#### C. Control Field Identification `L_control_field_identification`

为了让 `g_phi(h,u)` 真正表示采样控制方向，反事实采样模块生成轻量可见性视图，但这些视图不用于一致性训练，只用于一阶控制场识别：

```text
h_cf = Encoder(do(u_r, x))
target_drift = stopgrad(h_cf - h_factual)
L_control_id = SmoothL1(g_phi(h_factual, u_r), target_drift)
```

关键区别：

- 我们不要求 `h_cf` 与 `h_factual` 接近；
- 相反，我们允许它们不同，并用差值监督控制场；
- 这个控制场随后只进入 barrier inequality，而不直接进入分类头。

#### D. Minimal Barrier Slack `L_minimal_barrier_slack`

若模型把所有 margin 拉到无限大，barrier loss 会变得平凡。OCBC 加入最小松弛约束：

```text
L_minimal_barrier_slack =
  relu(B_y(h) - b_max)^2
  + relu(control_norm - g_max)^2
```

它鼓励模型给出足够但不过度的安全边界，避免靠 logit 爆炸或巨大控制场抵消来伪造鲁棒性。

### 2.3 改 Dataloader：返回 Observer-Control Recipe Bank

新增 `ObserverControlCollator`。每个 batch 返回：

1. `factual_batch`：原始不规则事件。
2. `control_recipe_bank`：一组采样控制 recipe：
   - `routine_budget_cut`：变量级预算减少；
   - `alarm_detail_expand`：报警后局部 detail 暴露增加；
   - `panel_synchrony_shift`：同步 panel 扩张或拆分；
   - `late_window_suppress`：晚期窗口观测被设备策略削弱；
   - `quality_degrade`：观测质量或测量误差恶化。
3. `counterfactual_batch_bank`：由 recipe 生成的轻量反事实观测，仅用于识别 `g_phi(h,u)`。
4. `control_radius`：每个 recipe 的部署可达强度，用于报告证书。

这些 recipe 与历史机制区别明确：

- 不是 hazard do-resampling。
- 不是 policy simplex randomized smoothing。
- 不是 stopping recipes。
- 不是 censor recipe。
- 不是 chart/gauge move。
- 不是 knockoff calendar。
- 不是 gate threshold phase sweep。
- 不是 meet/join lattice view。

它们只定义 observation-control directions，并服务于 HJ barrier inequality。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value process 保留为 `ValueObserverEncoder`。
- 现有 sampling branch 改为 `PolicyControlField`：从采样 recipe 和 hidden state 估计 latent control drift。
- 现有 counterfactual intervention 改为 `ObserverControlRecipeBank`：生成可部署观测控制动作，用于识别控制场和检验 barrier。
- 推理阶段：
  - 不需要生成反事实视图即可预测；
  - 可用标准 control recipe bank 计算 `barrier_violation_score`；
  - 若某个样本在采样控制方向上接近决策边界，输出“policy-control unsafe”诊断。

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


def true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logit = logits.gather(1, labels[:, None]).squeeze(1)
    rival_logit = logits.masked_fill(
        F.one_hot(labels, logits.size(-1)).bool(),
        -1e4,
    ).max(dim=-1).values
    return true_logit - rival_logit


class ValueObserverEncoder(nn.Module):
    """Encode irregular observations into a differentiable observer state."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, batch: dict) -> torch.Tensor:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(value))

        horizon = (time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_h = self.var_embed(var_id.clamp_min(0))
        event_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(meas_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * event_mask.unsqueeze(-1)
        seq_h, _ = self.rnn(event_h)
        return masked_mean(seq_h, event_mask, dim=1)


class PolicyControlField(nn.Module):
    """Map observation-control recipes to latent drifts around the observer state."""

    def __init__(self, hidden_dim: int, recipe_dim: int):
        super().__init__()
        self.recipe_proj = nn.Sequential(
            nn.Linear(recipe_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.field = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, recipe: torch.Tensor) -> torch.Tensor:
        recipe_h = self.recipe_proj(recipe)
        x = torch.cat([state, recipe_h], dim=-1)
        direction = self.field(x)
        direction = F.normalize(direction, dim=-1)
        return direction * self.scale(x)


class ObserverControlBarrierClassifier(nn.Module):
    """Sampling-policy robust classifier with control-barrier certificates."""

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        num_classes: int,
        recipe_dim: int,
    ):
        super().__init__()
        self.encoder = ValueObserverEncoder(num_vars=num_vars, hidden_dim=hidden_dim)
        self.control = PolicyControlField(hidden_dim=hidden_dim, recipe_dim=recipe_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict) -> dict:
        state = self.encoder(batch)
        logits = self.classifier(state)
        return {"state": state, "logits": logits}

    def barrier_directional_derivative(
        self,
        state: torch.Tensor,
        labels: torch.Tensor,
        recipe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Recompute logits from a leaf-like state so autograd can form grad_h margin.
        state_for_grad = state.requires_grad_(True)
        logits = self.classifier(state_for_grad)
        margin = true_margin(logits, labels)
        grad_margin = torch.autograd.grad(
            margin.sum(),
            state_for_grad,
            create_graph=True,
            retain_graph=True,
        )[0]
        drift = self.control(state_for_grad, recipe)
        lie_derivative = (grad_margin * drift).sum(dim=-1)
        return margin, lie_derivative, drift

    def hj_barrier_loss(
        self,
        state: torch.Tensor,
        labels: torch.Tensor,
        recipe_bank: torch.Tensor,
        alpha: float = 0.4,
        safety_margin: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        violations = []
        drift_norms = []
        for recipe in recipe_bank.unbind(dim=1):
            margin, lie, drift = self.barrier_directional_derivative(state, labels, recipe)
            # HJ / control-barrier condition:
            # dB/dh * g(h,u) + alpha * B(h) >= safety_margin.
            violation = F.relu(safety_margin - lie - alpha * margin).pow(2)
            violations.append(violation)
            drift_norms.append(drift.norm(dim=-1))
        return torch.stack(violations, dim=1).mean(), torch.stack(drift_norms, dim=1).mean()

    def control_identification_loss(
        self,
        factual_state: torch.Tensor,
        cf_state_bank: torch.Tensor,
        recipe_bank: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for cf_state, recipe in zip(cf_state_bank.unbind(dim=1), recipe_bank.unbind(dim=1)):
            pred_drift = self.control(factual_state, recipe)
            target_drift = (cf_state - factual_state).detach()
            losses.append(F.smooth_l1_loss(pred_drift, target_drift))
        return torch.stack(losses).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_bar: float = 0.25,
        lambda_id: float = 0.15,
        lambda_min: float = 0.02,
        b_max: float = 8.0,
        g_max: float = 1.0,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)

        barrier_loss, drift_norm = self.hj_barrier_loss(
            state=out["state"],
            labels=labels,
            recipe_bank=batch["control_recipe_bank"],
        )

        cf_states = []
        for cf_batch in batch["counterfactual_batch_bank"]:
            cf_states.append(self.encoder(cf_batch))
        cf_state_bank = torch.stack(cf_states, dim=1)
        control_id_loss = self.control_identification_loss(
            factual_state=out["state"],
            cf_state_bank=cf_state_bank,
            recipe_bank=batch["control_recipe_bank"],
        )

        margin = true_margin(out["logits"], labels)
        slack_loss = F.relu(margin - b_max).pow(2).mean() + F.relu(drift_norm - g_max).pow(2)

        total = cls_loss + lambda_bar * barrier_loss + lambda_id * control_id_loss + lambda_min * slack_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "hj_barrier_loss": barrier_loss.detach(),
            "control_identification_loss": control_id_loss.detach(),
            "minimal_slack_loss": slack_loss.detach(),
            "mean_margin": margin.mean().detach(),
            "mean_control_drift_norm": drift_norm.detach(),
        }

    @torch.no_grad()
    def barrier_diagnostic(self, batch: dict) -> dict:
        out = self.forward(batch)
        labels = out["logits"].argmax(dim=-1)
        # Diagnostics can be run with gradients enabled in evaluation code if exact
        # Lie derivatives are needed. Here we return the factual margin only.
        return {
            "pred": labels,
            "margin": true_margin(out["logits"], labels),
        }


@torch.no_grad()
def build_observer_control_bank(batch: dict, recipe_dim: int = 5) -> tuple[torch.Tensor, list[dict]]:
    """Create observation-control recipes and counterfactual batches.

    These views are used to identify latent control fields, not to enforce
    representation/logit consistency.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    meas_std = batch.get("measurement_std", torch.zeros_like(value))
    bsz, num_events = value.shape
    device = value.device

    recipes = []
    cf_batches = []

    def clone_with(value_new, time_new, var_new, mask_new, std_new):
        out = dict(batch)
        out["event_value"] = value_new
        out["event_time"] = time_new
        out["event_var_id"] = var_new
        out["event_mask"] = mask_new
        out["measurement_std"] = std_new
        return out

    eye = torch.eye(recipe_dim, device=device, dtype=value.dtype)

    # 1. Routine budget cut: keep at most two observations per variable.
    budget_mask = torch.zeros_like(mask)
    for var_idx in torch.unique(var_id[mask > 0]).tolist():
        hit = ((var_id == int(var_idx)) & (mask > 0)).to(mask.dtype)
        order = hit.cumsum(dim=1)
        budget_mask = torch.maximum(budget_mask, hit * (order <= 2).to(mask.dtype))
    recipes.append(eye[0].expand(bsz, -1))
    cf_batches.append(clone_with(value * budget_mask, time, var_id, budget_mask, meas_std))

    # 2. Alarm detail expand: amplify local detail around high absolute deviations.
    centered = value - masked_mean(value, mask, dim=1).unsqueeze(1)
    high_detail = (centered.abs() >= masked_mean(centered.abs(), mask, dim=1).unsqueeze(1)).to(mask.dtype) * mask
    detail_value = value * (1.0 + 0.15 * high_detail)
    recipes.append(eye[1].expand(bsz, -1))
    cf_batches.append(clone_with(detail_value, time, var_id, mask, meas_std))

    # 3. Panel synchrony shift: snap nearby event times to coarse bins.
    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    coarse_time = torch.round(time / horizon * 8.0) / 8.0 * horizon
    recipes.append(eye[2].expand(bsz, -1))
    cf_batches.append(clone_with(value, coarse_time, var_id, mask, meas_std))

    # 4. Late-window suppress: remove late observations.
    late_keep = ((time / horizon) < 0.67).to(mask.dtype) * mask
    recipes.append(eye[3].expand(bsz, -1))
    cf_batches.append(clone_with(value * late_keep, time, var_id, late_keep, meas_std))

    # 5. Quality degrade: same calendar, worse measurement quality.
    noisy_std = meas_std + 0.25 * value.detach().abs().mean(dim=1, keepdim=True)
    recipes.append(eye[4].expand(bsz, -1))
    cf_batches.append(clone_with(value, time, var_id, mask, noisy_std))

    recipe_bank = torch.stack(recipes, dim=1)
    return recipe_bank, cf_batches
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `detail-exposure shift`：训练环境报警后密集 detail 采样，测试环境只保留 routine 采样。
   - `panel-synchrony shift`：训练环境同步 panel，测试环境拆成异步变量。
   - `budget-control shift`：设备或医院预算导致变量级观测数量削减。
   - `late-suppression shift`：晚期窗口在测试环境中缺失或低质量。
   - `quality-control shift`：同样采样日历下测量误差分布改变。

2. **对比方法**
   - 普通 irregular encoder。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - CDI-style context-detail gate baseline。
   - MVC-CDE / kernel smoothing baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、TPHN 等。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - barrier violation rate：`dot_B + alpha B < margin` 的样本比例。
   - minimum control safety margin：所有 control recipes 下的最小 HJ slack。
   - unsafe-error overlap：错误预测是否集中在高 barrier violation 样本。
   - policy-control diagnostic AUC：barrier violation 是否能预测跨策略失败。

4. **消融实验**
   - 去掉 `L_hj_barrier`，验证采样控制方向是否能推穿 margin。
   - 去掉 `L_control_field_identification`，验证控制场是否退化为任意方向。
   - 将 control recipes 替换为随机 mask，验证收益来自可解释 observer-control 结构。
   - 去掉 minimal slack，检查是否出现 logit 爆炸式伪证书。
   - 扫描 control recipe bank 大小，评估安全证书覆盖度与训练开销。

## 5. 预期创新性

1. **从采样去偏转向观测控制安全性**：不问采样概率、不删除采样信息，而是问采样控制能否把分类 margin 推穿边界。
2. **从 gate/hysteresis 转向 control-barrier certificate**：吸收 context-detail event detection 中“粗筛后细查”的观察控制启发，但不扫描 gate 阈值、不做相位平台或滞回损失。
3. **从一致性增强转向 HJ 不穿越条件**：不同采样控制下 hidden state 和 logits 可以变化；只要真实类 barrier 不被控制方向推穿即可。
4. **从经验 worst-policy 鲁棒转向可诊断安全边界**：模型输出 barrier violation，能解释某个预测是否对可部署采样控制不安全。
5. **与采样解耦/反事实干预框架低侵入兼容**：counterfactual sampler 只需提供 observer-control recipes；value encoder 和分类主干可以沿用现有实现。

## 6. 一句话投稿卖点

**OCBC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“观测控制场推动 observer hidden state 穿越类别安全边界”的问题，并通过 Policy Control Field、Hamilton-Jacobi Barrier Loss 与 Observer-Control Recipe Bank，让模型在不依赖危险率、对抗、一致性、后验商、随机平滑、停时鞅、拓扑、gauge、纠错码、knockoff、evidential uncertainty、信息格、solver trace 或 trigger-phase hysteresis 的前提下，获得可诊断的采样策略偏移鲁棒分类边界。**
