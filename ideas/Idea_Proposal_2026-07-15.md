# Title: RKHS Cubature Debiaser：面向采样策略偏移的再生核求积去偏分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已读取自动化记忆 `MEMORIES.md`：确认多轮历史任务同样未发现 `my_work_summary.md`，并记录了当前工作区未检出的历史提案核心机制。
- 已读取近期论文记录：`paper_daily.md`、`paper_daily_2026-07-13.md`。
- 已读取当前工作区内全部历史提案：`Idea_Proposal_2026-06-12.md`、`2026-06-13.md`、`2026-06-14.md`、`2026-06-16.md`、`2026-06-19.md`、`2026-06-21.md`、`2026-06-22.md`、`2026-06-23.md`、`2026-06-25.md`、`2026-06-26.md`、`2026-07-12.md`、`2026-07-13.md`、`2026-07-14.md`。
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案机制：`2026-06-17`、`2026-06-20`、`2026-06-24`、`2026-06-27`。

### 历史核心机制黑名单

为避免与历史提案和 `paper_daily.md` 中已有机制发生思维重合，本提案明确避开以下方向作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer。
8. 分类梯度与采样 score 的零空间正交化。
9. hazard-driven counterfactual resampling。
10. 多个 `do(policy)` 视图的 risk variance 约束。
11. 生理流算子与采样算子的交换子手术。
12. value graph / policy graph 的交换或分离损失。
13. policy residual sink 作为采样残差收纳槽。
14. additive evidence market、protocol tax、token-level evidence budget 或边际证据审计。
15. 后验商动力学、模型空间 posterior quotient、采样似然因子相除、干预积分分类。
16. reconstruction error cartography、ANOVA-style state/policy error projection、VQ semantic clauses、policy-sensitive acquisition clauses、HSIC redaction checksum。
17. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler、policy coverage loss。
18. 采样测度上的 Radon-Nikodym density ratio、doubly robust target-measure correction、influence-bound regularization。
19. previsible martingale query、continuous-discrete standardized innovation、optional-stopping moment control、stopping recipe collator、predictability barrier。
20. soft excursion topology、censored persistence interval likelihood、censor envelope、fragmentation sobriety。
21. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
22. policy-only negative film、dual-exposure value/shadow encoders、latent shadow eraser/stencil、shadow nullification high-entropy loss、value retention buckets。
23. latent packet codeword、learnable parity-check、syndrome locator、syndrome-guided packet repair、codeword closure。
24. conditional knockoff calendar、calendar adapter、knockoff group statistics、soft knockoff-FDR firewall、swap symmetry calibration。
25. measurement-Jacobian/Fisher-style observability witness、counterfactual observability probe bank、low-quantile state-coordinate gate、observability-routed classification。
26. subjective-logic / Dirichlet evidential classification、policy-induced ignorance/vacuity mass、class-wise evidence discount、policy stress audit bank。
27. observation-set policy lattice、meet/join visibility masks、信息偏序单调性、次模边际契约、quality-order loss、shortcut curvature penalty。
28. trace-instrumented kernel CDE、solver-trace mediator、NFE/roughness/step-size/bandwidth proxy、reference trace bank、front-door trace-standardized readout、trace do-slope penalty。
29. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing、MedMamba 的 frequency/adaptive graph branch、MedSpaformer 的普通 token sparsification、MILM 的 value-redacted classifier、StarEmbed 的直接 foundation embedding、LLMTS 的普通 LLM alignment 或 MVC-CDE 的普通多视图平滑路径作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗，不要求多采样视图表征一致，不做后验除法、随机平滑认证、停时鞅、拓扑、gauge、纠错码、knockoff、观测性门控、evidential uncertainty、信息格契约或求解轨迹中和；而是把非规则观测看成对连续时间状态证据积分的离散求积点。采样策略偏移首先造成的是求积公式偏差：某些时间窗、变量或 panel 因被过密采样而在积分中被重复计票，另一些区域因稀疏而权重不足。RKHS Cubature Debiaser 用再生核矩精确约束直接求解一组可微 cubature 权重，让分类器近似消费同一个连续状态证据积分，而不是消费训练政策制造的事件计数。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 `paper_daily_2026-07-13.md` 中的 **Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing (MVC-CDE)** 指出：不规则采样影响的不只是缺失率，还会改变 control path 的几何粗糙度；用 kernel / Gaussian-process smoothing 可以把离散观测提升为更平滑的连续路径。历史最新提案已经把 MVC-CDE 的启发推进到 solver trace 中介层，因此本提案不再使用 NFE、step-size、roughness trace 或 front-door trace readout，而是转向 **continuous evidence integral 的离散求积偏差**。

许多 irregular classifier 的最终 readout 本质上都近似：

```text
continuous state evidence integral ~= sum_i quadrature_weight_i * event_evidence_i
```

采样政策一变，离散求积点也随之改变：报警后密集复测会让短时间窗口重复计票；同步 panel 会让某个变量组过度可见；设备预算会让晚期窗口证据被低估；多波段 cadence 改变会让相位或变量覆盖的核矩偏移。普通 pooling 或 attention 很容易把“事件多”解释为“证据强”，从而把训练医院或设备的采样日程学成类别边界。

**RKHS Cubature Debiaser (RKCD)** 的目标是：用近期 kernel smoothing 机制启发出的 RKHS 特征 `psi(t, variable, quality)` 定义标签无关的参考核矩 `mu_reference`，并为每条非规则序列求解一组可微 cubature weights：

```text
sum_i w_i psi_i ~= mu_reference
```

分类器最终看到的是 `sum_i w_i h_i`，即连续证据积分的核矩校准估计，而不是训练政策诱导的原始事件计数。这样，RKCD 保留 value evidence，但压低采样政策造成的过密/过稀积分偏差。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：Kernel-Cubature Event Encoder

1. **Value Event Lift**：输入 `(value, time, variable_id, measurement_std, event_mask)`，输出 value-driven event evidence `h_i`。该模块可接现有 mTAND、CDE、SSM、event Transformer 或轻量 GRU；不拼接 policy id，不把 missingness pattern 直接作为类别特征。

2. **RKHS Sampling Coordinates**：为每个事件构造核特征：

```text
psi_i = [RBF_time(t_i), onehot(variable_i), RBF_quality(std_i), low-rank time x variable features]
```

这些特征描述事件在连续积分域中的坐标。它们受 MVC-CDE 的 kernel smoothing 启发，但目的不是平滑 control path 或降低 solver NFE，而是定义求积公式应满足的再生核矩。

3. **Differentiable Cubature Solver**：给定基础非负权重 `a_i` 与核特征矩阵 `Psi`，解一个 ridge KKT 投影：

```text
min_w ||w - a||_2^2 + rho ||w||_2^2
s.t.  Psi^T w = mu_reference
      sum_i w_i = 1
      w_i ~= 0 for padded events
```

若精确约束不可行，则返回最小残差权重，并将 `moment_residual` 作为诊断。它不是 Radon-Nikodym ratio，不估计 `p_test / p_train`，也不建模 sampling hazard。

4. **Cubature Evidence Readout**：

```text
z_cubature = sum_i w_i h_i
logits = Classifier(z_cubature)
```

过密采样区域不会因事件数多而自动贡献更多；稀疏但覆盖关键核矩的事件可以获得适当权重；如果测试策略造成不可恢复的覆盖缺口，moment residual 会升高。

### 2.2 改 Loss：Reproducing Moment Exactness

总目标：

```text
L = L_cls
  + lambda_moment * L_kernel_moment_exactness
  + lambda_cv     * L_policy_control_variate
  + lambda_ess    * L_weight_sobriety
  + lambda_jack   * L_cubature_jackknife
```

#### A. 分类损失 `L_cls`

```text
L_cls = CE(Classifier(sum_i w_i h_i), y)
```

区别于普通 attention pooling：`w_i` 由核矩精确约束控制，不由采样密度自由决定。

#### B. Kernel Moment Exactness `L_kernel_moment_exactness`

对事实采样和反事实 sampling stencils 生成的可见子集，都计算 cubature residual：

```text
L_kernel_moment_exactness = || sum_i w_i psi_i - mu_reference ||_2^2
```

这不是多视图一致性：不同 stencil 的 logits 和 representation 可以不同；若某个 stencil 删除了关键区域，moment residual 可以变大，作为“该策略无法支撑同等连续积分”的诊断。

#### C. Policy Control Variate `L_policy_control_variate`

采样解耦框架中的 sampling branch 改为输出 control-variate basis `c_i`，描述事件密度、panel burst、局部同步、变量预算等 policy-sensitive 坐标。它不进入 classifier，只要求 cubature weights 对纯政策基函数积分为零或参考常数：

```text
L_policy_control_variate = || sum_i w_i c_i - c_reference ||_2^2
```

这不是 adversarial、density ratio、evidence tax、knockoff 或 solver trace；它只是数值求积中的控制变量思想。

#### D. Weight Sobriety `L_weight_sobriety`

防止 KKT 求解器用极端权重满足矩条件：

```text
ESS = 1 / sum_i w_i^2
L_weight_sobriety =
  relu(ess_min - ESS)^2
  + mean relu(-w_i)^2
  + mean relu(|w_i| - w_max)^2
```

#### E. Cubature Jackknife `L_cubature_jackknife`

对高杠杆采样组做 leave-one-group-out：

```text
leverage_g = || z_full - z_without_group_g ||_2
moment_gap_g = || mu_full - mu_without_group_g ||_2
L_cubature_jackknife = mean_g relu(leverage_g - kappa * moment_gap_g - margin)^2
```

如果某个 panel group 一删掉 logits 就崩，但核矩缺口并不大，说明模型可能仍在利用采样政策制造的过密 token。

### 2.3 改 Dataloader：Cubature Stencil Bank

新增 `CubatureStencilCollator`。每个 batch 返回：

1. 原始事件：`event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_std`。
2. `reference_moment`：标签无关核矩目标，例如均匀时间覆盖、变量组均衡、测量质量基线、低阶 time-variable 交互。
3. `stencil_recipe_bank` 与 `stencil_visibility_bank`：
   - `dense_burst_thinning`：报警后高频窗口稀疏化；
   - `panel_unbatching`：同步 panel 拆成异步事件；
   - `late_window_budget`：晚期窗口预算收缩；
   - `quality_degrade`：测量质量或异方差改变。
4. `control_variate_target`：通常为零向量或数据集定义的参考政策常数。

这些 stencil 不构成对比正样本，不要求风险方差稳定，也不产生 certified radius；它们只检查在不同采样政策下，同一连续证据积分是否仍可由可见事件稳定求积。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 保留，输出 event evidence `h_i`。
- 现有 sampling branch 改为 `PolicyControlVariateHead`，输出只用于求积约束的 `c_i`，不进入分类头。
- 现有 counterfactual intervention 改为 `CubatureStencilBank`，提供不同采样政策下的可见事件与核矩可行性诊断。
- 推理阶段无需已知策略标签：只根据事实观测计算 RKHS features、cubature weights、moment residual 和分类 logits。
- 部署诊断输出：
  - `moment_residual`：当前采样策略是否支持参考连续积分；
  - `effective_sample_size`：预测是否依赖少数高权重事件；
  - `policy_control_variate_residual`：是否仍存在采样密度/panel 同步泄漏；
  - `jackknife_leverage`：哪些变量组或时间窗具有异常求积杠杆。

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


class RKHSFeatureMap(nn.Module):
    """Build label-free RKHS coordinates for cubature moment constraints."""

    def __init__(self, num_vars: int, num_time_centers: int = 8, num_quality_centers: int = 4):
        super().__init__()
        self.num_vars = num_vars
        self.register_buffer("time_centers", torch.linspace(0.0, 1.0, num_time_centers))
        self.register_buffer("quality_centers", torch.linspace(0.0, 2.0, num_quality_centers))
        self.log_time_bw = nn.Parameter(torch.tensor(-1.0))
        self.log_quality_bw = nn.Parameter(torch.tensor(-0.5))

    def forward(self, event_time: torch.Tensor, event_var_id: torch.Tensor, event_mask: torch.Tensor, measurement_std: torch.Tensor) -> torch.Tensor:
        horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        t = (event_time / horizon).clamp(0.0, 1.0)
        time_bw = F.softplus(self.log_time_bw) + 1e-3
        quality_bw = F.softplus(self.log_quality_bw) + 1e-3

        time_feat = torch.exp(-0.5 * ((t[..., None] - self.time_centers) / time_bw).pow(2))
        quality = torch.log1p(measurement_std).clamp(0.0, 4.0)
        quality_feat = torch.exp(-0.5 * ((quality[..., None] - self.quality_centers) / quality_bw).pow(2))
        var_feat = F.one_hot(event_var_id.clamp_min(0), self.num_vars).to(event_time.dtype)

        # A compact time-variable interaction catches panel/window over-coverage.
        interaction = var_feat * (2.0 * t.unsqueeze(-1) - 1.0)
        psi = torch.cat([time_feat, quality_feat, var_feat, interaction], dim=-1)
        return psi * event_mask.unsqueeze(-1)


class EventEvidenceLift(nn.Module):
    """Value-driven event evidence; sampling coordinates are not class features."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id.clamp_min(0))
        x = torch.cat(
            [var_h, value.unsqueeze(-1), torch.log1p(delta_t).unsqueeze(-1), torch.log1p(meas_std).unsqueeze(-1)],
            dim=-1,
        )
        return self.net(x) * mask.unsqueeze(-1)


class CubatureWeightSolver(nn.Module):
    """Differentiable ridge KKT solver for RKHS moment-matching cubature."""

    def __init__(self, hidden_dim: int, ridge: float = 1e-3):
        super().__init__()
        self.base_weight = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.ridge = ridge

    def forward(self, event_h: torch.Tensor, psi: torch.Tensor, mask: torch.Tensor, reference_moment: torch.Tensor) -> dict:
        bsz, num_events, _ = event_h.shape
        base_logits = self.base_weight(event_h).squeeze(-1).masked_fill(mask == 0, -1e4)
        a = torch.softmax(base_logits, dim=-1)

        # Constraint matrix C contains [sum-to-one, RKHS moments].
        ones = mask.unsqueeze(-1).to(event_h.dtype)
        cmat = torch.cat([ones, psi], dim=-1)  # [B, N, M + 1]
        target = torch.cat([torch.ones(bsz, 1, device=event_h.device, dtype=event_h.dtype), reference_moment], dim=-1)

        # Projection: w = a - C (C^T C + ridge I)^-1 (C^T a - target).
        gram = torch.bmm(cmat.transpose(1, 2), cmat)
        eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype).unsqueeze(0)
        rhs = torch.bmm(cmat.transpose(1, 2), a.unsqueeze(-1)).squeeze(-1) - target
        correction_coeff = torch.linalg.solve(gram + self.ridge * eye, rhs.unsqueeze(-1)).squeeze(-1)
        correction = torch.bmm(cmat, correction_coeff.unsqueeze(-1)).squeeze(-1)
        weights = (a - correction) * mask

        moment = torch.bmm(weights.unsqueeze(1), psi).squeeze(1)
        ess = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-6)
        return {"weights": weights, "base_weights": a, "moment": moment, "ess": ess}


class PolicyControlVariateHead(nn.Module):
    """Policy-sensitive basis used only for cubature constraints."""

    def __init__(self, num_vars: int, control_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.proj = nn.Sequential(nn.Linear(num_vars + 5, control_dim), nn.SiLU(), nn.Linear(control_dim, control_dim))

    def forward(self, batch: dict) -> torch.Tensor:
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        t = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_onehot = F.one_hot(var_id.clamp_min(0), self.num_vars).to(time.dtype)
        local_density = 1.0 / (1.0 + delta_t)
        early = (t <= 0.33).to(time.dtype)
        middle = ((t > 0.33) & (t <= 0.66)).to(time.dtype)
        late = (t > 0.66).to(time.dtype)
        coord = torch.cat([var_onehot, t.unsqueeze(-1), local_density.unsqueeze(-1), early.unsqueeze(-1), middle.unsqueeze(-1), late.unsqueeze(-1)], dim=-1)
        return self.proj(coord) * mask.unsqueeze(-1)


class RKHSCubatureDebiaser(nn.Module):
    """Sampling-policy robust classifier through RKHS cubature debiasing."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int, control_dim: int = 8):
        super().__init__()
        self.evidence = EventEvidenceLift(num_vars, hidden_dim)
        self.rkhs = RKHSFeatureMap(num_vars)
        self.solver = CubatureWeightSolver(hidden_dim)
        self.control = PolicyControlVariateHead(num_vars, control_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, batch: dict, visibility: torch.Tensor | None = None) -> dict:
        mask = batch["event_mask"] if visibility is None else batch["event_mask"] * visibility
        local_batch = dict(batch)
        local_batch["event_mask"] = mask
        event_h = self.evidence(local_batch)
        meas_std = local_batch.get("measurement_std", torch.zeros_like(local_batch["event_value"]))
        psi = self.rkhs(local_batch["event_time"], local_batch["event_var_id"], mask, meas_std)
        reference = local_batch["reference_moment"]
        cub = self.solver(event_h, psi, mask, reference)
        z = torch.bmm(cub["weights"].unsqueeze(1), event_h).squeeze(1)
        logits = self.classifier(z)
        control = self.control(local_batch)
        cv_moment = torch.bmm(cub["weights"].unsqueeze(1), control).squeeze(1)
        return {**cub, "logits": logits, "state": z, "psi": psi, "control_moment": cv_moment}

    def training_loss(
        self,
        batch: dict,
        lambda_moment: float = 0.25,
        lambda_cv: float = 0.15,
        lambda_ess: float = 0.05,
        lambda_jack: float = 0.05,
        ess_min: float = 4.0,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]
        cls_loss = F.cross_entropy(out["logits"], labels)
        moment_loss = (out["moment"] - batch["reference_moment"]).pow(2).mean()
        cv_target = batch.get("control_variate_target", torch.zeros_like(out["control_moment"]))
        cv_loss = (out["control_moment"] - cv_target).pow(2).mean()
        ess_loss = F.relu(ess_min - out["ess"]).pow(2).mean() + F.relu(-out["weights"]).pow(2).mean()

        jack_losses = []
        if "stencil_visibility_bank" in batch:
            for visibility in batch["stencil_visibility_bank"].unbind(dim=1):
                stencil_out = self.forward(batch, visibility=visibility)
                state_gap = (out["state"].detach() - stencil_out["state"]).pow(2).sum(dim=-1).sqrt()
                moment_gap = (batch["reference_moment"] - stencil_out["moment"]).pow(2).sum(dim=-1).sqrt()
                jack_losses.append(F.relu(state_gap - 2.0 * moment_gap - 0.05).pow(2).mean())
        jack_loss = torch.stack(jack_losses).mean() if jack_losses else torch.zeros((), device=out["logits"].device)

        total = cls_loss + lambda_moment * moment_loss + lambda_cv * cv_loss + lambda_ess * ess_loss + lambda_jack * jack_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "kernel_moment_loss": moment_loss.detach(),
            "policy_control_variate_loss": cv_loss.detach(),
            "weight_sobriety_loss": ess_loss.detach(),
            "cubature_jackknife_loss": jack_loss.detach(),
            "mean_effective_sample_size": out["ess"].mean().detach(),
        }


@torch.no_grad()
def build_reference_moment(batch: dict, num_vars: int, num_time_centers: int = 8, num_quality_centers: int = 4) -> torch.Tensor:
    """A simple label-free reference moment: uniform time, balanced variables, baseline quality."""
    device = batch["event_time"].device
    dtype = batch["event_time"].dtype
    time_moment = torch.ones(batch["event_time"].size(0), num_time_centers, device=device, dtype=dtype) / num_time_centers
    quality_moment = torch.zeros(batch["event_time"].size(0), num_quality_centers, device=device, dtype=dtype)
    quality_moment[:, 0] = 1.0
    var_moment = torch.ones(batch["event_time"].size(0), num_vars, device=device, dtype=dtype) / num_vars
    interaction_moment = torch.zeros(batch["event_time"].size(0), num_vars, device=device, dtype=dtype)
    return torch.cat([time_moment, quality_moment, var_moment, interaction_moment], dim=-1)


@torch.no_grad()
def build_cubature_stencil_visibility(batch: dict) -> torch.Tensor:
    """Create policy stencils for cubature feasibility diagnostics, not consistency pairs."""
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device
    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    t = time / horizon

    late_budget = ((t < 0.67) | (torch.arange(num_events, device=device)[None] % 2 == 0)).to(mask.dtype) * mask
    burst_thin = ((torch.arange(num_events, device=device)[None] % 2 == 0).to(mask.dtype)) * mask
    even_vars = ((var_id % 2) == 0).to(mask.dtype) * mask
    early_only = (t <= 0.75).to(mask.dtype) * mask
    return torch.stack([late_budget, burst_thin, even_vars, early_only], dim=1)
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `density quadrature shift`：训练环境报警后密集复测，测试环境固定低频采样。
   - `panel cubature shift`：训练环境同步 panel，测试环境拆成异步项目。
   - `late-window under-integration shift`：晚期窗口因设备预算被稀疏化。
   - `quality-kernel shift`：测量异方差或返回延迟改变 quality kernel moment。

2. **对比方法**
   - 普通 attention / set pooling irregular encoder。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - MVC-CDE / attentive kernel smoothing baseline。
   - DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN 等历史方案。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - kernel moment residual under shift。
   - effective sample size of cubature weights。
   - policy control-variate residual。
   - high-leverage stencil failure rate：错误预测是否依赖少数采样组高权重。

4. **消融实验**
   - 去掉 KKT cubature solver，改为 softmax attention pooling。
   - 去掉 `L_policy_control_variate`，检查 panel / density shortcut 是否回流。
   - 去掉 `L_weight_sobriety`，检查是否出现极端负权重或少数事件支配。
   - 将 reference moment 替换为训练集经验采样矩，验证必须使用标签无关连续积分目标。
   - 将 stencil bank 替换为随机 mask，验证收益来自 cubature feasibility 诊断而非普通增强。

## 5. 预期创新性

1. **从采样概率去偏转向求积公式去偏**：首次把 sampling-policy shift 表述为 continuous evidence integral 的 cubature bias。
2. **从 kernel smoothing 转向 kernel moment exactness**：吸收 MVC-CDE 的核机制启发，但不做普通多视图平滑路径或 solver trace，而是用 RKHS 特征约束离散观测的积分权重。
3. **从 policy branch 分类/对抗转向 control variate**：采样分支只提供求积控制变量，不进入分类头、不估计环境、不计算密度比。
4. **从反事实视图一致转向求积可行性诊断**：counterfactual intervention 只用于检查不同采样 stencil 是否仍可满足参考核矩，不要求 logits 一致。
5. **部署解释性明确**：moment residual、ESS 和 jackknife leverage 可以直接告诉用户预测是否受某个采样政策导致的过密/过稀求积偏差影响。

## 6. 一句话投稿卖点

**RKCD 首次把非规则采样时间序列分类中的 sampling-policy shift 形式化为“离散观测对连续状态证据积分的求积偏差”，并通过 RKHS kernel moment exactness、可微 cubature KKT solver 与 policy control variates，让分类器消费核矩校准后的连续证据积分，而不是训练医院、设备或巡天 cadence 制造的事件计数捷径。**
