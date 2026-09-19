# Title: Do-Robin Boundary Immunizer：面向采样策略偏移的语义-时间边界免疫分类器

## 0. 强制读取记录与思维黑名单

### 已读取与检索材料

- 已尝试读取 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已检索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化持久记忆 `MEMORIES.md` 及 2026-07-24 至 2026-09-18 的历史提案记忆条目。
- 已读取/抽取当前仓库 `ideas/Idea_Proposal_*.md` 的全部历史提案标题、核心机制与多份完整正文。
- 已读取近期 `paper_daily_2026-09-14.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`，并从兼容入口 `paper_daily.md` 抽取最新论文索引与 Sampling-Policy Shift 启发。

### 历史核心机制黑名单

本提案显式避开以下已出现机制，避免与历史方向发生思维重合：

1. 危险率 point-process scorer、hazard-driven 反事实重采样、分类梯度与采样 score 零空间正交。
2. 生理流/采样算子交换子、图交换子、policy residual sink。
3. additive evidence market、protocol tax、token 边际审计、证据预算。
4. 模型空间后验商、采样似然因子相除、干预积分分类。
5. 重构误差地图、ANOVA state/policy error projection、VQ 语义 clauses、HSIC redaction。
6. policy-simplex randomized smoothing、认证半径、logit-normal / Dirichlet do-sampler。
7. Radon-Nikodym 测度比、doubly robust target-measure correction、RKHS cubature。
8. previsible martingale query、optional-stopping moment control、停时 recipe。
9. excursion topology capsule、censored persistence interval、topology envelope。
10. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
11. policy-only negative film、shadow eraser/stencil。
12. latent packet codeword、parity-check、syndrome locator、packet repair。
13. calendar knockoff、soft knockoff-FDR、swap symmetry。
14. observability witness、evidential shield、policy-lattice submodular margins、solver-trace front-door。
15. measurement-action bisimulation、signature renormalizer、thermodynamic free-energy、copula/rank marginal stripping。
16. triage queue debt、Sinkhorn detail canonicalizer、MDL episode transducer、causal sheaf gluing。
17. trigger-phase hysteresis、control barrier certificate、regret escrow router、principal-stratum compiler。
18. conformal risk sleeves、IV structural purifier、jury rank tribunal、Krylov mode annihilator、determinantal/Nystrom volume basis、tropical support routes。
19. fixed clinical viva、temporal sequent proof、disease-progress poset clock、feasible hull、IRT-DIF、RG fixed point。
20. bitemporal causal curtain、Kaczmarz clinical tomography、matched risk-set conditional likelihood、sampling differential privacy cloak、synthetic falsification forge。
21. dialectical JEPA referee、PID semantic prism、meta workflow immunization、collider seal transformer、source-atlas antipodal adapter。
22. Schwarzian warp conditioner、renewal exposure meter、palimpsest homeostatic memory、typed SSA observation IR / semantic switchboard / dead-code elimination。
23. 将 RoMAE 连续 RoPE、SLAN switch layer、CHARM channel semantics 或 ECG latent ODE 直接作为主创新而不做 sampling-policy 因果改造。

本提案选择新的正交切入点：**不把采样政策当作要删除、投影、征税、证明、校准、隐私化、编译或记忆疲劳的对象；而是把不规则观测看成在“语义通道 × 连续时间”域上给潜在病理场施加的边界条件。采样政策可以改变哪些边界点被夹持、哪些边界只提供通量约束，但不应改变病理场在域内部满足的语义-时间方程。分类器只读取由 Robin 边界免疫求解器恢复的 interior pathology field，而不是直接读取采样开关、日历密度或通道可见性。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中有三条新的启发很适合与当前“采样解耦/反事实干预”框架结合：

- **RoMAE** 说明连续位置编码能让通用 Transformer 处理不规则多变量坐标，但也提醒我们：位置表达能力越强，越容易吸收巡天 cadence、医院 routine 或设备 duty cycle。
- **SLAN** 说明非插补式 switch 更新比伪造缺失值更适合临床 IMTS；但 switch 何时打开本身可能是医院政策，而不一定是病理状态。
- **CHARM** 说明通道语义描述能提升异构传感器迁移；但通道命名、单位、设备配置和联测习惯也可能混入部署政策。

这些机制共同指向一个更底层的问题：

> 非规则观测不是完整场本身，而是对一个潜在语义-时间病理场的边界读取。训练医院决定了哪些边界点被读取、何时被读取、以何种通道语义被读取；测试医院换一套边界条件后，普通模型会把边界几何误当成类别证据。

**Do-Robin Boundary Immunizer (DRBI)** 将 sampling-policy shift 重新表述为 **边界条件偏移**：

- 被真实测到的值是 Dirichlet 边界夹持：`u(t, channel)=observed value`；
- 只知道某变量被 routine 打开、pending、低频读取或设备 duty-cycle 限制时，是 Neumann/Robin 式的 policy flux 约束；
- 真正可迁移的病理证据不是“哪些边界点出现了”，而是由这些边界条件恢复出的域内病理场 `u_state(t, semantic_channel)`。

这与历史提案显著不同：DRBI 不估计采样概率，不做多视图 logits 一致，不输出证据税，不做后验除法，不求解 graph commutator，不做 SSA 程序死代码消除。它把反事实干预模块用于改变**边界条件类型**，训练一个对边界采样政策稳定的语义-时间场求解器。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从事件池化改为 Semantic-Time Boundary Field Solver

给定不规则事件流：

```text
E = {(t_i, d_i, x_i, unit_i, device_i)}_{i=1}^N
```

DRBI 先构造一个连续的语义-时间坐标：

```text
c_i = ChannelSemanticEncoder(description_d, unit_d, device_d)
p_i = ContinuousTimePosition(t_i)
xi_i = [p_i, c_i]
```

然后在语义-时间域上学习一个病理场：

```text
u_theta(xi) : semantic-time coordinate -> latent pathology field
```

事件值不直接被池化进分类器，而是作为边界条件约束这个场：

1. **Dirichlet Value Boundary**
   - 对真实观测值，要求 `u_theta(xi_i)` 能解释 value embedding。
   - 这继承 SLAN 的“只有观测到才更新”的精神，但不让 switch 激活次数成为分类证据。

2. **Robin Policy Boundary**
   - sampling branch 只看时间、通道可见性、单位/设备/联测描述，输出边界混合系数 `alpha_i` 和 policy flux `g_i`：

```text
alpha_i * u(xi_i) + (1 - alpha_i) * normal_grad u(xi_i) = g_i
```

   - `alpha_i, g_i` 不进入分类头，只描述当前采样政策给病理场施加了什么边界压力。

3. **Green Kernel Interior Readout**
   - 在一组 learnable / semantic anchor interior points 上，用可微 Green kernel 从边界条件恢复域内场。
   - 分类器只读取 interior field summary，而不是读取原始 mask、switch count、policy code 或 boundary flux。

### 2.2 改 Loss：从一致性/投影转向 Robin 边界免疫

总目标：

```text
L = L_cls
  + lambda_bc   * L_boundary_fit
  + lambda_pde  * L_interior_equation
  + lambda_do   * L_robin_intervention
  + lambda_flux * L_policy_flux_sobriety
```

#### A. 分类损失 `L_cls`

分类器只接收 interior pathology field：

```text
U_int = GreenSolver(value_boundary, robin_policy_boundary, anchors)
logits = Classifier(pool(U_int))
L_cls = CE(logits, y)
```

这里不是固定问答表，也不是 clinical viva anchor；anchor 是语义-时间域中的求解点，用来恢复连续场。

#### B. Boundary Fit `L_boundary_fit`

观测值应能作为 Dirichlet 边界被场解释：

```text
L_boundary_fit = mean_i || ProjectValue(u(xi_i)) - EmbedValue(x_i, d_i) ||^2
```

这不是 reconstruction-error pseudo observation；它只约束边界处的场与真实观测一致，分类仍来自 interior field。

#### C. Interior Equation `L_interior_equation`

对 interior anchors，约束病理场满足一个语义-时间椭圆型平滑方程：

```text
L_interior_equation = || L_semantic_time u - q_value ||^2
```

其中 `L_semantic_time` 是由通道语义和连续时间坐标诱导的核 Laplacian，`q_value` 是 value branch 产生的源项。关键点是：采样政策只能改变边界读取，不应改变域内方程本身。

#### D. Robin Intervention Loss `L_robin_intervention`

当前反事实干预模块生成边界条件改写，而不是生成一致性正样本：

- `dirichlet_to_robin`：某变量从真实数值观测变成只知道被低频读取；
- `robin_to_dirichlet`：某通道在反事实设备中增加少量真实读取；
- `semantic_unit_swap`：同一生理量用不同单位/设备名称记录；
- `panel_boundary_split`：panel 联测从同一边界簇拆成异步边界点。

对每个反事实边界，DRBI 不要求 logits 或 representation 完全一致，只要求**域内方程残差与边界解释是自洽的**：

```text
L_robin_intervention =
  mean_r || L u_r - q_value ||^2
  + mean_r || RobinResidual(u_r, boundary_r) ||^2
```

直觉：采样政策可以改变“边界条件怎样写”，但如果模型恢复的是同一个病理场，其域内方程不应被 policy-only boundary perturbation 改写。

#### E. Policy Flux Sobriety `L_policy_flux_sobriety`

防止 sampling branch 把标签信息藏进 flux：

```text
L_policy_flux_sobriety =
  ||g_policy||_2^2
  + relu(corr(g_policy, stopgrad(label_margin)) - eps)^2
```

这不是 adversarial environment classifier；它不预测医院或策略标签，也不反向欺骗分类表示。它只限制 Robin flux 成为隐藏分类通道。

### 2.3 改 Dataloader：返回 Boundary-Condition Bank

新增 `BoundaryConditionCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `channel_description_id` 或离线文本 embedding：来自 CHARM 式通道语义。
3. `unit_id`、`device_id`、`panel_id`：只用于构造边界类型，不直接进入分类头。
4. `boundary_type`：Dirichlet / Robin / low-rate Robin / pending Robin。
5. `boundary_recipe_bank`：反事实边界改写 recipe。
6. `interior_anchor_coord`：语义-时间域中的求解点。

与历史机制的区别：

- 不是 SLAN 式 switch 激活分类；switch 只决定哪些边界点存在。
- 不是 CHARM 式 channel-aware JEPA；通道语义只定义域坐标和核，不做 latent debate 或语义棱镜。
- 不是 RoMAE 式连续 RoPE/MAE；连续位置只参与边界值场求解，不做 masked reconstruction pretraining。
- 不是 SSA switchboard；没有程序 IR、def-use、liveness 或 dead-code elimination。
- 不是 Kaczmarz tomography；不是用 observation rays 重建字段，而是用 Dirichlet/Robin 边界条件求解语义-时间场。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value process 改为 `BoundaryValueEncoder`：把观测值投影成 Dirichlet 边界源项。
- 现有 sampling process 改为 `RobinPolicyBoundaryHead`：只输出边界混合系数与 flux，不能进入分类器。
- 现有 counterfactual intervention 改为 `BoundaryRewriteBank`：生成不同采样政策下的边界类型/单位/panel 改写。
- 推理阶段只需事实观测：先构造边界条件，再恢复 interior pathology field 并分类。
- 可解释输出包括：
  - 每个变量/时间窗的 Dirichlet vs Robin 边界占比；
  - policy flux norm；
  - interior equation residual；
  - 哪些预测依赖边界几何而非域内病理场。

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


class SemanticTimeCoordinate(nn.Module):
    """Build semantic-time coordinates for boundary points."""

    def __init__(self, num_vars: int, num_units: int, coord_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, coord_dim)
        self.unit_embed = nn.Embedding(num_units, coord_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(2, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
        )
        self.mix = nn.Sequential(
            nn.Linear(3 * coord_dim, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
        )

    def forward(
        self,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        unit_id: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = event_time / horizon
        delta_t = torch.zeros_like(event_time)
        delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
        time_h = self.time_proj(torch.stack([time_norm, torch.log1p(delta_t)], dim=-1))
        var_h = self.var_embed(event_var_id.clamp_min(0))
        unit_h = self.unit_embed(unit_id.clamp_min(0))
        coord = self.mix(torch.cat([time_h, var_h, unit_h], dim=-1))
        return F.normalize(coord, dim=-1) * event_mask.unsqueeze(-1)


class BoundaryValueEncoder(nn.Module):
    """Encode observed values as Dirichlet boundary sources."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, event_value: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        value_x = torch.stack([event_value, quality], dim=-1)
        return self.net(value_x)


class RobinPolicyBoundaryHead(nn.Module):
    """Estimate policy boundary coefficients without exposing them to the classifier."""

    def __init__(self, coord_dim: int, hidden_dim: int, recipe_dim: int):
        super().__init__()
        self.recipe_proj = nn.Linear(recipe_dim, coord_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * coord_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.alpha = nn.Linear(hidden_dim, 1)
        self.flux = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        coord: torch.Tensor,
        event_time: torch.Tensor,
        event_mask: torch.Tensor,
        boundary_type: torch.Tensor,
        recipe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # boundary_type: [B, N] with larger values indicating more policy-defined boundaries.
        horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = event_time / horizon
        delta_t = torch.zeros_like(event_time)
        delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
        density = 1.0 / (1.0 + delta_t)
        recipe_h = self.recipe_proj(recipe)[:, None, :].expand_as(coord)
        stats = torch.stack([time_norm, torch.log1p(delta_t), density, boundary_type.float()], dim=-1)
        h = self.net(torch.cat([coord, recipe_h, stats], dim=-1))
        alpha = torch.sigmoid(self.alpha(h)).squeeze(-1) * event_mask
        flux = self.flux(h) * event_mask.unsqueeze(-1)
        return alpha, flux


class GreenKernelSolver(nn.Module):
    """Recover interior pathology fields from boundary values and Robin flux."""

    def __init__(self, coord_dim: int, hidden_dim: int, num_anchors: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(num_anchors, coord_dim) * 0.05)
        self.log_length = nn.Parameter(torch.zeros(coord_dim))
        self.source = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [B, M, C], y: [B, N, C]
        length = F.softplus(self.log_length).view(1, 1, 1, -1) + 1e-3
        diff = (x[:, :, None, :] - y[:, None, :, :]) / length
        return torch.exp(-0.5 * diff.pow(2).sum(dim=-1))

    def forward(
        self,
        coord: torch.Tensor,
        boundary_value: torch.Tensor,
        alpha: torch.Tensor,
        flux: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> dict:
        bsz = coord.size(0)
        anchors = F.normalize(self.anchor, dim=-1)[None].expand(bsz, -1, -1)
        k_ab = self.kernel(anchors, coord) * event_mask[:, None, :]
        k_norm = k_ab / k_ab.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # Dirichlet values and Robin flux are blended as boundary sources.
        boundary_source = alpha.unsqueeze(-1) * boundary_value + (1.0 - alpha).unsqueeze(-1) * flux
        interior = torch.einsum("ban,bnh->bah", k_norm, self.source(boundary_source))

        # One recurrent relaxation step makes the layer behave like a small neural PDE solver.
        flat_old = interior.reshape(-1, interior.size(-1))
        flat_new = self.update(flat_old, torch.zeros_like(flat_old))
        interior = flat_new.view_as(interior)

        return {"interior": interior, "anchors": anchors, "kernel": k_norm}


def interior_equation_loss(interior: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Penalize high semantic-time Laplacian residual over anchor fields."""

    # Anchor-to-boundary kernel induces a smoothness proxy: nearby boundary support
    # should not produce abrupt anchor field jumps.
    anchor_affinity = torch.bmm(kernel, kernel.transpose(1, 2))
    degree = anchor_affinity.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    smoothed = torch.bmm(anchor_affinity, interior) / degree
    return (interior - smoothed).pow(2).mean()


def boundary_fit_loss(
    coord_field: torch.Tensor,
    boundary_value: torch.Tensor,
    event_mask: torch.Tensor,
) -> torch.Tensor:
    raw = (coord_field - boundary_value).pow(2).sum(dim=-1)
    return (raw * event_mask).sum() / event_mask.sum().clamp_min(1.0)


def flux_sobriety_loss(
    flux: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    event_mask: torch.Tensor,
    eps: float = 0.05,
) -> torch.Tensor:
    flux_norm = (flux.pow(2).sum(dim=-1) * event_mask).sum() / event_mask.sum().clamp_min(1.0)
    margin = logits.gather(1, labels[:, None]).squeeze(1)
    rival = logits.masked_fill(F.one_hot(labels, logits.size(-1)).bool(), -1e4).max(dim=-1).values
    label_margin = (margin - rival).detach()
    flux_summary = masked_mean(flux.norm(dim=-1), event_mask, dim=1)
    flux_summary = flux_summary - flux_summary.mean()
    label_margin = label_margin - label_margin.mean()
    corr = (flux_summary * label_margin).mean()
    corr = corr / torch.sqrt(flux_summary.pow(2).mean() * label_margin.pow(2).mean() + 1e-6)
    return 0.01 * flux_norm + F.relu(corr.abs() - eps).pow(2)


class DoRobinBoundaryImmunizer(nn.Module):
    """Sampling-policy robust classifier via semantic-time Robin boundary solving."""

    def __init__(
        self,
        num_vars: int,
        num_units: int,
        coord_dim: int,
        hidden_dim: int,
        num_classes: int,
        recipe_dim: int,
        num_anchors: int = 32,
    ):
        super().__init__()
        self.coord = SemanticTimeCoordinate(num_vars, num_units, coord_dim)
        self.value = BoundaryValueEncoder(hidden_dim)
        self.policy_boundary = RobinPolicyBoundaryHead(coord_dim, hidden_dim, recipe_dim)
        self.solver = GreenKernelSolver(coord_dim, hidden_dim, num_anchors)
        self.boundary_projector = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def solve_with_recipe(self, batch: dict, recipe: torch.Tensor) -> dict:
        coord = self.coord(
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
            unit_id=batch["unit_id"],
            event_mask=batch["event_mask"],
        )
        boundary_value = self.value(
            event_value=batch["event_value"],
            quality=batch.get("event_quality", torch.ones_like(batch["event_value"])),
        )
        alpha, flux = self.policy_boundary(
            coord=coord,
            event_time=batch["event_time"],
            event_mask=batch["event_mask"],
            boundary_type=batch["boundary_type"],
            recipe=recipe,
        )
        solved = self.solver(
            coord=coord,
            boundary_value=boundary_value,
            alpha=alpha,
            flux=flux,
            event_mask=batch["event_mask"],
        )
        pooled = solved["interior"].mean(dim=1)
        logits = self.classifier(pooled)
        solved.update({
            "coord": coord,
            "boundary_value": boundary_value,
            "alpha": alpha,
            "flux": flux,
            "logits": logits,
        })
        return solved

    def forward(self, batch: dict) -> dict:
        zero_recipe = torch.zeros(
            batch["event_value"].size(0),
            batch["boundary_recipe_bank"].size(-1),
            device=batch["event_value"].device,
            dtype=batch["event_value"].dtype,
        )
        return self.solve_with_recipe(batch, zero_recipe)

    def training_loss(
        self,
        batch: dict,
        lambda_bc: float = 0.2,
        lambda_pde: float = 0.1,
        lambda_do: float = 0.2,
        lambda_flux: float = 0.05,
    ) -> dict:
        labels = batch["labels"]
        factual = self.forward(batch)
        cls_loss = F.cross_entropy(factual["logits"], labels)

        # Approximate boundary field by projecting nearest interior support back to events.
        event_support = torch.bmm(factual["kernel"].transpose(1, 2), factual["interior"])
        bc_loss = boundary_fit_loss(
            coord_field=self.boundary_projector(event_support),
            boundary_value=factual["boundary_value"],
            event_mask=batch["event_mask"],
        )
        pde_loss = interior_equation_loss(factual["interior"], factual["kernel"])

        do_terms = []
        for recipe, boundary_type in zip(
            batch["boundary_recipe_bank"].unbind(dim=1),
            batch["boundary_type_bank"].unbind(dim=1),
        ):
            cf_batch = dict(batch)
            cf_batch["boundary_type"] = boundary_type
            cf = self.solve_with_recipe(cf_batch, recipe)
            robin_residual = (
                cf["alpha"].unsqueeze(-1) * cf["boundary_value"]
                + (1.0 - cf["alpha"]).unsqueeze(-1) * cf["flux"]
            )
            robin_residual = (robin_residual.pow(2).sum(dim=-1) * cf_batch["event_mask"]).sum()
            robin_residual = robin_residual / cf_batch["event_mask"].sum().clamp_min(1.0)
            do_terms.append(interior_equation_loss(cf["interior"], cf["kernel"]) + 0.01 * robin_residual)
        do_loss = torch.stack(do_terms).mean() if do_terms else torch.zeros((), device=cls_loss.device)

        flux_loss = flux_sobriety_loss(factual["flux"], factual["logits"], labels, batch["event_mask"])

        total = (
            cls_loss
            + lambda_bc * bc_loss
            + lambda_pde * pde_loss
            + lambda_do * do_loss
            + lambda_flux * flux_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "boundary_fit_loss": bc_loss.detach(),
            "interior_equation_loss": pde_loss.detach(),
            "robin_intervention_loss": do_loss.detach(),
            "policy_flux_sobriety": flux_loss.detach(),
            "mean_robin_alpha": factual["alpha"].mean().detach(),
        }


@torch.no_grad()
def build_boundary_condition_bank(batch: dict, recipe_dim: int = 4) -> dict:
    """Create counterfactual boundary rewrites for DRBI.

    boundary_type convention:
      0 = observed Dirichlet value
      1 = low-rate Robin boundary
      2 = pending / delayed Robin boundary
      3 = panel split / unit swap Robin boundary
    """

    event_time = batch["event_time"]
    event_mask = batch["event_mask"]
    event_var_id = batch["event_var_id"]
    bsz, num_events = event_time.shape
    device = event_time.device

    recipes = []
    boundary_types = []
    base_type = batch.get("boundary_type", torch.zeros_like(event_time))

    # Recipe 1: low-rate device boundary in late window.
    horizon = event_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
    late = (event_time / horizon > 0.66) & (event_mask > 0)
    r1 = torch.zeros(bsz, recipe_dim, device=device)
    r1[:, 0] = 1.0
    recipes.append(r1)
    boundary_types.append(torch.where(late, torch.ones_like(base_type), base_type))

    # Recipe 2: pending boundary for alternating events.
    alternating = (torch.arange(num_events, device=device)[None] % 2 == 1) & (event_mask > 0)
    r2 = torch.zeros(bsz, recipe_dim, device=device)
    r2[:, 1] = 1.0
    recipes.append(r2)
    boundary_types.append(torch.where(alternating, torch.full_like(base_type, 2), base_type))

    # Recipe 3: panel split boundary for high-frequency variables.
    repeated = event_var_id[:, 1:] == event_var_id[:, :-1]
    repeated = F.pad(repeated, (1, 0), value=False) & (event_mask > 0)
    r3 = torch.zeros(bsz, recipe_dim, device=device)
    r3[:, 2] = 1.0
    recipes.append(r3)
    boundary_types.append(torch.where(repeated, torch.full_like(base_type, 3), base_type))

    # Recipe 4: Dirichlet restoration probe for sparse early events.
    early = (event_time / horizon < 0.33) & (event_mask > 0)
    r4 = torch.zeros(bsz, recipe_dim, device=device)
    r4[:, 3] = 1.0
    recipes.append(r4)
    boundary_types.append(torch.where(early, torch.zeros_like(base_type), base_type))

    batch["boundary_recipe_bank"] = torch.stack(recipes, dim=1)
    batch["boundary_type_bank"] = torch.stack(boundary_types, dim=1)
    batch["boundary_type"] = base_type
    return batch
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `dirichlet-to-robin shift`：训练环境有真实数值读数，测试环境只留下低频/摘要/pending 记录。
   - `unit-device boundary shift`：同一生理量由不同单位或设备通道记录，考察通道语义坐标是否稳定。
   - `panel boundary split`：训练环境同步 panel 作为一个密集边界簇，测试环境拆成异步边界点。
   - `duty-cycle boundary thinning`：可穿戴设备或 ICU 监测因电量/流程改变边界读取频率。

2. **对比方法**
   - imputation baseline、mask dropout、missingness-aware encoder。
   - SLAN / switch-layer 非插补模型。
   - RoMAE 连续位置 MAE。
   - CHARM channel-aware representation。
   - 历史方案中的 DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、DSSS 等。

3. **核心指标**
   - in-policy accuracy 与 worst-policy accuracy。
   - boundary-shift calibration error。
   - interior equation residual under counterfactual boundaries。
   - policy flux reliance：错误预测是否伴随异常高 policy flux norm。
   - Dirichlet/Robin robustness gap：数值边界被替换为 policy boundary 后性能下降幅度。

4. **消融实验**
   - 去掉 Robin boundary，只用 Dirichlet value pooling，验证是否退化为 switch 激活捷径。
   - 去掉 interior equation loss，验证分类器是否直接读边界几何。
   - 去掉 flux sobriety，验证 policy flux 是否携带标签信息。
   - 将通道语义坐标替换为随机变量 id，验证 CHARM 式语义通道对跨设备迁移的贡献。
   - 将 boundary rewrite 替换为随机 mask，验证收益来自边界条件改写而非普通增强。

## 5. 预期创新性

1. **从采样策略去偏转向边界条件免疫**：采样政策不再被当作 nuisance feature 删除，而被解释为潜在病理场的边界条件变化。
2. **从 switch 激活转向 Dirichlet/Robin 边界类型**：吸收 SLAN 的非插补思想，但避免把“哪些 switch 打开”直接变成分类证据。
3. **从连续位置编码转向语义-时间场求解**：吸收 RoMAE 的连续坐标启发，但不做 masked reconstruction 或位置对齐；坐标用于定义病理场域。
4. **从通道语义表示转向边界核几何**：吸收 CHARM 的通道描述思想，但不做 JEPA 辩论、PID 或语义程序类型检查；通道语义只决定 Green kernel 与边界传播。
5. **从反事实采样视图转向边界改写自洽性**：counterfactual intervention 不要求 logits 一致，不做认证、保形、投票、证明或程序消除，只检查 policy-only 边界改写是否保持同一 interior equation。

## 6. 一句话投稿卖点

**DRBI 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“潜在语义-时间病理场的边界条件偏移”问题，并通过 Dirichlet/Robin 边界分解、Green kernel interior field solver 与反事实边界改写自洽损失，让分类器依赖域内病理场而不是依赖训练医院或设备制造的采样开关、边界密度、panel 同步和通道可见性捷径。**
