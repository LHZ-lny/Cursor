# Title: Do-Boehm Knot Firewall：面向采样策略偏移的病程样条结点防火墙

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化持久记忆 `MEMORIES.md`，并补读当前工作区未落盘的历史 proposal 记忆条目，包括 2026-07-24、2026-07-25、2026-07-26、2026-07-27、2026-07-29、2026-07-31、2026-08-01、2026-08-04、2026-08-07、2026-08-10、2026-08-11、2026-08-21、2026-09-12、2026-09-23 等。
- 已读取/抽取当前仓库 `ideas/Idea_Proposal_*.md` 的历史提案标题、黑名单、核心机制段与最近完整正文，覆盖当前工作区已落盘的 2026-06-12 至 2026-09-26 提案。
- 已分段读取 `paper_daily.md` 的近期记录，重点纳入 2026-09 后段的 RoMAE、SLAN、CHARM、TimeCHEAT、INTERVenE、SLiCE、MedFuse、ReDiTT、DynaMamba、Delta-XAI 与 **TD-HNODE** 等机制。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下主机制：

1. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
2. 生理流/采样算子交换子、state-policy graph 拆分、policy residual sink。
3. protocol tax、additive evidence market、token 边际审计、证据预算。
4. posterior quotient、Radon-Nikodym density ratio、doubly robust correction、RKHS cubature。
5. reconstruction error cartography、ANOVA projection、VQ clauses、HSIC redaction。
6. optional-stopping martingale、拓扑胶囊、policy gauge、syndrome code、knockoff calendar。
7. observability witness、evidential vacuity、policy lattice、solver trace front-door、conformal sleeve、IV/control-function。
8. Borda jury、Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock。
9. trigger-phase hysteresis、control-barrier certificate、regret escrow、principal stratum compiler、Do-IV structural purifier。
10. RG fixed point、bitemporal curtain、clinical tomography、matched risk-set likelihood、policy privacy cloak。
11. synthetic falsification forge、dialectical JEPA referee、PID semantic prism、Noether semantic action、meta workflow immunization。
12. collider seal、source-atlas antipodal retrieval、Schwarzian warp canonicalizer、renewal exposure meter、palimpsest memory。
13. typed SSA observation program、Robin boundary field solver、Blackwell experiment order、elicitable functional compass。
14. Lyapunov interval observer、BBP / Marchenko-Pastur random-matrix spike sieve、RIP sparse pathology recovery。
15. factorial cumulant thinning algebra、repeat/panel cumulant subtraction、Poisson clutter null。
16. Delta-XAI / SWING 风险跃迁熔断、prediction-change attribution consistency、policy interrupt buffer。
17. 单纯 state-policy 双分支、对抗环境分类器、跨视图 logits/representation 一致性、频域掩码对比学习、missingness pattern 直接分类、普通 retrieval-augmented classifier、普通 dual-SSM classifier。

本提案选择新的正交切入点：**不把采样政策建成概率、图、谱、证据预算、随机矩阵、高阶计数、风险跃迁解释或状态空间记忆；而是把非规则观测时间表看成样条 knot vector。采样政策偏移会插入、删除、滑动或提高某些 knots 的重数；真实病程曲线在 Boehm knot insertion / removal 代数下应保持同一条曲线。分类器只读取被投影到 canonical disease-knot basis 的控制点，不能直接读取 policy knot residual。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 的新增机制共同指向一个仍未被历史方案占据的空白：

- **DeepFRC** 提醒我们不规则分类不只是缺点问题，也包含时间相位错位；但它主要做可微时间 warping，历史 proposal 已黑名单化 Schwarzian / phase registration 路线。
- **RoMAE** 表明连续时间坐标可以被通用 Transformer/MAE 强力吸收；但强位置表达能力也会把采样制度作为类别证据。
- **SLiCE / Structured Linear CDE** 让多策略连续时间编码更可扩展；但 control path 仍由观测时间表决定。
- **TD-HNODE** 把慢病进展路径与不规则 encounter 时间耦合起来；它同时暴露一个关键风险：并发症或 marker 的首次记录时间，可能是 disease clock，也可能只是 screening / visit policy。

这些工作仍主要围绕“如何表达不规则时间”或“如何沿不规则时间演化”。本提案换一个问题问：

> 如果同一条潜在病程曲线在医院 A 被 routine round 稀疏观测，在医院 B 被告警后密集复测，在医院 C 被 panel 同步返回，那么这些采样政策到底改变了什么？  
> 它们改变的是 spline knot vector，而不应改变 canonical disease control polygon。

在样条理论中，B-spline 曲线由 knot vector `U`、degree `p` 和 control points `P` 决定：

```text
C(t) = sum_j B_{j,p}(t; U) P_j
```

Boehm knot insertion 告诉我们：向 `U` 中插入一个新 knot，可以得到更细的控制点 `P'`，但表示的曲线 `C(t)` 完全不变：

```text
C(t; U, P) == C(t; U', P'),    U' = U + {new_knot}
P' = R(U -> U') P
```

这正好对应采样策略偏移：

- 告警后密集复测 = 在异常附近插入多个 knots；
- 低频设备 duty cycle = 删除或合并 knots；
- lab panel 同步返回 = 在同一时间点提高 knot multiplicity；
- 医院查房节律 = 将 knots snap 到固定 routine grid；
- follow-up / screening policy 改变 = 某些病程阶段 knots 提前或延后。

如果分类器直接消费 fine-knot control points，它会把“哪里被插入更多 knots”“哪些 knots 具有高重数”“哪些阶段被更密集观测”误当成类别证据。**Do-Boehm Knot Firewall (DBKF)** 的核心主张是：

> 先允许采样政策形成任意 fine knot basis 来拟合观测；再用 Boehm refinement / knot removal algebra 把 fine controls 投影回一个 canonical disease-knot basis。分类头只能读取 canonical controls；fine-knot residual 只能作为 policy diagnostic 或 reject score。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 决定样条曲线在各时间点的观测值；
- sampling process 决定 knot insertion / deletion / multiplicity / slide；
- counterfactual intervention 不再做 logits 一致、风险跃迁解释、谱筛选或高阶计数代数，而是生成 **knot surgery bank**，训练模型识别哪些控制点变化只是 knot refinement artifact。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Knot Surgery Bank

新增 `BoehmKnotCollator`。每个样本返回原始不规则事件流，同时返回一组对时间 knot vector 的可控手术：

```text
event_i = {
    value_i,
    time_i,
    variable_i,
    quality_i,
    event_mask_i,
}
```

`knot_surgery_bank` 包括：

1. `alarm_knot_insert`
   - 在高异常或告警窗口插入多个近邻 knots，但不创造新病程。
   - 模拟复测、床旁监测加密、报警后采样。

2. `routine_knot_snap`
   - 将观测时间吸附到固定查房 / 护理节律 knots。
   - 模拟医院 routine workflow。

3. `device_knot_drop`
   - 删除部分 knots 或合并长 gap，模拟可穿戴低频 duty cycle / 电池策略。

4. `panel_multiplicity`
   - 同一时间点多变量同步返回，提高局部 knot multiplicity。
   - 不应凭同步本身提高分类 margin。

5. `visit_delay_slide`
   - 将 encounter 或 lab-return knots 整体平移，模拟随访延迟、排队、回填。
   - 这里只建模 knot 几何，不复用 bitemporal curtain 或 record-time anti-retrocausal loss。

6. `semantic_event_insert`
   - 插入真正有新 value support 的观测，允许改变 canonical controls。

这些 counterfactual views 不是 contrastive positives，不要求 logits 一致；它们只用于学习 **哪些 fine control changes 可被 Boehm algebra 解释为同一条曲线的 knot refinement，哪些是真实病程曲线改变**。

### 2.2 改 Encoder：Event-to-Spline Control Firewall

#### A. Event Value Stem

先把每个非规则观测事件编码成变量条件化的值 embedding：

```text
e_i = Stem(value_i, variable_i, quality_i)
```

这里可以吸收 MedFuse / CHARM 的 value-feature 语义，但用途不同：

- embedding 不直接进入分类头；
- 它只作为样条拟合的多维观测值 `Y_i`；
- variable description 可以调节不同通道共享哪些 canonical disease controls。

#### B. Fine-Knot Spline Fitter

对每个样本，根据事实或反事实采样时间构造 fine knot vector `U_fine`，用 B-spline design matrix `B_fine(t; U_fine)` 拟合 fine controls：

```text
P_fine = argmin_P || M * (B_fine P - Y) ||_2^2 + lambda ||D2 P||_2^2
```

其中 `D2` 是二阶差分平滑项。这里不是 MVC-CDE kernel smoothing，也不是 FlowPath 可逆路径；它是可微 ridge spline control solving。

#### C. Canonical Disease-Knot Projection

定义一个低重数、跨医院共享的 canonical disease knot vector `U_can`。它可以是：

- 固定的病程阶段 anchors；
- 数据驱动但只由训练集平均 coverage 初始化；
- 或由临床阶段弱标签给定。

用数值 Boehm refinement matrix `R_can_to_fine` 将 canonical controls 映射到 fine basis：

```text
P_fine_expected = R_can_to_fine P_can
```

反向求 `P_can`：

```text
P_can = argmin_P || P_fine - R_can_to_fine P ||_2^2 + lambda ||D2 P||_2^2
```

最终分类器只读取：

```text
logits = Classifier(pool(P_can))
```

#### D. Policy Knot Residual Buffer

fine controls 中无法由 canonical controls 的 Boehm refinement 解释的部分定义为：

```text
E_policy = P_fine - R_can_to_fine P_can
```

`E_policy` 不进入分类头，只输出：

- `knot_policy_score`
- `alarm_insert_residual`
- `panel_multiplicity_residual`
- `visit_delay_residual`
- `coverage_gap_uncertainty`

如果 `E_policy` 在 policy-only surgery 中很大，但 `P_can` 和 logits 也大幅变化，说明模型仍在把 sampling knot geometry 当成疾病证据。

### 2.3 改 Loss：从不变性/去偏转向 Boehm Knot Algebra Discipline

总目标：

```text
L = L_cls
  + lambda_rec  * L_spline_reconstruction
  + lambda_ref  * L_boehm_refinement
  + lambda_rem  * L_policy_knot_removal
  + lambda_qnt  * L_canonical_control_stability
  + lambda_leak * L_knot_leakage_probe
```

#### A. Safe Classification `L_cls`

只用 canonical disease controls 分类：

```text
L_cls = CE(Classifier(pool(P_can_factual)), y)
```

分类头不接收 raw mask、delta-t histogram、event count、fine knot density、knot multiplicity、panel id、hospital id 或 policy residual。

#### B. Spline Reconstruction `L_spline_reconstruction`

fine controls 必须解释观测值：

```text
L_rec = || M * (B_fine P_fine - Y) ||_1
```

这保证模型不是随意生成 latent embedding，而是在当前观测 knots 下形成可重构的连续病程曲线。

#### C. Boehm Refinement Closure `L_boehm_refinement`

对 `alarm_knot_insert`、`routine_knot_snap`、`panel_multiplicity` 等 policy-only knot surgeries，插入或细化 knots 后的 fine controls 应可由同一个 `P_can` 通过 refinement matrix 解释：

```text
L_ref = || P_fine_policy - R_can_to_policy P_can_factual ||_1
```

直觉：如果只是采样政策插入更多 knots，同一条病程曲线不应改变 canonical controls。

#### D. Policy Knot Removal `L_policy_knot_removal`

对 policy-only dense regions，要求高重数 / 高密度 knots 可以被删除而不显著改变 canonical prediction：

```text
remove_err = || C_fine(t_eval) - C_removed(t_eval) ||_1
L_rem = remove_err_policy_only + relu(logit_shift_policy_only - eps)^2
```

这不是 risk-jump 熔断，也不是 regret escrow。它直接使用样条 knot removal 的几何含义：可移除的政策 knots 不应支撑类别 margin。

#### E. Canonical Control Stability `L_canonical_control_stability`

对同一潜在样本的多种采样政策，canonical controls 应在 disease subspace 稳定；但这里不约束所有 representation 或 logits 一致，只约束 **经 Boehm 投影后的控制点**：

```text
L_qnt = || P_can_policy - stopgrad(P_can_factual) ||_1 * policy_only_mask
```

若 `semantic_event_insert` 带来真实新 value support，则允许对应局部 controls 改变，并用 `semantic_insert_mask` 放开约束。

#### F. Knot Leakage Probe `L_knot_leakage_probe`

训练一个审计 probe 用 knot-only features 预测标签：

```text
knot_features = [knot_density, multiplicity, gap_stats, snap_distance, inserted_knot_count]
leak_logits = KnotProbe(stopgrad(knot_features))
L_leak = relu(leak_auc_proxy - tau)^2
```

该 probe 不做对抗分类，也不把 policy branch 接入主模型；它只作为数据与模型审计，报告当前 split 中采样 knot geometry 的标签泄漏强度。

### 2.4 推理阶段

推理时只需要事实观测：

1. 根据实际观测时间构造 `U_fine`。
2. 拟合 `P_fine`。
3. 投影到 canonical `P_can`。
4. 分类器读取 `P_can` 输出 logits。
5. 同时报告 `E_policy` 和 knot coverage uncertainty。

若部署医院的 sampling policy 发生变化，模型不会直接把新增 knots、panel multiplicity 或 routine snap 当成风险证据；这些变化先被尝试解释为 Boehm refinement / removable knots。

---

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_open_uniform_knots(num_ctrl: int, degree: int, device, dtype) -> torch.Tensor:
    """Open uniform knot vector on [0, 1]."""
    num_knots = num_ctrl + degree + 1
    interior = num_knots - 2 * (degree + 1)
    if interior > 0:
        mid = torch.linspace(0.0, 1.0, interior + 2, device=device, dtype=dtype)[1:-1]
        knots = torch.cat(
            [
                torch.zeros(degree + 1, device=device, dtype=dtype),
                mid,
                torch.ones(degree + 1, device=device, dtype=dtype),
            ],
            dim=0,
        )
    else:
        knots = torch.cat(
            [
                torch.zeros(degree + 1, device=device, dtype=dtype),
                torch.ones(degree + 1, device=device, dtype=dtype),
            ],
            dim=0,
        )
    return knots


def bspline_basis(times: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Cox-de Boor basis.

    Args:
        times: [B, T] normalized to [0, 1]
        knots: [K + degree + 1]
    Returns:
        basis: [B, T, K]
    """
    eps = 1e-8
    num_ctrl = knots.numel() - degree - 1
    t = times.unsqueeze(-1)

    left = knots[:-1].view(1, 1, -1)
    right = knots[1:].view(1, 1, -1)
    basis = ((t >= left) & (t < right)).to(times.dtype)
    basis[..., num_ctrl - 1] = torch.where(
        times.eq(1.0),
        torch.ones_like(times),
        basis[..., num_ctrl - 1],
    )

    for p in range(1, degree + 1):
        next_basis = []
        for j in range(num_ctrl):
            left_den = (knots[j + p] - knots[j]).clamp_min(eps)
            right_den = (knots[j + p + 1] - knots[j + 1]).clamp_min(eps)

            left_term = (times - knots[j]) / left_den * basis[..., j]
            if j + 1 < basis.size(-1):
                right_term = (knots[j + p + 1] - times) / right_den * basis[..., j + 1]
            else:
                right_term = torch.zeros_like(left_term)
            next_basis.append(left_term + right_term)
        basis = torch.stack(next_basis, dim=-1)
    return basis


def second_difference_matrix(num_ctrl: int, device, dtype) -> torch.Tensor:
    if num_ctrl < 3:
        return torch.zeros(0, num_ctrl, device=device, dtype=dtype)
    eye = torch.eye(num_ctrl, device=device, dtype=dtype)
    return eye[:-2] - 2.0 * eye[1:-1] + eye[2:]


def ridge_solve(design: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, ridge: float) -> torch.Tensor:
    """
    Batched ridge least squares for spline controls.

    design: [B, T, K]
    target: [B, T, H]
    mask: [B, T]
    returns: [B, K, H]
    """
    weight = mask.to(design.dtype).unsqueeze(-1)
    x = design * weight
    y = target * weight
    xtx = x.transpose(1, 2) @ x
    k = xtx.size(-1)
    eye = torch.eye(k, device=design.device, dtype=design.dtype).unsqueeze(0)
    xty = x.transpose(1, 2) @ y
    return torch.linalg.solve(xtx + ridge * eye, xty)


def numerical_refinement_matrix(
    old_knots: torch.Tensor,
    new_knots: torch.Tensor,
    degree: int,
    num_eval: int = 96,
) -> torch.Tensor:
    """
    Approximate Boehm refinement matrix R such that B_old(t) ~= B_new(t) R.
    In production this can be replaced by exact Boehm/Oslo insertion.
    """
    device, dtype = old_knots.device, old_knots.dtype
    grid = torch.linspace(0.0, 1.0, num_eval, device=device, dtype=dtype).view(1, -1)
    b_old = bspline_basis(grid, old_knots, degree).squeeze(0)  # [E, K_old]
    b_new = bspline_basis(grid, new_knots, degree).squeeze(0)  # [E, K_new]
    ridge = 1e-5 * torch.eye(b_new.size(1), device=device, dtype=dtype)
    # Solve B_new R = B_old.
    return torch.linalg.solve(b_new.T @ b_new + ridge, b_new.T @ b_old)


class EventValueStem(nn.Module):
    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_emb = nn.Embedding(num_vars, hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.quality_proj = nn.Linear(1, hidden_dim)
        self.out = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor, var_id: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        value_feat = torch.stack([values, torch.tanh(values)], dim=-1)
        h = self.var_emb(var_id) * torch.sigmoid(self.value_proj(value_feat))
        h = h + self.quality_proj(quality.unsqueeze(-1))
        return self.out(h)


class BoehmKnotFirewall(nn.Module):
    def __init__(
        self,
        num_vars: int,
        num_classes: int,
        hidden_dim: int = 128,
        degree: int = 3,
        num_canonical_ctrl: int = 12,
        num_fine_ctrl: int = 24,
        ridge: float = 1e-3,
    ):
        super().__init__()
        self.degree = degree
        self.num_canonical_ctrl = num_canonical_ctrl
        self.num_fine_ctrl = num_fine_ctrl
        self.ridge = ridge

        self.stem = EventValueStem(num_vars, hidden_dim)
        self.control_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(num_canonical_ctrl * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(num_fine_ctrl * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _knots(self, num_ctrl: int, device, dtype) -> torch.Tensor:
        return make_open_uniform_knots(num_ctrl, self.degree, device, dtype)

    def fit_controls(
        self,
        times: torch.Tensor,
        values: torch.Tensor,
        var_id: torch.Tensor,
        quality: torch.Tensor,
        mask: torch.Tensor,
        num_ctrl: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stem(values, var_id, quality)
        knots = self._knots(num_ctrl, times.device, times.dtype)
        design = bspline_basis(times.clamp(0.0, 1.0), knots, self.degree)
        controls = ridge_solve(design, h, mask, self.ridge)
        return controls, design, knots

    def project_to_canonical(self, fine_controls: torch.Tensor, fine_knots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        can_knots = self._knots(self.num_canonical_ctrl, fine_knots.device, fine_knots.dtype)
        refine = numerical_refinement_matrix(can_knots, fine_knots, self.degree)  # [K_fine, K_can]
        design = refine.unsqueeze(0).expand(fine_controls.size(0), -1, -1)
        mask = torch.ones(fine_controls.size(0), fine_controls.size(1), device=fine_controls.device, dtype=fine_controls.dtype)
        canonical = ridge_solve(design, fine_controls, mask, self.ridge)
        expected_fine = design @ canonical
        residual = fine_controls - expected_fine
        return canonical, residual

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fine_controls, design, fine_knots = self.fit_controls(
            batch["event_time"],
            batch["event_value"],
            batch["event_var_id"],
            batch.get("measurement_quality", torch.ones_like(batch["event_value"])),
            batch["event_mask"],
            self.num_fine_ctrl,
        )
        canonical, residual = self.project_to_canonical(fine_controls, fine_knots)
        canonical = self.control_norm(canonical)
        logits = self.classifier(canonical.flatten(start_dim=1))
        policy_score = self.policy_head(residual.flatten(start_dim=1)).squeeze(-1)
        recon = design @ fine_controls
        return {
            "logits": logits,
            "fine_controls": fine_controls,
            "canonical_controls": canonical,
            "policy_residual": residual,
            "policy_score": policy_score,
            "reconstruction": recon,
            "design": design,
            "fine_knots": fine_knots,
        }


class BoehmKnotLoss(nn.Module):
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_refine: float = 0.5,
        lambda_remove: float = 0.2,
        lambda_stability: float = 0.5,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_refine = lambda_refine
        self.lambda_remove = lambda_remove
        self.lambda_stability = lambda_stability

    def forward(
        self,
        factual: dict[str, torch.Tensor],
        policy_view: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        labels = batch["label"]
        loss_cls = F.cross_entropy(factual["logits"], labels)

        target = factual.get("event_embedding_target")
        if target is None:
            target = factual["reconstruction"].detach()
        mask = batch["event_mask"].unsqueeze(-1).to(factual["reconstruction"].dtype)
        loss_rec = ((factual["reconstruction"] - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)

        # Policy-only knot surgeries should be explained as refinement residual,
        # not as canonical disease-control movement.
        policy_only = batch.get("policy_only_surgery", torch.ones(labels.size(0), device=labels.device))
        while policy_only.dim() < factual["canonical_controls"].dim():
            policy_only = policy_only.unsqueeze(-1)

        loss_stability = (
            (policy_view["canonical_controls"] - factual["canonical_controls"].detach()).abs()
            * policy_only
        ).mean()

        loss_refine = (policy_view["policy_residual"].abs() * policy_only).mean()

        logit_shift = (policy_view["logits"] - factual["logits"].detach()).pow(2).sum(dim=-1).sqrt()
        loss_remove = (logit_shift * batch.get("removable_knot_mask", torch.ones_like(logit_shift))).mean()

        total = (
            loss_cls
            + self.lambda_rec * loss_rec
            + self.lambda_refine * loss_refine
            + self.lambda_remove * loss_remove
            + self.lambda_stability * loss_stability
        )
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_rec": loss_rec,
            "loss_refine": loss_refine,
            "loss_remove": loss_remove,
            "loss_stability": loss_stability,
        }
```

### Collator 草稿

```python
def boehm_knot_surgery_view(batch: dict, recipe: str) -> dict:
    """
    Sketch only. Real implementation should preserve raw values while changing
    observation times / masks according to deployable sampling-policy recipes.
    """
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    time = out["event_time"]
    mask = out["event_mask"]

    if recipe == "alarm_knot_insert":
        # Duplicate high-surprise observations by inserting near-identical times.
        # Values stay unchanged; this should be removable by the knot firewall.
        surprise = out.get("value_surprise", torch.zeros_like(time))
        pick = (surprise > surprise.quantile(0.75, dim=1, keepdim=True)).to(mask.dtype) * mask
        out["event_time"] = (time + 0.002 * pick).clamp(0.0, 1.0)
        out["policy_only_surgery"] = torch.ones(time.size(0), device=time.device)
        out["removable_knot_mask"] = torch.ones(time.size(0), device=time.device)

    elif recipe == "routine_knot_snap":
        grid = torch.round(time * 24.0) / 24.0
        out["event_time"] = torch.where(mask.bool(), grid, time)
        out["policy_only_surgery"] = torch.ones(time.size(0), device=time.device)
        out["removable_knot_mask"] = torch.ones(time.size(0), device=time.device)

    elif recipe == "device_knot_drop":
        keep = ((torch.arange(time.size(1), device=time.device) % 2) == 0).view(1, -1)
        out["event_mask"] = mask * keep.to(mask.dtype)
        out["policy_only_surgery"] = torch.ones(time.size(0), device=time.device)
        out["removable_knot_mask"] = torch.ones(time.size(0), device=time.device)

    elif recipe == "semantic_event_insert":
        # A true new value-support edit; canonical controls are allowed to move.
        out["policy_only_surgery"] = torch.zeros(time.size(0), device=time.device)
        out["removable_knot_mask"] = torch.zeros(time.size(0), device=time.device)

    else:
        raise ValueError(f"unknown knot surgery recipe: {recipe}")
    return out
```

---

## 4. 实验切入点

1. **跨采样政策鲁棒性**
   - P12 / P19 / MIMIC-IV / eICU / PhysioNet Sepsis。
   - 构造 routine、alarm-burst、panel、low-duty-cycle、visit-delay 五类反事实采样政策。
   - 报告 in-policy、cross-policy、counterfactual-policy AUROC / AUPRC。

2. **Knot-only leakage probe**
   - 只用 knot density、gap distribution、multiplicity、snap distance、inserted-knot count 训练分类器。
   - 若 probe 很强，说明数据中存在采样时间表 shortcut；DBKF 应显著降低主模型对这些特征的依赖。

3. **Boehm closure ablation**
   - 去掉 `L_boehm_refinement`。
   - 去掉 `L_policy_knot_removal`。
   - 让 classifier 直接读取 `P_fine`。
   - 将 exact / numerical Boehm matrix 替换为普通 MLP projection。

4. **与近期 paper mechanism 的对比**
   - RoMAE continuous-position MAE：检查其位置编码是否泄漏 policy。
   - SLAN switch model：检查 switch frequency 是否支撑类别 margin。
   - TimeCHEAT / SLiCE / MedFuse：作为强 irregular encoder baseline。
   - TD-HNODE-style Neural ODE/hypergraph：比较 visit schedule 改变时 disease-marker logits 是否稳定。

5. **可解释性输出**
   - 展示 canonical disease control polygon 与 policy knot residual。
   - 对 alarm-burst 与 panel-multiplicity 环境，显示新增 fine knots 被 residual buffer 吸收，而 canonical controls 基本不动。

---

## 5. 预期创新性

1. **从 sampling time encoding 转向 knot algebra**  
   历史方案大多把时间戳当特征、图边、控制路径、风险跃迁或观测概率；DBKF 把采样时间表提升为 spline knot vector，并用 Boehm insertion/removal 的已知代数约束它。

2. **从反事实一致性转向曲线等价性**  
   不要求不同采样视图 logits 或 representation 一致；只要求 policy-only knot surgery 不改变同一条 latent disease curve 的 canonical control polygon。

3. **从插值/平滑转向控制点防火墙**  
   普通 spline / CDE / kernel smoothing 会直接把采样 knots 变成路径几何；DBKF 允许 fine knots 拟合观测，但用 canonical-knot projection 防止政策 knots 支撑分类 margin。

4. **自然解释 TD-HNODE 暴露的问题**  
   首次记录时间、visit interval 和 marker timing 可以被看作 observation-knot geometry。DBKF 将其分解为 canonical disease controls 与 policy knot residual，避免把筛查/随访政策当作病程进展。

5. **低侵入兼容现有框架**  
   现有 value encoder、counterfactual sampler、irregular backbone 都可保留；只需在事件表征与分类头之间加入 spline control solver 与 knot firewall。

## 6. 一句话投稿卖点

**Do-Boehm Knot Firewall 首次把非规则采样策略偏移表述为样条 knot vector 的插入、删除、滑动与重数变化，通过 Boehm knot refinement / removal 代数把政策 knots 隔离为 residual buffer，让分类器只读取 canonical disease control polygon，从而在不依赖概率去偏、图解耦、谱筛选、累积量或风险跃迁解释的前提下获得采样策略鲁棒性。**
