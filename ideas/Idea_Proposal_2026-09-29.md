# Title: Do-Cauchy Residue Firewall：面向采样策略偏移的病程奇点留数防火墙

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化持久记忆 `MEMORIES.md`，并补读最新 `idea_2026-09-28.md`、`idea_2026-09-27.md`、`idea_2026-09-26.md`、`idea_2026-09-25.md`、`idea_2026-09-24.md`、`idea_2026-09-23.md` 等历史 proposal 摘要。
- 已读取/检索当前仓库 `ideas/Idea_Proposal_*.md` 的全部历史标题、黑名单与核心机制段，覆盖当前工作区已落盘的 2026-06-12 至 2026-09-28 提案。
- 已读取近期 `paper_daily_2026-09-27.md`、`paper_daily_2026-09-26.md`、`paper_daily_2026-09-25.md`、`paper_daily_2026-09-24.md`、`paper_daily_2026-09-22.md`，并从兼容入口 `paper_daily.md` 抽取近期机制；本轮重点吸收：
  - **TD-HNODE** 对 disease clock 与 observation clock 混叠的提醒；
  - **SLiCE / Structured Linear CDEs** 对可并行连续时间编码和多策略视图训练效率的启发；
  - **ReTAMamba / DynaMamba** 对 Mamba/SSM 长程记忆可能固化采样政策痕迹的警示；
  - **ReDiTT** 对异步事件未来分布多模态和条件生成的启发，但不使用 retrieval classifier；
  - **Delta-XAI** 对风险变化审计的启发，但不复用 SWING / delta circuit breaker。

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
9. trigger-phase hysteresis、control-barrier certificate、regret escrow、principal stratum compiler、Do-IV purifier。
10. IRT-DIF、RG fixed point、bitemporal curtain、clinical tomography、matched risk-set likelihood、policy privacy cloak。
11. synthetic falsification forge、dialectical JEPA referee、PID semantic prism、Noether semantic action、meta workflow immunization。
12. collider seal、source-atlas antipodal retrieval、Schwarzian warp canonicalizer、renewal exposure meter、palimpsest memory。
13. typed SSA observation program、Robin boundary field solver、Blackwell experiment order、elicitable functional compass。
14. Lyapunov interval observer、BBP / Marchenko-Pastur random-matrix spike sieve、RIP sparse pathology recovery。
15. factorial cumulant thinning algebra、repeat/panel cumulant subtraction、Poisson clutter null。
16. Delta-XAI / SWING 风险跃迁熔断、prediction-change attribution consistency、policy interrupt buffer。
17. B-spline knot-vector framing、Boehm knot insertion/removal、canonical disease control polygon、knot residual buffer。
18. ReTAMamba reliability tensor 原子解混、nonnegative anchor-cone freshness demixer、state/policy freshness atoms、policy cone absorption。
19. 单纯 state-policy 双分支、对抗环境分类器、跨视图 logits/representation 一致性、频域掩码对比学习、missingness pattern 直接分类、普通 retrieval-augmented classifier、普通 dual-SSM classifier。

本提案选择新的正交切入点：**不把采样政策建成概率、图、谱、样条 knot、可靠性 cone、程序、队列、证据市场、稀疏测量矩阵或在线风险跃迁；而是把一条非规则观测事件流嵌入复平面中的观测轮廓。采样政策偏移主要改变轮廓如何绕行、拉伸、折返或形成小回路；真正的病程突变对应由 value 支持的 meromorphic disease pole。分类器只读取这些病程奇点的留数，policy-only contour deformation 的积分必须在 Cauchy 意义下闭合为零。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily` 中的几条机制共同暴露了一个尚未被历史方案占据的核心矛盾：

- **TD-HNODE** 说明首次记录时间并不等同于真实病程发生时间；筛查、随访、编码与就诊制度会改变 observation clock。
- **SLiCE** 让多策略连续时间编码更可扩展，但如果 control path 本身由采样制度塑形，高效 CDE 可能更快吸收 policy shortcut。
- **DynaMamba / ReTAMamba** 提醒我们，长程 memory、reliability gate 与 multi-scale routing 会把“何时测、测多久、测哪些变量”长期保存在 hidden state 中。
- **ReDiTT** 说明异步事件流有多模态未来，但普通 retrieval memory 很容易按 policy fingerprint 而不是病程状态找邻居。

这些工作仍默认分类器直接消费时间路径、状态记忆、可靠性权重或检索参考。本提案换一个数学对象：

> 同一条潜在病程在不同医院或设备下被不同方式采样，本质上像同一个复平面 meromorphic field 被不同观测轮廓积分。只要轮廓没有跨过新的 disease pole，Cauchy 积分与留数不应改变；如果 value 真的出现阈值跨越、趋势反转或跨变量生理共振，则相当于轮廓包围了新的病程奇点，其留数才允许改变分类证据。

因此，**Do-Cauchy Residue Firewall (DCRF)** 把 sampling-policy shift 从“时间戳分布偏移”改写成“观测轮廓变形”：

```text
event stream  -> complex observation contour Gamma
value stem    -> meromorphic evidence field f_theta(z)
classifier    -> residues of disease poles enclosed by Gamma
policy shift  -> contour deformation without disease-pole crossing
```

复分析中的核心事实是：

```text
Integral_Gamma f(z) dz = 2 pi i * sum residues of enclosed poles
```

如果一个反事实采样策略只是让观测更密、更稀、吸附到 routine grid、panel 同步返回、告警后重复测量，轮廓形状可以变化，但只要它没有包围新的病程奇点，留数和分类证据不应改变。相反，如果观测值真的引入新的生理语义，例如 lactate 突破危险阈值、SOFA 相关变量共同恶化、趋势从恢复转为恶化，则 meromorphic field 中应出现或强化 disease pole，留数应被释放给分类器。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 负责生成复平面中的 disease-pole 候选：值新息、趋势反转、跨变量支持和临床阈值跨越；
- sampling process 只生成 contour deformation：routine wiggle、panel loop、alarm loop、gap stretch、visit delay、device duty-cycle detour；
- counterfactual intervention 不再要求 logits 一致，也不做密度比、风险方差、可靠性解混或样条 knot 投影；它只验证 policy-only deformation 的 contour integral 是否无新增留数。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Contour Deformation Bank

新增 `CauchyContourCollator`。每个样本返回事实事件流，同时返回复平面轮廓参数和反事实轮廓变形：

```text
event_i = {
    value_i,
    time_i,
    variable_i,
    quality_i,
    mask_i,
}

contour_i = {
    z_i      = tau(time_i) + j * psi(value_i, variable_i),
    dz_i     = z_{i+1} - z_i,
    segment_mask_i,
}
```

反事实 `contour_deformation_bank` 包括：

1. `routine_wiggle`
   - 将时间点局部吸附到护理/查房节律，再在复平面实轴附近形成小幅摆动。
   - 不改变观测值，不应产生新 disease residue。

2. `panel_micro_loop`
   - 把同步 panel 展开成一个微小闭合回路，或把近邻异步变量收缩成 panel loop。
   - 如果只是联测制度改变，闭合小回路的积分应接近 0。

3. `alarm_detour`
   - 在告警窗口附近增加重复测量 detour，但不引入新的 value novelty。
   - 模拟告警后密集复测，不能靠重复绕行放大留数。

4. `gap_stretch`
   - 拉伸相邻事件间实轴距离，模拟低频 duty cycle 或随访延迟。
   - 只改变轮廓速度与长度，不应改变包围的 disease poles。

5. `record_delay_hook`
   - 将某些 lab-return 或 diagnosis-code 时间形成向后/向前钩状 detour。
   - 用于隔离 observation clock 与 disease clock，不复用 bitemporal curtain。

6. `semantic_pole_injection`
   - 保持采样轮廓基本不变，但注入 value-supported novelty：阈值跨越、趋势方向改变、跨变量生理一致恶化。
   - 应释放新的 disease residue。

这些 views 不是 contrastive positives，也不是 policy jury 或 risk-jump windows；它们只提供 Cauchy 意义下的“是否跨过新病程奇点”的训练契约。

### 2.2 改 Encoder：Event-to-Contour Meromorphic Encoder

#### A. Complex Event Stem

先把每个观测事件嵌入复平面：

```text
h_i = Stem(value_i, variable_i, quality_i)
z_i = tau(time_i) + j * psi(h_i)
```

- `tau(time)` 是单调时间坐标，可以吸收 SLiCE 式连续时间主干的高效时间编码；
- `psi(h)` 是 value-conditioned imaginary coordinate，表示当前事件离“病程语义轴”的偏移；
- 变量语义可以调节 `psi`，但不直接进入分类头。

#### B. Learnable Disease Poles

学习一组 class-shared / class-specific meromorphic poles：

```text
p_k = a_k + j b_k
rho_k = residue vector for pole k
```

每个 pole 表示一个可能的病程奇点，例如“持续低血压后乳酸上升”“呼吸频率与白细胞共同恶化”“肾功能指标跨阈值并伴随尿量下降”。pole 不是采样触发器；它必须被 value-supported contour 包围或接近时才释放 residue。

#### C. Soft Cauchy Residue Readout

离散事件流形成多段 contour。对每段计算 soft winding / residue weight：

```text
w_{i,k} = soft_winding(z_i, z_{i+1}, p_k)
residue_k = sum_i w_{i,k} * phi(h_i, h_{i+1})
```

其中 `soft_winding` 近似某条线段绕 pole 的角度变化：

```text
angle_delta = atan2(z_{i+1} - p_k) - atan2(z_i - p_k)
w_{i,k} = angle_delta / (2 pi)
```

`phi(h_i, h_{i+1})` 是值语义门控，防止仅靠轮廓几何绕 pole 产生证据。最终分类器只读取：

```text
safe_residue = ResidueNormalize(residue_1, ..., residue_K)
logits = Classifier(safe_residue)
```

分类头不接收 raw mask、delta-t histogram、event count、panel id、routine bucket、contour length、loop count、hospital id 或 policy deformation label。

#### D. Policy Contour Sink

policy-only deformation 的积分残差进入单独诊断头：

```text
policy_integral_sink = Integral(policy_loop, f_theta(z)) - 2 pi i * released_state_residue
```

它只输出：

- `contour_policy_load`
- `micro_loop_leakage`
- `alarm_detour_amplification`
- `record_delay_hook_score`
- `deployment_contour_shift_alarm`

这些诊断不进入分类器。

### 2.3 改 Loss：Cauchy Residue Discipline

总目标：

```text
L = L_cls
  + lambda_rec  * L_meromorphic_reconstruction
  + lambda_def  * L_cauchy_deformation_null
  + lambda_loop * L_policy_loop_zero
  + lambda_rel  * L_semantic_pole_release
  + lambda_sep  * L_pole_policy_separation
  + lambda_leak * L_contour_leakage_probe
```

#### A. Safe Classification `L_cls`

只用 disease residues 分类：

```text
L_cls = CE(Classifier(safe_residue_factual), y)
```

这一步不做 logits consistency，也不要求不同反事实视图的 hidden state 相同；分类器的输入对象从 event token / hidden memory 变成 meromorphic residue ledger。

#### B. Meromorphic Reconstruction `L_meromorphic_reconstruction`

为了避免 pole 变成任意 latent code，要求 meromorphic field 能重构局部 value increment：

```text
delta_hat_i = Re( sum_k rho_k / (z_mid_i - p_k + eps) ) * dz_i
L_rec = SmoothL1(delta_hat_i, value_{i+1} - value_i)
```

这保证 disease poles 解释的是连续事件之间的值变化，而不是纯采样几何。

#### C. Cauchy Deformation Null `L_cauchy_deformation_null`

对 `routine_wiggle`、`gap_stretch`、`record_delay_hook` 这类 policy-only deformation：

```text
L_def = || residue(policy_deformed_view) - residue(factual_view) ||_1
```

注意这里约束的是 **留数账本**，不是 logits 或普通 representation。若 policy-only deformation 只改变轮廓形状而没有跨过 disease pole，留数应保持 Cauchy 等价。

#### D. Policy Loop Zero `L_policy_loop_zero`

对 `panel_micro_loop` 与 `alarm_detour`：

```text
loop_integral = sum_{segments in loop} f_theta(z_i) * dz_i
L_loop = |loop_integral|_1
```

如果模型通过重复测量、panel 同步或告警复测小回路积累分类证据，`L_loop` 会直接惩罚这种捷径。

#### E. Semantic Pole Release `L_semantic_pole_release`

对 `semantic_pole_injection`：

```text
target_release_k = clinical_novelty_score_k
L_rel = BCE(sigmoid(residue_gain_k), target_release_k)
```

只有 value-supported novelty 才允许新增留数；这避免模型把所有 contour deformation 都压成不变，从而丢失真实病程突变。

#### F. Pole-Policy Separation `L_pole_policy_separation`

如果某个 pole 的 winding 主要由 policy-only views 触发，而不是由 value novelty 触发，则将其推入 policy sink：

```text
policy_trigger_rate_k = mean(|residue_k(policy_only) - residue_k(factual)|)
state_trigger_rate_k  = mean(|residue_k(semantic_injection) - residue_k(factual)|)
L_sep = relu(policy_trigger_rate_k - state_trigger_rate_k + margin)
```

#### G. Contour Leakage Probe `L_contour_leakage_probe`

训练一个 stop-gradient probe 从 `safe_residue` 预测 policy recipe / hospital / sampling density：

```text
L_probe = CE(Probe(safe_residue.detach()), policy_id)
L_leak  = - CE(Probe(safe_residue), uniform_policy)
```

这只是审计 residue 中是否仍残留采样政策，不是对抗环境分类器作为主机制；核心约束仍来自 Cauchy deformation 与 policy loop zero。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `ComplexEventStem`：输出 value-conditioned imaginary coordinate 与 local value increment。
- 现有 sampling branch 改为 `ContourDeformationGenerator`：只产生轮廓变形 recipes，不把 policy descriptor 输入分类头。
- 现有 counterfactual intervention 改为 `ContourDeformationBank`：生成 policy-only deformation 和 semantic-pole injection，用于训练留数是否释放。
- 分类主路径改为 `CauchyResidueClassifier`：只读取 disease residues。
- 推理阶段无需知道测试采样策略标签；模型从事实事件构建 contour，读取 disease pole residues，同时报告 contour policy sink 作为部署偏移告警。

---

## 3. Code Draft: PyTorch 核心模块草稿

```python
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CauchyBatch:
    values: torch.Tensor          # [B, T, V_in] observed value features
    times: torch.Tensor           # [B, T]
    variables: torch.Tensor       # [B, T] long
    quality: torch.Tensor         # [B, T, Q]
    mask: torch.Tensor            # [B, T]
    labels: torch.Tensor          # [B]
    policy_id: Optional[torch.Tensor] = None
    novelty_target: Optional[torch.Tensor] = None  # [B, K] semantic pole release target


class ComplexEventStem(nn.Module):
    """Map irregular events to complex contour coordinates and local value states."""

    def __init__(self, value_dim: int, quality_dim: int, num_vars: int, d_model: int):
        super().__init__()
        self.var_emb = nn.Embedding(num_vars, d_model)
        self.value_proj = nn.Linear(value_dim + quality_dim + d_model, d_model)
        self.imag_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.value_state = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, values, times, variables, quality):
        var_h = self.var_emb(variables)
        x = torch.cat([values, quality, var_h], dim=-1)
        h = self.value_state(self.value_proj(x))

        # Real coordinate is normalized observation time; imaginary coordinate is value semantics.
        tau = (times - times[:, :1]) / (times[:, -1:].clamp_min(1e-3) - times[:, :1] + 1e-3)
        psi = self.imag_head(h).squeeze(-1)
        z = torch.complex(tau, psi)
        return z, h


class CauchyResidueEncoder(nn.Module):
    """Learn meromorphic disease poles and read out soft residues from a contour."""

    def __init__(self, d_model: int, num_poles: int, num_classes: int):
        super().__init__()
        self.num_poles = num_poles
        self.pole_real = nn.Parameter(torch.linspace(0.05, 0.95, num_poles))
        self.pole_imag = nn.Parameter(torch.zeros(num_poles))
        self.residue_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_poles),
        )
        self.field_readout = nn.Linear(num_poles, d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_poles * d_model),
            nn.Linear(num_poles * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )
        self.policy_probe = nn.Sequential(
            nn.LayerNorm(num_poles * d_model),
            nn.Linear(num_poles * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 8),  # recipe / environment bins; resize in implementation
        )

    @staticmethod
    def _angle_delta(z0: torch.Tensor, z1: torch.Tensor, poles: torch.Tensor):
        # z0/z1: [B, T-1], poles: [K]
        a0 = torch.angle(z0.unsqueeze(-1) - poles)
        a1 = torch.angle(z1.unsqueeze(-1) - poles)
        delta = a1 - a0
        # unwrap to [-pi, pi] for stable soft winding.
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        return delta / (2 * math.pi)  # [B, T-1, K]

    def forward(self, z: torch.Tensor, h: torch.Tensor, mask: torch.Tensor):
        poles = torch.complex(self.pole_real, self.pole_imag)  # [K]
        z0, z1 = z[:, :-1], z[:, 1:]
        h0, h1 = h[:, :-1], h[:, 1:]
        seg_mask = (mask[:, :-1] * mask[:, 1:]).unsqueeze(-1)

        winding = self._angle_delta(z0, z1, poles) * seg_mask
        semantic_gate = torch.sigmoid(self.residue_gate(torch.cat([h0, h1], dim=-1)))
        weighted_winding = winding.unsqueeze(-1) * semantic_gate.unsqueeze(-1) * h1.unsqueeze(-2)
        residues = weighted_winding.sum(dim=1)  # [B, K, D]

        flat = residues.flatten(start_dim=1)
        logits = self.classifier(flat)
        return {
            "logits": logits,
            "residues": residues,
            "flat_residue": flat,
            "poles": poles,
            "winding": winding,
        }

    def loop_integral(self, z: torch.Tensor, h: torch.Tensor, mask: torch.Tensor):
        # A lightweight meromorphic field integral proxy for policy-only micro loops.
        poles = torch.complex(self.pole_real, self.pole_imag)
        z_mid = 0.5 * (z[:, 1:] + z[:, :-1])
        dz = z[:, 1:] - z[:, :-1]
        seg_mask = (mask[:, :-1] * mask[:, 1:]).unsqueeze(-1)

        inv_dist = 1.0 / (z_mid.unsqueeze(-1) - poles + (1e-4 + 0.0j))
        field_coeff = self.residue_gate(torch.cat([h[:, :-1], h[:, 1:]], dim=-1))
        field = (field_coeff.to(inv_dist.dtype) * inv_dist).sum(dim=-1)
        integral = (field * dz * seg_mask.squeeze(-1)).sum(dim=1)
        return torch.stack([integral.real, integral.imag], dim=-1)


class DoCauchyResidueFirewall(nn.Module):
    def __init__(self, value_dim, quality_dim, num_vars, d_model, num_poles, num_classes):
        super().__init__()
        self.stem = ComplexEventStem(value_dim, quality_dim, num_vars, d_model)
        self.encoder = CauchyResidueEncoder(d_model, num_poles, num_classes)

    def encode(self, batch: CauchyBatch):
        z, h = self.stem(batch.values, batch.times, batch.variables, batch.quality)
        return self.encoder(z, h, batch.mask), z, h

    def forward(self, factual: CauchyBatch, views: Optional[Dict[str, CauchyBatch]] = None):
        out, z, h = self.encode(factual)
        losses = {"cls": F.cross_entropy(out["logits"], factual.labels)}

        if views is not None:
            factual_res = out["residues"].detach()

            # Policy-only deformations should preserve the residue ledger.
            null_terms = []
            loop_terms = []
            for name in ["routine_wiggle", "gap_stretch", "record_delay_hook"]:
                if name in views:
                    v_out, _, _ = self.encode(views[name])
                    null_terms.append((v_out["residues"] - factual_res).abs().mean())
            for name in ["panel_micro_loop", "alarm_detour"]:
                if name in views:
                    z_v, h_v = self.stem(
                        views[name].values,
                        views[name].times,
                        views[name].variables,
                        views[name].quality,
                    )
                    loop_terms.append(self.encoder.loop_integral(z_v, h_v, views[name].mask).abs().mean())

            if null_terms:
                losses["cauchy_deformation_null"] = torch.stack(null_terms).mean()
            if loop_terms:
                losses["policy_loop_zero"] = torch.stack(loop_terms).mean()

            # Semantic novelty should release selected disease poles.
            if "semantic_pole_injection" in views and factual.novelty_target is not None:
                s_out, _, _ = self.encode(views["semantic_pole_injection"])
                gain = (s_out["residues"] - out["residues"]).norm(dim=-1)
                losses["semantic_pole_release"] = F.binary_cross_entropy_with_logits(
                    gain, factual.novelty_target.float()
                )

        if factual.policy_id is not None:
            probe_logits = self.encoder.policy_probe(out["flat_residue"].detach())
            losses["policy_probe_audit"] = F.cross_entropy(probe_logits, factual.policy_id)

            # Leakage penalty pushes the live residue away from policy-identifiable shortcuts.
            live_probe_logits = self.encoder.policy_probe(out["flat_residue"])
            uniform = torch.full_like(live_probe_logits, 1.0 / live_probe_logits.size(-1))
            losses["contour_leakage"] = F.kl_div(
                F.log_softmax(live_probe_logits, dim=-1),
                uniform,
                reduction="batchmean",
            )

        return out, losses


def total_loss(losses, weights):
    loss = losses["cls"]
    for key, value in losses.items():
        if key == "cls":
            continue
        loss = loss + weights.get(key, 1.0) * value
    return loss
```

---

## 4. 为什么它与历史方案显著正交

1. **不是 sampling probability / hazard / density ratio**：DCRF 不估计观测点过程，也不做 Radon-Nikodym 校正；采样策略只作为 contour deformation。
2. **不是拓扑胶囊或 persistent homology**：虽然提到轮廓绕行，但核心训练对象是 meromorphic residue 与 Cauchy integral，不计算 persistence diagram、Betti 数或拓扑寿命。
3. **不是样条 knot / Boehm algebra**：9 月 27 日方案把采样视为 knot vector surgery；本方案完全不构造 B-spline basis 或 control polygon，而是把事件流视为复平面 contour。
4. **不是可靠性锚锥解混**：9 月 28 日方案解混 reliability tensor；本方案不学习 freshness atoms，也不让 reliability gate 主导 token 权重。
5. **不是风险跃迁解释**：9 月 26 日方案围绕 prediction delta / SWING circuit breaker；本方案不解释在线 logits 变化，而在分类前阻断 policy-only contour loops 的留数释放。
6. **不是检索增强分类器**：ReDiTT 的启发仅用于认识异步事件多模态，不建立 neighbor memory，也不把 reference tokens 输入分类器。
7. **不是普通跨视图一致性**：DCRF 约束的是“policy-only deformation 不跨 disease pole 时留数不变”和“semantic novelty 应释放新留数”，不是粗暴要求 representation 或 logits 对齐。
8. **解释性直接对应部署风险**：如果某医院的 routine grid、panel 同步或告警复测导致 `policy_loop_zero` 与 `contour_policy_load` 升高，就能明确指出模型正在把观测轮廓的绕行当作病程奇点。

---

## 5. 实验设计建议

1. **数据集与偏移构造**
   - PhysioNet 2012 / 2019、MIMIC-III / MIMIC-IV、eICU。
   - 构造 cross-policy split：routine-heavy、alarm-dense、panel-synchronous、lab-delay、low-duty-cycle。
   - 额外生成 counterfactual contour deformation bank：只改采样轮廓、不改 value novelty。

2. **基线**
   - GRU-D、mTAN、Raindrop、QuITE、DBGL、SLAN、TimeCHEAT、ReTAMamba/DynaMamba-like SSM。
   - 现有“采样解耦/反事实干预”框架的普通双分支版本。
   - 不带 Cauchy losses 的 complex contour encoder。

3. **核心指标**
   - in-policy AUROC/AUPRC。
   - cross-policy AUROC/AUPRC drop。
   - policy-only contour residue gain。
   - micro-loop leakage。
   - semantic-pole release recall。
   - safe_residue 中 policy_id 可预测性。

4. **关键消融**
   - 去掉 `L_policy_loop_zero`，验证 panel / alarm repeat 是否放大风险。
   - 去掉 `L_semantic_pole_release`，验证模型是否过度不变而忽略真实病程突变。
   - 将 contour residue 替换为普通 Mamba hidden state，验证收益不是复数 stem 参数量带来的。
   - 将 semantic gate 去掉，只保留几何 winding，验证 value support 对防止 policy loop 伪留数的必要性。

5. **可视化**
   - 每个样本的复平面 observation contour。
   - disease poles 与 policy-only loops 的相对位置。
   - factual vs counterfactual deformation 的 residue ledger。
   - 部署环境中 contour_policy_load 的漂移趋势。

---

## 6. 一句话卖点

**Do-Cauchy Residue Firewall 把非规则采样分类从“沿观测时间记忆事件”改成“只读取病程奇点留数”：采样政策可以任意弯曲观测轮廓、制造小回路或改变采样速度，但只要没有真实 value-supported disease pole crossing，就不能向分类器释放新的风险证据。**
