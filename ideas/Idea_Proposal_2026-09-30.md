# Title: Do-Microlocal Semantic Wavefront Shield：面向采样策略偏移的语义波前防火墙

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化持久记忆 `MEMORIES.md`，并纳入其中记录但当前工作区未完全落盘的历史 proposal 摘要。
- 已完整读取当前仓库 `ideas/Idea_Proposal_*.md` 的全部历史提案文件，覆盖 2026-06-12 至 2026-09-29 的已落盘提案。
- 已读取近期 `paper_daily_2026-09-27.md`、`paper_daily_2026-09-26.md`、`paper_daily_2026-09-25.md`、`paper_daily_2026-09-24.md`、`paper_daily_2026-09-22.md`、`paper_daily_2026-09-21.md`，并分段读取兼容入口 `paper_daily.md` 的近期索引。

### 历史核心机制黑名单

本提案显式避开以下已出现主机制：

1. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
2. 生理流/采样算子交换子、state-policy graph 拆分、policy residual sink。
3. protocol tax、additive evidence market、token 边际审计、证据预算。
4. posterior quotient、Radon-Nikodym density ratio、doubly robust correction、RKHS cubature。
5. reconstruction error cartography、ANOVA projection、VQ clauses、HSIC redaction。
6. optional-stopping martingale、拓扑胶囊、policy gauge、syndrome code、knockoff calendar。
7. observability witness、evidential vacuity、policy lattice、solver trace front-door、conformal sleeve、IV/control-function。
8. Borda jury、Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock。
9. IRT-DIF、RG fixed point、bitemporal curtain、clinical tomography、matched risk-set likelihood、policy privacy cloak。
10. synthetic falsification forge、dialectical JEPA referee、PID semantic prism、Noether semantic action、meta workflow immunization。
11. collider seal、source-atlas antipodal retrieval、Schwarzian warp canonicalizer、renewal exposure meter、palimpsest memory。
12. typed SSA observation program、Robin boundary field solver、Blackwell experiment order、elicitable functional compass。
13. Lyapunov interval observer、BBP / Marchenko-Pastur random-matrix spike sieve、RIP sparse pathology recovery。
14. factorial cumulant thinning algebra、repeat/panel cumulant subtraction、Poisson clutter null。
15. Delta-XAI / SWING 风险跃迁熔断、prediction-change attribution consistency、policy interrupt buffer。
16. B-spline knot-vector framing、Boehm knot insertion/removal、canonical disease control polygon、knot residual buffer。
17. ReTAMamba reliability tensor 原子解混、nonnegative anchor-cone freshness demixer、state/policy freshness atoms。
18. Cauchy contour / meromorphic disease pole / residue firewall。
19. 单纯 state-policy 双分支、对抗环境分类器、跨视图 logits/representation 一致性、频域掩码对比学习、missingness pattern 直接分类、普通 retrieval-augmented classifier、普通 dual-SSM classifier。

本轮选择新的正交切入点：**不把采样政策建成概率、图、谱、样条、可靠性锥、复轮廓、程序、边界、记忆、实验序或高阶计数；而是把不规则事件流看成时间-通道-数值语义空间中的离散分布。采样政策制造的是沿 routine grid、panel 同步、重复复测、pending 返回等方向的 policy conormal singularities；真实病理突变制造的是由 value novelty 支持的 semantic wavefront。分类器只读取避开 policy conormal cone 的病理波前能量。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中有几条机制给出新的组合机会：

- **ReTAMamba** 强调变量特异 freshness、multi-scale aggregation 与 token routing，但 reliability 权重可能混入医院复测制度和 router budget 偏好。
- **TD-HNODE** 提醒我们首次记录时间与真实 disease progression time 混叠，观测时刻可能来自筛查/随访政策。
- **Delta-XAI** 说明在线风险变化常被 forward-fill、窗口滑动和 irregular sampling artifact 误归因。
- **ReDiTT / DynaMamba** 显示异步事件流与长程选择性记忆很强，但也可能长期保存 policy fingerprint。

这些工作仍默认模型在某种 token、路径、可靠性或记忆空间中累计证据。本提案换一个更底层的微局部视角：

> 采样政策不是一种普通特征，而是在观测测度中制造特定方向的奇异性。routine round 会沿固定日历方向产生格点边；panel 同步会在通道方向产生瞬时共现边；repeat echo 会在短时同变量方向产生高密度边；pending / value-return 会在“记录已出现但值语义未变”方向产生伪突变。真正的病理证据，则应表现为 value-supported semantic wavefront：阈值跨越、趋势反转、跨变量生理响应或疾病阶段跃迁。

因此，**Do-Microlocal Semantic Wavefront Shield (DMSWS)** 不再问“哪些 token 可靠”，也不问“不同采样视图 logits 是否一致”，而是问：

```text
当前分类 margin 来自哪一类局部方向奇异性？
它是否落在采样政策的 conormal cone 内？
它是否有真实 value novelty / pathology-bin transition 支持？
```

如果某个方向只在 routine / panel / repeat / pending 的 policy cone 内高能，它可以被报告为采样伪影，但不能进入分类头。如果某个方向同时有 value novelty、趋势反转或跨变量语义支撑，则作为病理 wavefront 被放行。

这与当前“采样解耦/反事实干预”框架自然兼容：

- value process 负责产生 semantic event field 与 value-supported jump scores；
- sampling process 负责生成 policy conormal directions 与反事实 policy cone recipes；
- counterfactual intervention 不生成一致性正样本，而是生成 policy-only conormal edits 与 semantic-wavefront edits；
- classifier 只读取 state wavefront energy，不读取 raw mask、delta-t、policy cone energy、event count 或 router budget。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Wavefront Cone Audit Bank

新增 `WavefrontConeCollator`。每个样本返回事实事件流与一组局部方向审计视图：

```text
event_i = {
    value_i,
    time_i,
    variable_i,
    semantic_channel_i,
    quality_i,
    mask_i,
}
```

反事实视图包括：

1. `routine_conormal`
   - 将事件吸附到固定查房 / 护理 round。
   - 只改变日历方向，不改变 value。
   - 应激活 policy cone，不应激活 state wavefront。

2. `panel_conormal`
   - 把一组变量压到同一时间，或将同步 panel 拆成异步返回。
   - 只改变通道-时间共现方向。

3. `repeat_echo_conormal`
   - 在同变量短间隔复制相似值。
   - 形成短时同变量 conormal ridge，但不应产生病理波前。

4. `pending_return_conormal`
   - pending flag 消失或值返回，但值与最近有效值一致。
   - 应被识别为记录流程突变。

5. `semantic_wavefront_insert`
   - 保持采样几何基本不变，插入真实 value novelty、阈值跨越、趋势反转或跨变量生理响应。
   - 应释放 state wavefront。

6. `mixed_probe`
   - 同时发生 policy conormal 改动与 semantic wavefront。
   - 用于检验模型是否能把 policy conormal energy 与 state wavefront energy 分账。

这些视图不是 contrastive positives，不做 logits consistency、不做 conformal / privacy / Blackwell / Cauchy / spline / NMF；它们只监督局部方向能量归因。

### 2.2 改 Encoder：Semantic Event Distribution -> Directional Wavefront Probes

#### A. 语义-时间坐标

每个事件被映射到连续坐标：

```text
xi_i = [tau(time_i), channel_semantic_i, value_phase_i]
h_i  = EventStem(value_i, variable_i, quality_i)
```

- `tau(time)` 是连续时间坐标；
- `channel_semantic` 来自变量语义、单位和设备描述，但不直接进入分类头；
- `value_phase` 是由 value bin / surprise 形成的语义坐标；
- `h_i` 是局部值语义状态。

#### B. Policy Conormal Dictionary

sampling branch 只输出一组 policy conormal directions：

```text
D_policy = {
    d_routine,       # 日历 round / cadence 方向
    d_panel,         # 同步 panel / cross-channel co-measurement 方向
    d_repeat,        # 同变量短延迟复测方向
    d_pending,       # record/value-return 方向
    d_budget         # variable budget / low-duty-cycle 方向
}
```

这些方向不进入分类器；它们只定义“哪些局部奇异性应被视为采样政策造成”。

#### C. State Wavefront Directions

value branch 学习少量 state wavefront directions：

```text
D_state = {
    threshold_cross,
    trend_flip,
    cross_variable_response,
    persistent_abnormality,
    disease_stage_jump
}
```

与 policy directions 不同，state directions 必须由 value novelty 和 pathology-bin transition 支持。

#### D. Directional Wavefront Energy

对事件对 `(i, j)`，计算局部坐标差和语义差：

```text
delta_xi_ij = xi_j - xi_i
delta_h_ij  = h_j - h_i
```

若 `delta_xi_ij` 与某个方向 `d` 对齐，且距离足够局部，则累积定向波前能量：

```text
E_d = sum_{i,j} K_local(xi_i, xi_j)
              * cone_align(delta_xi_ij, d)
              * || Project_d(delta_h_ij) ||^2
```

最终将能量分解为：

```text
E_state  = wavefront energy outside policy conormal cone and supported by value novelty
E_policy = wavefront energy inside policy conormal cone
```

分类器只读取 `E_state` 与其语义摘要：

```text
logits = Classifier(E_state, state_direction_summary)
```

### 2.3 改 Loss：从视图一致转向 Conormal Shield Discipline

总目标：

```text
L = L_cls
  + lambda_rec  * L_directional_reconstruction
  + lambda_pol  * L_policy_conormal_silence
  + lambda_sem  * L_semantic_wavefront_release
  + lambda_mix  * L_mixed_cone_additivity
  + lambda_sep  * L_cone_margin_separation
  + lambda_leak * L_policy_energy_leakage
```

#### A. Safe Wavefront Classification `L_cls`

只用 state wavefront energy 分类：

```text
L_cls = CE(Classifier(E_state), y)
```

#### B. Directional Reconstruction `L_directional_reconstruction`

为了避免方向能量成为任意 latent code，要求定向能量能解释局部 value jump：

```text
value_jump_hat_ij = Decoder(direction_energy_ij, semantic_direction_ij)
L_rec = SmoothL1(value_jump_hat_ij, value_j - value_i)
```

这不是 raw reconstruction，也不是 iTimER reconstruction-error pseudo observation；它只约束 wavefront energy 对真实局部值突变有语义解释。

#### C. Policy Conormal Silence `L_policy_conormal_silence`

对 policy-only 视图：

```text
L_pol =
  relu(E_state(policy_view) - E_state(factual) - semantic_gain - eps)^2
  + relu(policy_energy_min - E_policy(policy_view))^2
```

policy-only 改动应该被 policy cone 接住，而不应该增加 state wavefront。

#### D. Semantic Wavefront Release `L_semantic_wavefront_release`

对 `semantic_wavefront_insert`：

```text
semantic_target = value_surprise + threshold_cross + trend_flip
L_sem = BCE(sigmoid(Delta E_state), semantic_target)
```

这避免模型过度抑制所有方向突变；真实病理突变必须被 state wavefront 放行。

#### E. Mixed Cone Additivity `L_mixed_cone_additivity`

对同时包含 policy conormal 与 semantic wavefront 的 view：

```text
Delta_state_mixed  ~= Delta_state_semantic
Delta_policy_mixed ~= Delta_policy_policy
```

这让模型面对“告警后复测且数值确实恶化”时，能把复测几何和真实恶化拆开。

#### F. Cone Margin Separation `L_cone_margin_separation`

要求 policy conormal directions 与 state wavefront directions 在 learnable metric 下有角间隔：

```text
L_sep = mean relu(cos_M(d_state, d_policy) - cos_max)^2
```

它不是 latent gauge projection，也不是零空间 score surgery；分离发生在观测测度的局部方向锥，而不是隐藏表示维度。

#### G. Policy Energy Leakage `L_policy_energy_leakage`

训练一个只读 `E_policy` 的审计 probe：

```text
policy_probe_logits = Probe(stopgrad(E_policy))
L_lkg = CE(policy_probe_logits, y)
```

若 `E_policy` 单独能强预测标签，说明数据中存在严重 policy-label coupling。分类器本身不读取 `E_policy`，该损失主要用于报告与轻量约束。

---

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


class SemanticCoordinateStem(nn.Module):
    """Build semantic-time-value coordinates and local event states."""

    def __init__(self, num_vars: int, coord_dim: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, coord_dim),
            nn.SiLU(),
            nn.Linear(coord_dim, coord_dim),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp_min(0)
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_h = self.var_embed(var_id)
        value_h = self.value_proj(torch.stack([value, torch.tanh(value), quality], dim=-1))
        state_h = self.state_proj(var_h + value_h) * mask.unsqueeze(-1)

        coord_input = torch.cat(
            [
                var_h,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                value_h.norm(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        coord = self.coord_proj(coord_input)
        coord = F.normalize(coord, dim=-1) * mask.unsqueeze(-1)
        return {"coord": coord, "state": state_h, "mask": mask}


class DirectionConeBank(nn.Module):
    """Learn state wavefront directions and derive policy conormal directions."""

    def __init__(self, coord_dim: int, num_state_dirs: int = 8, num_policy_dirs: int = 6):
        super().__init__()
        self.state_dirs = nn.Parameter(torch.randn(num_state_dirs, coord_dim) * 0.02)
        self.policy_dirs = nn.Parameter(torch.randn(num_policy_dirs, coord_dim) * 0.02)
        self.metric_raw = nn.Parameter(torch.eye(coord_dim))

    def metric(self) -> torch.Tensor:
        return self.metric_raw.T @ self.metric_raw + 1e-3 * torch.eye(
            self.metric_raw.size(0),
            device=self.metric_raw.device,
            dtype=self.metric_raw.dtype,
        )

    def normalized_dirs(self) -> tuple[torch.Tensor, torch.Tensor]:
        metric = self.metric()

        def norm_dir(d: torch.Tensor) -> torch.Tensor:
            md = d @ metric
            norm = (md * d).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            return d / norm

        return norm_dir(self.state_dirs), norm_dir(self.policy_dirs)


class LocalWavefrontProbe(nn.Module):
    """Estimate directional singular energy over event-pair local cones."""

    def __init__(self, hidden_dim: int, coord_dim: int, num_state_dirs: int, num_policy_dirs: int):
        super().__init__()
        self.cones = DirectionConeBank(coord_dim, num_state_dirs, num_policy_dirs)
        self.state_proj = nn.Linear(hidden_dim, num_state_dirs)
        self.policy_proj = nn.Linear(hidden_dim, num_policy_dirs)
        self.decoder = nn.Sequential(
            nn.Linear(num_state_dirs + num_policy_dirs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def directional_energy(self, coord: torch.Tensor, state: torch.Tensor, mask: torch.Tensor) -> dict:
        bsz, steps, coord_dim = coord.shape
        state_dirs, policy_dirs = self.cones.normalized_dirs()
        metric = self.cones.metric()

        dx = coord[:, :, None, :] - coord[:, None, :, :]  # [B, T, T, C]
        dh = state[:, :, None, :] - state[:, None, :, :]  # [B, T, T, H]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        eye = torch.eye(steps, device=mask.device, dtype=torch.bool)[None]
        pair_mask = pair_mask.masked_fill(eye, 0.0)

        # Local kernel: only nearby semantic-time coordinates contribute.
        mdx = torch.einsum("btic,cd->btid", dx, metric)
        dist2 = (mdx * dx).sum(dim=-1)
        local = torch.exp(-4.0 * dist2) * pair_mask

        def cone_energy(dirs: torch.Tensor, proj_layer: nn.Linear) -> torch.Tensor:
            align = torch.einsum("btic,kc->btik", dx, dirs)
            align = align.pow(2) / dist2.unsqueeze(-1).clamp_min(1e-6)
            semantic_jump = proj_layer(dh).pow(2)
            energy = (local.unsqueeze(-1) * align * semantic_jump).sum(dim=(1, 2))
            denom = local.sum(dim=(1, 2), keepdim=False).clamp_min(1.0).unsqueeze(-1)
            return energy / denom

        state_energy = cone_energy(state_dirs, self.state_proj)
        policy_energy = cone_energy(policy_dirs, self.policy_proj)
        all_energy = torch.cat([state_energy, policy_energy], dim=-1)
        jump_recon = self.decoder(all_energy).squeeze(-1)
        return {
            "state_energy": state_energy,
            "policy_energy": policy_energy,
            "jump_recon": jump_recon,
            "state_dirs": state_dirs,
            "policy_dirs": policy_dirs,
        }


class DoMicrolocalSemanticWavefrontShield(nn.Module):
    """Classify only from value-supported wavefront energy outside policy conormal cones."""

    def __init__(
        self,
        num_vars: int,
        coord_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_state_dirs: int = 8,
        num_policy_dirs: int = 6,
    ):
        super().__init__()
        self.stem = SemanticCoordinateStem(num_vars, coord_dim, hidden_dim)
        self.wavefront = LocalWavefrontProbe(hidden_dim, coord_dim, num_state_dirs, num_policy_dirs)
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_state_dirs),
            nn.Linear(num_state_dirs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.policy_probe = nn.Sequential(
            nn.LayerNorm(num_policy_dirs),
            nn.Linear(num_policy_dirs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, batch: dict) -> dict:
        stem = self.stem(batch)
        wf = self.wavefront.directional_energy(stem["coord"], stem["state"], stem["mask"])
        logits = self.classifier(wf["state_energy"])
        policy_logits = self.policy_probe(wf["policy_energy"].detach())
        return {**stem, **wf, "logits": logits, "policy_logits": policy_logits}

    def cone_separation_loss(self, out: dict, cos_max: float = 0.20) -> torch.Tensor:
        state_dirs = out["state_dirs"]
        policy_dirs = out["policy_dirs"]
        metric = self.wavefront.cones.metric()
        cos = torch.einsum("kc,cd,ld->kl", state_dirs, metric, policy_dirs)
        return F.relu(cos.abs() - cos_max).pow(2).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_rec: float = 0.10,
        lambda_pol: float = 0.30,
        lambda_sem: float = 0.25,
        lambda_mix: float = 0.15,
        lambda_sep: float = 0.05,
        lambda_leak: float = 0.03,
    ) -> dict:
        labels = batch["labels"]
        factual = self.encode(batch["factual_view"])
        cls_loss = F.cross_entropy(factual["logits"], labels)

        # Directional reconstruction target: local value variation magnitude.
        rec_target = batch.get(
            "value_jump_summary",
            batch["factual_view"]["event_value"].std(dim=1),
        )
        rec_loss = F.smooth_l1_loss(factual["jump_recon"], rec_target.detach())

        policy_losses = []
        for view in batch.get("policy_conormal_views", []):
            out = self.encode(view)
            semantic_gain = view.get("semantic_gain", torch.zeros_like(factual["state_energy"].sum(dim=-1)))
            excess_state = out["state_energy"].sum(dim=-1) - factual["state_energy"].sum(dim=-1).detach()
            policy_load = out["policy_energy"].sum(dim=-1)
            policy_losses.append(
                F.relu(excess_state - semantic_gain - 0.05).pow(2).mean()
                + F.relu(0.05 - policy_load).pow(2).mean()
            )
        policy_loss = torch.stack(policy_losses).mean() if policy_losses else cls_loss.new_zeros(())

        semantic_view = batch.get("semantic_wavefront_view")
        if semantic_view is not None:
            sem = self.encode(semantic_view)
            target = semantic_view.get("semantic_target", torch.ones_like(sem["state_energy"].sum(dim=-1)))
            gain = sem["state_energy"].sum(dim=-1) - factual["state_energy"].sum(dim=-1).detach()
            sem_loss = F.binary_cross_entropy_with_logits(gain, target.float())
        else:
            sem_loss = cls_loss.new_zeros(())

        mixed_view = batch.get("mixed_probe_view")
        if mixed_view is not None and semantic_view is not None and policy_losses:
            mixed = self.encode(mixed_view)
            delta_state_expected = sem["state_energy"] - factual["state_energy"]
            # Use first policy view as a practical anchor in this sketch.
            pol_anchor = self.encode(batch["policy_conormal_views"][0])
            delta_policy_expected = pol_anchor["policy_energy"] - factual["policy_energy"]
            mix_loss = F.smooth_l1_loss(
                mixed["state_energy"] - factual["state_energy"],
                delta_state_expected.detach(),
            ) + F.smooth_l1_loss(
                mixed["policy_energy"] - factual["policy_energy"],
                delta_policy_expected.detach(),
            )
        else:
            mix_loss = cls_loss.new_zeros(())

        sep_loss = self.cone_separation_loss(factual)
        leak_loss = F.cross_entropy(factual["policy_logits"], labels)

        total = (
            cls_loss
            + lambda_rec * rec_loss
            + lambda_pol * policy_loss
            + lambda_sem * sem_loss
            + lambda_mix * mix_loss
            + lambda_sep * sep_loss
            + lambda_leak * leak_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "directional_reconstruction_loss": rec_loss.detach(),
            "policy_conormal_silence_loss": policy_loss.detach(),
            "semantic_wavefront_release_loss": sem_loss.detach(),
            "mixed_cone_additivity_loss": mix_loss.detach(),
            "cone_margin_separation_loss": sep_loss.detach(),
            "policy_energy_leakage_loss": leak_loss.detach(),
            "mean_state_wavefront": factual["state_energy"].sum(dim=-1).mean().detach(),
            "mean_policy_wavefront": factual["policy_energy"].sum(dim=-1).mean().detach(),
        }
```

### Collator 草稿

```python
import torch


@torch.no_grad()
def build_wavefront_cone_audit_batch(batch: dict) -> dict:
    """Create microlocal policy-conormal and semantic-wavefront audit views."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    quality = batch.get("measurement_quality", torch.ones_like(value))
    device = value.device
    steps = value.size(1)
    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, new_quality):
        view = dict(batch)
        view["event_value"] = new_value
        view["event_time"] = new_time
        view["event_var_id"] = new_var
        view["event_mask"] = new_mask
        view["measurement_quality"] = new_quality
        return view

    policy_views = []

    # 1. Routine conormal: snap to clinical rounds.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    v = clone_with(value * mask, rounded_time, var_id, mask, quality)
    v["semantic_gain"] = torch.zeros(value.size(0), device=device)
    policy_views.append(v)

    # 2. Panel conormal: synchronize cross-variable near-neighbors.
    packed_time = time.clone()
    packed_time[:, 1:] = torch.where(mask[:, 1:] > 0, time[:, :-1], time[:, 1:])
    v = clone_with(value * mask, packed_time, var_id, mask, quality)
    v["semantic_gain"] = torch.zeros(value.size(0), device=device)
    policy_views.append(v)

    # 3. Repeat echo conormal: duplicate similar values via short lag.
    repeated = torch.where((time_norm > 0.66), value.roll(shifts=1, dims=1), value)
    v = clone_with(repeated * mask, time, var_id, mask, quality)
    v["semantic_gain"] = torch.zeros(value.size(0), device=device)
    policy_views.append(v)

    # 4. Pending-return conormal: record change without value novelty.
    pending_quality = quality * 0.25
    v = clone_with(value * mask, time, var_id, mask, pending_quality)
    v["semantic_gain"] = torch.zeros(value.size(0), device=device)
    policy_views.append(v)

    # Semantic wavefront: inject value-supported novelty in high-surprise region.
    surprise = batch.get("value_surprise", value.abs())
    threshold = surprise.quantile(0.75, dim=1, keepdim=True)
    inject = (surprise >= threshold).to(value.dtype) * mask
    semantic_value = value + 0.5 * inject * torch.sign(value + 1e-3)
    semantic_view = clone_with(semantic_value, time, var_id, mask, quality)
    semantic_view["semantic_target"] = (inject.sum(dim=1) > 0).to(value.dtype)

    # Mixed probe: routine snapping plus semantic novelty.
    mixed_view = clone_with(semantic_value, rounded_time, var_id, mask, quality)
    mixed_view["semantic_target"] = semantic_view["semantic_target"]

    out = {
        "factual_view": batch,
        "policy_conormal_views": policy_views,
        "semantic_wavefront_view": semantic_view,
        "mixed_probe_view": mixed_view,
        "labels": batch["labels"],
        "value_jump_summary": value.diff(dim=1, prepend=value[:, :1]).abs().mean(dim=1),
    }
    return out
```

---

## 4. 实验切入点

1. **跨采样政策 split**
   - routine-heavy -> alarm-dense；
   - panel-pack -> panel-split；
   - pending-heavy -> value-return-clean；
   - high-duty-cycle -> low-duty-cycle；
   - hospital schema / channel semantic shift。

2. **对比方法**
   - ReTAMamba / DynaMamba-style reliability-Mamba backbone；
   - TD-HNODE-style hypergraph ODE；
   - Delta-XAI/SWING post-hoc audit；
   - ReDiTT-style latent event model without retrieval for classification；
   - 历史方案：DAFD、DCRF、DBKF、DSDCB、DBSES、DEFC、DSSS、DRBI、DLIO、DBBP-SS、DRIP-SPL、Do-FaCuT 等。

3. **核心指标**
   - in-policy AUROC / AUPRC；
   - worst-policy AUROC / AUPRC；
   - policy-conormal energy leakage：`E_policy` 对标签的 probe AUC；
   - semantic-wavefront release recall：真实阈值跨越 / 趋势反转是否释放 state energy；
   - routine / panel / repeat / pending conormal silence rate；
   - mixed additivity residual：同时存在 workflow 与真实恶化时能否分账。

4. **消融实验**
   - 去掉 policy conormal silence，让分类器直接读取全部 directional energy；
   - 去掉 semantic release，观察是否过度鲁棒、漏掉真实急性恶化；
   - 将 conormal directions 替换为随机方向，验证收益来自结构化采样方向；
   - 让 `E_policy` 进入分类头作为反例，验证院内性能可能升高但跨政策退化；
   - 用普通 time/channel attention 替代 directional pair probe，验证不是参数量带来的收益。

---

## 5. 预期创新性

1. **从采样特征去偏转向微局部奇异性归因**：DMSWS 不估计采样概率、不做图/谱/边界/样条/复轮廓，而是识别观测测度中不同方向的局部奇异能量。
2. **从可靠性标量转向 conormal shielding**：ReTAMamba 的 reliability 问“这个 token 还可靠吗”；DMSWS 问“这个 token 引起的局部突变沿哪个方向发生，是否落入采样 conormal cone”。
3. **从风险变化解释转向分类前的方向防火墙**：Delta-XAI 解释预测为什么变；DMSWS 在分类前阻止 policy conormal energy 进入 state wavefront。
4. **从病程/观测时钟混叠转向 cotangent-level 分离**：TD-HNODE 暴露首次记录时间污染 disease clock；DMSWS 在时间-通道-数值语义空间的方向层面分离记录流程与真实病理突变。
5. **不强迫过度一致**：低频、pending 或 panel shift 可以改变 policy energy 和诊断告警；只有 value-supported semantic wavefront 才能改变分类证据。

## 6. 一句话投稿卖点

**Do-Microlocal Semantic Wavefront Shield 首次把非规则采样策略偏移表述为“采样流程在观测测度中制造 policy conormal singularities”的问题，通过语义-时间坐标、局部方向波前探针、policy conormal silence 与 semantic wavefront release，让分类器只读取由真实病理值新息支撑的 state wavefront，而不是读取 routine cadence、panel 同步、repeat echo、pending 返回或 router 预算造成的采样方向伪突变。**
