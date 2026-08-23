# Title: Do-CauKer Orthogonal Falsification Forge：面向采样策略偏移的正交合成反例锻炉

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代总结文件。
- 已读取 `paper_daily.md`，重点纳入最新记录中的 **CauKer**，以及近期记录中反复出现的 **PULSE / TCF**。
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
  - `ideas/Idea_Proposal_2026-07-30.md`
  - `ideas/Idea_Proposal_2026-08-05.md`
  - `ideas/Idea_Proposal_2026-08-06.md`
  - `ideas/Idea_Proposal_2026-08-08.md`
  - `ideas/Idea_Proposal_2026-08-09.md`
  - `ideas/Idea_Proposal_2026-08-22.md`
- 已读取自动化记忆 `MEMORIES.md` 与其中全部额外历史 proposal 摘要：
  - `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`
  - `idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`
  - `idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-05.md`、`idea_2026-08-06.md`
  - `idea_2026-08-07.md`、`idea_2026-08-08.md`、`idea_2026-08-09.md`
  - `idea_2026-08-10.md`、`idea_2026-08-11.md`、`idea_2026-08-21.md`

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮永久避开以下核心机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、采样 score 零空间、hazard-driven resampling、do-risk variance。
8. 生理流算子与采样算子交换子、value/policy graph commutator、policy residual sink。
9. additive evidence market、protocol tax、token-level evidence budget、边际证据审计。
10. 模型空间 posterior quotient、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、ANOVA projection、VQ semantic / acquisition clauses、HSIC redaction。
12. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler。
13. 采样测度 density ratio、doubly robust target-measure correction、influence-bound。
14. optional-stopping martingale queries、standardized innovation、stopping recipe moment control。
15. soft excursion topology、censored persistence interval、censor envelope、fragmentation sobriety。
16. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
17. policy-only negative film、shadow eraser / stencil、shadow nullification。
18. latent packet codeword、parity-check、syndrome locator、packet repair decoder。
19. conditional knockoff calendar、soft knockoff-FDR、swap symmetry。
20. observability witness、counterfactual observability probe、observability-routed classification。
21. subjective-logic / Dirichlet evidential shield、policy-induced vacuity、class-wise evidence discount。
22. observation-set policy lattice、meet/join masks、monotone/submodular margin、shortcut curvature。
23. solver-trace front-door、NFE/roughness trace mediator、reference trace bank。
24. measurement-action bisimulation、policy-word signature renormalization、thermodynamic free-energy、Sklar copula stripping、triage queue debt、Sinkhorn detail canonicalization、MDL episode transducer、causal sheaf gluing、trigger hysteresis、control barrier certificates、regret escrow、principal-stratum status compiler。
25. counterfactual conformal risk sleeves、counterfactual sampling instruments、Borda / majority rank tribunal、Krylov policy subspace、determinantal / Nystrom volume basis、tropical support routes。
26. fixed clinical viva question bank、pathology sequent proof bank、disease-progress poset clock、pathology feasible hull、IRT latent trait / DIF firewall。
27. observation-resolution RG beta flow、event-time vs record-time causal curtain、clinical tomography ray design、matched policy risk sets、Gaussian privacy cloak。

本提案选择新的正交切入点：**不在真实数据上事后删除、平滑、校准、投影或证明采样信息，而是在预训练数据层构造一个“采样政策 shortcut 必然被反例推翻”的合成宇宙。借鉴 CauKer 的 GP kernel composition + SCM 合成分类世界，但额外加入可控 observation-policy SCM，并用正交数组设计让 label、状态机制、采样政策、噪声质量和中心流程在合成 batch 中被系统性交叉。模型通过 policy-cell worst-case risk 与机制卡监督学习状态因果因素；任何只依赖采样政策的规则都会在同一锻炉中遇到反标签、同标签跨政策、同政策跨状态的最小反例。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 最新记录中的 **CauKer** 给了一个非常重要的方向：time series classification foundation model 不一定只能靠更多真实数据预训练，结构化合成数据也可以通过 GP kernel composition 与 Structural Causal Models 产生可泛化的分类机制。它的缺口是，原始 CauKer 主要覆盖状态轨迹、变量依赖、趋势、周期、非平稳等 **value-generating mechanisms**，但还没有把 ICU/EHR/科学观测中的 **observation policy** 作为独立可控生成因子。

而当前“非规则采样时间序列的采样策略偏移鲁棒分类”最难的地方恰恰是：真实数据里的 label 与 sampling policy 往往天然缠在一起。PULSE 告诉我们跨 HiRID / MIMIC-IV / eICU 的中心差异会同时改变病人分布、变量 schema、采样频率、流程和标签基率；TCF 告诉我们 EHR 事件的时间与病理分箱有强语义，但未来事件出现本身又混合 patient state 与 care process。只在真实数据上训练时，模型很容易学到一个在训练医院有效、在测试医院失效的采样捷径，而且事后再做对抗、校准或投影往往只能修补已经形成的偏差。

**Do-CauKer Orthogonal Falsification Forge (DCOFF)** 的核心直觉：

> 在模型学习真实医院数据之前，先把它放进一个可控合成锻炉：同一种采样政策会配到不同标签，同一个标签会配到所有采样政策，同一个潜在病程会被多种政策观察，同一种政策也会观察到不同病程。采样捷径不是被惩罚掉，而是被合成世界中的最小反例系统性 falsify。

这与历史方案的关键区别是：

- 不是跨视图一致性：不要求同一潜在轨迹在不同采样政策下 logits 或 representation 相同。
- 不是 policy adversarial：不训练分类表示去欺骗环境判别器。
- 不是后验商、密度比、危险率、保形、knockoff、IRT、RG 或 DP privacy。
- 主创新发生在 **Dataloader / pretraining curriculum**：通过正交合成因子设计打断 label-policy 偶然相关，并用 worst-cell risk 逼迫模型在每个政策单元格都正确分类。

如果一个模型试图使用“报警后密集复测 = 死亡风险高”这类规则，DCOFF 会在合成锻炉里立即提供反例：报警后密集复测也会出现在低风险病程；高风险病程也会以 routine-round、panel-split、value-pending 或低覆盖策略出现。模型只有学习由 GP+SCM 产生的状态机制和 TCF-style pathology bins，才能在所有正交 policy cell 上同时降低风险。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：从真实 batch 改为 Orthogonal Falsification Forge

新增 `OrthogonalFalsificationForge`，每个训练 epoch 同时产生真实 batch 与合成 forge batch。合成样本由五类因子组成：

1. **State mechanism factor `m_state`**
   - 来自 CauKer：组合 RBF、periodic、linear、Matérn-like、change-point 等 GP kernel。
   - 通过 SCM 将多个变量耦合，产生连续 latent pathology field。
   - label 由 latent state 的稳定机制决定，而不是由采样政策决定。

2. **Pathology bin factor `m_path`**
   - 借鉴 TCF 的 pathology-focused binning，将连续值转为变量特异的 low / normal / high / critical 区间。
   - 这里只作为合成 label 语义和辅助机制卡，不做 fixed viva、proof、IRT、foreseeing 或 hull。

3. **Observation-policy factor `m_policy`**
   - routine-round：固定查房窗口。
   - alarm-dense：状态越接近异常，局部复测越密集。
   - panel-pack：多个变量同步联测。
   - panel-split：同一 panel 被拆为异步事件。
   - variable-budget：变量组有观测预算。
   - pending-latency：事件已发生但值延迟返回。

4. **Noise / quality factor `m_quality`**
   - 异方差测量误差、变量级噪声、时间窗遮蔽、中心级单位偏移。
   - 吸收 StarEmbed / PULSE 中真实观测质量差异的启发，但不输出 evidential uncertainty。

5. **Center recipe factor `m_center`**
   - 组合不同变量 schema、采样覆盖、panel 习惯和记录延迟，模拟 PULSE 式跨中心部署。

核心是用 **orthogonal array / factorial design** 采样这些因子。对任意 label 与 policy 的组合，batch 中都近似均衡；对任意二阶组合 `(label, policy)`、`(state mechanism, policy)`、`(quality, policy)`，都覆盖到足够多单元格。这样数据层直接打断 policy shortcut 的统计基础。

### 2.2 改 Encoder：只加机制卡辅助头，不把政策卡送入分类器

DCOFF 不要求重写主干。现有 irregular encoder 继续接收：

```text
event_value, event_time, event_var_id, event_mask
```

输出：

```text
h_state, logits = EncoderClassifier(events)
```

新增两个只在预训练中使用的轻量 head：

1. **State Mechanism Card Head**
   - 从 `h_state` 预测合成世界的 `m_state` 与 `m_path`。
   - 作用是让表示学到“这条序列由什么状态机制生成”，而不是只学采样日历。

2. **Policy Report Head**
   - 不接 `h_state`，只接 dataloader 给出的采样坐标摘要 `policy_report`。
   - 预测 `m_policy / m_quality / m_center`，用于检查 forge 是否覆盖足够政策多样性。
   - 其输出不进入分类头，也不参与最终 real-data 推理。

关键设计：分类器从不接收机制卡、policy id、center id 或合成 metadata。机制卡只是训练时的 oracle annotation，用于让合成世界像“带答案的因果实验台”。

### 2.3 改 Loss：从一致性约束转向 Orthogonal Cell DRO

总目标：

```text
L = L_real_cls
  + lambda_synth * L_synth_cls
  + lambda_cell  * L_policy_cell_dro
  + lambda_card  * L_state_mechanism_card
  + lambda_forge * L_forge_coverage_audit
```

#### A. Real Classification `L_real_cls`

真实数据仍按普通分类训练：

```text
L_real_cls = CE(logits_real, y_real)
```

如果真实数据有 TCF-style pathology bins 或临床分箱标签，可作为普通辅助监督，但不是本提案核心。

#### B. Synthetic Classification `L_synth_cls`

合成样本带有 oracle label：

```text
L_synth_cls = CE(logits_synth, y_synth)
```

由于合成锻炉正交覆盖 label-policy 组合，单靠采样政策无法在所有单元格上同时降低该损失。

#### C. Policy-Cell DRO `L_policy_cell_dro`

对每个 `(label, policy, quality, center)` cell 计算平均分类损失，再用 soft worst-case 聚合：

```text
cell_loss[c] = mean_{i in cell c} CE(logits_i, y_i)
L_policy_cell_dro = tau * logsumexp_c(cell_loss[c] / tau)
```

这不是 risk variance，也不是多视图一致性。它不比较同一轨迹的不同 policy view，只要求模型在每个被正交设计覆盖的采样单元格中都能独立分类。若模型依赖某个训练政策捷径，最差 cell 会立刻升高。

#### D. State Mechanism Card `L_state_mechanism_card`

合成世界知道每条序列的状态生成机制：

```text
L_state_mechanism_card =
  CE(StateHead(h_state), m_state)
  + CE(PathologyHead(h_state), m_path)
```

它帮助 encoder 关注 GP+SCM 的状态因子，例如趋势、周期、突变、跨变量因果耦合和病理分箱，而不是采样政策。注意它不是 proof、IRT、poset clock 或 model-space posterior；只是合成预训练中的机制卡监督。

#### E. Forge Coverage Audit `L_forge_coverage_audit`

这项约束 dataloader 产生的 batch 足够正交，而不是约束模型表示：

```text
L_forge_coverage_audit =
  || P(label, policy) - P(label)P(policy) ||_F^2
  + || P(state, policy) - P(state)P(policy) ||_F^2
```

在实现上它主要用于监控和重采样；若用可学习 generator，也可反向更新 generator logits，使 forge 持续产生更均衡、更难的反例 cell。

### 2.4 反事实干预如何进入

当前“采样解耦/反事实干预”框架中的 counterfactual sampler 不再承担一致性、平滑、保形、risk-set、knockoff 或 privacy neighbor 的角色，而是变成 **反例锻造器**：

1. 在合成 latent state 上渲染多种 observation policy。
2. 保证 policy 与 label 在 batch / epoch 级别正交。
3. 根据模型最差 cell 自动加密采样薄弱区域：
   - 如果模型在 `alarm-dense + low-risk` cell 错误率高，下一轮增加该 cell。
   - 如果模型在 `routine-round + high-risk + pending` cell 错误率高，生成更多同类反例。
4. 真实数据 fine-tuning 时保留少量 forge replay，防止模型重新被真实医院中的偶然 policy-label 相关牵引。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ForgeConfig:
    batch_size: int = 128
    num_events: int = 64
    num_vars: int = 12
    num_classes: int = 2
    num_state_mechanisms: int = 6
    num_pathology_cards: int = 5
    num_policies: int = 6
    num_quality_cards: int = 4
    num_centers: int = 4
    hidden_dim: int = 128
    cell_temperature: float = 0.20


def orthogonal_factor_table(config: ForgeConfig, device: torch.device) -> dict:
    """Create a compact factorial table with balanced label-policy cells.

    This is a pragmatic orthogonal-array sketch: cyclic modular columns give
    balanced low-order interactions without requiring an external OA package.
    """

    n = config.batch_size
    row = torch.arange(n, device=device)
    label = row % config.num_classes
    policy = (row // config.num_classes + row) % config.num_policies
    state = (row // (config.num_classes * config.num_policies) + 2 * row) % config.num_state_mechanisms
    path = (state + label + row // 3) % config.num_pathology_cards
    quality = (policy + row // 5) % config.num_quality_cards
    center = (policy + state + row // 7) % config.num_centers
    return {
        "label": label,
        "policy_id": policy,
        "state_id": state,
        "pathology_id": path,
        "quality_id": quality,
        "center_id": center,
    }


class CauKerStateSCM(nn.Module):
    """Generate latent multivariate trajectories from GP-like basis functions and SCM mixing."""

    def __init__(self, config: ForgeConfig):
        super().__init__()
        self.config = config
        self.state_embed = nn.Embedding(config.num_state_mechanisms, config.hidden_dim)
        self.path_embed = nn.Embedding(config.num_pathology_cards, config.hidden_dim)
        self.label_shift = nn.Embedding(config.num_classes, config.num_vars)
        self.scm_edges = nn.Parameter(torch.randn(config.num_state_mechanisms, config.num_vars, config.num_vars) * 0.05)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 6, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.num_vars),
        )

    def forward(self, factors: dict) -> dict:
        bsz = factors["label"].size(0)
        device = factors["label"].device
        t = torch.linspace(0.0, 1.0, self.config.num_events, device=device)
        t = t[None].expand(bsz, -1)

        state_h = self.state_embed(factors["state_id"])
        path_h = self.path_embed(factors["pathology_id"])
        mech_h = state_h + path_h

        freq = 1.0 + (factors["state_id"].float() % 5.0)[:, None]
        phase = 0.17 * factors["pathology_id"].float()[:, None]
        trend = (factors["pathology_id"].float()[:, None] - 2.0) * (t - 0.5)
        periodic = torch.sin(2.0 * math.pi * freq * t + phase)
        local_bump = torch.exp(-0.5 * ((t - 0.25 - 0.08 * factors["label"].float()[:, None]) / 0.12).pow(2))
        changepoint = torch.tanh(10.0 * (t - 0.55 + 0.05 * factors["state_id"].float()[:, None]))
        rough = torch.sin(2.0 * math.pi * (freq + 2.0) * t) * torch.cos(2.0 * math.pi * t)
        label_carrier = self.label_shift(factors["label"])[:, None, :]

        basis = torch.stack([t, trend, periodic, local_bump, changepoint, rough], dim=-1)
        base = self.value_head(torch.cat([mech_h[:, None].expand(-1, self.config.num_events, -1), basis], dim=-1))
        edge = self.scm_edges[factors["state_id"]]
        mixed = base + 0.25 * torch.einsum("bij,bnj->bni", edge, base)
        value = mixed + 0.35 * label_carrier
        return {"time_grid": t, "full_value": value}


class ObservationPolicyRenderer(nn.Module):
    """Render irregular observations under policy, quality, and center factors."""

    def __init__(self, config: ForgeConfig):
        super().__init__()
        self.config = config
        self.center_var_bias = nn.Embedding(config.num_centers, config.num_vars)
        self.quality_noise = nn.Embedding(config.num_quality_cards, config.num_vars)

    def forward(self, full: dict, factors: dict) -> dict:
        value = full["full_value"]
        time = full["time_grid"]
        bsz, steps, num_vars = value.shape
        device = value.device

        center_bias = torch.sigmoid(self.center_var_bias(factors["center_id"]))[:, None, :]
        quality = F.softplus(self.quality_noise(factors["quality_id"]))[:, None, :] + 0.02

        time_norm = time[:, :, None]
        policy = factors["policy_id"]
        routine = ((torch.arange(steps, device=device)[None, :, None] % 8) == 0).float()
        alarm_score = torch.sigmoid(2.0 * value.abs())
        late = (time_norm > 0.66).float()
        early = (time_norm < 0.33).float()
        alternating = ((torch.arange(steps, device=device)[None, :, None] % 2) == 0).float()
        panel_pack = ((torch.arange(num_vars, device=device)[None, None, :] % 3) == 0).float()
        variable_budget = ((torch.arange(num_vars, device=device)[None, None, :] + policy[:, None, None]) % 2 == 0).float()
        pending_gate = 1.0 - 0.35 * late

        gates = []
        gates.append(0.15 + 0.85 * routine.expand(bsz, -1, num_vars))
        gates.append(0.10 + 0.70 * alarm_score + 0.20 * late)
        gates.append(0.15 + 0.75 * panel_pack.expand(bsz, steps, -1))
        gates.append(0.20 + 0.65 * (1.0 - panel_pack).expand(bsz, steps, -1) * alternating)
        gates.append(0.10 + 0.80 * variable_budget.expand(-1, steps, -1))
        gates.append((0.20 + 0.65 * early + 0.25 * routine).expand(bsz, -1, num_vars) * pending_gate)
        gate_bank = torch.stack(gates, dim=1).clamp(0.02, 0.98)
        chosen_gate = gate_bank[torch.arange(bsz, device=device), policy]
        chosen_gate = chosen_gate * center_bias

        obs_mask = torch.bernoulli(chosen_gate).to(value.dtype)
        noisy_value = value + quality * torch.randn_like(value)
        observed_value = noisy_value * obs_mask

        event_value, event_time, event_var, event_mask = dense_to_event_stream(
            observed_value=observed_value,
            time_grid=time,
            obs_mask=obs_mask,
            max_events=self.config.num_events,
        )
        policy_report = summarize_policy_report(event_time, event_var, event_mask, self.config.num_vars)
        return {
            "event_value": event_value,
            "event_time": event_time,
            "event_var_id": event_var,
            "event_mask": event_mask,
            "policy_report": policy_report,
            "measurement_std": quality.mean(dim=-1).expand(-1, steps),
        }


def dense_to_event_stream(
    observed_value: torch.Tensor,
    time_grid: torch.Tensor,
    obs_mask: torch.Tensor,
    max_events: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten dense synthetic observations into padded irregular event streams."""

    bsz, steps, num_vars = observed_value.shape
    device = observed_value.device
    event_value = torch.zeros(bsz, max_events, device=device)
    event_time = torch.zeros(bsz, max_events, device=device)
    event_var = torch.zeros(bsz, max_events, device=device, dtype=torch.long)
    event_mask = torch.zeros(bsz, max_events, device=device)

    for b in range(bsz):
        hit_t, hit_v = torch.where(obs_mask[b] > 0)
        if hit_t.numel() == 0:
            hit_t = torch.tensor([0], device=device)
            hit_v = torch.tensor([0], device=device)
        order = torch.argsort(time_grid[b, hit_t] + 1e-4 * hit_v.float())
        hit_t = hit_t[order][:max_events]
        hit_v = hit_v[order][:max_events]
        n = hit_t.numel()
        event_value[b, :n] = observed_value[b, hit_t, hit_v]
        event_time[b, :n] = time_grid[b, hit_t]
        event_var[b, :n] = hit_v
        event_mask[b, :n] = 1.0
    return event_value, event_time, event_var, event_mask


def summarize_policy_report(
    event_time: torch.Tensor,
    event_var: torch.Tensor,
    event_mask: torch.Tensor,
    num_vars: int,
) -> torch.Tensor:
    """Low-order observation-policy summary used only for forge diagnostics."""

    horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = event_time / horizon
    var_rate = F.one_hot(event_var.clamp(0, num_vars - 1), num_vars).to(event_time.dtype)
    var_rate = (var_rate * event_mask.unsqueeze(-1)).sum(dim=1)
    var_rate = var_rate / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    early = ((time_norm <= 0.33).to(event_time.dtype) * event_mask).mean(dim=1, keepdim=True)
    middle = (((time_norm > 0.33) & (time_norm <= 0.66)).to(event_time.dtype) * event_mask).mean(dim=1, keepdim=True)
    late = ((time_norm > 0.66).to(event_time.dtype) * event_mask).mean(dim=1, keepdim=True)
    delta_t = torch.zeros_like(event_time)
    delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
    mean_gap = (torch.log1p(delta_t) * event_mask).sum(dim=1, keepdim=True)
    mean_gap = mean_gap / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.cat([var_rate, early, middle, late, mean_gap], dim=-1)


class OrthogonalFalsificationForge(nn.Module):
    """CauKer-style synthetic world plus observation-policy orthogonalization."""

    def __init__(self, config: ForgeConfig):
        super().__init__()
        self.config = config
        self.state_scm = CauKerStateSCM(config)
        self.policy_renderer = ObservationPolicyRenderer(config)

    def forward(self, device: torch.device) -> dict:
        factors = orthogonal_factor_table(self.config, device)
        full = self.state_scm(factors)
        obs = self.policy_renderer(full, factors)
        obs["labels"] = factors["label"].long()
        obs["state_id"] = factors["state_id"].long()
        obs["pathology_id"] = factors["pathology_id"].long()
        obs["policy_id"] = factors["policy_id"].long()
        obs["quality_id"] = factors["quality_id"].long()
        obs["center_id"] = factors["center_id"].long()
        obs["cell_id"] = make_policy_cell_id(factors, self.config)
        return obs


def make_policy_cell_id(factors: dict, config: ForgeConfig) -> torch.Tensor:
    return (
        factors["label"]
        + config.num_classes * factors["policy_id"]
        + config.num_classes * config.num_policies * factors["quality_id"]
        + config.num_classes * config.num_policies * config.num_quality_cards * factors["center_id"]
    )


def cell_dro_loss(loss_per_sample: torch.Tensor, cell_id: torch.Tensor, temperature: float) -> torch.Tensor:
    """Soft worst-cell risk over orthogonal policy cells."""

    unique = torch.unique(cell_id)
    cell_losses = []
    for cid in unique:
        cell_losses.append(loss_per_sample[cell_id == cid].mean())
    cell_losses = torch.stack(cell_losses)
    return temperature * torch.logsumexp(cell_losses / temperature, dim=0)


class MechanismCardHead(nn.Module):
    """Auxiliary heads used only during forge pretraining."""

    def __init__(self, hidden_dim: int, config: ForgeConfig):
        super().__init__()
        self.state_head = nn.Linear(hidden_dim, config.num_state_mechanisms)
        self.path_head = nn.Linear(hidden_dim, config.num_pathology_cards)
        self.policy_report_head = nn.Sequential(
            nn.Linear(config.num_vars + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.num_policies + config.num_quality_cards + config.num_centers),
        )
        self.config = config

    def forward(self, value_state: torch.Tensor, policy_report: torch.Tensor) -> dict:
        policy_logits = self.policy_report_head(policy_report)
        p_end = self.config.num_policies
        q_end = p_end + self.config.num_quality_cards
        return {
            "state_logits": self.state_head(value_state),
            "pathology_logits": self.path_head(value_state),
            "policy_logits": policy_logits[:, :p_end],
            "quality_logits": policy_logits[:, p_end:q_end],
            "center_logits": policy_logits[:, q_end:],
        }


class DCOFFPretrainingWrapper(nn.Module):
    """Wrap an irregular classifier with the orthogonal falsification forge."""

    def __init__(self, base_model: nn.Module, forge: OrthogonalFalsificationForge, hidden_dim: int):
        super().__init__()
        self.base_model = base_model
        self.forge = forge
        self.cards = MechanismCardHead(hidden_dim, forge.config)

    def encode(self, batch: dict) -> dict:
        out = self.base_model(batch)
        if "value_state" not in out:
            out["value_state"] = out["logits"]
        return out

    def forge_loss(
        self,
        real_batch: dict,
        lambda_synth: float = 1.0,
        lambda_cell: float = 0.7,
        lambda_card: float = 0.2,
    ) -> dict:
        device = real_batch["event_value"].device
        synth = self.forge(device)

        real_out = self.encode(real_batch)
        real_loss = F.cross_entropy(real_out["logits"], real_batch["labels"])

        synth_out = self.encode(synth)
        sample_loss = F.cross_entropy(synth_out["logits"], synth["labels"], reduction="none")
        synth_loss = sample_loss.mean()
        cell_loss = cell_dro_loss(
            sample_loss,
            synth["cell_id"],
            temperature=self.forge.config.cell_temperature,
        )

        card = self.cards(synth_out["value_state"], synth["policy_report"])
        card_loss = (
            F.cross_entropy(card["state_logits"], synth["state_id"])
            + F.cross_entropy(card["pathology_logits"], synth["pathology_id"])
            + 0.25 * F.cross_entropy(card["policy_logits"], synth["policy_id"])
            + 0.25 * F.cross_entropy(card["quality_logits"], synth["quality_id"])
            + 0.25 * F.cross_entropy(card["center_logits"], synth["center_id"])
        )

        total = real_loss + lambda_synth * synth_loss + lambda_cell * cell_loss + lambda_card * card_loss
        return {
            "loss": total,
            "real_cls_loss": real_loss.detach(),
            "synth_cls_loss": synth_loss.detach(),
            "policy_cell_dro_loss": cell_loss.detach(),
            "mechanism_card_loss": card_loss.detach(),
        }
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `synthetic orthogonal pretraining`：先在 DCOFF forge 上预训练，再在 P12 / P19 / MIMIC / eICU / HiRID 风格真实数据上 fine-tune。
   - `PULSE-style cross-center shift`：把中心、变量 schema、采样频率、panel 习惯和 pending 延迟作为测试环境。
   - `TCF pathology bridge`：用 pathology-focused bins 定义合成状态机制与真实 EHR 数值分箱之间的桥接辅助任务。
   - `shortcut reversal stress`：训练真实数据中 label-policy 相关为正，测试中反转，例如 alarm-dense 不再代表高风险。

2. **对比方法**
   - CauKer 原始合成预训练，但不含 observation-policy SCM。
   - 普通 mask dropout / random missing augmentation。
   - 只做真实数据训练的 irregular Transformer / STAR-Set / VP-GNN / TCF-style encoder。
   - 普通 domain randomization，但不做正交数组和 policy-cell DRO。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DIPF、DRG-SFF、DPPC 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy worst-cell AUROC / AUPRC。
   - policy shortcut reversal error：训练相关反转后错误率。
   - worst synthetic policy-cell loss：锻炉中最差 `(label, policy, quality, center)` 单元格风险。
   - label-policy mutual information in forge：验证 dataloader 是否真的打断合成捷径。
   - fine-tuning shortcut relapse：真实数据 fine-tune 后，synthetic forge worst-cell 是否重新恶化。

4. **消融实验**
   - 去掉 orthogonal array，只随机采样合成政策，验证是否出现 label-policy 偶然相关。
   - 去掉 policy-cell DRO，只做平均合成 CE，检查少数困难政策 cell 是否被忽略。
   - 去掉 state mechanism card，检查 encoder 是否仍能学到 CauKer-style GP+SCM 状态机制。
   - 去掉 forge replay，只在真实数据 fine-tune，检查是否重新吸收真实医院 sampling shortcut。
   - 只用 observation-policy SCM、不用 CauKer GP+SCM，验证状态机制多样性对迁移的必要性。

## 5. 预期创新性

1. **从事后去偏转向事前反例锻造**：历史方案大多在真实数据表示上做约束；DCOFF 先构造一个 policy shortcut 必然失败的合成因果训练宇宙。
2. **从普通合成预训练转向采样政策正交合成预训练**：吸收 CauKer 的 GP+SCM 合成机制，但新增 observation-policy SCM、center recipe、quality factor 与正交数组设计。
3. **从平均风险转向 policy-cell worst-case risk**：不用多视图一致性或风险方差，而是在正交 policy cells 上做 soft worst-case DRO，直接惩罚只在某些采样政策上有效的规则。
4. **从反事实增强转向最小反例课程**：counterfactual sampler 不制造 positive pairs，而是为模型当前失败的 label-policy 单元格生成更多合成反例。
5. **与现有框架低侵入兼容**：不要求重写 Encoder；DCOFF 主要替换 dataloader / pretraining loop，适配任意 irregular classifier。

## 6. 一句话投稿卖点

**DCOFF 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“训练分布缺少足以推翻采样捷径的反例”问题，通过 CauKer-style GP+SCM 状态生成、observation-policy SCM、正交数组 factor crossing、policy-cell DRO 与 forge replay，让模型在进入真实 ICU/EHR 跨中心数据前已经见过同标签跨政策、同政策跨标签和弱势政策单元格的系统性反例，从数据生成层面阻断采样政策捷径。**
