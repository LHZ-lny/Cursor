# Title: Policy-Thermodynamic Free-Energy Annealer：面向采样策略偏移的策略热力学自由能退火器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md`：其中记录了多轮自动化任务均未发现 `my_work_summary.md`，并列出当前工作区未检出的历史机制。
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-06-12.md`
  - `paper_daily_2026-06-25.md`
  - `paper_daily_2026-06-26.md`
  - `paper_daily_2026-07-13.md`
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
  - `2026-06-17`: counterfactual error cartography / VQ semantic clauses / HSIC redaction。
  - `2026-06-20`: do-measure doubly robust risk rewriting / Radon-Nikodym density ratio。
  - `2026-06-24`: policy-shadow negative film / shadow eraser。
  - `2026-06-27`: observability-witness routing / Fisher-style observability gates。
  - `2026-07-15`: RKHS cubature debiaser / differentiable cubature KKT / control-variate exactness。
  - `2026-07-16`: measurement-action bisimulation lens / canonical measurement response heads。
  - `2026-07-17`: policy-word signature renormalizer / shuffle-algebra counterterms。

### 历史核心机制黑名单

为避免与历史提案和 `paper_daily.md` 中已有机制发生思维重合，本提案明确避开以下方向作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、hazard thinning、hazard-driven do-resampling。
8. 分类梯度与采样 score 的零空间正交化。
9. 多个 `do(policy)` 视图的 risk variance 约束。
10. 生理流算子与采样算子的交换子手术、value graph / policy graph 的交换或分离损失、policy residual sink。
11. additive evidence market、protocol tax、token-level evidence budget、边际证据审计。
12. 模型空间 posterior quotient、采样似然因子相除、干预积分分类。
13. reconstruction error cartography、ANOVA-style state/policy error projection、VQ semantic clauses、HSIC redaction checksum。
14. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler、policy coverage loss。
15. 采样测度 Radon-Nikodym density ratio、doubly robust target-measure correction、influence-bound regularization。
16. previsible martingale query、continuous-discrete standardized innovation、optional-stopping moment control。
17. soft excursion topology、censored persistence interval likelihood、censor envelope、fragmentation sobriety。
18. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
19. policy-only negative film、dual-exposure shadow encoders、latent shadow eraser/stencil。
20. latent packet codeword、learnable parity-check、syndrome locator、syndrome-guided repair。
21. conditional knockoff calendar、knockoff group statistics、soft knockoff-FDR firewall、swap symmetry calibration。
22. subjective-logic / Dirichlet evidential classification、policy-induced ignorance/vacuity mass。
23. observation-set policy lattice、meet/join visibility masks、单调/次模边际契约。
24. solver-trace mediator、NFE/step-size/front-door trace standardization、trace do-slope penalty。
25. RKHS cubature weights、kernel moment exactness、differentiable KKT cubature solver。
26. measurement-action bisimulation、canonical response battery、measurement action Bellman closure。
27. policy-word signature、mixed value-policy iterated integrals、shuffle-algebra counterterms。
28. 单纯复用 FlowPath/MVC-CDE 的可逆路径或多视图平滑、GSNF/DBGL/GARLIC 的图衰减、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec/MILM/LLMTS 的文本化或普通 LLM alignment、QuITE 的普通 query token、MTM/MedSpaformer 的 token selection/sparsification、MedMamba 的 frequency/adaptive graph、StarEmbed 的直接 foundation embedding。

本提案选择新的正交切入点：**不估计采样概率，不做对抗、不做一致性、不做后验除法、不做随机平滑认证、不做停时鞅、不做拓扑/gauge/纠错/knockoff/evidential/信息格/solver-trace/RKHS/measurement-action/signature；而是把采样策略偏移看成“观测能量景观的温度改变”。模型显式分解 value-driven internal energy 与 policy-induced entropy，并通过反事实退火路径约束类别自由能对采样温度的热容量，阻止分类边界依赖某个训练策略下的高熵观测流程。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列中的策略偏移，本质上常常改变的是“同一底层状态被观测系统释放了多少可用信息”：

- 医院 A 在报警后密集复测，观测系统像低温状态：短时间内释放大量细粒度证据；
- 医院 B 只在固定查房时刻采样，观测系统像高温状态：信息更稀疏、更混合、更不确定；
- 天文巡天中的天气、昼夜窗口、多波段 cadence 和异方差误差改变了某类物理状态的可恢复性；
- ICU 中 value-pending、化验延迟和测量质量变化让“已下单但值未返回”的采样事件携带高策略熵。

近期 `paper_daily.md` 中三类前沿机制给出启发：

1. **MVC-CDE / attentive kernel smoothing** 说明，控制路径的粗糙度和平滑带宽决定模型能否稳定读取不规则观测；但普通多视图平滑仍可能把某个采样策略下“更低噪、更密集、更容易平滑”的状态当成类别证据。
2. **StarEmbed** 强调异步多波段、异方差与 OOD detection，提示 sampling-policy shift 影响的是任务相关信息的可恢复性，而不仅是 mask ratio。
3. **Rethinking LLMs for ICU irregular classification** 指出前端 encoder 对时间戳、缺失和异步观测的处理比后端语义 alignment 更关键，因此鲁棒性应在 encoder/readout 层建立，而不是依赖大模型“理解”采样偏移。

**Policy-Thermodynamic Free-Energy Annealer (PT-FEA)** 的核心想法是：

> 把采样政策看成调节观测系统温度 `T` 的外部控制量。观测值提供 internal energy `E_y`，采样坐标、测量质量和异步结构提供 policy entropy `S_y`。分类不直接使用事实 logits，而使用在一条反事实温度退火路径上稳定的 class free energy：
>
> `F_y(T) = E_y(T) - T * S_y(T)`。

如果某个类别只在训练医院的密集复测、同步 panel 或低噪声 band coverage 下成立，那么它的自由能 margin 会对 `T` 极其敏感，表现为高 policy heat capacity。PT-FEA 不要求不同采样视图 logits 一致，也不把采样信息完全删除；它只惩罚“类别 margin 对采样温度的二阶热响应过大”。这样可以保留真实 informative sampling 的一阶可用性，同时压制策略捷径造成的陡峭相变。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通 event encoder 改为 Energy-Entropy 双读出

PT-FEA 可以包裹现有 irregular encoder，也可以作为新的前端：

1. **Value Internal Energy Encoder**
   - 输入事件 `(value, time, variable, delta_t, measurement_std)`。
   - 输出类别级 internal energy `E_y`，数值越低表示越支持类别 `y`。
   - 该分支主要由观测值、局部变化、测量质量校准后的 value token 决定。
   - 它不估计 hazard，不输出 posterior quotient，不做 graph/gauge/topology/code。

2. **Policy Entropy Meter**
   - 输入只包含采样坐标摘要、异步度、局部观测密度、测量噪声、value-pending 标记和反事实退火温度。
   - 输出类别级 entropy `S_y(T)`，表示当前采样温度下某类证据有多少来自观测流程释放的信息容量。
   - `S_y` 不单独作为分类 logit；它只通过自由能公式调节 energy readout。

3. **Free-Energy Classifier**
   - 对每个温度 `T` 计算：

```text
F_y(T) = E_y(T) - T * S_y(T)
logits_y(T) = -F_y(T)
```

   - 训练分类主损失使用一个 **annealed free-energy readout**，例如在 `T in {T_low, ..., T_high}` 的少量确定性温度点上做稳定积分：

```text
logits_anneal = - mean_T F(T)
```

这里的积分不是 randomized smoothing，也不给鲁棒半径；它只是用热力学路径估计“分类证据是否只在某个采样温度下成立”。

### 2.2 改 Dataloader：返回 Counterfactual Temperature Ladder

新增 `PolicyTemperatureCollator`。它不返回一致性正样本、不做 risk variance、不做 density ratio，也不生成 knockoff/stop/censor/channel views；它只为同一事实事件流构造一组可微温度控制：

1. **低温 `T < 1`：信息释放更充分**
   - 提高局部可见性权重；
   - 降低 `measurement_std`；
   - 保留 dense burst 与同步 panel；
   - 模拟“训练医院密集复测/低噪声设备”。

2. **高温 `T > 1`：信息更混合、更不确定**
   - 增大测量噪声；
   - 软化可见性；
   - 将同步 panel 拆成异步弱可见事件；
   - 模拟“稀疏查房、value-pending、天气/设备质量下降”。

3. **温度路径不是采样概率**
   - `T` 不等于 `p(observed)`，也不估计采样策略 likelihood；
   - 它只作为反事实干预旋钮，控制“观测过程释放多少可判别信息”；
   - 训练时用 deterministic ladder，避免与 DS-CS 的随机 policy-simplex smoothing 重合。

### 2.3 改 Loss：从一致性/对抗转向 Free-Energy Heat-Capacity Control

总目标：

```text
L = L_free_energy_cls
  + lambda_heat * L_policy_heat_capacity
  + lambda_maxwell * L_energy_entropy_maxwell
  + lambda_melt * L_melting_point_order
```

#### A. 自由能分类损失 `L_free_energy_cls`

对温度退火后的平均自由能做分类：

```text
L_free_energy_cls = CE(-mean_T F(T), y)
```

它不要求每个 `T` 下预测相同；若高温视图信息确实不足，单点预测可以变差。模型只需在整条温度路径的自由能证据上给出稳定类别。

#### B. 策略热容量惩罚 `L_policy_heat_capacity`

对真实类 margin：

```text
m_y(T) = -F_y(T) - max_{k != y} -F_k(T)
```

计算温度二阶差分：

```text
C_policy = |d^2 m_y(T) / dT^2|
```

若某个类别证据只在特定采样温度附近突然出现，`C_policy` 会很大，类似“策略相变”。惩罚：

```text
L_policy_heat_capacity = mean relu(C_policy - c_max)^2
```

这不是 consistency loss：它不把所有温度下的 logits 拉平，只限制 margin 曲线不能有由采样政策造成的尖锐二阶响应。真实急性状态可以在所有温度下保持低自由能；策略捷径通常只在某个温度段突然降低自由能。

#### C. Energy-Entropy Maxwell Relation `L_energy_entropy_maxwell`

自由能定义要求 entropy 近似满足：

```text
S_y(T) ≈ - dF_y(T) / dT
```

因此加入：

```text
L_energy_entropy_maxwell = SmoothL1(S(T), -finite_diff_T(F(T)))
```

这项防止 `S_y` 退化为任意 gating 分支。它不是 evidential uncertainty，也不是 policy residual；它是自由能分解的物理一致性约束。

#### D. Melting-Point Order Loss `L_melting_point_order`

借鉴 StarEmbed 的异方差和 OOD 启发，高噪声、低覆盖、value-pending 样本应该有更高的“熔点风险”：随着 `T` 升高，真实类 margin 更早变小。

定义：

```text
T_melt = min_T where margin_y(T) < margin_floor
```

用可微 soft-min 近似，并要求高质量样本的 `T_melt` 不低于低质量样本：

```text
L_melting_point_order =
  relu(T_melt_low_quality + delta - T_melt_high_quality)^2
```

它不是不确定性盾，也不是 OOD score；它只把观测质量与自由能退火曲线的崩塌顺序绑定起来，作为采样温度鲁棒性的诊断。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `ValueEnergyEncoder`，输出类别级 `E_y(T)`。
- 现有 sampling branch 改为 `PolicyEntropyMeter`，只输出 `S_y(T)` 与温度路径诊断，不进入 classifier。
- 现有 counterfactual intervention 改为 `TemperatureLadder`，生成测量质量、可见性、异步度和 value-pending 的温度控制，而不是 policy views、risk views、knockoff calendars、censor recipes、channel corruptions 或 trace interventions。
- 推理阶段：
  - **fast mode**：只用 `T = {0.8, 1.0, 1.2}` 三点自由能平均；
  - **diagnostic mode**：扫描温度曲线，输出 `policy heat capacity`、`melting temperature`、`entropy reliance ratio`；
  - **deployment warning**：若事实 `T=1` 预测很强但 `T` 轻微升高即崩塌，提示模型可能依赖训练采样策略释放的信息。

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


def true_margin_from_free_energy(free_energy: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # free_energy: [B, K, C], lower is better.
    logits = -free_energy
    true_logit = logits.gather(2, labels[:, None, None].expand(-1, logits.size(1), 1)).squeeze(-1)
    rival_logit = logits.masked_fill(
        F.one_hot(labels, logits.size(-1)).bool()[:, None, :],
        -1e4,
    ).max(dim=-1).values
    return true_logit - rival_logit


class PolicyTemperatureAdapter(nn.Module):
    """Apply deterministic temperature interventions to event visibility and quality."""

    def forward(self, batch: dict, temperature: torch.Tensor) -> dict:
        # temperature: [B], where T > 1 means higher policy entropy.
        out = dict(batch)
        value = batch["event_value"]
        mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(value))

        temp = temperature[:, None].to(value.dtype)
        visibility_scale = torch.sigmoid(2.0 - temp)
        noise_scale = temp.clamp_min(0.25)

        out["temperature"] = temperature
        out["event_mask_soft"] = mask * visibility_scale
        out["measurement_std"] = meas_std * noise_scale + 0.05 * (temp - 1.0).clamp_min(0.0)

        # High temperature weakens value precision without inventing pseudo-values.
        out["event_value_tempered"] = value / noise_scale.sqrt().clamp_min(1e-4)
        return out


class ValueEnergyEncoder(nn.Module):
    """Encode irregular values into class-wise internal energy."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        value = batch.get("event_value_tempered", batch["event_value"])
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch.get("event_mask_soft", batch["event_mask"])
        meas_std = batch.get("measurement_std", torch.zeros_like(value))
        temperature = batch["temperature"][:, None].expand_as(value)

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id.clamp_min(0))
        event_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(meas_std).unsqueeze(-1),
                torch.log1p(temperature).unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * mask.unsqueeze(-1)
        seq_h, _ = self.mixer(event_h)
        pooled = masked_mean(seq_h, mask, dim=1)

        # Lower energy should indicate stronger class support.
        return self.energy_head(pooled)


class PolicyEntropyMeter(nn.Module):
    """Measure class-wise policy entropy from sampling coordinates and temperature."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(time))
        temperature = batch["temperature"][:, None]

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_obs = F.one_hot(var_id.clamp_min(0), self.num_vars).to(time.dtype) * mask.unsqueeze(-1)
        var_rate = var_obs.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        early = (time_norm <= 0.33).to(time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype)
        late = (time_norm > 0.66).to(time.dtype)
        value_pending = batch.get("value_pending", torch.zeros_like(mask))

        stats = torch.cat(
            [
                mask.mean(dim=1, keepdim=True),
                (early * mask).mean(dim=1, keepdim=True),
                (middle * mask).mean(dim=1, keepdim=True),
                (late * mask).mean(dim=1, keepdim=True),
                torch.log1p(delta_t).mean(dim=1, keepdim=True),
                meas_std.mean(dim=1, keepdim=True),
                value_pending.mean(dim=1, keepdim=True),
                temperature,
            ],
            dim=-1,
        )
        return self.net(torch.cat([var_rate, stats], dim=-1))


class PolicyThermodynamicFreeEnergyAnnealer(nn.Module):
    """Sampling-policy robust classifier based on free-energy annealing."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.temperature_adapter = PolicyTemperatureAdapter()
        self.energy = ValueEnergyEncoder(num_vars, hidden_dim, num_classes)
        self.entropy = PolicyEntropyMeter(num_vars, hidden_dim, num_classes)
        self.num_classes = num_classes

    def free_energy_curve(self, batch: dict, temperatures: torch.Tensor) -> dict:
        energy_list = []
        entropy_list = []
        free_list = []
        for idx in range(temperatures.size(1)):
            temp = temperatures[:, idx]
            tempered = self.temperature_adapter(batch, temp)
            internal_energy = self.energy(tempered)
            policy_entropy = self.entropy(tempered)
            free_energy = internal_energy - temp[:, None] * policy_entropy
            energy_list.append(internal_energy)
            entropy_list.append(policy_entropy)
            free_list.append(free_energy)
        return {
            "energy": torch.stack(energy_list, dim=1),
            "entropy": torch.stack(entropy_list, dim=1),
            "free_energy": torch.stack(free_list, dim=1),
        }

    def training_loss(
        self,
        batch: dict,
        lambda_heat: float = 0.20,
        lambda_maxwell: float = 0.10,
        lambda_melt: float = 0.05,
        heat_capacity_cap: float = 0.35,
        margin_floor: float = 0.15,
    ) -> dict:
        labels = batch["labels"]
        bsz = labels.size(0)
        device = labels.device
        dtype = batch["event_value"].dtype

        base_t = torch.tensor([0.70, 0.90, 1.00, 1.15, 1.35], device=device, dtype=dtype)
        temperatures = base_t[None, :].expand(bsz, -1)
        curve = self.free_energy_curve(batch, temperatures)
        free_energy = curve["free_energy"]

        annealed_logits = -free_energy.mean(dim=1)
        cls_loss = F.cross_entropy(annealed_logits, labels)

        margin = true_margin_from_free_energy(free_energy, labels)
        dt = temperatures[:, 1:] - temperatures[:, :-1]
        first = (margin[:, 1:] - margin[:, :-1]) / dt.clamp_min(1e-6)
        mid_dt = 0.5 * (dt[:, 1:] + dt[:, :-1])
        second = (first[:, 1:] - first[:, :-1]) / mid_dt.clamp_min(1e-6)
        heat_loss = F.relu(second.abs() - heat_capacity_cap).pow(2).mean()

        # Maxwell-style relation: S ~= -dF/dT per class.
        free_diff = (free_energy[:, 1:] - free_energy[:, :-1]) / dt.unsqueeze(-1).clamp_min(1e-6)
        entropy_mid = 0.5 * (curve["entropy"][:, 1:] + curve["entropy"][:, :-1])
        maxwell_loss = F.smooth_l1_loss(entropy_mid, -free_diff.detach())

        # Soft melting temperature: lower values mean the margin collapses earlier.
        melt_weight = torch.softmax(-(margin - margin_floor).abs() * 8.0, dim=1)
        t_melt = (melt_weight * temperatures).sum(dim=1)
        quality = batch.get("sample_quality", 1.0 / (1.0 + batch.get("measurement_std", torch.zeros_like(batch["event_value"])).mean(dim=1))).detach()
        hi_quality = quality > quality.median()
        if hi_quality.any() and (~hi_quality).any():
            melt_loss = F.relu(t_melt[~hi_quality].mean() + 0.05 - t_melt[hi_quality].mean()).pow(2)
        else:
            melt_loss = torch.zeros((), device=device, dtype=dtype)

        total = cls_loss + lambda_heat * heat_loss + lambda_maxwell * maxwell_loss + lambda_melt * melt_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "policy_heat_capacity_loss": heat_loss.detach(),
            "maxwell_loss": maxwell_loss.detach(),
            "melting_order_loss": melt_loss.detach(),
            "mean_policy_heat_capacity": second.abs().mean().detach(),
            "mean_melting_temperature": t_melt.mean().detach(),
        }


@torch.no_grad()
def build_temperature_ladder(batch: dict) -> torch.Tensor:
    """Return deterministic policy-temperature ladder for diagnostics."""

    bsz = batch["event_value"].size(0)
    device = batch["event_value"].device
    dtype = batch["event_value"].dtype
    ladder = torch.tensor([0.70, 0.90, 1.00, 1.15, 1.35], device=device, dtype=dtype)
    return ladder[None, :].expand(bsz, -1)
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `temperature inversion shift`：训练环境高风险类低温密集复测，测试环境改为高温稀疏查房。
   - `heteroskedastic band shift`：借鉴 StarEmbed，多变量/多波段测量噪声和覆盖率共同改变。
   - `value-pending thermal shift`：借鉴 MILM，采样事件已出现但数值延迟返回，提高 policy entropy。
   - `kernel-smoothing temperature shift`：借鉴 MVC-CDE，用不同平滑带宽模拟低温精细观测与高温粗糙观测。

2. **对比方法**
   - 普通 irregular encoder。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - MVC-CDE / attentive kernel smoothing。
   - StarEmbed / foundation embedding + OOD baseline。
   - LLMTS / Record2Vec / MILM-style 语义或 value-redacted baseline。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN、RKHS Cubature、Measurement-Action Bisimulation、Policy-Word Signature。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - policy heat capacity：真实类 margin 对采样温度的二阶敏感度。
   - melting temperature：预测随采样温度升高而崩塌的温度点。
   - entropy reliance ratio：`T * S_y / |E_y|`。
   - thermal calibration under shift：高温高熵样本是否更容易被低置信或拒识。

4. **消融实验**
   - 去掉 `L_policy_heat_capacity`，检查模型是否出现策略相变式 margin。
   - 去掉 Maxwell relation，验证 entropy meter 是否退化成任意 gating。
   - 只用 `T=1` 事实自由能分类，验证收益不是额外参数量。
   - 将 deterministic temperature ladder 替换为随机 mask dropout，验证收益来自热力学路径而非普通增强。
   - 扫描温度点数量，评估推理开销与 heat-capacity 诊断稳定性。

## 5. 预期创新性

1. **从采样概率/结构去偏转向采样温度建模**：不问某个点为什么被采样，而问当前采样政策释放了多少可判别信息容量。
2. **从 logits 一致性转向自由能曲率控制**：不同温度下的预测可以不同；模型只需避免类别 margin 对采样温度出现尖锐二阶相变。
3. **从不确定性输出转向 energy-entropy 分解**：`S_y` 不是 evidential vacuity，也不是 OOD score，而是由 `S ~= -dF/dT` 约束的策略熵。
4. **吸收 MVC-CDE / StarEmbed / LLMTS 的前沿启发但不复用其机制**：将 path roughness、异方差可恢复性和 encoder 审计统一为 temperature ladder，而不是多视图平滑、foundation embedding 或 LLM alignment。
5. **与采样解耦/反事实干预框架低侵入兼容**：value branch 输出 internal energy，sampling branch 输出 entropy meter，counterfactual branch 输出温度阶梯。
6. **部署诊断直观**：如果一个预测只在低温密集采样下成立、在高温稀疏采样下迅速熔化，`policy heat capacity` 与 `melting temperature` 会直接暴露采样策略捷径。

## 6. 一句话投稿卖点

**PT-FEA 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“观测系统温度改变导致类别自由能曲线发生策略相变”的问题，并通过 energy-entropy 双读出、反事实温度退火路径、Maxwell-style 自由能分解和 policy heat-capacity penalty，让分类器依赖跨采样温度平滑存在的状态证据，而不是依赖训练环境中特定密度、低噪声、同步 panel 或 value-pending 流程释放出的策略性信息。**
