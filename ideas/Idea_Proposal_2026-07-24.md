# Title: Trigger-Phase Hysteresis Neutralizer：面向采样策略偏移的触发相位滞回中和分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md`：其中记录了多轮自动化任务均未发现 `my_work_summary.md`，并列出了历史提案机制黑名单。
- 已读取近期论文记录：`paper_daily.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-13.md`。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案机制：`2026-06-17`、`2026-06-20`、`2026-06-24`、`2026-06-27`、`2026-07-15`、`2026-07-16`、`2026-07-17`、`2026-07-18`、`2026-07-19`、`2026-07-20`、`2026-07-21`、`2026-07-22`、`2026-07-23`。

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
28. solver trace / NFE / roughness 中介、reference trace bank、front-door trace-standardized readout、trace do-slope penalty。
29. queue debt recurrence、service discipline bank、debt-neutral detail gate。
30. context/detail anchors、Sinkhorn detail canonicalization、exposure absorption。
31. MDL episode transducer、artifact edit program、label automaton over compressed episodes。
32. causal sheaf gluing、local sections、restriction maps、global section solver。
33. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing、MedMamba 的 frequency/adaptive graph branch、MedSpaformer 的普通 token sparsification、MILM 的 value-redacted classifier、StarEmbed 的直接 foundation embedding、LLMTS 的普通 LLM alignment、MVC-CDE 的普通多视图平滑路径，或最新 sparse event detection 论文中的普通 state-policy 双门控作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗，不要求多采样视图 logits 一致，不做后验除法、随机平滑认证、停时鞅、拓扑、gauge、纠错码、knockoff、观测性、证据无知、信息格、solver trace、队列债务、Sinkhorn 锚点、MDL 或 sheaf gluing；而是把最新 sparse event detection 的 context-detail adaptive gate 重新解释为“观测系统决定何时打开高分辨率细节检查”的触发阈值。采样政策偏移会改变这个触发阈值、触发延迟和触发预算。本方法让分类器不依赖某个事实触发阈值，而依赖一条内部 gate-threshold response curve 上的宽平台，并用滞回环面积和边界敏感性惩罚切断训练策略特有的触发捷径。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 `paper_daily_2026-07-19.md` 中的 **Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction** 提出了一个很有启发的 coarse-to-fine 机制：先由 global context explorer 判断是否存在可疑片段，再由 local detail inspector 在可能有事件的区域做精查，Adaptive Gating Module 负责控制上下文与细节何时交互。

这个机制对非规则采样分类非常关键，但直接照搬会引入新的 sampling-policy shortcut：

- 医院 A 可能只在粗粒度风险超过阈值后密集采样局部细节，医院 B 可能按固定查房细查；
- 设备 A 的报警阈值较低，会打开更多 detail windows，设备 B 的阈值较高，会只记录最强事件；
- 某类训练样本因为流程规则更容易触发高分辨率测量，AGM 学到的 gate 可能在无意中成为“采样政策触发器”；
- 如果分类器只在事实 gate 打开的位置读局部细节，那么换一个触发阈值、触发延迟或资源预算后，类别边界会随采样政策漂移。

历史方案大多把 sampling-policy shift 放在输入、后验、图、token、证据、信息格、数值求解或局部覆盖结构上。本提案关注一个尚未被历史机制覆盖的中介：

```text
sampling policy -> context-detail trigger threshold / latency / budget -> detail inspection phase -> logits
```

这里的关键不是“有没有 gate”，而是 **分类器到底依赖哪一个 gate phase**。如果模型只在某个医院协议特有的阈值附近突然变得高置信，那么它很可能利用了采样触发捷径；如果真实类别在一段较宽的 gate threshold 区间内都能被稳定识别，那么预测更接近底层状态，而不是某个事实采样策略。

**Trigger-Phase Hysteresis Neutralizer (TPHN)** 的核心目标是：

> 把 context-detail gate 从一个单点开关改造成一条可扫描的触发相位曲线；最终分类来自曲线上的稳定平台，而不是事实采样策略给出的单一触发阈值。模型可以输出“在哪些阈值下细节被打开”，但不能让真实类边际只在训练策略特有的窄触发相位上尖峰化。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从单阈值 AGM 改为 Trigger-Phase Context-Detail Encoder

TPHN 保留最新论文的 coarse-to-fine 精神，但把 AGM 的事实 gate 升级成可干预的触发相位轴。

1. **Global Context Explorer**
   - 输入不规则事件 `(value, time, variable, mask, measurement_std)`。
   - 输出粗粒度上下文状态 `c_t` 与触发能量 `a_t`。
   - `a_t` 表示“若资源允许，此处是否值得打开 detail inspector”，而不是直接作为分类证据。

2. **Local Detail Inspector**
   - 在事件局部窗口内抽取高分辨率细节 `d_t`，例如局部斜率、短时异常、变量间近邻交互或文本/化验细节。
   - 与普通 AGM 不同，`d_t` 不由一个事实 gate 直接决定是否进入分类头。

3. **Trigger-Phase Sweep Readout**
   - 构造一组内部触发阈值：

```text
tau_1 < tau_2 < ... < tau_Q
```

   - 对每个阈值生成 detail opening mask：

```text
w_t(tau_q) = sigmoid((a_t - tau_q) / temperature)
```

   - 每个阈值对应一个相位表示：

```text
h(tau_q) = Pool_t[ c_t + w_t(tau_q) * DetailAdapter(d_t) ]
logits(tau_q) = Classifier(h(tau_q))
```

   - 最终预测不是事实阈值下的 `logits(tau_factual)`，而是稳定平台上的 trimmed phase mean：

```text
logits_TPHN = TrimmedMean_{tau in stable plateau} logits(tau)
```

这不是 policy-simplex randomized smoothing：`tau` 不是外部采样政策扰动，也不输出认证半径；它是模型内部 context-detail gate 的触发相位扫描。TPHN 不要求不同采样视图 logits 一致，只要求分类边界不在单个政策特有触发阈值上尖峰化。

### 2.2 改 Loss：从普通 gate supervision 转向触发相位平台 + 滞回中和

总目标：

```text
L = L_phase_cls
  + lambda_flat * L_phase_margin_flatness
  + lambda_hys  * L_trigger_hysteresis
  + lambda_bnd  * L_boundary_quarantine
  + lambda_cov  * L_detail_coverage_sobriety
```

#### A. Phase-Plateau Classification Loss `L_phase_cls`

只用平台均值 logits 做分类：

```text
L_phase_cls = CE(TrimmedMean_tau logits(tau), y)
```

其中 trimmed mean 会去掉最早和最晚若干阈值，避免模型通过“所有 detail 全关”或“所有 detail 全开”的退化策略获益。

#### B. True-Margin Phase Flatness `L_phase_margin_flatness`

计算真实类 margin 随触发阈值变化的离散导数：

```text
margin_y(tau) = logit_y(tau) - max_{k != y} logit_k(tau)
phase_slope   = margin_y(tau_{q+1}) - margin_y(tau_q)
```

在平台区域惩罚过大斜率：

```text
L_phase_margin_flatness = mean_{tau in plateau} phase_slope^2
```

直觉：若模型只有在某个特定 gate threshold 刚好打开某类训练医院流程细节时才高置信，margin curve 会出现窄尖峰；如果类别证据来自稳定状态，margin 会在一段 threshold 区间内保持相对平坦。

这不是多视图一致性，因为它不比较两个反事实采样 view，也不要求所有 `logits(tau)` 相等；它只限制真实类边际对内部触发相位的一阶敏感性。

#### C. Trigger Hysteresis Loss `L_trigger_hysteresis`

采样政策常常有滞回：一旦报警触发，后续一段时间会继续密集复测；而阈值降低/升高的路径并不对称。TPHN 显式扫描两条内部路径：

- opening sweep：阈值从高到低，detail windows 逐步打开；
- closing sweep：阈值从低到高，detail windows 逐步关闭，并带一个轻量 memory gate。

得到 `logits_open(tau)` 与 `logits_close(tau)`。若预测强依赖触发历史而非状态证据，两条曲线会围成较大滞回环：

```text
L_trigger_hysteresis =
  mean_tau || softmax(logits_open(tau)) - softmax(logits_close(tau)) ||_1
```

这不是 sheaf gluing、queue recurrence 或停时鞅；它只把 gate 开关路径本身的滞回作为采样政策触发捷径的诊断与训练约束。

#### D. Boundary Quarantine Loss `L_boundary_quarantine`

detail gate 最危险的区域是阈值边界附近，因为那里少量采样政策变化就能让某个局部细节突然进入或离开分类器。TPHN 计算：

```text
boundary_t(tau) = w_t(tau) * (1 - w_t(tau))
```

若某个 token 只有在 boundary 区域才产生极大真实类 logit 贡献，说明模型可能依赖“刚好被触发”的策略性细节。惩罚边界贡献：

```text
L_boundary_quarantine =
  mean_{t,tau} boundary_t(tau) * attribution_t(tau)^2
```

这不同于 protocol tax / evidence market：没有证据预算、税费或拍卖；也不同于 observability gate：不估计 latent coordinate 可观测性。它只隔离触发阈值边界上的不稳定 detail 贡献。

#### E. Detail Coverage Sobriety `L_detail_coverage_sobriety`

为了避免模型通过永远打开或永远关闭 detail branch 逃避相位约束，加入覆盖清醒度：

```text
coverage(tau) = mean_t w_t(tau)
L_detail_coverage_sobriety =
  relu(coverage_min - mean_tau coverage)^2
  + relu(mean_tau coverage - coverage_max)^2
  + entropy_penalty_if_all_thresholds_identical
```

这让 detail inspector 真正参与决策，但不允许它退化成事实采样 mask 的硬编码复刻。

### 2.3 改 Dataloader：返回 Trigger-Phase Intervention Bank

新增 `TriggerPhaseCollator`。每个 batch 返回原始事件外，还返回一组只改变“触发细节检查方式”的 recipe：

1. `threshold_raise`：提高 detail opening 阈值，模拟保守医院或高报警阈值设备。
2. `threshold_lower`：降低 detail opening 阈值，模拟低阈值报警系统。
3. `trigger_latency`：让 detail window 延迟若干事件打开。
4. `budget_topk`：每个样本只允许打开前 `k` 个 detail windows。
5. `burst_collapse`：把报警后连续密集 detail 合并为一个粗 detail。

这些 recipe 的作用不是制造对比学习正样本，也不是要求 recipe 之间 logits 一致。它们只用于扩大触发相位曲线的扫描范围，并产生诊断指标：

- phase plateau width：真实类 margin 平台有多宽；
- hysteresis area：打开/关闭路径是否明显不对称；
- boundary reliance：预测是否集中依赖触发边界 token；
- detail coverage：模型是否退化为全开或全关。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为或包裹为 `ContextDetailPhaseEncoder`：输出粗上下文、局部细节与触发能量。
- 现有 sampling branch 改为 `TriggerRecipeGenerator`：不输出 policy code、不做对抗、不做概率估计，只生成触发阈值、延迟和预算 recipe。
- 现有 counterfactual intervention 改为 `TriggerPhaseInterventionBank`：定义内部 gate phase 的扫描和滞回路径，不构造一致性 view。
- 分类头只消费 `phase_plateau_logits`。
- 推理阶段输出：
  - `pred_label`；
  - `phase_plateau_width`；
  - `hysteresis_area`；
  - `boundary_reliance_score`；
  - 若某样本只有极窄阈值区间能被正确分类，则提示“预测可能依赖采样触发策略”。

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
    true_logit = logits.gather(-1, labels[:, None]).squeeze(-1)
    rival = logits.masked_fill(
        F.one_hot(labels, logits.size(-1)).bool(),
        -1e4,
    ).max(dim=-1).values
    return true_logit - rival


def trimmed_phase_mean(logits: torch.Tensor, trim: int = 1) -> torch.Tensor:
    # logits: [B, Q, C]
    if logits.size(1) <= 2 * trim:
        return logits.mean(dim=1)
    return logits[:, trim:-trim].mean(dim=1)


class ContextDetailPhaseEncoder(nn.Module):
    """Coarse context, local detail, and trigger energy for irregular events."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.context_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.detail_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.trigger_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.detail_adapter = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch: dict) -> dict:
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
        context, _ = self.context_rnn(event_h)

        # Detail inspector is local and high-resolution; the phase gate decides
        # how much of it can influence each threshold readout.
        detail = self.detail_conv(event_h.transpose(1, 2)).transpose(1, 2)
        detail = self.detail_adapter(detail) * event_mask.unsqueeze(-1)

        trigger_energy = self.trigger_head(context).squeeze(-1)
        trigger_energy = trigger_energy.masked_fill(event_mask == 0, -1e4)
        return {
            "context": context,
            "detail": detail,
            "trigger_energy": trigger_energy,
            "event_mask": event_mask,
        }


class TriggerPhaseReadout(nn.Module):
    """Scan internal gate thresholds and classify from phase-stable logits."""

    def __init__(self, hidden_dim: int, num_classes: int, num_thresholds: int = 9, temperature: float = 0.15):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.temperature = temperature
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.close_memory = nn.GRUCell(hidden_dim, hidden_dim)

    def threshold_bank(self, trigger_energy: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        masked = trigger_energy.masked_fill(event_mask == 0, float("nan"))
        # Use fixed quantile levels per sample; production code can replace this
        # with robust nanquantile utilities for older PyTorch versions.
        clean = masked.nan_to_num(nan=0.0)
        low = clean.masked_fill(event_mask == 0, 1e4).amin(dim=1)
        high = clean.masked_fill(event_mask == 0, -1e4).amax(dim=1)
        alpha = torch.linspace(0.10, 0.90, self.num_thresholds, device=clean.device, dtype=clean.dtype)
        return low[:, None] * (1.0 - alpha[None]) + high[:, None] * alpha[None]

    def phase_logits(self, context: torch.Tensor, detail: torch.Tensor, trigger_energy: torch.Tensor, event_mask: torch.Tensor) -> dict:
        tau = self.threshold_bank(trigger_energy, event_mask)
        logits = []
        gates = []
        pooled_states = []
        for idx in range(tau.size(1)):
            gate = torch.sigmoid((trigger_energy - tau[:, idx : idx + 1]) / self.temperature) * event_mask
            fused = context + gate.unsqueeze(-1) * detail
            pooled = masked_mean(fused, event_mask, dim=1)
            logits.append(self.classifier(pooled))
            gates.append(gate)
            pooled_states.append(pooled)
        return {
            "tau": tau,
            "logits_tau": torch.stack(logits, dim=1),
            "gate_tau": torch.stack(gates, dim=1),
            "pooled_tau": torch.stack(pooled_states, dim=1),
        }

    def hysteresis_logits(self, context: torch.Tensor, detail: torch.Tensor, trigger_energy: torch.Tensor, event_mask: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        open_logits = []
        close_logits = []
        close_state = torch.zeros(context.size(0), context.size(-1), device=context.device, dtype=context.dtype)

        for idx in range(tau.size(1)):
            gate_open = torch.sigmoid((trigger_energy - tau[:, idx : idx + 1]) / self.temperature) * event_mask
            open_pooled = masked_mean(context + gate_open.unsqueeze(-1) * detail, event_mask, dim=1)
            open_logits.append(self.classifier(open_pooled))

        for idx in reversed(range(tau.size(1))):
            gate_close = torch.sigmoid((trigger_energy - tau[:, idx : idx + 1]) / self.temperature) * event_mask
            close_pooled = masked_mean(context + gate_close.unsqueeze(-1) * detail, event_mask, dim=1)
            close_state = self.close_memory(close_pooled, close_state)
            close_logits.append(self.classifier(close_state))

        close_logits = list(reversed(close_logits))
        return torch.stack(open_logits, dim=1), torch.stack(close_logits, dim=1)


class TriggerPhaseHysteresisNeutralizer(nn.Module):
    """Sampling-policy robust classifier based on gate-threshold phase plateaus."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int, num_thresholds: int = 9):
        super().__init__()
        self.encoder = ContextDetailPhaseEncoder(num_vars, hidden_dim)
        self.readout = TriggerPhaseReadout(hidden_dim, num_classes, num_thresholds=num_thresholds)
        self.boundary_probe = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict) -> dict:
        enc = self.encoder(batch)
        phase = self.readout.phase_logits(
            context=enc["context"],
            detail=enc["detail"],
            trigger_energy=enc["trigger_energy"],
            event_mask=enc["event_mask"],
        )
        logits = trimmed_phase_mean(phase["logits_tau"], trim=1)
        open_logits, close_logits = self.readout.hysteresis_logits(
            context=enc["context"],
            detail=enc["detail"],
            trigger_energy=enc["trigger_energy"],
            event_mask=enc["event_mask"],
            tau=phase["tau"],
        )
        return {**enc, **phase, "logits": logits, "open_logits": open_logits, "close_logits": close_logits}

    def phase_flatness_loss(self, logits_tau: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bsz, num_tau, num_classes = logits_tau.shape
        flat_logits = logits_tau.reshape(bsz * num_tau, num_classes)
        flat_labels = labels[:, None].expand(-1, num_tau).reshape(-1)
        margin = true_margin(flat_logits, flat_labels).view(bsz, num_tau)
        if num_tau <= 2:
            return torch.zeros((), device=logits_tau.device, dtype=logits_tau.dtype)
        slope = margin[:, 1:] - margin[:, :-1]
        # Focus on the inner plateau, not degenerate all-off/all-on boundaries.
        if slope.size(1) > 2:
            slope = slope[:, 1:-1]
        return slope.pow(2).mean()

    def hysteresis_loss(self, open_logits: torch.Tensor, close_logits: torch.Tensor) -> torch.Tensor:
        open_prob = F.softmax(open_logits, dim=-1)
        close_prob = F.softmax(close_logits, dim=-1)
        return (open_prob - close_prob).abs().mean()

    def boundary_quarantine_loss(self, out: dict, labels: torch.Tensor) -> torch.Tensor:
        gate = out["gate_tau"]  # [B, Q, N]
        boundary = gate * (1.0 - gate)
        detail_logits = self.boundary_probe(out["detail"])  # [B, N, C]
        true_contrib = detail_logits.gather(
            dim=-1,
            index=labels[:, None, None].expand(-1, detail_logits.size(1), 1),
        ).squeeze(-1)
        raw = boundary * true_contrib[:, None].pow(2)
        mask = out["event_mask"][:, None]
        return (raw * mask).sum() / (mask.sum() * gate.size(1)).clamp_min(1.0)

    def coverage_sobriety_loss(self, gate_tau: torch.Tensor, event_mask: torch.Tensor, low: float = 0.10, high: float = 0.85) -> torch.Tensor:
        coverage = (gate_tau * event_mask[:, None]).sum(dim=-1) / event_mask[:, None].sum(dim=-1).clamp_min(1.0)
        mean_coverage = coverage.mean(dim=1)
        spread = coverage.std(dim=1)
        return (
            F.relu(low - mean_coverage).pow(2)
            + F.relu(mean_coverage - high).pow(2)
            + F.relu(0.05 - spread).pow(2)
        ).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_flat: float = 0.20,
        lambda_hys: float = 0.15,
        lambda_bnd: float = 0.05,
        lambda_cov: float = 0.05,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]

        cls_loss = F.cross_entropy(out["logits"], labels)
        flat_loss = self.phase_flatness_loss(out["logits_tau"], labels)
        hys_loss = self.hysteresis_loss(out["open_logits"], out["close_logits"])
        bnd_loss = self.boundary_quarantine_loss(out, labels)
        cov_loss = self.coverage_sobriety_loss(out["gate_tau"], out["event_mask"])

        total = (
            cls_loss
            + lambda_flat * flat_loss
            + lambda_hys * hys_loss
            + lambda_bnd * bnd_loss
            + lambda_cov * cov_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "phase_flatness_loss": flat_loss.detach(),
            "trigger_hysteresis_loss": hys_loss.detach(),
            "boundary_quarantine_loss": bnd_loss.detach(),
            "coverage_sobriety_loss": cov_loss.detach(),
        }


@torch.no_grad()
def build_trigger_phase_intervention_bank(batch: dict) -> dict:
    """Create trigger-phase recipes for diagnostics, not logit consistency pairs."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = value.shape
    device = value.device

    recipe = []
    recipe_name = []

    # 1. Conservative threshold: keep only high-amplitude candidate details.
    amp = value.abs()
    amp_cut = amp.masked_fill(mask == 0, -1e4).quantile(0.70, dim=1, keepdim=True)
    high_trigger = (amp >= amp_cut).to(mask.dtype) * mask
    recipe.append(high_trigger)
    recipe_name.append("threshold_raise")

    # 2. Liberal threshold: open more detail windows.
    low_cut = amp.masked_fill(mask == 0, 1e4).quantile(0.30, dim=1, keepdim=True)
    low_trigger = (amp >= low_cut).to(mask.dtype) * mask
    recipe.append(low_trigger)
    recipe_name.append("threshold_lower")

    # 3. Trigger latency: delay detail opening by one event.
    delayed = torch.zeros_like(mask)
    delayed[:, 1:] = mask[:, :-1]
    recipe.append(delayed * mask)
    recipe_name.append("trigger_latency")

    # 4. Budget top-k: only first few events per variable can open details.
    budget = torch.zeros_like(mask)
    for var_idx in torch.unique(var_id[mask > 0]).tolist():
        var_hit = ((var_id == int(var_idx)) & (mask > 0)).to(mask.dtype)
        order = var_hit.cumsum(dim=1)
        budget = torch.maximum(budget, ((order <= 2).to(mask.dtype) * var_hit))
    recipe.append(budget * mask)
    recipe_name.append("budget_topk")

    # 5. Burst collapse approximation: keep alternating events inside dense runs.
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    recipe.append(alternating.expand(bsz, -1) * mask)
    recipe_name.append("burst_collapse")

    out = dict(batch)
    out["trigger_recipe_visibility_bank"] = torch.stack(recipe, dim=1)
    out["trigger_recipe_names"] = recipe_name
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `threshold shift`：训练环境低触发阈值，测试环境高触发阈值，或反向。
   - `latency shift`：训练医院报警后立即细查，测试医院延迟若干事件/小时细查。
   - `detail budget shift`：训练环境允许高频局部观测，测试环境每变量只允许有限 detail windows。
   - `burst collapse shift`：训练环境保留报警后密集复测，测试环境合并为粗粒度记录。

2. **对比方法**
   - 普通 irregular encoder。
   - 普通 context-detail AGM。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - MILM-style value-redacted sampling classifier。
   - MedSpaformer-style token sparsification。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN、queue debt、Sinkhorn anchor、MDL transducer、sheaf glue。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - phase plateau width：真实类 margin 保持高于阈值的触发相位宽度。
   - hysteresis area：opening / closing sweep 的概率曲线面积差。
   - boundary reliance score：预测依赖 gate boundary token 的程度。
   - trigger-shift calibration error：触发阈值变化下高置信错误比例。

4. **消融实验**
   - 去掉 `L_phase_margin_flatness`，验证模型是否回到窄阈值尖峰。
   - 去掉 `L_trigger_hysteresis`，验证触发路径记忆是否污染分类。
   - 去掉 `L_boundary_quarantine`，检查边界 token 是否成为策略捷径。
   - 只用事实 gate 单阈值分类，验证收益不是 context-detail 结构本身。
   - 将 trigger recipes 替换为随机 mask，验证收益来自触发相位结构而非普通增强。

## 5. 预期创新性

1. **从采样模式偏移转向触发相位偏移**：把 sampling-policy shift 表述为 context-detail gate 的阈值、延迟与预算改变，而不是 mask ratio、density ratio、hazard、图结构或后验因子改变。
2. **从单点 gate 转向相位曲线分类**：吸收最新 sparse event detection 的 adaptive gate 机制，但不复用普通 state-policy 双门控；分类来自 gate-threshold response curve 的稳定平台。
3. **从一致性增强转向平台几何约束**：不要求反事实采样视图 logits 相同，只抑制真实类 margin 对内部触发阈值的一阶尖峰依赖。
4. **从采样去偏转向滞回中和**：opening / closing sweep 的滞回环直接诊断模型是否依赖“触发历史”，适合医疗报警、设备阈值和资源预算改变场景。
5. **部署诊断直接可读**：若预测只在极窄触发相位成立或边界依赖过高，系统可提示“需要补采样/降低触发策略依赖”，比单纯输出不确定性或 policy score 更贴近操作流程。

## 6. 一句话投稿卖点

**TPHN 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“context-detail inspection trigger phase 的阈值/延迟/预算偏移”问题，并通过触发阈值相位扫描、平台式分类读出、opening/closing 滞回环惩罚与边界贡献隔离，让模型利用稀疏事件检测中的 coarse-to-fine 细查能力，同时避免把训练医院或设备特有的细查触发规则学成类别捷径。**
