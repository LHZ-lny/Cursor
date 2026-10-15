# Title: Observability-Witness State Routing：面向采样策略偏移的可观测性证人状态路由

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md`，其中记录了当前工作区未检出的历史提案机制摘要。
- 已读取近期 paper daily：
  - `paper_daily.md`
  - `paper_daily_2026-06-12.md`
  - `paper_daily_2026-06-25.md`
  - `paper_daily_2026-06-26.md`
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案核心机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`
  - `Idea_Proposal_2026-06-24.md`

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
24. sampling calendar abstraction、conditional knockoff calendar、group-level knockoff statistics、soft knockoff-FDR firewall、swap symmetry calibration。
25. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing、MedMamba 的 frequency/adaptive graph branch、MedSpaformer 的普通 token sparsification、MILM 的 value-redacted classifier 或 LLM 文本摘要作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不删除采样信息，不做对抗、不做一致性、不做 knockoff、不做纠错码、不做拓扑/规范/鞅/平滑认证；而是把 sampling-policy shift 视为“某些潜在状态坐标在新采样策略下变得不可观测”。模型为每个状态坐标构造可观测性证人分数，分类头只能高权重使用在反事实采样策略族下仍可被观测值识别的状态坐标。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列的真正危险，不只是观测点变少，而是**模型以为自己看见了某个状态坐标，其实这个坐标只是在训练采样策略下被“照亮”了**：

- 医院 A 在报警后密集复测，使短期恶化速度可见；医院 B 只在固定查房窗口测量，该速度坐标在测试时不可辨识。
- 天文巡天 A 的 cadence 覆盖关键亮度变化段，巡天 B 因天气或波段调度漏掉关键片段，同一个类别的形态坐标可恢复性发生改变。
- ICU 数据中某些 lab panel 在训练机构中同步出现，模型能间接识别一个组合状态；换成异步或延迟返回后，该组合状态不再被当前观测支持。

近期 `paper_daily_2026-06-26.md` 的两个机制给出关键启发：

1. **StarEmbed** 提醒我们，采样策略会改变任务相关信息的可恢复性。对变量星来说，不同 cadence、波段覆盖和测量误差会决定同一物理类别的相位、周期形态或异常源结构是否能被看见。迁移鲁棒性不只是 representation distance 小，而是要知道哪些语义坐标在当前观测政策下仍可被识别。
2. **Rethinking LLMs for Irregular Time Series** 说明，大模型或语义对齐不是捷径；前端 encoder 是否尊重不规则观测、缺失、时间戳和异步结构，比后端 LLM alignment 更关键。因此，应该在 encoder 端显式审计“某个隐状态维度是否真的由观测值支持”，而不是把序列粗暴转文本或依赖后端语义模块自动修正采样偏差。

Observability-Witness State Routing (OWSR) 的核心直觉是：

> 对 sampling-policy shift 鲁棒的分类器，不应只问“哪个状态维度对类别有用”，还应问“这个状态维度在训练、测试和反事实采样策略下是否仍被观测算子支持”。若某个维度只在训练医院的特定复测节奏下可见，它可以进入不确定性诊断，但不应成为高置信分类主证据。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 继续学习潜在状态 `z`；
- sampling process 不输出 hazard、ratio、policy residual、calendar knockoff、syndrome 或 gauge frame，而是输出反事实观测算子族；
- counterfactual intervention 不用于一致性、不用于风险方差、不用于平滑认证，而是用于计算每个状态坐标在不同采样策略下的 **observability witness**；
- classifier 通过 state-coordinate routing 使用可观测坐标，对不可观测坐标降权并转入不确定性头。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从 pooled representation 改为带可观测性证人的状态坐标

给定事件流 `(t_i, d_i, x_i)`，现有 irregular encoder 输出一个样本级潜在状态：

```text
z = StateEncoder({x_i, t_i, d_i}) in R^H
```

OWSR 额外加入一个轻量 measurement decoder，用于回答：如果给定潜在状态 `z`、变量 `d` 和时间 `t`，它能解释当前观测值到什么程度？

```text
mu_i, sigma_i = MeasurementDecoder(z, t_i, d_i)
```

这里的 decoder 不是 iTimER 式重构误差地图，也不生成伪观测。它只服务一个目的：用观测值对潜在状态坐标的局部敏感性，估计当前采样策略下每个 `z_k` 是否被观测算子支持。

对每个潜在维度 `k`，计算对角 Fisher-style 可观测性证人：

```text
O_k = sum_i m_i * (d mu_i / d z_k)^2 / sigma_i^2
```

直觉：

- 若 `O_k` 高，说明当前采样到的观测值对状态坐标 `z_k` 有足够约束；
- 若 `O_k` 低，说明该坐标即使对分类有用，也主要是模型从训练分布先验、采样捷径或相关变量外推中“猜出来”的；
- 当反事实采样策略改变后，若某个维度的 `O_k` 大幅下降，它就是 sampling-policy shift 下的脆弱坐标。

### 2.2 改 Dataloader：新增 Observability Probe Bank

新增 `ObservabilityProbeCollator`。它不构造一致性正样本、不做 hazard thinning、不做 knockoff、不做 syndrome corruption，而是为每条样本返回一组观测算子 probe：

1. `factual_probe`：事实观测事件。
2. `cadence_probe`：改变观测 cadence，例如稀疏化早期或晚期窗口。
3. `bandwidth_probe`：限制变量预算，例如某类 lab 只保留前几次观测。
4. `asynchrony_probe`：拆散 panel 或延迟某些变量，改变同步可观测性。
5. `uncertainty_probe`：保留事件但增大测量噪声或降低可信度，用于模拟 StarEmbed 中异方差观测误差。

这些 probe 只用于估计 `O_k^{(r)}`。训练时不要求不同 probe 的 logits 相同；也不要求表示一致。我们只计算每个状态坐标在 probe bank 下的低分位可观测性：

```text
O_k^min = quantile_r(O_k^{(r)}, q)
g_k = sigmoid((log(1 + O_k^min) - beta_k) / tau)
```

`g_k` 是状态坐标路由门：它不是 token sparsification，因为它不选择观测 token；也不是 policy branch，因为它不编码策略标签；它选择的是**哪些潜在状态坐标被当前及反事实观测算子可靠支持**。

### 2.3 改 Classifier：State Routing，而不是 policy embedding 分类

分类头只高权重读取可观测坐标：

```text
z_route = g * z + (1 - g) * stopgrad(z) * rho
logits = Classifier(z_route)
u = UncertaintyHead((1 - g) * z)
```

其中 `rho` 是一个小系数，表示不可观测坐标可以作为弱背景上下文，但不能成为主证据。推理阶段输出：

- 分类概率；
- 每个状态坐标的 observability witness；
- low-observability reliance score；
- 若高置信预测主要依赖低 `g_k` 坐标，则触发 policy-shift warning。

### 2.4 改 Loss：从不变性/对抗转向可观测性支持约束

总目标：

```text
L = L_cls
  + lambda_meas * L_measurement_nll
  + lambda_route * L_low_observability_reliance
  + lambda_order * L_probe_ordering
  + lambda_unc   * L_unobservability_calibration
```

#### A. 分类损失 `L_cls`

使用路由后的状态分类：

```text
L_cls = CE(Classifier(z_route), y)
```

OWSR 不禁止模型学习复杂状态，也不强迫各采样策略预测一致。它只要求高置信分类证据来自可观测坐标。

#### B. Measurement NLL `L_measurement_nll`

measurement decoder 需要提供有意义的局部观测算子：

```text
L_measurement_nll =
  sum_i m_i * 0.5 * ((x_i - mu_i) / sigma_i)^2 + log sigma_i
```

这不是重构误差伪观测，也不是 error cartography。误差值不进入分类器，也不被投影成 state/policy effect；它只用于让 Jacobian-based observability witness 有物理含义。

#### C. Low-Observability Reliance `L_low_observability_reliance`

惩罚分类头在低可观测坐标上放大权重：

```text
R_low = sum_k ||W_cls[:, k]||_2 * stopgrad(1 - g_k) * |z_k|
L_low_observability_reliance = mean(R_low)
```

这不同于梯度与采样 score 正交化：OWSR 不计算采样 score，也不约束分类梯度和采样梯度夹角。它只检查分类证据是否由观测算子可识别的状态坐标支持。

#### D. Probe Ordering `L_probe_ordering`

反事实 probe 的采样更弱时，可观测性不应反常变高。对由 dataloader 标注的 probe 强弱序关系 `(r_strong, r_weak)`：

```text
L_probe_ordering =
  mean_k relu(O_k^{weak} - O_k^{strong} - margin)^2
```

它不是跨视图一致性，因为不比较 representation 或 logits；只校准 observability witness 的单调语义。

#### E. Unobservability Calibration `L_unobservability_calibration`

当模型必须依赖大量低可观测坐标才能预测时，应降低置信度或提高不确定性：

```text
low_obs_mass = mean_k (1 - g_k) * |z_k|
L_unc = BCE(UncertaintyHead(low_obs_mass), wrong_or_abstain_target)
```

训练中可用在线错误指示的 stop-gradient 近似，或用 probe bank 下低可观测质量超过阈值作为弱监督。这不是 certified smoothing，也不输出鲁棒半径；它只是部署校准。

### 2.5 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 保留为 `StateEncoder`。
- 现有 sampling branch 改为 `ProbePolicyGenerator`：输出一组观测算子 probe，不估计采样概率，不进入分类器。
- 现有 counterfactual intervention 改为 `ObservabilityProbeBank`：提供不同 cadence、预算、异步、噪声策略下的观测支持检查。
- 推理阶段无需测试环境标签；对事实观测和少量标准 probe 估计 `g_k`，再进行 routed classification。
- 若已有 LLM/semantic 模块，它只能消费 routed state 与 observability report，不能直接消费原始采样文本摘要。

### 2.6 为什么与历史提案显著正交

OWSR 的主创新是 **状态坐标可观测性证人 + 分类状态路由**：

- 不是 missingness encoder：采样模式不直接进入分类头。
- 不是 hazard/null-space：不估计点过程危险率，也不做分类梯度与采样 score 正交。
- 不是 graph/commutator：不学习生理图或策略图，也不计算算子交换子。
- 不是 evidence market：没有 token 购买、协议税或边际证据预算。
- 不是 posterior quotient：不在模型空间除去采样因子。
- 不是 error cartography：不把重构误差分解成 state/policy 主效应。
- 不是 randomized smoothing：不平均 policy logits，也不给 certified radius。
- 不是 density ratio / doubly robust：不估计目标采样测度。
- 不是 martingale/topology/gauge/shadow/syndrome/knockoff：没有停时矩、拓扑胶囊、横向投影、负片擦除、纠错码或 FDR 负控。
- 不是普通 token sparsification：路由对象是 latent state coordinates 的可观测性，而不是选择 token、query 或 calendar group。

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


class MeasurementDecoder(nn.Module):
    """Local observation operator used to build observability witnesses."""

    def __init__(self, latent_dim: int, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, 1)
        self.log_sigma = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # z: [B, H], event_time/event_var_id: [B, N]
        bsz, num_events = event_time.shape
        z_event = z[:, None, :].expand(bsz, num_events, -1)
        var_h = self.var_embed(event_var_id.clamp_min(0))
        horizon = event_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_x = (event_time / horizon).unsqueeze(-1)
        h = self.net(torch.cat([z_event, var_h, time_x], dim=-1))
        mu = self.mu(h).squeeze(-1)
        sigma = F.softplus(self.log_sigma(h).squeeze(-1)) + 1e-3
        return mu, sigma


def measurement_nll(
    value: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    raw = 0.5 * ((value - mu) / sigma).pow(2) + torch.log(sigma)
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


class ObservabilityWitness(nn.Module):
    """Estimate per-coordinate observability under a bank of sampling probes."""

    def __init__(self, measurement: MeasurementDecoder, latent_dim: int):
        super().__init__()
        self.measurement = measurement
        self.log_threshold = nn.Parameter(torch.zeros(latent_dim))

    def diagonal_fisher(
        self,
        z: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Enable per-sample latent sensitivity. H is usually modest, so this
        # autograd sketch is acceptable for research prototyping.
        z_probe = z.detach().clone().requires_grad_(True)
        mu, sigma = self.measurement(z_probe, event_time, event_var_id)
        weighted_mu = mu * event_mask
        fisher_terms = []
        for event_idx in range(event_time.size(1)):
            grad = torch.autograd.grad(
                weighted_mu[:, event_idx].sum(),
                z_probe,
                retain_graph=True,
                create_graph=True,
            )[0]
            weight = event_mask[:, event_idx : event_idx + 1] / sigma[:, event_idx : event_idx + 1].pow(2)
            fisher_terms.append(grad.pow(2) * weight)
        return torch.stack(fisher_terms, dim=0).sum(dim=0)  # [B, H]

    def forward(self, z: torch.Tensor, batch: dict, quantile: float = 0.25) -> dict:
        witnesses = []
        for probe_time, probe_var, probe_mask in zip(
            batch["probe_event_time_bank"].unbind(dim=1),
            batch["probe_event_var_bank"].unbind(dim=1),
            batch["probe_event_mask_bank"].unbind(dim=1),
        ):
            witnesses.append(self.diagonal_fisher(z, probe_time, probe_var, probe_mask))
        witness_bank = torch.stack(witnesses, dim=1)  # [B, R, H]
        low_witness = torch.quantile(witness_bank, q=quantile, dim=1)
        gate = torch.sigmoid(torch.log1p(low_witness) - self.log_threshold.view(1, -1))
        return {
            "witness_bank": witness_bank,
            "low_witness": low_witness,
            "gate": gate,
        }


class ObservabilityRoutedClassifier(nn.Module):
    """Route classification through state coordinates supported by observations."""

    def __init__(
        self,
        state_encoder: nn.Module,
        latent_dim: int,
        num_vars: int,
        hidden_dim: int,
        num_classes: int,
        leak: float = 0.05,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.measurement = MeasurementDecoder(latent_dim, num_vars, hidden_dim)
        self.witness = ObservabilityWitness(self.measurement, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.leak = leak

    def encode_state(self, batch: dict) -> torch.Tensor:
        out = self.state_encoder(
            event_value=batch["event_value"],
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
        )
        if isinstance(out, dict):
            return out["state"]
        return out

    def forward(self, batch: dict) -> dict:
        z = self.encode_state(batch)
        obs = self.witness(z, batch)
        gate = obs["gate"]
        z_route = gate * z + (1.0 - gate) * z.detach() * self.leak
        logits = self.classifier(z_route)
        low_obs_mass = ((1.0 - gate) * z.abs()).mean(dim=-1, keepdim=True)
        uncertainty_logit = self.uncertainty(low_obs_mass).squeeze(-1)
        return {
            "z": z,
            "z_route": z_route,
            "logits": logits,
            "uncertainty_logit": uncertainty_logit,
            **obs,
        }

    def training_loss(
        self,
        batch: dict,
        lambda_meas: float = 0.2,
        lambda_route: float = 0.05,
        lambda_order: float = 0.1,
        lambda_unc: float = 0.03,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]

        cls_loss = F.cross_entropy(out["logits"], labels)

        mu, sigma = self.measurement(out["z"], batch["event_time"], batch["event_var_id"])
        meas_loss = measurement_nll(batch["event_value"], mu, sigma, batch["event_mask"])

        # Penalize class-head reliance on latent coordinates with weak observability.
        final_linear = self.classifier[-1]
        head_strength = final_linear.weight.norm(dim=0)  # [H]
        low_obs = (1.0 - out["gate"]).detach()
        route_loss = (low_obs * out["z"].abs() * head_strength.view(1, -1)).mean()

        order_loss = probe_ordering_loss(
            out["witness_bank"],
            batch["probe_strength_order"],
        )

        # A weak deployment-calibration target: high low-observability mass should
        # increase uncertainty when the routed classifier is likely unreliable.
        with torch.no_grad():
            pred_wrong = (out["logits"].argmax(dim=-1) != labels).to(out["logits"].dtype)
            weak_target = torch.maximum(pred_wrong, batch.get("low_observability_target", pred_wrong))
        unc_loss = F.binary_cross_entropy_with_logits(out["uncertainty_logit"], weak_target)

        total = (
            cls_loss
            + lambda_meas * meas_loss
            + lambda_route * route_loss
            + lambda_order * order_loss
            + lambda_unc * unc_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "measurement_nll": meas_loss.detach(),
            "route_loss": route_loss.detach(),
            "probe_order_loss": order_loss.detach(),
            "uncertainty_loss": unc_loss.detach(),
            "mean_gate": out["gate"].mean().detach(),
        }


def probe_ordering_loss(witness_bank: torch.Tensor, order: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """Enforce monotone observability for annotated strong -> weak probe pairs.

    witness_bank: [B, R, H]
    order: [P, 2], each row is [strong_probe_idx, weak_probe_idx].
    """
    if order.numel() == 0:
        return witness_bank.new_zeros(())
    losses = []
    for strong_idx, weak_idx in order.tolist():
        strong = torch.log1p(witness_bank[:, strong_idx])
        weak = torch.log1p(witness_bank[:, weak_idx])
        losses.append(F.relu(weak - strong + margin).pow(2).mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def build_observability_probe_bank(
    event_time: torch.Tensor,
    event_var_id: torch.Tensor,
    event_mask: torch.Tensor,
    num_vars: int,
) -> dict:
    """Create policy probes used only for observability estimation."""

    bsz, num_events = event_time.shape
    device = event_time.device
    times, vars_, masks = [], [], []

    # Probe 0: factual observations.
    times.append(event_time)
    vars_.append(event_var_id)
    masks.append(event_mask)

    horizon = event_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = event_time / horizon

    # Probe 1: late-window sparse cadence.
    late_sparse = event_mask * ((time_norm < 0.66) | ((torch.arange(num_events, device=device)[None] % 2) == 0)).to(event_mask.dtype)
    times.append(event_time)
    vars_.append(event_var_id)
    masks.append(late_sparse)

    # Probe 2: variable budget, keeping at most two observations per variable.
    budget_mask = torch.zeros_like(event_mask)
    for var_idx in range(num_vars):
        is_var = ((event_var_id == var_idx) & (event_mask > 0)).to(event_mask.dtype)
        rank = is_var.cumsum(dim=1)
        budget_mask = torch.where((is_var > 0) & (rank <= 2), torch.ones_like(budget_mask), budget_mask)
    times.append(event_time)
    vars_.append(event_var_id)
    masks.append(budget_mask)

    # Probe 3: asynchrony by delaying odd variables without changing values.
    odd_var = (event_var_id % 2 == 1).to(event_time.dtype)
    mean_gap = torch.zeros_like(event_time)
    mean_gap[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
    mean_gap = mean_gap.sum(dim=1, keepdim=True) / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    delayed_time = event_time + 0.5 * mean_gap * odd_var
    times.append(delayed_time)
    vars_.append(event_var_id)
    masks.append(event_mask)

    # Factual should be at least as informative as the sparsified probes.
    order = torch.tensor([[0, 1], [0, 2]], device=device, dtype=torch.long)

    return {
        "probe_event_time_bank": torch.stack(times, dim=1),
        "probe_event_var_bank": torch.stack(vars_, dim=1),
        "probe_event_mask_bank": torch.stack(masks, dim=1),
        "probe_strength_order": order,
    }
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cadence recoverability shift`：训练环境覆盖关键窗口，测试环境稀疏或延迟覆盖。
   - `variable-budget shift`：某些变量组在测试策略中只有少量可见观测。
   - `panel-to-asynchrony shift`：训练环境同步 panel，测试环境拆散或延迟返回。
   - `heteroskedastic sensor shift`：同一观测事件存在不同测量噪声，模拟 StarEmbed 式异方差观测。

2. **对比方法**
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - MILM-style value-redacted leakage probe。
   - Rethinking LLMs 中强调的 mTAND / irregular-aware encoder baseline。
   - DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF 等历史方案。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - mean observability gate：预测依赖的状态坐标是否被观测支持。
   - low-observability reliance score：错误高置信样本是否依赖低可观测坐标。
   - probe-order violation rate：弱采样 probe 是否反常产生更强可观测性。
   - abstention-aware accuracy：当低可观测依赖过高时拒识后的有效精度。

4. **消融实验**
   - 去掉 `L_low_observability_reliance`，验证分类头是否重新利用不可观测坐标。
   - 去掉 measurement decoder，只用普通 encoder gate，验证 witness 是否需要观测算子语义。
   - 将 probe bank 替换为随机 mask，验证收益来自结构化采样策略 probe。
   - 只用 factual witness，不用反事实低分位 witness，验证跨策略可观测性的重要性。
   - 扫描 leak 系数 `rho`，验证不可观测坐标作为弱上下文和主证据之间的 trade-off。

## 5. 预期创新性

1. **从采样去偏转向状态可观测性路由**：不问采样策略概率，也不删除采样信息；只问分类所用状态坐标是否由当前及反事实观测算子支持。
2. **从 token 可见性转向 latent coordinate identifiability**：吸收 MedSpaformer/StarEmbed 对可见信息差异的警示，但核心不是选 token，而是审计状态维度是否可辨识。
3. **从 LLM 后端转向 encoder 前端审计**：吸收 Rethinking LLMs 的结论，把采样鲁棒性前移到 irregular encoder 和 measurement operator，而不是寄希望于文本摘要或对齐策略。
4. **从反事实一致性转向反事实可观测性估计**：counterfactual intervention 不要求 logits/representation 一致，只提供观测算子族来估计哪些状态坐标在政策变化下仍可见。
5. **部署解释性直接**：模型能指出“这次预测依赖哪些可观测状态坐标，哪些坐标在当前采样策略下不可辨识”，比单纯输出 policy score 或 token attention 更接近实际可靠性审计。

## 6. 一句话投稿卖点

**OWSR 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“分类所需状态坐标在新观测策略下失去可观测性”的问题，并通过 measurement-Jacobian observability witness、反事实 probe bank 与 state-coordinate routing，让分类器优先依赖跨采样策略仍被观测值支持的潜在状态维度，而不是依赖训练采样政策偶然照亮的不可辨识坐标。**
