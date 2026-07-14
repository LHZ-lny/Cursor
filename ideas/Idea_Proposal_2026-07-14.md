# Title: Solver-Trace Front-Door Neutralizer：面向采样策略偏移的求解轨迹前门中和器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md`：其中记录了多轮自动化任务均未发现 `my_work_summary.md`，并列出了历史提案机制黑名单。
- 已读取近期论文记录：`paper_daily.md`、`paper_daily_2026-07-13.md`。
- 已读取当前工作区内全部历史提案：`Idea_Proposal_2026-06-12.md`、`2026-06-13.md`、`2026-06-14.md`、`2026-06-16.md`、`2026-06-19.md`、`2026-06-21.md`、`2026-06-22.md`、`2026-06-23.md`、`2026-06-25.md`、`2026-06-26.md`、`2026-07-12.md`、`2026-07-13.md`。
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
28. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing、MedMamba 的 frequency/adaptive graph branch、MedSpaformer 的普通 token sparsification、MILM 的 value-redacted classifier、StarEmbed 的直接 foundation embedding、LLMTS 的普通 LLM alignment 或 MVC-CDE 的普通多视图平滑路径作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗，不要求多采样视图表征一致，不做后验除法、随机平滑认证、停时鞅、拓扑、gauge、纠错码、knockoff、观测性门控、证据不确定性或信息格边际约束；而是把 Neural CDE / kernel-smoothed continuous-time encoder 在求解时产生的数值轨迹，例如 NFE、局部误差、步长收缩、路径粗糙度和带宽选择，看作 sampling-policy shortcut 的可观测因果中介。分类器通过前门式 trace standardization 在统一的求解轨迹分布下做决策，从而避免把“某个采样政策更难求解/更粗糙/更容易触发 solver 小步长”误当成类别证据。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 `paper_daily_2026-07-13.md` 中的 **Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing (MVC-CDE)** 给了一个此前历史提案没有充分利用的信号：不规则采样不仅改变输入 token、mask、图结构或路径几何，还会改变模型的 **数值求解过程**。

当离散观测被提升为 Neural CDE 的 control path 后，采样政策会沿着以下路径进入分类器：

- 某医院报警后密集复测，control path 局部高曲率，ODE solver 被迫缩小步长，NFE 升高；
- 某设备低电量时稀疏采样，kernel smoothing 带宽更大，路径更平滑但状态细节更不确定；
- 某变量 panel 同步出现，multi-view smoothing 中某些带宽视图更容易获得低误差；
- 某些类别在训练环境中天然被更高频测量，模型可能把“高 NFE / 高 roughness / 特定 step histogram”作为类别捷径。

历史方案大多把采样偏移放在输入、表示、后验、图、token、证据、可观测性或不确定性层面处理；但 MVC-CDE 暴露了一个新的中介层：

```text
sampling policy -> control path roughness -> solver trace -> hidden trajectory/logits
```

其中 `solver trace` 包括 NFE、accepted/rejected step count、local truncation error、adaptive step-size histogram、kernel bandwidth usage、path curvature 与 roughness score。真实急性状态变化也可能导致路径粗糙和 solver 变慢；问题在于训练环境中 policy-induced roughness 也会制造同样的数值轨迹。若分类器直接利用事实求解轨迹，它可能完全不需要显式读取 mask，就能从 NFE 和 step histogram 中读出训练医院协议。

**Solver-Trace Front-Door Neutralizer (ST-FDN)** 的核心目标是：

> 允许连续时间 encoder 利用 kernel smoothing 和 adaptive solver 的表达力，但最终分类必须在一个被标准化的 solver-trace 分布下完成；模型可以知道当前样本是否难求解，用于诊断和校准，但不能让 policy-specific NFE / roughness trace 直接决定类别边界。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通 MVC-CDE 改为 Trace-Instrumented Kernel CDE

ST-FDN 不把 MVC-CDE 的 multi-view smoothing 当作最终创新，而是把它升级为可观测因果中介提取器。

1. **Kernel Control Path Builder**
   - 输入不规则事件 `(value, time, variable, mask, measurement_std)`。
   - 产生若干 kernel-smoothed control path hidden states。
   - 与 MVC-CDE 一样，避免强行穿过每个噪声点；但本提案不把多视图 attention 或平滑路径本身作为主鲁棒机制。

2. **Adaptive Solver Trace Recorder**
   - 每次 CDE forward 记录数值轨迹：`normalized_nfe`、`mean_step_size`、`min_step_size`、`local_error_mean`、`local_error_max`、`roughness_score`、`bandwidth_usage`。
   - 若实际 solver 不方便返回所有内部量，可用 differentiable proxy：路径一阶/二阶差分、kernel residual、带宽注意力熵、分段曲率近似。

3. **State Encoder 与 Trace Mediator 分离**
   - `h_state`：连续状态表示，只来自 value-driven CDE hidden trajectory。
   - `m_trace`：求解轨迹中介，只描述“这条路径如何被数值求解”，不直接进入最终分类头。

### 2.2 改 Classifier：Trace-Standardized Front-Door Readout

普通 CDE 分类器直接计算：

```text
logits = Classifier(h_state)
```

ST-FDN 改成：

```text
logits_tau = Classifier(FiLM(h_state, tau))
logits_fd  = mean_{tau in TraceReferenceBank} logits_tau
```

其中 `tau` 是从 reference trace bank 采样的标准化求解轨迹码，而不是当前样本事实 trace。直觉是：事实 trace 可以被记录、编码、诊断；但最终预测要回答的是 **如果这条状态表示在一组标准化 solver trace 下被读取，类别是什么**。因此，采样政策通过 `solver trace` 影响预测的路径被中和。

这是一种前门式中和：

```text
sampling policy -> solver trace -> logits
```

ST-FDN 不尝试估计 `p(policy)` 或 `p(sampled | state)`，也不除以采样似然；它只在一个可观测中介 `solver trace` 上执行 `do(trace = tau_ref)` 的标准化读出。

### 2.3 改 Loss：从不变性/对抗转向 Solver-Trace Mediation Control

总目标：

```text
L = L_frontdoor_cls
  + lambda_trace * L_trace_reconstruction
  + lambda_slope * L_trace_do_slope
  + lambda_mix   * L_state_trace_separation
```

#### A. Front-Door Classification Loss `L_frontdoor_cls`

只用 trace-standardized logits 做分类：

```text
L_frontdoor_cls = CE(mean_tau Classifier(FiLM(h_state, tau_ref)), y)
```

它不是 randomized smoothing over policies，因为 `tau_ref` 不是采样政策扰动，而是数值求解轨迹中介的参考分布；也不输出 certified radius。

#### B. Trace Reconstruction Loss `L_trace_reconstruction`

为了让 trace mediator 真实刻画求解过程，训练一个 `TraceEncoder` 从路径 proxy 中重构 observed solver trace：

```text
L_trace_reconstruction = SmoothL1(trace_hat, stopgrad(trace_observed))
```

这项只保证中介可测，不把 trace 当作类别证据。

#### C. Trace Do-Slope Penalty `L_trace_do_slope`

对同一个 `h_state` 注入多组 reference trace code，计算真实类 margin 对 trace effort 的线性斜率：

```text
margin_tau = logit_y(h_state, tau) - max_{k != y} logit_k(h_state, tau)
slope = Cov(margin_tau, effort_tau) / Var(effort_tau)
L_trace_do_slope = slope^2
```

它惩罚“保持状态表示不变时，真实类边际随 NFE / roughness 单调变大或变小”。这与历史的一致性损失不同：它不要求所有 `logits_tau` 相同，不比较不同采样 view，只阻断类别 margin 对 **数值求解 effort** 的一阶依赖。

#### D. State-Trace Separation Loss `L_state_trace_separation`

为了避免 `h_state` 偷偷编码完整 trace，使用一个非对抗的协方差分离项：

```text
L_state_trace_separation = || Corr(StopGrad(trace_code), state_projection) ||_F^2
```

它不是 environment adversarial，也不是 policy probe。它只限制 continuous state embedding 与数值求解中介过度耦合，防止 classifier 在 `h_state` 里重新读出 NFE shortcut。

### 2.4 改 Dataloader / Collator：返回 Trace Intervention Bank

新增 `SolverTraceInterventionCollator`。每个 batch 返回：

1. 原始事件：`event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_std`。
2. trace recipes：
   - `bandwidth_narrow`：较窄 kernel，保留局部粗糙；
   - `bandwidth_wide`：较宽 kernel，降低路径粗糙；
   - `tolerance_loose`：更粗 solver tolerance；
   - `tolerance_strict`：更细 solver tolerance；
   - `curvature_clip`：对局部高曲率片段做可微预条件。
3. reference trace bank：
   - 来自 batch 内不同样本、不同 trace recipes 的 solver trace code；
   - 也可以维护一个 EMA queue，作为训练期间的标准 trace 分布；
   - reference bank 不包含 label、policy id 或 environment id。
4. diagnostics：`observed_trace`、`roughness_score`、`nfe_proxy`、`bandwidth_usage`。

注意：collator 不返回对比正样本，不要求多个 recipe 的预测一致；它只提供中介 `solver trace` 的可干预读出条件。

### 2.5 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `TraceInstrumentedKernelCDE`，输出 `h_state` 与 `trace_observed`。
- 现有 sampling branch 改为 `TraceRecipeGenerator`，只定义 smoothing bandwidth、solver tolerance 和 path preconditioner 的干预 recipe。
- 现有 counterfactual intervention 改为 `TraceInterventionBank`，用于生成 reference trace code，而不是生成一致性 view、risk variance view、policy simplex samples、knockoff calendar 或 lattice meet/join。
- 推理阶段：
  - 快速模式：使用训练集 EMA reference trace bank 的少量 trace code 平均预测；
  - 诊断模式：同时输出 factual trace reliance、NFE-label slope、roughness-mediated gap；
  - 部署告警：若 factual logits 与 front-door logits 差距过大，提示模型面对了 policy-induced numerical shortcut。

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


class KernelPathTraceProxy(nn.Module):
    """Build a smoothed event representation and differentiable solver-trace proxies."""

    def __init__(self, num_vars: int, hidden_dim: int, num_bandwidths: int = 4):
        super().__init__()
        self.num_bandwidths = num_bandwidths
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bandwidth_logits = nn.Parameter(torch.zeros(num_bandwidths))
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.state_mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, batch: dict, recipe: torch.Tensor | None = None) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id.clamp_min(0))
        event_x = torch.cat(
            [var_h, value.unsqueeze(-1), torch.log1p(delta_t).unsqueeze(-1), torch.log1p(meas_std).unsqueeze(-1)],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * event_mask.unsqueeze(-1)

        if recipe is None:
            recipe = torch.zeros(value.size(0), self.num_bandwidths, device=value.device, dtype=value.dtype)
        bandwidth_weight = torch.softmax(self.bandwidth_logits[None] + recipe, dim=-1)

        first_diff = torch.zeros_like(event_h)
        first_diff[:, 1:] = event_h[:, 1:] - event_h[:, :-1]
        second_diff = torch.zeros_like(event_h)
        second_diff[:, 2:] = first_diff[:, 2:] - first_diff[:, 1:-1]

        roughness = masked_mean(second_diff.pow(2).sum(dim=-1), event_mask, dim=1)
        gap_mean = masked_mean(torch.log1p(delta_t), event_mask, dim=1)
        gap_min = (delta_t + (1.0 - event_mask) * 1e6).amin(dim=1).clamp_max(10.0)
        quality_noise = masked_mean(torch.log1p(meas_std), event_mask, dim=1)
        bandwidth_entropy = -(bandwidth_weight * bandwidth_weight.clamp_min(1e-8).log()).sum(dim=-1)
        bandwidth_effort = (bandwidth_weight * torch.linspace(1.0, 0.25, self.num_bandwidths, device=value.device)).sum(dim=-1)
        nfe_proxy = roughness.sqrt() * (1.0 + bandwidth_effort) + 0.1 * quality_noise

        trace_observed = torch.stack(
            [nfe_proxy, roughness, gap_mean, gap_min, quality_noise, bandwidth_entropy, bandwidth_effort],
            dim=-1,
        )
        seq_h, _ = self.state_mixer(event_h)
        state = masked_mean(seq_h, event_mask, dim=1)
        return {"state": state, "trace_observed": trace_observed, "bandwidth_weight": bandwidth_weight}


class TraceMediator(nn.Module):
    def __init__(self, trace_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(trace_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, trace_dim))

    def forward(self, trace: torch.Tensor) -> dict:
        code = self.encoder(trace)
        return {"trace_code": code, "trace_recon": self.decoder(code)}


class TraceFiLMClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.trace_to_film = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, state: torch.Tensor, trace_code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.trace_to_film(trace_code).chunk(2, dim=-1)
        conditioned = state * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        return self.classifier(conditioned)


class SolverTraceFrontDoorNeutralizer(nn.Module):
    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int, trace_dim: int = 7, reference_samples: int = 8):
        super().__init__()
        self.encoder = KernelPathTraceProxy(num_vars, hidden_dim)
        self.trace = TraceMediator(trace_dim, hidden_dim)
        self.classifier = TraceFiLMClassifier(hidden_dim, num_classes)
        self.state_trace_probe = nn.Linear(hidden_dim, hidden_dim)
        self.reference_samples = reference_samples

    def reference_codes(self, trace_code: torch.Tensor, batch: dict) -> torch.Tensor:
        if "reference_trace_code" in batch:
            ref = batch["reference_trace_code"]
            return ref[:, : self.reference_samples]
        return torch.stack([trace_code.roll(shifts=i + 1, dims=0) for i in range(self.reference_samples)], dim=1)

    def frontdoor_logits(self, state: torch.Tensor, ref_code: torch.Tensor) -> torch.Tensor:
        logits = [self.classifier(state, ref_code[:, i]) for i in range(ref_code.size(1))]
        return torch.stack(logits, dim=1).mean(dim=1)

    def forward(self, batch: dict) -> dict:
        enc = self.encoder(batch)
        trace = self.trace(enc["trace_observed"])
        ref = self.reference_codes(trace["trace_code"].detach(), batch)
        return {
            **enc,
            **trace,
            "reference_trace_code": ref,
            "logits": self.frontdoor_logits(enc["state"], ref),
            "logits_factual_trace": self.classifier(enc["state"], trace["trace_code"]),
        }

    def trace_do_slope_loss(self, state: torch.Tensor, ref_code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        effort = ref_code.norm(dim=-1)
        margins = []
        for idx in range(ref_code.size(1)):
            margins.append(true_margin(self.classifier(state, ref_code[:, idx]), labels))
        margin = torch.stack(margins, dim=1)
        effort_c = effort - effort.mean(dim=1, keepdim=True)
        margin_c = margin - margin.mean(dim=1, keepdim=True)
        slope = (effort_c * margin_c).mean(dim=1) / effort_c.pow(2).mean(dim=1).clamp_min(1e-6)
        return slope.pow(2).mean()

    def state_trace_separation_loss(self, state: torch.Tensor, trace_code: torch.Tensor) -> torch.Tensor:
        state_z = self.state_trace_probe(state)
        state_z = (state_z - state_z.mean(dim=0)) / state_z.std(dim=0).clamp_min(1e-6)
        trace_z = (trace_code.detach() - trace_code.detach().mean(dim=0)) / trace_code.detach().std(dim=0).clamp_min(1e-6)
        return ((state_z.T @ trace_z) / max(state.size(0) - 1, 1)).pow(2).mean()

    def training_loss(self, batch: dict, lambda_trace: float = 0.15, lambda_slope: float = 0.20, lambda_mix: float = 0.05) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]
        cls_loss = F.cross_entropy(out["logits"], labels)
        trace_loss = F.smooth_l1_loss(out["trace_recon"], out["trace_observed"].detach())
        slope_loss = self.trace_do_slope_loss(out["state"], out["reference_trace_code"], labels)
        sep_loss = self.state_trace_separation_loss(out["state"], out["trace_code"])
        total = cls_loss + lambda_trace * trace_loss + lambda_slope * slope_loss + lambda_mix * sep_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "trace_reconstruction_loss": trace_loss.detach(),
            "trace_do_slope_loss": slope_loss.detach(),
            "state_trace_separation_loss": sep_loss.detach(),
        }


@torch.no_grad()
def build_trace_recipe_bank(batch_size: int, num_bandwidths: int, device: torch.device) -> torch.Tensor:
    """Create bandwidth/tolerance recipes for solver-trace interventions."""
    recipes = torch.zeros(batch_size, 4, num_bandwidths, device=device)
    recipes[:, 0, 0] = 1.5      # narrow bandwidth: rough path
    recipes[:, 1, -1] = 1.5     # wide bandwidth: smooth path
    recipes[:, 2] = -0.5        # loose numerical effort
    recipes[:, 3] = 0.5         # strict numerical effort
    return recipes
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `roughness shift`：训练环境中高风险样本被密集复测，测试环境改为稀疏定时采样。
   - `solver-effort shift`：训练和测试在相同底层轨迹上使用不同采样 cadence，使 NFE / step-size histogram 改变。
   - `bandwidth-preference shift`：训练环境适合窄带宽，测试环境噪声更大、宽带宽更优。
   - `panel-curvature shift`：同步 panel 造成局部路径急转，测试环境拆成异步观测。

2. **对比方法**
   - 普通 Neural CDE。
   - MVC-CDE / attentive kernel smoothing。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - NFE-label leakage：只用 solver trace 预测标签的能力。
   - trace-mediated gap：`CE(logits_factual_trace, y) - CE(logits_frontdoor, y)`。
   - trace do-slope：真实类 margin 对 reference trace effort 的斜率。
   - roughness-shift calibration error：高 roughness 样本是否被过度自信分类。

4. **消融实验**
   - 去掉 `L_trace_do_slope`，检查模型是否重新利用 NFE shortcut。
   - 去掉 front-door readout，直接用 factual trace FiLM 分类。
   - 将 reference trace bank 替换为同样本 factual trace，验证收益不是 FiLM 参数量带来的。
   - 用真实 solver callback 替换 proxy trace，检查数值轨迹信号是否更强。
   - 扫描 reference trace bank 大小，评估鲁棒性与推理开销。

## 5. 预期创新性

1. **从输入采样偏移转向数值求解中介偏移**：首次把 sampling-policy shortcut 定位到 Neural CDE 的 NFE、step-size、local error 和 roughness trace。
2. **从普通 path smoothing 转向 trace front-door standardization**：吸收 MVC-CDE 对 control path roughness 的启发，但不把多视图平滑作为鲁棒性本身，而是把求解轨迹作为要中和的因果中介。
3. **从政策去偏转向数值中介干预**：不估计采样概率、不做对抗、不做密度比、不做后验商；只对 `solver trace` 执行 `do(trace = reference)` 读出。
4. **保留真实高粗糙状态信号**：ST-FDN 不惩罚高 NFE 本身，避免把急性状态变化误删；它只惩罚真实类边际对 trace effort 的直接斜率依赖。
5. **部署诊断清晰**：当事实 trace 分类与前门标准化分类差距大时，可直接报告“预测依赖数值求解轨迹，可能受采样政策影响”。

## 6. 一句话投稿卖点

**ST-FDN 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样政策通过 control-path roughness 改变 Neural CDE 求解轨迹，并由 solver trace 中介污染分类边界”的问题，通过 trace-instrumented kernel CDE、reference trace bank 与 front-door trace-standardized readout，让模型在不依赖危险率、对抗、一致性、后验商、随机平滑、拓扑、gauge、纠错码、knockoff、观测性门控、evidential shield 或信息格约束的前提下，切断 NFE/roughness/step-size 等数值求解捷径。**
