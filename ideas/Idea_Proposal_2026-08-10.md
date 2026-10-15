# Title: Do-Bitemporal Causal Curtain：面向采样策略偏移的双时轴因果幕帘分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取 `paper_daily.md`，并确认仓库存在 dated paper daily：`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`。
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
- 已读取自动化记忆 `MEMORIES.md`，纳入其中记录但当前工作区未完全落盘的历史机制摘要。

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
11. reconstruction error cartography、ANOVA projection、VQ semantic/acquisition clauses、HSIC redaction。
12. policy-simplex randomized smoothing、certified policy radius、logit-normal/Dirichlet do-sampler。
13. 采样测度 density ratio、doubly robust target-measure correction、influence-bound。
14. optional-stopping martingale queries、standardized innovation、stopping recipe moment control。
15. soft excursion topology、censored persistence interval、censor envelope、fragmentation sobriety。
16. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
17. policy-only negative film、shadow eraser/stencil、shadow nullification。
18. latent packet codeword、parity-check、syndrome locator、packet repair decoder。
19. conditional knockoff calendar、soft knockoff-FDR、swap symmetry。
20. observability witness、counterfactual observability probe、observability-routed classification。
21. subjective-logic / Dirichlet evidential shield、policy-induced vacuity、class-wise evidence discount。
22. observation-set policy lattice、meet/join masks、monotone/submodular margin、shortcut curvature。
23. solver-trace front-door、NFE/roughness trace mediator、reference trace bank。
24. measurement-action bisimulation、policy-word signature renormalization、thermodynamic free-energy、Sklar copula stripping、triage queue debt、Sinkhorn detail canonicalization、MDL episode transducer、causal sheaf gluing、trigger hysteresis、control barrier certificates、regret escrow、principal-stratum status compiler。
25. counterfactual conformal risk sleeves、counterfactual sampling instruments、Borda / majority rank tribunal、Krylov policy subspace、determinantal / Nystrom volume basis、tropical support routes。
26. fixed clinical viva question bank、pathology sequent proof bank、disease-progress poset clock、pathology feasible hull、IRT latent trait / DIF firewall。
27. 反事实观测分辨率 RG fixed point、irrelevant operator decay、scale beta function。

本提案选择新的正交切入点：**不把鲁棒性建立在后验商、图拆分、校准集合、证明、测量项、偏序、拓扑、纠错码、knockoff、陪审团或重整化尺度流上，而是把采样策略偏移表述为“临床/传感事件发生时间”和“该事件数值对模型可得的记录时间”之间的错位。分类器只能在部署时刻的 record-time 因果幕帘内决策；回填、pending、延迟入库、跨中心记录速度差异只允许进入 latency sidecar 与诊断，不允许把尚未可得的未来观测伪装成类别证据。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 最新记录中，最值得与当前“采样解耦/反事实干预”框架结合的是 **PULSE** 与 **TCF**。

**PULSE** 把 ICU time-series classification 放进 HiRID / MIMIC-IV / eICU 等跨中心设置，提示真实 sampling-policy shift 不只是“少测几个点”，还包括不同中心的数据可得性流程：

- 某中心化验下单后很快返回，另一中心存在长时间 value-pending；
- 某中心离线整理时会把晚到化验回填到较早 event time，另一中心按 record arrival time 保留；
- 某中心床旁设备实时入库，另一中心批量上传导致事件发生时间和记录时间错位；
- 某中心在病情恶化后密集记录，另一中心事后补录或摘要化。

这些差异会制造一种危险的 **retrocausal shortcut**：训练时离线表格中某个化验被标在早期 event time，但在真实部署时该值直到预测之后才返回。模型若按 event time 排序并读取它，就等价于看见了未来记录流程。

**TCF** 的 Pathology-Focused Binning、Dual-Calendar RoPE 与 Time-Conditioned Foreseeing 强调 EHR foundation model 必须理解“何时发生”和“未来哪个时间点会出现什么病理事件”。但 TCF 的 future-event 目标仍可能把 patient state 与 care process 混在一起：未来会不会有某项观测、何时入库、何时可用，本身就是医院记录政策的一部分。

因此，本轮提出 **Do-Bitemporal Causal Curtain (DBCC)**：

> 每个事件同时拥有 `event_time` 与 `record_time`。`event_time` 描述病理或传感状态发生的临床时间；`record_time` 描述模型在部署时何时真正能看到该值。采样策略偏移首先改变的是两者之间的 latency / pending / backfill 分布。DBCC 用 record-time 因果幕帘隔离“已发生但尚不可得”的观测，让分类器只读取 as-of 可得的病理状态；记录延迟由 sidecar 解释和预测，但不进入类别证据。

这个角度能解决采样偏移中的三个难点：

1. **跨中心记录延迟偏移**：同一病程在医院 A 中实时可得，在医院 B 中延迟可得；DBCC 在训练时模拟 record-time 幕帘，避免模型依赖未来才返回的值。
2. **value-pending 与回填污染**：化验已下单但 value 未返回时，普通模型可能把“下单事件”或离线回填值当成类别证据；DBCC 将 pending/latency 放到 sidecar，分类头只能用幕帘内 value。
3. **TCF future-event 混合问题**：DBCC 保留 pathology-focused foreseeing，但把 future pathology 与 future record availability 分成两个头；分类用前者，部署诊断用后者。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：从单时轴事件流改为 Bitemporal As-Of Batch

每个观测事件被表示为：

```text
e_i = (value_i, variable_i, event_time_i, record_time_i, value_available_i, pending_i)
```

新增 `BitemporalCurtainCollator`，在事实数据之外生成若干 record-time policy recipes：

1. `fast_return`：化验/传感值几乎实时可得。
2. `slow_lab`：部分变量存在长延迟返回。
3. `batch_upload`：多个事件在同一 record_time 批量入库。
4. `backfill_after_cutoff`：event_time 早于预测时刻，但 record_time 晚于预测时刻。
5. `value_pending`：记录显示已施测，但 value 仍不可得。

每个 batch 返回若干 `as_of_cutoff`。对 cutoff `c`，只有满足：

```text
record_time_i <= c
value_available_i = 1
```

的事件能进入分类主路径。`event_time_i <= c` 但 `record_time_i > c` 的事件属于“已发生但尚不可得”，只能用于 latency sidecar 监督，不能被分类器读取。

### 2.2 改 Encoder：Bitemporal Event Encoder + Causal Curtain Pooler

DBCC 可以包裹 STAR-Set、TCF-style Transformer、EHR event Transformer 或轻量 GRU，但前端必须区分两类时间：

1. **Pathology Event Atomizer**
   - 吸收 TCF 的 Pathology-Focused Binning，将 `(value_i, variable_i)` 转为病理 bin distribution。
   - `event_time_i` 用于表达病理发生位置。
   - `record_time_i - event_time_i` 用于 sidecar 记录延迟，不直接进入分类 logits。

2. **Record-Time Causal Curtain**
   - 对每个 `as_of_cutoff` 生成幕帘 mask：

```text
curtain_i(c) = 1[record_time_i <= c and value_available_i = 1]
```

   - 主 encoder 只聚合 `curtain_i(c)=1` 的 value atoms。
   - pending / delayed / backfilled tokens 只进入 `LatencySidecar`。

3. **As-Of Classifier**
   - 分类头读取 `h_asof(c)`：

```text
h_asof(c) = Pooler({atom_i : curtain_i(c)=1})
logits_c = Classifier(h_asof(c))
```

   - 训练时随机抽取多个 cutoff，让模型习惯在不同 record availability 下做当前时刻预测，而不是离线读取全量回填数据。

### 2.3 改 Loss：从一致性/证明/校准转向 Anti-Retrocausal Discipline

总目标：

```text
L = L_asof_cls
  + lambda_curtain * L_retrocausal_curtain
  + lambda_lat     * L_latency_sidecar
  + lambda_fore    * L_bitemporal_foreseeing
  + lambda_sep     * L_state_latency_separation
```

#### A. As-Of Classification `L_asof_cls`

只使用 record-time 幕帘内可得信息：

```text
L_asof_cls = CE(Classifier(h_asof(c)), y)
```

若样本在 cutoff `c` 时证据不足，模型可以低置信，但不能从 `record_time > c` 的回填值中偷取未来证据。

#### B. Retrocausal Curtain Loss `L_retrocausal_curtain`

训练时额外计算一个 `h_backfilled(c)`，它包含 `event_time <= c` 但 `record_time > c` 的离线回填 token。若回填视图让真实类 margin 相比 as-of 视图获得过大增益，说明模型可能依赖未来才可得的记录流程：

```text
margin_asof = logit_y(h_asof) - max_{k != y} logit_k(h_asof)
margin_back = logit_y(h_backfilled) - max_{k != y} logit_k(h_backfilled)
L_retrocausal_curtain = relu(margin_back - margin_asof - allowed_gain)^2
```

`allowed_gain` 只由幕帘内 value quality 估计，不能由未来 pending token 决定。注意这不是多采样视图一致性：DBCC 不要求 as-of 与 backfilled logits 相同；它只限制“未来记录回填”不能成为过大的类别捷径。

#### C. Latency Sidecar `L_latency_sidecar`

记录延迟必须被解释，但不能用于分类：

```text
delay_i = record_time_i - event_time_i
lat_hat_i = LatencySidecar(sampling_coordinates_i)
L_latency_sidecar = SmoothL1(lat_hat_i, delay_i) + BCE(available_hat_i, value_available_i)
```

这让模型显式报告跨中心记录流程差异：哪些变量延迟长、哪些事件会 pending、哪些中心容易 batch upload。

#### D. Bitemporal Foreseeing `L_bitemporal_foreseeing`

吸收 TCF 的 future-time conditioned foreseeing，但拆成两个头：

```text
p_path(bin, var | h_asof, future_event_time)
p_record(delay_bucket, available | h_asof, future_event_time)
```

训练目标：

```text
L_bitemporal_foreseeing =
  CE(p_path, future_pathology_bin)
  + beta * CE(p_record, future_record_availability)
```

分类头只共享 `p_path` 的状态表征；`p_record` 的隐藏层不接入分类器，只用于部署诊断和采样流程建模。

#### E. State-Latency Separation `L_state_latency_separation`

防止 `h_asof` 偷偷编码中心记录速度，用非对抗交叉协方差：

```text
L_state_latency_separation =
  || Corr(h_asof, stopgrad(latency_summary)) ||_F^2
```

它不是 environment adversarial classifier，也不预测策略标签；它只限制分类状态表示与记录延迟摘要的线性耦合。

### 2.4 与 PULSE / TCF 的结合方式

- **来自 PULSE**：HiRID / MIMIC-IV / eICU 的跨中心差异不仅作为 dataset shift，还被拆成 record latency、batch upload、pending ratio、backfill rate 等 bitemporal diagnostics。
- **来自 TCF**：保留 Pathology-Focused Binning 与 future-time query，但将 future pathology 与 future record availability 分离，避免 future observation process 污染分类状态。
- **与采样解耦/反事实干预框架结合**：
  - value process 产生 pathology atoms；
  - sampling process 产生 record-time policy recipes 与 latency sidecar targets；
  - counterfactual intervention 生成可解释的 record-time delays / pending / backfill；
  - classifier 只读取 causal curtain 内的 as-of pathology state。

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
    rival = logits.masked_fill(F.one_hot(labels, logits.size(-1)).bool(), -1e4)
    return true_logit - rival.max(dim=-1).values


class BitemporalPathologyAtomizer(nn.Module):
    """Lift bitemporal events into pathology atoms while retaining latency metadata."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.5, 2.5, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.atom_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_bins + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        event_time = batch["event_time"]
        record_time = batch["record_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        event_mask = batch["event_mask"]
        available = batch.get("value_available", event_mask)

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * event_mask.unsqueeze(-1) * available.unsqueeze(-1)

        horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        event_norm = event_time / horizon
        record_norm = record_time / horizon
        latency = (record_time - event_time).clamp_min(0.0)
        latency_norm = torch.log1p(latency) / torch.log1p(horizon)
        pending = (1.0 - available).clamp(0.0, 1.0)

        atom_x = torch.cat(
            [
                self.var_embed(var_id),
                bin_prob,
                value.unsqueeze(-1) * available.unsqueeze(-1),
                event_norm.unsqueeze(-1),
                record_norm.unsqueeze(-1),
                latency_norm.unsqueeze(-1) + pending.unsqueeze(-1),
            ],
            dim=-1,
        )
        atom_h = self.atom_proj(atom_x) * event_mask.unsqueeze(-1)
        return {
            "atom_h": atom_h,
            "bin_prob": bin_prob,
            "pathology_bin": bin_prob.argmax(dim=-1),
            "latency": latency,
            "event_mask": event_mask,
            "available": available,
        }


class CausalCurtainPooler(nn.Module):
    """Pool only events that are available by a record-time cutoff."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.event_context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.pool_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, atom_h: torch.Tensor, curtain_mask: torch.Tensor) -> torch.Tensor:
        ctx, _ = self.event_context(atom_h * curtain_mask.unsqueeze(-1))
        pooled = masked_mean(ctx, curtain_mask, dim=1)
        return self.pool_proj(pooled)


class LatencySidecar(nn.Module):
    """Predict record latency from sampling coordinates only; not used by classifier."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delay_head = nn.Linear(hidden_dim, 1)
        self.available_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        event_time = batch["event_time"]
        record_time = batch["record_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]

        horizon = (event_time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        event_norm = event_time / horizon
        record_norm = record_time / horizon
        delay = (record_time - event_time).clamp_min(0.0)

        var_onehot = F.one_hot(var_id, self.num_vars).to(event_time.dtype) * mask.unsqueeze(-1)
        var_rate = var_onehot.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        stats = torch.stack(
            [
                mask.mean(dim=1),
                masked_mean(event_norm, mask, dim=1),
                masked_mean(record_norm, mask, dim=1),
                masked_mean(torch.log1p(delay), mask, dim=1),
                (record_norm - event_norm).clamp_min(0.0).amax(dim=1),
                batch.get("value_available", mask).mean(dim=1),
            ],
            dim=-1,
        )
        h = self.net(torch.cat([var_rate, stats], dim=-1))
        return {
            "latency_summary": h,
            "delay_pred": F.softplus(self.delay_head(h)).squeeze(-1),
            "available_logit": self.available_head(h).squeeze(-1),
        }


class BitemporalForeseeHead(nn.Module):
    """Separate future pathology from future record availability."""

    def __init__(self, hidden_dim: int, num_vars: int, num_bins: int, num_delay_bins: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.num_delay_bins = num_delay_bins
        self.query = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.var_head = nn.Linear(hidden_dim, num_vars)
        self.bin_head = nn.Linear(hidden_dim, num_bins)
        self.delay_head = nn.Linear(hidden_dim, num_delay_bins)
        self.avail_head = nn.Linear(hidden_dim, 1)

    def forward(self, h_asof: torch.Tensor, query_event_time: torch.Tensor) -> dict:
        h = h_asof[:, None].expand(-1, query_event_time.size(1), -1)
        q = self.query(torch.cat([h, query_event_time.unsqueeze(-1)], dim=-1))
        return {
            "var_logits": self.var_head(q),
            "bin_logits": self.bin_head(q),
            "delay_logits": self.delay_head(q),
            "available_logit": self.avail_head(q).squeeze(-1),
        }


class DoBitemporalCausalCurtain(nn.Module):
    """Sampling-policy robust classifier with event-time / record-time separation."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        num_delay_bins: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.atomizer = BitemporalPathologyAtomizer(num_vars, num_bins, hidden_dim)
        self.pooler = CausalCurtainPooler(hidden_dim)
        self.sidecar = LatencySidecar(num_vars, hidden_dim)
        self.foresee = BitemporalForeseeHead(hidden_dim, num_vars, num_bins, num_delay_bins)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.state_probe = nn.Linear(hidden_dim, hidden_dim)
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.num_delay_bins = num_delay_bins

    def curtain_mask(self, batch: dict, cutoff: torch.Tensor, mode: str = "asof") -> torch.Tensor:
        mask = batch["event_mask"]
        available = batch.get("value_available", mask)
        if mode == "asof":
            return mask * available * (batch["record_time"] <= cutoff[:, None]).to(mask.dtype)
        if mode == "backfilled":
            return mask * available * (batch["event_time"] <= cutoff[:, None]).to(mask.dtype)
        raise ValueError(f"unknown curtain mode: {mode}")

    def encode_asof(self, batch: dict, cutoff: torch.Tensor, mode: str = "asof") -> dict:
        atom = self.atomizer(batch)
        curtain = self.curtain_mask(batch, cutoff, mode=mode)
        h_asof = self.pooler(atom["atom_h"], curtain)
        logits = self.classifier(h_asof)
        return {**atom, "curtain": curtain, "h_asof": h_asof, "logits": logits}

    def latency_sidecar_loss(self, batch: dict, sidecar: dict) -> torch.Tensor:
        mask = batch["event_mask"]
        delay = (batch["record_time"] - batch["event_time"]).clamp_min(0.0)
        target_delay = masked_mean(torch.log1p(delay), mask, dim=1)
        delay_loss = F.smooth_l1_loss(sidecar["delay_pred"], target_delay.detach())

        available = batch.get("value_available", mask)
        target_avail = available.mean(dim=1)
        avail_loss = F.binary_cross_entropy_with_logits(sidecar["available_logit"], target_avail)
        return delay_loss + avail_loss

    def retrocausal_curtain_loss(
        self,
        labels: torch.Tensor,
        asof_logits: torch.Tensor,
        backfilled_logits: torch.Tensor,
        allowed_gain: torch.Tensor,
    ) -> torch.Tensor:
        margin_asof = true_margin(asof_logits, labels)
        margin_back = true_margin(backfilled_logits, labels)
        return F.relu(margin_back - margin_asof - allowed_gain.detach()).pow(2).mean()

    def bitemporal_foreseeing_loss(self, batch: dict, h_asof: torch.Tensor) -> torch.Tensor:
        if "query_event_time" not in batch:
            return torch.zeros((), device=h_asof.device)
        pred = self.foresee(h_asof, batch["query_event_time"])
        target_var = batch["query_target_var"].clamp(0, self.num_vars - 1)
        target_bin = batch["query_target_bin"].clamp(0, self.num_bins - 1)
        target_delay = batch["query_target_delay_bin"].clamp(0, self.num_delay_bins - 1)
        target_avail = batch["query_target_available"].to(h_asof.dtype)

        var_loss = F.cross_entropy(pred["var_logits"].flatten(0, 1), target_var.flatten())
        bin_loss = F.cross_entropy(pred["bin_logits"].flatten(0, 1), target_bin.flatten())
        delay_loss = F.cross_entropy(pred["delay_logits"].flatten(0, 1), target_delay.flatten())
        avail_loss = F.binary_cross_entropy_with_logits(pred["available_logit"], target_avail)
        return var_loss + bin_loss + 0.35 * (delay_loss + avail_loss)

    def state_latency_separation_loss(self, h_asof: torch.Tensor, latency_summary: torch.Tensor) -> torch.Tensor:
        state_z = self.state_probe(h_asof)
        state_z = (state_z - state_z.mean(dim=0)) / state_z.std(dim=0).clamp_min(1e-6)
        lat_z = latency_summary.detach()
        lat_z = (lat_z - lat_z.mean(dim=0)) / lat_z.std(dim=0).clamp_min(1e-6)
        corr = state_z.T @ lat_z / max(h_asof.size(0) - 1, 1)
        return corr.pow(2).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_curtain: float = 0.30,
        lambda_lat: float = 0.18,
        lambda_fore: float = 0.20,
        lambda_sep: float = 0.05,
    ) -> dict:
        labels = batch["labels"]
        cutoff = batch["as_of_cutoff"]

        asof = self.encode_asof(batch, cutoff, mode="asof")
        backfilled = self.encode_asof(batch, cutoff, mode="backfilled")
        sidecar = self.sidecar(batch)

        cls_loss = F.cross_entropy(asof["logits"], labels)

        # Small value-quality allowance: if as-of visible evidence is strong,
        # backfilled margins may improve slightly, but not because of future records.
        visible_rate = asof["curtain"].sum(dim=1) / batch["event_mask"].sum(dim=1).clamp_min(1.0)
        allowed_gain = 0.05 + 0.10 * visible_rate
        curtain_loss = self.retrocausal_curtain_loss(
            labels=labels,
            asof_logits=asof["logits"],
            backfilled_logits=backfilled["logits"],
            allowed_gain=allowed_gain,
        )

        lat_loss = self.latency_sidecar_loss(batch, sidecar)
        fore_loss = self.bitemporal_foreseeing_loss(batch, asof["h_asof"])
        sep_loss = self.state_latency_separation_loss(asof["h_asof"], sidecar["latency_summary"])

        total = (
            cls_loss
            + lambda_curtain * curtain_loss
            + lambda_lat * lat_loss
            + lambda_fore * fore_loss
            + lambda_sep * sep_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "retrocausal_curtain_loss": curtain_loss.detach(),
            "latency_sidecar_loss": lat_loss.detach(),
            "bitemporal_foreseeing_loss": fore_loss.detach(),
            "state_latency_separation_loss": sep_loss.detach(),
            "visible_rate": visible_rate.mean().detach(),
        }


@torch.no_grad()
def build_bitemporal_record_policy_batch(batch: dict, num_vars: int) -> dict:
    """Create record-time policies for DBCC training.

    The generated views alter record_time and value availability. They are not
    contrastive positives and are not used for logits consistency.
    """

    event_time = batch["event_time"]
    event_var_id = batch["event_var_id"]
    event_mask = batch["event_mask"]
    value = batch["event_value"]
    bsz, num_events = event_time.shape
    device = event_time.device

    horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = event_time / horizon
    var_is_lab = (event_var_id % 2 == 1).to(event_time.dtype)
    late = (time_norm > 0.66).to(event_time.dtype)

    # Base policy: heterogeneous delay, longer for lab-like variables and late windows.
    base_delay = 0.03 * horizon + 0.12 * horizon * var_is_lab + 0.08 * horizon * late
    jitter = 0.02 * horizon * torch.rand_like(event_time)
    record_time = event_time + (base_delay + jitter) * event_mask

    # Value-pending: some lab values are ordered but unavailable by cutoff.
    pending_prob = (0.10 + 0.35 * var_is_lab + 0.20 * late).clamp(0.0, 0.8)
    value_available = (torch.rand_like(event_time) > pending_prob).to(event_time.dtype) * event_mask

    # Batch upload: snap a subset of records to shared upload rounds.
    upload_round = torch.round(record_time / horizon * 8.0) / 8.0 * horizon
    batch_upload = ((event_var_id % 3) == 0).to(event_time.dtype)
    record_time = torch.where(batch_upload.bool(), upload_round, record_time)

    # As-of cutoff: deployment decision time in record-time coordinates.
    cutoff_fraction = 0.55 + 0.35 * torch.rand(bsz, device=device, dtype=event_time.dtype)
    as_of_cutoff = (cutoff_fraction[:, None] * horizon).squeeze(1)

    out = dict(batch)
    out["record_time"] = record_time
    out["value_available"] = value_available
    out["as_of_cutoff"] = as_of_cutoff

    # TCF-style bitemporal query targets from visible or near-future events.
    query_idx = torch.randint(0, num_events, (bsz, 4), device=device)
    bidx = torch.arange(bsz, device=device)[:, None]
    out["query_event_time"] = (event_time[bidx, query_idx] / horizon).clamp(0.0, 1.0)
    out["query_target_var"] = event_var_id[bidx, query_idx].clamp(0, num_vars - 1)
    # A production collator should use pathology bin labels from the atomizer/bin config.
    out["query_target_bin"] = torch.bucketize(value[bidx, query_idx], torch.tensor([-1.0, 0.0, 1.0], device=device))
    delay = (record_time - event_time).clamp_min(0.0)
    delay_norm = delay[bidx, query_idx] / horizon
    out["query_target_delay_bin"] = torch.clamp((delay_norm * 4).long(), 0, 3)
    out["query_target_available"] = value_available[bidx, query_idx]
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `real-time-to-delayed shift`：训练中心近实时入库，测试中心化验值延迟返回。
   - `backfill shift`：离线训练数据允许按 event_time 回填，部署数据只能按 record_time as-of 决策。
   - `batch-upload shift`：设备或医院批量上传，导致多个事件共享 record_time。
   - `value-pending shift`：采样事件已发生但数值未返回，测试模型是否把“已下单/已记录”误当病理证据。

2. **对比方法**
   - 普通 TCF / EHR foundation model。
   - PULSE 中 LightGBM、传统深度模型与 LLM prompt / hybrid baseline。
   - STAR-Set、VP-GNN、MTM、MVC-CDE、QuITE 等 irregular / asynchronous baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DSPP、DCPD、DIPF、DRG-SFF 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-center worst-policy AUROC / AUPRC。
   - retrocausal gain：`margin_backfilled - margin_asof` 的分布。
   - curtain violation rate：回填视图带来异常大类别增益的比例。
   - pending leakage score：只用 pending/order/record delay 预测标签的能力。
   - bitemporal calibration：按 record latency bucket 分组后的 calibration error。
   - latency sidecar fidelity：延迟、pending、batch-upload 是否被 sidecar 解释，而不是进入分类状态。

4. **消融实验**
   - 去掉 record-time 幕帘，只按 event_time 排序，验证是否出现离线回填捷径。
   - 去掉 `L_retrocausal_curtain`，检查 backfilled margin 是否异常升高。
   - 去掉 `LatencySidecar`，检查 `h_asof` 是否吸收记录延迟。
   - 把 future pathology 与 future record availability 合并成一个 TCF 头，验证混合目标是否带来采样流程污染。
   - 将 record-time policy recipes 替换为随机 mask，验证收益来自双时轴可得性建模，而不是普通增强。

## 5. 预期创新性

1. **从采样是否发生转向数值何时可得**：首次把 sampling-policy shift 表述为 event-time 与 record-time 的双时轴错位，而不只是 mask、delta-t 或变量共现偏移。
2. **从离线回填分类转向 as-of 因果幕帘**：分类器不能读取部署时刻尚未返回的值，直接打击 ICU/EHR 中常见的 retrocausal shortcut。
3. **从 TCF future-event 预测转向双头 foreseeing**：病理未来与记录可得性未来分开建模，防止 future observation process 成为类别证据。
4. **从跨中心 benchmark 转向流程可得性诊断**：吸收 PULSE 的跨中心设置，进一步解释性能退化来自 record latency、pending ratio、batch upload 还是 backfill。
5. **与反事实采样低侵入兼容**：counterfactual sampler 只需改变 `record_time`、`value_available` 与 pending/backfill recipes，不需要 hazard、density ratio、posterior quotient、proof audit、IRT form、conformal calibration、policy jury、knockoff、gauge 或 RG flow。

## 6. 一句话投稿卖点

**DBCC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“事件发生时间与记录可得时间错位导致的回填/延迟/pending 捷径”问题，通过 Bitemporal Pathology Atomizer、record-time Causal Curtain、Anti-Retrocausal Margin Loss、Latency Sidecar 与双头 TCF-style foreseeing，让分类器只读取部署 as-of 时真正可得的病理证据，而不是依赖跨中心记录流程、离线回填、批量上传或 value-pending 制造的未来信息泄漏。**
