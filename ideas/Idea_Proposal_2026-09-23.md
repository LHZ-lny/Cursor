# Title: Do-BBP Semantic Spike Sieve：面向采样策略偏移的随机矩阵病理尖峰筛

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md`、`**/*work*summary*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化持久记忆 `MEMORIES.md`，并纳入其中记录但当前工作区未完全落盘的历史提案摘要。
- 已完整读取当前仓库 `ideas/Idea_Proposal_*.md` 的全部历史提案文件，覆盖 2026-06-12 至 2026-09-22 的已落盘提案。
- 已读取 `paper_daily.md` 的近期段落，重点纳入：
  - **SLiCE / Structured Linear CDEs**：结构化线性 CDE 让连续时间不规则序列主干具备并行、高表达的事件状态生成能力。
  - **MedFuse**：乘法式 value-feature fusion 说明临床 token 的数值语义必须条件化变量身份，但也会放大“某变量为什么被测”的 policy-conditioned value shortcut。
  - **RoMAE / SLAN / CHARM / TimeCHEAT / ECG latent ODE**：分别提示连续坐标、非插补 switch、通道语义、局部/全局通道策略和采样率可恢复性都重要，但历史提案已覆盖大量直接改造路线。

### 历史核心机制黑名单

本提案显式避开以下已出现主机制：

1. hazard / score null-space / hazard resampling / do-risk variance。
2. commutator graph surgery、policy residual sink、state-policy graph 拆分。
3. evidence market / protocol tax / marginal audit。
4. posterior quotient、density ratio、doubly robust、cubature。
5. optional stopping martingale、topology capsule、policy gauge、syndrome code、knockoff。
6. evidential vacuity、policy lattice、solver trace、conformal sleeve、IV、jury。
7. sequent proof、poset clock、IRT-DIF、RG fixed point、privacy cloak、synthetic forge、JEPA debate。
8. PID prism、Noether action、meta workflow adapter、collider seal、source atlas。
9. renewal exposure、palimpsest memory、SSA dead-code elimination、Robin boundary solver、Blackwell experiment order、elicitable functional compass、Lyapunov interval observer。

本轮选择一个新的正交切入点：**不估计采样概率，不要求多视图一致，不做校准集合、证明、编译、边界、记忆或实验序；而是把事件表示矩阵的随机矩阵谱分成 sampling-policy bulk 与 pathology spike。采样政策可以改变观测数、panel 密度、通道预算和 aspect ratio，从而改变协方差 bulk；但真正可迁移的病理语义应表现为超过 Marchenko-Pastur / BBP 阈值的稳定语义尖峰。分类器只读取 bulk-edge 校正后的 spike statistic。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样分类中，很多 shortcut 实际是 **样本协方差谱的 bulk 伪信号**：

- 报警后高频复测会把相似 token 成批塞入模型，使事件状态协方差出现大 bulk 能量；
- panel 同步联测会制造多个变量在短窗内的共同波动，普通 attention / pooling 可能把它当成稳定生理耦合；
- 变量预算或低采样率会改变事件数与 hidden dimension 的 aspect ratio，使同一病理状态下的统计噪声谱形不同；
- MedFuse 式乘法 token 很容易把“某变量在某政策下被测到”与数值语义相乘，形成 policy-conditioned covariance direction；
- SLiCE 式结构化线性 CDE 可以高效生成事件状态，但如果读出仍直接池化 token，就可能把采样密度和局部共测图吸收到分类边界。

随机矩阵理论给了一个不同于历史所有路线的表述：

> 对事件状态矩阵 `H in R^{N x D}`，采样政策主要改变的是有限样本噪声 bulk：`N/D`、重复观测、panel batch、低质量测量都会移动 Marchenko-Pastur bulk edge。真正由病理状态产生、能跨政策迁移的语义因子，应表现为越过 BBP phase transition 的 spiked components。

因此，本提案不问“某个点为什么被采样”，也不要求“不同采样策略输出相同”。它问：

> 当前分类 margin 是否来自超出 policy bulk edge 的语义 spike？如果一个方向只是在训练政策的 bulk 内漂浮，它不应成为类别证据。

这与采样解耦/反事实干预框架自然结合：

- value process 用 MedFuse / SLiCE 生成 value-conditioned event states；
- sampling process 只估计当前 observation experiment 的 aspect ratio、bulk noise scale、repeat/panel inflation，不进入分类头；
- counterfactual intervention 生成 policy-bulk audit views，校准“哪些谱变化只是采样 bulk 移动，哪些是病理 spike 真的出现”；
- classifier 只读取 BBP-stable spike statistic。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：MedFuse-SLiCE Event State + Spectrum Readout

给定不规则事件：

```text
e_i = (value_i, time_i, variable_i, quality_i, mask_i)
```

先用 MedFuse 式乘法融合形成 token：

```text
token_i = FeatureEmbed(variable_i) * ValueEmbed(value_i)
        + TimeCoord(time_i)
        + QualityEmbed(quality_i)
```

然后用结构化线性 CDE / parallel scan 主干得到事件状态：

```text
h_i = StructuredLinearCDE(token_{<=i}, time_{<=i})
H   = [h_1, ..., h_N]^T
```

这里吸收 SLiCE 的优势：可扩展、连续时间、适合反事实采样 view 批量训练。但核心创新不在 CDE 主干，而在 **随机矩阵谱读出**。

### 2.2 Sampling Branch：只估计 Bulk Edge，不进分类头

sampling branch 从观测坐标估计：

```text
q_policy      = effective_event_count / effective_hidden_dim
sigma_policy  = bulk noise scale
repeat_infl   = repeated-check inflation
panel_infl    = local co-measurement inflation
quality_infl  = measurement-noise inflation
```

由此得到 Marchenko-Pastur bulk edge：

```text
lambda_plus = sigma_policy^2 * (1 + sqrt(q_policy))^2
```

它只用于谱筛选和损失校准，不作为分类特征。

### 2.3 BBP Semantic Spike Sieve

对 centered event state 矩阵计算样本协方差：

```text
C = H_centered^T W H_centered / n_eff
eigvals, eigvecs = eig(C)
spike_mask_k = sigmoid((eigval_k - lambda_plus - margin) / tau)
```

然后用 shrinkage 后的 spike statistic 分类：

```text
spike_stat = sum_k spike_mask_k * shrink(eigval_k, lambda_plus) * eigvec_k
logits = Classifier(spike_stat)
```

这和历史机制的差异：

- 不是频域掩码：谱是 token covariance eigen-spectrum，不是时间频率；
- 不是 Nystrom / logdet volume：不选择样本基，也不最大化体积；
- 不是 Krylov / annihilator：不构造 policy operator 子空间；
- 不是 Blackwell 实验序：不学习 garbling kernel；
- 不是 renewal exposure / palimpsest memory：不改写证据累计单位或记忆写入权；
- 不是多视图一致性：反事实 views 只校准 bulk 与 spike，不要求 logits 相同。

## 3. Loss：从一致性转向 Bulk-Edge / Spike Discipline

总目标：

```text
L = L_cls
  + lambda_bulk * L_MP_bulk_calibration
  + lambda_bbp  * L_BBP_spike_semantics
  + lambda_leak * L_bulk_margin_leakage
  + lambda_cf   * L_counterfactual_bulk_audit
```

### A. Spike Classification `L_cls`

只用 spike statistic 分类：

```text
L_cls = CE(Classifier(spike_stat), y)
```

### B. MP Bulk Calibration `L_MP_bulk_calibration`

对每个 batch，要求 bulk eigenvalues 的低阶矩与 sampling branch 估计的 MP bulk 匹配：

```text
bulk_eigs = eigvals[eigvals <= lambda_plus]
L_bulk =
  SmoothL1(mean(bulk_eigs), sigma_policy^2)
  + SmoothL1(max_soft(bulk_eigs), lambda_plus)
```

这让模型明确把重复复测、panel packing、低质量测量、变量预算造成的谱能量解释为 policy bulk。

### C. BBP Spike Semantics `L_BBP_spike_semantics`

真正 spike 必须有 value semantics 支持。用病理分箱、value surprise 或临床摘要作为弱监督：

```text
pathology_hat = PathologyHead(spike_stat)
L_bbp = CE(pathology_hat, pathology_summary)
```

若一个方向越过 bulk edge，但无法解释病理值语义，就不能稳定成为分类证据。

### D. Bulk Margin Leakage `L_bulk_margin_leakage`

构造一个 bulk-only diagnostic readout：

```text
bulk_stat = sum_k (1 - spike_mask_k) * eigval_k * eigvec_k
bulk_logits = BulkProbe(stopgrad(bulk_stat))
```

训练时不让 `bulk_stat` 进入分类器，同时惩罚 bulk probe 对标签过强：

```text
L_leak = relu(label_predictability(bulk_logits) - tau)^2
```

这不是 adversarial 去偏；主分类器不需要欺骗 probe。它只是审计并限制 bulk 中残留的标签捷径。

### E. Counterfactual Bulk Audit `L_counterfactual_bulk_audit`

反事实采样模块生成 policy-only 谱扰动：

- `repeat_burst`: 复制近似相同 value token；
- `panel_pack`: 同步多个变量；
- `variable_budget`: 降低某些变量覆盖；
- `low_rate`: 稀疏化时间坐标；
- `quality_degrade`: 增加测量噪声。

对这些 view，不要求 logits 一致，只要求：

```text
policy-only edit -> lambda_plus may move
policy-only edit -> stable spike count should not increase
```

损失：

```text
L_cf = mean_view relu(spike_mass_view - spike_mass_factual - semantic_gain_view - eps)^2
```

其中 `semantic_gain_view` 由 value novelty / pathology-bin change 计算。若反事实 view 只是重复、打包或低质化，semantic gain 近似 0，spike mass 不应凭采样政策增加。

## 4. Dataloader：Spectrum Audit Collator

新增 `SpectrumAuditCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_quality`。
2. `pathology_summary`：病理分箱摘要、value surprise summary 或任务相关临床摘要。
3. `policy_spectrum_descriptor`：有效事件数、变量覆盖、panel density、repeat density、measurement noise、pending ratio。
4. `spectrum_policy_view_bank`：
   - repeat burst；
   - panel pack / split；
   - variable budget；
   - low-rate thinning；
   - quality degradation。
5. `semantic_gain_target`：每个 view 相对事实 view 的真实 value novelty 增益。

这些 views 不是 contrastive positives，不用于 logits consistency；它们只检验 policy-only 采样变化是否被吸收入 bulk，而不是制造新 spike。

## 5. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.ndim < x.ndim:
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


class MuFuseEventStem(nn.Module):
    """MedFuse-style value-feature multiplication for irregular event tokens."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.LayerNorm(hidden_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp_min(0)
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        feature = self.var_embed(var_id)
        value_h = self.value_proj(torch.stack([value, quality], dim=-1))
        time_h = self.time_proj(torch.stack([time_norm, torch.log1p(delta_t)], dim=-1))
        token = feature * value_h + time_h
        return self.out(token) * mask.unsqueeze(-1)


class StructuredLinearEventScan(nn.Module):
    """SLiCE-inspired lightweight structured scan over event states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.Sigmoid(),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, token: torch.Tensor, time: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, steps, hidden_dim = token.shape
        state = token.new_zeros(bsz, hidden_dim)
        states = []
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        for idx in range(steps):
            step = self.in_proj(token[:, idx])
            gate = self.gate(torch.cat([state, step, torch.log1p(delta_t[:, idx:idx + 1])], dim=-1))
            cand = self.update(step, state)
            state = torch.where(
                mask[:, idx:idx + 1] > 0,
                gate * cand + (1.0 - gate) * state,
                state,
            )
            states.append(state)
        return torch.stack(states, dim=1) * mask.unsqueeze(-1)


class PolicyBulkEdgeHead(nn.Module):
    """Estimate MP bulk parameters from observation coordinates only."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.to_sigma = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.to_q = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, batch: dict, hidden_dim: int) -> dict:
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(time))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_rate = F.one_hot(var_id, self.num_vars).to(time.dtype) * mask.unsqueeze(-1)
        var_rate = var_rate.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        close = (delta_t <= masked_mean(delta_t, mask, dim=1).unsqueeze(-1).clamp_min(1e-6)).to(time.dtype)
        var_change = torch.zeros_like(mask)
        var_change[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(time.dtype)
        stats = torch.cat(
            [
                mask.mean(dim=1, keepdim=True),
                masked_mean((time_norm <= 0.33).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean((time_norm > 0.66).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean(torch.log1p(delta_t), mask, dim=1).unsqueeze(-1),
                masked_mean(close * var_change, mask, dim=1).unsqueeze(-1),
                masked_mean(quality, mask, dim=1).unsqueeze(-1),
                mask.sum(dim=1, keepdim=True) / mask.size(1),
            ],
            dim=-1,
        )
        h = self.net(torch.cat([var_rate, stats], dim=-1))
        sigma = self.to_sigma(h).squeeze(-1) + 1e-4
        # Effective aspect ratio is bounded away from zero and scaled by hidden_dim.
        n_eff = mask.sum(dim=1).clamp_min(1.0)
        q_emp = n_eff / float(hidden_dim)
        q = 0.5 * q_emp + 0.5 * self.to_q(h).squeeze(-1).clamp(1e-3, 10.0)
        lambda_plus = sigma.pow(2) * (1.0 + torch.sqrt(q)).pow(2)
        return {"sigma": sigma, "q": q, "lambda_plus": lambda_plus}


class BBPSpikeReadout(nn.Module):
    """Separate MP bulk from BBP-stable semantic spikes."""

    def __init__(self, hidden_dim: int, num_classes: int, num_pathology_bins: int, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.pathology_head = nn.Linear(hidden_dim, num_pathology_bins)
        self.bulk_probe = nn.Linear(hidden_dim, num_classes)

    def forward(self, states: torch.Tensor, mask: torch.Tensor, lambda_plus: torch.Tensor) -> dict:
        # Centered weighted covariance per sample.
        mean = masked_mean(states, mask, dim=1).unsqueeze(1)
        centered = (states - mean) * mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        cov = torch.einsum("bnh,bnd->bhd", centered, centered) / denom[:, None, None]
        cov = 0.5 * (cov + cov.transpose(-1, -2))

        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(0.0)
        spike_mask = torch.sigmoid((eigvals - lambda_plus[:, None]) / self.temperature)
        shrink = F.relu(eigvals - lambda_plus[:, None])
        spike_stat = torch.einsum("bk,bhk->bh", spike_mask * shrink, eigvecs)
        bulk_stat = torch.einsum("bk,bhk->bh", (1.0 - spike_mask) * eigvals, eigvecs)
        logits = self.classifier(spike_stat)
        return {
            "logits": logits,
            "spike_stat": spike_stat,
            "bulk_stat": bulk_stat,
            "eigvals": eigvals,
            "spike_mask": spike_mask,
            "spike_mass": (spike_mask * shrink).sum(dim=-1),
            "pathology_logits": self.pathology_head(spike_stat),
            "bulk_logits": self.bulk_probe(bulk_stat.detach()),
        }


class DoBBPSemanticSpikeSieve(nn.Module):
    """Sampling-policy robust classifier via MP bulk stripping and BBP spike readout."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int, num_pathology_bins: int):
        super().__init__()
        self.stem = MuFuseEventStem(num_vars, hidden_dim)
        self.scan = StructuredLinearEventScan(hidden_dim)
        self.bulk = PolicyBulkEdgeHead(num_vars, hidden_dim)
        self.readout = BBPSpikeReadout(hidden_dim, num_classes, num_pathology_bins)
        self.hidden_dim = hidden_dim

    def encode(self, batch: dict) -> dict:
        token = self.stem(batch)
        states = self.scan(token, batch["event_time"], batch["event_mask"])
        bulk = self.bulk(batch, hidden_dim=self.hidden_dim)
        out = self.readout(states, batch["event_mask"], bulk["lambda_plus"])
        return {**bulk, **out, "states": states}

    def mp_bulk_calibration_loss(self, out: dict) -> torch.Tensor:
        eigvals = out["eigvals"]
        lambda_plus = out["lambda_plus"]
        soft_bulk = torch.sigmoid((lambda_plus[:, None] - eigvals) / 0.05)
        bulk_mean = (eigvals * soft_bulk).sum(dim=1) / soft_bulk.sum(dim=1).clamp_min(1.0)
        bulk_max = (eigvals * soft_bulk).amax(dim=1)
        loss_mean = F.smooth_l1_loss(bulk_mean, out["sigma"].pow(2).detach())
        loss_edge = F.smooth_l1_loss(bulk_max, lambda_plus.detach())
        return loss_mean + loss_edge

    def bulk_margin_leakage_loss(self, out: dict, labels: torch.Tensor, tau: float = 0.10) -> torch.Tensor:
        ce = F.cross_entropy(out["bulk_logits"], labels)
        uniform_ce = torch.log(torch.tensor(out["bulk_logits"].size(-1), device=ce.device, dtype=ce.dtype))
        return F.relu(uniform_ce - ce - tau).pow(2)

    def counterfactual_bulk_audit_loss(self, batch: dict, factual: dict) -> torch.Tensor:
        views = batch.get("spectrum_policy_view_bank", [])
        if not views:
            return torch.zeros((), device=factual["logits"].device)
        factual_mass = factual["spike_mass"].detach()
        losses = []
        gains = batch.get("semantic_gain_target", None)
        for idx, view in enumerate(views):
            cf = self.encode(view)
            if gains is None:
                gain = torch.zeros_like(factual_mass)
            else:
                gain = gains[:, idx].to(factual_mass.dtype)
            losses.append(F.relu(cf["spike_mass"] - factual_mass - gain - 0.05).pow(2).mean())
        return torch.stack(losses).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_bulk: float = 0.20,
        lambda_bbp: float = 0.20,
        lambda_leak: float = 0.08,
        lambda_cf: float = 0.25,
    ) -> dict:
        out = self.encode(batch)
        labels = batch["labels"]
        cls_loss = F.cross_entropy(out["logits"], labels)
        bulk_loss = self.mp_bulk_calibration_loss(out)
        pathology_loss = F.cross_entropy(out["pathology_logits"], batch["pathology_summary"].long())
        leak_loss = self.bulk_margin_leakage_loss(out, labels)
        cf_loss = self.counterfactual_bulk_audit_loss(batch, out)
        total = cls_loss + lambda_bulk * bulk_loss + lambda_bbp * pathology_loss + lambda_leak * leak_loss + lambda_cf * cf_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "mp_bulk_calibration_loss": bulk_loss.detach(),
            "bbp_spike_semantics_loss": pathology_loss.detach(),
            "bulk_margin_leakage_loss": leak_loss.detach(),
            "counterfactual_bulk_audit_loss": cf_loss.detach(),
            "mean_spike_mass": out["spike_mass"].mean().detach(),
            "mean_lambda_plus": out["lambda_plus"].mean().detach(),
        }
```

## 6. Spectrum Audit Collator 草稿

```python
import torch


@torch.no_grad()
def build_spectrum_audit_batch(batch: dict) -> dict:
    """Create policy-only spectrum perturbations for BBP/MP auditing.

    These views are not contrastive positives and are not used for logits consistency.
    They test whether policy-only sampling edits move the covariance bulk without
    creating new semantic spikes.
    """

    out = dict(batch)
    views = []
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    quality = batch.get("measurement_quality", torch.ones_like(value))
    bsz, num_events = time.shape
    device = time.device
    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, new_quality):
        view = dict(batch)
        view["event_value"] = new_value
        view["event_time"] = new_time
        view["event_var_id"] = new_var
        view["event_mask"] = new_mask
        view["measurement_quality"] = new_quality
        view.pop("spectrum_policy_view_bank", None)
        return view

    # Repeat burst: duplicates should inflate bulk, not create spikes.
    late = (time_norm > 0.66).to(mask.dtype)
    repeated_value = torch.where(late > 0, value.roll(shifts=1, dims=1), value)
    views.append(clone_with(repeated_value * mask, time, var_id, mask, quality))

    # Panel pack: synchronize nearby cross-variable events.
    packed_time = time.clone()
    packed_time[:, 1:] = torch.where(mask[:, 1:] > 0, time[:, :-1], time[:, 1:])
    views.append(clone_with(value * mask, packed_time, var_id, mask, quality))

    # Variable budget: reduce odd-variable coverage.
    budget_mask = mask * (1.0 - 0.5 * (var_id % 2 == 1).to(mask.dtype))
    views.append(clone_with(value * budget_mask, time, var_id, budget_mask, quality * budget_mask))

    # Low-rate thinning: changes aspect ratio and recoverability.
    low_rate = mask * ((torch.arange(num_events, device=device)[None] % 3) == 0).to(mask.dtype)
    views.append(clone_with(value * low_rate, time, var_id, low_rate, quality * low_rate))

    # Quality degradation: broadens bulk noise.
    noisy_quality = quality * 0.25
    views.append(clone_with(value * mask, time, var_id, mask, noisy_quality))

    out["spectrum_policy_view_bank"] = views
    out["semantic_gain_target"] = torch.zeros(bsz, len(views), device=device)
    return out
```

## 7. 实验切入点

1. **Policy shift 构造**
   - `repeat-burst shift`：训练医院高风险样本有密集复测，测试医院将同样复测应用到普通患者。
   - `panel-pack / panel-split shift`：训练中心局部 panel 共测强，测试中心拆为异步事件。
   - `variable-budget shift`：测试中心降低某些变量覆盖，改变 aspect ratio。
   - `low-rate morphology shift`：可穿戴 / ECG 从高采样率降为低采样率，细形态 spike 应下降。
   - `quality-noise shift`：测量噪声上升，MP bulk edge 应诚实提高。

2. **对比方法**
   - SLiCE / Neural CDE / Transformer / GRU-D / BAT。
   - MedFuse tokenization baseline。
   - RoMAE continuous-position baseline。
   - TimeCHEAT local-CD/global-CI baseline。
   - 历史方案中的 DREM、Do-Palimpsest、DSSS、DRBI、DBSES、DEFC、DLIO 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - worst-policy AUROC / AUPRC。
   - bulk-label leakage：bulk-only probe 的标签预测能力。
   - spike survival under true pathology：真实病理突变是否产生稳定 spike。
   - policy-only spike inflation：repeat/panel/budget/quality views 是否凭政策制造 spike mass。
   - MP edge calibration：`lambda_plus` 与经验 bulk edge 的误差。

4. **消融实验**
   - 去掉 MP bulk calibration，检查重复复测是否产生虚假 spike。
   - 去掉 BBP spike mask，直接池化 SLiCE states，检查采样密度 shortcut。
   - 去掉 counterfactual bulk audit，检查 panel-pack / repeat-burst 是否增加 spike mass。
   - 将 policy descriptor 直接拼入 classifier 作为反例，验证院内性能可能升高但跨政策性能下降。
   - 把 covariance spectrum 换成时间频谱，验证收益来自随机矩阵 bulk/spike，而不是传统频域特征。

## 8. 预期创新性

1. **从采样去偏转向随机矩阵谱筛**：首次把 sampling-policy shift 表述为事件状态协方差的 bulk edge 移动，而非缺失概率、图结构、校准集合或观测程序变化。
2. **从 MedFuse 表达力转向 value-policy 乘法后的谱审计**：保留值-变量乘法语义，但防止 policy-conditioned value token 在 bulk 内成为捷径。
3. **从 SLiCE 高效主干转向 BBP-stable 读出**：利用结构化线性 CDE 高效生成连续时间事件状态，但分类只读超过 MP edge 的语义 spike。
4. **从一致性视图转向 bulk / spike 因果诊断**：反事实采样 view 不要求 logits 相同，只检查 policy-only 编辑是否只移动 bulk 而不创造新 spike。
5. **解释性明确**：模型能报告当前预测依赖多少 spike、bulk edge 是否因采样政策抬高、以及错误样本是否来自 bulk-label leakage。

## 9. 一句话投稿卖点

**DBBP-SS 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样政策改变事件状态协方差的随机矩阵 bulk，而真实病理状态应表现为越过 BBP 阈值的语义 spike”的问题，通过 MedFuse-SLiCE event states、MP bulk-edge calibration、BBP spike readout 与 counterfactual bulk audit，让分类器只读取跨采样政策稳定的病理尖峰，而不是读取训练医院或设备制造的重复复测、panel 同步、变量预算和低质量测量 bulk 捷径。**
