# Title: Do-PAC Receptive Witness Fields：面向采样策略偏移的局部证据见证泛化场

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前 `/workspace` 未检出该文件；并已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md`、中文 `*总结*.md`，均未发现可替代总结文件。
- 已读取自动化持久记忆 `MEMORIES.md`，并补读 `idea_2026-08-07.md` 至 `idea_2026-10-14.md` 中未完整显示的历史 proposal 摘要。
- 已读取/抽取当前仓库 `ideas/Idea_Proposal_*.md` 的全部历史标题、黑名单与核心机制段，覆盖当前工作区已落盘的 43 份历史提案，直至 `Idea_Proposal_2026-10-14.md`。
- 已读取近期 `paper_daily_2026-10-13.md` 与 `paper_daily_2026-10-14.md`；本轮重点吸收：
  - **i-borf / Time-Aware Bag-of-Receptive-Fields**：原生不规则序列上的 time-weighted local receptive-field word，强调可解释局部证据与真实时间支持。
  - **ChronoFlow**：分层生成 observation counts、feature frequencies、observation times、co-observation patterns 与 values，为构造可控 sampling-policy counterfactuals 提供工具。
  - **PRISM**：显式把 sampling pattern shift 从普通 domain shift 中分离出来。
  - **MMNAR causal clinical representation**：把变量/模态可见性视为 clinical assignment mechanism，而非普通缺失噪声。

### 历史核心机制黑名单

本提案显式避开以下已出现主机制：hazard / score 零空间、图交换子、证据市场、后验商、密度比、RKHS cubature、误差制图、HSIC/VQ、停时鞅、审查拓扑、policy gauge、syndrome code、knockoff calendar、observability witness、evidential vacuity、policy lattice、solver trace front-door、conformal sleeves、IV/control-function、Borda jury、Krylov annihilator、Nystrom volume、tropical routes、viva examiner、sequent proof、disease poset clock、IRT-DIF、RG fixed point、bitemporal curtain、clinical tomography、matched risk-set likelihood、policy privacy cloak、synthetic falsification forge、dialectical JEPA、PID prism、Noether action、meta workflow immunization、collider seal、source-atlas retrieval、Schwarzian warp、renewal exposure、palimpsest memory、typed SSA、Robin boundary solver、Blackwell experiment order、elicitable functional compass、Lyapunov observer、BBP / Marchenko-Pastur random matrix、RIP sparse recovery、factorial cumulant thinning、SWING delta circuit breaker、Boehm spline knot firewall、freshness anchor-cone demixer、Cauchy residue firewall、microlocal wavefront shield、counterchoice utility balancer，以及普通 state-policy 双分支、对抗环境分类器、跨视图 logits/representation 一致性、频域掩码对比学习、missingness pattern 直接分类、普通 retrieval classifier。

本轮选择新的正交切入点：**不把采样政策建成概率、图、谱、样条、复轮廓、波前、选择集、程序、证明、记忆、实验序、可靠性锥或高阶计数；而是把分类证据拆成一组 time-weighted local receptive-field witnesses。ChronoFlow 风格 policy generator 只负责生成“同一潜在病程在不同观测制度下会暴露哪些 witness 的机会”，分类器学习一个 PAC-Bayes witness posterior，并最小化跨生成策略环境的泛化上界。模型不要求各策略视图 logits 完全一致，而是要求：在可控 observation-process 变化下，能证明被分类器依赖的局部 witness 组合具有低策略依赖复杂度和稳定的最坏策略风险。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 paper_daily 给出两个非常互补的信号：

1. **i-borf** 说明在不规则分类中，局部 receptive fields 可以成为可解释证据单位；time-delta weighting 能把“一个形态被多长真实时间支持”显式编码出来。
2. **ChronoFlow** 说明 observation process 本身可以被分层建模和生成：先生成观测次数与频率，再生成时间点与变量联测结构，最后生成 values。

历史方案已经覆盖了大量“把采样政策建成某种数学对象再防火墙化”的路线：概率、图、后验、拓扑、gauge、纠错码、隐私、程序、边界、谱、稀疏测量、样条、复分析、波前、choice set 等。本提案不再发明另一种结构空间，而是回到审稿人最容易理解的分类证据单位：

> 一个模型为什么判为高风险？因为哪些局部数值形态、趋势片段、阈值附近持续时间、跨变量相邻形态作为 witness 支持了该类别？

采样策略偏移真正危险的地方在于：它会改变 witness 被“看见”的机会。例如同一段乳酸缓慢上升：

- routine-heavy 医院可能只暴露一个长时间支持的粗 witness；
- alarm-dense 医院可能暴露多个短间隔重复 witness；
- panel-pack 医院可能暴露跨变量共现 witness；
- low-duty-cycle 设备可能只暴露缺口两端的形态 witness。

如果分类器把“witness 出现机会”当成病理强度，就会跨策略失效。**Do-PAC Receptive Witness Fields (DPRWF)** 的核心是：让分类器仍然使用可解释局部 witness，但用 ChronoFlow-style policy views 估计这些 witness 在不同 observation processes 下的暴露不确定性，并用 PAC-Bayes 上界惩罚对策略敏感、高复杂度、低覆盖的 witness posterior。

这与“采样解耦/反事实干预”框架结合后形成三层分工：

- value branch：产生 time-weighted local receptive-field witness features；
- policy generator：产生 observation-count / time / co-observation counterfactual policy views；
- classifier：学习一个只依赖 value witness 的 Gibbs / posterior ensemble，并通过跨策略 PAC-Bayes bound 控制 worst-policy risk，而不是让 policy descriptor 直接进入分类头。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：ChronoFlow Policy Witness Bank

新增 `PolicyWitnessBankCollator`。每个训练样本不只返回事实事件流，还返回同一潜在病程下的若干 observation-process views：

```text
factual_view:
  event_time, event_var, event_value, event_mask

policy_views:
  count_shift_view        # 改 observation counts / feature frequency
  time_shift_view         # 改 observation times / gaps
  coobs_shift_view        # 改 feature co-observation pattern
  alarm_dense_view        # 告警后密集复测
  low_duty_view           # 低频设备或资源受限采样

semantic_release_view:
  保持 observation process 近似不变，注入 value novelty / threshold crossing
```

这里的 policy views 借鉴 ChronoFlow 的层级：先改观测次数与变量频率，再改时间点与联测结构，最后保持 values 来模拟同一病程被不同政策观察。它不是用于训练一个生成器本身，而是作为分类器的 policy stress environment。

关键差异：

- 不估计 density ratio；
- 不构造 choice set；
- 不做图边拆分；
- 不要求所有视图 logits 一致；
- 只用这些视图估计 witness posterior 的跨策略风险与复杂度。

### 2.2 改 Encoder：Time-Weighted Receptive Witness Extractor

借鉴 i-borf，但不做固定词袋浅分类器。对每个变量和多尺度窗口，构造局部 witness：

```text
w_i = [
  time_weighted_mean,
  time_weighted_slope,
  time_weighted_variance,
  threshold_dwell,
  signed_area,
  value_novelty,
  local_support_duration,
  variable_pair_context
]
```

每个 witness 经过小型 MLP 生成两个对象：

```text
e_i = StateWitnessEmbed(w_i)        # value-supported local evidence
a_i = PolicyAuditEmbed(delta_t, mask_density, coobs_tag)  # 只用于审计
```

分类主路径只读取 `e_i`。`a_i` 不进入分类器，只用于估计 policy exposure 和训练泄漏 probe。

### 2.3 改 Classifier：PAC-Bayes Witness Posterior

模型学习一个样本级 witness posterior：

```text
q_phi(z | {e_i}) = categorical / Concrete distribution over local witnesses
p_0(z)           = policy-agnostic prior, shared across environments
```

分类器不是池化所有事件，也不是 attention 任意加权；它从 witness posterior 采样或软选择少量局部 witnesses：

```text
h = sum_i q_i * e_i
logits = Classifier(h)
```

其中 `q_i` 只能由 value witness features 产生，不能使用 raw mask、delta-t histogram、co-observation tag、hospital id、policy view id 或 ChronoFlow generator parameters。

### 2.4 改 Loss：Policy-Generated PAC-Bayes Bound

总目标：

```text
L = L_empirical_gibbs
  + lambda_pac   * L_policy_pac_bayes
  + lambda_stab  * L_witness_posterior_stability
  + lambda_rel   * L_semantic_witness_release
  + lambda_cov   * L_policy_coverage_floor
  + lambda_audit * L_policy_witness_audit
```

#### A. Empirical Gibbs Classification

对 factual view 与 policy-generated views 分别计算分类风险：

```text
L_emp = mean_v CE(GibbsClassifier(v), y)
```

这里允许不同 policy view 的 logits 有小幅变化；我们不使用 logits consistency 作为主约束。

#### B. Policy PAC-Bayes Bound

对每个样本，把 ChronoFlow-style policy views 看作同一潜在病程下的环境样本，估计最坏策略经验风险和 witness posterior 的复杂度：

```text
R_hat_max = max_v CE(logits_v, y)
KL_qp     = KL(q_phi(z | view_v) || p_0(z))
n_eff     = effective number of value-supported witnesses

L_pac = R_hat_max + sqrt((KL_qp + log(2 * sqrt(n_eff) / delta)) / (2 * n_eff))
```

直觉：如果模型依赖某些只在 alarm-dense 或 panel-pack 下出现的 witness，则 `R_hat_max` 与 `KL_qp` 会升高；如果 witness 组合在不同观测制度下仍能以低复杂度解释标签，PAC-Bayes bound 会收紧。

#### C. Witness Posterior Stability

不要求 logits 一致，但要求被选中的 witness 语义类型稳定。对 policy views 的 witness posterior 做语义核均值匹配：

```text
L_stab = MMD( sum_i q_i^factual * sem(e_i^factual),
              sum_j q_j^policy  * sem(e_j^policy) )
```

这里匹配的是 value-witness semantic kernel，不匹配 raw timestamp、mask 或 co-observation pattern。

#### D. Semantic Witness Release

对 `semantic_release_view`，真实数值新息必须能改变 witness posterior 和 logits：

```text
target = threshold_cross + trend_flip + abnormal_dwell_gain
L_rel = BCE(release_logit, target)
      + relu(min_gain - KL(q_semantic || q_factual))^2
```

这避免模型为了收紧 PAC-Bayes bound 而过度保守。

#### E. Policy Coverage Floor

若某个样本在所有 policy views 中都缺少足够 value-supported witnesses，模型应输出不确定而不是利用 mask shortcut：

```text
coverage = sum_i q_i * value_support_i
L_cov = relu(coverage_floor - coverage)^2
```

#### F. Policy Witness Audit

训练一个 stop-gradient probe 从 posterior summary 预测 policy recipe：

```text
policy_logits = Probe(stopgrad([q_entropy, support_duration, selected_policy_audit_stats]))
```

如果 probe 很强，说明 witness posterior 仍携带策略指纹；该指标用于诊断与轻量惩罚，不作为主分类输入。

---

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def pairwise_mmd(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)
    kxx = torch.exp(-xx / (2 * bandwidth**2)).mean()
    kyy = torch.exp(-yy / (2 * bandwidth**2)).mean()
    kxy = torch.exp(-xy / (2 * bandwidth**2)).mean()
    return kxx + kyy - 2.0 * kxy


class TimeWeightedWitnessExtractor(nn.Module):
    """Build i-borf-inspired local receptive-field witnesses for irregular events."""

    def __init__(self, num_vars: int, hidden_dim: int, window_sizes: Iterable[int] = (3, 5, 9)):
        super().__init__()
        self.window_sizes = tuple(window_sizes)
        self.var_emb = nn.Embedding(num_vars, hidden_dim)
        self.witness_proj = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.audit_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.support_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def _window_stats(self, value: torch.Tensor, time: torch.Tensor, mask: torch.Tensor, width: int) -> torch.Tensor:
        pad = width // 2
        v = F.pad(value, (pad, pad), mode="replicate").unfold(1, width, 1)
        t = F.pad(time, (pad, pad), mode="replicate").unfold(1, width, 1)
        m = F.pad(mask, (pad, pad), mode="constant", value=0.0).unfold(1, width, 1)

        dt = torch.zeros_like(t)
        dt[..., 1:] = (t[..., 1:] - t[..., :-1]).clamp_min(0.0)
        weight = (dt + 1e-3) * m
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        mean = (weight * v).sum(dim=-1)
        centered = v - mean.unsqueeze(-1)
        var = (weight * centered.pow(2)).sum(dim=-1)
        duration = ((t.amax(dim=-1) - t.amin(dim=-1)).clamp_min(0.0)) * (m.sum(dim=-1) > 0).float()
        slope = (v[..., -1] - v[..., 0]) / duration.clamp_min(1e-3)
        signed_area = (weight * torch.tanh(v)).sum(dim=-1)
        threshold_dwell = (weight * (v.abs() > 1.0).float()).sum(dim=-1)
        novelty = (v[..., -1] - mean).abs()
        density = m.sum(dim=-1) / float(width)

        return torch.stack(
            [mean, var.sqrt(), slope, duration, signed_area, threshold_dwell, novelty, density],
            dim=-1,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        value = batch["event_value"]
        time = batch["event_time"]
        mask = batch["event_mask"].float()
        var_id = batch["event_var_id"].clamp_min(0)

        witness_tokens = []
        audit_tokens = []
        token_masks = []
        for width in self.window_sizes:
            stats = self._window_stats(value, time, mask, width)
            var_h = self.var_emb(var_id)
            token = self.witness_proj(torch.cat([var_h, stats], dim=-1))
            witness_tokens.append(token)

            delta_t = torch.zeros_like(time)
            delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
            audit_feat = torch.stack(
                [
                    torch.log1p(delta_t),
                    stats[..., 3],
                    stats[..., 7],
                    batch.get("coobs_density", torch.zeros_like(value)),
                    mask,
                ],
                dim=-1,
            )
            audit_tokens.append(self.audit_proj(audit_feat))
            token_masks.append(mask)

        witness = torch.cat(witness_tokens, dim=1)
        audit = torch.cat(audit_tokens, dim=1)
        witness_mask = torch.cat(token_masks, dim=1)
        value_support = self.support_head(witness).squeeze(-1) * witness_mask
        return {
            "witness": witness,
            "audit": audit,
            "witness_mask": witness_mask,
            "value_support": value_support,
        }


class PACBayesWitnessClassifier(nn.Module):
    """Gibbs-style classifier whose posterior can only read value witnesses."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.posterior_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.prior_logit = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )
        self.release_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.policy_probe = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8),
        )

    def encode(self, features: Dict[str, torch.Tensor], temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        witness = features["witness"]
        mask = features["witness_mask"]
        support = features["value_support"]

        logits_q = self.posterior_head(witness).squeeze(-1) / temperature
        logits_q = logits_q.masked_fill(mask <= 0, -1e4)
        q = torch.softmax(logits_q, dim=-1) * mask
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        prior_logits = torch.full_like(logits_q, self.prior_logit.item()).masked_fill(mask <= 0, -1e4)
        p = torch.softmax(prior_logits, dim=-1) * mask
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        pooled = torch.einsum("bt,bth->bh", q, witness)
        pred = self.classifier(pooled)
        release_logit = self.release_head(witness).squeeze(-1)

        entropy = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum(dim=-1)
        coverage = (q * support).sum(dim=-1)
        audit_summary = torch.einsum("bt,bth->bh", q.detach(), features["audit"])
        probe_input = torch.cat([audit_summary, entropy[:, None], coverage[:, None], mask.sum(dim=1, keepdim=True)], dim=-1)
        policy_logits = self.policy_probe(probe_input.detach())

        kl_qp = (q * (q.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(dim=-1)
        n_eff = support.sum(dim=-1).clamp_min(2.0)
        return {
            "logits": pred,
            "q": q,
            "p": p,
            "kl_qp": kl_qp,
            "n_eff": n_eff,
            "pooled": pooled,
            "release_logit": release_logit,
            "coverage": coverage,
            "entropy": entropy,
            "policy_logits": policy_logits,
        }


class DoPACReceptiveWitnessFields(nn.Module):
    def __init__(self, num_vars: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.extractor = TimeWeightedWitnessExtractor(num_vars, hidden_dim)
        self.classifier = PACBayesWitnessClassifier(hidden_dim, num_classes)

    def forward_view(self, view: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = self.extractor(view)
        out = self.classifier.encode(feats)
        return {**feats, **out}

    def training_loss(
        self,
        batch: Dict[str, torch.Tensor],
        delta: float = 0.05,
        lambda_pac: float = 0.35,
        lambda_stab: float = 0.15,
        lambda_release: float = 0.25,
        lambda_coverage: float = 0.10,
        lambda_audit: float = 0.02,
        coverage_floor: float = 0.35,
    ) -> Dict[str, torch.Tensor]:
        labels = batch["labels"]
        factual = self.forward_view(batch["factual_view"])
        views: List[Dict[str, torch.Tensor]] = [batch["factual_view"]] + list(batch.get("policy_views", []))
        outs = [factual] + [self.forward_view(v) for v in batch.get("policy_views", [])]

        ce_terms = torch.stack([F.cross_entropy(o["logits"], labels, reduction="none") for o in outs], dim=0)
        empirical_gibbs = ce_terms.mean()
        worst_policy_risk = ce_terms.max(dim=0).values.mean()

        kl = torch.stack([o["kl_qp"] for o in outs], dim=0).mean(dim=0)
        n_eff = torch.stack([o["n_eff"] for o in outs], dim=0).amin(dim=0)
        bound_radius = torch.sqrt((kl + math.log(2.0 / delta)) / (2.0 * n_eff))
        pac_loss = worst_policy_risk + bound_radius.mean()

        # Match selected value-witness semantics, not logits or raw sampling coordinates.
        stab_terms = []
        fact_sem = factual["pooled"].detach()
        for o in outs[1:]:
            stab_terms.append(F.smooth_l1_loss(o["pooled"], fact_sem))
        stability_loss = torch.stack(stab_terms).mean() if stab_terms else empirical_gibbs.new_zeros(())

        semantic_view = batch.get("semantic_release_view")
        if semantic_view is not None:
            sem = self.forward_view(semantic_view)
            target = semantic_view.get("semantic_target", torch.ones_like(sem["release_logit"]))
            release_loss = F.binary_cross_entropy_with_logits(
                sem["release_logit"],
                target.float(),
                reduction="none",
            )
            release_loss = (release_loss * sem["witness_mask"]).sum() / sem["witness_mask"].sum().clamp_min(1.0)
            posterior_gain = (sem["q"] * (sem["q"].clamp_min(1e-8).log() - factual["q"].detach().clamp_min(1e-8).log())).sum(dim=-1)
            release_loss = release_loss + F.relu(0.05 - posterior_gain).pow(2).mean()
        else:
            release_loss = empirical_gibbs.new_zeros(())

        coverage_loss = F.relu(coverage_floor - factual["coverage"]).pow(2).mean()

        policy_id = batch.get("policy_id")
        if policy_id is not None:
            audit_loss = F.cross_entropy(factual["policy_logits"], policy_id)
        else:
            audit_loss = empirical_gibbs.new_zeros(())

        total = (
            empirical_gibbs
            + lambda_pac * pac_loss
            + lambda_stab * stability_loss
            + lambda_release * release_loss
            + lambda_coverage * coverage_loss
            + lambda_audit * audit_loss
        )
        return {
            "loss": total,
            "empirical_gibbs_loss": empirical_gibbs.detach(),
            "policy_pac_bayes_loss": pac_loss.detach(),
            "witness_posterior_stability_loss": stability_loss.detach(),
            "semantic_witness_release_loss": release_loss.detach(),
            "policy_coverage_loss": coverage_loss.detach(),
            "policy_witness_audit_loss": audit_loss.detach(),
            "mean_kl_qp": factual["kl_qp"].mean().detach(),
            "mean_witness_coverage": factual["coverage"].mean().detach(),
            "mean_witness_entropy": factual["entropy"].mean().detach(),
        }
```

### Collator 草稿

```python
import torch


@torch.no_grad()
def build_policy_witness_bank(batch: dict, num_views: int = 4) -> dict:
    """Create ChronoFlow-inspired policy views for witness-level PAC-Bayes training."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"].float()
    device = value.device
    bsz, steps = value.shape

    def clone_view() -> dict:
        return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

    views = []

    # Count / frequency shift: hide every other low-support observation without changing values.
    count_shift = clone_view()
    keep = torch.ones_like(mask)
    keep[:, 1::2] = 0.0
    count_shift["event_mask"] = mask * keep
    count_shift["policy_recipe"] = torch.full((bsz,), 0, device=device, dtype=torch.long)
    views.append(count_shift)

    # Time shift: move timestamps toward routine slots but keep the same value path.
    time_shift = clone_view()
    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    norm_time = time / horizon
    time_shift["event_time"] = torch.round(norm_time * 8.0) / 8.0 * horizon
    time_shift["policy_recipe"] = torch.full((bsz,), 1, device=device, dtype=torch.long)
    views.append(time_shift)

    # Co-observation shift: rotate variables locally to mimic panel pack/split exposure.
    coobs_shift = clone_view()
    coobs_shift["event_var_id"] = torch.where(mask.bool(), var_id.roll(shifts=1, dims=1), var_id)
    coobs_shift["coobs_density"] = mask.roll(shifts=1, dims=1) * mask
    coobs_shift["policy_recipe"] = torch.full((bsz,), 2, device=device, dtype=torch.long)
    views.append(coobs_shift)

    # Alarm-dense view: duplicate high-novelty values into nearby slots.
    alarm_dense = clone_view()
    novelty = batch.get("value_novelty", value.abs())
    high = novelty >= novelty.quantile(0.75, dim=1, keepdim=True)
    shifted_value = value.roll(shifts=1, dims=1)
    alarm_dense["event_value"] = torch.where(high, 0.5 * value + 0.5 * shifted_value, value)
    alarm_dense["event_mask"] = torch.maximum(mask, high.float())
    alarm_dense["policy_recipe"] = torch.full((bsz,), 3, device=device, dtype=torch.long)
    views.append(alarm_dense)

    # Semantic release view: observation process is stable, value novelty is real.
    semantic = clone_view()
    inject = high.float() * mask
    semantic["event_value"] = value + 0.5 * inject * torch.sign(value + 1e-3)
    semantic["semantic_target"] = inject.repeat(1, 3) if inject.dim() == 2 else inject

    return {
        "factual_view": batch,
        "policy_views": views[:num_views],
        "semantic_release_view": semantic,
        "labels": batch["labels"],
        "policy_id": batch.get("policy_id"),
    }
```

---

## 4. 实验切入点

1. **跨策略评测**
   - PRISM-style controlled sampling pattern shift；
   - ChronoFlow-generated count / time / co-observation policy views；
   - ICU routine-heavy -> alarm-dense；
   - panel-pack -> panel-split；
   - low-duty wearable -> high-duty bedside monitor。

2. **可解释 witness 审计**
   - 显示 top selected receptive-field witnesses；
   - 检查 selected witness 的真实时间支持、变量、趋势和阈值 dwell；
   - 比较 factual 与 policy-generated views 下 witness posterior 的语义核距离；
   - 报告 policy probe 是否能从 posterior summary 预测采样制度。

3. **核心对比**
   - PRISM / DG under sampling pattern shifts；
   - i-borf / BORF-style interpretable classifier；
   - ChronoFlow augmentation + ordinary classifier；
   - GRU-D、mTAN、Raindrop、TimeCHEAT、ReTAMamba、TANDEM；
   - 现有采样解耦/反事实干预普通双分支版本。

4. **关键消融**
   - 去掉 PAC-Bayes bound，只做 policy augmentation CE；
   - 用 logits consistency 替代 witness posterior bound；
   - 允许 posterior head 读取 delta-t / mask / co-observation，验证院内升高但跨策略退化；
   - 去掉 semantic release，检查是否过度保守；
   - 去掉 ChronoFlow-style count/time/coobs 分层，只做随机 mask，验证真实 observation-process views 的必要性。

---

## 5. 预期创新性

1. **从结构防火墙转向证据见证泛化界**  
   历史方案大多把采样政策映射到某个结构对象再隔离。本提案直接约束分类器依赖的局部证据 witness 在生成策略环境下的 PAC-Bayes 泛化复杂度。

2. **兼顾解释性与鲁棒性**  
   i-borf 的可解释 receptive-field word 被升级为深度 witness posterior；输出仍能定位到局部形态和真实时间支持，而不是只给隐藏表示。

3. **ChronoFlow 不作为分类器，而作为策略环境采样器**  
   ChronoFlow-style 层级 observation process 用于构造 policy stress environments，帮助估计 worst-policy risk 与 witness coverage。

4. **不是普通多视图一致性**  
   不强迫所有策略视图 logits 完全相同；真实语义新息可以改变 posterior。约束对象是 witness posterior 的 PAC-Bayes risk-complexity tradeoff。

5. **避免把 informative sampling 全部删除**  
   policy audit branch 保留采样制度信息用于偏移告警和拒识；分类主路径只读取通过 value-supported witness posterior 证明具有低策略复杂度的局部证据。

## 6. 一句话投稿卖点

**Do-PAC Receptive Witness Fields 把非规则采样策略偏移下的分类证据改写为一组可解释的 time-weighted local witnesses，并利用 ChronoFlow 风格的观测过程反事实视图优化 PAC-Bayes 跨策略泛化界：模型不是简单删除采样信息或强迫 logits 一致，而是证明自己依赖的局部病理证据在不同采样制度下仍具有低复杂度、足覆盖、可解释的最坏策略风险。**
