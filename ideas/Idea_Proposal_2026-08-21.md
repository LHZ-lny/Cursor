# Title: Do-RiskSet Nuisance Canceller：面向采样策略偏移的反事实匹配风险集条件似然分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代工作总结文件。
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
- 已读取自动化记忆 `MEMORIES.md` 与其中保存的额外历史提案摘要：
  - `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`
  - `idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`
  - `idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-05.md`、`idea_2026-08-06.md`
  - `idea_2026-08-07.md`、`idea_2026-08-08.md`、`idea_2026-08-09.md`
  - `idea_2026-08-10.md`、`idea_2026-08-11.md`
- 已读取 `paper_daily.md`，其近期前沿机制覆盖 PULSE 与 TCF，并兼容记录 STAR-Set、VP-GNN、EHR-SPC、LLM4EHR 等上下文。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮永久避开以下核心机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、采样 score 零空间、hazard-driven resampling、do-risk variance。
8. 生理流算子与采样干预算子交换子、value/policy graph commutator、policy residual sink。
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
25. counterfactual conformal risk sleeves、sampling instruments / control-function residualization、Borda / rank tribunal、Krylov policy subspace、determinantal / Nystrom volume basis、tropical support routes。
26. fixed clinical viva question bank、pathology sequent proofs、disease-progress poset clock、feasible hull、IRT latent trait / DIF firewall。
27. observation-resolution RG coarse-graining ladder、RG beta function、irrelevant policy operator decay、fixed-point pathology foreseeing。
28. event-time / record-time bitemporal causal curtain、anti-retrocausal margin、latency sidecar。
29. observation-ray design matrix、differentiable Kaczmarz / ART clinical tomography、ray leverage capping、angle coverage logdet。

本提案选择新的正交切入点：**不再试图删除、折扣、纠错、证明、校准或几何化采样信息；也不估计采样概率、密度比、后验商或危险率。我们把采样政策视为一个会给每个 matched stratum 添加任意类别截距的 nuisance term，然后通过“匹配风险集条件似然”在训练目标中把该截距条件消去。采样分支只负责构造同采样政策风险集；分类头只能凭同风险集内部的病理值差异胜出。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 中近期最值得与当前“采样解耦/反事实干预”框架结合的两个信号仍然是 **PULSE** 与 **TCF**。

第一，**PULSE** 把 ICU time-series classification 放入 HiRID / MIMIC-IV / eICU 跨中心环境，说明真实 sampling-policy shift 经常不是单点随机缺失，而是中心级护理流程、变量 schema、采样频率、value-pending 和记录习惯共同变化。很多模型的失败并不是因为它完全不会处理不规则时间戳，而是因为它学到了某个中心的“政策截距”：

```text
class score = pathology value score + center/protocol-specific sampling intercept
```

例如同样的 lactate abnormality，在一个中心是常规筛查项，在另一个中心只在高度怀疑时出现。若模型把“该项出现”或“某个 panel 共现”转成类别截距，跨中心后这个截距会直接失效。

第二，**TCF** 的 Pathology-Focused Binning 和 Time-Conditioned Foreseeing 提醒我们：EHR 数值应被翻译成病理语义，而不是被当作普通连续 token。但是 TCF 式未来事件预测仍可能混合 patient state 与 care process。为了保留 TCF 的病理值语义，同时避免 future observation process 变成类别捷径，本提案让 TCF-style pathology bins 只服务 **value score** 与 **pathology foreseeing head**，而采样日历、变量覆盖、pending、panel 和记录密度只服务 **risk-set matching**。

**Do-RiskSet Nuisance Canceller (DRNC)** 的核心直觉来自流行病学 matched case-control / conditional logistic likelihood：

> 如果一个风险集内的样本拥有近似相同的采样政策签名，那么任何只由采样政策诱导的类别偏置，在该风险集内都像共同截距。对该风险集的标签计数做条件化后，共同截距会从 likelihood 中消失。于是模型若想在同一风险集内分对类别，必须依赖观测值中的病理差异，而不能依赖采样政策本身。

这与历史方案的差异很关键：

- 不要求不同采样视图 logits 一致；
- 不估计 `p(sample | state)`、density ratio、hazard 或 posterior quotient；
- 不构造 knockoff、conformal set、jury、proof、IRT item、RG ladder、tomography rays 或 error-correcting code；
- 不把 sampling signature 输入分类头；
- 采样分支只做一件事：把样本放进“同政策、异病理”的 matched risk set，让条件似然自动消掉政策 nuisance intercept。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：Pathology-Value Encoder + Policy Signature Matcher 双通道

DRNC 使用两个严格分工的前端。

#### A. Pathology-Value Encoder：进入分类头

每个不规则事件 `(value_i, time_i, var_i)` 先经过 TCF 启发的病理分箱：

```text
bin_i = PathologyFocusedBin(value_i, var_i)
atom_i = Embed(var_i, bin_i, value_i, delta_t_i)
h_value = EventEncoder({atom_i})
score_y = Classifier_y(h_value)
```

`h_value` 和 `score_y` 是唯一进入分类决策的通道。它保留 TCF 的病理语义，但不预测“医院未来会记录什么事件”作为类别证据。

#### B. Policy Signature Matcher：只构造风险集，不进入分类头

采样分支只看观测过程坐标：

```text
p_sig = g_policy(times, var_ids, mask, panel_id, pending_flag)
```

`p_sig` 包含：

- 早/中/晚时间窗事件密度；
- 变量覆盖率与变量组覆盖率；
- panel-like 近同步共现率；
- value-pending 或延迟返回比例；
- 平均 gap、最大 gap、burst 率；
- 可选中心/设备的 label-free 结构摘要。

`p_sig` 不进入 classifier，也不被训练成策略分类器。它只用于在 dataloader 内构造 matched risk set。

### 2.2 改 Dataloader：从 policy view 增强改为 Matched Risk-Set Bank

新增 `RiskSetMatchingCollator`。对每个 anchor 样本 `i`，在 batch / memory queue / 跨中心缓存中寻找 `K` 个样本组成风险集 `R_i`：

```text
R_i = {j : distance(p_sig_j, p_sig_i) small}
```

同时保证：

1. **政策接近**：`p_sig` 距离小，采样政策 nuisance 在组内近似共同截距。
2. **标签有变化**：风险集内不能全是同一类，否则条件似然没有可学习信号。
3. **中心可混合**：优先跨 HiRID / MIMIC-IV / eICU 风格中心匹配，检验 PULSE 式跨中心政策偏移。
4. **反事实可加入**：当前反事实采样模块可以生成 routine、alarm、panel、pending 等 policy signatures，但只用于扩展 matching 候选，不用于 logits consistency。

风险集返回：

```text
riskset_indices: [B, R]
riskset_mask:    [B, R]
riskset_policy_distance: [B, R]
```

### 2.3 改 Loss：从不变性转向 Risk-Set Conditional Likelihood

总目标：

```text
L = L_cls
  + lambda_cond * L_matched_conditional
  + lambda_bal  * L_riskset_balance
  + lambda_fore * L_pathology_foreseeing
```

#### A. 标准分类损失 `L_cls`

事实样本仍用普通交叉熵训练，保证模型保持直接分类能力：

```text
L_cls = CE(score(x_i), y_i)
```

#### B. 匹配风险集条件似然 `L_matched_conditional`

设风险集 `R` 内第 `j` 个样本对类别 `c` 的 value score 为 `s_{j,c}`。若采样政策只给类别 `c` 添加共同 nuisance 截距 `b_{R,c}`，则：

```text
raw score_{j,c} = s_{j,c} + b_{R,c}
```

在风险集内条件化类别 `c` 的阳性数量 `n_c` 后，`b_{R,c}` 从分子和分母中同时抵消。二分类 conditional likelihood 的多类 one-vs-rest 版本为：

```text
P(Y_c = observed positives | n_c, R)
  = exp(sum_{j in positives(c)} s_{j,c})
    / e_{n_c}(exp(s_{1,c}), ..., exp(s_{|R|,c}))
```

其中 `e_{n}` 是 elementary symmetric polynomial。对应损失：

```text
L_cond =
  - sum_c [
      sum_{j: y_j=c} s_{j,c}
      - log e_{n_c}(exp(s_{1,c}), ..., exp(s_{|R|,c}))
    ]
```

这一步是 DRNC 的核心创新：**不估计采样偏置有多大，而是通过条件化标签计数让任意组内共同采样截距自动消失**。若模型想在同一个采样政策风险集内分辨真实类别，它必须使用 pathology-value score，而不能使用 policy signature。

#### C. Risk-Set Balance `L_riskset_balance`

匹配质量必须可审计。对每个风险集施加轻量平衡约束：

```text
L_balance =
  mean_R mean_j distance(p_sig_j, p_sig_anchor)^2
  + relu(min_label_entropy - entropy(label_histogram_R))^2
```

该项主要约束 dataloader / sampler 质量，不让风险集变成随意 batch。它不是对抗学习，也不是 policy-invariance loss；它只是保证 conditional likelihood 的“共同 nuisance 截距”假设更可信。

#### D. Pathology-Only Foreseeing `L_pathology_foreseeing`

吸收 TCF 的时间条件预训练，但只预测病理分箱，不预测 observation administration：

```text
p_hat(var, pathology_bin | h_value, query_time)
L_fore = CE(p_hat, observed_future_pathology_bin)
```

它给 `h_value` 提供临床语义锚点，避免 value encoder 只为 matched likelihood 学局部排序。future event 是否会被医院记录、是否 pending、是否 panel 出现，不作为 foreseeing target。

### 2.4 推理阶段

推理时不需要风险集，也不需要采样政策标签：

1. 输入事实不规则事件流；
2. 经过 Pathology-Value Encoder 得到 `h_value`；
3. 分类头输出 logits；
4. 可选诊断输出 `nearest_training_policy_distance` 与 `riskset_support_count`，提示当前样本是否落在训练过的 policy stratum 附近。

关键点：risk-set conditional likelihood 是训练时的 nuisance-canceling discipline；部署时模型是普通单样本 classifier，不依赖其他患者样本。

### 2.5 与 PULSE / TCF / 采样解耦框架的结合方式

- **来自 PULSE**：HiRID / MIMIC-IV / eICU 被视为不同采样政策的自然来源；训练时跨中心构造 matched risk sets，使中心级 sampling intercept 在条件似然中被抵消。
- **来自 TCF**：保留 pathology-focused bin 与 future-time query，但 foreseeing 只绑定病理值语义，不绑定未来记录流程。
- **与采样解耦/反事实干预框架结合**：
  - value process：产生 pathology atoms 与 class scores；
  - sampling process：产生 `p_sig`，只服务 risk-set matching；
  - counterfactual intervention：生成额外 policy signatures 和匹配候选，不生成一致性 pair；
  - classifier：只读取 value representation。

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


class PathologyValueEncoder(nn.Module):
    """Encode irregular values into pathology-semantic state scores."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.5, 2.5, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.atom_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_bins + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.pool = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * mask.unsqueeze(-1)

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        atom_x = torch.cat(
            [
                self.var_embed(var_id),
                bin_prob,
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
            ],
            dim=-1,
        )
        atom_h = self.atom_proj(atom_x) * mask.unsqueeze(-1)
        ctx, _ = self.context(atom_h)
        pooled = self.pool(masked_mean(ctx, mask, dim=1))
        return {
            "value_state": pooled,
            "event_context": ctx,
            "bin_prob": bin_prob,
            "pathology_bin": bin_prob.argmax(dim=-1),
        }


class PolicySignatureEncoder(nn.Module):
    """Summarize observation policy for matching only, never for class logits."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        pending = batch.get("value_pending", torch.zeros_like(mask))
        panel_id = batch.get("panel_id", torch.full_like(var_id, -1))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_onehot = F.one_hot(var_id, self.num_vars).to(time.dtype) * mask.unsqueeze(-1)
        var_rate = var_onehot.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        early = (time_norm <= 0.33).to(time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype)
        late = (time_norm > 0.66).to(time.dtype)
        close_gap = (delta_t <= masked_mean(delta_t, mask, dim=1).unsqueeze(-1).clamp_min(1e-6)).to(time.dtype)
        changed_var = torch.zeros_like(mask)
        changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(time.dtype)

        stats = torch.cat(
            [
                mask.mean(dim=1, keepdim=True),
                masked_mean(early, mask, dim=1).unsqueeze(-1),
                masked_mean(middle, mask, dim=1).unsqueeze(-1),
                masked_mean(late, mask, dim=1).unsqueeze(-1),
                masked_mean(torch.log1p(delta_t), mask, dim=1).unsqueeze(-1),
                delta_t.masked_fill(mask == 0, 0.0).amax(dim=1, keepdim=True),
                masked_mean(close_gap * changed_var, mask, dim=1).unsqueeze(-1),
                masked_mean(pending, mask, dim=1).unsqueeze(-1),
                ((panel_id >= 0).to(time.dtype) * mask).mean(dim=1, keepdim=True),
            ],
            dim=-1,
        )
        return F.normalize(self.net(torch.cat([var_rate, stats], dim=-1)), dim=-1)


def log_elementary_symmetric(scores: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Compute log elementary symmetric polynomials over active items.

    Args:
        scores: [B, R], log weights.
        active: [B, R], binary risk-set mask.
    Returns:
        log_e: [B, R + 1], where log_e[:, k] = log e_k(exp(scores)).
    """

    bsz, set_size = scores.shape
    neg_inf = torch.full((bsz,), -1e9, device=scores.device, dtype=scores.dtype)
    log_e = [torch.zeros(bsz, device=scores.device, dtype=scores.dtype)]
    log_e.extend([neg_inf.clone() for _ in range(set_size)])

    masked_scores = scores.masked_fill(active == 0, -1e9)
    for item_idx in range(set_size):
        item_score = masked_scores[:, item_idx]
        new_log_e = [part.clone() for part in log_e]
        item_active = active[:, item_idx] > 0
        for k in range(item_idx + 1, 0, -1):
            candidate = log_e[k - 1] + item_score
            new_value = torch.logaddexp(log_e[k], candidate)
            new_log_e[k] = torch.where(item_active, new_value, log_e[k])
        log_e = new_log_e
    return torch.stack(log_e, dim=1)


def matched_conditional_likelihood_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    riskset_indices: torch.Tensor,
    riskset_mask: torch.Tensor,
) -> torch.Tensor:
    """One-vs-rest conditional likelihood within matched policy risk sets.

    Conditioning on class counts cancels arbitrary class-specific intercepts
    shared by all samples in the same matched sampling-policy stratum.
    """

    bsz, set_size = riskset_indices.shape
    num_classes = logits.size(-1)
    safe_idx = riskset_indices.clamp(0, logits.size(0) - 1)

    set_logits = logits[safe_idx]       # [B, R, C]
    set_labels = labels[safe_idx]       # [B, R]
    set_active = riskset_mask.to(logits.dtype)
    active_count = set_active.sum(dim=1)

    losses = []
    for cls_idx in range(num_classes):
        cls_scores = set_logits[:, :, cls_idx]
        positive = ((set_labels == cls_idx).to(logits.dtype) * set_active)
        n_pos = positive.sum(dim=1).long()
        informative = (n_pos > 0) & (n_pos < active_count.long())
        if not informative.any():
            continue

        log_e = log_elementary_symmetric(cls_scores, set_active)
        numerator = (cls_scores * positive).sum(dim=1)
        denominator = log_e.gather(1, n_pos[:, None]).squeeze(1)
        losses.append((denominator - numerator)[informative])

    if not losses:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return torch.cat(losses).mean()


class PathologyForeseeHead(nn.Module):
    """TCF-style pathology-bin foreseeing without predicting observation process."""

    def __init__(self, hidden_dim: int, num_vars: int, num_bins: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.query = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.var_head = nn.Linear(hidden_dim, num_vars)
        self.bin_head = nn.Linear(hidden_dim, num_bins)

    def forward(self, value_state: torch.Tensor, query_time: torch.Tensor) -> dict:
        state = value_state[:, None].expand(-1, query_time.size(1), -1)
        q = self.query(torch.cat([state, query_time.unsqueeze(-1)], dim=-1))
        return {"var_logits": self.var_head(q), "bin_logits": self.bin_head(q)}


class DoRiskSetNuisanceCanceller(nn.Module):
    """Sampling-policy robust classifier trained by matched risk-set likelihood."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.value_encoder = PathologyValueEncoder(num_vars, num_bins, hidden_dim)
        self.policy_signature = PolicySignatureEncoder(num_vars, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.foresee = PathologyForeseeHead(hidden_dim, num_vars, num_bins)
        self.num_vars = num_vars
        self.num_bins = num_bins

    def forward(self, batch: dict) -> dict:
        value = self.value_encoder(batch)
        logits = self.classifier(value["value_state"])
        # Detach by default: policy signatures organize the loss, not class evidence.
        policy_sig = self.policy_signature(batch).detach()
        return {**value, "logits": logits, "policy_signature": policy_sig}

    def riskset_balance_loss(self, out: dict, batch: dict) -> torch.Tensor:
        if "riskset_indices" not in batch:
            return torch.zeros((), device=out["logits"].device)
        idx = batch["riskset_indices"].clamp(0, out["logits"].size(0) - 1)
        mask = batch["riskset_mask"].to(out["logits"].dtype)
        sig = out["policy_signature"]
        anchor = sig[:, None, :]
        neighbor = sig[idx]
        distance = (neighbor - anchor).pow(2).sum(dim=-1)
        distance_loss = (distance * mask).sum() / mask.sum().clamp_min(1.0)

        labels = batch["labels"][idx]
        label_hist = []
        for cls_idx in range(out["logits"].size(-1)):
            label_hist.append(((labels == cls_idx).to(mask.dtype) * mask).sum(dim=1))
        hist = torch.stack(label_hist, dim=-1)
        prob = hist / hist.sum(dim=-1, keepdim=True).clamp_min(1.0)
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1)
        min_entropy = batch.get("riskset_min_entropy", torch.tensor(0.35, device=entropy.device))
        entropy_loss = F.relu(min_entropy - entropy).pow(2).mean()
        return distance_loss + 0.1 * entropy_loss

    def pathology_foreseeing_loss(self, batch: dict, out: dict) -> torch.Tensor:
        if "query_time" not in batch:
            return torch.zeros((), device=out["logits"].device)
        pred = self.foresee(out["value_state"], batch["query_time"])
        target_var = batch["query_target_var"].clamp(0, self.num_vars - 1)
        target_bin = batch["query_target_bin"].clamp(0, self.num_bins - 1)
        var_loss = F.cross_entropy(pred["var_logits"].flatten(0, 1), target_var.flatten())
        bin_loss = F.cross_entropy(pred["bin_logits"].flatten(0, 1), target_bin.flatten())
        return var_loss + bin_loss

    def training_loss(
        self,
        batch: dict,
        lambda_cond: float = 0.35,
        lambda_bal: float = 0.05,
        lambda_fore: float = 0.15,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)

        if "riskset_indices" in batch:
            cond_loss = matched_conditional_likelihood_loss(
                logits=out["logits"],
                labels=labels,
                riskset_indices=batch["riskset_indices"],
                riskset_mask=batch["riskset_mask"],
            )
        else:
            cond_loss = torch.zeros((), device=out["logits"].device)

        balance_loss = self.riskset_balance_loss(out, batch)
        fore_loss = self.pathology_foreseeing_loss(batch, out)
        total = cls_loss + lambda_cond * cond_loss + lambda_bal * balance_loss + lambda_fore * fore_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "matched_conditional_loss": cond_loss.detach(),
            "riskset_balance_loss": balance_loss.detach(),
            "pathology_foreseeing_loss": fore_loss.detach(),
        }


@torch.no_grad()
def build_policy_matched_risksets(
    labels: torch.Tensor,
    policy_signature: torch.Tensor,
    set_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build within-batch matched policy risk sets.

    Production code can extend this with a cross-batch memory queue and
    center-aware candidate pools. This sketch keeps the closest policy
    signatures while ensuring each anchor is included.
    """

    bsz = labels.size(0)
    device = labels.device
    dist = torch.cdist(policy_signature, policy_signature, p=2)

    all_indices = []
    all_masks = []
    for anchor in range(bsz):
        order = torch.argsort(dist[anchor])
        same_label = labels[order] == labels[anchor]
        different_label = ~same_label

        # Include the anchor, then alternate near-policy different-label and same-label samples.
        chosen = [anchor]
        for pool in [order[different_label], order[same_label]]:
            for idx in pool.tolist():
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= set_size:
                    break
            if len(chosen) >= set_size:
                break

        # If the batch is small or label variety is limited, pad with nearest remaining samples.
        for idx in order.tolist():
            if len(chosen) >= set_size:
                break
            if idx not in chosen:
                chosen.append(idx)

        valid = len(chosen)
        if valid < set_size:
            chosen = chosen + [anchor] * (set_size - valid)
        all_indices.append(torch.tensor(chosen[:set_size], device=device, dtype=torch.long))
        mask = torch.zeros(set_size, device=device, dtype=policy_signature.dtype)
        mask[: min(valid, set_size)] = 1.0
        all_masks.append(mask)

    return torch.stack(all_indices, dim=0), torch.stack(all_masks, dim=0)
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cross-center matched shift`：借鉴 PULSE，在 HiRID / MIMIC-IV / eICU 风格中心之间训练和测试，风险集跨中心匹配相似采样签名。
   - `routine-vs-alarm intercept shift`：训练中心中 alarm burst 与标签强相关，测试中心改为 routine sampling；检查条件似然是否减少政策截距依赖。
   - `panel-policy intercept shift`：训练中心同步 panel 是高风险 shortcut，测试中心常规 panel 化；风险集在 panel rate 上匹配。
   - `value-pending shift`：某中心化验已下单但值未返回的比例高；匹配 pending signature，分类只能依赖已返回病理值。

2. **对比方法**
   - 普通 TCF / EHR foundation model。
   - PULSE 中 LightGBM、传统深度模型与 LLM prompt / hybrid baseline。
   - STAR-Set、VP-GNN、MTM、MVC-CDE、QuITE 等 irregular / asynchronous baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、D-IVSP、DJRT、KPMA、DVNB、DSPP、DCPD、DIPF、DRG-SFF、DBCC、DKCT 等。

3. **核心指标**
   - in-policy accuracy / AUPRC。
   - cross-center worst-policy accuracy。
   - policy-only intercept leakage：只用 `p_sig` 预测标签的 AUROC，作为任务偏移强度诊断。
   - matched conditional gain：加入 `L_cond` 后 within-riskset ranking accuracy 的提升。
   - risk-set balance distance：风险集内采样签名均方距离。
   - cross-center matched ECE：按 policy stratum 分桶后的校准误差。
   - pathology foreseeing AUPRC：只预测病理 bins 的 TCF-style foreseeing 质量。

4. **消融实验**
   - 去掉 `L_matched_conditional`，只用普通 CE，检查跨中心政策截距是否重新污染分类。
   - 将 matched risk set 替换为随机 risk set，验证收益来自政策匹配而不是 batch 对比。
   - 让 `policy_signature` 直接拼入 classifier 作为反例，验证采样捷径会提升院内但损害跨中心。
   - 去掉 label-entropy balance，检查风险集全同类时条件似然是否失去约束。
   - 去掉 TCF-style pathology foreseeing，检查 value encoder 是否缺少病理语义锚。
   - 使用跨 batch memory queue vs 仅 batch 内 matching，评估大规模 PULSE 设置下的匹配质量。

## 5. 预期创新性

1. **从采样去偏转向条件似然消 nuisance**：不估计采样机制，不做对抗、密度比、后验除法或一致性；而是通过风险集条件化把组内共同采样截距数学上消掉。
2. **从 policy views 转向 matched risk sets**：反事实采样不再产生正样本或投票视图，而是产生政策签名和匹配候选，让模型在“同政策不同病理”的集合内学习。
3. **从跨中心 benchmark 转向可训练的截距抵消机制**：吸收 PULSE 的多中心偏移设置，但不止报告退化；直接把中心/协议造成的 nuisance intercept 放入条件似然框架。
4. **从 TCF future event 预测转向 pathology-only semantic anchor**：保留 TCF 的病理分箱和未来时间查询，但不预测 future observation administration，避免 care process 作为类别证据。
5. **与历史方案显著正交**：DRNC 不使用 proof、IRT、conformal、jury、knockoff、RG、tomography、bitemporal、gauge、topology、syndrome、martingale、density ratio 或 certified smoothing；它的核心是 matched conditional likelihood 的 nuisance-intercept cancellation。
6. **部署低侵入**：训练时使用风险集，推理时仍是单样本 classifier；适合嵌入现有 sampling decoupling / counterfactual intervention 代码库。

## 6. 一句话投稿卖点

**DRNC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“同一采样政策风险集内的类别 nuisance 截距污染”问题，通过 policy-signature matched risk sets 与 elementary-symmetric conditional likelihood，在不估计采样概率、不做一致性/对抗/校准/证明/纠错/几何分解的前提下，把医院协议、panel、pending、burst 等采样捷径从训练目标中条件消去，让分类器只能依赖同风险集内部真正区分类别的 pathology-value signal。**
