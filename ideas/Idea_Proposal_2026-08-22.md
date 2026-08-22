# Title: Do-Policy Privacy Cloak：面向采样策略偏移的反事实采样差分隐私披风

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代总结文件。
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
- 已读取自动化记忆 `MEMORIES.md` 及其中全部未落盘/额外历史提案摘要：
  - `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`
  - `idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`
  - `idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-05.md`、`idea_2026-08-06.md`
  - `idea_2026-08-07.md`、`idea_2026-08-08.md`、`idea_2026-08-09.md`
  - `idea_2026-08-10.md`、`idea_2026-08-11.md`、`idea_2026-08-21.md`
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-08-02.md`

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
27. observation-resolution RG beta flow、semigroup closure、irrelevant policy-operator decay、fixed-point foreseeing。
28. event-time vs record-time causal curtain、anti-retrocausal margin、latency sidecar、bitemporal foreseeing。
29. Kaczmarz / ART clinical tomography、observation-ray design matrix、ray leverage capping、angle coverage logdet。
30. matched policy risk sets、conditional logistic likelihood、elementary symmetric polynomial nuisance cancellation。
31. RKHS cubature moment exactness、control-variate cubature、policy-word signatures、policy thermodynamics、copula rank stripping、queue debt neutralization。

本提案选择新的正交切入点：**不把采样政策当作要估计的概率、要删除的残差、要投票的 juror、要校准的集合、要证明的前提、要修复的信道错误、要粗粒化的尺度流、要匹配的 risk set 或要重建的 tomography ray；而是把采样政策当作对下游部署不应泄漏的敏感属性。模型先从 TCF 式病理分箱中抽取状态表示，再通过一个可学习的高斯差分隐私披风释放给分类器，要求同一潜在病程在相邻采样政策下的释放分布具有受控 Renyi privacy leakage。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-08-02.md` 中的 **PULSE** 与 **TCF** 给当前 AAAI 投稿方向提供了一个新的切入点。

第一，**PULSE** 说明真实部署中的 ICU time-series classification 要面对 HiRID / MIMIC-IV / eICU 这类跨中心环境。跨中心差异不仅是输入分布变化，更是采样政策、护理流程、变量 schema、记录习惯和 value-pending 延迟的共同变化。如果一个表示能被轻易反推出“这是哪个中心的采样日历 / 这是 alarm-dense 还是 routine-round / 这是 panel 同步还是异步拆单”，那么分类器即使没有显式输入 hospital id，也可能通过表示中的政策指纹获得 shortcut。

第二，**TCF** 的 Pathology-Focused Binning 与 Time-Conditioned Foreseeing 提醒我们，EHR 数值应先进入有临床意义的病理区间；但 TCF 的未来事件预测仍可能把 patient state 与 care process 混在一起。换言之，我们既需要 TCF 式病理语义，又要限制病理表示向分类器泄漏采样政策。

历史方案已经尝试过很多处理采样政策的方式：危险率、后验商、停时鞅、拓扑审查、gauge、纠错码、knockoff、evidential vacuity、conformal sleeves、proof、poset clock、IRT-DIF、RG fixed point、bitemporal curtain、Kaczmarz tomography 和 matched risk-set likelihood。本轮换一个信息安全视角：

> 采样政策偏移可以被表述为“状态表示对采样政策不够私密”。如果同一潜在病程在两个相邻采样政策下产生的分类表示可被高置信地区分，那么分类头就有空间把政策身份当作类别证据。因此鲁棒分类器不只是要学到病理状态，还要把病理状态以一个对采样政策差分隐私的方式发布给分类头。

**Do-Policy Privacy Cloak (DPPC)** 的核心直觉：

1. value process 将不规则观测编译成 TCF-style pathology tokens。
2. sampling process 只生成“相邻采样政策”的 counterfactual neighbors，例如 routine/alarm、panel split、pending、cross-center exposure。
3. encoder 输出一个 state mean `mu_state`，再由 Privacy Cloak 释放随机表示：

```text
z_release = mu_state + sigma_policy * epsilon, epsilon ~ Normal(0, I)
```

4. 对任意相邻政策 `p` 与 `p'`，要求释放分布满足 Renyi-DP 风格上界：

```text
D_alpha( Q(z | do(p)) || Q(z | do(p')) ) <= epsilon_policy
```

5. 分类器只读 `z_release`，不读采样摘要、policy id、privacy noise scale 或 counterfactual recipe。

这样解决采样偏移的关键是：模型不是强行让所有采样视图表征相同，也不是用对抗器猜 policy；它允许同一病程在不同政策下的 state mean 有有限差异，但这个差异必须被高斯披风限制到“分类器难以利用政策身份”的隐私预算内。若某个预测必须依赖训练中心特有的采样日历，它会导致 privacy leakage 超预算；若预测来自病理值本身，模型可以用较小噪声保留分类效用。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从裸状态表示改为 Pathology State + Policy-Private Release

DPPC 可以包裹当前任意 irregular encoder，但推荐前端改成三层。

#### A. TCF-Style Pathology Tokenizer

每个事件 `(value_i, time_i, variable_i)` 先被转成病理语义 token：

```text
path_token_i = Embed(variable_i, pathology_bin_i, log(1 + delta_t_i), relative_time_i)
```

其中 pathology bin 借鉴 TCF 的密度/病理意义分箱。这里不构造固定 viva question，不做 IRT item response，不做 proof literal，不做 feasible hull facet，也不做 tomography ray。

#### B. State Mean Encoder

用 causal/event Transformer 或 GRU 得到病理状态均值：

```text
mu_state = StateEncoder({path_token_i})
```

`mu_state` 是分类前的未发布状态表示。它可以包含细粒度状态信息，但不能直接交给分类器；否则其中可能混入 PULSE-style center protocol 指纹。

#### C. Policy Privacy Cloak

采样支路只从观察坐标中估计两个量：

```text
policy_stress = Summary(times, vars, mask, panel, pending)
sigma_policy  = softplus(NoiseHead(policy_stress, sensitivity))
```

`sigma_policy` 不进入分类头，只决定发布噪声强度。发布表示为：

```text
z_release = mu_state + sigma_policy * epsilon
```

推理时可用均值近似或少量 Monte Carlo sample；训练时对 `z_release` 做 reparameterization。

关键区别：

- 不是 policy adversarial：没有环境判别器，也没有 gradient reversal。
- 不是 representation consistency：不要求两个 counterfactual view 的 `mu_state` 相同。
- 不是 randomized smoothing：不对多个政策样本平均预测，也不输出 certified radius。
- 不是 evidential uncertainty：`sigma_policy` 是隐私发布噪声，不是类别无知质量。
- 不是 conformal set：不输出预测集或覆盖率。

### 2.2 改 Loss：从去偏/一致性转向 Policy-Privacy Release Discipline

总目标：

```text
L = L_cls_private
  + lambda_priv * L_policy_renyi_privacy
  + lambda_util * L_pathology_utility
  + lambda_noise * L_noise_minimality
  + lambda_fore * L_private_pathology_foreseeing
```

#### A. Private Classification Loss `L_cls_private`

分类器只读发布后的表示：

```text
logits = Classifier(z_release)
L_cls_private = CE(logits, y)
```

若某个类别只能通过采样政策指纹识别，模型必须提高 `sigma_policy` 才能满足 privacy bound，从而该 shortcut 的分类效用会被噪声削弱。

#### B. Renyi Policy Privacy Loss `L_policy_renyi_privacy`

对同一事实样本，counterfactual sampler 生成相邻采样政策 `p'`，得到另一组状态均值 `mu_neighbor`。两个释放分布都是同方差对角高斯：

```text
Q_p  = Normal(mu_state, sigma_policy^2 I)
Q_p' = Normal(mu_neighbor, sigma_policy^2 I)
```

高斯机制的 Renyi divergence 有闭式近似：

```text
D_alpha(Q_p || Q_p') =
  alpha * ||mu_state - mu_neighbor||_2^2 / (2 * sigma_policy^2)
```

训练约束：

```text
L_policy_renyi_privacy =
  mean_neighbor relu(D_alpha - epsilon_policy)^2
```

这项不要求 `mu_state == mu_neighbor`，只要求“同一病程换相邻采样政策后，分类器可观察到的发布分布不能高泄漏政策身份”。

#### C. Pathology Utility Loss `L_pathology_utility`

为了避免 Privacy Cloak 靠无限噪声获得平凡隐私，增加病理效用保持：

```text
z_clean = stopgrad(mu_state)
z_priv  = z_release
L_pathology_utility = SmoothL1(UtilityHead(z_priv), UtilityHead(z_clean))
```

`UtilityHead` 只预测 TCF pathology bins / 当前窗口病理摘要，不预测 policy id、mask pattern 或中心标签。它不同于 proof、IRT、hull、tomography 或 foreseeing risk-set；只是保证私有发布后仍保留病理值语义。

#### D. Noise Minimality Loss `L_noise_minimality`

隐私噪声越大，分类越不稳定。DPPC 让噪声只在政策敏感样本上升高：

```text
target_sigma = stopgrad(sensitivity_norm / privacy_budget)
L_noise_minimality =
  SmoothL1(sigma_policy, target_sigma)
  + eta * mean(log_sigma^2)
```

直觉：如果 counterfactual neighbor 下 `mu_state` 变化很小，说明表示本身不泄漏政策，噪声可以小；如果变化大，说明政策身份进入了表示，就需要更强披风。

#### E. Private Pathology Foreseeing `L_private_pathology_foreseeing`

吸收 TCF 的未来时间条件思想，但只在私有发布表示上预测 **pathology outcome**，不预测 future observation administration：

```text
p_hat(bin, var | z_release, query_time) = ForeseeHead(z_release, query_time)
```

训练目标：

```text
L_private_pathology_foreseeing =
  CE(var_logits, target_var) + CE(bin_logits, target_bin)
```

这样保留 TCF 对不规则 EHR 时间语义的优势，同时避免模型为了预测“未来会被记录什么”而重新学习医院流程。

### 2.3 改 Dataloader：返回 Policy Neighbor Bank，而不是一致性增强样本

新增 `PolicyPrivacyNeighborCollator`，每个 batch 返回：

1. 原始事件：`event_value`、`event_time`、`event_var_id`、`event_mask`。
2. TCF pathology bin：`pathology_bin_id` 或可微 `bin_prob`。
3. 采样坐标摘要：变量覆盖、时间窗覆盖、panel 共现、burst 密度、value-pending 比例。
4. `policy_neighbor_bank`：与事实政策相邻、但保留同一潜在病程值语义的反事实邻居：
   - `routine_round_neighbor`：把事件吸附到固定查房时间。
   - `alarm_sparse_neighbor`：报警后密集复测改为稀疏复查。
   - `panel_split_neighbor`：同步 panel 拆成异步事件。
   - `pending_neighbor`：保留施测事件但隐藏部分 value。
   - `cross_center_exposure_neighbor`：模拟 PULSE 式中心变量覆盖变化。
5. `foreseeing_query_bank`：
   - TCF-style future time query；
   - 目标只包括 pathology bin / variable，不包括未来是否被记录。

关键区别：

- 不生成 contrastive positive pair。
- 不要求 logits、representation 或 risk 在多个 view 下相同。
- 不估计 hazard、density ratio、posterior quotient、DIF、risk-set intercept 或 tomography ray。
- 不用 policy-only branch 分类。
- 反事实邻居只用于估计表示对采样政策的 sensitivity，并据此约束差分隐私发布。

### 2.4 与 PULSE / TCF 的结合方式

- **来自 PULSE**：把 HiRID / MIMIC-IV / eICU 的中心流程差异解释为 policy-neighbor family。DPPC 可报告每个中心的 `privacy_leakage`、`required_sigma` 与 `private-worst-center AUPRC`，回答模型是否通过中心采样指纹做预测。
- **来自 TCF**：保留 Pathology-Focused Binning 与 future-time pathology queries，但不预测 future administration process；所有下游分类与 foreseeing 都在 `z_release` 上完成。
- **与采样解耦/反事实干预框架结合**：value process 产生病理状态均值，sampling process 产生相邻政策和敏感度，counterfactual intervention 提供 privacy neighbor，classifier 只读经过差分隐私披风发布的状态表示。

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


class PathologyTokenizer(nn.Module):
    """Convert irregular observations into TCF-style pathology tokens."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.5, 2.5, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_bins + 3, hidden_dim),
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

        token_x = torch.cat(
            [
                self.var_embed(var_id),
                bin_prob,
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
            ],
            dim=-1,
        )
        token_h = self.token_proj(token_x) * mask.unsqueeze(-1)
        return {
            "token_h": token_h,
            "bin_prob": bin_prob,
            "pathology_bin": bin_prob.argmax(dim=-1),
            "event_mask": mask,
        }


class PathologyStateEncoder(nn.Module):
    """Encode pathology tokens into an unreleased state mean."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.to_state = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, token_h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ctx, _ = self.context(token_h)
        pooled = masked_mean(ctx, mask, dim=1)
        return self.to_state(pooled)


class PolicyStressEncoder(nn.Module):
    """Summarize observation-policy coordinates for privacy noise only."""

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

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_onehot = F.one_hot(var_id, self.num_vars).to(time.dtype) * mask.unsqueeze(-1)
        var_rate = var_onehot.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        early = masked_mean((time_norm <= 0.33).to(time.dtype), mask, dim=1).unsqueeze(-1)
        middle = masked_mean(((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype), mask, dim=1).unsqueeze(-1)
        late = masked_mean((time_norm > 0.66).to(time.dtype), mask, dim=1).unsqueeze(-1)
        mean_gap = masked_mean(torch.log1p(delta_t), mask, dim=1).unsqueeze(-1)
        event_rate = mask.mean(dim=1, keepdim=True)

        close = (delta_t <= delta_t.mean(dim=1, keepdim=True).clamp_min(1e-6)).to(time.dtype)
        var_change = torch.zeros_like(mask)
        var_change[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(time.dtype)
        panel_like = masked_mean(close * var_change, mask, dim=1).unsqueeze(-1)

        pending = batch.get("value_pending", torch.zeros_like(mask))
        pending_rate = masked_mean(pending, mask, dim=1).unsqueeze(-1)
        burst_rate = masked_mean(close, mask, dim=1).unsqueeze(-1)
        form_size = mask.sum(dim=1, keepdim=True) / mask.size(1)

        stats = torch.cat(
            [early, middle, late, mean_gap, event_rate, panel_like, pending_rate, burst_rate, form_size],
            dim=-1,
        )
        return self.net(torch.cat([var_rate, stats], dim=-1))


class GaussianPrivacyCloak(nn.Module):
    """Release state embeddings through a learned Gaussian privacy mechanism."""

    def __init__(self, hidden_dim: int, min_sigma: float = 0.03, max_sigma: float = 2.0):
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.noise_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        mu_state: torch.Tensor,
        policy_h: torch.Tensor,
        sensitivity: torch.Tensor,
        sample: bool = True,
    ) -> dict:
        raw = self.noise_head(torch.cat([mu_state.detach(), policy_h, sensitivity.unsqueeze(-1)], dim=-1))
        sigma = F.softplus(raw) + self.min_sigma
        sigma = sigma.clamp(max=self.max_sigma)
        if sample:
            eps = torch.randn_like(mu_state)
            z_release = mu_state + sigma * eps
        else:
            z_release = mu_state
        return {"z_release": z_release, "sigma": sigma}


def gaussian_renyi_divergence(
    mean_a: torch.Tensor,
    mean_b: torch.Tensor,
    sigma: torch.Tensor,
    alpha: float = 8.0,
) -> torch.Tensor:
    """RDP divergence for equal-covariance diagonal Gaussian mechanisms."""
    sq_dist = (mean_a - mean_b).pow(2)
    var = sigma.pow(2).clamp_min(1e-6)
    return alpha * (sq_dist / (2.0 * var)).sum(dim=-1)


class PrivatePathologyForeseeHead(nn.Module):
    """Predict pathology bins from policy-private state releases."""

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

    def forward(self, z_release: torch.Tensor, query_time: torch.Tensor) -> dict:
        # query_time: [B, Q]
        z = z_release[:, None].expand(-1, query_time.size(1), -1)
        q = self.query(torch.cat([z, query_time.unsqueeze(-1)], dim=-1))
        return {"var_logits": self.var_head(q), "bin_logits": self.bin_head(q)}


class DoPolicyPrivacyCloak(nn.Module):
    """Sampling-policy robust classifier via policy-private pathology releases."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        hidden_dim: int,
        num_classes: int,
        privacy_epsilon: float = 1.0,
        renyi_alpha: float = 8.0,
    ):
        super().__init__()
        self.tokenizer = PathologyTokenizer(num_vars, num_bins, hidden_dim)
        self.state = PathologyStateEncoder(hidden_dim)
        self.policy = PolicyStressEncoder(num_vars, hidden_dim)
        self.cloak = GaussianPrivacyCloak(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )
        self.foresee = PrivatePathologyForeseeHead(hidden_dim, num_vars, num_bins)
        self.privacy_epsilon = privacy_epsilon
        self.renyi_alpha = renyi_alpha
        self.num_bins = num_bins
        self.num_vars = num_vars

    def encode_mu(self, batch: dict) -> dict:
        token = self.tokenizer(batch)
        mu = self.state(token["token_h"], token["event_mask"])
        policy_h = self.policy(batch)
        return {**token, "mu_state": mu, "policy_h": policy_h}

    def forward(self, batch: dict, sample: bool = True) -> dict:
        enc = self.encode_mu(batch)
        neighbor_bank = batch.get("policy_neighbor_bank", [])
        if neighbor_bank:
            neighbor_mu = []
            for neighbor in neighbor_bank:
                neighbor_mu.append(self.encode_mu(neighbor)["mu_state"])
            neighbor_mu = torch.stack(neighbor_mu, dim=1)
            sensitivity = (neighbor_mu - enc["mu_state"][:, None]).norm(dim=-1).amax(dim=1)
        else:
            neighbor_mu = enc["mu_state"][:, None]
            sensitivity = torch.zeros(enc["mu_state"].size(0), device=enc["mu_state"].device)

        release = self.cloak(enc["mu_state"], enc["policy_h"], sensitivity, sample=sample)
        logits = self.classifier(release["z_release"])
        return {**enc, **release, "neighbor_mu": neighbor_mu, "sensitivity": sensitivity, "logits": logits}

    def privacy_loss(self, out: dict) -> torch.Tensor:
        if out["neighbor_mu"].size(1) == 0:
            return torch.zeros((), device=out["logits"].device)
        divs = []
        for idx in range(out["neighbor_mu"].size(1)):
            div = gaussian_renyi_divergence(
                out["mu_state"],
                out["neighbor_mu"][:, idx].detach(),
                out["sigma"],
                alpha=self.renyi_alpha,
            )
            divs.append(div)
        rdp = torch.stack(divs, dim=1)
        return F.relu(rdp - self.privacy_epsilon).pow(2).mean()

    def pathology_utility_loss(self, out: dict) -> torch.Tensor:
        clean_target = self.utility_head(out["mu_state"].detach())
        private_pred = self.utility_head(out["z_release"])
        return F.smooth_l1_loss(private_pred, clean_target)

    def noise_minimality_loss(self, out: dict) -> torch.Tensor:
        target_sigma = (out["sensitivity"] / (self.privacy_epsilon ** 0.5 + 1e-6)).unsqueeze(-1)
        target_sigma = target_sigma.clamp(min=self.cloak.min_sigma, max=self.cloak.max_sigma).detach()
        return F.smooth_l1_loss(out["sigma"], target_sigma.expand_as(out["sigma"])) + 0.001 * out["sigma"].pow(2).mean()

    def private_foreseeing_loss(self, batch: dict, out: dict) -> torch.Tensor:
        if "query_time" not in batch:
            return torch.zeros((), device=out["logits"].device)
        pred = self.foresee(out["z_release"], batch["query_time"])
        target_var = batch["query_target_var"].clamp(0, self.num_vars - 1)
        target_bin = batch["query_target_bin"].clamp(0, self.num_bins - 1)
        var_loss = F.cross_entropy(pred["var_logits"].flatten(0, 1), target_var.flatten())
        bin_loss = F.cross_entropy(pred["bin_logits"].flatten(0, 1), target_bin.flatten())
        return var_loss + bin_loss

    def training_loss(
        self,
        batch: dict,
        lambda_priv: float = 0.35,
        lambda_util: float = 0.10,
        lambda_noise: float = 0.05,
        lambda_fore: float = 0.20,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch, sample=True)
        cls_loss = F.cross_entropy(out["logits"], labels)
        priv_loss = self.privacy_loss(out)
        util_loss = self.pathology_utility_loss(out)
        noise_loss = self.noise_minimality_loss(out)
        fore_loss = self.private_foreseeing_loss(batch, out)

        total = (
            cls_loss
            + lambda_priv * priv_loss
            + lambda_util * util_loss
            + lambda_noise * noise_loss
            + lambda_fore * fore_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "policy_renyi_privacy_loss": priv_loss.detach(),
            "pathology_utility_loss": util_loss.detach(),
            "noise_minimality_loss": noise_loss.detach(),
            "private_pathology_foreseeing_loss": fore_loss.detach(),
            "mean_sigma": out["sigma"].mean().detach(),
            "mean_policy_sensitivity": out["sensitivity"].mean().detach(),
        }


@torch.no_grad()
def build_policy_privacy_neighbors(batch: dict) -> list[dict]:
    """Create adjacent sampling-policy neighbors for privacy auditing.

    These neighbors are not contrastive positives and are not used for logits
    consistency. They estimate how much unreleased state would reveal sampling
    policy identity.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = value.shape
    device = value.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, pending=None):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        if pending is not None:
            out["value_pending"] = pending
        out.pop("policy_neighbor_bank", None)
        return out

    neighbors = []

    # Routine-round neighbor: change calendar coordinates but keep observed values.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    neighbors.append(clone_with(value * mask, rounded_time, var_id, mask))

    # Alarm-sparse neighbor: thin dense late observations.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_sparse_mask = torch.where(late > 0, mask * alternating, mask)
    neighbors.append(clone_with(value * alarm_sparse_mask, time, var_id, alarm_sparse_mask))

    # Panel-split neighbor: weaken near-synchronous cross-variable observations.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_split_mask = mask * (1.0 - 0.5 * close * changed_var)
    neighbors.append(clone_with(value * panel_split_mask, time, var_id, panel_split_mask))

    # Pending neighbor: administration is visible but selected values are hidden.
    pending = mask
    pending_value = torch.zeros_like(value)
    neighbors.append(clone_with(pending_value, time, var_id, mask, pending=pending))

    # Cross-center exposure neighbor: simulate variable coverage schema change.
    odd_var = (var_id % 2 == 1).to(mask.dtype)
    exposure_mask = mask * (1.0 - 0.4 * odd_var)
    neighbors.append(clone_with(value * exposure_mask, time, var_id, exposure_mask))

    return neighbors
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cross-center privacy shift`：借鉴 PULSE，在 HiRID / MIMIC-IV / eICU 风格中心之间迁移，报告 policy identity probe 是否仍能从 `z_release` 识别中心。
   - `routine-vs-alarm privacy shift`：训练中 alarm-dense，测试中 routine-round，检查 Renyi leakage 是否与 worst-center error 相关。
   - `panel-split privacy shift`：训练中心同步 panel，测试中心异步拆单，检查未发布 `mu_state` 与发布后 `z_release` 的政策可辨性差距。
   - `value-pending privacy shift`：只保留采样事件但隐藏 value，检查模型是否把“已下单/待返回”通过状态表示泄漏给分类器。
   - `TCF future-process shift`：训练中心 future event 记录完整，测试中心记录延迟；DPPC 只做 private pathology foreseeing，不预测 future administration。

2. **对比方法**
   - 普通 TCF / EHR foundation model。
   - PULSE 中 LightGBM、传统深度模型和 LLM prompt / hybrid baseline。
   - STAR-Set、VP-GNN、MTM、QuITE、MVC-CDE 等 irregular / asynchronous baseline。
   - 普通 policy adversarial baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、D-IVSP、DJRT、KPMA、DVNB、DSPP、DCPD、DFFH、DIPF、DRG-SFF、DBCC、DKCT、DRNC 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-center worst-policy AUROC / AUPRC。
   - policy leakage probe AUC：从 `mu_state` 与 `z_release` 分别预测采样政策/中心的能力，要求发布后显著下降。
   - Renyi privacy budget violation rate：`D_alpha > epsilon_policy` 的样本比例。
   - privacy-utility frontier：不同 `epsilon_policy` 下的分类性能与 leakage trade-off。
   - required sigma by center：哪个中心/政策需要更强披风，解释采样流程 shortcut。
   - private pathology foreseeing accuracy：私有发布后仍能预测 pathology bin 的能力。

4. **消融实验**
   - 去掉 `L_policy_renyi_privacy`，检查发布表示是否重新暴露中心/采样政策。
   - 去掉 `L_pathology_utility`，检查模型是否靠过大噪声获得平凡隐私并损失病理语义。
   - 去掉 `L_noise_minimality`，检查 `sigma_policy` 是否无约束膨胀。
   - 将 policy neighbor bank 替换为随机 mask，验证收益来自结构化相邻采样政策，而不是普通数据增强。
   - 直接用 `mu_state` 分类作为反例，验证 policy leakage 与跨中心退化。
   - 让 `policy_h` 进入分类头作为反例，验证采样政策敏感属性泄漏的危害。

## 5. 预期创新性

1. **从采样去偏转向采样政策差分隐私发布**：不删除、不投票、不证明、不修复、不做后验商或 risk-set cancellation，而是把分类前状态表示作为需要对采样政策私密发布的对象。
2. **从简单对抗不可辨转向闭式 Renyi privacy accountant**：不用 policy adversarial classifier，而是对相邻反事实采样政策下的高斯发布分布直接约束 Renyi divergence。
3. **从 TCF 事件预测转向 private pathology foreseeing**：保留 TCF 病理分箱与未来时间查询，但只预测病理值语义，不预测未来观测流程。
4. **从 PULSE 跨中心结果转向 leakage 解释**：不仅报告跨中心性能，还能说明某个中心的退化是否来自 `mu_state` 对采样政策身份泄漏过强。
5. **与反事实采样框架低侵入兼容**：counterfactual sampler 只需生成相邻 policy neighbors；无需 hazard、density ratio、knockoff、conformal、IRT、RG、Kaczmarz 或 matched risk-set 结构。
6. **部署可调**：`epsilon_policy` 成为明确旋钮。高风险医疗部署可用更小隐私预算降低政策捷径；资源有限场景可放宽预算换取精度。

## 6. 一句话投稿卖点

**DPPC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“分类状态表示对采样政策身份泄漏过强”的问题，通过 TCF-style pathology tokenizer、counterfactual policy-neighbor bank、Gaussian Privacy Cloak 与 Renyi policy-privacy loss，让分类器只能读取对医院 routine/alarm、panel、pending 和 cross-center exposure 差异具有差分隐私保护的病理状态发布，从而在保留病理 foreseeing 语义的同时抑制采样政策捷径。**
