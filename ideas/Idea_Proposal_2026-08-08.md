# Title: Do-IRT Policy-DIF Firewall：面向采样策略偏移的反事实测量项功能差异防火墙

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*work*.md`、`**/*总结*.md`：当前工作区未发现可替代总结文件。
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
- 已读取自动化记忆 `MEMORIES.md` 与其中记录的全部未落盘/额外历史提案摘要：
  - `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`
  - `idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`
  - `idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-05.md`、`idea_2026-08-06.md`、`idea_2026-08-07.md`
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-08-02.md`
  - `paper_daily_2026-07-27.md`
  - `paper_daily_2026-07-26.md`

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
25. counterfactual conformal risk sleeves、policy-conditional nonconformity set、leave-policy-out conformal calibration。
26. counterfactual sampling instruments、first-stage structural equation、control-function residualized structure readout。
27. counterfactual policy jury、Borda / majority rank tribunal、no-dictator / no-structural-bribery loss。
28. Krylov policy subspace、annihilating polynomial、policy-mode annihilation。
29. determinantal / Nystrom state-volume basis、policy-overload logdet sobriety。
30. tropical support routes、soft-min bottleneck、route survival under policy。
31. fixed clinical viva question bank、canonical answer sheet、source-chart-to-answer translator。
32. pathology sequent proof bank、provable / unproved / refuted proof audit、availability sobriety。
33. disease-progress poset clock、progress-conditioned pathology foreseeing、order-ideal classifier。
34. pathology feasible hull / convex half-space facets、Chebyshev-like hull center、worst-case hull margin。

本提案选择新的正交切入点：**不把鲁棒性建成证明、校准、纠错、排序、拓扑、gauge、后验商、凸包、固定问答或集合契约，而是把不同采样政策看成同一病理潜变量上的不同“施测表单”。模型用 Item Response Theory (IRT) 学习病理潜变量，用 Differential Item Functioning (DIF) 防火墙识别某个观测项在特定采样政策下是否改变了难度/偏置。分类器只读 policy-DIF 校正后的 latent trait，而不是读 item 是否被施测、panel 是否共现或某中心是否更爱测某项。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-08-02.md` 中的 **PULSE** 和 **TCF** 给当前 AAAI 投稿方向提供了一个很自然但历史方案尚未使用的视角。

第一，**PULSE** 把 ICU time-series classification 放到 HiRID / MIMIC-IV / eICU 跨中心环境中比较。跨中心部署不只是输入分布改变，更像病人被不同医院做了不同“测验表单”：同一个潜在病程，在医院 A 会被常规测某些化验，在医院 B 只在告警后测，在医院 C 变量 schema 和记录密度又不同。若分类器直接把“测了 lactate”或“panel 同步出现”当成类别证据，它本质上是在记忆试卷表单，而不是估计病人 latent clinical trait。

第二，**TCF** 的 Pathology-Focused Binning 与 Time-Conditioned Foreseeing 提醒我们，EHR 中的数值应先转成具有病理语义的离散/序数区间；但 TCF 式未来事件预测仍可能混合 patient state 与 care process。也就是说，一个病理 bin 观测项的出现和取值，都可能受到采样政策影响：

- item exposure：某项是否被施测、何时被施测；
- item response：施测后观测到 normal / high / critical 的概率；
- item bias：同一潜在病理状态下，不同医院政策让某个 item 更容易显示异常或更容易被记录为 pending。

历史方案已经覆盖了很多强机制：危险率、后验商、鞅、保形、knockoff、纠错码、拓扑、gauge、jury、固定问答、证明、偏序时钟和凸包。本轮换一个测量学角度：

> 采样政策偏移不是简单让 token 缺失，而是改变了“哪些测量项被发给病人，以及这些测量项在不同中心是否保持同样的功能”。因此鲁棒分类器应先估计一个跨表单可比的 latent disease trait，再把 policy-induced item difficulty / discrimination shift 隔离为 DIF，而不是把 item 暴露模式当成类别捷径。

**Do-IRT Policy-DIF Firewall (DIPF)** 的核心思想：

1. 把每个不规则观测事件转成一个 TCF 风格 pathology item response，例如 `(variable=lactate, bin=critical, phase=early)`。
2. 用可微 IRT 估计病理潜变量 `theta`：同一 `theta` 下，不同 item 的 difficulty / discrimination 决定观测到某个 pathology bin 的概率。
3. 用 sampling branch 只估计 **policy-DIF offset** 与 **item administration propensity**，解释不同采样政策让某些 item 更容易出现或更容易异常。
4. 分类头只读取校正 DIF 后的 `theta` posterior，而不读取 item exposure、policy code、panel 共现或 value-pending 本身。

这与当前“采样解耦/反事实干预”框架兼容：

- value process 负责估计 item response 与 latent trait；
- sampling process 负责估计 item administration 与 DIF offset；
- counterfactual intervention 生成不同表单/施测政策，用于审计哪些 item 出现了 policy-specific DIF；
- classifier 只读跨表单可比的 `theta`。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从 event embedding 改为 Pathology Item Response Encoder

DIPF 不需要固定 clinical viva question bank，也不需要预设证明规则。它把真实观测事件动态编译成 item response：

1. **Pathology Itemizer**
   - 吸收 TCF 的 Pathology-Focused Binning：每个观测值根据变量特异的病理分箱得到 ordinal response `r_i in {0, ..., B-1}`。
   - item id 不是固定问答题，而是由 `(variable_id, coarse clinical phase, bin family)` 组成的可学习 item embedding。
   - `coarse phase` 来自相对住院进程或事件局部上下文，不是 learnable reference point，也不是固定 viva time anchor。

2. **Multi-Dimensional Latent Trait Posterior**
   - 用观测值和 item response 估计病理潜变量：

```text
q(theta | item responses) = Normal(mu_theta, diag(sigma_theta^2))
```

   - `theta` 可以有多维，例如 hemodynamic trait、inflammation trait、renal trait、respiratory trait。
   - 分类器只读 `theta` 的均值和方差。

3. **Policy-DIF Head**
   - sampling branch 只输入时间戳、变量可见性、panel 共现、value-pending、中心/recipe 描述等采样坐标。
   - 输出两个量：

```text
admin_logit_i = Pr(item_i administered | policy)
dif_offset_i  = item_i 在当前 policy 下的 difficulty shift
```

   - `admin_logit` 与 `dif_offset` 不进入分类头，只进入 item response likelihood 的校正项。

### 2.2 改 Loss：从一致性/证明/集合校准转向 IRT-DIF Discipline

总目标：

```text
L = L_cls
  + lambda_irt   * L_item_response
  + lambda_admin * L_administration
  + lambda_dif   * L_counterfactual_DIF
  + lambda_leak  * L_trait_policy_leakage
  + lambda_info  * L_form_information_honesty
```

#### A. 分类损失 `L_cls`

事实观测下，分类器只读 DIF 校正后的 latent trait posterior：

```text
logits = Classifier([mu_theta, logvar_theta])
L_cls = CE(logits, y)
```

#### B. Item Response Likelihood `L_item_response`

对每个观测 item，使用 ordinal IRT：

```text
score_i = a_item_i^T theta - b_item_i - dif_offset_i
Pr(r_i <= k | theta, item_i, policy) = sigmoid(cut_k - score_i)
```

其中：

- `a_item` 是 item discrimination；
- `b_item` 是 item base difficulty；
- `dif_offset` 是当前采样政策造成的 item 功能差异；
- 分类头不看 `dif_offset`，只看 `theta`。

直觉：如果某医院只有在高度怀疑时才测某项，模型可以把该政策对 item difficulty 的改变吸收到 `dif_offset`，而不是错误提高 `theta` 或类别 logit。

#### C. Administration Likelihood `L_administration`

sampling branch 学习“哪些 item 被施测”，但该信息只用于校正选择偏差：

```text
L_admin = BCE(admin_logit_i, observed_admin_i)
```

这不是 missingness pattern encoder 直接分类。item 是否出现不产生类别 logit，只告诉模型当前表单覆盖了哪些测量项、哪些 item response likelihood 应被观察。

#### D. Counterfactual DIF Moment `L_counterfactual_DIF`

反事实采样模块生成不同施测表单：

- `routine_form`：固定查房式 item 施测；
- `alarm_form`：告警后密集 item 施测；
- `panel_form`：多变量联测表单；
- `pending_form`：item 已施测但 response 未返回；
- `cross_center_form`：PULSE 风格跨中心 item exposure 迁移。

DIPF 不要求不同表单 logits 一致，也不要求 `theta` 完全一致。它只要求：在控制 `theta` 后，item residual 的系统性 policy shift 应被 DIF head 解释。

```text
residual_i = observed_response_i - E_IRT(response_i | theta, item_i, dif_i)
L_counterfactual_DIF =
  mean_policy || mean_item residual_i(policy) ||_2^2
```

若某个 policy 下残差均值仍偏移，说明 DIF head 没有把 item 功能差异吸收干净，分类器可能仍把政策差异误当 latent trait。

#### E. Trait-Policy Leakage `L_trait_policy_leakage`

防止 `theta` 偷偷编码表单/采样政策。使用非对抗的交叉协方差约束：

```text
L_trait_policy_leakage = || Corr(stopgrad(policy_summary), theta) ||_F^2
```

它不是 environment adversarial classifier，也不是梯度反转；只是限制 latent trait posterior 与采样表单摘要的线性泄漏。policy 信息仍然可以存在于 `dif_offset` 与 `admin_logit` 中。

#### F. Form Information Honesty `L_form_information_honesty`

不同表单提供的信息量不同。DIPF 不假装所有采样视图同样可靠，而是让 `theta` posterior 方差反映 item coverage：

```text
observed_info = sum_i ||a_item_i||^2 * admin_i
posterior_precision = exp(-logvar_theta)
L_form_information_honesty =
  SmoothL1(mean_dim posterior_precision, stopgrad(observed_info))
```

这区别于 evidential vacuity、conformal set 或 feasible hull：这里的不确定性来自测量学中的 test information，而不是主观逻辑、覆盖率或几何半径。

### 2.3 改 Dataloader：返回 Counterfactual Test Forms

新增 `PolicyDIFItemCollator`，每个 batch 返回：

1. 原始事件：`event_value`、`event_time`、`event_var_id`、`event_mask`。
2. 病理 item response：`item_id`、`ordinal_response`、`response_mask`。
3. 表单摘要：变量覆盖、时间窗覆盖、panel 共现、value-pending 比例、局部 burst 强度。
4. `counterfactual_form_bank`：
   - routine form；
   - alarm form；
   - panel batching / debatching form；
   - value-pending form；
   - cross-center exposure form。
5. `admin_target_bank`：各表单下哪些 item 被施测。

关键区别：

- 不生成 consistency pair。
- 不固定 viva question bank。
- 不做 proof / sequent / poset / hull。
- 不估计 hazard、density ratio、posterior quotient 或 conformal nonconformity。
- 不用 item exposure 直接分类；item exposure 只训练 administration 与 DIF 校正。

### 2.4 与 PULSE / TCF 的结合方式

- **来自 PULSE**：HiRID / MIMIC-IV / eICU 可视为不同 test form families；DIPF 直接报告跨中心 item DIF，解释哪个变量/时间窗在新中心不再等价。
- **来自 TCF**：Pathology-Focused Binning 提供 ordinal item response；Dual-Calendar / foreseeing 的启发转化为 item phase 和 future item administration，而不是直接预测未来事件作为类别证据。
- **与反事实框架结合**：counterfactual sampler 生成可解释施测表单，DIF head 学习表单诱导的 item 功能差异，classifier 只读 latent trait。

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


class PathologyItemizer(nn.Module):
    """Convert irregular observations into ordinal pathology item responses."""

    def __init__(self, num_vars: int, num_bins: int, num_phases: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.num_phases = num_phases
        self.num_items = num_vars * num_phases
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.phase_embed = nn.Embedding(num_phases, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.5, 2.5, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.item_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_bins + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        phase = torch.floor((time / horizon).clamp(0, 0.999) * self.num_phases).long()
        item_id = var_id * self.num_phases + phase

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * mask.unsqueeze(-1)
        ordinal_response = bin_prob.argmax(dim=-1)

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        item_h = self.item_proj(
            torch.cat(
                [
                    self.var_embed(var_id),
                    self.phase_embed(phase),
                    bin_prob,
                    value.unsqueeze(-1),
                    torch.log1p(delta_t).unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        item_h = item_h * mask.unsqueeze(-1)
        return {
            "item_id": item_id,
            "item_h": item_h,
            "bin_prob": bin_prob,
            "ordinal_response": ordinal_response,
            "response_mask": mask,
            "phase": phase,
        }


class PolicyFormEncoder(nn.Module):
    """Summarize the administered test form for DIF and administration modeling only."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 8, hidden_dim),
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
        early = ((time_norm <= 0.33).to(time.dtype) * mask).mean(dim=1, keepdim=True)
        middle = (((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype) * mask).mean(dim=1, keepdim=True)
        late = ((time_norm > 0.66).to(time.dtype) * mask).mean(dim=1, keepdim=True)
        mean_gap = masked_mean(torch.log1p(delta_t), mask, dim=1).unsqueeze(-1)
        burst = (delta_t <= delta_t.mean(dim=1, keepdim=True).clamp_min(1e-6)).to(time.dtype)
        burst_rate = masked_mean(burst, mask, dim=1).unsqueeze(-1)
        event_rate = mask.mean(dim=1, keepdim=True)
        pending = batch.get("value_pending", torch.zeros_like(mask))
        pending_rate = masked_mean(pending, mask, dim=1).unsqueeze(-1)
        panel_id = batch.get("panel_id", torch.full_like(var_id, -1))
        panel_rate = ((panel_id >= 0).to(time.dtype) * mask).mean(dim=1, keepdim=True)
        form_size = mask.sum(dim=1, keepdim=True) / mask.size(1)

        stats = torch.cat(
            [early, middle, late, mean_gap, burst_rate, event_rate, pending_rate, panel_rate, form_size],
            dim=-1,
        )
        # Keep exactly num_vars + 8 by dropping form_size if needed for older configs.
        stats = stats[:, :8]
        return self.net(torch.cat([var_rate, stats], dim=-1))


class LatentTraitPosterior(nn.Module):
    """Amortize a multi-dimensional clinical trait posterior from item responses."""

    def __init__(self, hidden_dim: int, trait_dim: int):
        super().__init__()
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.to_mu = nn.Linear(2 * hidden_dim, trait_dim)
        self.to_logvar = nn.Linear(2 * hidden_dim, trait_dim)

    def forward(self, item_h: torch.Tensor, mask: torch.Tensor) -> dict:
        ctx, _ = self.context(item_h)
        pooled = masked_mean(ctx, mask, dim=1)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(-6.0, 4.0)
        return {"theta_mu": mu, "theta_logvar": logvar, "item_context": ctx}


class IRTDIFHead(nn.Module):
    """Ordinal IRT likelihood with policy-specific DIF offsets."""

    def __init__(self, num_items: int, trait_dim: int, num_bins: int, form_dim: int):
        super().__init__()
        self.num_items = num_items
        self.num_bins = num_bins
        self.discrimination = nn.Embedding(num_items, trait_dim)
        self.difficulty = nn.Embedding(num_items, 1)
        self.cutpoints = nn.Parameter(torch.linspace(-2.0, 2.0, num_bins - 1))
        self.form_to_dif = nn.Sequential(
            nn.Linear(form_dim, form_dim),
            nn.SiLU(),
            nn.Linear(form_dim, 1),
        )
        self.form_to_admin = nn.Sequential(
            nn.Linear(form_dim, form_dim),
            nn.SiLU(),
            nn.Linear(form_dim, 1),
        )
        self.item_form_gate = nn.Embedding(num_items, form_dim)

    def item_score(self, theta: torch.Tensor, item_id: torch.Tensor, form_h: torch.Tensor) -> dict:
        discr = self.discrimination(item_id)
        discr = F.normalize(discr, dim=-1)
        base = self.difficulty(item_id).squeeze(-1)
        item_form = self.item_form_gate(item_id)
        gated_form = form_h[:, None, :] * torch.tanh(item_form)
        dif = self.form_to_dif(gated_form).squeeze(-1)
        admin_logit = self.form_to_admin(gated_form).squeeze(-1)
        score = (discr * theta[:, None, :]).sum(dim=-1) - base - dif
        return {"score": score, "dif_offset": dif, "admin_logit": admin_logit, "discrimination": discr}

    def ordinal_log_probs(self, score: torch.Tensor) -> torch.Tensor:
        # Cumulative ordinal model converted to class probabilities.
        cut = torch.sort(self.cutpoints)[0]
        cdf = torch.sigmoid(cut.view(1, 1, -1) - score.unsqueeze(-1))
        first = cdf[..., :1]
        middle = cdf[..., 1:] - cdf[..., :-1]
        last = 1.0 - cdf[..., -1:]
        prob = torch.cat([first, middle, last], dim=-1).clamp_min(1e-6)
        prob = prob / prob.sum(dim=-1, keepdim=True)
        return prob.log()

    def forward(self, theta: torch.Tensor, item_id: torch.Tensor, form_h: torch.Tensor) -> dict:
        scored = self.item_score(theta, item_id, form_h)
        scored["response_log_prob"] = self.ordinal_log_probs(scored["score"])
        return scored


class DoIRTPolicyDIFFirewall(nn.Module):
    """Policy-shift robust classifier based on IRT latent traits and DIF isolation."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        num_phases: int,
        hidden_dim: int,
        trait_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.itemizer = PathologyItemizer(num_vars, num_bins, num_phases, hidden_dim)
        self.form = PolicyFormEncoder(num_vars, hidden_dim)
        self.trait = LatentTraitPosterior(hidden_dim, trait_dim)
        self.irt = IRTDIFHead(
            num_items=num_vars * num_phases,
            trait_dim=trait_dim,
            num_bins=num_bins,
            form_dim=hidden_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * trait_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_bins = num_bins

    def forward(self, batch: dict) -> dict:
        item = self.itemizer(batch)
        form_h = self.form(batch)
        trait = self.trait(item["item_h"], item["response_mask"])
        irt = self.irt(trait["theta_mu"], item["item_id"], form_h)
        logits = self.classifier(torch.cat([trait["theta_mu"], trait["theta_logvar"]], dim=-1))
        return {**item, **trait, **irt, "form_h": form_h, "logits": logits}

    def item_response_loss(self, out: dict) -> torch.Tensor:
        target = out["ordinal_response"].clamp(0, self.num_bins - 1)
        logp = out["response_log_prob"].gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = -(logp * out["response_mask"]).sum() / out["response_mask"].sum().clamp_min(1.0)
        return loss

    def administration_loss(self, out: dict) -> torch.Tensor:
        target = out["response_mask"]
        raw = F.binary_cross_entropy_with_logits(out["admin_logit"], target, reduction="none")
        return (raw * target.clamp_min(0.1)).mean()

    def counterfactual_dif_loss(self, batch: dict, factual: dict) -> torch.Tensor:
        form_bank = batch.get("counterfactual_form_bank", [])
        if not form_bank:
            return torch.zeros((), device=factual["logits"].device)

        losses = []
        for form_batch in form_bank:
            cf = self.forward(form_batch)
            # Residual moment after DIF correction should not systematically shift by form.
            expected = torch.softmax(cf["response_log_prob"], dim=-1)
            response = F.one_hot(cf["ordinal_response"].clamp(0, self.num_bins - 1), self.num_bins).to(expected.dtype)
            residual = (response - expected) * cf["response_mask"].unsqueeze(-1)
            residual_mean = residual.sum(dim=1) / cf["response_mask"].sum(dim=1, keepdim=True).clamp_min(1.0)
            losses.append(residual_mean.pow(2).mean())
        return torch.stack(losses).mean()

    def trait_policy_leakage_loss(self, out: dict) -> torch.Tensor:
        theta = out["theta_mu"]
        form_h = out["form_h"].detach()
        theta_z = (theta - theta.mean(dim=0)) / theta.std(dim=0).clamp_min(1e-6)
        form_z = (form_h - form_h.mean(dim=0)) / form_h.std(dim=0).clamp_min(1e-6)
        corr = theta_z.T @ form_z / max(theta.size(0) - 1, 1)
        return corr.pow(2).mean()

    def form_information_honesty_loss(self, out: dict) -> torch.Tensor:
        info = out["discrimination"].pow(2).sum(dim=-1) * out["response_mask"]
        observed_info = info.sum(dim=1) / out["response_mask"].sum(dim=1).clamp_min(1.0)
        precision = torch.exp(-out["theta_logvar"]).mean(dim=-1)
        return F.smooth_l1_loss(precision, observed_info.detach())

    def training_loss(
        self,
        batch: dict,
        lambda_irt: float = 0.35,
        lambda_admin: float = 0.10,
        lambda_dif: float = 0.25,
        lambda_leak: float = 0.05,
        lambda_info: float = 0.05,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)
        irt_loss = self.item_response_loss(out)
        admin_loss = self.administration_loss(out)
        dif_loss = self.counterfactual_dif_loss(batch, out)
        leak_loss = self.trait_policy_leakage_loss(out)
        info_loss = self.form_information_honesty_loss(out)

        total = (
            cls_loss
            + lambda_irt * irt_loss
            + lambda_admin * admin_loss
            + lambda_dif * dif_loss
            + lambda_leak * leak_loss
            + lambda_info * info_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "item_response_loss": irt_loss.detach(),
            "administration_loss": admin_loss.detach(),
            "counterfactual_dif_loss": dif_loss.detach(),
            "trait_policy_leakage_loss": leak_loss.detach(),
            "form_information_honesty_loss": info_loss.detach(),
            "mean_abs_dif": out["dif_offset"].abs().mean().detach(),
        }


@torch.no_grad()
def build_counterfactual_test_forms(batch: dict) -> list[dict]:
    """Create policy test forms for DIF auditing, not for logits consistency."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = value.shape
    device = value.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, pending=None, panel=None):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        if pending is not None:
            out["value_pending"] = pending
        if panel is not None:
            out["panel_id"] = panel
        out.pop("counterfactual_form_bank", None)
        return out

    forms = []

    # Routine form: snap administration times to coarse clinical rounds.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    forms.append(clone_with(value * mask, rounded_time, var_id, mask))

    # Alarm form: late-window dense follow-up, early routine thinning.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_mask = torch.where(late > 0, mask, mask * alternating)
    forms.append(clone_with(value * alarm_mask, time, var_id, alarm_mask))

    # Panel debatching form: break near-synchronous cross-variable panels.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_mask = mask * (1.0 - 0.5 * close * changed_var)
    panel_id = torch.where(close.bool(), torch.arange(num_events, device=device)[None].expand_as(var_id), -torch.ones_like(var_id))
    forms.append(clone_with(value * panel_mask, time, var_id, panel_mask, panel=panel_id))

    # Value-pending form: item is administered but ordinal response is unavailable.
    pending = mask
    forms.append(clone_with(torch.zeros_like(value), time, var_id, mask, pending=pending))

    return forms
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cross-center form shift`：借鉴 PULSE，在 HiRID / MIMIC-IV / eICU 风格中心之间切换 item exposure。
   - `selective item DIF shift`：训练中心某项只在疑似风险时施测，测试中心变成常规施测，检查 item difficulty 是否改变。
   - `panel form shift`：训练中心同步 panel，测试中心拆成异步 item。
   - `value-pending form shift`：item 已施测但 response 延迟返回，测试模型是否把“已下单”误当 latent trait。
   - `TCF future-form shift`：训练中未来事件表单按中心 A 的流程出现，测试中换成中心 B 的流程。

2. **对比方法**
   - 普通 TCF / EHR foundation model。
   - 普通 irregular Transformer / STAR-Set / VP-GNN。
   - PULSE 中 LightGBM、传统深度时序模型和 LLM prompt / hybrid baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、D-IVSP、DJRT、KPMA、DVNB、DSPP、DCPD、DFFH 等。

3. **核心指标**
   - in-policy accuracy。
   - cross-center worst-policy accuracy。
   - DIF magnitude by item：哪些变量/时间窗 item 在跨中心中功能差异最大。
   - item exposure leakage：只用 item administration 预测标签的能力。
   - trait-policy correlation：`theta` 与表单摘要的相关性，越低越不依赖采样政策。
   - form information calibration：posterior precision 是否随有效 item information 合理变化。
   - DIF-corrected residual shift：DIF 校正后 item residual 在各 policy form 下是否仍有系统偏移。

4. **消融实验**
   - 去掉 `PolicyDIFHead`，检查 latent trait 是否吸收中心采样政策。
   - 去掉 `L_counterfactual_DIF`，检查跨表单 residual shift 是否重新出现。
   - 去掉 `L_trait_policy_leakage`，检查 `theta` 是否记住 item exposure。
   - 去掉 `L_form_information_honesty`，检查少量 item 表单是否被过度自信分类。
   - 将 TCF 式 pathology-focused bins 替换为均匀分箱，验证病理 item response 的价值。
   - 让 `admin_logit` 直接进入分类头作为反例，验证 item exposure shortcut 的危害。

## 5. 预期创新性

1. **从采样表征去偏转向测量项功能差异建模**：首次把 sampling-policy shift 表述为不同采样政策下 pathology item 的 administration 与 response function 改变。
2. **从 EHR 事件预测转向 IRT latent trait**：吸收 TCF 的 pathology-focused bin，但不把未来事件出现当作类别证据，而是把观测值变成可校正 DIF 的 item response。
3. **从跨中心 benchmark 转向 item-level 可解释诊断**：吸收 PULSE 的跨中心设置，进一步指出哪些 item 在哪个中心发生 DIF，解释性能退化来源。
4. **从固定问答/证明/凸包转向概率测量学**：DIPF 不预设固定 clinical viva，不学习 sequent proof，不构造 feasible hull；它学习 item discrimination、difficulty、DIF offset 与 latent trait posterior。
5. **与反事实采样低侵入兼容**：counterfactual sampler 只生成 test forms，用于 DIF moment audit；不需要对 logits/representation 做一致性约束，也不估计 hazard、density ratio、posterior quotient 或 conformal set。

## 6. 一句话投稿卖点

**DIPF 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“不同采样政策改变了病理测量项的施测概率与功能差异”的问题，通过 Pathology Itemizer、可微多维 IRT latent trait、Policy-DIF Head 与 counterfactual test-form residual moment，让分类器只依赖跨中心可比的病理潜变量，而不是依赖医院 routine/alarm 表单、panel 共现、item exposure 或 value-pending 采样捷径。**
