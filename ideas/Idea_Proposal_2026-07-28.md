# Title: Counterfactual Conformal Risk Sleeves：面向采样策略偏移的反事实保形风险护套分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md` 以及持久记忆中的 `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`。
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
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-07-13.md`
  - `paper_daily_2026-07-19.md`
  - `paper_daily_2026-07-26.md`
  - `paper_daily_2026-07-27.md`

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮永久避开以下核心机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、分类梯度与采样 score 零空间正交、hazard-driven resampling、do-risk variance。
8. 生理流算子与采样算子交换子、value/policy graph commutator、policy residual sink。
9. additive evidence market、protocol tax、token-level evidence budget、边际证据审计。
10. 模型空间 posterior quotient、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、ANOVA projection、VQ semantic clauses、HSIC redaction。
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
25. 单纯把 STAR-Set 的 temporal/variable attention bias 或 VP-GNN 的 variable/patch graph 拆成 state-policy 双分支并做一致性约束。

本提案选择一个正交切入点：**不要求不同采样策略下 logits 相同，不删除采样信息，不做 uncertainty vacuity，也不做 graph/bias 分解；而是把采样偏移转化为“分类声明何时仍可保形校准”的问题。模型只有在反事实采样策略族下的 nonconformity 分数落入可校准风险护套时，才输出稳定 singleton 类别；否则输出多类别候选集或请求补采样。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-07-27.md` 中的两个机制给了新的前沿信号：

- **STAR-Set Transformer** 用 temporal locality penalty 和 variable-type affinity 把网格结构先验注入 point-set Transformer。它说明 attention bias 本身会记录“同一变量多久还有效”“哪些变量常在局部时间窗共现”。
- **VP-GNN** 同时建模 variable-wise graph 与 patch-wise graph。它说明采样策略偏移可以沿着变量消息传递和时间 patch 聚合两条路径改变分类边界。

历史思路大多试图把这类结构偏移拆掉、隔离、审计或约束一致。但真实部署中，一个更尖锐的问题是：

> 当测试医院的采样流程改变后，模型是否仍有资格输出一个单点类别，而不是只给出一个高置信但不可校准的猜测？

Sampling-policy shift 的危险不只是 representation 漂移，而是 **错误的 singleton decision**：训练环境中某个变量共现、时间邻域或 patch 选择让某类 logit 特别高；测试环境一换，这个 logit 仍然高，但它已经不再具备跨策略校准意义。普通 softmax、evidential uncertainty 或 attention 解释都无法直接回答“该单点预测在采样策略变化下是否仍覆盖真实标签”。

**Counterfactual Conformal Risk Sleeves (C-CRS)** 的核心直觉是：

> 对每个类别学习一条围绕 nonconformity score 的反事实保形护套。采样策略可以改变 attention bias、变量图、patch 图和 logits；我们不强行让它们不变，而是要求真实类别在所有可部署策略下仍落入可校准的风险护套。只有当候选集收缩为 singleton 时，模型才输出确定分类。

这与当前“采样解耦/反事实干预”框架高度兼容：

- value encoder 继续输出分类 logits 和结构诊断。
- sampling branch 不进入分类头，而是生成 policy signature 和反事实采样 recipe。
- counterfactual intervention 不用于一致性、不用于平滑半径、不用于 knockoff；它只生成校准用的 policy-conditional nonconformity 样本。
- 最终 readout 从 `argmax softmax` 改为 **policy-calibrated conformal set**；鲁棒性目标从“表示不变”转为“跨策略覆盖有效且 singleton 足够尖锐”。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：增加结构诊断输出，但不做 state-policy 图分解

C-CRS 可以包裹 STAR-Set、VP-GNN、MTM、CDE 或普通 irregular Transformer。基础模型需要额外返回三个诊断量：

1. **Value logits `logits`**
   - 标准分类输出。
   - 训练 CE 仍保留，避免 conformal head 替代主任务学习。

2. **Structure trace `s_struct`**
   - 来自 STAR-Set：attention bias contribution、temporal scale usage、variable affinity energy。
   - 来自 VP-GNN：variable message energy、patch aggregation entropy、selected patch depth。
   - 它不是分类特征的新捷径；只输入 conformal calibrator，用来判断当前结构证据在采样政策下是否可校准。

3. **Policy signature `p_sig`**
   - 由 sampling branch 从时间、变量、mask、局部共现、patch 可见性提取。
   - 不拼入分类 logits。
   - 只用于预测 nonconformity 护套的条件分位数。

关键差异：C-CRS 不把结构分成 state graph / policy graph，不要求 attention bias 在多策略下一致，也不对结构做 knockoff/FDR；它只问“给定这些结构诊断，真实标签的 nonconformity 是否仍被护套覆盖”。

### 2.2 改 Loss：从不变性约束转向 Policy-Conditional Conformal Calibration

定义类别级 nonconformity：

```text
a_y(x) = max_{k != y} logit_k(x) - logit_y(x)
```

若 `a_y` 越小，类别 `y` 越可信。C-CRS 学习一个条件分位数护套：

```text
q_y = Q_phi(y, p_sig, s_struct, alpha)
```

训练目标：

```text
L = L_cls
  + lambda_q   * L_counterfactual_quantile
  + lambda_cov * L_soft_coverage
  + lambda_eff * L_singleton_efficiency
  + lambda_loo * L_leave_policy_out
```

#### A. 分类损失 `L_cls`

事实样本仍用标准 CE：

```text
L_cls = CE(logits_factual, y)
```

#### B. 反事实分位数损失 `L_counterfactual_quantile`

对同一患者轨迹施加多种采样 recipe，得到 nonconformity 分数 `a_y^r`。Calibrator 预测真实类在该策略结构下的分位数 `q_y^r`，用 pinball loss 训练：

```text
L_quantile = rho_{1-alpha}(a_y^r - q_y^r)
```

其中 `alpha` 是允许的错误率，例如 0.1。直觉：护套不是要求所有策略下 logits 一样，而是学习“在这个采样结构下，真实类别 nonconformity 应该有多宽才算可校准”。

#### C. Soft Coverage Loss `L_soft_coverage`

对每个 batch 和每个 policy recipe，用可微覆盖率近似：

```text
covered = sigmoid((q_y^r - a_y^r) / tau)
L_soft_coverage = relu((1 - alpha) - mean_r,b covered)^2
```

这直接优化跨策略覆盖，而不是优化 policy radius、risk variance 或 evidential uncertainty。

#### D. Singleton Efficiency Loss `L_singleton_efficiency`

保形预测集定义为：

```text
C(x) = { c : a_c(x) <= q_c(x) }
```

若集合总是很大，覆盖率容易但分类无用。因此加入 soft set size 惩罚：

```text
set_size = sum_c sigmoid((q_c - a_c) / tau)
L_eff = relu(set_size - 1 - delta)^2
```

这鼓励模型在 value evidence 足够稳定时输出 singleton；在采样偏移导致结构不可校准时，允许输出多类别集合或拒识。

#### E. Leave-Policy-Out Calibration `L_leave_policy_out`

为了防止 calibrator 记住训练 recipe，训练时随机遮蔽一个 policy recipe，只用其他 recipe 的分位数统计预测被遮蔽 recipe 的护套：

```text
q_holdout = Q_phi(y, p_sig_holdout, aggregate(other recipes))
L_loo = pinball(a_y_holdout - q_holdout)
```

它不是对抗学习，也不是一致性；它模拟“部署时来了一个没见过的采样流程”，迫使护套学习可迁移的校准规律。

### 2.3 改 Dataloader：返回 Conformal Policy Sleeve Bank

新增 `ConformalSleeveCollator`，每个 batch 返回：

1. `factual_batch`：原始事件。
2. `policy_recipe_bank`：可部署采样策略族，例如：
   - `locality_scale_shift`：改变 STAR-Set temporal locality 可支持的时间邻域。
   - `variable_affinity_shift`：改变变量局部共现与联测机会。
   - `patch_budget_shift`：改变 VP-GNN patch 可见性与聚合深度。
   - `alarm_dense_to_routine`：告警后密集测量改成固定查房式。
3. `cf_batch_bank`：每个 recipe 下的反事实观测事件。
4. `policy_signature_bank`：每个 recipe 的采样结构摘要。
5. `calibration_split_id`：batch 内分为 train-calib / holdout-policy 两部分，用于 leave-policy-out。

与历史机制区别：

- 不生成正样本用于对比学习。
- 不要求 representation 或 logits 一致。
- 不估计 hazard、density ratio、posterior factor 或 solver trace。
- 不输出 evidential vacuity 或 certified radius。
- 不做 knockoff FDR、信息格 meet/join、graph 分解、gauge 投影或 syndrome repair。
- 只把反事实采样视图用作 **保形 nonconformity 校准样本**。

### 2.4 推理阶段

给定测试样本：

1. base encoder 输出 `logits` 与 `s_struct`。
2. sampling branch 输出 `p_sig`。
3. conformal head 对每个类别输出 `q_c`。
4. 构造候选集：

```text
C(x) = { c : max_{k != c} logit_k - logit_c <= q_c }
```

决策规则：

- 若 `|C(x)| = 1`：输出 singleton 类别。
- 若 `|C(x)| > 1`：输出候选集，并触发补采样建议，例如补测能最大缩小 set size 的变量/时间窗。
- 若 `|C(x)| = 0`：说明结构诊断 OOD，可回退为 `argmax` 但标记为 nonconformal prediction。

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


def class_nonconformity(logits: torch.Tensor) -> torch.Tensor:
    """Return a_c = max_{k != c} logit_k - logit_c for every candidate class."""
    num_classes = logits.size(-1)
    rival = []
    for cls_idx in range(num_classes):
        masked = logits.masked_fill(
            F.one_hot(torch.full((logits.size(0),), cls_idx, device=logits.device), num_classes).bool(),
            -1e4,
        )
        rival.append(masked.max(dim=-1).values - logits[:, cls_idx])
    return torch.stack(rival, dim=-1)


def gather_true_score(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return scores.gather(1, labels[:, None]).squeeze(1)


def pinball_loss(error: torch.Tensor, quantile: float) -> torch.Tensor:
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


class PolicySignatureEncoder(nn.Module):
    """Summarize sampling structure for calibration only, not for logits."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        event_time = batch["event_time"]
        event_var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]

        horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = event_time / horizon
        delta_t = torch.zeros_like(event_time)
        delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)

        var_obs = F.one_hot(event_var_id.clamp_min(0), self.num_vars).to(event_time.dtype)
        var_rate = (var_obs * event_mask.unsqueeze(-1)).sum(dim=1)
        var_rate = var_rate / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        early = (time_norm <= 0.33).to(event_time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(event_time.dtype)
        late = (time_norm > 0.66).to(event_time.dtype)

        # Local co-observation proxy: adjacent events close in time and from different variables.
        close = (delta_t <= delta_t[event_mask > 0].mean().clamp_min(1e-6)).to(event_time.dtype)
        var_change = torch.zeros_like(event_time)
        var_change[:, 1:] = (event_var_id[:, 1:] != event_var_id[:, :-1]).to(event_time.dtype)
        co_obs = masked_mean(close * var_change, event_mask, dim=1).unsqueeze(-1)

        stats = torch.cat(
            [
                event_mask.mean(dim=1, keepdim=True),
                masked_mean(early, event_mask, dim=1).unsqueeze(-1),
                masked_mean(middle, event_mask, dim=1).unsqueeze(-1),
                masked_mean(late, event_mask, dim=1).unsqueeze(-1),
                masked_mean(torch.log1p(delta_t), event_mask, dim=1).unsqueeze(-1),
                delta_t.amax(dim=1, keepdim=True),
                co_obs,
                event_mask.sum(dim=1, keepdim=True) / event_mask.size(1),
            ],
            dim=-1,
        )
        return self.net(torch.cat([var_rate, stats], dim=-1))


class StructureTraceEncoder(nn.Module):
    """Compress STAR-Set / VP-GNN style structural diagnostics into calibration features."""

    def __init__(self, trace_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trace_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, model_out: dict) -> torch.Tensor:
        if "structure_trace" in model_out:
            trace = model_out["structure_trace"]
        else:
            logits = model_out["logits"]
            # Fallback diagnostics: confidence shape only. Production models should
            # pass attention-bias or graph/patch statistics here.
            prob = torch.softmax(logits, dim=-1)
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            top2 = prob.topk(2, dim=-1).values
            margin = top2[:, :1] - top2[:, 1:2]
            trace = torch.cat([entropy, margin, prob.mean(dim=-1, keepdim=True)], dim=-1)
        return self.net(trace)


class ConformalSleeveHead(nn.Module):
    """Predict class-conditional nonconformity sleeves."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.alpha_proj = nn.Linear(1, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        policy_h: torch.Tensor,
        struct_h: torch.Tensor,
        value_h: torch.Tensor,
        alpha: float = 0.10,
    ) -> torch.Tensor:
        bsz = policy_h.size(0)
        classes = torch.arange(self.num_classes, device=policy_h.device)
        class_h = self.class_embed(classes)[None].expand(bsz, -1, -1)

        alpha_tensor = torch.full((bsz, self.num_classes, 1), alpha, device=policy_h.device)
        alpha_h = self.alpha_proj(alpha_tensor)

        policy = policy_h[:, None].expand_as(class_h)
        struct = struct_h[:, None].expand_as(class_h)
        value = value_h[:, None].expand_as(class_h)
        raw_q = self.net(torch.cat([policy, struct, value, class_h + alpha_h], dim=-1)).squeeze(-1)
        # Nonconformity thresholds can be negative, but bounding prevents exploding sleeves.
        return 4.0 * torch.tanh(raw_q / 4.0)


class CounterfactualConformalRiskSleeves(nn.Module):
    """Wrap an irregular classifier with policy-conditional conformal sleeves."""

    def __init__(
        self,
        base_model: nn.Module,
        num_vars: int,
        num_classes: int,
        value_dim: int,
        policy_hidden: int = 128,
        trace_dim: int = 3,
        alpha: float = 0.10,
    ):
        super().__init__()
        self.base_model = base_model
        self.policy_encoder = PolicySignatureEncoder(num_vars, policy_hidden)
        self.trace_encoder = StructureTraceEncoder(trace_dim, policy_hidden)
        self.value_proj = nn.Linear(value_dim, policy_hidden)
        self.sleeve = ConformalSleeveHead(policy_hidden, num_classes)
        self.num_classes = num_classes
        self.alpha = alpha

    def encode_once(self, batch: dict) -> dict:
        out = self.base_model(batch)
        logits = out["logits"]
        if "value_state" in out:
            value_state = out["value_state"]
        else:
            value_state = logits
        value_h = self.value_proj(value_state)
        policy_h = self.policy_encoder(batch)
        struct_h = self.trace_encoder(out)
        q = self.sleeve(policy_h, struct_h, value_h, alpha=self.alpha)
        score = class_nonconformity(logits)
        return {
            **out,
            "policy_h": policy_h,
            "struct_h": struct_h,
            "value_h": value_h,
            "sleeve_q": q,
            "nonconformity": score,
        }

    def prediction_set(self, batch: dict, temperature: float = 0.05) -> dict:
        out = self.encode_once(batch)
        soft_member = torch.sigmoid((out["sleeve_q"] - out["nonconformity"]) / temperature)
        hard_member = out["nonconformity"] <= out["sleeve_q"]
        return {**out, "soft_member": soft_member, "prediction_set": hard_member}

    def training_loss(
        self,
        batch: dict,
        lambda_q: float = 0.30,
        lambda_cov: float = 0.30,
        lambda_eff: float = 0.05,
        lambda_loo: float = 0.15,
        temperature: float = 0.05,
        set_slack: float = 0.25,
    ) -> dict:
        labels = batch["labels"]
        factual = self.encode_once(batch)
        cls_loss = F.cross_entropy(factual["logits"], labels)

        quantile_losses = []
        cover_terms = []
        set_sizes = []
        holdout_errors = []

        cf_batches = batch.get("cf_batch_bank", [])
        if not cf_batches:
            cf_batches = [batch]

        for idx, cf_batch in enumerate(cf_batches):
            cf = self.encode_once(cf_batch)
            score_y = gather_true_score(cf["nonconformity"], labels)
            q_y = gather_true_score(cf["sleeve_q"], labels)
            quantile_losses.append(pinball_loss(score_y - q_y, quantile=1.0 - self.alpha))

            covered = torch.sigmoid((q_y - score_y) / temperature)
            cover_terms.append(covered.mean())

            soft_member = torch.sigmoid((cf["sleeve_q"] - cf["nonconformity"]) / temperature)
            set_sizes.append(soft_member.sum(dim=-1).mean())

            # Leave-policy-out: predict held-out recipe sleeve from factual value state
            # plus held-out policy/structure, without forcing logits consistency.
            if len(cf_batches) > 1:
                other_idx = [j for j in range(len(cf_batches)) if j != idx]
                other_policy = []
                other_struct = []
                for j in other_idx:
                    other = self.encode_once(cf_batches[j])
                    other_policy.append(other["policy_h"].detach())
                    other_struct.append(other["struct_h"].detach())
                agg_policy = torch.stack(other_policy, dim=0).mean(dim=0)
                agg_struct = torch.stack(other_struct, dim=0).mean(dim=0)
                q_holdout = self.sleeve(agg_policy, agg_struct, factual["value_h"].detach(), alpha=self.alpha)
                q_holdout_y = gather_true_score(q_holdout, labels)
                holdout_errors.append(pinball_loss(score_y.detach() - q_holdout_y, quantile=1.0 - self.alpha))

        quantile_loss = torch.stack(quantile_losses).mean()
        coverage = torch.stack(cover_terms).mean()
        coverage_loss = F.relu((1.0 - self.alpha) - coverage).pow(2)

        set_size = torch.stack(set_sizes).mean()
        efficiency_loss = F.relu(set_size - 1.0 - set_slack).pow(2)

        if holdout_errors:
            loo_loss = torch.stack(holdout_errors).mean()
        else:
            loo_loss = torch.zeros((), device=labels.device)

        total = (
            cls_loss
            + lambda_q * quantile_loss
            + lambda_cov * coverage_loss
            + lambda_eff * efficiency_loss
            + lambda_loo * loo_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "quantile_loss": quantile_loss.detach(),
            "coverage_loss": coverage_loss.detach(),
            "singleton_efficiency_loss": efficiency_loss.detach(),
            "leave_policy_out_loss": loo_loss.detach(),
            "soft_coverage": coverage.detach(),
            "soft_set_size": set_size.detach(),
        }


@torch.no_grad()
def build_conformal_policy_sleeve_bank(batch: dict, num_recipes: int = 4) -> list[dict]:
    """Create counterfactual policy views for conformal sleeve calibration."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = value.shape
    device = value.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon
    cf_batches = []

    def clone_with(new_value, new_time, new_var, new_mask):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        return out

    # 1. Temporal locality scale shift: keep sparse anchors in late window.
    late_sparse = mask.clone()
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    late_sparse = torch.where(late > 0, late * alternating, late_sparse) * mask
    cf_batches.append(clone_with(value * late_sparse, time, var_id, late_sparse))

    # 2. Variable affinity shift: remove local co-observation opportunities.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    close = gap <= masked_mean(gap, mask, dim=1).unsqueeze(-1).clamp_min(1e-6)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    affinity_mask = mask * (1.0 - (close.to(mask.dtype) * changed_var * 0.5))
    cf_batches.append(clone_with(value * affinity_mask, time, var_id, affinity_mask))

    # 3. Patch budget shift: keep only representative events in each coarse window.
    patch_keep = torch.zeros_like(mask)
    for start, end in [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]:
        in_patch = ((time_norm >= start) & (time_norm < end)).to(mask.dtype) * mask
        rank = in_patch.cumsum(dim=1)
        patch_keep = torch.maximum(patch_keep, ((rank <= 2).to(mask.dtype) * in_patch))
    cf_batches.append(clone_with(value * patch_keep, time, var_id, patch_keep))

    # 4. Routine-round shift: snap timestamps to coarse rounds without changing values.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    cf_batches.append(clone_with(value * mask, rounded_time, var_id, mask))

    return cf_batches[:num_recipes]
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `attention-bias shift`：改变局部时间邻域，使 STAR-Set 的 temporal bias 学到的有效时间尺度失效。
   - `variable-affinity shift`：训练环境中变量经常联测，测试环境拆成异步测量。
   - `patch-budget shift`：VP-GNN 中某些高权重 patch 在测试政策下不可见。
   - `routine-vs-alarm shift`：告警触发密集测量切换为固定查房式测量。

2. **对比方法**
   - STAR-Set / VP-GNN 原模型。
   - 普通 temperature scaling / Platt calibration。
   - Evidential uncertainty baseline。
   - Conformal prediction without policy signatures。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、PSSC。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy singleton accuracy。
   - policy-shift conformal coverage：真实标签是否落入 `C(x)`。
   - singleton rate at target coverage。
   - invalid singleton rate：错误且集合大小为 1 的比例。
   - mean set size under shift。
   - structure-conditioned calibration gap：按 temporal bias / variable affinity / patch entropy 分桶后的覆盖误差。

4. **消融实验**
   - 去掉 policy signature，只做普通 conformal calibration。
   - 去掉 structure trace，只看采样统计，验证 STAR-Set/VP-GNN 结构诊断的必要性。
   - 去掉 leave-policy-out，检查未知采样流程下覆盖是否退化。
   - 将 counterfactual policy bank 替换为随机 mask，验证收益来自结构化采样策略而非普通增强。
   - 强制输出 singleton，验证高置信错误是否重新上升。

## 5. 预期创新性

1. **从表示鲁棒转向决策可校准**：不再把目标限定为 representation/logits 不变，而是直接约束“采样策略改变后，分类声明是否仍有覆盖意义”。
2. **从 uncertainty mass 转向 conformal set**：不同于 evidential shield 的主观无知，C-CRS 输出的是 policy-conditional nonconformity prediction set，能以覆盖率和 singleton rate 量化部署风险。
3. **从结构分解转向结构条件校准**：吸收 STAR-Set attention bias 和 VP-GNN graph/patch 诊断，但不把它们拆成 state-policy 双图；只把它们作为保形护套的条件变量。
4. **从反事实一致性转向反事实校准样本**：counterfactual intervention 不用于拉近表示，也不做平滑认证，而是扩展 calibration distribution，使未知采样流程下仍能控制错误 singleton。
5. **部署动作明确**：若输出多类别集合，系统可以请求补采样；补采样目标可选为最大化预期 set-size 收缩的变量或时间窗。

## 6. 一句话投稿卖点

**C-CRS 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“单点分类声明在跨采样策略下是否仍可保形校准”的问题，并通过 policy-conditional nonconformity sleeves、反事实采样校准库与 leave-policy-out 覆盖训练，让模型在不依赖危险率、对抗、一致性、后验商、随机平滑、拓扑、gauge、纠错码、knockoff、evidential uncertainty、信息格或 solver trace 的前提下，显式减少采样偏移下的高置信错误 singleton。**
