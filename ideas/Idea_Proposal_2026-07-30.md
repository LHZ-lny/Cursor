# Title: Do-Jury Rank Tribunal：面向采样策略偏移的反事实政策陪审团排序裁决器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md` 以及 `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`、`idea_2026-07-29.md`。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`
  - `Idea_Proposal_2026-06-24.md`
  - `Idea_Proposal_2026-06-27.md`
  - `Idea_Proposal_2026-07-15.md`
  - `Idea_Proposal_2026-07-16.md`
  - `Idea_Proposal_2026-07-17.md`
  - `Idea_Proposal_2026-07-18.md`
  - `Idea_Proposal_2026-07-19.md`
  - `Idea_Proposal_2026-07-20.md`
  - `Idea_Proposal_2026-07-21.md`
  - `Idea_Proposal_2026-07-22.md`
  - `Idea_Proposal_2026-07-23.md`
  - `Idea_Proposal_2026-07-24.md`
  - `Idea_Proposal_2026-07-25.md`
  - `Idea_Proposal_2026-07-26.md`
  - `Idea_Proposal_2026-07-27.md`
  - `Idea_Proposal_2026-07-29.md`
- 已读取近期论文记录：
  - `paper_daily.md`
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
7. 连续时间危险率 point-process scorer、采样 score 零空间、hazard-driven resampling、do-risk variance。
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
25. counterfactual conformal risk sleeves、policy-conditional nonconformity set、leave-policy-out conformal calibration。
26. counterfactual sampling instruments、first-stage structural equation、control-function residualized structure readout、weak-instrument guard。
27. 单纯把 STAR-Set 的 temporal/variable attention bias 或 VP-GNN 的 variable/patch graph 拆成 state-policy 双分支并做一致性约束。

本提案选择新的正交切入点：**不删除采样信息、不拆图、不做保形集合、不做 IV 残差化、不做对抗或一致性拉齐；而是把每一种反事实采样政策视为一名“陪审员”，让模型在类别排序层面接受可微社会选择裁决。采样政策可以改变单个 juror 的 logits，但不能让某个政策成为左右最终标签的独裁者。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-07-27.md` 中的两个前沿机制给出非常清晰的结构信号：

- **STAR-Set Transformer** 用 temporal locality penalty 和 variable-type affinity 恢复异步事件集合中的局部时间结构与变量兼容结构。
- **VP-GNN** 同时建模 variable-wise graph 与 patch-wise graph，说明采样策略偏移会沿着变量联测与时间片段可见性两条路径进入分类器。

历史提案已经尝试过把这些结构信号拆成 state/policy 双图、当作 IV treatment、作为 conformal calibrator 的条件变量，或通过控制屏障、regret、principal strata 等方式约束其风险。本轮换一个更像 AAAI 机制创新的角度：

> 在真实部署中，我们未必能判定哪个采样策略视图最接近“真状态”。但我们可以要求最终分类不是由某一个医院协议、某一种 temporal bias、某一种 patch budget 或某一种变量共现规则单独决定，而是由一组反事实采样政策陪审员的稳定类别排序共同裁决。

Sampling-policy shortcut 的典型表现不是每个反事实视图都错，而是某个特定政策视图给出极高置信的错误 logit，并在普通平均、最大池化或注意力读出中成为“独裁者”。例如：

- 训练医院中 `lactate + WBC` 经常联测，某个 variable-affinity juror 会把这种共现直接排到阳性类别前面；
- 报警后密集复测在训练环境中强关联死亡风险，某个 patch-budget juror 会把 late dense patch 的局部证据放大；
- STAR-Set 的 temporal bias 在训练采样间隔下很有效，但换医院后同一时间尺度不再代表相同临床语义。

**Do-Jury Rank Tribunal (DJRT)** 的核心思想是把输出从单模型 logits 改为 **反事实政策陪审团的类别排序裁决**：

1. 反事实采样模块生成多名 juror：routine-round、alarm-dense、variable-affinity、patch-budget、panel-split 等。
2. 每名 juror 共享 value encoder，但在其政策视图下独立产生类别偏好排序。
3. 最终标签由可微 Borda / pairwise majority tribunal 产生，而不是由某个视图的绝对 logit 直接支配。
4. 训练目标惩罚“单一政策独裁”和“结构贿赂”：如果移除某个 juror 后最终真类排名剧烈变化，说明分类器仍被某种采样政策绑架。

这样与当前“采样解耦/反事实干预”框架高度兼容：sampling branch 不进入分类头，而是负责生成政策陪审团；counterfactual intervention 不用于一致性、不用于平滑认证、不用于保形校准，而是用于构造一组可审讯的 policy jurors；readout 从 softmax 置信度变成社会选择式排名裁决。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：返回 Counterfactual Policy Jury Bank

新增 `PolicyJuryCollator`，每个 batch 返回事实视图和 `R` 个反事实政策视图：

1. **Temporal locality juror**
   - 模拟 STAR-Set temporal bias 失配。
   - 改变局部时间邻域：晚期稀疏、早期密集、固定查房式 rounded timestamps。

2. **Variable-affinity juror**
   - 模拟变量兼容矩阵偏移。
   - 打散训练医院特有的 panel 共现，或把异步变量合并成联测窗口。

3. **Patch-budget juror**
   - 模拟 VP-GNN patch-wise graph 可见性变化。
   - 每个粗窗口仅保留固定预算事件，或强调报警后局部 patch。

4. **Panel-split juror**
   - 将同一时间附近的多变量联测拆分成更异步的序列。
   - 检验分类器是否依赖“同窗共现”而非观测值语义。

5. **Routine-vs-alarm juror**
   - 在固定查房采样与告警触发采样之间切换。
   - 对应现实 ICU / EHR 中最常见的政策偏移。

这些 juror 不构成正负对比样本，也不要求 logits 一致。它们只提供多种合法采样制度下的“投票意见”。

### 2.2 改 Encoder：共享 Value Encoder + Juror-Specific Structural Lens

基础 irregular encoder 可以是 STAR-Set、VP-GNN、MTM、CDE 或普通 event Transformer。DJRT 只要求输出两类量：

```text
logits_r = BaseClassifier(do(policy_r, x))
trace_r  = structural trace under policy_r
```

其中 `trace_r` 可来自：

- STAR-Set：temporal bias energy、variable affinity energy、attention-bias contribution；
- VP-GNN：variable message energy、patch aggregation entropy、selected patch depth；
- 普通 Transformer：attention entropy、event coverage、variable coverage 等轻量 proxy。

关键点：

- `trace_r` 不被拆成 state graph / policy graph；
- `trace_r` 不进入普通分类 logits；
- `trace_r` 只用于给 juror 分配有限的 deliberation reliability，防止结构异常视图成为独裁者。

### 2.3 改 Loss：从 logits 不变转向 Social-Choice Robust Classification

每名 juror 输出类别 logits `L_r in R^K`。先把 logits 转成可微 pairwise preference：

```text
P_r[a, b] = sigmoid((L_r[a] - L_r[b]) / tau)
```

再用 juror reliability `w_r` 做加权多数：

```text
M[a, b] = sum_r w_r * P_r[a, b]
Borda[a] = sum_{b != a} M[a, b]
```

最终 `Borda` 作为分类 logit。总目标：

```text
L = L_tribunal_ce
  + lambda_maj * L_true_majority
  + lambda_dict * L_no_dictator
  + lambda_bribe * L_no_structural_bribery
  + lambda_div * L_jury_diversity
```

#### A. Tribunal 分类损失 `L_tribunal_ce`

不再对单一事实视图 logits 做最终决策，而是对 Borda 裁决分数做 CE：

```text
L_tribunal_ce = CE(Borda, y)
```

这不是概率平均，也不是随机平滑认证；它使用的是 ordinal preference aggregation。绝对 logit 标定可以随政策变化，但类别相对偏好需要通过多数裁决站得住。

#### B. True-Class Majority Loss `L_true_majority`

要求真实类别在多数陪审员中击败每个竞争类别：

```text
majority_yk = M[y, k] / (M[y, k] + M[k, y] + eps)
L_true_majority = mean_k relu(m0 - majority_yk)^2
```

这不是跨视图一致性：允许少数 juror 反对真实类别，只要求不存在由某个采样政策单独制造的反向多数。

#### C. No-Dictator Loss `L_no_dictator`

逐个移除 juror，重新计算 Borda。若移除某个政策视图后真实类别排名或最终 top-1 大幅改变，说明该视图拥有过高决策杠杆：

```text
delta_r = || softmax(Borda_all) - softmax(Borda_without_r) ||_1
L_no_dictator = mean_r relu(delta_r - delta_max)^2
```

它不同于 leave-policy-out conformal calibration：这里没有覆盖率、nonconformity 或预测集，只检查社会选择裁决是否被单一政策绑架。

#### D. No-Structural-Bribery Loss `L_no_structural_bribery`

利用 STAR-Set / VP-GNN 的结构 trace 构造 juror 的“贿赂风险”：

```text
bribe_r = zscore(temporal_bias_energy_r)
        + zscore(variable_affinity_energy_r)
        + zscore(patch_selection_concentration_r)
```

若某个 juror 的结构 trace 显示它高度依赖政策诱导的局部共现或 patch 可见性，则它不能同时获得过高 reliability：

```text
L_no_structural_bribery = mean_r relu(w_r * bribe_r - c)^2
```

这不是把结构分解成 state/policy 双图，也不是 IV residualization；它只是社会选择中的反贿赂规则：结构异常的 juror 可以发言，但不能用异常结构购买裁决权。

#### E. Jury Diversity Loss `L_jury_diversity`

若所有 juror 完全相同，陪审团失去意义。用轻量熵项防止权重塌缩：

```text
L_jury_diversity = - entropy(w)
```

注意该项只作用于 juror 权重，不要求 representation 或 logits 一致，也不扩大 policy-simplex 平滑半径。

### 2.4 推理阶段

给定测试样本：

1. 生成一组部署可接受的 policy jurors。
2. 每名 juror 在对应采样视图下输出类别 logits。
3. Tribunal 计算 pairwise majority 和 Borda 分数。
4. 输出：
   - `pred_label = argmax(Borda)`；
   - `jury_margin = Borda_top1 - Borda_top2`；
   - `dictator_score = max_r delta_r`；
   - `minority_report`：哪些 policy juror 反对最终标签。

当 `dictator_score` 高或 `jury_margin` 低时，系统不是给出保形集合或 evidential vacuity，而是报告“当前预测被某类采样政策强支配”，并建议补采样打破争议，例如补测能最大降低 pairwise cycle 的变量或时间窗。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_zscore(x: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(eps)
    else:
        weight = mask.to(x.dtype)
        while weight.dim() < x.dim():
            weight = weight.unsqueeze(-1)
        denom = weight.sum(dim=0, keepdim=True).clamp_min(1.0)
        mean = (x * weight).sum(dim=0, keepdim=True) / denom
        var = ((x - mean).pow(2) * weight).sum(dim=0, keepdim=True) / denom
        std = var.sqrt().clamp_min(eps)
    return (x - mean) / std


def pairwise_preference(logits: torch.Tensor, temperature: float = 0.25) -> torch.Tensor:
    """Return soft pairwise preferences P[a, b] = Pr(a beats b).

    Args:
        logits: [B, R, K], juror-specific class logits.
    Returns:
        prefs: [B, R, K, K].
    """
    diff = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    prefs = torch.sigmoid(diff / temperature)
    eye = torch.eye(logits.size(-1), device=logits.device, dtype=torch.bool)
    return prefs.masked_fill(eye.view(1, 1, logits.size(-1), logits.size(-1)), 0.0)


class JurorReliabilityHead(nn.Module):
    """Assign deliberation weights from structural traces only."""

    def __init__(self, trace_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trace_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        # traces: [B, R, D]
        raw = self.net(traces).squeeze(-1)
        return torch.softmax(raw, dim=1)


class DoJuryRankTribunal(nn.Module):
    """Wrap an irregular classifier with counterfactual policy-jury rank aggregation."""

    def __init__(
        self,
        base_model: nn.Module,
        trace_dim: int,
        hidden_dim: int,
        num_classes: int,
        temperature: float = 0.25,
        majority_margin: float = 0.58,
        dictator_cap: float = 0.35,
        bribe_cap: float = 0.50,
    ):
        super().__init__()
        self.base_model = base_model
        self.reliability = JurorReliabilityHead(trace_dim, hidden_dim)
        self.num_classes = num_classes
        self.temperature = temperature
        self.majority_margin = majority_margin
        self.dictator_cap = dictator_cap
        self.bribe_cap = bribe_cap

    def encode_jurors(self, batch: dict) -> dict:
        juror_batches = batch.get("policy_jury_bank", [batch])
        logits_bank = []
        trace_bank = []

        for juror_batch in juror_batches:
            out = self.base_model(juror_batch)
            logits_bank.append(out["logits"])
            if "structure_trace" in out:
                trace_bank.append(out["structure_trace"])
            else:
                prob = torch.softmax(out["logits"], dim=-1)
                entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
                margin = prob.topk(2, dim=-1).values
                margin = margin[:, :1] - margin[:, 1:2]
                coverage = juror_batch["event_mask"].float().mean(dim=1, keepdim=True)
                trace_bank.append(torch.cat([entropy, margin, coverage], dim=-1))

        logits = torch.stack(logits_bank, dim=1)  # [B, R, K]
        traces = torch.stack(trace_bank, dim=1)   # [B, R, D]
        weights = self.reliability(traces)        # [B, R]
        return {"juror_logits": logits, "juror_traces": traces, "juror_weights": weights}

    def tribunal_scores(self, logits: torch.Tensor, weights: torch.Tensor) -> dict:
        prefs = pairwise_preference(logits, temperature=self.temperature)
        majority = (prefs * weights[:, :, None, None]).sum(dim=1)
        borda = majority.sum(dim=-1)
        return {"prefs": prefs, "majority": majority, "borda": borda}

    def without_one_scores(self, logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        scores = []
        num_jurors = logits.size(1)
        for idx in range(num_jurors):
            keep = torch.ones(num_jurors, device=logits.device, dtype=torch.bool)
            keep[idx] = False
            kept_logits = logits[:, keep]
            kept_weights = weights[:, keep]
            kept_weights = kept_weights / kept_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            scores.append(self.tribunal_scores(kept_logits, kept_weights)["borda"])
        return torch.stack(scores, dim=1)

    def forward(self, batch: dict) -> dict:
        jurors = self.encode_jurors(batch)
        tribunal = self.tribunal_scores(jurors["juror_logits"], jurors["juror_weights"])
        leave_one = self.without_one_scores(jurors["juror_logits"], jurors["juror_weights"])
        return {**jurors, **tribunal, "leave_one_borda": leave_one}

    def loss(self, batch: dict) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        borda = out["borda"]
        majority = out["majority"]
        weights = out["juror_weights"]
        traces = out["juror_traces"]
        leave_one = out["leave_one_borda"]

        tribunal_ce = F.cross_entropy(borda, labels)

        true_vs_all = majority.gather(
            1, labels[:, None, None].expand(-1, 1, self.num_classes)
        ).squeeze(1)
        competitor_mask = ~F.one_hot(labels, self.num_classes).bool()
        true_majority = true_vs_all[competitor_mask].view(labels.size(0), self.num_classes - 1)
        majority_loss = F.relu(self.majority_margin - true_majority).pow(2).mean()

        full_prob = torch.softmax(borda, dim=-1)
        leave_prob = torch.softmax(leave_one, dim=-1)
        dictator_delta = (leave_prob - full_prob[:, None]).abs().sum(dim=-1)
        no_dictator_loss = F.relu(dictator_delta - self.dictator_cap).pow(2).mean()

        # The first trace channels should encode structural stress proxies such as
        # temporal-bias energy, variable-affinity concentration, and patch entropy.
        bribe_risk = masked_zscore(traces[..., : min(3, traces.size(-1))]).mean(dim=-1).relu()
        no_bribe_loss = F.relu(weights * bribe_risk - self.bribe_cap).pow(2).mean()

        jury_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
        diversity_loss = -jury_entropy

        total = (
            tribunal_ce
            + 0.35 * majority_loss
            + 0.20 * no_dictator_loss
            + 0.15 * no_bribe_loss
            + 0.03 * diversity_loss
        )
        return {
            "loss": total,
            "tribunal_ce": tribunal_ce.detach(),
            "true_majority_loss": majority_loss.detach(),
            "no_dictator_loss": no_dictator_loss.detach(),
            "no_structural_bribery_loss": no_bribe_loss.detach(),
            "jury_entropy": jury_entropy.detach(),
            "dictator_score": dictator_delta.max(dim=1).values.detach(),
        }


@torch.no_grad()
def build_policy_jury_bank(batch: dict) -> list[dict]:
    """Construct counterfactual sampling-policy jurors without creating labels."""
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        return out

    jurors = [batch]

    # Temporal locality juror: routine-round timestamps.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    jurors.append(clone_with(value * mask, rounded_time, var_id, mask))

    # Variable-affinity juror: weaken adjacent cross-variable co-observation.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    affinity_mask = mask * (1.0 - 0.5 * close * changed_var)
    jurors.append(clone_with(value * affinity_mask, time, var_id, affinity_mask))

    # Patch-budget juror: retain a small budget in each coarse time patch.
    patch_keep = torch.zeros_like(mask)
    for start, end in [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]:
        in_patch = ((time_norm >= start) & (time_norm < end)).to(mask.dtype) * mask
        rank = in_patch.cumsum(dim=1)
        patch_keep = torch.maximum(patch_keep, (rank <= 2).to(mask.dtype) * in_patch)
    jurors.append(clone_with(value * patch_keep, time, var_id, patch_keep))

    # Alarm-dense juror: emphasize late observations while thinning early routine events.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_mask = torch.where(late > 0, mask, mask * alternating)
    jurors.append(clone_with(value * alarm_mask, time, var_id, alarm_mask))

    return jurors
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `temporal-bias shift`：改变 STAR-Set temporal locality 支持的时间尺度。
   - `variable-affinity shift`：训练环境变量联测，测试环境拆成异步测量。
   - `patch-budget shift`：VP-GNN 中高权重 patch 在测试政策下被压缩或不可见。
   - `routine-vs-alarm shift`：固定查房式采样与告警触发式采样互换。

2. **对比方法**
   - STAR-Set / VP-GNN 原模型。
   - 平均 logits 的 counterfactual ensemble。
   - 最大置信度选择的 policy ensemble。
   - 普通 uncertainty / calibration baseline。
   - 历史方案 C-CRS、D-IVSP、PSSC、OCBC、RE-CDR、ST-FDN 等。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - jury-margin under shift。
   - dictator score：单一 policy juror 对最终预测的最大影响。
   - minority-report consistency：反对最终标签的 juror 是否集中在某类采样策略。
   - structural bribery gap：高 temporal/variable/patch stress juror 是否获得异常裁决权。

4. **消融实验**
   - 去掉 No-Dictator Loss，仅做 Borda 聚合。
   - 去掉 No-Structural-Bribery Loss，让高结构压力 juror 自由获得权重。
   - 用普通概率平均替代 rank tribunal，验证排序社会选择而非 ensemble 数量带来收益。
   - 只用事实 juror，验证反事实政策陪审团的必要性。
   - 把 STAR-Set/VP-GNN trace 替换为随机 trace，验证结构反贿赂规则的作用。

## 5. 预期创新性

1. **从表示不变转向社会选择裁决**：不要求各采样视图 logits 一致，而是让类别偏好通过多数排序规则抵抗单一政策捷径。
2. **从结构拆分转向反独裁规则**：吸收 STAR-Set/VP-GNN 的结构诊断，但不拆 state/policy 图；只限制结构异常 juror 获得过大裁决权。
3. **从 calibration set 转向 minority report**：不同于保形集合或 evidential vacuity，DJRT 输出陪审团争议来源，能直接解释“哪种采样政策在反对最终诊断”。
4. **从反事实增强转向政策审讯**：counterfactual intervention 不再服务对比、一致性、平滑、风险方差或覆盖率，而是生成多种可部署采样制度下的 juror 证词。
5. **审稿卖点清晰**：AAAI 审稿人能直接看到 social choice / rank aggregation 与 sampling-policy robustness 的新结合，机制上与过往危险率、IV、保形、纠错、拓扑、gauge、knockoff 等路线显著正交。

## 6. 一句话投稿卖点

**Do-Jury Rank Tribunal 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“不同采样制度陪审员之间的类别排序裁决问题”，通过可微 Borda 多数、反独裁裁决损失与结构反贿赂规则，阻止 STAR-Set/VP-GNN 式 temporal bias、variable affinity 和 patch visibility 在部署时变成单一采样政策独裁者。**
