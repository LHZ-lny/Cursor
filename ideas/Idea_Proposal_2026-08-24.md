# Title: Do-Dialectical JEPA Referee：面向采样策略偏移的反事实双辩潜表征裁判

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`、`*summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未检出可读取的工作总结文件。
- 已读取 `paper_daily.md`，重点纳入最新记录中的 **JETS** 与 **TRIAGE**，并参考近期 **CauKer / PULSE / TCF** 的启发。
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
  - `ideas/Idea_Proposal_2026-08-22.md`
  - `ideas/Idea_Proposal_2026-08-23.md`
- 已读取自动化记忆 `MEMORIES.md` 与额外历史提案摘要：`idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`、`idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`、`idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-07.md`、`idea_2026-08-10.md`、`idea_2026-08-11.md`、`idea_2026-08-21.md`。

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
11. reconstruction error cartography、ANOVA projection、VQ semantic / acquisition clauses、HSIC redaction。
12. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler。
13. 采样测度 density ratio、doubly robust target-measure correction、influence-bound。
14. optional-stopping martingale queries、standardized innovation、stopping recipe moment control。
15. soft excursion topology、censored persistence interval、censor envelope、fragmentation sobriety。
16. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
17. policy-only negative film、shadow eraser / stencil、shadow nullification。
18. latent packet codeword、parity-check、syndrome locator、packet repair decoder。
19. conditional knockoff calendar、soft knockoff-FDR、swap symmetry。
20. observability witness、counterfactual observability probe、observability-routed classification。
21. subjective-logic / Dirichlet evidential shield、policy-induced vacuity、class-wise evidence discount。
22. observation-set policy lattice、meet/join masks、monotone/submodular margin、shortcut curvature。
23. solver-trace front-door、NFE/roughness trace mediator、reference trace bank。
24. measurement-action bisimulation、policy-word signature renormalization、thermodynamic free-energy、Sklar copula stripping、triage queue debt、Sinkhorn detail canonicalization、MDL episode transducer、causal sheaf gluing、trigger hysteresis、control barrier certificates、regret escrow、principal-stratum status compiler。
25. counterfactual conformal risk sleeves、counterfactual sampling instruments、Borda / majority rank tribunal、Krylov policy subspace、determinantal / Nystrom volume basis、tropical support routes。
26. fixed clinical viva question bank、pathology sequent proof bank、disease-progress poset clock、pathology feasible hull、IRT latent trait / DIF firewall。
27. observation-resolution RG beta flow、event-time vs record-time causal curtain、clinical tomography ray design、matched policy risk sets、Gaussian privacy cloak。
28. CauKer-style observation-policy orthogonal falsification forge、policy-cell DRO、合成反例锻炉。

本提案选择新的正交切入点：**不删除、不投影、不校准、不投票、不证明、不测验、不隐私化采样政策，也不在数据层制造正交合成世界；而是把采样政策捷径表述为“只有单方辩词、缺少反方可预测潜表征支撑”的伪论证。借鉴 JETS 的 latent predictive learning 与 TRIAGE 的正反结局辩证思想，让每个类别必须通过 outcome-conditioned JEPA 预测目标潜表征，并让 policy-only decoy 在双辩裁判前保持沉默。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 最新记录中的 **JETS** 和 **TRIAGE** 提供了两个与历史机制明显不同的前沿信号。

第一，**JETS** 将可穿戴健康行为数据表示为 `(timestamp, value, metric type)` triplets，并采用 JEPA 式 joint-embedding predictive architecture：不是重构原始噪声点，而是在 latent space 中从部分观测预测完整视图表示。这对非规则采样很关键，因为很多缺口、短尖峰、佩戴/充电断点和设备 duty cycle 并不是稳定病理状态，逐点重构会把设备策略当作目标；latent prediction 更适合学习高层状态语义。

第二，**TRIAGE** 指出临床风险预测不应只为单一结论找理由，而要让 competing outcomes 同时提出正反证据，避免 LLM 在不规则医疗时序上产生 risk polarization。虽然 TRIAGE 走的是解释/LLM 路线，但它背后的思想更通用：一个稳健分类声明应经得起“为什么是这个类别”和“为什么不是竞争类别”的双向检验。

采样策略偏移最危险的地方，往往是模型只听到单方辩词：

- 某医院报警后密集复测，模型听到“高风险”的单方论据，却没有检查这些复测值是否真的能预测高风险状态的潜表征；
- 某可穿戴设备夜间缺测，模型听到“睡眠异常/低活跃”的单方论据，却没有检查反方类别是否同样能解释 latent target；
- 某中心的 panel 同步出现，模型听到“炎症/脓毒症”的单方论据，却没有让竞争类别用同一批 value evidence 作出反驳。

**Do-Dialectical JEPA Referee (DD-JEPA)** 的核心观点是：

> 采样政策可以改变“谁被允许发言”，但不能让 policy-only 的发言赢得类别裁决。每个类别都必须通过一个 outcome-conditioned latent predictor 解释被遮蔽的 value target；最终分类来自正反预测误差的辩证差，而不是来自 mask、日历、panel 或 value-pending 本身。

与当前“采样解耦/反事实干预”框架结合时：

- value process 负责编码 pathology / behavior triplets；
- sampling process 不进入分类头，只构造 policy-only decoy 与反事实可见性；
- counterfactual intervention 不生成对比正样本、不做一致性、不做保形/隐私/IRT/RG，而是制造“只有采样坐标、没有 value target 支撑”的沉默辩词；
- classifier 不直接读取 pooled logits，而读取每个类别在 latent JEPA 目标上的 **affirm-vs-rebut predictive gap**。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：Outcome-Conditioned Dialectical JEPA

DD-JEPA 将不规则事件流拆成三种视图：

1. **Context view `x_ctx`**
   - 事实观测或反事实采样下可见的 value triplets。
   - 输入包含 `(value, time, variable, delta_t, quality)`。
   - 不直接输入 policy id 或中心 id。

2. **State target view `x_tgt`**
   - 由 value-grounded target sampler 选择的被遮蔽 value tokens。
   - 目标由 EMA target encoder 编码，停止梯度：

```text
z_tgt = stopgrad(EMA_TargetEncoder(x_tgt))
```

   - 与 iTimER 不同，这里不重构原始值、不使用 reconstruction error 分布、不生成 pseudo-observation，也不做 Wasserstein 对齐。

3. **Policy-only decoy `x_decoy`**
   - 保留时间戳、变量 id、panel/pending/density 等采样坐标，但遮蔽或置零 value。
   - decoy 的作用不是分类，也不是 value-redacted classifier；它只检测“单靠采样日历能否让某个类别在双辩中获胜”。

核心模块是 **Outcome-Conditioned Dialectical Predictor**。对每个类别 `c`，学习两组 predictor：

```text
z_affirm_c = P_affirm(h_ctx, embed(c))
z_rebut_c  = P_rebut(h_ctx, embed(c))
```

- `affirm_c` 表示“若类别为 c，当前 context 应如何预测 target latent”；
- `rebut_c` 表示“若类别不是 c，当前 context 对 target latent 的反向解释应是什么”；
- 类别分数不是普通 softmax logit，而是预测误差差值：

```text
score_c = dist(z_rebut_c, z_tgt) - dist(z_affirm_c, z_tgt)
```

如果 `affirm_c` 比 `rebut_c` 更能预测 target latent，则 `score_c` 变大。若某个类别只是由采样日历支撑，`x_decoy` 也会给出类似辩词，此时会被 decoy-silence loss 压制。

### 2.2 改 Loss：从不变性转向双辩预测裁判

总目标：

```text
L = L_dialectical_cls
  + lambda_jepa   * L_latent_predict
  + lambda_silent * L_policy_decoy_silence
  + lambda_rebut  * L_counter_rebuttal_margin
  + lambda_ema    * L_target_variance_sobriety
```

#### A. Dialectical Classification `L_dialectical_cls`

用双辩预测差作为分类 logit：

```text
score_c = d(z_rebut_c, z_tgt) - d(z_affirm_c, z_tgt)
L_dialectical_cls = CE(score, y)
```

它不要求不同采样 view 的 logits 一致；同一病程在稀疏政策下 target latent 可能更难预测，score 可以下降。关键是：正确类别的 affirm predictor 必须比其 rebut predictor 更能解释 value target。

#### B. Latent Predictive JEPA `L_latent_predict`

对真实类别的 affirm predictor 做 latent prediction：

```text
L_latent_predict = || normalize(z_affirm_y) - normalize(z_tgt) ||_2^2
```

这吸收 JETS 的 latent-space predictive learning，但不做 raw reconstruction、不做 contrastive negative、不要求 pairwise consistency。

#### C. Policy Decoy Silence `L_policy_decoy_silence`

将 `x_decoy` 输入同一个 context encoder，得到 `score_decoy_c`。如果仅有采样坐标而没有 value evidence，任何类别都不应在双辩中获得明显优势：

```text
L_policy_decoy_silence =
  KL( softmax(score_decoy) || Uniform(K) )
  + mean_c relu(|score_decoy_c| - delta_silent)^2
```

这不是 policy adversarial，因为没有训练表示去欺骗策略判别器；也不是 value-redacted classifier，因为 decoy 的目标是“沉默”，不是预测标签。它直接阻止 MILM/TRIAGE 式“只靠采样行为也能讲出强结论”的捷径。

#### D. Counter-Rebuttal Margin `L_counter_rebuttal_margin`

TRIAGE 的启发不是文本 rationale，而是正反结局都必须被比较。DD-JEPA 要求真实类别的 affirm 优于竞争类别 affirm，同时竞争类别的 rebut 应能解释“为什么不是它”：

```text
L_counter_rebuttal =
  mean_{k != y} relu(m + d_affirm_y - d_affirm_k)^2
  + mean_{k != y} relu(m + d_rebut_k - d_affirm_y)^2
```

它不是 proof/sequent，也不是 jury/social choice；没有规则证明、排序投票或 minority report。它只在 latent prediction space 中训练 outcome-specific pro/contra predictors。

#### E. Target Variance Sobriety `L_target_variance_sobriety`

为了防止 EMA target encoder 塌缩，使用 JETS/JEPA 常见的方差与协方差约束：

```text
L_target_variance =
  mean_dim relu(gamma - std(z_tgt_dim))^2
  + offdiag(cov(z_tgt))^2
```

它只约束 target latent 的信息量，不涉及 policy smoothing、certified radius、privacy noise 或 evidential uncertainty。

### 2.3 改 Dataloader：返回 Dialectical JEPA Views

新增 `DialecticalJEPACollator`，每个 batch 返回：

1. `context_batch`：事实可见 value triplets。
2. `target_batch`：value-grounded 被遮蔽 target tokens，用于 EMA target encoder。
3. `policy_decoy_batch`：value 被遮蔽、仅保留采样坐标的 decoy。
4. `rebuttal_policy_bank`：可选的反事实采样可见性，用于制造更难的反方场景：
   - wearable duty-cycle drop；
   - routine-round ICU 采样；
   - alarm-dense follow-up；
   - panel split / pack；
   - value-pending。
5. `target_mask`：保证 target latent 来自 value tokens，而不是 policy-only tokens。

关键区别：

- 不生成 contrastive positive pairs。
- 不要求 counterfactual views 的 representation / logits / risk 一致。
- 不估计 hazard、density ratio、posterior quotient、DIF、privacy divergence、RG beta 或 synthetic policy-cell。
- 不输出 proof、conformal set、evidential uncertainty、jury vote 或 IRT trait。
- sampling branch 的唯一角色是构造 decoy 与 rebuttal visibility，让采样日历无法单方胜诉。

### 2.4 与当前框架的结合方式

- 现有 value encoder 可替换为 `TripletContextEncoder`。
- 现有 sampling branch 改为 `DecoyViewBuilder`：生成 policy-only decoy 和 target-safe masks，不进入分类头。
- 现有 counterfactual intervention 改为 `RebuttalPolicyBank`：只用于测试某个类别辩词是否仍由 value target 支撑。
- 推理阶段可不使用 target encoder：用训练后的 target prototype memory 或从当前窗口内自举 target anchors 计算 dialectical scores；低分差表示需要补采样。

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


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class TripletEventEncoder(nn.Module):
    """Encode irregular triplets without exposing policy ids to the classifier."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon

        event_x = torch.cat(
            [
                self.var_embed(var_id.clamp_min(0)),
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * mask.unsqueeze(-1)
        seq_h, _ = self.context(event_h)
        seq_h = self.out(seq_h) * mask.unsqueeze(-1)
        pooled = masked_mean(seq_h, mask, dim=1)
        return {"event_state": seq_h, "pooled_state": pooled}


class EMAEncoder(nn.Module):
    """Momentum target encoder wrapper for JEPA-style latent targets."""

    def __init__(self, online: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.target = online
        self.momentum = momentum
        for param in self.target.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        for tgt, src in zip(self.target.parameters(), online.parameters()):
            tgt.data.mul_(self.momentum).add_(src.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def forward(self, batch: dict) -> dict:
        return self.target(batch)


class DialecticalPredictor(nn.Module):
    """Class-conditioned affirm/rebut JEPA predictors."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.affirm = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rebut = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.num_classes = num_classes

    def forward(self, context_state: torch.Tensor) -> dict:
        bsz = context_state.size(0)
        classes = torch.arange(self.num_classes, device=context_state.device)
        class_h = self.class_embed(classes)[None].expand(bsz, -1, -1)
        ctx = context_state[:, None].expand_as(class_h)
        joint = torch.cat([ctx, class_h], dim=-1)
        return {
            "affirm": self.affirm(joint),
            "rebut": self.rebut(joint),
        }


def squared_distance_bank(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (pred - target[:, None]).pow(2).sum(dim=-1)


class DoDialecticalJEPAReferee(nn.Module):
    """Sampling-policy robust classifier via outcome-conditioned latent debates."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.context_encoder = TripletEventEncoder(num_vars, hidden_dim)
        self.target_encoder = EMAEncoder(TripletEventEncoder(num_vars, hidden_dim))
        self.predictor = DialecticalPredictor(hidden_dim, num_classes)
        self.num_classes = num_classes

    def dialectical_scores(self, context_batch: dict, target_batch: dict) -> dict:
        ctx = self.context_encoder(context_batch)
        with torch.no_grad():
            tgt = self.target_encoder(target_batch)
        target_z = tgt["pooled_state"].detach()

        pred = self.predictor(ctx["pooled_state"])
        d_affirm = squared_distance_bank(pred["affirm"], target_z)
        d_rebut = squared_distance_bank(pred["rebut"], target_z)
        scores = d_rebut - d_affirm
        return {
            "scores": scores,
            "d_affirm": d_affirm,
            "d_rebut": d_rebut,
            "target_z": target_z,
            "context_state": ctx["pooled_state"],
            "affirm": pred["affirm"],
            "rebut": pred["rebut"],
        }

    def decoy_silence_loss(self, decoy_batch: dict, target_batch: dict, delta: float = 0.15) -> torch.Tensor:
        decoy = self.dialectical_scores(decoy_batch, target_batch)
        prob = torch.softmax(decoy["scores"], dim=-1)
        uniform = torch.full_like(prob, 1.0 / prob.size(-1))
        kl_to_uniform = F.kl_div(prob.clamp_min(1e-8).log(), uniform, reduction="batchmean")
        amplitude = F.relu(decoy["scores"].abs() - delta).pow(2).mean()
        return kl_to_uniform + amplitude

    def target_variance_loss(self, target_z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        z = (target_z - target_z.mean(dim=0)) / target_z.std(dim=0).clamp_min(1e-6)
        std_loss = F.relu(gamma - target_z.std(dim=0)).pow(2).mean()
        cov = (z.T @ z) / max(z.size(0) - 1, 1)
        cov_loss = off_diagonal(cov).pow(2).mean()
        return std_loss + 0.01 * cov_loss

    def training_loss(
        self,
        batch: dict,
        lambda_jepa: float = 0.35,
        lambda_silent: float = 0.30,
        lambda_rebut: float = 0.20,
        lambda_ema: float = 0.03,
        margin: float = 0.25,
    ) -> dict:
        labels = batch["labels"]
        out = self.dialectical_scores(batch["context_batch"], batch["target_batch"])
        cls_loss = F.cross_entropy(out["scores"], labels)

        true_affirm = out["d_affirm"].gather(1, labels[:, None]).squeeze(1)
        true_rebut = out["d_rebut"].gather(1, labels[:, None]).squeeze(1)
        jepa_loss = true_affirm.mean()

        rival_affirm = out["d_affirm"].masked_fill(
            F.one_hot(labels, self.num_classes).bool(),
            1e4,
        ).min(dim=-1).values
        rebuttal_loss = (
            F.relu(margin + true_affirm - rival_affirm).pow(2)
            + F.relu(margin + true_affirm - true_rebut).pow(2)
        ).mean()

        silent_loss = self.decoy_silence_loss(batch["policy_decoy_batch"], batch["target_batch"])
        variance_loss = self.target_variance_loss(out["target_z"])

        total = (
            cls_loss
            + lambda_jepa * jepa_loss
            + lambda_silent * silent_loss
            + lambda_rebut * rebuttal_loss
            + lambda_ema * variance_loss
        )
        return {
            "loss": total,
            "dialectical_cls_loss": cls_loss.detach(),
            "latent_predict_loss": jepa_loss.detach(),
            "policy_decoy_silence_loss": silent_loss.detach(),
            "counter_rebuttal_loss": rebuttal_loss.detach(),
            "target_variance_loss": variance_loss.detach(),
        }


@torch.no_grad()
def build_dialectical_jepa_views(batch: dict, target_ratio: float = 0.30) -> dict:
    """Create context/target/decoy views for dialectical JEPA training.

    These views are not contrastive positives and are not used for logits
    consistency. The decoy keeps sampling coordinates but removes value support.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = value.shape
    device = value.device

    rand = torch.rand_like(mask)
    target_mask = ((rand < target_ratio).to(mask.dtype) * mask).clamp(0.0, 1.0)
    context_mask = (mask - target_mask).clamp_min(0.0)

    # Keep at least one context token per sample.
    empty_context = context_mask.sum(dim=1) == 0
    if empty_context.any():
        first_idx = mask[empty_context].argmax(dim=1)
        context_mask[empty_context, first_idx] = 1.0
        target_mask[empty_context, first_idx] = 0.0

    context_batch = dict(batch)
    context_batch["event_value"] = value * context_mask
    context_batch["event_mask"] = context_mask
    context_batch["event_time"] = time
    context_batch["event_var_id"] = var_id

    target_batch = dict(batch)
    target_batch["event_value"] = value * target_mask
    target_batch["event_mask"] = target_mask
    target_batch["event_time"] = time
    target_batch["event_var_id"] = var_id

    # Policy-only decoy: sampling coordinates survive, value evidence is silenced.
    decoy_batch = dict(batch)
    decoy_batch["event_value"] = torch.zeros_like(value)
    decoy_batch["event_mask"] = mask
    decoy_batch["event_time"] = time
    decoy_batch["event_var_id"] = var_id
    decoy_batch["measurement_quality"] = torch.zeros_like(value)

    out = dict(batch)
    out["context_batch"] = context_batch
    out["target_batch"] = target_batch
    out["policy_decoy_batch"] = decoy_batch
    out["target_mask"] = target_mask
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `wearable duty-cycle shift`：借鉴 JETS，可穿戴设备因佩戴/充电/固件 duty cycle 改变采样。
   - `routine-vs-alarm shift`：ICU 中固定查房采样与报警后密集复测互换。
   - `panel-pending shift`：panel 同步、拆单异步与 value-pending 互换。
   - `sampling-only shortcut shift`：训练时 policy-only decoy 强预测标签，测试时反转该相关性。

2. **对比方法**
   - JETS-style JEPA 但不做 dialectical predictor / decoy silence。
   - TRIAGE-style LLM/rationale 风格风险预测。
   - MILM value-redacted sampling classifier。
   - 普通 irregular Transformer / STAR-Set / VP-GNN / TCF-style encoder。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DIPF、DRG-SFF、DPPC、DCOFF 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy worst AUROC / AUPRC。
   - decoy win rate：policy-only decoy 是否能让某个类别在双辩中获胜，越低越好。
   - dialectical gap stability：真实类别 `d_rebut - d_affirm` 在 policy shift 下是否保持相对优势。
   - rebuttal failure rate：错误预测中，竞争类别 affirm 是否无法被真实类别 rebuttal 反驳。
   - latent target collapse score：EMA target latent 的维度方差与协方差是否健康。

4. **消融实验**
   - 去掉 `L_policy_decoy_silence`，检查采样日历是否重新成为单方胜诉证据。
   - 去掉 `P_rebut`，只保留 affirm JEPA，验证双辩结构是否必要。
   - 将 latent target 换成 raw reconstruction，验证 JETS 式 latent prediction 是否优于重构噪声。
   - 将 decoy 直接作为分类输入，验证 value-redacted shortcut 的危害。
   - 将 target mask 改为 sampling-policy mask，验证 target 必须 value-grounded 而非 policy-grounded。

## 5. 预期创新性

1. **从去偏/校准/证明转向 latent debate**：不是删除采样信息，也不是输出不确定性或保形集合，而是要求类别声明在 latent target prediction 中同时通过正方和反方检验。
2. **从 raw reconstruction 转向 JEPA target**：吸收 JETS 的 latent predictive learning，避免把设备噪声、佩戴断点、value-pending 或采样日历当作逐点重构目标。
3. **从 LLM dialectical rationale 转向可微双辩预测器**：吸收 TRIAGE 的 competing outcomes 思想，但不依赖文本 rationale；所有裁判发生在 latent representation space。
4. **从 policy-only 利用转向 decoy silence**：采样坐标被允许作为 decoy 出庭，但没有 value target 支撑时必须保持沉默，不能单方赢得分类。
5. **与反事实采样框架低侵入兼容**：counterfactual sampler 只需生成 context / target / decoy / rebuttal visibility；不需要 hazard、density ratio、knockoff、IRT、RG、privacy、conformal 或 synthetic forge。

## 6. 一句话投稿卖点

**DD-JEPA 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样日历制造了缺少反方潜表征支撑的单方类别辩词”的问题，通过 JETS-style latent target prediction、outcome-conditioned affirm/rebut predictors 与 policy-only decoy silence，让分类器只有在观测值能预测目标状态潜表征、且竞争类别反驳失败时才输出高分，从而阻止 routine/alarm、panel、pending、wearable duty-cycle 等采样政策捷径在跨策略部署中单方胜诉。**
