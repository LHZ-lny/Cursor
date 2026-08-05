# Title: Do-Sequent Pathology Proofs：面向采样策略偏移的病理时序蕴含证明分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `**/*summary*.md`、`**/*Summary*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md` 以及其中记录的历史提案摘要，包括但不限于 2026-06-12 至 2026-08-04 的机制黑名单。
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
- 已读取近期论文记录 `paper_daily.md`，重点纳入 2026-08-02 追加的：
  - PULSE: Benchmarking Large Language Models for ICU Time Series Classification
  - Time-Conditioned Foreseeing: An EHR-Specific Foundation Model for Irregular Dynamics and Calendrical Time

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
26. counterfactual sampling instruments、first-stage structural equation、control-function residualized structure readout、weak-instrument guard。
27. counterfactual policy jury、Borda / majority rank tribunal、no-dictator / no-structural-bribery loss。
28. Krylov policy subspace、annihilating polynomial、policy-mode annihilation。
29. determinantal / Nystrom state-volume basis、policy-overload logdet sobriety。
30. tropical support routes、soft-min bottleneck、route survival under policy。
31. fixed clinical viva question bank、canonical answer sheet、source-chart-to-answer translator。

本提案选择新的正交切入点：**不再把采样鲁棒性表述为表示不变、校准集合、概率因子剥离、图结构拆分、固定问答或排序裁决，而是把分类声明变成一组可微的“病理时序蕴含证明”。采样政策可以让某些前提不可证明，但不能把“不可证明”伪装成“可证明的类别证据”。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 最新记录中，PULSE 和 TCF 给了两个对当前 AAAI 方向很关键的启发。

第一，**PULSE** 把 ICU time-series classification 放到 HiRID、MIMIC-IV、eICU 等跨中心环境中审计，说明 sampling-policy shift 不应只在单数据集内用随机 mask 模拟。跨中心部署时，模型面对的是一整套护理流程、变量定义、采样频率、事件记录习惯和终点标签分布的变化。一个模型即使在院内强，也可能在新中心因依赖采样流程 shortcut 而产生高置信错误。

第二，**TCF** 的 Pathology-Focused Binning 与 Time-Conditioned Foreseeing 提醒我们：EHR 的数值不是普通 token，异常高低值、病理区间、未来时间条件和临床事件时间共同决定语义。单纯预测下一个事件或把时间戳加进 RoPE 不够；模型需要理解“某个病理状态在某个临床时间窗内是否被支持”。

当前“采样解耦/反事实干预”框架已经把 value process 与 sampling process 分开，但多数历史方案仍围绕：

- 如何删除、折扣、平滑、校准或审计采样信息；
- 如何让不同采样策略下的表示、风险、边际或预测声明更稳定；
- 如何把采样政策做成图、测度、后验、证据、拓扑、gauge、syndrome、jury 或固定问答。

**Do-Sequent Pathology Proofs (DSPP)** 换一个角度：

> 真实部署中，医生或科学系统并不只需要一个概率分数，而需要知道“这个类别声明由哪些病理前提蕴含出来”。采样政策偏移最危险的地方，不是某个证据消失，而是模型把“由于训练医院恰好测了某项指标”误当作“该病理前提被证明”。因此，鲁棒分类器应输出 proof-carrying class claim：只有当 pathology-focused temporal sequents 的前提在当前观测下可证明、且在反事实采样审计下没有被观测反例推翻时，该前提才允许贡献类别分数。

这里的核心不是固定 clinical question bank，也不是 VQ semantic clauses。DSPP 学习的是 **类别级时序蕴含规则**：

```text
若在相对临床时间窗 W1 中出现病理 bin A，
且随后在时间窗 W2 中变量 B 进入异常区间，
且不存在已观测的反证 bin C，
则类别 y 的证明强度提高。
```

采样政策可以让某些前提未观测；DSPP 把它们标记为 **unproved**，而不是当作 false，也不是把它们转成 uncertainty mass 或 conformal set。采样政策只有在提供真实 pathology-bin 前提或观测反例时，才影响证明；仅有“某变量被测了/没测”不能直接成为证明。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从连续 logits 改为 Pathology Sequent Encoder

DSPP 保留一个普通 value encoder，但在分类读出前加入 pathology sequent 层。

1. **Pathology Atomizer**
   - 吸收 TCF 的 Pathology-Focused Binning：每个观测值先被映射到变量特异的病理区间，例如 low / normal / high / critical。
   - 每个 atom 包含 `(variable, pathology_bin, relative_time, value_strength, observed_flag)`。
   - 时间编码可使用 TCF 式绝对/相对临床时间，但不作为自适应 reference point；它只帮助判断某个前提是否落在 sequent 的时间窗内。

2. **Differentiable Sequent Bank**
   - 对每个类别学习若干 soft sequents。
   - 每条 sequent 包含多个 antecedent literals 和少量 refuter literals：

```text
antecedent: 需要被证明的病理前提
refuter:    若已观测为真，则推翻该条证明
```

   - 每个 literal 由变量分布、病理 bin 分布、相对时间窗、极性和最小观测质量构成。
   - sequent 的证明分数来自 antecedent 的 soft Lukasiewicz conjunction；可用性分数来自这些前提在当前采样策略下是否真的被观测到。

3. **Proof-Carrying Readout**
   - 分类 logit 不直接来自采样 mask 或 policy signature，而来自通过 proof gate 的 sequent：

```text
class_logit_y = base_logit_y + sum_j proof_strength_{y,j} * no_refutation_{y,j}
```

   - 若某条 sequent 的关键前提未观测，它不会变成负证据；它只是不能为类别提供证明。
   - 若反事实采样视图中出现观测反例，则该 sequent 的类别贡献被压低。

### 2.2 改 Sampling Branch：从 policy 表征改为 Proof Availability Auditor

sampling branch 不输出策略标签、不进入 logits、不生成 jury 或 conformal sleeves。它只负责计算每条 sequent 在当前采样结构下的 **proof availability**：

- `provable`: antecedent 的必要病理 literal 至少有可用观测支持；
- `unproved`: 关键 literal 因采样政策未观测，不能贡献正证明；
- `refuted`: refuter literal 已观测为真，必须压制该 sequent；
- `policy-only`: 只有采样坐标变化、没有病理 bin 支持，不能成为证明。

这与 sampling decoupling 框架结合得很自然：

- value process 产生 pathology atom truth；
- sampling process 只决定 sequent 是否有资格被证明；
- counterfactual intervention 生成不同 sampling visibility 下的 proof audit；
- classifier 只消费 proof-carrying pathology evidence。

### 2.3 改 Loss：从一致性/校准/排序转向 Proof Obligation Discipline

总目标：

```text
L = L_cls
  + lambda_proof * L_true_class_proof
  + lambda_refute * L_counterfactual_refutation
  + lambda_avail * L_availability_sobriety
  + lambda_anchor * L_pathology_anchor
```

#### A. 分类损失 `L_cls`

事实视图下使用 proof-carrying logits 训练：

```text
L_cls = CE(logits_proof, y)
```

#### B. True-Class Proof Obligation `L_true_class_proof`

要求真实类别至少有若干条可证明 sequent 支撑，而不是只靠 base logit：

```text
proof_y = sum_j proof_strength_{y,j} * availability_{y,j} * no_refutation_{y,j}
L_true_class_proof = relu(m_proof - proof_y)^2
```

这不是 evidence market：没有 token 价格、预算或拍卖；它只要求类别声明带有可检查的病理时序证明。

#### C. Counterfactual Refutation Loss `L_counterfactual_refutation`

反事实采样干预生成若干 policy audit views。DSPP 不要求这些 views 的 logits 一致，也不要求风险方差小。它只检查：

- 若某条真实类 sequent 在事实视图中贡献很高；
- 但在某个反事实采样视图中出现了已观测 refuter；
- 则该 sequent 不应继续高贡献。

```text
L_refute = mean_r,j proof_weight_j * observed_refuter_{r,j}
```

这能阻止模型把训练医院特有的“采样可见性”当作证明，而保留真正由 pathology bins 支撑的规则。

#### D. Availability Sobriety `L_availability_sobriety`

防止模型使用未证明前提：

```text
L_availability_sobriety =
  mean_{y,j} relu(proof_strength_{y,j} - availability_{y,j} - eps)^2
```

含义是：如果某条 sequent 的病理前提因为采样策略不可见，它就不能凭空产生类别证明。

#### E. Pathology Anchor Loss `L_pathology_anchor`

防止 sequent 退化成“只看时间窗、变量是否被测、panel 是否出现”的 policy-only 规则。每条高权重 sequent 必须至少锚定一个 pathology bin literal：

```text
anchor_mass = sum_l sequent_weight_l * pathology_bin_specificity_l
L_anchor = mean relu(anchor_min - anchor_mass)^2
```

这与 DCVE 的固定问题不同：DSPP 不预设问卷，也不把所有样本翻译成统一 answer sheet；它学习类别相关的 temporal pathology entailments，并用 proof audit 防止采样坐标成为证明。

### 2.4 改 Dataloader：返回 Proof Audit Views

新增 `SequentProofAuditCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `pathology_bin_id` 或连续 `bin_strength`。
3. `measurement_quality`：来自测量误差、重复观测稳定性或 TCF 式病理 bin 可信度。
4. `proof_audit_view_bank`：
   - `value_pending`: 采样事件可见但 value/bin 不可用，测试模型是否把“已下单”当证明；
   - `routine_round`: 时间戳被规整为固定查房式，测试时间窗前提是否仍可证明；
   - `center_panel_split`: 联测 panel 被拆成异步观测，测试 sequent 是否依赖共现；
   - `critical_bin_drop`: 移除少数 critical bin 观测，检查类别证明是否诚实消失。
5. `audit_visibility`：每个 view 下哪些 antecedent/refuter literal 可被证明、不可证明或被反驳。

### 2.5 与 PULSE / TCF 的结合方式

- **来自 TCF**：Pathology-Focused Binning、临床相对时间、未来时间条件启发，被转化为 pathology atoms 和 temporal sequents。
- **来自 PULSE**：跨中心 shift 不再只报告 AUROC 退化，而是检查不同中心下：
  - 哪些 class claims 仍有 proof；
  - 哪些预测只剩 unproved antecedents；
  - 哪些中心的采样流程制造了 policy-only pseudo-proof。
- **与现有反事实框架结合**：反事实采样模块不再输出一致性 pair、jury、conformal sleeves 或 viva answer sheet，只输出 proof audit views，用于判断某条病理蕴含是否有资格作为类别证据。

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


class PathologyAtomizer(nn.Module):
    """Convert irregular events into soft pathology-bin atoms."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.0, 2.0, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.atom_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_bins + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        event_mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * event_mask.unsqueeze(-1)

        horizon = (time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_h = self.var_embed(var_id)
        atom_x = torch.cat(
            [
                var_h,
                bin_prob,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        atom_h = self.atom_proj(atom_x) * event_mask.unsqueeze(-1)
        return {
            "atom_h": atom_h,
            "bin_prob": bin_prob,
            "time_norm": time_norm,
            "event_mask": event_mask,
            "quality": quality,
        }


class TemporalLiteralBank(nn.Module):
    """Learn soft pathology literals used by class-level sequents."""

    def __init__(self, num_classes: int, num_sequents: int, num_literals: int, num_vars: int, num_bins: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_sequents = num_sequents
        self.num_literals = num_literals
        self.num_vars = num_vars
        self.num_bins = num_bins

        self.var_logits = nn.Parameter(torch.randn(num_classes, num_sequents, num_literals, num_vars) * 0.02)
        self.bin_logits = nn.Parameter(torch.randn(num_classes, num_sequents, num_literals, num_bins) * 0.02)
        self.time_center = nn.Parameter(torch.rand(num_classes, num_sequents, num_literals))
        self.time_width = nn.Parameter(torch.zeros(num_classes, num_sequents, num_literals))
        self.literal_weight = nn.Parameter(torch.zeros(num_classes, num_sequents, num_literals))
        self.refuter_weight = nn.Parameter(torch.zeros(num_classes, num_sequents, num_literals))

    def literal_truth(self, atom: dict, event_var_id: torch.Tensor) -> dict:
        """Return antecedent and refuter truth for every class/sequent/literal."""

        # Shapes:
        # bin_prob: [B, N, Bin]
        # output: [B, C, S, L]
        bin_prob = atom["bin_prob"]
        time_norm = atom["time_norm"]
        event_mask = atom["event_mask"]
        quality = atom["quality"]

        var_select = torch.softmax(self.var_logits, dim=-1)
        bin_select = torch.softmax(self.bin_logits, dim=-1)

        event_var_onehot = F.one_hot(event_var_id.clamp(0, self.num_vars - 1), self.num_vars).to(bin_prob.dtype)
        var_match = torch.einsum("bnv,cslv->bncsl", event_var_onehot, var_select)
        bin_match = torch.einsum("bnd,csld->bncsl", bin_prob, bin_select)

        center = self.time_center.sigmoid()
        width = F.softplus(self.time_width) + 0.05
        time_gate = torch.exp(-0.5 * ((time_norm[:, :, None, None, None] - center[None, None]) / width[None, None]).pow(2))

        event_truth = var_match * bin_match * time_gate * quality[:, :, None, None, None]
        event_truth = event_truth * event_mask[:, :, None, None, None]

        # A literal is proven if any observed event satisfies it.
        truth = 1.0 - torch.prod(1.0 - event_truth.clamp(0.0, 0.95), dim=1)
        availability = (event_truth > 0.05).to(bin_prob.dtype).amax(dim=1)
        return {"literal_truth": truth, "literal_available": availability}

    def sequent_scores(self, literal_truth: torch.Tensor, literal_available: torch.Tensor) -> dict:
        antecedent_w = torch.sigmoid(self.literal_weight)
        refuter_w = torch.sigmoid(self.refuter_weight)

        # Lukasiewicz-style soft conjunction. Missing antecedents reduce availability
        # but are not treated as observed false refuters.
        antecedent_violation = antecedent_w[None] * (1.0 - literal_truth)
        proof_strength = F.relu(1.0 - antecedent_violation.sum(dim=-1))

        availability_violation = antecedent_w[None] * (1.0 - literal_available)
        proof_availability = F.relu(1.0 - availability_violation.sum(dim=-1))

        refuter_strength = (refuter_w[None] * literal_truth).sum(dim=-1).clamp(0.0, 1.0)
        no_refutation = 1.0 - refuter_strength

        safe_proof = proof_strength * proof_availability * no_refutation
        return {
            "proof_strength": proof_strength,
            "proof_availability": proof_availability,
            "refuter_strength": refuter_strength,
            "safe_proof": safe_proof,
            "antecedent_weight": antecedent_w,
            "refuter_weight": refuter_w,
        }


class DoSequentPathologyProofs(nn.Module):
    """Proof-carrying classifier for sampling-policy robust irregular TS classification."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        hidden_dim: int,
        num_classes: int,
        num_sequents: int = 6,
        num_literals: int = 4,
    ):
        super().__init__()
        self.atomizer = PathologyAtomizer(num_vars, num_bins, hidden_dim)
        self.literal_bank = TemporalLiteralBank(num_classes, num_sequents, num_literals, num_vars, num_bins)
        self.base_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.base_head = nn.Linear(hidden_dim, num_classes)
        self.proof_readout = nn.Parameter(torch.randn(num_classes, num_sequents) * 0.02)
        self.num_classes = num_classes

    def forward(self, batch: dict) -> dict:
        atom = self.atomizer(batch)
        seq_h, _ = self.base_encoder(atom["atom_h"])
        pooled = masked_mean(seq_h, atom["event_mask"], dim=1)
        base_logits = self.base_head(pooled)

        literal = self.literal_bank.literal_truth(atom, batch["event_var_id"])
        sequent = self.literal_bank.sequent_scores(
            literal_truth=literal["literal_truth"],
            literal_available=literal["literal_available"],
        )

        proof_weight = F.softplus(self.proof_readout)
        proof_contrib = (proof_weight[None] * sequent["safe_proof"]).sum(dim=-1)
        proof_logits = base_logits + proof_contrib

        return {
            "logits": proof_logits,
            "base_logits": base_logits,
            "literal_truth": literal["literal_truth"],
            "literal_available": literal["literal_available"],
            **sequent,
        }

    def training_loss(
        self,
        batch: dict,
        lambda_proof: float = 0.20,
        lambda_refute: float = 0.25,
        lambda_avail: float = 0.15,
        lambda_anchor: float = 0.05,
        proof_margin: float = 0.35,
        availability_eps: float = 0.05,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)

        proof_y = out["safe_proof"].gather(
            1,
            labels[:, None, None].expand(-1, 1, out["safe_proof"].size(-1)),
        ).squeeze(1)
        true_proof_mass = proof_y.sum(dim=-1)
        proof_loss = F.relu(proof_margin - true_proof_mass).pow(2).mean()

        # A high proof score is illegal when the antecedents are not available.
        avail_violation = F.relu(out["proof_strength"] - out["proof_availability"] - availability_eps)
        availability_loss = avail_violation.pow(2).mean()

        refute_losses = []
        for audit_batch in batch.get("proof_audit_view_bank", []):
            audit_out = self.forward(audit_batch)
            factual_weight = proof_y.detach()
            audit_refuter = audit_out["refuter_strength"].gather(
                1,
                labels[:, None, None].expand(-1, 1, audit_out["refuter_strength"].size(-1)),
            ).squeeze(1)
            refute_losses.append((factual_weight * audit_refuter).mean())
        if refute_losses:
            refute_loss = torch.stack(refute_losses).mean()
        else:
            refute_loss = torch.zeros((), device=labels.device)

        # Each useful sequent must be anchored in at least one pathology-bin-specific literal.
        bin_specificity = torch.softmax(self.literal_bank.bin_logits, dim=-1).amax(dim=-1).values
        antecedent_w = torch.sigmoid(self.literal_bank.literal_weight)
        anchor_mass = (antecedent_w * bin_specificity).sum(dim=-1)
        anchor_loss = F.relu(0.60 - anchor_mass).pow(2).mean()

        total = (
            cls_loss
            + lambda_proof * proof_loss
            + lambda_refute * refute_loss
            + lambda_avail * availability_loss
            + lambda_anchor * anchor_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "true_class_proof_loss": proof_loss.detach(),
            "counterfactual_refutation_loss": refute_loss.detach(),
            "availability_sobriety_loss": availability_loss.detach(),
            "pathology_anchor_loss": anchor_loss.detach(),
            "mean_true_proof_mass": true_proof_mass.mean().detach(),
        }


@torch.no_grad()
def build_sequent_proof_audit_views(batch: dict) -> list[dict]:
    """Create proof audit views.

    These are not consistency pairs. They test whether a pathology sequent is
    provable, unproved, or refuted under plausible sampling policies.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    quality = batch.get("measurement_quality", torch.ones_like(value))
    bsz, num_events = value.shape
    device = value.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, new_quality):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        out["measurement_quality"] = new_quality
        return out

    views = []

    # 1. Value-pending: order event is visible, pathology value is not yet provable.
    pending_mask = mask
    pending_value = torch.zeros_like(value)
    pending_quality = quality * 0.05
    views.append(clone_with(pending_value, time, var_id, pending_mask, pending_quality))

    # 2. Routine-round policy: sampling time becomes coarse clinical rounds.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    views.append(clone_with(value * mask, rounded_time, var_id, mask, quality))

    # 3. Center panel split: remove near-synchronous cross-variable observations.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    split_mask = mask * (1.0 - 0.5 * close * changed_var)
    views.append(clone_with(value * split_mask, time, var_id, split_mask, quality * split_mask))

    # 4. Critical-bin drop proxy: remove high-magnitude rare observations.
    abs_value = value.abs() * mask
    threshold = abs_value.quantile(0.80, dim=1, keepdim=True)
    critical = (abs_value >= threshold).to(mask.dtype)
    critical_drop = mask * (1.0 - critical)
    views.append(clone_with(value * critical_drop, time, var_id, critical_drop, quality * critical_drop))

    return views
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cross-center routine shift`：借鉴 PULSE，在 HiRID / MIMIC-IV / eICU 风格采样频率之间切换。
   - `value-pending shift`：化验事件已出现但值尚未返回，测试模型是否把“下单”误当 proof。
   - `panel-split shift`：训练中心同步 panel，测试中心拆成异步事件。
   - `critical-bin visibility shift`：某中心更频繁捕捉 critical bins，另一中心低频观测导致前提不可证明。

2. **对比方法**
   - 普通 irregular Transformer / mTAND / STAR-Set / VP-GNN。
   - TCF-style foreseeing backbone without proof layer。
   - PULSE 中强传统基线与 LLM prompt / hybrid baseline。
   - 历史方案：C-CRS、DJRT、DCVE、PSSC、PIIES、PLSM、ST-FDN、CKCF、SCSC 等。

3. **核心指标**
   - in-policy accuracy。
   - cross-center worst-policy accuracy。
   - proof coverage：真实类预测中有多少带至少一条可证明 sequent。
   - unproved-positive rate：高置信预测却缺少可证明前提的比例。
   - refuted-proof rate：反事实 audit view 中被观测反例推翻的 sequent 比例。
   - policy-only proof leakage：只靠观测可见性而无 pathology-bin anchor 的证明比例。

4. **消融实验**
   - 去掉 `L_true_class_proof`，检查是否回到普通 softmax shortcut。
   - 去掉 `L_counterfactual_refutation`，检查事实视图高贡献 sequent 是否被反事实观测反例推翻。
   - 去掉 `L_availability_sobriety`，检查未观测前提是否被模型当作正证明。
   - 去掉 `L_pathology_anchor`，检查 sequent 是否退化成 sampling-calendar 规则。
   - 将 pathology-focused bins 替换为均匀分箱，验证 TCF 式病理分箱的价值。

## 5. 预期创新性

1. **从概率预测转向证明携带预测**：模型输出的不只是 logits，而是由 pathology-focused temporal sequents 支撑的 class claim。
2. **从采样不变转向证明资格审计**：反事实采样不再用于一致性、风险、平滑、校准集合或陪审团，而是用于判断一条病理蕴含是否 provable / unproved / refuted。
3. **从 fixed viva 转向 learned sequent calculus**：不同于固定临床问答表，DSPP 学习类别相关的病理时序蕴含，并要求每条高贡献规则锚定病理 bin。
4. **从 value-redacted shortcut 转向 proof availability**：采样事件本身可以告诉模型“可能需要检查某项指标”，但只有 pathology value/bin 被观测并满足前提时，才能构成证明。
5. **与 PULSE / TCF 自然耦合**：PULSE 提供跨中心 policy-shift 评测场；TCF 提供病理分箱和时间条件语义；DSPP 则把二者推进到可解释、可审计的证明鲁棒分类机制。

## 6. 一句话投稿卖点

**DSPP 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“类别声明的病理时序前提是否可证明”的问题，通过 pathology-focused atomizer、learned temporal sequent bank 与 counterfactual proof audit，让模型只在观测值真正证明病理蕴含时输出类别证据，从而避免把跨中心采样日历、value-pending、panel 共现或 critical-bin 可见性误当作可迁移诊断证明。**
