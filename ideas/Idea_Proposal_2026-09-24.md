# Title: Do-RIP Sparse Pathology Lens：面向采样策略偏移的受限等距病理稀疏透镜

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md` 及 2026-09-15 至 2026-09-23 的最新记忆条目，纳入未必完整检出于工作区的历史机制。
- 已读取/抽取 `paper_daily.md` 的近期记录，重点覆盖 RoMAE、SLAN、CHARM、TimeCHEAT、INTERVenE、SLiCE、MedFuse 等 9 月新增机制。
- 已读取当前工作区历史提案与标题/核心卖点索引，覆盖 `ideas/Idea_Proposal_2026-06-12.md` 至 `ideas/Idea_Proposal_2026-09-23.md` 的历史方向。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下核心机制作为主创新：

1. 危险率 point-process、采样 score 零空间、hazard-driven resampling、do-risk variance。
2. 生理流/采样算子交换子、value/policy graph commutator、policy residual sink。
3. additive evidence market、protocol tax、token budget、边际证据审计。
4. posterior quotient、模型空间采样似然因子相除、policy marginalization。
5. reconstruction error cartography、ANOVA projection、VQ clauses、HSIC redaction。
6. policy-simplex randomized smoothing、certified radius、Dirichlet/logit-normal do-sampler。
7. Radon-Nikodym density ratio、doubly robust target-measure correction、influence bound。
8. optional-stopping martingale、standardized innovation、stopping recipe moment control。
9. soft excursion topology、censored persistence interval、censor envelope、fragmentation sobriety。
10. policy gauge frame、horizontal transport、chart span supervision、vertical blindness。
11. policy shadow film、latent eraser/stencil、negative-film high-entropy silence。
12. parity-check codeword、syndrome locator、packet repair decoder。
13. conditional knockoff calendar、soft knockoff-FDR、swap symmetry。
14. observability witness、measurement Jacobian/Fisher gate、low-observability routing。
15. subjective-logic evidential shield、policy-induced vacuity、class-wise discount。
16. observation-set policy lattice、meet/join masks、monotone/submodular margin。
17. solver trace front-door、NFE/roughness mediator、reference trace bank。
18. conformal risk sleeves、policy-conditional nonconformity set、leave-policy-out calibration。
19. social-choice rank tribunal、Borda jury、no-dictator / no-bribery losses。
20. synthetic falsification forge、orthogonal array SCM pretraining、policy-cell DRO。
21. policy privacy cloak、Renyi policy leakage、Gaussian privacy release。
22. PID semantic prism、cross-representation redundant ray、policy-synergy suppression。
23. Noether semantic action、policy work、semantic charge conservation。
24. meta workflow immunization、fast/slow adapter split、source-atlas antipodal retrieval。
25. collider seal、marked-event explain-away gate、workflow-swap path-specific seal。
26. palimpsest memory、repeated-write fatigue、write-budget audit。
27. renewal exposure meter、reward-per-exposure readout、renewal-balance residual。
28. SSA observation program、type checking、dead-code elimination、phi-node policy instructions。
29. Robin boundary / Green solver、Blackwell experiment sieve、elicitable functional compass。
30. Lyapunov interval observer、policy impulse work、counterfactual contraction.
31. BBP / Marchenko-Pastur random-matrix bulk edge、semantic spike readout。
32. DeepFRC-style Schwarzian warp canonicalizer、minimal-curvature biological phase registration。
33. 单纯 state-policy 双分支、对抗环境分类器、跨视图 logits/representation 一致性、频域掩码对比学习、missingness pattern 直接分类。

本提案选择新的正交切入点：**不把采样政策建成概率、图边、日历、边界、程序、记忆、校准集合或随机矩阵谱；而是把每次非规则观测看作对一个稀疏病理原子向量的线性/局部线性测量。采样政策改变的是测量矩阵的行选择和相干性。分类器不读取 raw mask 或 policy descriptor，而只读取通过受限等距约束可恢复的 sparse pathology code。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 的几条机制给出一个共同信号，但历史 proposal 还没有把它们统一成“可恢复性几何”：

- **MedFuse** 强调单个事件 token 中 value 与 feature identity 的乘法融合很关键；但同一个 value-feature 组合在不同医院测量政策下可能代表不同先验风险。
- **CHARM** 提示通道语义、单位、设备描述会影响跨数据集迁移；但通道是否存在、何时出现，也可能是政策痕迹。
- **TimeCHEAT** 说明局部跨通道借用很有用；但局部 CD 边权会被医院联测流程牵引。
- **SLiCE** 让多策略连续时间编码更可扩展；但高效主干仍可能高效吸收采样 shortcut。
- **RoMAE/SLAN** 说明连续位置编码和 switch action 很适合不规则观测；但位置和开关本身也可能泄漏采样政策。

这些工作都在回答“如何更好利用不规则观测”。Sampling-policy shift 下更本质的问题是：

> 当前采样政策提供的观测集合，是否足以从底层病理原子中稳定恢复分类所需的稀疏支持？如果不能，模型是否正在把测量矩阵本身的相干结构误当成类别证据？

在医疗和可穿戴场景中，真实病理证据往往是稀疏的：少数变量、少数时间窗、少数趋势组合决定风险。采样政策可以改变观测矩阵 `A_pi`：

- routine policy 产生相对均匀但低分辨率的测量行；
- alarm policy 在风险窗口密集追加高度相似的行；
- panel policy 产生一批强相关变量的共同行；
- device-budget policy 删除某些变量或拉长采样间隔。

如果分类器直接消费所有事件 token，它会把 `A_pi` 的行相干性、重复测量和联测结构当作类别捷径。**Do-RIP Sparse Pathology Lens (DRIP-SPL)** 的核心假设是：

```text
观测事件 y_pi ≈ A_pi c_state + noise
```

其中 `c_state` 是稀疏病理原子系数，`A_pi` 由当前采样政策决定。只有当 `A_pi` 对真实支持集近似满足受限等距性质，`c_state` 才可稳定恢复。模型的任务不是删除采样信息，而是学习一个“测量透镜”：将事件流转成测量矩阵，恢复稀疏病理 code，并把高相干、低可恢复性的政策结构隔离在 recovery diagnostic 中。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 负责产生 value-conditioned measurement rows；
- sampling process 负责描述测量矩阵的 coherence / RIP risk，但不进入分类头；
- counterfactual intervention 负责生成不同采样政策下的测量矩阵，训练稀疏恢复器在可恢复范围内稳定，而不是做 logits 一致、投票、校准或对抗。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从事件 pooling 改为 Measurement Operator Builder

每个观测事件 `(t_i, d_i, x_i, q_i)` 被提升成一行测量向量 `a_i`：

```text
a_i = RowBuilder(value=x_i, variable=d_i, time=t_i, quality=q_i, channel_text=d_desc)
```

设计上吸收 MedFuse 与 CHARM，但不直接把 token embedding 输入分类器：

1. **Value-feature multiplicative row**
   - `feature_embedding(d_i)` 与 `value_embedding(x_i)` 做 Hadamard modulation，得到变量条件化的数值语义。
   - 与 MedFuse 的区别：该 embedding 不是分类 token，而是测量矩阵的一行。

2. **Channel semantic conditioning**
   - 可选输入变量描述、单位、设备或科室上下文，生成 row basis selection。
   - 与 CHARM 的区别：通道语义只影响“该观测测量哪些病理原子”，不直接作为类别证据。

3. **Continuous coordinate row phase**
   - 使用轻量 continuous RoPE / log-delta-time 形成时间相位，但只用于测量行。
   - 与 RoMAE 的区别：不重构 masked token，也不让 positional code 单独预测类别。

最终得到：

```text
A_pi in R^{N_obs x K_atoms}
y_pi in R^{N_obs}
c_hat = SparseDecoder(A_pi, y_pi)
logits = Classifier(c_hat)
```

分类器只读取稀疏病理系数 `c_hat`，不读取 `A_pi`、mask、delta-t summary、policy descriptor 或 coherence diagnostic。

### 2.2 改 Sparse Decoder：Unrolled ISTA / LISTA 病理恢复器

使用可微稀疏恢复层：

```text
c^{l+1} = SoftThreshold(c^l + eta A^T (y - A c^l), lambda)
```

优点：

- 可解释：`c_hat` 的非零项对应病理原子；
- 低侵入：可以包裹任意事件 encoder；
- 与采样政策关系清晰：采样政策只通过 `A_pi` 改变恢复条件，而不直接变成分类 evidence。

### 2.3 改 Loss：从一致性/对抗转向 Restricted-Isometry Recovery Contracts

总目标：

```text
L = L_cls
  + lambda_rec  * L_measurement_reconstruction
  + lambda_rip  * L_support_rip
  + lambda_coh  * L_policy_coherence_sobriety
  + lambda_stab * L_sparse_support_recoverability
```

#### A. 分类损失 `L_cls`

只用恢复后的稀疏病理 code：

```text
L_cls = CE(Classifier(c_hat_factual), y)
```

#### B. 测量重构 `L_measurement_reconstruction`

让 `A_pi c_hat` 能解释实际观测值：

```text
L_rec = || M_obs * (A_pi c_hat - y_pi) ||_1
```

这保证 `c_hat` 不是普通 embedding，而是能被观测测量矩阵解释的病理 code。

#### C. 支持集 RIP 约束 `L_support_rip`

对当前 batch 恢复出的 top-k 支持集 `S`，检查事实与反事实采样政策下的测量矩阵是否在该支持集上近似等距：

```text
G_S = A_S^T A_S
L_rip = || G_S - I ||_F^2
```

直觉：如果某个采样政策只提供重复、共线、联测的观测行，`G_S` 会高相干，稀疏病理 code 不可稳定恢复；模型不能把这种高相干当成类别捷径，而必须通过 loss 促使 RowBuilder 学会把政策重复行压成低增益测量。

这不同于 observability witness：不是按坐标估计 Fisher/Jacobian 可观测性；也不同于 parity code：不是把表示变成 codeword 并修复错误包；它直接约束“采样政策诱导的测量矩阵是否能稳定恢复稀疏病理原子”。

#### D. Policy Coherence Sobriety `L_policy_coherence_sobriety`

对明显由政策造成的重复测量、panel 共测或短间隔 burst，惩罚测量行之间的过高互相干：

```text
mu_policy = max_{i != j, same_policy_group} |<a_i, a_j>|
L_coh = relu(mu_policy - mu_max)^2
```

它不是 protocol tax，也不是证据预算。模型可以保留高价值观测；只是重复政策行不应在测量矩阵中变成多个几乎相同但共同放大某个类别的行。

#### E. Sparse Support Recoverability `L_sparse_support_recoverability`

反事实采样模块生成多种 `A_pi^r`。不要求不同视图 logits 或 representation 相同，只要求在每个视图的可恢复支持上，top-k 病理原子不被纯政策行创造出来：

```text
support_gain_r = positive_part(|c_hat_r| - recoverability_r)
L_stab = mean support_gain_r outside factual_value_supported_atoms
```

这里 `recoverability_r` 由 `diag((A_S^T A_S)^{-1})` 或 Gram 条件数估计。若某个反事实政策视图因为 panel 同步或 alarm burst 产生新的大系数，但测量矩阵本身条件数很差，该系数会被判定为 policy-created sparse atom。

### 2.4 改 Dataloader：返回 Measurement Policy Bank

新增 `RIPSparsePolicyCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`、`measurement_std`。
2. `channel_description_id` 或变量单位/设备文本 embedding 索引。
3. `policy_group_id`：routine、alarm-burst、panel、variable-budget、device-duty、pending 等观测组，仅用于 coherence audit。
4. `counterfactual_measurement_bank`：
   - `panel_decorrelate`：拆开联测 panel；
   - `alarm_burst_thin`：稀疏化报警后重复测量；
   - `variable_budget_drop`：删除低预算变量；
   - `routine_round_snap`：将观测映射到固定查房节律；
   - `device_duty_cycle`：模拟可穿戴低频 duty cycle。
5. `recoverability_mask`：标记哪些观测仍有足够 value/quality 支持，防止把真实高价值稀有观测误判为政策行。

这些反事实视图不构成对比正样本，不做风险方差、不做投票、不做保形、不做隐私、不做程序 rewrite；它们只用于估计测量矩阵的相干性、条件数和稀疏恢复稳定性。

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


class MeasurementRowBuilder(nn.Module):
    """Build value-conditioned measurement rows for sparse pathology recovery."""

    def __init__(self, num_vars: int, num_atoms: int, hidden_dim: int, desc_dim: int = 0):
        super().__init__()
        self.num_atoms = num_atoms
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.desc_proj = nn.Linear(desc_dim, hidden_dim) if desc_dim > 0 else None
        self.row_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atoms),
        )
        self.value_readout = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp_min(0)
        mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(value))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_h = self.var_embed(var_id)
        value_x = torch.stack(
            [value, torch.log1p(delta_t), torch.log1p(meas_std)],
            dim=-1,
        )
        value_h = self.value_proj(value_x)

        # Multiplicative value-feature semantics become measurement-row semantics,
        # not direct classification features.
        h = var_h * (1.0 + torch.tanh(value_h))
        if self.desc_proj is not None and "channel_desc" in batch:
            h = h + self.desc_proj(batch["channel_desc"])

        # Continuous position phase modulates row orientation but does not enter logits.
        phase = torch.stack(
            [torch.sin(time_norm), torch.cos(time_norm)],
            dim=-1,
        ).mean(dim=-1, keepdim=True)
        h = h * (1.0 + 0.1 * phase)

        row = self.row_head(h)
        row = F.normalize(row, p=2, dim=-1) * mask.unsqueeze(-1)
        y_value = self.value_readout(h).squeeze(-1) * mask
        return {"A": row, "y_proxy": y_value}


def soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * F.relu(x.abs() - threshold)


class UnrolledSparseDecoder(nn.Module):
    """LISTA-style sparse recovery from irregular measurement rows."""

    def __init__(self, num_atoms: int, num_steps: int = 6):
        super().__init__()
        self.log_step = nn.Parameter(torch.zeros(num_steps))
        self.log_lam = nn.Parameter(torch.full((num_steps,), -2.0))
        self.num_atoms = num_atoms
        self.num_steps = num_steps

    def forward(self, A: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # A: [B, N, K], y/mask: [B, N]
        c = torch.zeros(A.size(0), self.num_atoms, device=A.device, dtype=A.dtype)
        y = y * mask
        for idx in range(self.num_steps):
            pred = torch.einsum("bnk,bk->bn", A, c)
            residual = (y - pred) * mask
            grad = torch.einsum("bnk,bn->bk", A, residual)
            step = F.softplus(self.log_step[idx])
            lam = F.softplus(self.log_lam[idx])
            c = soft_threshold(c + step * grad, lam)
        return c


def support_rip_loss(A: torch.Tensor, code: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Restricted-isometry proxy on recovered top-k atom supports."""

    bsz, _, num_atoms = A.shape
    k = min(k, num_atoms)
    support = code.abs().topk(k, dim=-1).indices
    losses = []
    eye = torch.eye(k, device=A.device, dtype=A.dtype)
    for bidx in range(bsz):
        As = A[bidx, :, support[bidx]]
        gram = As.transpose(0, 1) @ As / As.size(0).clamp_min(1) if isinstance(As.size(0), torch.Tensor) else As.transpose(0, 1) @ As / max(As.size(0), 1)
        losses.append((gram - eye).pow(2).mean())
    return torch.stack(losses).mean()


def policy_coherence_loss(A: torch.Tensor, group_id: torch.Tensor, mask: torch.Tensor, cap: float = 0.35) -> torch.Tensor:
    """Penalize highly coherent measurement rows inside policy-induced groups."""

    losses = []
    for bidx in range(A.size(0)):
        active_groups = torch.unique(group_id[bidx][mask[bidx] > 0])
        for gid in active_groups:
            idx = (group_id[bidx] == gid) & (mask[bidx] > 0)
            if idx.sum() <= 1:
                continue
            rows = F.normalize(A[bidx, idx], p=2, dim=-1)
            corr = rows @ rows.transpose(0, 1)
            off_diag = corr - torch.eye(corr.size(0), device=A.device, dtype=A.dtype)
            losses.append(F.relu(off_diag.abs().max() - cap).pow(2))
    if not losses:
        return torch.zeros((), device=A.device, dtype=A.dtype)
    return torch.stack(losses).mean()


def sparse_support_recoverability_loss(
    factual_code: torch.Tensor,
    cf_code: torch.Tensor,
    cf_A: torch.Tensor,
    recoverability_mask: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """Discourage policy-created atoms unsupported by recoverable measurements."""

    support = factual_code.abs().topk(min(k, factual_code.size(-1)), dim=-1).indices
    supported = torch.zeros_like(factual_code)
    supported.scatter_(1, support, 1.0)

    gram_diag = cf_A.pow(2).sum(dim=1).clamp_min(1e-6)
    recoverability = (gram_diag / gram_diag.mean(dim=1, keepdim=True).clamp_min(1e-6)).clamp(0.0, 2.0)
    recoverability = recoverability * recoverability_mask

    policy_created = F.relu(cf_code.abs() - factual_code.abs().detach())
    unsupported = (1.0 - supported) * F.relu(1.0 - recoverability)
    return (policy_created * unsupported).mean()


class DoRIPSparsePathologyLens(nn.Module):
    """Classify through RIP-constrained sparse pathology recovery."""

    def __init__(
        self,
        num_vars: int,
        num_atoms: int,
        hidden_dim: int,
        num_classes: int,
        desc_dim: int = 0,
    ):
        super().__init__()
        self.rows = MeasurementRowBuilder(num_vars, num_atoms, hidden_dim, desc_dim)
        self.decoder = UnrolledSparseDecoder(num_atoms)
        self.classifier = nn.Sequential(
            nn.Linear(num_atoms, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def recover(self, batch: dict) -> dict:
        meas = self.rows(batch)
        # Use observed value as the measurement target; y_proxy can serve as a
        # denoising auxiliary target when values are normalized per variable.
        y = batch.get("measurement_target", batch["event_value"])
        code = self.decoder(meas["A"], y, batch["event_mask"])
        logits = self.classifier(code)
        recon = torch.einsum("bnk,bk->bn", meas["A"], code)
        return {**meas, "code": code, "logits": logits, "recon": recon}

    def training_loss(
        self,
        batch: dict,
        lambda_rec: float = 0.15,
        lambda_rip: float = 0.20,
        lambda_coh: float = 0.10,
        lambda_stab: float = 0.20,
    ) -> dict:
        labels = batch["labels"]
        factual = self.recover(batch)

        cls_loss = F.cross_entropy(factual["logits"], labels)
        target = batch.get("measurement_target", batch["event_value"])
        rec_raw = (factual["recon"] - target).abs() * batch["event_mask"]
        rec_loss = rec_raw.sum() / batch["event_mask"].sum().clamp_min(1.0)
        rip_loss = support_rip_loss(factual["A"], factual["code"])
        coh_loss = policy_coherence_loss(
            factual["A"],
            batch["policy_group_id"],
            batch["event_mask"],
        )

        stab_losses = []
        for cf_batch in batch.get("counterfactual_measurement_bank", []):
            cf = self.recover(cf_batch)
            rec_mask = cf_batch.get(
                "recoverability_mask",
                torch.ones_like(factual["code"]),
            )
            stab_losses.append(
                sparse_support_recoverability_loss(
                    factual_code=factual["code"],
                    cf_code=cf["code"],
                    cf_A=cf["A"],
                    recoverability_mask=rec_mask,
                )
            )
        if stab_losses:
            stab_loss = torch.stack(stab_losses).mean()
        else:
            stab_loss = torch.zeros((), device=target.device, dtype=target.dtype)

        total = (
            cls_loss
            + lambda_rec * rec_loss
            + lambda_rip * rip_loss
            + lambda_coh * coh_loss
            + lambda_stab * stab_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "measurement_reconstruction_loss": rec_loss.detach(),
            "support_rip_loss": rip_loss.detach(),
            "policy_coherence_loss": coh_loss.detach(),
            "sparse_recoverability_loss": stab_loss.detach(),
            "mean_sparsity": (factual["code"].abs() > 1e-3).float().sum(dim=-1).mean().detach(),
        }


@torch.no_grad()
def build_counterfactual_measurement_bank(batch: dict) -> list[dict]:
    """Create policy edits that perturb the measurement operator, not labels."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = value.shape
    device = value.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, group_shift: int):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        out["policy_group_id"] = batch["policy_group_id"] + group_shift
        return out

    views = []

    # 1. Alarm burst thinning: remove every other late dense observation.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    burst_keep = torch.where(late > 0, alternating, torch.ones_like(mask)) * mask
    views.append(clone_with(value * burst_keep, time, var_id, burst_keep, 10))

    # 2. Panel decorrelation: jitter close cross-variable timestamps.
    jitter = 0.03 * horizon * torch.randn_like(time)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_time = (time + jitter * changed_var * mask).clamp_min(0.0)
    views.append(clone_with(value * mask, panel_time, var_id, mask, 20))

    # 3. Variable budget drop: keep early evidence for odd variables only.
    odd_var = (var_id % 2 == 1).to(mask.dtype)
    early = (time_norm <= 0.5).to(mask.dtype)
    budget_keep = mask * torch.maximum(1.0 - odd_var, early)
    views.append(clone_with(value * budget_keep, time, var_id, budget_keep, 30))

    # 4. Routine-round snapping: fixed clinical rounds.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    views.append(clone_with(value * mask, rounded_time, var_id, mask, 40))
    return views
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `panel coherence shift`：训练环境中变量联测产生高度相干测量行，测试环境拆分 panel。
   - `alarm burst shift`：训练医院报警后短窗口重复测量，测试医院按 routine round 采样。
   - `variable budget shift`：不同机构对高成本变量采用不同测量预算。
   - `device duty-cycle shift`：可穿戴设备在低电量或夜间降低采样频率。
   - `value-feature prior shift`：同一变量值在常规筛查 vs 疑似高危下被测，先验风险不同。

2. **对比方法**
   - 普通 event Transformer / MedFuse-style MuFuse。
   - TimeCHEAT / SLAN / RoMAE-style irregular baselines。
   - SLiCE 或 Neural CDE 主干。
   - missingness-aware encoder、policy adversarial baseline、mask dropout。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、DSSS、DLIO、DBBP-SS 等。

3. **核心指标**
   - in-policy accuracy / worst-policy accuracy。
   - policy-only classifier AUC：仅凭 `A_pi` 的 row statistics 是否能预测标签。
   - support RIP violation：`||A_S^T A_S - I||`。
   - policy row coherence：同一 panel / burst / routine group 内的最大行相干。
   - recovery condition number：top-k 支持上的 Gram 条件数。
   - spurious atom rate：反事实政策视图中新出现且 recoverability 低的稀疏原子比例。

4. **消融实验**
   - 去掉 `L_support_rip`，检查模型是否重新利用 panel/burst 高相干 shortcut。
   - 去掉 `L_policy_coherence_sobriety`，检查重复测量是否线性放大类别证据。
   - 用普通 pooled token 替代 sparse decoder，验证收益不是 RowBuilder 参数量带来的。
   - 将反事实 measurement bank 替换为随机 mask，验证收益来自结构化测量矩阵审计。
   - 扫描稀疏原子数与 top-k 支持大小，评估恢复稳定性和分类信息保留的 trade-off。

## 5. 预期创新性

1. **从采样去偏转向稀疏可恢复性**：不删除、不投影、不征税、不修复、不校准采样信息，而是问当前观测矩阵是否足以稳定恢复病理稀疏 code。
2. **从 value-feature token 分类转向 value-feature measurement row**：吸收 MedFuse 的乘法语义，但把它用于构造测量算子，而不是直接作为分类证据。
3. **从通道语义迁移转向通道测量几何**：吸收 CHARM 的通道描述优势，但通道文本只定义测量行方向，不直接进入 logits。
4. **从局部 CD / switch / CDE 主干转向恢复契约**：TimeCHEAT、SLAN、SLiCE 可作为 RowBuilder 或序列主干，但核心约束是 RIP / coherence / condition number。
5. **反事实干预用途全新**：counterfactual sampler 不制造一致性视图、不做投票、不做保形、不做风险方差，而是产生测量矩阵扰动来审计 sparse recovery。
6. **解释性直接对应部署风险**：若某个预测依赖高相干 panel 或 alarm burst，support RIP violation 与 spurious atom rate 会升高，能直接指出“这不是病理原子，而是采样矩阵病态造成的假原子”。

## 6. 一句话投稿卖点

**DRIP-SPL 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样政策改变病理稀疏 code 的测量矩阵与可恢复性”的问题，通过 value-feature measurement rows、unrolled sparse recovery、support-RIP 与 policy coherence 审计，让分类器只读取可由低相干观测矩阵稳定恢复的病理原子，而不是读取训练医院或设备制造的 panel 同步、报警复测、变量预算和 duty-cycle 测量病态。**
