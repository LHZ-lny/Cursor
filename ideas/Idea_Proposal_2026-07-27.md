# Title: Principal-Stratum Status Compiler：面向采样策略偏移的潜在观测主阶层状态编译器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md`，其中记录了多轮自动化任务均未发现 `my_work_summary.md`，并补充了当前工作区未检出的历史提案机制。
- 已读取近期论文记录：`paper_daily.md`，并确认工作区存在 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：`2026-06-17`、`2026-06-20`、`2026-06-24`、`2026-06-27`、`2026-07-15` 至 `2026-07-26`。

### 历史核心机制黑名单

为避免与历史提案发生思维重合，本提案明确避开以下主创新路线：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、采样 score 零空间、hazard-driven resampling、do-risk variance。
8. 生理流算子与采样算子的交换子、value graph / policy graph 分离、policy residual sink。
9. additive evidence market、protocol tax、token evidence budget、边际证据审计。
10. 后验商动力学、模型空间 posterior quotient、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、ANOVA-style error projection、VQ semantic clauses、HSIC redaction checksum。
12. policy-simplex randomized smoothing、certified policy radius、Dirichlet/logit-normal do-sampler、coverage loss。
13. Radon-Nikodym density ratio、doubly robust target-measure correction、influence-bound regularization。
14. optional-stopping martingale query、standardized innovation、predictability barrier。
15. soft excursion topology、censored persistence interval likelihood、censor envelope、fragmentation sobriety。
16. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
17. policy-only negative film、shadow eraser/stencil、shadow nullification。
18. latent packet codeword、parity-check、syndrome locator、packet repair。
19. conditional knockoff calendar、soft knockoff-FDR firewall、swap symmetry calibration。
20. observability witness、counterfactual observability probe bank、observability-routed classification。
21. subjective-logic evidential shields、policy-induced ignorance/vacuity、class-wise evidence discount。
22. observation-set policy lattice、meet/join visibility masks、submodular margin contracts、quality-order loss。
23. solver-trace mediator、NFE/roughness trace bank、front-door trace-standardized readout。
24. RKHS cubature weights、measurement-action bisimulation、policy-word signature renormalization、free-energy annealing、copula rank stripping、triage queue debt、Sinkhorn anchor canonicalization、MDL episode transduction、causal sheaf glue、trigger-phase hysteresis、observer-control barrier certificates、regret-escrow detail routing。
25. 单纯复用 FlowPath 可逆路径、GSNF/DBGL/GARLIC 图衰减、iTimER 误差伪观测、Record2Vec 摘要嵌入、QuITE 普通 query、MTM 普通 token mixing、MedMamba frequency/adaptive graph、MedSpaformer token sparsification、MILM value-redacted classifier、LLMTS/LLM4EHR 普通 LLM alignment 或 EHR-SPC 普通 future status set prediction 作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗，不要求多视图 logits/representation 一致，不做证据折扣、拓扑、gauge、纠错、knockoff、barrier 或 regret release；而是把不同采样政策下“同一个临床状态事件是否会被观测到”视为潜在结果，学习每个未来状态原子的 principal stratum：always-observed、policy-triggered、policy-suppressed 或 never-informative。分类器只消费由 principal strata 编译出的稳定状态原子，而不是消费事实采样政策偶然暴露的事件集合。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 `paper_daily.md` 中的两条线索很适合与“采样解耦/反事实干预”框架结合：

1. **EHR-SPC / Status-Aware Self-Supervised Forecasting** 说明，不规则 EHR 表征不应只重构局部缺失值，而应预测未来 status set。问题是，未来 event set 本身也受医院采样和记录政策影响：同一潜在病程在医院 A 会产生“乳酸复查 + 护理记录 + 用药事件”，在医院 B 可能只产生“常规生命体征记录”。
2. **LLM4EHR** 强调医学事件序列提供语义锚点，但事件语义也混合 patient-state semantics 与 hospital-policy semantics。检查、医嘱、用药、护理记录既描述患者状态，也描述医院如何观察和干预患者。

因此，直接把未来 event set 当作 SSL 目标，或直接把 EHR event sequence 与 time-series embedding 做语义对齐，都可能把“未来会被记录什么”误当成“患者未来状态是什么”。采样策略偏移下，模型学到的 status token 可能其实是 protocol token。

**Principal-Stratum Status Compiler (PSSC)** 的核心问题是：

> 对某个未来状态原子 `a`，如果我们把同一条潜在病程放到多种采样/记录政策下，它会在哪些政策中被观测到？

这对应因果推断里的 principal stratification。对每个状态原子 `a`，考虑一组反事实采样政策 `p = 1...K` 下的潜在观测结果：

```text
O_a(p) in {0, 1}: 在政策 p 下，状态原子 a 是否会被记录/观测到
```

由向量 `O_a(1:K)` 可以定义状态原子的主阶层：

- `always-state`: 多数合理政策下都会出现，说明它更可能是稳定病程语义。
- `policy-triggered`: 只在告警、复测、panel 或特定记录制度下出现，说明它更像采样/流程触发的可见性。
- `policy-suppressed`: 只在低成本、稀疏或延迟政策下消失，说明它是易被政策遮蔽的状态证据。
- `never-informative`: 在反事实政策下都不可稳定支持，不应用于分类主证据。

PSSC 不要求不同政策视图预测一致，也不删除 informative sampling。它先把未来状态预测目标拆成潜在观测主阶层，再把分类器限制为主要依赖 `always-state` 与经过 `policy-suppressed` 标记的状态原子。这样，模型不会把某家医院特有的“未来事件集合形状”当成可迁移类别证据。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：从普通 future status target 改为 Potential-Outcome Status Bank

新增 `PrincipalStratumStatusCollator`。对每条患者轨迹切分为过去窗口 `X_past` 与未来窗口 `E_future`，然后生成一组反事实采样/记录政策：

1. `routine_round`: 固定查房式观测。
2. `alarm_followup`: 异常后密集复测。
3. `panel_batch`: 多变量 panel 同步记录。
4. `budget_sparse`: 成本受限或设备受限的稀疏记录。
5. `value_pending`: 事件坐标已出现但数值尚未返回。

对未来事件先构造语义状态原子 `status atom`：

```text
a_j = (semantic_type_j, coarse_time_bin_j, value_bucket_j, event_family_j)
```

其中 `semantic_type` 可以来自医学代码、变量类型、事件类型或冻结医学文本编码器的离散类别，但 PSSC 不做 LLM 对齐训练，也不把自由文本 embedding 直接作为分类主通道。

collator 返回一个潜在观测矩阵：

```text
potential_obs[j, k] = 1 if atom j would be visible under policy k else 0
```

注意：这不是 meet/join 信息格，不计算单调/次模边际；也不是 knockoff 或随机平滑。它只把“未来状态原子在不同政策下是否可见”作为潜在结果矩阵。

### 2.2 改 Encoder：Past Encoder + Principal-Stratum Set Decoder

PSSC 保留当前 value process / sampling process 的解耦思想，但角色改变：

1. **Past Irregular Encoder**
   - 输入过去窗口的 `(value, time, variable, mask)`。
   - 输出 `h_past`，表示到当前时刻为止的病程状态。
   - 不把 policy id、missingness pattern 或 future potential matrix 输入分类头。

2. **Status Query Set Decoder**
   - 借鉴 EHR-SPC 的 query-based set prediction，但目标不是普通 future event set，而是带潜在观测向量的 status atom set。
   - 每个 query 输出：

```text
atom_embedding_q
semantic_logits_q
time_bin_logits_q
value_bucket_logits_q
potential_obs_logits_q in R^K
```

3. **Principal-Stratum Compiler**
   - 将 `potential_obs_logits_q` 编译为主阶层概率：

```text
stratum_q = Compiler(sigmoid(potential_obs_logits_q))
```

   - `always-state` 原子进入分类主路径。
   - `policy-suppressed` 原子可以进入主路径但带有显式 suppression flag，表示“这是状态证据，但容易被某些政策遮蔽”。
   - `policy-triggered` 原子只用于未来状态预测辅助目标和偏移诊断，不直接进入分类聚合。
   - `never-informative` 原子被丢弃或只用于 reconstruction sanity check。

### 2.3 改 Loss：从 future set prediction 转向 Principal-Stratum Sufficiency

总目标：

```text
L = L_cls
  + lambda_set * L_status_set_prediction
  + lambda_po  * L_potential_observation_matrix
  + lambda_str * L_principal_stratum_compilation
  + lambda_sup * L_suppression_sufficiency
```

#### A. 分类损失 `L_cls`

分类器不从事实 future event set 分类，也不从 policy-triggered atom 分类。它只读取主阶层编译后的稳定状态摘要：

```text
z_stable = Pool({atom_q | stratum_q in always-state or policy-suppressed})
logits = Classifier(z_stable)
L_cls = CE(logits, y)
```

#### B. 状态集合预测 `L_status_set_prediction`

使用 DETR-style set prediction 预测未来 status atoms 的语义类型、粗时间桶和数值桶。这里吸收 EHR-SPC 的 set prediction 优势，但预测对象不是原始未来事件列表，而是 policy-potential status atoms。

#### C. 潜在观测矩阵损失 `L_potential_observation_matrix`

对匹配到的 atom 预测它在每个反事实政策下是否可见：

```text
L_potential_observation_matrix =
  BCE(potential_obs_logits_q, potential_obs_target_q)
```

这项把采样偏移从“事实 mask 变化”提升为“同一状态原子在政策集合上的潜在观测响应”。

#### D. 主阶层编译损失 `L_principal_stratum_compilation`

根据 `potential_obs_target` 生成弱主阶层标签：

- 全 1 或高可见率：`always-state`
- 仅在 triggered recipes 中可见：`policy-triggered`
- 在 sparse/budget recipes 中消失但 routine/triggered 中可见：`policy-suppressed`
- 全 0 或低可见率：`never-informative`

训练 compiler：

```text
L_principal_stratum_compilation = CE(stratum_logits, stratum_target)
```

#### E. Suppression Sufficiency Loss `L_suppression_sufficiency`

为了避免模型把所有困难原子都推成 `always-state`，对 `policy-suppressed` 原子加一个 sufficiency 检查：当它被反事实稀疏政策遮蔽时，模型不能把“未观测”直接当负证据；当它可见时，可以作为状态证据补充。

实现上比较两个摘要：

```text
z_always = Pool(always-state atoms)
z_safe   = Pool(always-state + policy-suppressed atoms)
```

约束：

```text
L_suppression_sufficiency =
  CE(Classifier(z_always), y) * w_always
  + relu(CE(Classifier(z_safe), y) - CE(Classifier(z_always), y) - margin)^2
```

直觉：`always-state` 应足以给出保守预测；`policy-suppressed` 只能改善或细化状态证据，不能把分类完全建立在“某政策下才看得见”的未来记录事件上。

这不是多视图一致性，因为没有要求不同采样政策 logits 相同；它只约束不同主阶层集合的分类角色。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `PastIrregularEncoder`，负责从过去观测中预测未来潜在状态原子。
- 现有 sampling branch 改为 `PolicyPotentialGenerator`，只定义未来窗口的反事实采样/记录政策，用于生成 `potential_obs` 标签。
- 现有 counterfactual intervention 不再生成一致性 pair、risk variance、knockoff、barrier 或 regret stress，而是生成 potential-outcome observation matrix。
- 推理阶段不需要真实 future event，也不需要 policy id；模型从过去窗口预测 principal-stratum status atoms，并用 stable strata 做分类。
- 部署诊断输出：
  - `always_state_mass`: 分类证据中有多少来自 always-state atoms。
  - `triggered_atom_leakage`: 预测是否过度依赖 policy-triggered atoms。
  - `suppressed_state_gap`: 稀疏政策可能遮蔽的真实状态证据比例。
  - `principal_stratum_shift`: 测试环境中事实观测的 atom stratum 分布是否偏离训练。

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


class PastIrregularEncoder(nn.Module):
    """Encode past irregular observations into a patient-state context."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, batch: dict) -> torch.Tensor:
        value = batch["past_value"]
        time = batch["past_time"]
        var_id = batch["past_var_id"]
        mask = batch["past_mask"]
        measurement_std = batch.get("past_measurement_std", torch.zeros_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id.clamp_min(0))
        event_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(measurement_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * mask.unsqueeze(-1)
        seq_h, _ = self.rnn(event_h)
        return masked_mean(seq_h, mask, dim=1)


class PrincipalStatusSetDecoder(nn.Module):
    """Predict future status atoms and their potential observability matrix."""

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_semantic_types: int,
        num_time_bins: int,
        num_value_buckets: int,
        num_policies: int,
        num_strata: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.semantic_head = nn.Linear(hidden_dim, num_semantic_types)
        self.time_head = nn.Linear(hidden_dim, num_time_bins)
        self.value_head = nn.Linear(hidden_dim, num_value_buckets)
        self.obs_head = nn.Linear(hidden_dim, num_policies)
        self.stratum_head = nn.Sequential(
            nn.Linear(hidden_dim + num_policies, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_strata),
        )

    def forward(self, h_past: torch.Tensor) -> dict:
        bsz = h_past.size(0)
        query = self.query[None].expand(bsz, -1, -1)
        memory = h_past[:, None, :]
        atom_h = self.decoder(query, memory)
        potential_obs_logits = self.obs_head(atom_h)
        obs_prob = torch.sigmoid(potential_obs_logits)
        stratum_logits = self.stratum_head(torch.cat([atom_h, obs_prob], dim=-1))
        return {
            "atom_h": atom_h,
            "semantic_logits": self.semantic_head(atom_h),
            "time_logits": self.time_head(atom_h),
            "value_logits": self.value_head(atom_h),
            "potential_obs_logits": potential_obs_logits,
            "stratum_logits": stratum_logits,
        }


def pool_by_stratum(atom_h: torch.Tensor, stratum_prob: torch.Tensor, stratum_ids: list[int]) -> torch.Tensor:
    weight = stratum_prob[..., stratum_ids].sum(dim=-1)
    return masked_mean(atom_h, weight, dim=1)


class PrincipalStratumStatusCompiler(nn.Module):
    """Compile potential-observation status atoms into policy-robust classification evidence."""

    ALWAYS = 0
    POLICY_TRIGGERED = 1
    POLICY_SUPPRESSED = 2
    NEVER_INFORMATIVE = 3

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        num_queries: int,
        num_semantic_types: int,
        num_time_bins: int,
        num_value_buckets: int,
        num_policies: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = PastIrregularEncoder(num_vars, hidden_dim)
        self.decoder = PrincipalStatusSetDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_semantic_types=num_semantic_types,
            num_time_bins=num_time_bins,
            num_value_buckets=num_value_buckets,
            num_policies=num_policies,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict) -> dict:
        h_past = self.encoder(batch)
        dec = self.decoder(h_past)
        stratum_prob = torch.softmax(dec["stratum_logits"], dim=-1)

        z_always = pool_by_stratum(dec["atom_h"], stratum_prob, [self.ALWAYS])
        z_stable = pool_by_stratum(dec["atom_h"], stratum_prob, [self.ALWAYS, self.POLICY_SUPPRESSED])
        logits = self.classifier(z_stable)
        logits_always = self.classifier(z_always)
        return {**dec, "logits": logits, "logits_always": logits_always, "h_past": h_past}

    def matched_status_loss(self, out: dict, batch: dict) -> torch.Tensor:
        """Sketch: assumes the collator provides query-to-target assignment indices."""

        q_idx = batch["match_query_idx"]
        t_idx = batch["match_target_idx"]
        b_idx = torch.arange(q_idx.size(0), device=q_idx.device)[:, None]

        sem_loss = F.cross_entropy(
            out["semantic_logits"][b_idx, q_idx].flatten(0, 1),
            batch["target_semantic_type"].gather(1, t_idx).flatten(),
        )
        time_loss = F.cross_entropy(
            out["time_logits"][b_idx, q_idx].flatten(0, 1),
            batch["target_time_bin"].gather(1, t_idx).flatten(),
        )
        value_loss = F.cross_entropy(
            out["value_logits"][b_idx, q_idx].flatten(0, 1),
            batch["target_value_bucket"].gather(1, t_idx).flatten(),
        )
        return sem_loss + time_loss + value_loss

    def potential_observation_loss(self, out: dict, batch: dict) -> torch.Tensor:
        q_idx = batch["match_query_idx"]
        t_idx = batch["match_target_idx"]
        b_idx = torch.arange(q_idx.size(0), device=q_idx.device)[:, None]
        pred = out["potential_obs_logits"][b_idx, q_idx]
        target = batch["target_potential_obs"][b_idx, t_idx].to(pred.dtype)
        return F.binary_cross_entropy_with_logits(pred, target)

    def stratum_loss(self, out: dict, batch: dict) -> torch.Tensor:
        q_idx = batch["match_query_idx"]
        t_idx = batch["match_target_idx"]
        b_idx = torch.arange(q_idx.size(0), device=q_idx.device)[:, None]
        pred = out["stratum_logits"][b_idx, q_idx].flatten(0, 1)
        target = batch["target_principal_stratum"].gather(1, t_idx).flatten()
        return F.cross_entropy(pred, target)

    def suppression_sufficiency_loss(self, out: dict, labels: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
        ce_always = F.cross_entropy(out["logits_always"], labels, reduction="none")
        ce_stable = F.cross_entropy(out["logits"], labels, reduction="none")
        # Stable atoms may improve the always-only prediction, but should not become a brittle shortcut.
        return (ce_always + F.relu(ce_stable - ce_always - margin).pow(2)).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_set: float = 0.30,
        lambda_po: float = 0.40,
        lambda_str: float = 0.20,
        lambda_sup: float = 0.10,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)
        set_loss = self.matched_status_loss(out, batch)
        po_loss = self.potential_observation_loss(out, batch)
        stratum_loss = self.stratum_loss(out, batch)
        suff_loss = self.suppression_sufficiency_loss(out, labels)
        total = (
            cls_loss
            + lambda_set * set_loss
            + lambda_po * po_loss
            + lambda_str * stratum_loss
            + lambda_sup * suff_loss
        )
        with torch.no_grad():
            stratum_prob = torch.softmax(out["stratum_logits"], dim=-1)
            triggered_mass = stratum_prob[..., self.POLICY_TRIGGERED].mean()
            always_mass = stratum_prob[..., self.ALWAYS].mean()
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "status_set_loss": set_loss.detach(),
            "potential_observation_loss": po_loss.detach(),
            "principal_stratum_loss": stratum_loss.detach(),
            "suppression_sufficiency_loss": suff_loss.detach(),
            "always_state_mass": always_mass,
            "triggered_atom_mass": triggered_mass,
        }


@torch.no_grad()
def derive_principal_stratum(potential_obs: torch.Tensor) -> torch.Tensor:
    """Heuristic target compiler for principal strata.

    potential_obs: [B, M, K] with policy order:
    routine, alarm_followup, panel_batch, budget_sparse, value_pending.
    """

    visible_rate = potential_obs.float().mean(dim=-1)
    routine = potential_obs[..., 0].bool()
    alarm = potential_obs[..., 1].bool()
    panel = potential_obs[..., 2].bool()
    sparse = potential_obs[..., 3].bool()
    pending = potential_obs[..., 4].bool()

    target = torch.full(potential_obs.shape[:2], 3, device=potential_obs.device, dtype=torch.long)
    target[visible_rate >= 0.80] = 0  # always-state
    triggered = (alarm | panel) & ~routine & (visible_rate > 0.0)
    target[triggered] = 1  # policy-triggered
    suppressed = routine & ~sparse & (visible_rate >= 0.40)
    target[suppressed] = 2  # policy-suppressed
    never = (visible_rate <= 0.20) & ~pending
    target[never] = 3
    return target
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `future-recording shift`：训练环境未来窗口记录密集，测试环境记录稀疏。
   - `alarm-followup shift`：训练医院异常后立刻复测，测试医院延迟复测。
   - `panel-batch shift`：训练医院同步 panel 记录，测试医院拆成异步事件。
   - `value-pending shift`：测试时未来某些事件坐标已出现但数值未返回。
   - `semantic-event shift`：医学事件序列中医嘱/护理记录制度改变，但底层标签语义不变。

2. **对比方法**
   - 普通 irregular encoder。
   - EHR-SPC 普通 future status set prediction。
   - LLM4EHR 普通 event-sequence contrastive alignment。
   - MILM-style value-redacted sampling classifier。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN、以及后续记忆中记录的 RKHS Cubature、Bisimulation Lens、Signature Renormalizer、Queue Debt、Sheaf Glue、Hysteresis、Barrier、Regret Escrow 等历史方案。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - principal-stratum purity：预测 atom 的主阶层是否与潜在观测矩阵一致。
   - triggered-atom leakage：分类 logit 对 policy-triggered atoms 的依赖比例。
   - always-state sufficiency：仅用 always-state atoms 的保守分类能力。
   - suppressed-state recovery：稀疏政策遮蔽的状态原子在可见时是否改善分类。

4. **消融实验**
   - 去掉 `L_potential_observation_matrix`，验证模型是否退化为普通 future set prediction。
   - 去掉 `L_principal_stratum_compilation`，检查 policy-triggered atoms 是否进入分类主路径。
   - 去掉 `L_suppression_sufficiency`，检查模型是否把分类建立在 policy-suppressed atoms 上。
   - 将潜在政策 bank 替换为随机 mask，验证收益来自可解释采样/记录政策而非普通增强。
   - 只用 LLM semantic type、只用数值状态 atom、两者组合，验证 LLM4EHR 式事件语义是否需要 principal-stratum 编译才能稳健。

## 5. 预期创新性

1. **从 future status prediction 转向 potential-outcome status prediction**：EHR-SPC 预测未来状态集合，PSSC 进一步预测每个未来状态原子在多种采样政策下是否会被观测到。
2. **从语义对齐转向主阶层编译**：吸收 LLM4EHR 对医学事件语义的重视，但不做普通 contrastive alignment；事件语义先被放进 principal strata，区分 patient-state semantics 与 hospital-policy semantics。
3. **从采样去偏转向潜在观测阶层化**：不估计采样概率、不做对抗、不做一致性、不做 knockoff 或证书；只问“这个状态原子属于哪类潜在观测阶层”。
4. **保留 informative sampling 的细粒度价值**：policy-suppressed 原子不是被删除，而是被标记为容易被某些政策遮蔽的状态证据；policy-triggered 原子也可用于诊断和未来状态预测，但不直接支配分类边界。
5. **与当前反事实框架低侵入兼容**：现有 counterfactual sampler 只需从生成 mask/view 改为生成未来 status atom 的 potential-observation matrix。

## 6. 一句话投稿卖点

**PSSC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“未来状态原子的潜在观测主阶层错配”问题：通过 potential-outcome status bank、principal-stratum set decoder 与 stable-strata classifier，模型区分哪些未来事件语义是跨政策稳定病程状态，哪些只是医院采样/记录制度触发的可见性，从而在不依赖危险率、对抗、一致性、后验商、随机平滑、拓扑、gauge、纠错、knockoff、evidential shield、barrier 或 regret router 的前提下，提升跨采样政策分类鲁棒性。**
