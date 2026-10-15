# Title: Regret-Escrow Context-Detail Router：面向采样策略偏移的反事实遗憾托管上下文-细节路由器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已读取自动化记忆：确认历史多轮任务中同样未发现 `my_work_summary.md` 或可替代总结文件，并读取了 `idea_2026-07-24.md`、`idea_2026-07-25.md`。
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-07-19.md`
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案机制：2026-06-17、06-20、06-24、06-27、07-15 至 07-25。

### 历史核心机制黑名单

为避免与历史提案和 `paper_daily.md` 中已有机制发生思维重合，本提案明确避开以下方向作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、hazard-driven counterfactual resampling、分类梯度与采样 score 零空间正交化。
8. 生理流算子与采样算子的交换子、value graph / policy graph 分离、policy residual sink。
9. additive evidence market、protocol tax、token-level evidence budget、边际证据审计。
10. 后验商动力学、模型空间 posterior quotient、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、ANOVA-style error projection、VQ semantic clauses、HSIC redaction。
12. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler、policy coverage loss。
13. 采样测度 density ratio、doubly robust correction、influence-bound regularization。
14. previsible martingale query、standardized innovation、optional-stopping moment control。
15. soft excursion topology、censored persistence interval、censor envelope、fragmentation sobriety。
16. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
17. policy-only negative film、latent shadow eraser/stencil。
18. latent packet codeword、parity-check、syndrome locator、packet repair decoder。
19. conditional knockoff calendar、soft knockoff-FDR firewall、swap symmetry calibration。
20. observability witness、counterfactual observability probe、low-observability routed classification。
21. subjective-logic / Dirichlet evidential classification、policy-induced ignorance/vacuity mass。
22. observation-set policy lattice、meet/join masks、单调/次模边际契约、shortcut curvature。
23. solver trace / NFE / roughness front-door standardization。
24. trigger-threshold phase sweep、phase-plateau classification、opening/closing trigger hysteresis、boundary quarantine。
25. observation-control field、Hamilton-Jacobi / control-barrier certificate。

本提案的核心机制是 **反事实遗憾托管**：不再问“gate 在哪个阈值打开”，也不把 detail 分支直接做成不变表示；而是要求 local detail inspector 的每一份细节增益先进入 escrow，只有当它相对 coarse context 在一组反事实采样政策下都能产生正的低分位遗憾收益时，才允许释放进最终 logits。

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 `paper_daily_2026-07-19.md` 中的 **Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction** 提供了一个很适合迁移到非规则采样分类的结构启发：稀疏医疗事件需要先由 global context explorer 判断是否值得精查，再由 local detail inspector 处理局部边界和形态。

但在 sampling-policy shift 场景中，local detail 是最危险也最有价值的部分：

- 它最能捕捉短时异常、报警后复测、局部峰值和事件边界；
- 也最容易吸收训练医院的复测习惯、标注覆盖、panel 触发、设备高频窗口；
- 如果直接让 adaptive gate 学“何时打开 detail”，gate 可能学到的是训练政策下的触发规则，而不是跨政策稳定的状态证据。

7/24 的历史提案已经围绕 trigger phase、阈值扫描、phase plateau 和 hysteresis 做过机制设计，因此本提案避开“门控相位”方向，改用决策论视角：

> local detail 不应因为“看起来更精细”就进入分类器；它必须证明自己相对 coarse context 的分类遗憾在多种 `do(policy)` 下都有稳健下降。若一个细节只在事实采样政策下有用、换一个反事实复测/稀疏/异步策略后就无用或有害，它只能留在 escrow 中，不能污染最终分类边界。

这样能同时保留 context-detail 机制的优势与采样偏移鲁棒性：

1. coarse context explorer 作为低分辨率、低策略敏感的安全基线；
2. local detail inspector 继续捕捉稀疏高价值事件；
3. sampling branch / counterfactual intervention 不用于一致性、不用于对抗、不用于证据税，而是生成“细节是否稳健降低遗憾”的审计场景；
4. 最终分类只释放跨政策稳健有用的 detail 增量。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通 adaptive gate 改为 Regret-Escrow Router

模型包含三条路径：

1. **Coarse Context Explorer**
   - 输入完整不规则事件流的低分辨率摘要，例如粗时间窗 pooling、变量组摘要、长程状态 token。
   - 输出 `logits_context` 和 `h_context`。
   - 它是安全基线：即使 detail 被禁用，也应能给出不依赖局部采样触发的初步分类。

2. **Local Detail Inspector**
   - 在 context 提示的候选区域内读取局部事件片段、短时变化、稀疏高幅度异常。
   - 输出细节增量 `delta_logits_detail`，但该增量默认不直接加到最终 logits。
   - 与 CDI-TS 的 detail inspector 对齐，但本提案不使用 gate phase / hysteresis。

3. **Regret Escrow Router**
   - sampling branch 生成若干反事实采样 recipe，例如 burst thinning、panel desynchronization、late-window detail hiding、routine-round resampling。
   - 对每个 recipe，比较 context-only 与 context+detail 的逐样本损失：

```text
regret_k = CE(logits_context^k, y) - CE(logits_fused^k, y)
```

   - 若 `regret_k > 0`，说明 detail 在该政策下相对 context 降低了分类遗憾；若 `regret_k < 0`，说明 detail 在该政策下伤害了决策。
   - 计算低分位稳健遗憾：

```text
regret_lcb = Quantile_q({regret_k}_{k=1}^K), q in [0.1, 0.3]
```

   - 最终 detail release：

```text
release = gate_raw * sigmoid(a * regret_lcb)
logits = logits_context + release * delta_logits_detail
```

直觉：detail 分支不是被删除，而是被“托管”。只有当它对多种采样政策都能稳定减少相对遗憾时，才被释放。

### 2.2 改 Loss：从一致性/对抗转向 Counterfactual Regret Escrow

总目标：

```text
L = L_factual_cls
  + lambda_ctx * L_context_floor
  + lambda_reg * L_regret_escrow
  + lambda_tail * L_tail_regret
  + lambda_use * L_detail_use_sobriety
```

#### A. Factual Classification Loss `L_factual_cls`

事实样本上使用 escrow 后的 logits：

```text
L_factual_cls = CE(logits_context + release * delta_logits_detail, y)
```

#### B. Context Floor `L_context_floor`

coarse context 是安全基线，不能完全退化：

```text
L_context_floor = CE(logits_context, y)
```

这不是多视图一致性，也不是 policy adversarial；它只保证“无细节时的最低可用分类能力”。

#### C. Regret Escrow Loss `L_regret_escrow`

若低分位遗憾 `regret_lcb` 非正，detail release 应接近 0；若 `regret_lcb` 为正，release 可以增加：

```text
allowed_release = sigmoid(b * regret_lcb.detach())
L_regret_escrow = ReLU(release_strength - allowed_release)^2
```

该项不是 evidence tax：没有给 token 定价，也没有预算；它只把 detail 使用权绑定到“相对 context 的跨政策决策收益”。

#### D. Tail Regret Loss `L_tail_regret`

detail inspector 应避免只在平均意义上有用，却在最坏采样政策下伤害分类：

```text
L_tail_regret = ReLU(tau - regret_lcb)^2 * stopgrad(detail_attempt)
```

其中 `detail_attempt` 表示模型确实尝试释放 detail。若样本不需要 detail，该项不强迫 detail 有用；若模型想用 detail，就必须让低分位遗憾不为负。

#### E. Detail Use Sobriety `L_detail_use_sobriety`

为了防止 release 全开或全关，加入轻量清醒度：

```text
L_detail_use_sobriety =
  mean(ReLU(release_strength - r_max)^2)
  + mean(ReLU(r_min - release_strength)^2 * positive_regret_flag)
```

它只约束路由器的使用习惯，不约束不同 policy views 的 logits 相同。

### 2.3 改 Dataloader：返回 Counterfactual Detail Stress Bank

新增 `CounterfactualDetailRegretCollator`，每个 batch 返回：

1. `factual_batch`：原始不规则事件。
2. `cf_views`：一组反事实采样视图，仅用于计算 detail 相对 context 的 regret：
   - `burst_thin`：报警后高频复测稀疏化；
   - `panel_desync`：同步 panel 拆成异步观测；
   - `late_detail_hide`：隐藏晚期局部细节，只保留 coarse context；
   - `routine_round`：把事件触发式观测变为粗查房式窗口。
3. `detail_region_mask`：哪些事件属于 detail inspector 可读的局部高分辨率区域。
4. `context_visibility_mask`：coarse context 可见的低分辨率摘要。

这些 view 不用于 pairwise consistency，不用于 risk variance，不用于 randomized smoothing，也不用于 density ratio。它们只回答一个问题：

> 在这个采样政策下，detail 是否比 context-only 更值得信任？

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 拆为 coarse context stem 与 local detail inspector。
- 现有 sampling branch 改为 detail stress recipe generator，不输出采样概率、policy residual、barrier field 或 uncertainty mass。
- 现有 counterfactual intervention 保留，但目标从“让视图一致”改成“估计 detail 的跨政策 regret lower bound”。
- 推理阶段：
  - 快速模式：使用事实样本的 context/detail 与训练期 EMA regret prior 估计 release；
  - 稳健模式：对少量标准 stress recipes 做轻量前向，估计 `regret_lcb` 后再释放 detail；
  - 诊断输出：`regret_lcb`、release strength、detail harm rate、context fallback rate。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="none")


def lower_confidence_regret(regret: torch.Tensor, quantile: float = 0.2) -> torch.Tensor:
    """Return a low-quantile regret estimate over counterfactual policies.

    regret: [K, B], where positive values mean detail improves over context.
    """

    return torch.quantile(regret, q=quantile, dim=0)


class ContextDetailRegretRouter(nn.Module):
    """Release local detail only when it robustly reduces regret vs coarse context."""

    def __init__(
        self,
        context_encoder: nn.Module,
        detail_encoder: nn.Module,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.detail_encoder = detail_encoder
        self.context_head = nn.Linear(hidden_dim, num_classes)
        self.detail_head = nn.Linear(hidden_dim, num_classes)
        self.release_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def encode_view(self, batch: dict) -> dict:
        h_context = self.context_encoder(
            values=batch["values"],
            times=batch["times"],
            mask=batch["context_mask"],
        )
        h_detail = self.detail_encoder(
            values=batch["values"],
            times=batch["times"],
            mask=batch["detail_mask"],
            context=h_context,
        )

        logits_context = self.context_head(h_context)
        delta_logits = self.detail_head(h_detail)
        raw_release = self.release_head(torch.cat([h_context, h_detail], dim=-1)).squeeze(-1)
        return {
            "h_context": h_context,
            "h_detail": h_detail,
            "logits_context": logits_context,
            "delta_logits": delta_logits,
            "raw_release": raw_release,
        }

    def fuse_logits(
        self,
        logits_context: torch.Tensor,
        delta_logits: torch.Tensor,
        raw_release: torch.Tensor,
        regret_lcb: torch.Tensor | None = None,
        regret_scale: float = 6.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if regret_lcb is None:
            release = raw_release
        else:
            # Negative robust regret locks the detail contribution in escrow.
            regret_gate = torch.sigmoid(regret_scale * regret_lcb.detach())
            release = raw_release * regret_gate
        logits = logits_context + release.unsqueeze(-1) * delta_logits
        return logits, release

    def counterfactual_regrets(self, batch: dict) -> tuple[torch.Tensor, list[dict]]:
        labels = batch["labels"]
        regrets = []
        encoded_views = []
        for view in batch["cf_views"]:
            out = self.encode_view(view)
            logits_fused, release = self.fuse_logits(
                out["logits_context"],
                out["delta_logits"],
                out["raw_release"],
                regret_lcb=None,
            )
            ce_context = cross_entropy_per_sample(out["logits_context"], labels)
            ce_fused = cross_entropy_per_sample(logits_fused, labels)
            regrets.append(ce_context - ce_fused)
            out["logits_fused_unescrowed"] = logits_fused
            out["release_unescrowed"] = release
            encoded_views.append(out)
        return torch.stack(regrets, dim=0), encoded_views

    def forward(self, batch: dict) -> dict:
        factual = self.encode_view(batch)
        if "cf_views" in batch and batch["cf_views"]:
            regret, _ = self.counterfactual_regrets(batch)
            regret_lcb = lower_confidence_regret(regret)
        else:
            regret_lcb = torch.zeros(
                batch["labels"].size(0),
                device=factual["logits_context"].device,
                dtype=factual["logits_context"].dtype,
            )
        logits, release = self.fuse_logits(
            factual["logits_context"],
            factual["delta_logits"],
            factual["raw_release"],
            regret_lcb=regret_lcb,
        )
        factual.update({
            "logits": logits,
            "release": release,
            "regret_lcb": regret_lcb,
        })
        return factual

    def training_loss(
        self,
        batch: dict,
        lambda_ctx: float = 0.4,
        lambda_reg: float = 0.25,
        lambda_tail: float = 0.15,
        lambda_use: float = 0.02,
        target_regret: float = 0.02,
        max_release: float = 0.85,
        min_release_when_useful: float = 0.10,
    ) -> dict:
        labels = batch["labels"]

        factual_base = self.encode_view(batch)
        regret, _ = self.counterfactual_regrets(batch)
        regret_lcb = lower_confidence_regret(regret)

        logits, release = self.fuse_logits(
            factual_base["logits_context"],
            factual_base["delta_logits"],
            factual_base["raw_release"],
            regret_lcb=regret_lcb,
        )

        factual_loss = F.cross_entropy(logits, labels)
        context_loss = F.cross_entropy(factual_base["logits_context"], labels)

        allowed_release = torch.sigmoid(6.0 * regret_lcb.detach())
        escrow_loss = F.relu(release - allowed_release).pow(2).mean()

        detail_attempt = factual_base["raw_release"].detach()
        tail_loss = (
            F.relu(target_regret - regret_lcb).pow(2) * detail_attempt
        ).mean()

        useful = (regret_lcb.detach() > target_regret).to(release.dtype)
        use_sobriety = (
            F.relu(release - max_release).pow(2)
            + useful * F.relu(min_release_when_useful - release).pow(2)
        ).mean()

        total = (
            factual_loss
            + lambda_ctx * context_loss
            + lambda_reg * escrow_loss
            + lambda_tail * tail_loss
            + lambda_use * use_sobriety
        )
        return {
            "loss": total,
            "factual_loss": factual_loss.detach(),
            "context_loss": context_loss.detach(),
            "regret_escrow_loss": escrow_loss.detach(),
            "tail_regret_loss": tail_loss.detach(),
            "detail_use_sobriety": use_sobriety.detach(),
            "mean_regret_lcb": regret_lcb.mean().detach(),
            "mean_release": release.mean().detach(),
        }


@torch.no_grad()
def build_counterfactual_detail_views(batch: dict) -> list[dict]:
    """Sketch counterfactual sampling views for regret auditing.

    These views are not consistency pairs. They only evaluate whether local detail
    still improves over the coarse context under plausible sampling policies.
    """

    values = batch["values"]
    times = batch["times"]
    mask = batch["mask"]
    bsz, seq_len, _ = values.shape
    device = values.device

    horizon = times[:, -1:].clamp_min(1e-6)
    time_norm = times / horizon
    late = (time_norm > 0.66).to(mask.dtype).unsqueeze(-1)
    alternating = ((torch.arange(seq_len, device=device)[None, :, None] % 2) == 0).to(mask.dtype)

    views = []

    # 1. Burst thinning: local high-frequency details are sparsified.
    burst_mask = mask * alternating
    views.append({
        **batch,
        "values": values * burst_mask,
        "context_mask": mask,
        "detail_mask": burst_mask,
    })

    # 2. Late detail hiding: coarse context remains, late high-resolution detail vanishes.
    late_detail_mask = mask * (1.0 - late)
    views.append({
        **batch,
        "values": values * late_detail_mask,
        "context_mask": mask,
        "detail_mask": late_detail_mask,
    })

    # 3. Routine-round approximation: detail sees only coarse periodic checks.
    round_mask = mask * ((torch.arange(seq_len, device=device)[None, :, None] % 3) == 0).to(mask.dtype)
    views.append({
        **batch,
        "values": values * round_mask,
        "context_mask": mask,
        "detail_mask": round_mask,
    })

    # 4. Panel desynchronization proxy: stagger odd variables by suppressing alternating events.
    var_idx = torch.arange(mask.size(-1), device=device).view(1, 1, -1)
    odd_var = (var_idx % 2 == 1).to(mask.dtype)
    desync_keep = torch.where(odd_var > 0, alternating, torch.ones_like(alternating))
    desync_mask = mask * desync_keep
    views.append({
        **batch,
        "values": values * desync_mask,
        "context_mask": mask,
        "detail_mask": desync_mask,
    })

    return views
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `burst-to-routine shift`：训练环境报警后密集复测，测试环境只保留查房式观测；
   - `panel-desync shift`：训练环境同步 panel，测试环境拆成异步化验；
   - `late-detail blackout`：测试设备或流程丢失晚期局部细节；
   - `annotation-coverage shift`：某些事件边界在训练数据中高覆盖，测试中仅有粗粒度记录。

2. **对比方法**
   - 普通 irregular encoder；
   - CDI-style context-detail adaptive gate；
   - mask dropout / random missing augmentation；
   - missingness-aware encoder；
   - policy adversarial baseline；
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、TPHN、OCBC。

3. **核心指标**
   - in-policy accuracy；
   - worst-policy accuracy；
   - detail harm rate：`regret_k < 0` 的比例；
   - regret-lower-bound calibration：`regret_lcb` 是否能预测跨政策 detail 可靠性；
   - context fallback accuracy：detail 被 escrow 锁住时 coarse context 的准确率；
   - released-detail reliance：高置信预测中有多少依赖正 `regret_lcb` 的 detail。

4. **消融实验**
   - 去掉 `L_regret_escrow`，验证 detail gate 是否重新依赖训练政策局部触发；
   - 去掉 `L_context_floor`，检查 coarse context 是否退化导致 escrow 失效；
   - 用平均 regret 替代低分位 regret，验证 tail policy 对鲁棒性的重要性；
   - 将 counterfactual detail views 替换为随机 mask，验证收益来自采样政策语义而非普通增强；
   - 只用 context 或只用 detail 分类，验证“托管释放”优于单一路径。

## 5. 预期创新性

1. **从门控相位转向遗憾托管**：吸收 context-detail interaction 的前沿结构，但不扫描 gate threshold、不做 hysteresis、不做 boundary quarantine；detail 释放权由跨政策 regret lower bound 决定。
2. **从采样去偏转向相对决策收益审计**：不估计采样概率、不做对抗、不做一致性、不输出不确定性；只问 detail 相对 coarse context 是否在多种 `do(policy)` 下稳健降低损失。
3. **从“细节越多越好”转向“细节必须自证无害”**：局部高分辨率证据只有通过反事实遗憾审计后才能进入 logits，能抑制报警复测、panel 共现和标注覆盖带来的细节捷径。
4. **与采样解耦/反事实干预框架低侵入兼容**：保留 value process 与 counterfactual sampler，只重定义 sampling branch 的产物为 detail stress recipes。
5. **部署解释性清晰**：当模型回退到 context-only 时，可以报告“detail 在某些采样政策下产生负遗憾”；当 detail 被释放时，可以报告其 `regret_lcb`，直接说明为什么这份局部细节值得信任。

## 6. 一句话投稿卖点

**RE-CDR 首次把非规则采样时间序列分类中的 context-detail 采样捷径表述为“local detail 相对 coarse context 的跨政策反事实遗憾是否为正”的决策托管问题，并通过 Regret-Escrow Router 只释放能在多种采样干预下稳定降低分类遗憾的局部细节，从而在不依赖危险率、对抗、一致性、后验商、随机平滑、拓扑、gauge、纠错码、knockoff、evidential shield、信息格、solver trace、trigger hysteresis 或 control barrier 的前提下提升采样策略偏移鲁棒性。**
