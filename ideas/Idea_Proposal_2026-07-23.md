# Title: Causal Sheaf-Glue Classifier：面向采样策略偏移的反事实观测覆盖粘合分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件；自动化记忆也记录了历史多轮任务均未发现该文件或可替代总结文件。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案机制：
  - 2026-06-17：反事实误差语义制图、ANOVA state/policy error projection、VQ semantic clauses、HSIC redaction。
  - 2026-06-20：采样测度 Radon-Nikodym 比、doubly robust target-measure correction。
  - 2026-06-24：policy-only negative film、shadow eraser/stencil。
  - 2026-06-27：observability witness、观测性 probe bank、低观测性 gating。
  - 2026-07-15 至 2026-07-22：RKHS cubature debiaser、measurement-action bisimulation、policy-word signature renormalizer、thermodynamic free-energy、Sklar copula marginal stripping、triage queue debt、Sinkhorn detail canonicalizer、MDL episode transducer。

### 历史核心机制黑名单

本提案明确避开以下已经出现过的核心机制，避免与历史方案思维重合：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、hazard-driven resampling、分类梯度与采样 score 零空间正交。
8. 生理流算子与采样算子的交换子、value graph / policy graph 交换或分离。
9. additive evidence market、protocol tax、token-level evidence budget、边际证据审计。
10. posterior quotient、模型空间后验除以采样似然因子、干预积分分类。
11. reconstruction error cartography、误差语义编译、HSIC redaction。
12. policy-simplex randomized smoothing、certified policy radius。
13. 采样测度 density ratio、doubly robust correction、influence-bound。
14. optional-stopping martingale query、标准化创新鞅、停时矩控制。
15. soft excursion topology、censored persistence interval、censor envelope、fragmentation sobriety。
16. policy-gauge horizontal transport、chart span supervision、vertical blindness。
17. policy-only negative film、shadow eraser/stencil。
18. latent packet syndrome code、parity check、syndrome repair。
19. conditional knockoff calendar、soft knockoff-FDR firewall、swap symmetry。
20. observability witness / measurement-Jacobian / low-observability routing。
21. subjective-logic evidential shield、policy-induced vacuity、class-wise evidence discount。
22. observation-set policy lattice、meet/join visibility masks、submodular margin。
23. solver trace front-door、NFE/roughness/step-size trace standardization。
24. RKHS cubature weights、measurement-action bisimulation、policy-word signature、thermodynamic annealing、copula marginal stripping、queue debt、Sinkhorn detail canonicalization、MDL episode compression。

本轮选择一个新的正交切入点：**不估计采样概率，不删除或折扣采样信息，不做跨策略 logits/representation 一致，不做对抗、平滑、后验除法、拓扑持久性、gauge 投影、纠错码、knockoff、evidential uncertainty、信息格边际或 solver trace 中和；而是把每种采样策略诱导出的可见片段看成潜在状态域上的一个“观测覆盖”，把局部 value evidence 学成覆盖上的局部截面，并通过可微层论粘合约束学习一个能跨覆盖存在的全局状态截面。分类器只读取可粘合的全局截面，采样策略造成的局部不兼容只表现为粘合残差诊断，不直接进入类别边界。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列中的 sampling-policy shift 经常不是单点缺失，也不是单一 mask ratio 变化，而是“同一条潜在状态轨迹被切成了不同的局部可见片段”：

- ICU 医院 A 把报警后的生命体征、乳酸和血气作为一个短窗口 panel 共同观测，医院 B 把它们拆成跨班次异步片段。
- 可穿戴设备 A 在夜间只保留低频全局上下文，设备 B 在运动后保留局部高频细节。
- 天文或多传感器数据中，不同 band / channel 的可见窗口由天气、设备排程或资源预算决定，导致同一物理状态被不同 patch 覆盖。

近期 `paper_daily.md` 里的两类机制给了直接启发：

1. **Enhancing Sparse Event Detection via Adaptive Gate of Context-Detail Interaction** 强调稀疏医疗事件需要全局 context 与局部 detail 协同，而不是在所有时间点平均处理。对采样偏移来说，不同策略实际上改变了哪些 detail patch 能与哪些 context patch 发生联系。
2. **MVC-CDE / Attentive Kernel Smoothing** 与 **StarEmbed** 都提醒我们：真实不规则观测的“覆盖结构”本身很关键。前者关注 control path 由哪些观测支撑，后者展示多 band / cadence / coverage 变化会影响类别语义可恢复性。

历史方案已经充分探索了采样概率、负控、证据预算、后验因子、停时、拓扑、gauge、纠错码和 solver trace 等方向。本提案换一个问题表述：

> 如果多个局部观测 patch 真的来自同一个底层状态，它们在重叠区域上应该能被粘合成一个一致的全局状态截面；如果某个预测只依赖训练医院特有的观测切片方式，那么这些局部截面会在反事实覆盖变换下无法稳定粘合。

这和当前“采样解耦/反事实干预”框架天然兼容：

- value process 负责在每个局部 patch 上生成状态截面；
- sampling process 不预测策略标签、不估计 hazard、不输出 residual sink，而是生成观测覆盖与 patch-overlap 结构；
- counterfactual intervention 不生成一致性增强样本，而是生成 cover refinement / coarsening / split / merge，用于训练“局部截面是否能按覆盖关系粘合”；
- 分类器只读取由全局粘合层求出的 `global section`，而不是读取 mask、policy id、局部粘合残差或采样日历。

直觉上，稳定状态证据应该能跨不同观测覆盖被局部一致地粘合；训练策略特有的 sampling shortcut 往往只在某种 patch 切法下显得有用，一旦 panel 被拆开、context-detail patch 被改写或局部 detail 被合并，它就会表现为无法粘合的局部截面冲突。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：从反事实视图改为 Counterfactual Observation Cover

新增 `CounterfactualCoverCollator`。它不返回用于一致性学习的多个完整序列，而是返回一组覆盖结构：

1. **事实覆盖 `Cover_f`**
   - 把事件流划分为若干 patch，例如：
     - `global_context_patch`：长窗口粗粒度上下文；
     - `local_detail_patch`：疑似事件附近的细粒度片段；
     - `variable_panel_patch`：同一 panel 或变量组；
     - `band/channel_patch`：同一传感器、band 或临床变量族。
   - 每个 patch 是事件子集，不含 label 或 policy id。

2. **重叠图 `OverlapGraph`**
   - 两个 patch 若共享事件、共享时间窗、共享变量组或在时间上邻近，就形成一条 overlap edge。
   - edge 上记录 `overlap_mask` 与 `restriction_type`，例如 time-overlap、variable-overlap、context-detail-overlap。

3. **反事实覆盖操作 `CoverBank`**
   - `panel_split`：把同步 panel 拆成异步变量 patch。
   - `detail_merge`：把高频 detail patch 合并到粗 context patch。
   - `late_context_drop`：删除晚期上下文覆盖。
   - `band_recover`：恢复某个缺失 band / channel 的覆盖占位，但不伪造数值。

这些操作只改变“局部观测如何覆盖潜在状态域”，不是普通 mask dropout、不是 policy smoothing、不是 meet/join lattice，也不要求不同覆盖下 logits 相同。

### 2.2 改 Encoder：从序列 pooled representation 改为 Local Section Encoder

每个 patch 独立编码为局部截面：

```text
s_p = LocalSectionEncoder(events in patch p)
```

`s_p` 只来自观测值、变量 id、时间戳、测量质量等 value-side 信息；patch 的可见性结构只用于定义覆盖和重叠，不作为类别特征拼进分类头。

随后为每类重叠学习一个 restriction map：

```text
r_{p -> e} = R_type(s_p)
r_{q -> e} = R_type(s_q)
```

若两个 patch 在重叠区域描述的是同一个潜在状态，限制到重叠域后应该一致：

```text
glue_residual_e = r_{p -> e} - r_{q -> e}
```

这不是跨采样视图一致性，因为它不要求完整 representation 或 logits 相同；它只要求同一覆盖内部的局部截面在 overlap 上满足可粘合条件。

### 2.3 新增 Global Section Solver：分类只读可粘合全局截面

给定局部截面 `{s_p}` 和 overlap residual，使用可微加权最小二乘求一个全局截面 `g`：

```text
g = argmin_z sum_p w_p ||A_p z - s_p||^2
          + sum_{(p,q)} w_{pq} ||R_p s_p - R_q s_q||^2
```

实际实现中可用一个轻量 `SheafGluingSolver` 近似：

- 用 patch quality 和 overlap residual 产生 reliability weights；
- 对局部截面做加权聚合得到 `global_section`；
- 高粘合残差的 patch 不被删除，也不进入 policy residual sink，而是降低其对全局截面的可信度；
- 分类器只接收 `global_section`。

这与历史方案区别：

- 不是 evidence market：没有购买 token、税率或预算。
- 不是 knockoff：没有真实/负控日历统计。
- 不是 topology capsule：不计算峰谷持久性或审查区间。
- 不是 gauge：不学习垂直/横向子空间。
- 不是 lattice：不约束 meet/join margin。
- 不是 solver trace：不处理中介 NFE 或 roughness。

### 2.4 改 Loss：从一致性/去偏转向 Sheaf Gluing Axioms

总目标：

```text
L = L_cls
  + lambda_glue  * L_overlap_gluing
  + lambda_cover * L_cover_functoriality
  + lambda_rel   * L_reliability_calibration
  + lambda_rank  * L_section_noncollapse
```

#### A. 分类损失 `L_cls`

```text
L_cls = CE(Classifier(global_section_factual), y)
```

分类器不读取局部 patch id、coverage code、policy recipe、粘合残差向量或反事实覆盖标签。

#### B. Overlap Gluing Loss `L_overlap_gluing`

对事实覆盖内每条 overlap edge：

```text
L_overlap_gluing =
  mean_e ||R_type(s_p) - R_type(s_q)||_2^2 * overlap_weight_e
```

它鼓励局部截面在重叠区域可粘合。若某个 patch 的信息主要来自采样策略捷径，它往往只在自身 patch 内有效，限制到 overlap 后无法与邻近 context/detail patch 对齐。

#### C. Cover Functoriality Loss `L_cover_functoriality`

反事实覆盖操作提供 parent-child patch 关系。例如 panel split 后，原 panel patch 的局部截面应能限制到各个变量子 patch；detail merge 后，多个 detail patch 应能共同限制到粗 context patch。

```text
L_cover_functoriality =
  mean_{parent -> child}
  ||R_parent_child(s_parent) - stopgrad(s_child_on_shared_events)||_2^2
```

这不是多视图表征一致。它不要求事实覆盖和反事实覆盖的全局表示或 logits 相同，只要求覆盖细化/粗化满足“限制映射可复合”的层论结构：先限制到 parent 再限制到 child，与直接在 child patch 编码共享事件，应给出相容状态描述。

#### D. Reliability Calibration `L_reliability_calibration`

`SheafGluingSolver` 会给每个 patch 一个 reliability weight。该权重不能任意压低困难 patch，否则模型会只保留少数容易粘合但无信息的片段。

用 overlap residual 与 patch value quality 校准：

```text
target_rel = sigmoid(value_quality - normalized_glue_residual)
L_reliability_calibration = SmoothL1(rel_weight, stopgrad(target_rel))
```

注意这里的 reliability 不是 uncertainty shield，也不是 observability witness；它只表示“该局部截面能否与邻域粘合”，不直接作为类别概率或拒识分数。

#### E. Section Non-Collapse `L_section_noncollapse`

为了避免所有局部截面塌缩成常数从而 trivially glue，加入批内方差和局部重构约束：

```text
L_section_noncollapse =
  relu(var_floor - Var(global_section))^2
  + local_value_reconstruction_loss
```

该项确保粘合层保留 value-side 状态信息，而不是通过抹平所有 patch 来获得低 gluing residual。

### 2.5 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `LocalSectionEncoder`，每个 patch 独立生成局部状态截面。
- 现有 sampling branch 改为 `CoverBuilder`：只定义 patch、overlap、cover operation，不输出 policy logits 或分类特征。
- 现有 counterfactual intervention 改为 `CoverBank`：生成 split/merge/drop/recover 等覆盖变换，用于训练 restriction maps 的函子性。
- 推理阶段只需事实覆盖：
  1. 构造 patches 与 overlap edges；
  2. 编码局部截面；
  3. 粘合成全局截面；
  4. 分类。
- 部署诊断输出：
  - patch-level glue residual；
  - context-detail mismatch score；
  - panel split instability；
  - global section reliability。

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


class LocalSectionEncoder(nn.Module):
    """Encode one observation patch into a value-driven local section."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.patch_mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        event_value: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
        patch_visibility: torch.Tensor,
        measurement_std: torch.Tensor | None = None,
    ) -> dict:
        # event_*: [B, N], patch_visibility: [B, P, N]
        if measurement_std is None:
            measurement_std = torch.zeros_like(event_value)

        delta_t = torch.zeros_like(event_time)
        delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
        horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = event_time / horizon

        var_h = self.var_embed(event_var_id.clamp_min(0))
        event_x = torch.cat(
            [
                var_h,
                event_value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(measurement_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * event_mask.unsqueeze(-1)

        sections = []
        qualities = []
        for patch_mask in patch_visibility.unbind(dim=1):
            active = (patch_mask * event_mask).clamp(0.0, 1.0)
            patch_h = event_h * active.unsqueeze(-1)
            seq_h, _ = self.patch_mixer(patch_h)
            section = masked_mean(seq_h, active, dim=1)
            quality = self.quality_head(section).squeeze(-1)
            sections.append(section)
            qualities.append(quality)

        return {
            "section": torch.stack(sections, dim=1),      # [B, P, H]
            "quality": torch.stack(qualities, dim=1),     # [B, P]
        }


class RestrictionMaps(nn.Module):
    """Restriction maps from patch sections to overlap coordinates."""

    def __init__(self, hidden_dim: int, num_overlap_types: int):
        super().__init__()
        self.maps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_overlap_types)
            ]
        )

    def forward(self, section: torch.Tensor, overlap_type: torch.Tensor) -> torch.Tensor:
        # section: [B, E, H], overlap_type: [B, E]
        outs = []
        for type_idx, module in enumerate(self.maps):
            out = module(section)
            mask = (overlap_type == type_idx).to(section.dtype).unsqueeze(-1)
            outs.append(out * mask)
        return torch.stack(outs, dim=0).sum(dim=0)


def gather_patch_sections(section: torch.Tensor, patch_index: torch.Tensor) -> torch.Tensor:
    """Gather patch sections for each overlap edge."""

    # section: [B, P, H], patch_index: [B, E]
    hidden_dim = section.size(-1)
    idx = patch_index.unsqueeze(-1).expand(-1, -1, hidden_dim)
    return section.gather(dim=1, index=idx)


class SheafGluingSolver(nn.Module):
    """Approximate global section by reliability-weighted gluing."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.rel_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.global_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        section: torch.Tensor,
        quality: torch.Tensor,
        patch_glue_residual: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> dict:
        # section: [B, P, H], quality/residual/patch_mask: [B, P]
        rel_input = torch.cat(
            [
                section,
                quality.unsqueeze(-1),
                patch_glue_residual.unsqueeze(-1),
            ],
            dim=-1,
        )
        raw_rel = self.rel_head(rel_input).squeeze(-1)
        raw_rel = raw_rel.masked_fill(patch_mask == 0, -1e4)
        weight = torch.softmax(raw_rel, dim=1)
        global_section = torch.einsum("bp,bph->bh", weight, section)
        global_section = self.global_refine(global_section)
        return {"global_section": global_section, "reliability": weight}


class CausalSheafGlueClassifier(nn.Module):
    """Sampling-policy robust classifier based on counterfactual cover gluing."""

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        num_classes: int,
        num_overlap_types: int,
    ):
        super().__init__()
        self.local_encoder = LocalSectionEncoder(num_vars, hidden_dim)
        self.restrict = RestrictionMaps(hidden_dim, num_overlap_types)
        self.gluing = SheafGluingSolver(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.local_decoder = nn.Linear(hidden_dim, 1)

    def encode_cover(self, batch: dict, prefix: str = "") -> dict:
        enc = self.local_encoder(
            event_value=batch[f"{prefix}event_value"],
            event_time=batch[f"{prefix}event_time"],
            event_var_id=batch[f"{prefix}event_var_id"],
            event_mask=batch[f"{prefix}event_mask"],
            patch_visibility=batch[f"{prefix}patch_visibility"],
            measurement_std=batch.get(f"{prefix}measurement_std"),
        )
        section = enc["section"]

        src = gather_patch_sections(section, batch[f"{prefix}overlap_src"])
        dst = gather_patch_sections(section, batch[f"{prefix}overlap_dst"])
        overlap_type = batch[f"{prefix}overlap_type"]
        edge_mask = batch[f"{prefix}overlap_mask"]

        src_res = self.restrict(src, overlap_type)
        dst_res = self.restrict(dst, overlap_type)
        edge_residual = (src_res - dst_res).pow(2).mean(dim=-1) * edge_mask

        patch_residual = torch.zeros_like(enc["quality"])
        patch_count = torch.zeros_like(enc["quality"])
        for edge_idx in range(edge_residual.size(1)):
            res = edge_residual[:, edge_idx]
            s_idx = batch[f"{prefix}overlap_src"][:, edge_idx]
            d_idx = batch[f"{prefix}overlap_dst"][:, edge_idx]
            patch_residual.scatter_add_(1, s_idx[:, None], res[:, None])
            patch_residual.scatter_add_(1, d_idx[:, None], res[:, None])
            patch_count.scatter_add_(1, s_idx[:, None], edge_mask[:, edge_idx : edge_idx + 1])
            patch_count.scatter_add_(1, d_idx[:, None], edge_mask[:, edge_idx : edge_idx + 1])
        patch_residual = patch_residual / patch_count.clamp_min(1.0)

        glued = self.gluing(
            section=section,
            quality=enc["quality"],
            patch_glue_residual=patch_residual,
            patch_mask=batch[f"{prefix}patch_mask"],
        )
        logits = self.classifier(glued["global_section"])
        return {
            **enc,
            **glued,
            "logits": logits,
            "edge_residual": edge_residual,
            "patch_residual": patch_residual,
        }

    def forward(self, batch: dict) -> dict:
        return self.encode_cover(batch, prefix="")

    def overlap_gluing_loss(self, out: dict, batch: dict) -> torch.Tensor:
        edge_mask = batch["overlap_mask"]
        return out["edge_residual"].sum() / edge_mask.sum().clamp_min(1.0)

    def cover_functoriality_loss(self, batch: dict, factual: dict) -> torch.Tensor:
        losses = []
        for prefix in batch["cover_bank_prefixes"]:
            cf = self.encode_cover(batch, prefix=prefix)
            parent_idx = batch[f"{prefix}parent_patch_index"]
            child_idx = batch[f"{prefix}child_patch_index"]
            relation_mask = batch[f"{prefix}parent_child_mask"]

            parent = gather_patch_sections(factual["section"], parent_idx)
            child = gather_patch_sections(cf["section"], child_idx)
            rel_type = batch[f"{prefix}parent_child_type"]
            parent_restricted = self.restrict(parent, rel_type)
            child_restricted = self.restrict(child.detach(), rel_type)
            raw = (parent_restricted - child_restricted).pow(2).mean(dim=-1)
            losses.append((raw * relation_mask).sum() / relation_mask.sum().clamp_min(1.0))
        if not losses:
            return torch.zeros((), device=factual["section"].device)
        return torch.stack(losses).mean()

    def reliability_calibration_loss(self, out: dict, batch: dict) -> torch.Tensor:
        residual = out["patch_residual"].detach()
        quality = out["quality"].detach()
        target = torch.sigmoid(quality - residual)
        raw = F.smooth_l1_loss(out["reliability"], target, reduction="none")
        return (raw * batch["patch_mask"]).sum() / batch["patch_mask"].sum().clamp_min(1.0)

    def section_noncollapse_loss(self, out: dict, batch: dict, var_floor: float = 0.05) -> torch.Tensor:
        global_var = out["global_section"].var(dim=0).mean()
        var_loss = F.relu(var_floor - global_var).pow(2)

        # Lightweight local reconstruction to keep sections value-bearing.
        patch_pred = self.local_decoder(out["section"]).squeeze(-1)
        patch_target = []
        for patch_mask in batch["patch_visibility"].unbind(dim=1):
            active = patch_mask * batch["event_mask"]
            patch_target.append(masked_mean(batch["event_value"], active, dim=1))
        patch_target = torch.stack(patch_target, dim=1)
        recon_raw = F.smooth_l1_loss(patch_pred, patch_target, reduction="none")
        recon_loss = (recon_raw * batch["patch_mask"]).sum() / batch["patch_mask"].sum().clamp_min(1.0)
        return var_loss + recon_loss

    def training_loss(
        self,
        batch: dict,
        lambda_glue: float = 0.20,
        lambda_cover: float = 0.15,
        lambda_rel: float = 0.05,
        lambda_rank: float = 0.05,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]
        cls_loss = F.cross_entropy(out["logits"], labels)
        glue_loss = self.overlap_gluing_loss(out, batch)
        cover_loss = self.cover_functoriality_loss(batch, out)
        rel_loss = self.reliability_calibration_loss(out, batch)
        noncollapse_loss = self.section_noncollapse_loss(out, batch)

        total = (
            cls_loss
            + lambda_glue * glue_loss
            + lambda_cover * cover_loss
            + lambda_rel * rel_loss
            + lambda_rank * noncollapse_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "overlap_gluing_loss": glue_loss.detach(),
            "cover_functoriality_loss": cover_loss.detach(),
            "reliability_calibration_loss": rel_loss.detach(),
            "section_noncollapse_loss": noncollapse_loss.detach(),
            "mean_patch_residual": out["patch_residual"].mean().detach(),
        }
```

### Counterfactual Cover Collator 草稿

```python
@torch.no_grad()
def build_counterfactual_cover(batch: dict, num_vars: int, max_patches: int = 8) -> dict:
    """Create observation patches, overlap edges and cover operations.

    This sketch keeps the data structure explicit. Production code can replace
    the heuristic patches with dataset-specific panels, bands or event windows.
    """

    event_time = batch["event_time"]
    event_var_id = batch["event_var_id"]
    event_mask = batch["event_mask"]
    bsz, num_events = event_time.shape
    device = event_time.device

    horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = event_time / horizon

    patches = []
    # Global context patches.
    patches.append(event_mask)
    patches.append(((time_norm <= 0.5).to(event_mask.dtype) * event_mask))
    patches.append(((time_norm > 0.5).to(event_mask.dtype) * event_mask))

    # Variable-family patches.
    even_var = (event_var_id % 2 == 0).to(event_mask.dtype) * event_mask
    odd_var = (event_var_id % 2 == 1).to(event_mask.dtype) * event_mask
    patches.extend([even_var, odd_var])

    # Detail patches around dense regions.
    gap = torch.zeros_like(event_time)
    gap[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
    dense = (gap <= gap.masked_fill(event_mask == 0, 0.0).mean(dim=1, keepdim=True)).to(event_mask.dtype)
    patches.append(dense * event_mask)

    while len(patches) < max_patches:
        patches.append(torch.zeros_like(event_mask))
    patch_visibility = torch.stack(patches[:max_patches], dim=1)
    patch_mask = (patch_visibility.sum(dim=-1) > 0).to(event_mask.dtype)

    # Build overlap edges by patch intersections.
    src, dst, typ, mask = [], [], [], []
    for i in range(max_patches):
        for j in range(i + 1, max_patches):
            inter = patch_visibility[:, i] * patch_visibility[:, j]
            active = (inter.sum(dim=1) > 0).to(event_mask.dtype)
            src.append(torch.full((bsz,), i, device=device, dtype=torch.long))
            dst.append(torch.full((bsz,), j, device=device, dtype=torch.long))
            # 0: time overlap, 1: variable-family overlap, 2: context-detail overlap.
            typ.append(torch.full((bsz,), 0 if i < 3 and j < 3 else 1 if i >= 3 and j >= 3 else 2, device=device, dtype=torch.long))
            mask.append(active)

    out = dict(batch)
    out["patch_visibility"] = patch_visibility
    out["patch_mask"] = patch_mask
    out["overlap_src"] = torch.stack(src, dim=1)
    out["overlap_dst"] = torch.stack(dst, dim=1)
    out["overlap_type"] = torch.stack(typ, dim=1)
    out["overlap_mask"] = torch.stack(mask, dim=1)

    # A small cover bank: panel split and detail merge.
    out["cover_bank_prefixes"] = ["cf_split_", "cf_merge_"]
    for prefix in out["cover_bank_prefixes"]:
        out[f"{prefix}event_value"] = batch["event_value"]
        out[f"{prefix}event_time"] = batch["event_time"]
        out[f"{prefix}event_var_id"] = batch["event_var_id"]
        out[f"{prefix}event_mask"] = event_mask

    # Split: emphasize variable-family patches.
    split_visibility = patch_visibility.clone()
    split_visibility[:, 0] = 0.0
    out["cf_split_patch_visibility"] = split_visibility
    out["cf_split_patch_mask"] = (split_visibility.sum(dim=-1) > 0).to(event_mask.dtype)
    out["cf_split_overlap_src"] = out["overlap_src"]
    out["cf_split_overlap_dst"] = out["overlap_dst"]
    out["cf_split_overlap_type"] = out["overlap_type"]
    out["cf_split_overlap_mask"] = out["overlap_mask"]

    # Merge: emphasize global context and suppress detail.
    merge_visibility = patch_visibility.clone()
    merge_visibility[:, 5:] = 0.0
    out["cf_merge_patch_visibility"] = merge_visibility
    out["cf_merge_patch_mask"] = (merge_visibility.sum(dim=-1) > 0).to(event_mask.dtype)
    out["cf_merge_overlap_src"] = out["overlap_src"]
    out["cf_merge_overlap_dst"] = out["overlap_dst"]
    out["cf_merge_overlap_type"] = out["overlap_type"]
    out["cf_merge_overlap_mask"] = out["overlap_mask"]

    # Parent-child relation sketches for functoriality.
    for prefix in out["cover_bank_prefixes"]:
        relation_count = min(4, max_patches)
        out[f"{prefix}parent_patch_index"] = torch.arange(relation_count, device=device)[None].expand(bsz, -1)
        out[f"{prefix}child_patch_index"] = torch.arange(relation_count, device=device)[None].expand(bsz, -1)
        out[f"{prefix}parent_child_type"] = torch.zeros(bsz, relation_count, device=device, dtype=torch.long)
        out[f"{prefix}parent_child_mask"] = torch.ones(bsz, relation_count, device=device)

    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `context-detail shift`：训练环境保留报警后的局部 detail patch，测试环境只保留粗 context 或延迟 detail。
   - `panel split/merge shift`：训练环境同步 panel，测试环境拆成异步变量 patch。
   - `band/channel cover shift`：多 band 或多传感器覆盖发生系统性变化。
   - `late-cover blackout shift`：晚期窗口覆盖被设备预算或临床流程遮蔽。

2. **对比方法**
   - 普通 irregular encoder。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - context-detail adaptive gate baseline。
   - MVC-CDE / kernel smoothing baseline。
   - StarEmbed / foundation embedding + OOD score baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN、RKHS Cubature、MABL、PWSR、PTFEA、SSB、TQDN、CASDC、PAMET。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - cover-gluing violation：局部 patch 在 overlap 上的平均粘合残差。
   - context-detail mismatch score：detail patch 与 global context 的限制不一致程度。
   - panel split instability：panel 被拆开后 global section 的分类风险变化。
   - reliability-error calibration：低 reliability patch 是否更容易对应错误预测。

4. **消融实验**
   - 去掉 `L_overlap_gluing`，检查模型是否重新依赖 patch-specific shortcut。
   - 去掉 `L_cover_functoriality`，检查 panel split / detail merge 下是否退化。
   - 去掉 reliability calibration，检查 gluing solver 是否任意丢弃困难 patch。
   - 把 cover operations 替换为随机 mask，验证收益来自覆盖结构而非普通增强。
   - 只用 global context patch 或只用 local detail patch，验证粘合机制是否确实整合二者。

## 5. 预期创新性

1. **从采样机制估计转向观测覆盖粘合**：不问某事件为何被采样，而问不同采样策略切出的局部观测 patch 是否能粘合成同一个全局状态截面。
2. **从 context-detail gate 转向 sheaf gluing**：吸收稀疏事件检测中 context/detail 交互的启发，但不用 gate 决定何时看 detail；而是让 context 与 detail 在 overlap 上接受可粘合性检验。
3. **从 coverage/OOD 评估转向覆盖结构训练**：吸收 StarEmbed 对 band coverage / cadence 的启发，把覆盖变化变成 patch cover 与 restriction map，而不是直接用 foundation embedding 或 OOD score。
4. **从反事实增强转向覆盖函子性**：counterfactual intervention 只提供 split/merge/drop/recover 等 cover operations，训练 restriction maps 的可复合结构，不要求跨覆盖 logits 或表征一致。
5. **与采样解耦框架高度兼容**：value process 生成局部截面，sampling process 生成覆盖，classifier 只读 global section；采样策略信息不会以 policy id、mask pattern 或残差槽形式进入类别边界。
6. **解释性直接对应采样偏移**：粘合残差能定位“哪个局部 patch 与全局 context 不相容”，比单纯输出不确定性、负控 FDR 或 curvature 更能解释具体采样覆盖问题。

## 6. 一句话投稿卖点

**CSG 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“同一潜在状态域被不同采样政策切成不同观测覆盖，导致局部状态截面难以粘合”的问题，并通过 Local Section Encoder、Restriction Maps 与可微 Global Section Solver，让分类器只依赖跨 context/detail、panel split/merge 和 band coverage 变化仍能粘合的全局状态截面，从而在不复用危险率、对抗、一致性、后验商、平滑认证、停时鞅、拓扑、gauge、纠错码、knockoff、evidential、lattice 或 solver-trace 机制的前提下提升采样策略偏移鲁棒性。**
