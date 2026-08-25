# Title: Do-PID Semantic Prism：面向采样策略偏移的跨表示语义棱镜分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`、`*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未检出可读取的工作总结文件。
- 已读取 `paper_daily.md` 最新末段，重点纳入：
  - **MATA-Former & SIICU**：semantic-aware temporal alignment、医学语义有效期、text-intensive ICU event-wise risk modeling。
  - **Cross-Representation Benchmarking**：统一比较网格多变量时序、事件流、文本事件流，提示 representation choice 会显著改变 sampling-policy shortcut 的可见路径。
- 已读取当前工作区全部历史 proposal 文件：
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
  - `ideas/Idea_Proposal_2026-08-24.md`
- 已读取自动化记忆 `MEMORIES.md` 及历史摘要：
  - `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`
  - `idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`
  - `idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-05.md`、`idea_2026-08-06.md`
  - `idea_2026-08-07.md`、`idea_2026-08-08.md`、`idea_2026-08-09.md`
  - `idea_2026-08-10.md`、`idea_2026-08-11.md`、`idea_2026-08-21.md`

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下机制作为主创新：

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
29. outcome-conditioned JEPA latent debate、affirm/rebut predictor、policy-only decoy silence。

本提案选择新的正交切入点：**不删除、不投影、不证明、不投票、不校准集合、不隐私化、不合成反例、不做表示一致或潜表征预测；而是把同一 EHR 原始轨迹渲染成网格、事件流和文本流三种表示，用部分信息分解（Partial Information Decomposition, PID）区分“跨表示冗余的病理语义”“单一表示独有的格式线索”和“采样政策参与后才出现的协同捷径”。最终分类器只读取跨表示冗余语义棱镜折射出的 shared semantic ray。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 最新记录中的两篇工作给出一个此前历史方案没有直接覆盖的缝隙。

第一，**MATA-Former** 把时间注意力从纯几何距离提升到“医学语义有效期”：同样过去 12 小时的事件，对于感染、循环衰竭、肾功能异常等风险可能有完全不同的有效期。这个思想非常适合非规则 ICU，因为文本、护理记录、化验、操作和生命体征并不共享一个统一时间衰减律。但它的风险也很明显：文本事件和护理记录本身高度受医院记录流程影响，semantic-aware temporal bias 可能把“某类记录为何被写下”误当成“某个病理证据仍有效”。

第二，**Cross-Representation Benchmarking** 指出，同一 EHR 可以被整理为三种完全不同的输入表示：规则网格、多变量事件流、文本事件流。不同表示不是中性的工程选择：网格会把采样政策压缩为填补值和 mask；事件流会显式暴露事件可见性、顺序和 panel 共现；文本流会把“未测、下单、待返回、复查”等流程写成语言线索。因此，同一个 sampling-policy shortcut 在三种表示中的可见形态不同。

历史方案大多沿着单一表示内部去处理采样偏移：在时间、图、后验、拓扑、证据、校准、证明、测量项或隐私发布上做机制创新。但如果表示选择本身会改变 shortcut 的暴露方式，那么一个更基础的问题是：

> 哪些分类信息在 grid / event / text 三种渲染中都以医学语义形式重复出现，哪些信息只在某个表示中出现，哪些信息必须依赖采样政策摘要和特定表示共同出现才具有标签预测力？

**Do-PID Semantic Prism (DPSP)** 的核心直觉是：

- 真正跨采样政策稳定的病理状态，通常会在多种表示中形成 **冗余语义信息**：网格趋势、事件值序列、文本事件描述都会指向同一类临床状态。
- 采样政策捷径往往是 **表示独有信息**：例如网格 imputation artifact、事件流 panel 同窗、文本流中的“复查/待返回”语言模板。
- 更危险的是 **policy-synergistic information**：单看表示或单看采样摘要都不强，但二者组合后突然强预测标签，例如“文本中出现复查 + late dense sampling”在训练中心代表高风险，换中心后失效。

因此，DPSP 不要求三种表示 logits 一致，不让它们投票，也不把它们做成保形校准或 JEPA target。它把三种表示当作一块语义棱镜：只有穿过三面仍能保留的冗余病理光线进入分类器；表示独有色散和 policy-synergy 被单独度量并从主分类路径中隔离。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Tri-Render Counterfactual Batch

新增 `TriRenderCollator`，对同一条原始 EHR 轨迹构造三种同步渲染：

1. **Grid view**
   - 按小时或临床窗口聚合 vitals / labs。
   - 保留 value、mask、简单 quality；不把 policy id 输入分类头。

2. **Event view**
   - 保留 `(time, variable, value, quality, semantic_type)` 事件序列。
   - 可表示 MATA-Former 中的 structured + text-intensive ICU events。

3. **Text view**
   - 将事件流转成简短、模板化、可控的 clinical event snippets。
   - 文本只服务语义编码，不把“医院 A / 中心 B / policy recipe”写入内容。

4. **Policy summary**
   - 由 sampling branch 从观测坐标抽取：密度、gap、panel-like 共现、pending 率、文本记录频率、变量覆盖。
   - 只用于 PID 估计和 synergy 审计，不进入最终分类器。

5. **Counterfactual render recipes**
   - `grid_impute_shift`：改变网格填补/聚合窗口。
   - `event_panel_split`：改变事件流 panel 同步形态。
   - `text_documentation_shift`：改变文本事件出现频率和措辞粒度。
   - 这些 recipe 不形成 consistency pair，不做平滑、不做 knockoff、不做 proof audit；它们只帮助估计某个信息是否依赖表示格式与采样政策的协同出现。

### 2.2 改 Encoder：Semantic Prism Encoder

DPSP 使用三个轻量 representation encoder，并共享一个 MATA-style semantic validity stem：

```text
h_grid  = GridEncoder(grid_view)
h_event = EventEncoder(event_view, semantic_validity_bias)
h_text  = TextEventEncoder(text_view, semantic_validity_bias)
```

其中 semantic validity stem 借鉴 MATA-Former，但做两个关键改造：

- 它只输出 **state-validity embedding**，表示医学事件对病理状态的有效期。
- 它不直接把 documentation density、recording latency 或 note frequency 作为分类 bias；这些记录流程特征只进入 policy summary，用于后续 PID synergy 审计。

然后用一个 **PID Prism Splitter** 将每个表示分成三类通道：

```text
r_shared : 三种表示都能支持的冗余语义 ray
u_view   : 某个表示独有的格式 ray
s_policy : 表示与 policy summary 共同出现才带来的协同 ray
```

最终分类头只读取 `r_shared`：

```text
logits = Classifier(r_shared)
```

关键区别：

- 不是三视图 logits 一致。
- 不是 contrastive alignment。
- 不是 Borda / jury / ensemble。
- 不是 conformal / evidential / privacy。
- 不是把三种表示翻译成固定问卷、证明、IRT item 或 synthetic forge。
- 不是要求 grid/event/text 产生同一个 latent；只要求用于分类的那部分信息在 PID 意义上主要来自跨表示冗余语义，而不是表示独有或 policy-synergistic 成分。

### 2.3 改 Loss：从不变性转向 Partial Information Discipline

总目标：

```text
L = L_shared_cls
  + lambda_red * L_redundant_sufficiency
  + lambda_uni * L_unique_label_quarantine
  + lambda_syn * L_policy_synergy_suppression
  + lambda_sem * L_semantic_validity_grounding
```

#### A. Shared Classification `L_shared_cls`

只用冗余语义 ray 分类：

```text
L_shared_cls = CE(Classifier(r_shared), y)
```

这里不平均三种表示 logits，也不使用最强表示。若某个表示独有线索很强，它可以被 `u_view` 记录，但不能直接给最终分类器。

#### B. Redundant Sufficiency `L_redundant_sufficiency`

冗余语义 ray 必须足够解释标签。实现上让 `r_shared` 经过一个共享 label head，预测真实标签：

```text
L_redundant_sufficiency = CE(LabelHead(r_shared), y)
```

注意它不是跨表示一致性：`h_grid`、`h_event`、`h_text` 可以不同，三个表示的 private channel 可以保留各自格式差异。约束只作用在被 splitter 明确抽出的 shared ray 上。

#### C. Unique Label Quarantine `L_unique_label_quarantine`

每个表示独有通道 `u_grid/u_event/u_text` 允许解释格式和渲染差异，但不应单独强预测标签。用冻结式 label probe 做非对抗审计：

```text
p_unique_y = Probe_y(stopgrad(u_view))
L_unique_label_quarantine =
  mean_view relu(CE_uniform - CE(p_unique_y, y))^2
```

含义是：如果某个 private channel 单独就能高预测标签，很可能是该表示暴露了训练政策捷径，例如文本模板、网格填补模式或事件 panel 同步。

这不是 policy adversarial，因为 probe 不预测环境或政策，也不反向欺骗主 encoder；它只让 private channel 的标签可用性保持在可审计低水平。

#### D. Policy Synergy Suppression `L_policy_synergy_suppression`

最危险的 shortcut 是“表示 private channel + policy summary”共同预测标签。DPSP 显式训练一个 synergy probe：

```text
logits_syn = Probe_syn([u_view, policy_summary])
gain_syn = CE(Probe_y(u_view), y) - CE(logits_syn, y)
L_policy_synergy_suppression = relu(gain_syn - tau_syn)^2
```

直觉：如果加入采样摘要后，某个表示独有通道突然显著增强标签预测，说明存在 policy-synergistic shortcut。该损失压低这种增益，而不是要求所有 counterfactual views 输出相同预测。

#### E. Semantic Validity Grounding `L_semantic_validity_grounding`

吸收 MATA-Former 的 semantic-aware temporal alignment，但只用它给 shared ray 提供医学有效期：

```text
validity_weight = sigmoid(g_semantic(event_type, risk_query, delta_t))
r_shared = weighted_prism_pool(h_grid, h_event, h_text, validity_weight)
```

用 Plateau-Gaussian soft labels 或 event-wise risk labels 监督 shared ray 的时间有效期：

```text
L_semantic_validity =
  BCE(RiskCurveHead(r_shared, query_time), soft_risk_curve)
```

它不是 disease-progress poset、proof sequent 或 TCF foreseeing；没有偏序边、证明前提或未来事件生成。它只校准“某类医学语义在多长时间内仍应影响当前风险”。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value process 改为三个 render-specific encoders：`GridEncoder / EventEncoder / TextEventEncoder`。
- 现有 sampling branch 改为 `PolicySummaryEncoder`：只为 PID synergy 审计提供采样摘要，不进入分类头。
- 现有 counterfactual intervention 改为 `TriRenderRecipeBank`：生成 grid / event / text 各自的渲染变化，用于估计 unique 与 synergy，而不是做一致性、平滑、proof、conformal、privacy 或 synthetic pretraining。
- 推理阶段默认只需事实三渲染，输出：
  - `pred_label`；
  - `shared_semantic_energy`：冗余语义支撑强度；
  - `unique_label_leakage`：各表示 private channel 的标签泄漏；
  - `policy_synergy_gain`：采样摘要加入后 private channel 的标签增益；
  - `representation_shortcut_map`：shortcut 主要来自 grid artifact、event panel 还是 text documentation。

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


def uniform_cross_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return -log_prob.mean(dim=-1).mean()


class SemanticValidityStem(nn.Module):
    """MATA-style semantic validity without feeding documentation density to logits."""

    def __init__(self, num_semantic_types: int, hidden_dim: int):
        super().__init__()
        self.type_embed = nn.Embedding(num_semantic_types, hidden_dim)
        self.validity = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        event_type: torch.Tensor,
        risk_query: torch.Tensor,
        delta_t: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        type_h = self.type_embed(event_type.clamp_min(0))
        query_h = risk_query[:, None].expand_as(type_h)
        x = torch.cat(
            [
                type_h,
                query_h,
                torch.log1p(delta_t).unsqueeze(-1),
                event_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        return torch.sigmoid(self.validity(x)).squeeze(-1) * event_mask


class EventEncoder(nn.Module):
    def __init__(self, num_vars: int, num_semantic_types: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.type_embed = nn.Embedding(num_semantic_types, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, batch: dict, validity_weight: torch.Tensor | None = None) -> torch.Tensor:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        sem_type = batch["event_semantic_type"]
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon

        x = torch.cat(
            [
                self.var_embed(var_id.clamp_min(0)),
                self.type_embed(sem_type.clamp_min(0)),
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(x)
        if validity_weight is not None:
            event_h = event_h * validity_weight.unsqueeze(-1)
        event_h = event_h * mask.unsqueeze(-1)
        seq_h, _ = self.context(event_h)
        seq_h = self.out(seq_h)
        return masked_mean(seq_h, mask, dim=1)


class GridEncoder(nn.Module):
    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3 * num_vars, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, grid: dict) -> torch.Tensor:
        value = grid["grid_value"]
        mask = grid["grid_mask"]
        quality = grid.get("grid_quality", torch.ones_like(value))
        x = torch.cat([value, mask, quality], dim=-1)
        h = self.proj(x)
        seq_h, _ = self.temporal(h)
        step_mask = (mask.sum(dim=-1) > 0).to(value.dtype)
        return masked_mean(seq_h, step_mask, dim=1)


class TextEventEncoder(nn.Module):
    """Encode controlled text-event embeddings; upstream text encoder can be frozen."""

    def __init__(self, text_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, text: dict, validity_weight: torch.Tensor | None = None) -> torch.Tensor:
        emb = text["text_event_embedding"]
        time = text["text_event_time"]
        mask = text["text_event_mask"]
        doc_quality = text.get("documentation_quality", torch.ones_like(time))
        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        x = torch.cat([emb, (time / horizon).unsqueeze(-1), doc_quality.unsqueeze(-1)], dim=-1)
        h = self.proj(x)
        if validity_weight is not None and validity_weight.shape == mask.shape:
            h = h * validity_weight.unsqueeze(-1)
        h = h * mask.unsqueeze(-1)
        seq_h, _ = self.context(h)
        return masked_mean(seq_h, mask, dim=1)


class PolicySummaryEncoder(nn.Module):
    """Summarize observation process for PID probes only, never for final logits."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, event_batch: dict) -> torch.Tensor:
        time = event_batch["event_time"]
        var_id = event_batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = event_batch["event_mask"]
        pending = event_batch.get("value_pending", torch.zeros_like(mask))
        text_seen = event_batch.get("text_documented", torch.zeros_like(mask))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_rate = F.one_hot(var_id, self.num_vars).to(time.dtype) * mask.unsqueeze(-1)
        var_rate = var_rate.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        close = (delta_t <= masked_mean(delta_t, mask, dim=1).unsqueeze(-1).clamp_min(1e-6)).to(time.dtype)
        var_change = torch.zeros_like(mask)
        var_change[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(time.dtype)

        stats = torch.cat(
            [
                mask.mean(dim=1, keepdim=True),
                masked_mean((time_norm <= 0.33).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean(((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean((time_norm > 0.66).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean(torch.log1p(delta_t), mask, dim=1).unsqueeze(-1),
                masked_mean(close * var_change, mask, dim=1).unsqueeze(-1),
                masked_mean(pending, mask, dim=1).unsqueeze(-1),
                masked_mean(text_seen, mask, dim=1).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(torch.cat([var_rate, stats], dim=-1))


class PIDPrismSplitter(nn.Module):
    """Split tri-render representations into redundant, unique, and policy-synergy rays."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.shared_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.shared_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.unique_proj = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(3)
            ]
        )

    def forward(self, h_grid: torch.Tensor, h_event: torch.Tensor, h_text: torch.Tensor) -> dict:
        tri = torch.cat([h_grid, h_event, h_text], dim=-1)
        gate = self.shared_gate(tri)
        shared = self.shared_proj(tri) * gate

        views = [h_grid, h_event, h_text]
        unique = []
        for idx, view in enumerate(views):
            unique.append(self.unique_proj[idx](torch.cat([view, shared.detach()], dim=-1)))
        return {
            "shared": shared,
            "unique_grid": unique[0],
            "unique_event": unique[1],
            "unique_text": unique[2],
            "shared_gate": gate,
        }


class DoPIDSemanticPrism(nn.Module):
    """Tri-render PID classifier for sampling-policy robust irregular EHR classification."""

    def __init__(
        self,
        num_vars: int,
        num_semantic_types: int,
        text_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.validity = SemanticValidityStem(num_semantic_types, hidden_dim)
        self.risk_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.grid_encoder = GridEncoder(num_vars, hidden_dim)
        self.event_encoder = EventEncoder(num_vars, num_semantic_types, hidden_dim)
        self.text_encoder = TextEventEncoder(text_dim, hidden_dim)
        self.policy_encoder = PolicySummaryEncoder(num_vars, hidden_dim)
        self.prism = PIDPrismSplitter(hidden_dim)

        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_classes))
        self.shared_head = nn.Linear(hidden_dim, num_classes)
        self.unique_probe = nn.ModuleDict(
            {
                "grid": nn.Linear(hidden_dim, num_classes),
                "event": nn.Linear(hidden_dim, num_classes),
                "text": nn.Linear(hidden_dim, num_classes),
            }
        )
        self.synergy_probe = nn.ModuleDict(
            {
                "grid": nn.Linear(2 * hidden_dim, num_classes),
                "event": nn.Linear(2 * hidden_dim, num_classes),
                "text": nn.Linear(2 * hidden_dim, num_classes),
            }
        )
        self.risk_curve_head = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch: dict) -> dict:
        event = batch["event_view"]
        bsz = event["event_value"].size(0)
        risk_query = self.risk_query.expand(bsz, -1)
        delta_t = torch.zeros_like(event["event_time"])
        delta_t[:, 1:] = (event["event_time"][:, 1:] - event["event_time"][:, :-1]).clamp_min(0.0)
        validity_weight = self.validity(
            event_type=event["event_semantic_type"],
            risk_query=risk_query,
            delta_t=delta_t,
            event_mask=event["event_mask"],
        )

        h_grid = self.grid_encoder(batch["grid_view"])
        h_event = self.event_encoder(event, validity_weight=validity_weight)
        h_text = self.text_encoder(batch["text_view"])
        policy_h = self.policy_encoder(event)
        prism = self.prism(h_grid, h_event, h_text)
        logits = self.classifier(prism["shared"])
        return {
            **prism,
            "logits": logits,
            "shared_logits": self.shared_head(prism["shared"]),
            "policy_h": policy_h,
            "validity_weight": validity_weight,
        }

    def semantic_validity_loss(self, out: dict, batch: dict) -> torch.Tensor:
        if "soft_risk_curve" not in batch:
            return torch.zeros((), device=out["logits"].device)
        query_time = batch["risk_query_time"]
        shared = out["shared"][:, None].expand(-1, query_time.size(1), -1)
        x = torch.cat([shared, query_time.unsqueeze(-1)], dim=-1)
        pred = self.risk_curve_head(x).squeeze(-1)
        return F.binary_cross_entropy_with_logits(pred, batch["soft_risk_curve"].to(pred.dtype))

    def pid_probe_losses(
        self,
        out: dict,
        labels: torch.Tensor,
        synergy_tau: float = 0.08,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_losses = []
        synergy_losses = []
        for name in ["grid", "event", "text"]:
            u = out[f"unique_{name}"]
            unique_logits = self.unique_probe[name](u.detach())
            unique_ce = F.cross_entropy(unique_logits, labels)
            unique_losses.append(F.relu(uniform_cross_entropy(unique_logits) - unique_ce).pow(2))

            syn_logits = self.synergy_probe[name](torch.cat([u, out["policy_h"].detach()], dim=-1))
            syn_ce = F.cross_entropy(syn_logits, labels)
            gain = (unique_ce.detach() - syn_ce).clamp_min(0.0)
            synergy_losses.append(F.relu(gain - synergy_tau).pow(2))
        return torch.stack(unique_losses).mean(), torch.stack(synergy_losses).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_red: float = 0.25,
        lambda_uni: float = 0.15,
        lambda_syn: float = 0.30,
        lambda_sem: float = 0.10,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        shared_cls = F.cross_entropy(out["logits"], labels)
        redundant_loss = F.cross_entropy(out["shared_logits"], labels)
        unique_loss, synergy_loss = self.pid_probe_losses(out, labels)
        sem_loss = self.semantic_validity_loss(out, batch)
        total = (
            shared_cls
            + lambda_red * redundant_loss
            + lambda_uni * unique_loss
            + lambda_syn * synergy_loss
            + lambda_sem * sem_loss
        )
        return {
            "loss": total,
            "shared_cls_loss": shared_cls.detach(),
            "redundant_sufficiency_loss": redundant_loss.detach(),
            "unique_label_quarantine_loss": unique_loss.detach(),
            "policy_synergy_suppression_loss": synergy_loss.detach(),
            "semantic_validity_loss": sem_loss.detach(),
        }
```

## 4. Tri-Render Collator 草稿

```python
@torch.no_grad()
def build_tri_render_batch(raw_batch: dict, text_embedder) -> dict:
    """Build grid/event/text views for PID semantic prism training.

    Counterfactual recipes alter rendering format only for PID diagnostics.
    They are not contrastive positives and are not used for logits consistency.
    """

    event_view = {
        "event_value": raw_batch["event_value"],
        "event_time": raw_batch["event_time"],
        "event_var_id": raw_batch["event_var_id"],
        "event_semantic_type": raw_batch["event_semantic_type"],
        "event_mask": raw_batch["event_mask"],
        "measurement_quality": raw_batch.get("measurement_quality", torch.ones_like(raw_batch["event_value"])),
        "value_pending": raw_batch.get("value_pending", torch.zeros_like(raw_batch["event_value"])),
        "text_documented": raw_batch.get("text_documented", torch.zeros_like(raw_batch["event_value"])),
    }

    grid_view = render_grid_view(
        event_value=event_view["event_value"],
        event_time=event_view["event_time"],
        event_var_id=event_view["event_var_id"],
        event_mask=event_view["event_mask"],
        num_vars=raw_batch["num_vars"],
        num_bins=raw_batch.get("num_grid_bins", 24),
    )
    text_view = render_text_event_view(event_view, text_embedder=text_embedder)

    out = dict(raw_batch)
    out["event_view"] = event_view
    out["grid_view"] = grid_view
    out["text_view"] = text_view
    return out


def render_grid_view(
    event_value: torch.Tensor,
    event_time: torch.Tensor,
    event_var_id: torch.Tensor,
    event_mask: torch.Tensor,
    num_vars: int,
    num_bins: int,
) -> dict:
    bsz, num_events = event_value.shape
    device = event_value.device
    horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    bin_id = ((event_time / horizon) * num_bins).long().clamp(0, num_bins - 1)
    grid_value = torch.zeros(bsz, num_bins, num_vars, device=device, dtype=event_value.dtype)
    grid_mask = torch.zeros_like(grid_value)
    for idx in range(num_events):
        b = torch.arange(bsz, device=device)
        t = bin_id[:, idx]
        v = event_var_id[:, idx].clamp(0, num_vars - 1)
        active = event_mask[:, idx] > 0
        grid_value[b[active], t[active], v[active]] = event_value[active, idx]
        grid_mask[b[active], t[active], v[active]] = 1.0
    return {
        "grid_value": grid_value,
        "grid_mask": grid_mask,
        "grid_quality": torch.ones_like(grid_value),
    }


def render_text_event_view(event_view: dict, text_embedder) -> dict:
    snippets = make_controlled_event_snippets(event_view)
    text_embedding = text_embedder(snippets)
    return {
        "text_event_embedding": text_embedding,
        "text_event_time": event_view["event_time"],
        "text_event_mask": event_view["event_mask"],
        "documentation_quality": 1.0 - 0.5 * event_view.get("value_pending", torch.zeros_like(event_view["event_mask"])),
    }
```

## 5. 实验切入点

1. **Policy shift 构造**
   - `representation-specific shortcut shift`：训练时 grid imputation artifact 与标签相关，测试时反转。
   - `event panel shift`：训练中心 panel 同步，测试中心 panel 拆分。
   - `text documentation shift`：训练中心用更细护理记录或复查文本，测试中心文本更粗或延迟记录。
   - `semantic validity shift`：同一医学事件在不同记录流程下出现时间不同，但病理有效期应保持相近。

2. **对比方法**
   - 单表示 grid / event / text baseline。
   - Cross-Representation Benchmarking 风格统一预处理 baseline。
   - MATA-Former-style semantic time bias baseline。
   - 普通三表示 logits average / concatenation。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DSPP、DCPD、DIPF、DRG-SFF、DPPC、DCOFF、DD-JEPA 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy worst AUROC / AUPRC。
   - shared semantic energy：最终预测中 shared ray 的能量占比。
   - unique label leakage：单表示 private channel 的标签 probe 能力。
   - policy synergy gain：`[u_view, policy_summary]` 相比 `u_view` 的标签预测增益。
   - representation shortcut map：错误样本中 shortcut 主要来自 grid、event 还是 text。
   - semantic validity calibration：MATA-style validity weight 与 event-wise soft risk label 的匹配度。

4. **消融实验**
   - 去掉 `L_policy_synergy_suppression`，检查表示独有通道是否与 policy summary 合谋预测标签。
   - 去掉 PID splitter，直接 concat 三表示，验证是否学到更强采样捷径。
   - 去掉 semantic validity grounding，只做三表示 PID，验证 MATA 式语义有效期对 text-intensive ICU 的价值。
   - 只用两种表示做 PID，检验三面棱镜相比双表示审计是否更能发现 shortcut。
   - 将 text snippets 中的 documentation phrases 去掉，验证文本流 shortcut 是否主要来自记录流程语言。

## 6. 预期创新性

1. **从单表示去偏转向跨表示信息分解**：历史方案大多在某一种 representation 内修补采样偏移；DPSP 直接把 representation choice 当作机制变量，用 PID 区分冗余语义、表示独有线索和 policy-synergy。
2. **从 MATA 时间偏置转向语义有效期审计**：吸收 MATA-Former 的医学语义时间有效期，但不让 documentation density 或 note timing 直接成为 temporal bias 捷径。
3. **从 Cross-Representation Benchmark 转向可训练鲁棒机制**：不只是比较 grid/event/text 谁更强，而是让三种表示共同揭示哪些信息可跨格式迁移、哪些只是格式与采样政策共同制造的捷径。
4. **从一致性/投票/校准转向 PID discipline**：不要求三视图 logits 一致，不做 jury、不做 conformal、不做 JEPA、不做 privacy；只约束最终分类信息来源必须主要来自 shared semantic ray。
5. **诊断粒度直接面向部署**：当跨中心失败时，可以指出问题来自网格填补、事件流 panel、文本记录模板，还是它们与采样摘要之间的协同信息。

## 7. 一句话投稿卖点

**DPSP 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“同一 EHR 原始轨迹在 grid / event / text 三种表示中暴露出不同 shortcut 色散”的问题，通过 MATA-style semantic validity、三表示 PID prism splitter、unique label quarantine 与 policy-synergy suppression，让分类器只读取跨表示冗余的病理语义 ray，从而避免把网格填补、事件 panel、文本记录流程及其与采样摘要的协同信号误当成可迁移类别证据。**
