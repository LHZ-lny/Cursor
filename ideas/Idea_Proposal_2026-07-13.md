# Title: Policy-Lattice Submodular Margins：面向采样策略偏移的采样信息格次模边际分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已读取自动化记忆：确认历史多轮任务中同样未发现 `my_work_summary.md` 或可替代总结文件。
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-06-12.md`
  - `paper_daily_2026-06-25.md`
  - `paper_daily_2026-06-26.md`
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`
  - `Idea_Proposal_2026-06-24.md`
  - `Idea_Proposal_2026-06-27.md`

### 历史核心机制黑名单

为避免与历史提案和 `paper_daily.md` 中已有机制发生思维重合，本提案明确避开以下方向作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer。
8. 分类梯度与采样 score 的零空间正交化。
9. hazard-driven counterfactual resampling。
10. 多个 `do(policy)` 视图的 risk variance 约束。
11. 生理流算子与采样算子的交换子手术。
12. value graph / policy graph 的交换或分离损失。
13. policy residual sink 作为采样残差收纳槽。
14. additive evidence market、protocol tax、token-level evidence budget 或边际证据审计。
15. 后验商动力学、模型空间 posterior quotient、采样似然因子相除、干预积分分类。
16. reconstruction error cartography、ANOVA-style state/policy error projection、VQ semantic clauses、policy-sensitive acquisition clauses、HSIC redaction checksum。
17. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler、policy coverage loss。
18. 采样测度上的 Radon-Nikodym density ratio、doubly robust target-measure correction、influence-bound regularization。
19. previsible martingale query、continuous-discrete standardized innovation、optional-stopping moment control、stopping recipe collator、predictability barrier。
20. soft excursion topology、censored persistence interval likelihood、censor envelope、fragmentation sobriety。
21. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
22. policy-only negative film、dual-exposure value/shadow encoders、latent shadow eraser/stencil、shadow nullification high-entropy loss、value retention buckets。
23. latent packet codeword、learnable parity-check、syndrome locator、syndrome-guided packet repair、codeword closure。
24. conditional knockoff calendar、calendar adapter、knockoff group statistics、soft knockoff-FDR firewall、swap symmetry calibration。
25. measurement-Jacobian/Fisher-style observability witness、counterfactual observability probe bank、low-quantile state-coordinate gate、observability-routed classification。
26. subjective-logic / Dirichlet evidential classification、policy-induced ignorance/vacuity mass、class-wise evidence discount、policy stress audit bank。
27. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing、MedMamba 的 frequency/adaptive graph branch、MedSpaformer 的普通 token sparsification、MILM 的 value-redacted classifier、StarEmbed 的直接 foundation embedding 或 LLMTS 的普通 LLM alignment 作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗，不要求多视图表示一致，不做后验除法、随机平滑、停时鞅、拓扑胶囊、gauge 投影、纠错码、knockoff、观测性 gating 或 evidential uncertainty；而是把不同采样政策产生的观测集合组织成一个“采样信息格”。分类器可以利用更多观测，但其真实类边际必须服从信息偏序上的单调性和次模性：额外观测带来的边际收益应递减，不能因为某个医院特有的 panel 共现、时间窗密集或变量组合突然产生超加性分类跳跃。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

采样策略偏移的危险经常不是单个 token、单条边或单个时间戳，而是某些观测组合在训练环境中被制度性地一起看见：

- 医院 A 把 lactate、WBC、CRP 作为同一 panel 同步下单，医院 B 只在特定病程阶段异步检测；
- 设备 A 在夜间低电量时只保留少量通道，设备 B 在夜间保持高密度监测；
- 训练机构中“报警后 2 小时密集复测”与某类标签高度相关，测试机构改为固定查房后这个组合消失；
- 天文巡天中的 band coverage、cadence 和 measurement noise 共同决定某些相位/频率信息是否可恢复。

很多历史方案尝试删除、折扣、审计、修复或隔离采样信息；但真实世界中，更多观测通常确实带来更多状态信息。关键不是禁止使用采样差异，而是给分类器一个更结构化的契约：

> 如果采样政策 B 比政策 A 包含更多、更高质量或覆盖更完整的观测，那么真实类边际不应无故下降；同时，两个采样子集的联合收益不应大于它们分别带来的收益之和，否则模型很可能在利用某种训练环境特有的“共现捷径”。

近期 `paper_daily.md` 中两个机制给出直接启发：

1. **StarEmbed** 强调真实不规则观测包含多波段覆盖、异步采样、异方差测量误差和 OOD 评估。它提示我们：采样政策偏移会改变“任务相关信息是否可恢复”，因此应当在训练目标中显式表达观测集合的覆盖/质量偏序，而不是只看 mask ratio。
2. **Rethinking LLMs for Irregular Time Series Classification in Critical Care** 指出，ICU 不规则分类的关键在前端 encoder 是否尊重时间戳、缺失和异步观测，而不是后端 LLM alignment。由此可见，与其继续堆叠大模型或语义摘要，不如给 irregular encoder 加上可验证的信息格契约。

**Policy-Lattice Submodular Margins (PLSM)** 的核心思想是：当前“采样解耦/反事实干预”框架不再生成一致性视图、风险方差视图或 policy-only 负控，而是生成一组可比较的观测子集：

```text
A, B: 两个不同采样政策下可见的观测集合
A meet B = A ∩ B: 两个政策共同可见的最小信息
A join B = A ∪ B: 两个政策合并后的最大信息
```

模型在这些集合上预测同一个标签，但训练目标不是让 logits 相同，而是约束真实类 margin 作为集合函数满足：

```text
Monotonicity:  margin(A join B) >= margin(A), margin(B)
Submodularity: margin(A) + margin(B) >= margin(A join B) + margin(A meet B)
```

这两个不等式能精准打击 sampling-policy shortcut：

- 若模型只依赖单个真实状态观测，增加观测通常会带来递减收益；
- 若模型依赖训练医院特有的 panel 共现或复测组合，`A join B` 往往会产生异常大的超加性 margin jump，违反次模性；
- 若换采样政策后真实类边际反而下降，说明 encoder 没有把“更多可恢复状态信息”转化为更稳定分类证据。

因此，PLSM 不把采样模式直接作为分类特征，也不把它彻底删掉，而是让采样干预模块为 encoder 提供 **信息偏序训练契约**。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通 event encoder 改为 Observation-Lattice Encoder

PLSM 可以包裹现有 irregular encoder，但推荐将前端改为事件集合式的 `ObservationLatticeEncoder`：

1. **Event Atom Lift**
   - 每个观测事件 `(value, time, variable, delta_t, measurement_std)` 被编码为一个 value-driven atom。
   - `measurement_std` 或 `quality_score` 只描述观测质量，借鉴 StarEmbed 的异方差启发。
   - 不把环境标签、policy id、value-redacted sampling pattern 或 missingness pattern 直接拼入分类头。

2. **Coverage-Aware Set Pooling**
   - encoder 接收任意观测子集，输出 `h(S)`。
   - 使用 event mask 指定当前 view 的可见集合 `S`。
   - 时间戳仍被编码，但只作为事件属性进入 atom，不构造 learnable reference points、不做 path flow、不做 graph decay。

3. **Margin Readout**
   - 分类头输出 logits。
   - 训练时把真实类 margin 看成集合函数：

```text
f_y(S) = logit_y(S) - max_{k != y} logit_k(S)
```

PLSM 的创新不在于新建一个复杂 backbone，而在于约束 `f_y(S)` 在采样信息格上具有合理形状：更多可用信息应提升或至少不损害真实类边际，但组合收益应递减。

### 2.2 改 Loss：从一致性/对抗/不确定性转向 Monotone-Submodular Margin Contracts

总目标：

```text
L = L_cls
  + lambda_mono * L_lattice_monotone
  + lambda_sub  * L_lattice_submodular
  + lambda_q    * L_quality_order
  + lambda_curv * L_shortcut_curvature
```

#### A. 分类损失 `L_cls`

事实观测集合直接训练分类：

```text
L_cls = CE(logits(S_factual), y)
```

这与普通分类器一致，但后续所有格约束都作用在同一个 encoder 上。

#### B. 信息偏序单调性 `L_lattice_monotone`

由反事实采样模块生成两个 policy views `A` 和 `B`，并构造 `join = A ∪ B`。若 `join` 包含更多观测或更高质量观测，它不应降低真实类边际：

```text
L_lattice_monotone =
  relu(f_y(A) + eps_gain(A, join) - f_y(join))^2
  + relu(f_y(B) + eps_gain(B, join) - f_y(join))^2
```

其中 `eps_gain` 由新增观测覆盖、测量质量和时间窗覆盖计算，通常是很小的非负 margin。它不是强制 logits 相同，而是要求“信息更丰富的 view 不应更不相信真实类别”。

#### C. 次模边际契约 `L_lattice_submodular`

构造 `meet = A ∩ B` 与 `join = A ∪ B`。对真实类 margin 施加可微次模不等式：

```text
L_lattice_submodular =
  relu(f_y(join) + f_y(meet) - f_y(A) - f_y(B) - slack)^2
```

直觉：

- `A` 和 `B` 各自已有的状态证据，合并后应表现为递减边际收益；
- 若 `join` 中某个训练策略特有的共现组合让 margin 暴涨，则违反次模性；
- 这种超加性恰恰是 panel shortcut、复测 burst shortcut、特定 band/cadence shortcut 的典型形态。

这不是 consistency loss，因为 `logits(A)`、`logits(B)`、`logits(join)` 可以不同；PLSM 只限制它们在信息格上的相对边际形状。

#### D. 质量偏序约束 `L_quality_order`

借鉴 StarEmbed 的异方差测量启发，若两个 view 的观测集合相同，但一个 view 的测量噪声更低或 band/变量覆盖更完整，则真实类 margin 不应更差：

```text
L_quality_order =
  relu(f_y(noisy_view) + q_margin - f_y(clean_view))^2
```

这把 sampling-policy shift 从“缺失率变化”提升为“可恢复信息质量变化”，尤其适合多波段科学观测、ICU 化验值延迟返回、设备质量漂移等场景。

#### E. Shortcut Curvature Penalty `L_shortcut_curvature`

为显式定位采样捷径，计算同一 batch 内的离散二阶差分：

```text
curvature = f_y(join) - f_y(A) - f_y(B) + f_y(meet)
L_shortcut_curvature = mean(relu(curvature - tau)^2)
```

高正曲率表示“两个采样子集单独看都不强，合在一起突然强预测”。这通常不是稳定生理/状态证据，而是训练机构采样流程制造的组合捷径。PLSM 把这种超加性作为训练信号压制，而不是像历史方案那样估计 hazard、做对抗、做 knockoff-FDR 或输出不确定性。

### 2.3 改 Dataloader：返回 Policy Lattice Views，而不是一致性增强样本

新增 `PolicyLatticeCollator`。每个 batch 返回：

1. `factual_batch`：原始观测事件。
2. `view_a`：由一个反事实采样 policy recipe 生成的可见子集，例如 early-window routine、variable-budget、band coverage、alarm-followup thinning。
3. `view_b`：另一个不同 policy recipe 的可见子集。
4. `view_meet = view_a ∩ view_b`：两个政策共同可见观测。
5. `view_join = view_a ∪ view_b`：两个政策合并可见观测。
6. `clean_quality_view` 与 `noisy_quality_view`：同一可见集合下不同测量质量，用于质量偏序。
7. `lattice_gain`：由新增观测比例、时间窗覆盖、变量覆盖和测量质量提升计算出的非负小 margin。

关键区别：

- 不生成对比学习 pair。
- 不要求多 view logits 一致。
- 不估计采样概率、密度比或危险率。
- 不把 policy id、mask pattern、value-redacted prediction 作为分类输入。
- 不做负控 knockoff、观测性 probe、evidential uncertainty 或 token 预算。
- 只利用反事实干预定义观测集合之间的偏序关系。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为或包裹为 `ObservationLatticeEncoder`，能在任意 event visibility mask 下前向。
- 现有 sampling branch 不进入分类头，只负责生成可比较的 policy recipes 和 lattice masks。
- 现有 counterfactual intervention 从“生成增强视图用于一致性/风险/平滑/审计”改为“生成 meet/join 信息格视图用于边际契约”。
- 推理阶段无需生成 lattice views；只用事实观测集合预测。
- 部署诊断可输出：
  - monotone violation score：事实样本附近是否存在“更多信息反而更差”的异常；
  - submodular curvature score：预测是否依赖超加性采样组合；
  - quality-order violation：低噪声/高覆盖 view 是否提升真实类边际。

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


def true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logit = logits.gather(1, labels[:, None]).squeeze(1)
    rival_logit = logits.masked_fill(
        F.one_hot(labels, logits.size(-1)).bool(),
        -1e4,
    ).max(dim=-1).values
    return true_logit - rival_logit


class ObservationLatticeEncoder(nn.Module):
    """Encode arbitrary visible subsets of irregular events."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.set_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict, visibility: torch.Tensor | None = None) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]
        measurement_std = batch.get("measurement_std", torch.zeros_like(value))

        if visibility is None:
            visibility = event_mask
        active = (event_mask * visibility).clamp(0.0, 1.0)

        horizon = (time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_h = self.var_embed(var_id.clamp_min(0))
        event_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(measurement_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        atom = self.event_proj(event_x) * active.unsqueeze(-1)
        pooled = masked_mean(atom, active, dim=1)
        state = self.set_mixer(pooled)
        logits = self.classifier(state)
        return {"logits": logits, "state": state, "active": active}


def lattice_gain(
    low_visibility: torch.Tensor,
    high_visibility: torch.Tensor,
    event_time: torch.Tensor,
    event_mask: torch.Tensor,
    scale: float = 0.05,
) -> torch.Tensor:
    """Small non-negative margin based on added coverage and time-span gain."""

    low = (low_visibility * event_mask).clamp(0.0, 1.0)
    high = (high_visibility * event_mask).clamp(0.0, 1.0)
    added = (high - low).clamp_min(0.0)

    added_rate = added.sum(dim=1) / event_mask.sum(dim=1).clamp_min(1.0)
    horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = event_time / horizon
    early_added = ((time_norm <= 0.33).to(event_time.dtype) * added).amax(dim=1).values
    late_added = ((time_norm >= 0.66).to(event_time.dtype) * added).amax(dim=1).values
    coverage_span = 0.5 * (early_added + late_added)
    return scale * (added_rate + coverage_span).clamp(0.0, 1.0)


class PolicyLatticeSubmodularMargins(nn.Module):
    """Sampling-policy robust classifier with lattice margin contracts."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.encoder = ObservationLatticeEncoder(num_vars, hidden_dim, num_classes)

    def forward(self, batch: dict, visibility: torch.Tensor | None = None) -> dict:
        return self.encoder(batch, visibility)

    def training_loss(
        self,
        batch: dict,
        lambda_mono: float = 0.25,
        lambda_sub: float = 0.20,
        lambda_q: float = 0.10,
        lambda_curv: float = 0.05,
        submod_slack: float = 0.05,
        curvature_tau: float = 0.10,
    ) -> dict:
        labels = batch["labels"]

        factual = self.forward(batch)
        cls_loss = F.cross_entropy(factual["logits"], labels)

        out_a = self.forward(batch, batch["view_a_visibility"])
        out_b = self.forward(batch, batch["view_b_visibility"])
        out_meet = self.forward(batch, batch["view_meet_visibility"])
        out_join = self.forward(batch, batch["view_join_visibility"])

        m_a = true_margin(out_a["logits"], labels)
        m_b = true_margin(out_b["logits"], labels)
        m_meet = true_margin(out_meet["logits"], labels)
        m_join = true_margin(out_join["logits"], labels)

        gain_a = lattice_gain(
            batch["view_a_visibility"],
            batch["view_join_visibility"],
            batch["event_time"],
            batch["event_mask"],
        )
        gain_b = lattice_gain(
            batch["view_b_visibility"],
            batch["view_join_visibility"],
            batch["event_time"],
            batch["event_mask"],
        )

        monotone_loss = (
            F.relu(m_a + gain_a - m_join).pow(2)
            + F.relu(m_b + gain_b - m_join).pow(2)
        ).mean()

        # Submodularity over the true-class margin set function.
        submod_violation = m_join + m_meet - m_a - m_b - submod_slack
        submod_loss = F.relu(submod_violation).pow(2).mean()

        # Same visible set, different measurement quality.
        clean = self.forward(batch, batch["quality_clean_visibility"])
        noisy_batch = dict(batch)
        if "measurement_std_noisy" in batch:
            noisy_batch["measurement_std"] = batch["measurement_std_noisy"]
        noisy = self.forward(noisy_batch, batch["quality_noisy_visibility"])
        m_clean = true_margin(clean["logits"], labels)
        m_noisy = true_margin(noisy["logits"], labels)
        quality_gain = batch.get("quality_margin", torch.full_like(m_clean, 0.03))
        quality_loss = F.relu(m_noisy + quality_gain - m_clean).pow(2).mean()

        curvature = m_join - m_a - m_b + m_meet
        curvature_loss = F.relu(curvature - curvature_tau).pow(2).mean()

        total = (
            cls_loss
            + lambda_mono * monotone_loss
            + lambda_sub * submod_loss
            + lambda_q * quality_loss
            + lambda_curv * curvature_loss
        )

        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "lattice_monotone_loss": monotone_loss.detach(),
            "lattice_submodular_loss": submod_loss.detach(),
            "quality_order_loss": quality_loss.detach(),
            "shortcut_curvature_loss": curvature_loss.detach(),
            "mean_submodular_curvature": curvature.mean().detach(),
            "mean_join_margin": m_join.mean().detach(),
        }


@torch.no_grad()
def build_policy_lattice_views(batch: dict, num_vars: int) -> dict:
    """Create meet/join visibility masks for policy-lattice margin contracts."""

    event_time = batch["event_time"]
    event_var_id = batch["event_var_id"]
    event_mask = batch["event_mask"]
    measurement_std = batch.get("measurement_std", torch.zeros_like(event_time))
    bsz, num_events = event_time.shape
    device = event_time.device

    horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = event_time / horizon

    # Policy A: routine early/middle coverage.
    early_middle = (time_norm <= 0.66).to(event_mask.dtype) * event_mask

    # Policy B: variable-budget view with alternating variable groups.
    even_var = (event_var_id % 2 == 0).to(event_mask.dtype)
    budget = torch.zeros_like(event_mask)
    for var_idx in range(num_vars):
        var_hit = ((event_var_id == var_idx) & (event_mask > 0)).to(event_mask.dtype)
        order = var_hit.cumsum(dim=1)
        keep = (order <= 3).to(event_mask.dtype) * var_hit
        budget = torch.maximum(budget, keep)
    view_b = torch.maximum(even_var * event_mask, budget * event_mask)

    view_a = early_middle
    view_meet = torch.minimum(view_a, view_b) * event_mask
    view_join = torch.maximum(view_a, view_b) * event_mask

    # Quality pair: identical visibility, but one view receives higher measurement noise.
    quality_clean = view_join
    quality_noisy = view_join
    noisy_std = measurement_std + 0.20 * batch["event_value"].detach().abs().mean(dim=1, keepdim=True)

    out = dict(batch)
    out["view_a_visibility"] = view_a
    out["view_b_visibility"] = view_b
    out["view_meet_visibility"] = view_meet
    out["view_join_visibility"] = view_join
    out["quality_clean_visibility"] = quality_clean
    out["quality_noisy_visibility"] = quality_noisy
    out["quality_margin"] = 0.03 + 0.02 * (
        noisy_std.mean(dim=1) > measurement_std.mean(dim=1)
    ).to(event_mask.dtype)

    out["measurement_std_noisy"] = noisy_std
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `panel-superadditivity shift`：训练环境中某些变量组合经常同步出现，测试环境拆成异步项目。
   - `coverage-quality shift`：训练环境测量误差低且覆盖完整，测试环境某些变量或 band 异方差增大。
   - `routine-vs-triggered shift`：固定查房式采样与报警触发式采样互换。
   - `cadence-coverage shift`：借鉴 StarEmbed，多 band / 多变量 coverage 与 cadence 同时改变，测试任务相关信息可恢复性的变化。

2. **对比方法**
   - 普通 irregular encoder。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - MILM-style value-redacted sampling classifier。
   - MedSpaformer-style token sparsification baseline。
   - StarEmbed / foundation embedding + OOD score baseline。
   - LLMTS 风格 encoder/alignment baseline。
   - DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES 等历史方案。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - lattice monotone violation rate：信息更多的 view 是否反而更差。
   - submodular curvature score：预测是否依赖超加性采样组合。
   - quality-order violation：测量质量更好时真实类边际是否提升。
   - shortcut localization：哪些变量组/时间窗组合导致最大正曲率。

4. **消融实验**
   - 去掉 `L_lattice_monotone`，检查更多观测是否仍可能降低真实类边际。
   - 去掉 `L_lattice_submodular`，验证 panel / 共现 shortcut 是否重新出现。
   - 去掉 `L_quality_order`，验证异方差 policy shift 下是否退化。
   - 将 meet/join views 替换为随机 mask，验证收益来自信息格结构而非普通增强。
   - 只保留单调性、不保留次模性，检查是否无法压制超加性组合捷径。

## 5. 预期创新性

1. **从采样去偏转向信息格边际契约**：不删除采样信息、不估计采样概率、不做对抗或一致性，而是约束分类边际在观测集合偏序上的合理形状。
2. **从 token 选择转向超加性捷径抑制**：吸收 MedSpaformer 对可见 token 分布偏移的关注，但不做 token sparsification；重点是发现并压制采样组合导致的正曲率。
3. **从 OOD/异方差评估转向质量偏序训练**：吸收 StarEmbed 对 band coverage、cadence 和 measurement noise 的启发，把测量质量纳入边际单调契约。
4. **从 LLM alignment 转向 encoder contract**：吸收 LLMTS 的组件审计结论，优先让 irregular encoder 满足采样信息格上的结构约束，而不是依赖后端语言模型。
5. **与反事实干预低侵入兼容**：counterfactual sampler 只需产出可见集合 `A/B/meet/join`，无需构造危险率、密度比、knockoff、停时矩、证据税、纠错 syndrome 或不确定性盾。
6. **部署解释性清晰**：若某个预测依赖训练医院特有的 panel/cadence 组合，submodular curvature 会显著升高，可直接作为采样捷径诊断。

## 6. 一句话投稿卖点

**PLSM 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“观测集合信息格上的分类边际形状失真”问题，并通过 meet/join policy-lattice views、单调边际约束与次模曲率惩罚，让模型利用更多可恢复状态信息的同时，抑制医院协议、设备 cadence 或多变量 panel 共现制造的超加性采样捷径。**
