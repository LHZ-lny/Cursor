# Title: Do-Lyapunov Interval Observer：面向采样策略偏移的反事实收缩区间观测器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md`、`**/*work*summary*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化持久记忆 `MEMORIES.md`，并额外读取最近历史摘要 `idea_2026-09-21.md`、`idea_2026-09-20.md`、`idea_2026-09-19.md`、`idea_2026-09-18.md`、`idea_2026-08-21.md`。
- 已读取/抽取当前仓库 `ideas/Idea_Proposal_*.md` 的全部历史提案标题、黑名单、核心方法段与近期完整正文，覆盖当前工作区已落盘的 2026-06-12 至 2026-09-21 提案。
- 已读取近期 `paper_daily_2026-09-16.md` 与 `paper_daily_2026-09-21.md` 完整内容，并从 `paper_daily.md`、`paper_daily_2026-09-*.md` 抽取近期前沿机制。重点纳入：
  - **INTERVenE**：把原始 ICU EHR 点观测转换为知识型 temporal-abstraction intervals，使分类解释落到临床命名区间、趋势、状态和上下文。
  - **RoMAE**：连续位置 / Axial RoPE 可表达不规则时间与通道坐标，但强位置编码也可能放大采样节律泄漏。
  - **SLAN**：switch layer 避免插补未观测变量，但 switch 激活频率本身可能成为医院政策捷径。
  - **TimeCHEAT**：局部 channel-dependent 与全局 channel-independent 的分层提醒我们，短窗跨通道协同与长期通道独立应被区别处理。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下机制作为主创新：

1. learnable reference points / adaptive time encoding、频域掩码修复、prototype 约束、missingness pattern 直接分类、简单 policy adversarial / IRM。
2. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
3. 生理流-采样算子交换子、value/policy graph 分离、policy residual sink。
4. protocol tax / additive evidence market、posterior quotient、density ratio / doubly robust、policy-simplex randomized smoothing。
5. reconstruction error cartography、VQ clauses、optional-stopping martingale、censored topology、policy gauge、syndrome code、knockoff calendar。
6. observability witness、evidential vacuity、information lattice、solver trace front-door、conformal sleeves、IV/control-function、Borda jury。
7. Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock、feasible hull、IRT-DIF、RG fixed point。
8. bitemporal curtain、clinical tomography、matched risk-set likelihood、Gaussian privacy cloak、CauKer orthogonal synthetic forge、JEPA 双辩裁判。
9. cross-representation PID prism、Noether semantic action、fast/slow meta workflow adapter、test-time workflow adaptation。
10. ORA collider factorization、explain-away responsibility gate、workflow-swap path-specific seal。
11. source atlas / policy-antipodal retrieval、frozen predictor input-adapter contract。
12. DeepFRC-inspired monotone registration、minimal-Schwarzian warp canonicalizer、warp-curvature sidecar。
13. alternating-renewal internal clocks、renewal-normalized reward-rate readout、renewal-balance residual、exposure duty calibration。
14. palimpsest / fixed-capacity pathology memory、write pressure、duplicate fatigue、homeostatic memory slots。
15. typed SSA observation IR、semantic switchboard、dead-code elimination。
16. Dirichlet/Robin boundary condition solver、Green kernel interior field、policy flux sobriety。
17. statistical experiment / Blackwell order / learned garbling kernel / Le Cam deficiency discipline。
18. elicitable pathology functionals、strict proper scoring rules、identification residual、recoverability-gated functional compass。
19. temporal-abstraction interval 直接 state/policy 双分支、interval vocabulary 分解、普通 counterfactual interval-stream consistency。

本提案选择新的正交切入点：**不把区间 token 直接分成 state interval 与 policy interval，不学习 Blackwell 实验筛，不做泛函适当评分，不编译 SSA，也不要求多采样视图 logits 一致；而是把分类器改造成一个带 Lyapunov 能量的连续病程观测器。采样政策可以制造更多 interval、repeat、pending、panel split 或 low-rate read，但这些只允许产生有限的 policy impulse work；若没有语义新息，观测器状态必须在反事实采样序列之间收缩到同一个病程吸引子。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

INTERVenE 的 temporal-abstraction interval 给了一个很有吸引力的接口：与其让 Transformer 直接在 `(time, variable, value)` token 上归因，不如先把 ICU/EHR 事件翻译成临床可命名的状态、趋势、干预和上下文区间。问题在于，**interval token 的数量、边界和持续时间本身也会被采样政策塑形**：

- 高频血糖监测会生成更多 hyperglycemia / trend intervals；低频监测可能只看到粗略状态。
- 某个医院把 lab panel 同步记录，另一个医院拆成异步返回，区间边界会被人为切碎。
- 报警后密集复测会产生大量短区间，普通 Transformer 很容易把“短区间很多”当作风险证据。
- pending、repeat check、护理记录习惯会制造 interval churn，即临床状态没变，但抽象词汇表频繁刷新。

历史方案已经覆盖了很多去偏路径：危险率、后验商、图交换、证据税、拓扑审查、保形护套、IRT、RG、SSA、Blackwell、泛函征询等。本轮换一个动态系统视角：

> 同一潜在病程在不同采样政策下会被切成不同的 interval stream，但如果这些差异只是记录制度造成的，病程观测器的内部状态不应被无限推动。真正应改变状态的，是能带来 clinical semantic innovation 的 interval；纯采样制度造成的 interval churn 只能产生可计量、受限且被阻尼的 impulse work。

**Do-Lyapunov Interval Observer (DLIO)** 将当前“采样解耦/反事实干预”框架改造成三层：

1. **Interval innovation process**：从 INTERVenE 式命名区间中估计临床语义新息 `nu_t`，例如状态首次进入危险区、趋势方向改变、跨变量关系出现新证据。
2. **Policy impulse process**：从 interval 数量、边界抖动、pending / repeat / panel split / low-rate read 中估计采样制度冲击 `pi_t`，但它不进入分类头。
3. **Lyapunov observer process**：用带可学习能量函数 `V(z)` 的状态观测器更新病程状态 `z_t`，要求纯 policy impulse 下状态能量收缩，而语义新息才允许有限扩张。

这能解决 sampling-policy shift 的关键原因是：

- **反复复测不会无限累积证据**：重复区间若没有语义新息，只能被 Lyapunov 阻尼，不能像普通 RNN 那样因为更新次数多而推高风险。
- **允许真实病程突变改变状态**：若 interval 显示真正的新趋势或新状态，收缩约束会被 semantic innovation budget 放宽，不会把急性恶化误删。
- **不强迫多视图一致**：低频采样确实可能错过细节；DLIO 不要求 logits 完全相同，而要求“由采样制度引起的额外状态功”被 Lyapunov 能量解释和限制。
- **解释性直接面向偏移**：模型能报告某次预测由哪些 semantic innovations 推动，以及哪些 interval churn 被识别为 policy impulse 并被阻尼。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Interval Impulse Bank，而不是 consistency views

新增 `LyapunovIntervalCollator`。每个样本先经过知识型 temporal abstraction，形成区间事件：

```text
interval_i = {
    concept_id: clinical state / trend / intervention / context,
    start_time, end_time,
    value_bin: low / normal / high / rising / falling / severe,
    channel_semantic: lab / vital / medication / intervention,
    boundary_quality: measured / interpolated / pending / administrative,
    source_recipe: factual / repeat_dense / panel_split / low_rate / pending_stub
}
```

反事实采样模块不再返回普通增强视图，而是返回 **Interval Impulse Bank**：

1. `factual_interval_stream`
   - 原始 KBTA / INTERVenE 式区间序列。

2. `counterfactual_interval_streams`
   - `repeat_dense`：在相同临床状态下加入短间隔复测，测试模型是否把复测次数当风险。
   - `panel_split`：把同步 panel 拆成异步返回，测试区间边界碎裂是否推动状态漂移。
   - `pending_stub`：加入 value-pending / administrative interval，测试无值记录是否注入证据。
   - `low_rate_read`：降低采样率，允许细粒度新息不可见，但不允许低频本身决定类别。
   - `boundary_jitter`：轻微移动 interval 边界，测试边界抖动的状态功是否受限。
   - `semantic_innovation_insert`：插入确有临床意义的趋势反转或严重状态，作为“应允许状态改变”的正例。

3. `alignment_index`
   - 用粗时间锚和 concept family 对不同反事实 stream 的阶段进行软对齐。
   - 它不是 optimal transport / Sinkhorn，不求全局匹配，只提供相邻阶段的对齐索引。

4. `innovation_target`
   - 弱监督哪些 interval 引入了真实语义新息：新 concept 出现、value bin 跨危险阈值、趋势方向改变、持续异常首次超过临床阈值。

5. `policy_impulse_target`
   - 弱监督哪些 interval 主要来自采样制度：repeat、pending、panel split、administrative boundary jitter、device duty cycle。

关键区别：

- 这些 stream 不是 contrastive positives；
- 不要求 logits / hidden state 完全一致；
- 不估计采样 hazard、density ratio、Blackwell order、proper functional score 或 posterior quotient；
- 不生成 proof、conformal set、program IR、boundary condition、memory slot、privacy noise 或 retrieval atlas；
- 反事实采样只用于估计“policy impulse 在 Lyapunov 能量中做了多少功”。

### 2.2 改 Encoder：Policy-Damped Lyapunov Interval Observer

每个 interval 先被编码成语义向量：

```text
h_i = IntervalStem(
    concept_id_i,
    value_bin_i,
    start_i,
    duration_i,
    channel_semantic_i,
    boundary_quality_i
)
```

这里吸收近期机制但改变用途：

- INTERVenE 的 interval abstraction 只提供可命名临床事件，不直接当分类 token 池化；
- RoMAE 式连续坐标用于计算 interval duration 与相对阶段，而不作为类别捷径；
- SLAN 式 switch 思想用于“只有真实有值 interval 才能触发 semantic innovation”，但 switch 激活次数不直接分类；
- TimeCHEAT 的局部/全局通道洞察用于定义 interval 的 semantic neighborhood，而不学习 state/policy 图边分解。

#### A. Semantic Innovation Gate

对每个 interval 输出临床新息：

```text
nu_i = sigmoid(InnovationHead(h_i, z_{i-1}))
```

`nu_i` 高表示该 interval 相对当前病程状态带来新语义，例如首次进入 severe bin、趋势反转、异常持续时间跨阈值。`nu_i` 低表示重复测量、边界抖动或行政刷新。

#### B. Policy Impulse Head

并行估计采样制度冲击：

```text
pi_i = sigmoid(PolicyImpulseHead(h_i, metadata_i))
```

`pi_i` 只进入 Lyapunov 正则和诊断，不进入分类头。它解释 repeat、pending、panel split、low-rate read、boundary jitter 这类由观测制度造成的 impulse。

#### C. Lyapunov-Damped Observer Update

观测器状态 `z_i` 表示当前病程。候选更新为：

```text
delta_i = CandidateUpdate(z_{i-1}, h_i)
```

然后用新息门与政策阻尼共同调制：

```text
damp_i = nu_i * (1 - stopgrad(pi_i))
z_i = z_{i-1} + damp_i * delta_i
```

直觉：

- 真实临床新息 `nu_i` 可以推动状态；
- policy impulse `pi_i` 越高，更新越被阻尼；
- 若一个 interval 同时有真实新息和政策冲击，模型必须通过 Lyapunov loss 证明状态能量变化由新息预算解释。

#### D. Learnable Lyapunov Energy

学习一个正定能量：

```text
V(z_a, z_b) = || L (z_a - z_b) ||_2^2
```

其中 `L` 是低秩或 Cholesky 参数化矩阵。对同一样本的事实 stream 与反事实 stream，在软对齐阶段上比较 `V(z^f_t, z^c_t)`。

### 2.3 改 Loss：从表示不变转向 Counterfactual Contraction Discipline

总目标：

```text
L = L_cls
  + lambda_con * L_counterfactual_contraction
  + lambda_work * L_policy_impulse_work
  + lambda_nov * L_semantic_innovation
  + lambda_rec * L_low_rate_recoverability
  + lambda_diag * L_impulse_diagnostics
```

#### A. Observer Classification `L_cls`

只使用最终病程观测器状态分类：

```text
logits = Classifier(z_T)
L_cls = CE(logits, y)
```

分类头不接收 interval count、stream recipe、center id、pending flag、panel id、policy descriptor 或 `pi_i`。

#### B. Counterfactual Contraction `L_counterfactual_contraction`

对事实 stream `f` 与反事实 stream `c` 的相邻对齐阶段：

```text
V_t = V(z^f_t, z^c_t)
V_prev = V(z^f_{t-1}, z^c_{t-1})
innovation_budget_t = alpha * (nu^f_t + nu^c_t)

L_con = relu(V_t - gamma * V_prev - innovation_budget_t - eps)^2
```

其中 `gamma < 1`。含义：

- 若反事实差异只是 repeat / pending / boundary jitter，则 `nu` 低，状态距离必须收缩；
- 若确有 semantic innovation，则允许状态距离增加，但增加量必须被新息预算解释；
- 这不是 logits consistency，也不是普通 representation matching，而是用 Lyapunov 能量约束采样政策造成的状态功。

#### C. Policy Impulse Work `L_policy_impulse_work`

估计每个 interval 对 Lyapunov 能量的瞬时做功：

```text
work_i = pi_i * max(0, V(z_i, z_{i-1}) - beta * nu_i)
L_work = mean(work_i)
```

若一个高 policy impulse interval 导致状态大幅移动，但没有对应 semantic innovation，就被惩罚。

#### D. Semantic Innovation Supervision `L_semantic_innovation`

用弱标签监督新息门：

```text
L_nov = BCE(nu_i, innovation_target_i)
```

弱标签可由区间概念变化、value bin 跨阈值、趋势方向改变、严重状态持续时间首次达标产生。它不是 fixed viva、sequent proof、IRT item 或 proper functional target。

#### E. Low-Rate Recoverability `L_low_rate_recoverability`

低频读取可能真的无法恢复某些细粒度病理。DLIO 不强迫低频与高频一致，而要求低频 view 的新息门更保守：

```text
L_rec = relu(nu_low_rate - nu_high_rate - margin)^2
```

对粗粒度节律 / 长期趋势可设置例外 mask，使低频仍能表达可恢复新息。

#### F. Impulse Diagnostics `L_impulse_diagnostics`

policy impulse head 需要能诊断采样制度，但其输出被 `stopgrad` 隔离出分类路径：

```text
L_diag = CE(PolicyRecipeHead(pi_summary), policy_recipe_id)
```

这不是 adversarial 去偏；它鼓励模型显式识别采样制度冲击，再通过 `L_work` 限制其做功。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

现有框架中的三类模块可以这样替换：

1. **Sampling branch**
   - 旧：估计或编码采样政策，并尝试从分类表示中剥离。
   - 新：输出 `policy impulse pi_i` 与 recipe diagnostics，只用于 Lyapunov work penalty 和偏移告警。

2. **Counterfactual intervention**
   - 旧：生成 mask/time/policy views 做一致性、对抗或风险约束。
   - 新：生成 interval impulse bank，专门区分 repeat/pending/panel/boundary jitter 与真实 semantic innovation。

3. **Classifier**
   - 旧：可能读取 pooled tokens、state-policy 分解后的 state branch 或校准后的表示。
   - 新：只读取 Lyapunov observer 的最终病程状态 `z_T`，并可报告每个 interval 的 `nu_i`、`pi_i` 与 `work_i`。

---

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x, mask, dim, eps=1e-6):
    mask = mask.to(dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)


class IntervalStem(nn.Module):
    """Encode INTERVenE-style temporal abstraction intervals."""

    def __init__(self, num_concepts, num_bins, num_channels, hidden_dim):
        super().__init__()
        self.concept = nn.Embedding(num_concepts, hidden_dim)
        self.value_bin = nn.Embedding(num_bins, hidden_dim)
        self.channel = nn.Embedding(num_channels, hidden_dim)
        self.boundary = nn.Embedding(4, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, concept_id, value_bin, channel_id, boundary_quality, start, end):
        duration = (end - start).clamp_min(0.0)
        time_feat = torch.stack([start, end, duration], dim=-1)
        h = (
            self.concept(concept_id)
            + self.value_bin(value_bin)
            + self.channel(channel_id)
            + self.boundary(boundary_quality)
            + self.time_proj(time_feat)
        )
        return self.out(h)


class LyapunovMetric(nn.Module):
    """Low-rank positive semidefinite energy V(a,b)=||L(a-b)||^2."""

    def __init__(self, state_dim, rank=None):
        super().__init__()
        rank = rank or state_dim
        self.factor = nn.Parameter(torch.randn(rank, state_dim) * 0.02)

    def forward(self, a, b):
        diff = a - b
        projected = F.linear(diff, self.factor)
        return projected.pow(2).sum(dim=-1)


class LyapunovIntervalObserver(nn.Module):
    def __init__(
        self,
        num_concepts,
        num_bins,
        num_channels,
        hidden_dim,
        state_dim,
        num_classes,
        num_policy_recipes,
    ):
        super().__init__()
        self.stem = IntervalStem(num_concepts, num_bins, num_channels, hidden_dim)
        self.init_state = nn.Parameter(torch.zeros(state_dim))
        self.to_state = nn.Linear(hidden_dim, state_dim)

        self.innovation_head = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.policy_impulse_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.policy_recipe_head = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_policy_recipes),
        )
        self.lyapunov = LyapunovMetric(state_dim)

    def encode_intervals(self, batch):
        return self.stem(
            batch["concept_id"],
            batch["value_bin"],
            batch["channel_id"],
            batch["boundary_quality"],
            batch["start"],
            batch["end"],
        )

    def rollout(self, batch):
        h = self.encode_intervals(batch)
        mask = batch["mask"].to(h.dtype)
        batch_size, steps, _ = h.shape
        z = self.init_state.unsqueeze(0).expand(batch_size, -1)

        states = []
        innovations = []
        impulses = []
        works = []

        for t in range(steps):
            h_t = h[:, t]
            active = mask[:, t].unsqueeze(-1)
            gate_input = torch.cat([h_t, z], dim=-1)

            nu = torch.sigmoid(self.innovation_head(gate_input))
            pi = torch.sigmoid(self.policy_impulse_head(h_t))
            delta = self.candidate(gate_input)

            # Policy impulse damps state movement but is detached from the
            # classification route, so diagnostics cannot become evidence.
            damp = nu * (1.0 - pi.detach())
            z_next = z + active * damp * delta

            step_work = pi.squeeze(-1) * self.lyapunov(z_next, z)
            z = torch.where(active.bool(), z_next, z)

            states.append(z)
            innovations.append(nu.squeeze(-1) * mask[:, t])
            impulses.append(pi.squeeze(-1) * mask[:, t])
            works.append(step_work * mask[:, t])

        states = torch.stack(states, dim=1)
        innovations = torch.stack(innovations, dim=1)
        impulses = torch.stack(impulses, dim=1)
        works = torch.stack(works, dim=1)

        final_state = states[:, -1]
        logits = self.classifier(final_state)
        impulse_summary = torch.stack(
            [
                masked_mean(innovations, mask, dim=1),
                masked_mean(impulses, mask, dim=1),
            ],
            dim=-1,
        )
        recipe_logits = self.policy_recipe_head(impulse_summary)

        return {
            "logits": logits,
            "states": states,
            "innovation": innovations,
            "impulse": impulses,
            "work": works,
            "recipe_logits": recipe_logits,
        }


def aligned_energy(metric, states_a, states_b, align_index, align_mask):
    """Compare factual/counterfactual states at soft-aligned phase indices."""
    batch, steps = align_index.shape
    dim = states_b.size(-1)
    gather_idx = align_index.clamp_min(0).unsqueeze(-1).expand(batch, steps, dim)
    b_aligned = torch.gather(states_b, 1, gather_idx)
    energy = metric(states_a, b_aligned)
    return energy * align_mask.to(energy.dtype)


def counterfactual_contraction_loss(model, out_f, out_cf, batch_pair, gamma=0.85, eps=1e-3):
    align_mask = batch_pair["align_mask"]
    energy = aligned_energy(
        model.lyapunov,
        out_f["states"],
        out_cf["states"],
        batch_pair["align_index"],
        align_mask,
    )
    prev = F.pad(energy[:, :-1], (1, 0), value=0.0)
    innovation_budget = batch_pair["innovation_scale"] * (
        out_f["innovation"] + torch.gather(
            out_cf["innovation"],
            1,
            batch_pair["align_index"].clamp_min(0),
        )
    )
    violation = energy - gamma * prev - innovation_budget - eps
    return (F.relu(violation).pow(2) * align_mask).sum() / align_mask.sum().clamp_min(1.0)


def dlio_training_loss(model, factual_batch, cf_batch, pair_batch, labels, weights):
    out_f = model.rollout(factual_batch)
    out_cf = model.rollout(cf_batch)

    cls_loss = F.cross_entropy(out_f["logits"], labels)
    con_loss = counterfactual_contraction_loss(model, out_f, out_cf, pair_batch)

    work_margin = factual_batch["innovation_target"].to(out_f["work"].dtype)
    work_loss = F.relu(out_f["work"] - weights["work_beta"] * work_margin).mean()

    nov_loss = F.binary_cross_entropy(
        out_f["innovation"],
        factual_batch["innovation_target"].to(out_f["innovation"].dtype),
        reduction="none",
    )
    nov_loss = (nov_loss * factual_batch["mask"]).sum() / factual_batch["mask"].sum().clamp_min(1.0)

    impulse_loss = F.binary_cross_entropy(
        out_f["impulse"],
        factual_batch["policy_impulse_target"].to(out_f["impulse"].dtype),
        reduction="none",
    )
    impulse_loss = (impulse_loss * factual_batch["mask"]).sum() / factual_batch["mask"].sum().clamp_min(1.0)

    recipe_loss = F.cross_entropy(out_f["recipe_logits"], factual_batch["policy_recipe_id"])

    total = (
        cls_loss
        + weights["con"] * con_loss
        + weights["work"] * work_loss
        + weights["nov"] * nov_loss
        + weights["impulse"] * impulse_loss
        + weights["diag"] * recipe_loss
    )

    return total, {
        "cls_loss": cls_loss.detach(),
        "contraction_loss": con_loss.detach(),
        "policy_work_loss": work_loss.detach(),
        "innovation_loss": nov_loss.detach(),
        "impulse_loss": impulse_loss.detach(),
        "recipe_loss": recipe_loss.detach(),
    }
```

### 3.1 Collator 草稿

```python
class LyapunovIntervalCollator:
    def __init__(self, abstraction_engine, intervention_bank):
        self.abstraction_engine = abstraction_engine
        self.intervention_bank = intervention_bank

    def __call__(self, samples):
        factual = []
        counterfactual = []
        pairs = []

        for sample in samples:
            intervals = self.abstraction_engine.to_intervals(sample["events"])
            recipe = self.intervention_bank.sample_recipe()
            cf_intervals = self.intervention_bank.apply(intervals, recipe)

            factual.append(self.tensorize(intervals, recipe_id=0))
            counterfactual.append(self.tensorize(cf_intervals, recipe_id=recipe.id))
            pairs.append(self.align(intervals, cf_intervals))

        return {
            "factual": self.pad_batch(factual),
            "counterfactual": self.pad_batch(counterfactual),
            "pair": self.pad_pair_batch(pairs),
            "label": torch.tensor([s["label"] for s in samples], dtype=torch.long),
        }

    def tensorize(self, intervals, recipe_id):
        # Weak labels are generated from interval semantics, not from the class.
        innovation_target = [
            int(x.crosses_risk_threshold or x.trend_reversal or x.first_severe_duration)
            for x in intervals
        ]
        policy_impulse_target = [
            int(x.source_recipe in {"repeat_dense", "pending_stub", "panel_split", "boundary_jitter"})
            for x in intervals
        ]
        return {
            "concept_id": torch.tensor([x.concept_id for x in intervals]),
            "value_bin": torch.tensor([x.value_bin for x in intervals]),
            "channel_id": torch.tensor([x.channel_id for x in intervals]),
            "boundary_quality": torch.tensor([x.boundary_quality for x in intervals]),
            "start": torch.tensor([x.start for x in intervals], dtype=torch.float32),
            "end": torch.tensor([x.end for x in intervals], dtype=torch.float32),
            "innovation_target": torch.tensor(innovation_target, dtype=torch.float32),
            "policy_impulse_target": torch.tensor(policy_impulse_target, dtype=torch.float32),
            "policy_recipe_id": torch.tensor(recipe_id, dtype=torch.long),
        }
```

---

## 4. 实验切入点

1. **数据集与 shift 构造**
   - MIMIC-IV / eICU / HiRID：先用 KBTA / rule-based temporal abstraction 生成 clinical interval streams。
   - P12 / P19：用变量分箱、趋势和持续异常规则构造轻量 interval abstraction。
   - 反事实采样策略：repeat dense、panel split、pending stub、low-rate read、boundary jitter、alarm-dense vs routine-round。

2. **对比方法**
   - 原始 INTERVenE-style interval Transformer。
   - RoMAE / continuous-position Transformer。
   - SLAN / switch non-imputation encoder。
   - TimeCHEAT-style local CD / global CI baseline。
   - 当前采样解耦框架中的 policy-adversarial、state/policy branch、counterfactual consistency 版本。

3. **关键指标**
   - In-policy AUROC / AUPRC。
   - Cross-policy AUROC / AUPRC drop。
   - Repeat-density sensitivity：同一状态重复复测次数增加时 logit 斜率。
   - Boundary-jitter work：区间边界抖动带来的 Lyapunov work。
   - Semantic-innovation recall：真实趋势反转 / severe duration 是否能触发 `nu_i`。
   - Policy-impulse work ratio：分类前状态移动中有多少来自 high-impulse intervals。

4. **消融**
   - 去掉 `L_counterfactual_contraction`：检验是否回到普通 interval Transformer shortcut。
   - 去掉 `L_policy_impulse_work`：检验 repeat / pending 是否重新推动状态。
   - 去掉 innovation supervision：检验模型是否过度阻尼真实病程突变。
   - 将 Lyapunov metric 替换为欧氏 hidden-state MSE：验证不是普通表示一致性带来的收益。
   - 允许 `pi_i` 进入分类头：检验 policy diagnostics 是否会变成 label shortcut。

---

## 5. 预期创新性

1. **从 interval token pooling 转向收缩观测器**：吸收 INTERVenE 的可解释区间抽象，但不把区间数量、边界和持续时间直接喂给分类器；区间必须通过 Lyapunov observer 才能改变病程状态。
2. **从表示一致性转向能量收缩纪律**：不是强迫多策略 logits / representation 一致，而是要求纯采样制度冲击在可学习 Lyapunov 能量下收缩；真实语义新息可以突破收缩。
3. **从采样政策分支转向 policy impulse work**：采样分支不负责分类，也不只是对抗去除，而是明确度量“采样制度对病程状态做了多少功”。
4. **能解释低频场景下的合理性能损失**：低采样率丢失细粒度形态时，新息门可以变保守，避免历史一致性方法的过度对齐。
5. **与现有框架低耦合**：只需把反事实采样输出改成 interval impulse bank，把 encoder 换成 Lyapunov observer，不依赖额外人工 proof、program IR、Blackwell order 或 proper scoring target。

## 6. 一句话投稿卖点

**DLIO 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样制度对病程观测器状态施加了多余冲击功”的问题，通过 INTERVenE 式临床区间抽象、policy impulse head 与反事实 Lyapunov 收缩损失，让模型在保留真实临床新息的同时，阻止复测频率、pending、panel split、低频读取和区间边界抖动在跨采样政策部署中累积成类别捷径。**
