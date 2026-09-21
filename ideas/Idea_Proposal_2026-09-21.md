# Title: Do-Elicitome Functional Compass：面向采样策略偏移的可征询病理泛函罗盘

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化持久记忆 `MEMORIES.md`，并纳入其中记录但当前工作区未完全落盘的历史提案摘要。
- 已读取/抽取当前仓库 `ideas/Idea_Proposal_*.md` 的历史提案标题、黑名单、核心机制与最近多份完整正文，覆盖 2026-06-12 至 2026-09-20 的已落盘提案。
- 已读取近期 `paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`，并从 `paper_daily.md` 抽取最新论文索引。重点纳入：
  - **Continuum Dropout**：连续时间内部随机正则化提醒我们区分模型内部不确定性与外部采样密度。
  - **RoMAE**：连续坐标 / axial RoPE 让通用 Transformer 处理不规则时间与通道坐标，但位置表达力越强越需要防止采样位置泄漏。
  - **SLAN**：switch layer 只在变量真实观测到时更新局部状态，但 switch 激活本身仍可能是医院/设备政策。
  - **CHARM**：通道语义描述有助于异构传感器迁移，但通道描述、单位和设备配置也可能携带部署政策。
  - **ECG latent ODE**：低采样率下连续形态可恢复性影响类别级鲁棒性，少数类尤其容易受采样率变化伤害。
  - **TimeCHEAT**：局部 channel-dependent、全局 channel-independent 的通道策略说明跨通道借用信息应有尺度边界，但局部共测边可能混入 policy shortcut。

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

本提案选择新的正交切入点：**不把采样政策估成概率、不做实验序关系、不求解边界场、不编译观测程序、不维护记忆板、不做多视图 logits 一致；而是把分类所需的病理证据声明为一组可征询的语义泛函（elicitable pathology functionals）。每个泛函都有严格适当评分规则和 identification function。采样政策可以改变某个泛函的可征询精度与不确定宽度，但不能让不可征询的 mask / switch / panel / cadence 本身绕过评分规则进入分类头。分类器只读取通过适当评分规则训练出的病理泛函罗盘。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列分类中的很多失败，并不是因为模型完全看不到病理状态，而是因为它把“病理泛函”和“观测政策痕迹”混成同一个 token 表征：

- ICU 中真正有用的可能是“乳酸在未来 6 小时的上分位趋势”“肌酐升高的持续时间”“SpO2 低于阈值的占空比”，而不是 lactate 被复测了几次；
- 可穿戴 ECG 中真正有用的是 QRS 宽度、RR irregularity、低频节律等形态/节律泛函；45 Hz 设备可能仍能征询节律泛函，却无法可靠征询少数类需要的细形态泛函；
- TimeCHEAT 的局部 CD / 全局 CI 提醒我们：有些泛函需要短窗跨通道协同，有些泛函应保持通道独立；但局部 panel 共测本身不能直接成为类别证据；
- CHARM 的通道语义适合定义“这个变量可用于征询哪个病理泛函”，而不是把通道描述直接拼进分类器；
- RoMAE 的连续坐标适合告诉泛函在什么时间与通道坐标上被评估，不能让强位置编码直接学习医院 routine cadence；
- SLAN 的 switch 思想适合说明“只有真实观测到的变量才能参与某个泛函的估计”，但 switch 是否打开不应绕过泛函估计直接决定标签。

历史方案常见目标是：删除采样信息、对齐多策略表征、惩罚 policy residue、或重写观测对象。本提案换一个统计学习视角：

> 若分类器需要使用某类病理证据，就必须先说明它在估计哪个可征询泛函，并用严格适当评分规则接受监督。采样政策只能影响这个泛函的可恢复性、置信宽度和 identification residual；任何不能通过评分规则变成病理泛函的观测痕迹，都不能进入分类头。

**Do-Elicitome Functional Compass (DEFC)** 将当前“采样解耦/反事实干预”框架改造成三层：

1. **Value process**：从事件值、时间、变量语义中估计一组病理泛函，如 quantile、expectile、threshold occupancy、robust slope、episode duration、cross-channel lag functional。
2. **Sampling process**：只输出每个泛函在当前采样政策下的 recoverability 与 observation support，不进入分类头。
3. **Counterfactual intervention**：改变采样视图后，不要求 logits 一致，而是检查同一个泛函是否仍满足自己的 identification equation；若低采样率确实无法征询某个细形态泛函，则允许不确定宽度上升，而不是强迫表示对齐。

这样解决 sampling-policy shift 的关键好处是：

- **避免 policy shortcut**：mask、switch、panel pack、cadence 只有在能改善某个病理泛函的适当评分时才有用；否则不会直接进入分类。
- **避免过度一致**：低采样率或稀疏策略可能真的让某些泛函不可征询，DEFC 允许宽度变大或泛函缺席，不强迫 logits 一样。
- **保留可解释性**：输出不是抽象 embedding，而是“哪些病理泛函被可靠征询、哪些因采样政策不可恢复、哪些泛函驱动了分类”。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Functional Elicitation Bank，而不是 consistency views

新增 `FunctionalElicitationCollator`。每个样本返回一组病理泛函规格，而不是只返回多视图增强：

```text
functional_spec_j = {
    semantic_scope: variables / channel descriptions,
    temporal_scope: window or continuous-time region,
    functional_type: quantile / expectile / threshold_occupancy / robust_slope / lag_coupling,
    resolution_requirement: minimum time density or morphology support,
    scoring_rule: pinball / asymmetric squared / Bernoulli proper score,
}
```

同时返回反事实采样干预产生的 support bank：

1. `factual_support`
   - 原始不规则事件流对每个泛函的实际支持。

2. `counterfactual_support_bank`
   - `low_rate_morphology_loss`：降低采样率，测试细形态泛函是否应变宽或失活；
   - `switch_sparse`：只保留真实观测 switch，测试未观测变量是否被伪造；
   - `local_cd_panel`：短窗口局部跨通道共测，测试跨通道 lag / occupancy 泛函；
   - `global_ci_channel`：按通道独立保留长期轨迹，测试变量个体趋势泛函；
   - `semantic_schema_swap`：变量名、单位、设备描述替换，测试 CHARM 式语义路由；
   - `routine_vs_alarm_cadence`：固定查房与报警密集复测，测试事件数是否被误当作泛函值。

3. `functional_target`
   - 来自可观测高质量窗口、离线临床分箱、同一患者高覆盖片段、或任务弱标签派生的泛函监督。
   - 不是重构所有原始值，也不是预测未来 observation process。

4. `recoverability_target`
   - 某个泛函在当前 support 下是否可征询，例如 45 Hz ECG 可征询 RR irregularity，但可能不可征询细粒度 QRS morphology。

关键区别：

- 这些 view 不是 contrastive positives；
- 不要求多 view logits / representation 一致；
- 不估计采样 hazard、density ratio、Blackwell order 或 posterior quotient；
- 不生成 program IR、boundary condition、memory slot、proof、conformal set 或 privacy noise；
- 反事实采样只用于审计“哪些病理泛函还可被适当评分规则征询”。

### 2.2 改 Encoder：Elicitome Functional Compass

给定事件：

```text
e_i = (value_i, time_i, variable_i, channel_description_i, unit_i, quality_i)
```

先做事件语义编码：

```text
h_i = EventStem(value_i, continuous_time_i, channel_semantic_i, unit_i, quality_i)
```

这里吸收近期机制但改变用途：

- RoMAE 式连续坐标只告诉泛函“在哪里被评估”，不作为分类 shortcut；
- CHARM 式通道描述只决定事件能服务哪些 semantic functional；
- SLAN 式 switch 只决定事件是否真实存在，不触发隐藏状态伪更新；
- TimeCHEAT 式 local-CD / global-CI 只决定泛函 scope，而不是把局部共测边直接分类。

#### A. Functional Query Router

每个泛函规格 `spec_j` 生成一个 query：

```text
q_j = FunctionalSpecEncoder(spec_j)
```

事件到泛函的注意力只由语义 scope、时间 scope 与 recoverability 共同决定：

```text
a_{j,i} = softmax(q_j^T W h_i + support_bias_{j,i})
u_j = sum_i a_{j,i} h_i
```

`u_j` 是第 `j` 个病理泛函的估计状态。分类器不读取原始 event count、mask density、center id、panel id 或 policy descriptor。

#### B. Proper Functional Heads

对不同泛函类型输出不同的可征询参数：

- `quantile`: 输出多个分位点 `Q_tau`;
- `expectile`: 输出 expectile `E_alpha`;
- `threshold_occupancy`: 输出阈值占空比概率 `P(x > c)`;
- `robust_slope`: 输出 Theil-Sen / Huber slope 的中心与宽度；
- `lag_coupling`: 输出短窗跨通道滞后泛函，但只在 local support 足够时激活。

每个输出都配套一个严格适当评分规则，而不是普通 MSE embedding loss。

#### C. Recoverability Gating

sampling branch 只输出：

```text
rho_j = P(functional_j is elicitable under current support)
w_j   = uncertainty width / scoring temperature
```

`rho_j, w_j` 不直接进入分类头；它们用于评分规则加权与诊断。最终分类器读取的是经过 recoverability 调制后的泛函中心值：

```text
z = Pool({ stop_policy(theta_j), rho_j })
logits = Classifier(z)
```

其中 `theta_j` 是由 proper functional head 输出的病理泛函参数；policy 支路不能绕过它直接贡献 logits。

### 2.3 改 Loss：从表示不变转向 Proper Elicitation Discipline

总目标：

```text
L = L_cls
  + lambda_prop * L_proper_scoring
  + lambda_id   * L_identification_residual
  + lambda_rec  * L_recoverability_order
  + lambda_do   * L_counterfactual_elicitation
  + lambda_gate * L_policy_nonelicitation
```

#### A. Functional Classification `L_cls`

只用病理泛函罗盘分类：

```text
logits = Classifier(Pool(theta_1, ..., theta_M))
L_cls = CE(logits, y)
```

这不是对所有事件 token 做 attention pooling；分类输入必须先通过 proper scoring head 形成可解释泛函。

#### B. Proper Scoring `L_proper_scoring`

不同泛函使用不同严格适当评分规则。

分位数用 pinball loss：

```text
L_quantile = mean_tau max(tau * (target - Q_tau),
                         (tau - 1) * (target - Q_tau))
```

expectile 用 asymmetric squared loss：

```text
L_expectile = |alpha - 1[target < E_alpha]| * (target - E_alpha)^2
```

阈值占空比用 Bernoulli proper score：

```text
L_occupancy = BCE(P(x > c), target_occupancy)
```

这一步强迫模型把观测值语义转化成可检验泛函，而不是自由学习 policy-entangled embedding。

#### C. Identification Residual `L_identification_residual`

每个 elicitable functional 都有 identification function `V(theta, target)`，正确时条件期望为 0。对 batch 内相同 functional type 的样本：

```text
L_id = || mean_b V(theta_b, target_b) ||^2
```

再检查 residual 不应被纯 policy descriptor 系统性解释：

```text
L_id_policy = || Corr(V(theta, target), policy_support_descriptor) ||_F^2
```

这不是 adversarial policy classifier，也不是 IRM；它只要求“泛函估计误差”不要按采样制度成系统偏移。

#### D. Recoverability Order `L_recoverability_order`

若反事实采样降低了某个泛函的分辨率支持，例如 ECG 细形态从 360 Hz 降到 45 Hz，则：

```text
width_low >= width_high
rho_low   <= rho_high
```

损失为：

```text
L_rec = relu(width_high - width_low + margin)^2
      + relu(rho_low - rho_high + margin)^2
```

如果 view 只是 semantic schema swap 或单位转换，recoverability 不应下降；如果 view 破坏局部形态或跨通道 lag 所需采样密度，recoverability 必须诚实下降。

#### E. Counterfactual Elicitation `L_counterfactual_elicitation`

对语义保持的反事实 rewrite，不比较 logits，而比较 identification equation 是否仍成立：

```text
L_do = mean_rewrite || mean_b V(theta_rewrite, target_same) ||^2
```

对信息破坏 rewrite，只要求评分宽度与 recoverability 反映损失，不强迫中心值或 logits 一致：

```text
L_do_damage = relu(score_confidence_damage - score_confidence_factual)^2
```

这样避免把真正不可恢复的信息硬对齐。

#### F. Policy Non-Elicitation `L_policy_nonelicitation`

防止模型用 policy support 直接预测标签而不经过泛函：

```text
logits_policy_only = PolicyProbe(stopgrad(policy_support_descriptor))
L_policy_probe = - CE(logits_policy_only, y)
```

实现时不把该 probe 反传到 classifier；它作为训练诊断或早停指标。主损失使用更直接的门控约束：

```text
L_gate = mean_j (1 - rho_j) * ||theta_j - stopgrad(theta_j_neutral)||^2
```

当某泛函不可征询时，它的中心估计被拉回中性值，不能靠 policy trace 编造高置信类别证据。

### 2.4 推理阶段

给定测试样本：

1. 事件编码器处理原始不规则事件；
2. functional query router 为每个病理泛函收集支持；
3. proper heads 输出 `theta_j`、`rho_j` 与 `width_j`；
4. classifier 只读取可征询泛函罗盘；
5. 同时报告：
   - 每个类别依赖的 top functional；
   - 哪些泛函因低采样率 / switch sparse / schema mismatch 不可恢复；
   - policy residual 是否系统性影响 identification residual；
   - local-CD 泛函与 global-CI 泛函的贡献比例。

---

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=dim)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    # pred: [B, M, T], target: [B, M, 1], tau: [T]
    err = target - pred
    tau = tau.view(*([1] * (pred.dim() - 1)), -1)
    return torch.maximum(tau * err, (tau - 1.0) * err).mean()


def expectile_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    err = target - pred
    weight = torch.where(err < 0, 1.0 - alpha, alpha)
    return (weight * err.square()).mean()


def corr_penalty(x: torch.Tensor, z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [B, M] residual, z: [B, P] policy descriptors.
    x = x - x.mean(dim=0, keepdim=True)
    z = z - z.mean(dim=0, keepdim=True)
    x = x / x.square().mean(dim=0, keepdim=True).add(eps).sqrt()
    z = z / z.square().mean(dim=0, keepdim=True).add(eps).sqrt()
    corr = torch.einsum("bm,bp->mp", x, z) / x.size(0)
    return corr.square().mean()


class EventStem(nn.Module):
    def __init__(self, value_dim: int, var_count: int, unit_count: int, hidden_dim: int):
        super().__init__()
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.var_emb = nn.Embedding(var_count, hidden_dim)
        self.unit_emb = nn.Embedding(unit_count, hidden_dim)
        self.quality_proj = nn.Linear(1, hidden_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        value: torch.Tensor,
        time_coord: torch.Tensor,
        var_id: torch.Tensor,
        unit_id: torch.Tensor,
        quality: torch.Tensor,
    ) -> torch.Tensor:
        # Continuous coordinates are represented with a small Fourier basis.
        time_feat = torch.stack(
            [
                time_coord,
                torch.sin(time_coord),
                torch.cos(time_coord),
                torch.log1p(time_coord.clamp_min(0.0)),
            ],
            dim=-1,
        )
        h = (
            self.value_proj(value)
            + self.time_proj(time_feat)
            + self.var_emb(var_id)
            + self.unit_emb(unit_id)
            + self.quality_proj(quality.unsqueeze(-1))
        )
        return self.out(h)


class FunctionalSpecEncoder(nn.Module):
    def __init__(self, num_functionals: int, num_types: int, hidden_dim: int):
        super().__init__()
        self.func_emb = nn.Embedding(num_functionals, hidden_dim)
        self.type_emb = nn.Embedding(num_types, hidden_dim)
        self.scope_proj = nn.Linear(6, hidden_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        func_id: torch.Tensor,
        func_type: torch.Tensor,
        scope_features: torch.Tensor,
    ) -> torch.Tensor:
        q = self.func_emb(func_id) + self.type_emb(func_type) + self.scope_proj(scope_features)
        return self.out(q)


class ElicitomeFunctionalCompass(nn.Module):
    def __init__(
        self,
        *,
        value_dim: int,
        var_count: int,
        unit_count: int,
        num_functionals: int,
        num_functional_types: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_quantiles: int = 5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles
        self.event_stem = EventStem(value_dim, var_count, unit_count, hidden_dim)
        self.spec_encoder = FunctionalSpecEncoder(
            num_functionals=num_functionals,
            num_types=num_functional_types,
            hidden_dim=hidden_dim,
        )
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.support_bias = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.quantile_head = nn.Linear(hidden_dim, num_quantiles)
        self.expectile_head = nn.Linear(hidden_dim, 1)
        self.occupancy_head = nn.Linear(hidden_dim, 1)
        self.center_head = nn.Linear(hidden_dim, 1)
        self.width_head = nn.Linear(hidden_dim, 1)
        self.recoverability_head = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_functionals * 3),
            nn.Linear(num_functionals * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        tau = torch.linspace(0.1, 0.9, num_quantiles)
        self.register_buffer("tau", tau)

    def route_functionals(
        self,
        event_h: torch.Tensor,
        event_mask: torch.Tensor,
        func_query: torch.Tensor,
        support_features: torch.Tensor,
    ) -> torch.Tensor:
        # event_h: [B, N, H], func_query: [B, M, H], support_features: [B, M, N, 4]
        key = self.key(event_h)
        val = self.value(event_h)
        logits = torch.einsum("bmh,bnh->bmn", func_query, key) / (self.hidden_dim ** 0.5)
        logits = logits + self.support_bias(support_features).squeeze(-1)
        support_mask = event_mask[:, None, :].expand_as(logits)
        attn = masked_softmax(logits, support_mask, dim=-1)
        return torch.einsum("bmn,bnh->bmh", attn, val)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        event_h = self.event_stem(
            value=batch["event_value"],
            time_coord=batch["event_time"],
            var_id=batch["event_var_id"],
            unit_id=batch["unit_id"],
            quality=batch["quality"],
        )
        func_query = self.spec_encoder(
            func_id=batch["functional_id"],
            func_type=batch["functional_type"],
            scope_features=batch["scope_features"],
        )
        u = self.route_functionals(
            event_h=event_h,
            event_mask=batch["event_mask"],
            func_query=func_query,
            support_features=batch["support_features"],
        )

        quantiles = self.quantile_head(u)
        expectile = self.expectile_head(u).squeeze(-1)
        occupancy_logit = self.occupancy_head(u).squeeze(-1)
        center = self.center_head(u).squeeze(-1)
        width = F.softplus(self.width_head(u).squeeze(-1)) + 1e-4
        recoverability = torch.sigmoid(self.recoverability_head(u).squeeze(-1))

        # Classification uses elicited functional values, not raw policy descriptors.
        compass = torch.stack([center, width.detach(), recoverability.detach()], dim=-1)
        logits = self.classifier(compass.flatten(start_dim=1))
        return {
            "logits": logits,
            "quantiles": quantiles,
            "expectile": expectile,
            "occupancy_logit": occupancy_logit,
            "center": center,
            "width": width,
            "recoverability": recoverability,
        }


class DEFLoss(nn.Module):
    def __init__(
        self,
        lambda_prop: float = 1.0,
        lambda_id: float = 0.2,
        lambda_rec: float = 0.2,
        lambda_gate: float = 0.1,
        expectile_alpha: float = 0.7,
    ):
        super().__init__()
        self.lambda_prop = lambda_prop
        self.lambda_id = lambda_id
        self.lambda_rec = lambda_rec
        self.lambda_gate = lambda_gate
        self.expectile_alpha = expectile_alpha

    def forward(self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        y = batch["label"]
        cls = F.cross_entropy(out["logits"], y)

        target = batch["functional_target"].unsqueeze(-1)
        quant = pinball_loss(out["quantiles"], target, batch["tau"])
        exp = expectile_loss(
            out["expectile"],
            batch["expectile_target"],
            alpha=self.expectile_alpha,
        )
        occ = F.binary_cross_entropy_with_logits(
            out["occupancy_logit"],
            batch["occupancy_target"],
            reduction="mean",
        )
        proper = quant + exp + occ

        # Identification residual for quantile median: tau - 1[target <= q_tau].
        mid = out["quantiles"].size(-1) // 2
        median = out["quantiles"][..., mid]
        tau_mid = batch["tau"][mid]
        id_residual = tau_mid - (batch["functional_target"] <= median).to(median.dtype)
        id_loss = id_residual.mean(dim=0).square().mean()
        id_policy = corr_penalty(id_residual, batch["policy_support_descriptor"])

        # Counterfactual recoverability order is supplied as pair indices.
        high_idx = batch["recoverability_high_idx"]
        low_idx = batch["recoverability_low_idx"]
        width_high = out["width"].gather(1, high_idx)
        width_low = out["width"].gather(1, low_idx)
        rho_high = out["recoverability"].gather(1, high_idx)
        rho_low = out["recoverability"].gather(1, low_idx)
        rec = F.relu(width_high - width_low + batch["recoverability_margin"]).square().mean()
        rec = rec + F.relu(rho_low - rho_high + batch["recoverability_margin"]).square().mean()

        neutral = batch["functional_neutral"]
        gate = ((1.0 - out["recoverability"]) * (out["center"] - neutral).square()).mean()

        loss = (
            cls
            + self.lambda_prop * proper
            + self.lambda_id * (id_loss + id_policy)
            + self.lambda_rec * rec
            + self.lambda_gate * gate
        )
        return {
            "loss": loss,
            "cls": cls.detach(),
            "proper": proper.detach(),
            "id": id_loss.detach(),
            "id_policy": id_policy.detach(),
            "recoverability": rec.detach(),
            "gate": gate.detach(),
        }
```

---

## 4. 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `EventStem + Functional Query Router`：事件先服务于具体病理泛函，而不是直接进入 pooled representation。
- 现有 sampling branch 改为 `Functional Support Estimator`：只预测每个泛函的 support、recoverability 与 scoring temperature。
- 现有 counterfactual intervention 改为 `Functional Support Rewrite Bank`：生成 low-rate、switch-sparse、local-CD、global-CI、schema-swap、routine/alarm cadence 等采样支持改写。
- 推理阶段只需要事实观测：输出可征询泛函罗盘、分类概率和不可恢复泛函告警。
- 诊断报告中可以直接回答：
  - 当前预测依赖哪些病理泛函？
  - 哪些泛函在部署采样政策下不可征询？
  - identification residual 是否随医院/设备 policy support 系统性偏移？
  - local-CD 泛函是否真的提供病理值，还是只反映 panel 共测？

---

## 5. 实验切入点

1. **Policy-shift split**
   - 按医院、采样密度、变量联测强度、routine/alarm cadence、设备采样率构造 train/test shift。

2. **Functional-only ablation**
   - 与普通 event pooling、mask/delta-t concatenation、switch-only classifier、policy-only classifier 比较，证明 DEFC 的分类收益来自病理泛函而非采样痕迹。

3. **Recoverability stress test**
   - 对 ECG / ICU 关键变量做 360 Hz -> 90 Hz -> 45 Hz 或 dense -> sparse 降采样，检查细形态泛函的 `rho` 是否下降、width 是否上升、少数类错误是否被提前告警。

4. **Schema and unit transfer**
   - 做 CHARM 式变量描述、单位和设备名称替换，检查同一泛函是否保持 identification residual 稳定。

5. **Local-CD vs Global-CI audit**
   - 用 TimeCHEAT 风格的短窗跨通道支持与长程单通道支持，分解分类贡献，识别哪些局部共测关系是病理互补，哪些只是 policy shortcut。

---

## 6. 预期创新性

- **机制新颖**：把采样偏移鲁棒分类转化为“可征询病理泛函 + 严格适当评分规则”的问题，不复用历史的实验序、边界场、SSA、记忆板、renewal、proof、privacy、density ratio 或多视图一致性路线。
- **理论抓手清晰**：elicitable functional 与 identification function 提供可检验统计条件，比泛泛的 representation invariance 更容易写成审稿人能理解的定理或诊断。
- **贴合近期 paper**：RoMAE、SLAN、CHARM、ECG latent ODE、TimeCHEAT 的机制都被吸收为“泛函可征询条件”，而不是直接照搬为主模型。
- **临床解释友好**：输出是 quantile、occupancy、slope、duration、lag-coupling 等语义泛函，天然适合解释“为什么采样政策变了以后这个预测还可信/不可信”。
- **避免过度鲁棒**：低信息采样下不强迫 logits 一致；模型可以诚实地说某个泛函不可征询，从而把信息缺失和 policy shortcut 区分开。

## 7. 一句话投稿卖点

**DEFC 把非规则采样分类从“学习一个尽量不受采样影响的黑箱表示”改写为“只允许经过严格适当评分规则征询出的病理泛函进入分类器”，从统计泛函层面切断采样政策 shortcut，同时保留低采样率和异构通道下真实信息缺失的不确定性。**
