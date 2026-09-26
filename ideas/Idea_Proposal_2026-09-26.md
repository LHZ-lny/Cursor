# Title: Do-SWING Delta Circuit Breaker：面向采样策略偏移的风险跃迁熔断分类器

## 0. 强制读取记录与思维黑名单

### 已读取与检索材料

- 已尝试读取 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已检索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化持久记忆 `MEMORIES.md`，并补读近期记忆条目 `idea_2026-09-23.md`、`idea_2026-09-24.md`、`idea_2026-09-25.md`。
- 已读取/抽取当前仓库 `ideas/Idea_Proposal_*.md` 的历史提案标题、黑名单、核心机制段与近期完整正文，覆盖当前工作区已落盘的 2026-06-12 至 2026-09-25 提案。
- 已读取近期 `paper_daily_2026-09-21.md`、`paper_daily_2026-09-22.md`、`paper_daily_2026-09-24.md`、`paper_daily_2026-09-25.md`，并从兼容入口 `paper_daily.md` 抽取近期前沿机制。重点纳入：
  - **Delta-XAI / SWING**：把在线时序解释对象从单点预测改为 `f(x_t)-f(x_{t-1})` 的预测变化，并用 shifted-window integrated gradients 追踪新增观测、移出窗口和延迟效应。
  - **ReDiTT**：异步事件流可在 latent event-token 空间中建模未来轨迹，但 retrieval reference 容易把采样政策邻居误当作病程邻居。
  - **DynaMamba**：选择性状态空间记忆适合长程临床 IMTS，但也可能把 mask、delta-t、panel 共测和复测政策长期写入 hidden state。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下主机制：

1. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
2. 生理流/采样算子交换子、state-policy graph 拆分、policy residual sink。
3. protocol tax、additive evidence market、token 边际审计、证据预算。
4. posterior quotient、Radon-Nikodym density ratio、doubly robust correction、RKHS cubature。
5. reconstruction error cartography、ANOVA projection、VQ clauses、HSIC redaction。
6. optional-stopping martingale、拓扑胶囊、policy gauge、syndrome code、knockoff calendar。
7. observability witness、evidential vacuity、policy lattice、solver trace front-door、conformal sleeve、IV/control-function。
8. Borda jury、Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock。
9. IRT-DIF、RG fixed point、bitemporal curtain、clinical tomography、matched risk-set likelihood、policy privacy cloak。
10. CauKer-style synthetic forge、dialectical JEPA referee、PID semantic prism、Noether semantic action、meta workflow immunization。
11. collider seal、source-atlas antipodal retrieval、Schwarzian warp canonicalizer、renewal exposure meter、palimpsest memory。
12. typed SSA observation program、Robin boundary field solver、Blackwell experiment order、elicitable functional compass。
13. Lyapunov interval observer、BBP / Marchenko-Pastur random-matrix spike sieve、RIP sparse pathology recovery。
14. factorial cumulant thinning algebra、repeat/panel cumulant subtraction、Poisson clutter null。
15. 单纯 state-policy 双分支、对抗环境分类器、跨视图 logits/representation 一致性、频域掩码对比学习、missingness pattern 直接分类、普通 retrieval-augmented classifier、普通 dual-SSM classifier。

本提案选择新的正交切入点：**不把采样政策建成概率、图、测量矩阵、谱、边界、程序、记忆或高阶计数；也不要求反事实视图输出一致。相反，它把在线分类器的每一次风险跃迁视为需要经过“熔断器”的控制信号：若该跃迁主要由新增真实病理值、趋势反转或临床语义新息触发，就被放行写入 state logit；若该跃迁主要由采样日历、pending、panel 打包、窗口滑出或重复复测触发，就被隔离到 policy interrupt buffer，不能继续累积成类别 margin。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

Delta-XAI 的关键启发是：临床在线监测中，医生更关心“风险为什么从 0.12 升到 0.47”，而不是只解释当前 0.47。它把解释目标从单点预测改成 `prediction change`，并用 SWING 避免普通 attribution 把 forward-fill、窗口重叠或延迟效应错归因到错误时间点。

这正好切中 sampling-policy shift 的盲区。跨医院部署时，分类器的绝对风险可能看起来稳定，但**风险跃迁的触发源**已经变了：

- 同一乳酸值在医院 A 是 routine round 的自然返回，在医院 B 是报警后密集复测，普通模型可能在 B 中把“刚被测到”当成风险跃迁原因。
- sliding window 中旧 token 移出、pending 值返回、panel 同步拆分，都会造成 logit 突变；这些突变可能是窗口和流程伪影，而不是病理改变。
- DynaMamba/SSM 的选择性写入可能把这些 logit 突变长期保存在 hidden state 中，使短期采样伪影变成长期类别证据。
- ReDiTT 式 event-token 轨迹建模说明未来事件结构有强先验，但若相似轨迹由 observation policy 决定，风险变化会被 policy-neighbor 放大。

**Do-SWING Delta Circuit Breaker (DSDCB)** 的核心直觉是：

> Sampling-policy shift 最危险的形态不是一次性错误预测，而是 policy-only 事件反复触发微小风险跃迁，并在在线模型中被累积、保留、解释成病理恶化。鲁棒分类器需要对每个风险跃迁做熔断审计：病理跃迁可以放行，采样跃迁必须进入中断缓冲区，不能直接写入分类 margin。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 产生 **state-confirmed delta**：真实数值变化、趋势反转、临床阈值跨越、跨变量病理关系。
- sampling process 产生 **policy interrupt delta**：时间戳改变、变量刚被测、pending 返回、panel 打包/拆分、窗口滑出、重复复测。
- counterfactual intervention 不再要求多视图 logits 一致，也不估计采样概率；它构造 **risk-jump audit pairs**，用于训练熔断器区分“该风险跳变是否应被放行”。
- classifier 不是直接读所有 event hidden state，而是只读经过熔断器放行的累计 state logit / state memory；policy interrupt buffer 只用于告警、拒识和解释。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Risk-Jump Audit Batch

新增 `DeltaCircuitBreakerCollator`，每个样本不再只返回一整条序列，而是返回在线相邻窗口对：

```text
prev_window  = x_{t-1-lookback:t-1}
curr_window  = x_{t-lookback:t}
delta_event  = curr_window - prev_window
label        = y_t
```

同时返回一组由现有反事实采样模块生成的 `risk_jump_recipe_bank`：

1. `value_reveal`
   - pending 值返回，并且数值相对历史/病理分箱有真实新息。
   - 应允许触发 state delta。

2. `policy_reveal_only`
   - pending 状态改变或变量“刚出现”，但值与最近有效值一致或无临床新息。
   - 应被熔断到 policy interrupt buffer。

3. `window_exit`
   - 旧窗口 token 移出导致 logit 改变，但当前没有新测量。
   - 不应直接改变 state margin。

4. `repeat_echo`
   - 短间隔重复复测，值几乎不变。
   - 不应通过多次出现累积风险。

5. `panel_pack_split`
   - 同一组变量同步返回或异步拆分。
   - 只允许改变解释中的 workflow trace，不允许凭共现本身推高类别 margin。

6. `semantic_jump`
   - 插入真实趋势反转、危险阈值跨越或跨变量生理不一致。
   - 应被放行，并在解释中显示为 state-confirmed risk jump。

每个窗口对额外返回弱监督：

- `value_surprise_score`
- `trend_flip_flag`
- `threshold_cross_flag`
- `pending_flag`
- `panel_membership`
- `window_exit_mask`
- `policy_only_recipe_id`

这些字段只训练 delta router，不进入最终分类头。

### 2.2 改 Encoder：Prediction-Difference Wrapper + Delta Circuit Breaker

#### A. Backbone Online Classifier

可以使用现有 irregular encoder，例如 Transformer、SLiCE、Mamba 或事件 token backbone：

```text
h_prev, raw_logits_prev = Backbone(prev_window)
h_curr, raw_logits_curr = Backbone(curr_window)
raw_delta_logits        = raw_logits_curr - raw_logits_prev
```

DSDCB 不把 `raw_logits_curr` 直接当最终预测，而是把 `raw_delta_logits` 交给熔断器审计。

#### B. SWING-Style Delta Attribution Surrogate

借鉴 Delta-XAI/SWING，但不把它作为事后解释工具，而是作为训练期的 delta routing 信号。对相邻窗口构造 shifted interpolation：

```text
h_alpha = align(prev_window, curr_window, alpha)
delta_attr_i = integral grad_f(h_alpha)_i * (h_curr_i - h_prev_i) d alpha
```

实现中可用少量插值步近似，或训练一个 `DeltaAttributionSurrogate` 直接预测每个 token / feature 对 `raw_delta_logits` 的贡献。

#### C. State Gate 与 Policy Interrupt Gate

对每个 delta token 产生两个门：

```text
state_gate_i  = sigmoid(StateGate(delta_token_i, swing_attr_i, value_surprise_i))
policy_gate_i = sigmoid(PolicyGate(delta_token_i, swing_attr_i, pending_i, panel_i, window_exit_i))
```

二者不是简单互补：一个事件可能既有真实病理值，也伴随 policy interrupt。熔断器要求它们的贡献被分账：

```text
state_delta_logits  = sum_i state_gate_i  * delta_logit_i
policy_delta_logits = sum_i policy_gate_i * delta_logit_i
```

#### D. Circuit-Broken State Logit

在线状态只累计被放行的 state delta：

```text
safe_logits_t = safe_logits_{t-1} + state_delta_logits
policy_buffer_t = decay * policy_buffer_{t-1} + policy_delta_logits
```

最终分类：

```text
logits = safe_logits_t
```

`policy_buffer_t` 不进入类别 logits，只输出：

- `interrupt_score`
- `policy_trip_reason`
- `delta_explanation_report`
- `manual_review / extra_measurement` 建议

这与历史机制的区别：

- 不是 evidence market：没有 token 价格、协议税或边际拍卖。
- 不是 regret escrow：不是决定是否释放 detail，而是对在线 risk delta 做即时熔断。
- 不是 Lyapunov contraction：不学习能量函数，也不要求反事实轨迹收缩。
- 不是 palimpsest/renewal：不改变记忆写入权或 reward/exposure 分母。
- 不是普通 Delta-XAI consistency：解释不是后处理指标，而是分类路径中的风险跃迁断路器。

### 2.3 改 Loss：从静态鲁棒性转向 Risk-Jump Circuit Discipline

总目标：

```text
L = L_cls
  + lambda_cmp  * L_delta_completeness
  + lambda_trip * L_policy_trip
  + lambda_rel  * L_state_release
  + lambda_buf  * L_buffer_noncompounding
  + lambda_swg  * L_swing_polarity
```

#### A. Safe Classification `L_cls`

只用熔断后的 `safe_logits_t` 分类：

```text
L_cls = CE(safe_logits_t, y_t)
```

如果训练初期担心熔断过强，可用 warm-up：

```text
logits_train = raw_logits_curr.detach() * beta + safe_logits_t * (1 - beta)
```

但最终模型部署时只读 `safe_logits_t`。

#### B. Delta Completeness `L_delta_completeness`

熔断器不能随意丢掉风险变化；它必须把原始 risk jump 分配到 state account 或 policy buffer：

```text
L_delta_completeness =
  || raw_delta_logits - state_delta_logits - policy_delta_logits ||_1
```

这保证 policy-only 变化不会消失，而是被显式报告为 interrupt。

#### C. Policy Trip Loss `L_policy_trip`

对 `policy_reveal_only / window_exit / repeat_echo / panel_pack_split` 等 policy-only recipe：

```text
L_policy_trip =
  || state_delta_logits_policy_only ||_2^2
  + relu(policy_trip_margin - || policy_delta_logits_policy_only ||_2)^2
```

直觉：如果风险变化只是由采样流程触发，state delta 应被熔断，policy buffer 应接住这次变化。

#### D. State Release Loss `L_state_release`

对 `value_reveal / semantic_jump` recipe，若存在真实数值新息：

```text
release_target = sigmoid(
    a * value_surprise
  + b * threshold_cross
  + c * trend_flip
)

L_state_release =
  BCE(mean(state_gate), release_target)
  + CE(safe_logits_after_release, y_t)
```

这样避免模型把所有风险跃迁都过度熔断。真正的急性恶化必须被及时放行。

#### E. Buffer Non-Compounding `L_buffer_noncompounding`

policy interrupt buffer 可以解释“为什么 raw model 会跳”，但不能长期变成隐形分类器：

```text
L_buffer_noncompounding =
  relu(corr(policy_buffer_norm, stopgrad(true_class_margin)) - eps)^2
  + || ClassProbe(stopgrad(policy_buffer_t)) ||_confident^2
```

这里不是 adversarial environment classifier；主模型不需要欺骗 probe。目标只是防止 policy buffer 与标签 margin 单调绑定。

#### F. SWING Polarity Loss `L_swing_polarity`

用 SWING attribution 的弱标签约束 delta 归因极性：

```text
state_attr_target  = normalize(value_surprise + threshold_cross + trend_flip)
policy_attr_target = normalize(pending + panel_membership + window_exit + repeat_echo)

L_swing_polarity =
  KL(normalize(abs(state_gate * swing_attr)),  state_attr_target)
  + KL(normalize(abs(policy_gate * swing_attr)), policy_attr_target)
```

这不是要求不同采样视图解释一致，而是要求每一次 risk jump 的 attribution polarity 与其触发源一致。

### 2.4 推理阶段

在线部署时：

1. 对每个新窗口计算 raw prediction change。
2. 用 Delta Circuit Breaker 判断本次 risk jump 是 state-confirmed 还是 policy-interrupted。
3. 只把 state delta 写入 `safe_logits`。
4. 若 `policy_buffer_norm` 或 `interrupt_score` 过高，输出偏移告警：
   - “风险上升主要来自 pending 返回 / panel 打包 / 窗口滑出，而不是新病理值。”
   - “建议等待真实值确认或追加某变量测量。”
5. 报告 `state_release_map` 与 `policy_trip_map`，作为 Delta-XAI 风格的 prediction-change 解释。

---

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


class DeltaTokenStem(nn.Module):
    """Builds token-level features for a prediction-change routing step."""

    def __init__(self, hidden_dim: int, meta_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3 + meta_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        h_prev: torch.Tensor,
        h_curr: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        delta_h = h_curr - h_prev
        x = torch.cat([h_prev, h_curr, delta_h, meta], dim=-1)
        return self.net(x)


class SwingAttributionSurrogate(nn.Module):
    """
    Lightweight differentiable surrogate for SWING-style delta attribution.
    It predicts token-class contributions to raw logit changes.
    """

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, delta_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attr = self.head(delta_tokens)
        return attr * mask.unsqueeze(-1).to(attr.dtype)


class DeltaCircuitBreaker(nn.Module):
    """
    Routes raw prediction changes into state-confirmed deltas and policy interrupts.
    Only state-confirmed deltas are allowed to update safe logits.
    """

    def __init__(self, hidden_dim: int, meta_dim: int, num_classes: int):
        super().__init__()
        self.delta_stem = DeltaTokenStem(hidden_dim, meta_dim)
        self.swing = SwingAttributionSurrogate(hidden_dim, num_classes)

        gate_in = hidden_dim + num_classes + meta_dim
        self.state_gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.policy_gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.delta_logit_head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        h_prev: torch.Tensor,
        h_curr: torch.Tensor,
        delta_meta: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta_tokens = self.delta_stem(h_prev, h_curr, delta_meta)
        swing_attr = self.swing(delta_tokens, mask)
        per_token_delta = self.delta_logit_head(delta_tokens)

        gate_features = torch.cat([delta_tokens, swing_attr.detach(), delta_meta], dim=-1)
        state_gate = torch.sigmoid(self.state_gate(gate_features)) * mask.unsqueeze(-1)
        policy_gate = torch.sigmoid(self.policy_gate(gate_features)) * mask.unsqueeze(-1)

        state_delta = (state_gate * per_token_delta).sum(dim=1)
        policy_delta = (policy_gate * per_token_delta).sum(dim=1)
        reconstructed_delta = state_delta + policy_delta

        return {
            "state_delta": state_delta,
            "policy_delta": policy_delta,
            "reconstructed_delta": reconstructed_delta,
            "per_token_delta": per_token_delta,
            "swing_attr": swing_attr,
            "state_gate": state_gate.squeeze(-1),
            "policy_gate": policy_gate.squeeze(-1),
            "delta_tokens": delta_tokens,
        }


class DoSwingDeltaCircuitBreaker(nn.Module):
    """
    Wrapper around an irregular time-series backbone.
    The backbone must return token states and logits for each online window.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        meta_dim: int,
        num_classes: int,
        buffer_decay: float = 0.8,
    ):
        super().__init__()
        self.backbone = backbone
        self.breaker = DeltaCircuitBreaker(hidden_dim, meta_dim, num_classes)
        self.num_classes = num_classes
        self.buffer_decay = buffer_decay

    def forward(
        self,
        prev_batch: dict[str, torch.Tensor],
        curr_batch: dict[str, torch.Tensor],
        delta_meta: torch.Tensor,
        token_mask: torch.Tensor,
        safe_logits_prev: torch.Tensor | None = None,
        policy_buffer_prev: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prev_out = self.backbone(prev_batch)
        curr_out = self.backbone(curr_batch)

        raw_delta_logits = curr_out["logits"] - prev_out["logits"]
        route = self.breaker(
            prev_out["tokens"],
            curr_out["tokens"],
            delta_meta,
            token_mask,
        )

        if safe_logits_prev is None:
            safe_logits_prev = prev_out["logits"].detach()
        if policy_buffer_prev is None:
            policy_buffer_prev = torch.zeros_like(raw_delta_logits)

        safe_logits = safe_logits_prev + route["state_delta"]
        policy_buffer = self.buffer_decay * policy_buffer_prev + route["policy_delta"]

        route.update(
            {
                "safe_logits": safe_logits,
                "raw_logits_curr": curr_out["logits"],
                "raw_delta_logits": raw_delta_logits,
                "policy_buffer": policy_buffer,
                "interrupt_score": policy_buffer.norm(dim=-1),
            }
        )
        return route


class DeltaCircuitLoss(nn.Module):
    def __init__(
        self,
        lambda_completeness: float = 1.0,
        lambda_policy_trip: float = 1.0,
        lambda_state_release: float = 1.0,
        lambda_buffer: float = 0.2,
        lambda_swing: float = 0.5,
        trip_margin: float = 0.1,
    ):
        super().__init__()
        self.lambda_completeness = lambda_completeness
        self.lambda_policy_trip = lambda_policy_trip
        self.lambda_state_release = lambda_state_release
        self.lambda_buffer = lambda_buffer
        self.lambda_swing = lambda_swing
        self.trip_margin = trip_margin

    def forward(
        self,
        out: dict[str, torch.Tensor],
        y: torch.Tensor,
        policy_only_mask: torch.Tensor,
        release_target: torch.Tensor,
        state_attr_target: torch.Tensor,
        policy_attr_target: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_cls = F.cross_entropy(out["safe_logits"], y)

        loss_completeness = F.smooth_l1_loss(
            out["reconstructed_delta"],
            out["raw_delta_logits"].detach(),
        )

        policy_only = policy_only_mask.to(out["state_delta"].dtype).unsqueeze(-1)
        state_norm_policy = (out["state_delta"].pow(2).sum(dim=-1, keepdim=True) * policy_only).mean()
        policy_norm = out["policy_delta"].norm(dim=-1, keepdim=True)
        policy_trip = (F.relu(self.trip_margin - policy_norm).pow(2) * policy_only).mean()
        loss_policy_trip = state_norm_policy + policy_trip

        mean_state_gate = masked_mean(out["state_gate"], token_mask, dim=1)
        loss_state_release = F.binary_cross_entropy(
            mean_state_gate.clamp(1e-5, 1 - 1e-5),
            release_target.to(mean_state_gate.dtype),
        )

        true_margin_proxy = out["safe_logits"].max(dim=-1).values.detach()
        buffer_norm = out["policy_buffer"].norm(dim=-1)
        centered_buffer = buffer_norm - buffer_norm.mean()
        centered_margin = true_margin_proxy - true_margin_proxy.mean()
        corr = (centered_buffer * centered_margin).mean()
        corr = corr / (centered_buffer.std().clamp_min(1e-4) * centered_margin.std().clamp_min(1e-4))
        loss_buffer = F.relu(corr.abs() - 0.05).pow(2)

        swing_abs = out["swing_attr"].abs().mean(dim=-1)
        state_dist = (out["state_gate"] * swing_abs * token_mask).clamp_min(0)
        policy_dist = (out["policy_gate"] * swing_abs * token_mask).clamp_min(0)
        state_dist = state_dist / state_dist.sum(dim=1, keepdim=True).clamp_min(1e-6)
        policy_dist = policy_dist / policy_dist.sum(dim=1, keepdim=True).clamp_min(1e-6)

        state_target = state_attr_target / state_attr_target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        policy_target = policy_attr_target / policy_attr_target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        loss_swing = (
            F.kl_div((state_dist + 1e-6).log(), state_target, reduction="batchmean")
            + F.kl_div((policy_dist + 1e-6).log(), policy_target, reduction="batchmean")
        )

        total = (
            loss_cls
            + self.lambda_completeness * loss_completeness
            + self.lambda_policy_trip * loss_policy_trip
            + self.lambda_state_release * loss_state_release
            + self.lambda_buffer * loss_buffer
            + self.lambda_swing * loss_swing
        )
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_completeness": loss_completeness,
            "loss_policy_trip": loss_policy_trip,
            "loss_state_release": loss_state_release,
            "loss_buffer": loss_buffer,
            "loss_swing": loss_swing,
        }
```

---

## 4. 实验切入点

1. **在线风险变化审计**
   - 数据：PhysioNet 2019 sepsis、MIMIC-III decompensation / mortality、P12。
   - 指标：AUROC/AUPRC 之外，报告 `policy_trip_rate`、`state_release_precision`、`risk_jump_false_alarm_rate`。

2. **反事实采样偏移**
   - 构造 pending、panel split、repeat echo、window exit、alarm dense 等 policy-only edits。
   - 检查 raw backbone 与 DSDCB 的风险跳变幅度差异。

3. **Delta-XAI 解释稳定性**
   - 对相同病程在不同采样政策下比较 prediction-change attribution。
   - 目标不是所有 attribution 一致，而是 policy attribution 被报告为 interrupt，state attribution 仍对应真实病理新息。

4. **SSM / Mamba hidden policy retention**
   - 用 DynaMamba 或 Mamba backbone 做 ablation。
   - 对比不加熔断时 hidden state 中 policy-only delta 的保留程度，以及 DSDCB 是否降低 policy-only AUPRC。

5. **临床可读诊断**
   - 输出每次风险跃迁的 `released` vs `tripped` 标签。
   - 人工检查高风险变化是否由真实值、趋势或阈值跨越触发，而非窗口滑出 / pending / panel。

---

## 5. 预期创新性

- **从静态风险鲁棒性转向动态风险跃迁鲁棒性**：历史方法多约束最终 logits、表示、测量矩阵、谱、边界或 cumulant；DSDCB 直接约束在线 risk jump 的触发源。
- **把 Delta-XAI 从事后解释升级为训练期熔断机制**：SWING 不只是解释为什么模型变了，而是决定这次变化是否允许写入可累积分类状态。
- **允许 policy delta 存在但禁止它复利**：不删除采样政策、不做 adversarial、不要求反事实 logits 一致；policy-only 风险变化被保留为 interrupt report，而不是成为类别 margin。
- **特别适配在线临床监测**：风险突然上升时，模型能区分“真实病理恶化”与“观测流程更新”，更符合 ICU/急诊中对报警可信度的需求。

## 6. 一句话投稿卖点

**Do-SWING Delta Circuit Breaker 将 Delta-XAI 的 prediction-change 解释变成分类器内部的风险跃迁熔断器：采样政策可以触发可审计的 interrupt，但不能把 policy-only 风险跳变复利成跨医院失效的临床分类证据。**
