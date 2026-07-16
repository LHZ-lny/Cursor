# Title: Measurement-Action Bisimulation Lens：面向采样策略偏移的测量动作双模拟透镜

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md`：其中记录了多轮自动化任务均未发现 `my_work_summary.md`，并列出了历史提案机制黑名单。
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-07-13.md`
- 已读取当前工作区内全部可见历史提案：
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
- 已尝试读取 `ideas/Idea_Proposal_2026-07-15.md`：当前工作区未检出该文件；其核心机制已从自动化记忆中纳入黑名单。
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：`2026-06-17`、`2026-06-20`、`2026-06-24`、`2026-06-27`、`2026-07-15`。

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
27. observation-set policy lattice、meet/join visibility masks、信息偏序单调性、次模边际契约、quality-order loss、shortcut curvature penalty。
28. solver trace / NFE / step-size / local-error / roughness 中介、front-door trace standardization、trace do-slope penalty。
29. RKHS cubature weights、kernel moment exactness、ridge KKT cubature solver、control-variate cubature constraints、jackknife leverage diagnostics。
30. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing、MedMamba 的 frequency/adaptive graph branch、MedSpaformer 的普通 token sparsification、MILM 的 value-redacted classifier、StarEmbed 的直接 foundation embedding、LLMTS 的普通 LLM alignment 或 MVC-CDE 的普通多视图平滑路径作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不删除采样信息，不做策略对抗、一致性、后验除法、随机平滑、测度校正、停时鞅、拓扑审查、gauge 投影、纠错码、knockoff、evidential uncertainty、信息格边际、solver trace 或 RKHS cubature；而是把非规则采样看成一个主动测量系统中的 action sequence。模型不根据事实采样日历直接分类，而是学习“如果我在一组标准化测量动作下继续追问，会看到什么”的 counterfactual measurement-response signature；分类器只消费这组标准化动作响应，因此从事实 policy shortcut 转向 action-bisimulation-stable state evidence。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily_2026-07-13.md` 中的 **Efficient Neural CDE via Attentive Kernel Smoothing** 说明，不规则观测进入连续时间模型前，需要先被转成某种可查询、可求解的 control path；路径构造方式会显著影响分类和效率。`paper_daily.md` 中的 **MILM** 又提示，采样 pattern 本身可以强预测标签；**Rethinking LLMs for Irregular Time Series Classification in Critical Care** 则强调，真正关键的是前端 encoder 是否尊重时间戳、缺失和异步观测，而不是后端语义模块有多复杂。

这些论文共同暴露了一个仍未被历史提案覆盖的视角：

> 采样策略不是一个静态 mask，也不只是一个概率 nuisance；它是一串 measurement actions。训练医院、设备或巡天系统不断在问：“此刻测哪个变量？多久后复测？用什么质量/成本测？是否把多个变量作为 panel 一起测？”

如果模型直接基于事实观测序列分类，它看到的是：

```text
latent state -> observer chooses measurement actions -> observed values -> classifier
```

一旦 observer 的 action policy 改变，分类器就可能把“被问了什么问题”误当成“状态是什么”。例如：

- 医院 A 只在高危时测 lactate，医院 B 把 lactate 变成常规项目；
- ICU 中某个化验值还未返回，但“已经下单”本身携带训练机构流程信息；
- 天文巡天 A 的 cadence 容易覆盖某个相位，巡天 B 的 cadence 覆盖另一个相位；
- MVC-CDE 的 kernel-smoothed path 在事实动作序列下表现良好，但换一套测量动作后，模型不知道哪些细节是状态，哪些是观察者提问方式。

**Measurement-Action Bisimulation Lens (MABL)** 的核心目标是：

> 把事实观测历史压缩成一个 latent belief，然后用一组与训练/测试采样政策无关的 canonical measurement actions 去“标准化追问”这个 belief。若两个观测历史在所有 canonical actions 下预测的潜在测量响应相同，则它们对分类器来说应被视为同一个可迁移状态；若响应不同，差异才有资格进入分类边界。

这类似强化学习中的 bisimulation，但这里的 action 不是治疗或控制动作，而是测量动作：

```text
a = (target variable, time offset, quality/cost level, panel flag)
response = p(value if do(measure a) | history)
```

最终分类器不消费事实 mask、事实采样频率、事实 panel 共现或 policy id，而消费：

```text
Lens(history) = {p(x under do(measure a_j) | history)}_{a_j in canonical action battery}
```

因此，MABL 的鲁棒性来自“标准化测量问答”，不是来自删除采样信息。它允许 informative sampling 存在，但把它从“事实日历捷径”转换为“在统一动作面板下可复现的状态响应”。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从事实事件编码改为 Measurement-Action Belief Encoder

MABL 把每个观测事件重新表示为 action-observation pair：

```text
event_i = (a_i, x_i)
a_i = (var_id_i, delta_t_i, quality_i, panel_flag_i)
```

1. **Causal Belief Encoder**
   - 输入按时间排序的 `(a_i, x_i)`。
   - 用 causal GRU / SSM / kernel-smoothed CDE stem 得到 `b_i = Belief(history up to i)`。
   - 这里可以吸收 MVC-CDE 的 kernel smoothing 作为稳定 value stem，但不把 smoothing、solver trace 或多视图路径作为主机制。

2. **Measurement Response Head**
   - 给定 belief `b` 和任意测量动作 `a`，输出潜在观测分布：

```text
mu_a, logvar_a = ResponseHead(b, a)
p(x | do(measure a), history) = Normal(mu_a, exp(logvar_a))
```

   - 这不是 missingness pattern encoder，因为 action 只作为“被问什么问题”的条件，不直接给分类头提供事实 policy。

3. **Canonical Action Battery**
   - 预定义或轻量学习一组标准测量动作 `A*`：
     - 每个变量在早/中/晚时间偏移下的查询；
     - 若干典型质量水平；
     - 单变量查询与小 panel 查询；
     - 可按 PYRREGULAR-style 统一接口跨数据集定义。
   - `A*` 不依赖当前样本事实采样日历，也不含 label 或 environment id。

4. **Bisimulation Lens Signature**

```text
R_A*(b) = concat_j [mu(b, a_j), logvar(b, a_j), response_entropy(b, a_j)]
logits = Classifier(R_A*(b_final))
```

分类器看到的是“在标准动作电池下，该病程/系统/天体会如何响应测量”，而不是“训练医院刚好测了什么”。

### 2.2 改 Sampling Branch：从 policy 表征改为 Action Planner Auditor

现有 sampling branch 不进入分类头，改成两个作用：

1. **Factual Action Parser**
   - 从 `(time, variable, mask, quality)` 构造事实动作 `a_i`。
   - 它只是把采样过程翻译成 action language，不输出采样概率、hazard、density ratio 或 protocol tax。

2. **Counterfactual Action Bank**
   - 当前反事实干预模块不再生成一致性 views、risk views、knockoff calendars 或 censor/stopping recipes。
   - 它生成一组 measurement action probes：
     - `routine_panel_query`：若按常规 panel 追问，会预测什么值；
     - `delayed_followup_query`：若延迟复测，会预测什么值；
     - `budgeted_single_var_query`：若只能测少量变量，会预测什么；
     - `quality_degraded_query`：若测量质量下降，预测不确定性是否合理上升。
   - 这些 probes 用于训练 response head 的世界模型，不直接生成分类增强样本。

### 2.3 改 Loss：从采样去偏转向 Measurement-Action Bisimulation Closure

总目标：

```text
L = L_lens_cls
  + lambda_obs  * L_observed_action_nll
  + lambda_dyn  * L_action_bellman_closure
  + lambda_cal  * L_response_calibration
  + lambda_lip  * L_label_lipschitz_on_response
```

#### A. Lens Classification Loss `L_lens_cls`

只用 canonical action lens signature 分类：

```text
R = ResponseSignature(b_final, A*)
L_lens_cls = CE(Classifier(R), y)
```

这一步是 MABL 与普通 action-conditioned model 的关键区别：事实采样动作可以帮助 belief 更新，但分类决策必须通过统一的 `A*` 追问接口完成。

#### B. Observed Action NLL `L_observed_action_nll`

对真实已观测事件做 teacher forcing。事件 `i` 发生前的 belief 为 `b_{i-1}`，事实动作为 `a_i`，观测值为 `x_i`：

```text
L_observed_action_nll =
  - log p(x_i | b_{i-1}, do(measure a_i))
```

这让 response head 学会回答“如果在这个历史下测这个变量，会看到什么”。它不是 iTimER 式重构误差伪观测，也不是 MILM 的 value-redacted classifier；采样 action 只是条件，监督来自观测值。

#### C. Action Bellman Closure `L_action_bellman_closure`

MABL 的双模拟来自一个闭环：如果模型预测在 belief `b` 下执行测量动作 `a` 会看到 `x`，那么把 `(a, x)` 写回 belief 后，新的 canonical response signature 应接近实际后续历史的 response signature。

对 held-out 事件 `i`：

```text
b_pred_next = BeliefTransition(b_{i-1}, a_i, x_i)
R_pred_next = ResponseSignature(b_pred_next, A*)
R_true_next = ResponseSignature(stopgrad(b_i), A*)
L_action_bellman_closure = SmoothL1(R_pred_next, R_true_next)
```

这不是跨采样 view 一致性。它不要求两个不同采样政策下的表示或 logits 相同；它只要求测量动作世界模型在“提问-得到答案-更新 belief”这一操作上闭合。

#### D. Response Calibration `L_response_calibration`

对观测动作的标准化残差做校准：

```text
eps_i = (x_i - mu_i) / sigma_i
L_response_calibration = mean_bucket(mean(eps)^2 + (var(eps) - 1)^2)
```

它与 OS-MQ 的停时鞅不同：这里不构造 stopped martingale，不用 optional-stopping moment；残差校准只服务于测量动作预测分布，使 canonical response 的方差可解释。

#### E. Label Lipschitz on Response Metric `L_label_lipschitz_on_response`

定义 measurement-action bisimulation distance：

```text
d_M(b, b') = mean_{a in A*} W2(
  p(x | b, do(a)),
  p(x | b', do(a))
)
```

若两个 belief 在所有 canonical actions 下响应接近，则分类 logits 不应出现过大跳变：

```text
L_label_lipschitz =
  relu(||logits(b) - logits(b')||_2 - kappa * d_M(b, b') - margin)^2
```

注意它不是要求同一样本的多采样 view logits 一致，也不是 contrastive learning。它是一个局部 Lipschitz 契约：分类器只能沿着 canonical measurement responses 真正变化的方向改变输出。事实 policy 导致的 action-history 表面差异，若没有改变 `A*` 下的响应，就无法制造大 logit jump。

### 2.4 改 Dataloader：返回 Measurement Action Probes，而不是 view pairs

新增 `MeasurementActionCollator`。每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `event_action`：
   - `action_var_id`
   - `action_delta_t`
   - `quality_code`
   - `panel_flag`
3. `canonical_action_bank`：统一动作电池 `A*`，不依赖事实 policy。
4. `heldout_action_index`：从已观测事件中抽样，训练 observed-action NLL 和 Bellman closure。
5. `counterfactual_action_probe_bank`：
   - 只定义“如果测什么”的 probes；
   - 不生成多份采样视图；
   - 不要求 logits/representation 一致；
   - 不估计 policy probability。
6. `response_pair_index`：batch 内 belief pairs，用于 response-metric Lipschitz 契约。

### 2.5 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `CausalBeliefEncoder`，输入 action-observation pair。
- 现有 sampling branch 改为 `FactualActionParser + ActionPlannerAuditor`，负责把采样日历翻译成测量动作，并生成 counterfactual probes。
- 现有 counterfactual intervention 改为“标准化测量问答”：不再重采样整条序列，而是对同一 belief 提出标准动作问题。
- 推理阶段：
  - 先用事实观测更新 belief；
  - 用固定 `A*` 生成 measurement-response signature；
  - 用 signature 分类；
  - 可输出哪些 canonical actions 对分类最敏感，作为“建议补测/补采样”的主动诊断。

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


def gaussian_nll(value: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (value - mean).pow(2) / logvar.exp().clamp_min(1e-6))


class MeasurementActionEmbedder(nn.Module):
    """Embed measurement actions: what variable to measure, when, and at what quality."""

    def __init__(self, num_vars: int, quality_bins: int, action_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, action_dim)
        self.quality_embed = nn.Embedding(quality_bins, action_dim)
        self.proj = nn.Sequential(
            nn.Linear(2 * action_dim + 2, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim),
        )

    def forward(
        self,
        action_var_id: torch.Tensor,
        action_delta_t: torch.Tensor,
        quality_code: torch.Tensor,
        panel_flag: torch.Tensor,
    ) -> torch.Tensor:
        var_h = self.var_embed(action_var_id.clamp_min(0))
        quality_h = self.quality_embed(quality_code.clamp_min(0))
        action_x = torch.cat(
            [
                var_h,
                quality_h,
                torch.log1p(action_delta_t).unsqueeze(-1),
                panel_flag.to(action_delta_t.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.proj(action_x)


class CausalBeliefEncoder(nn.Module):
    """Encode action-observation histories into causal beliefs."""

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_proj = nn.Sequential(
            nn.Linear(action_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        action_h: torch.Tensor,
        event_value: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> dict:
        obs_x = torch.cat([action_h, event_value.unsqueeze(-1)], dim=-1)
        step_h = self.obs_proj(obs_x) * event_mask.unsqueeze(-1)
        belief_seq, _ = self.rnn(step_h)

        # pre_belief at event i contains history before observing event i.
        pre_belief = torch.zeros_like(belief_seq)
        pre_belief[:, 1:] = belief_seq[:, :-1]
        final_belief = masked_mean(belief_seq, event_mask, dim=1)
        return {
            "belief_seq": belief_seq,
            "pre_belief": pre_belief,
            "final_belief": final_belief,
        }


class MeasurementResponseHead(nn.Module):
    """Predict potential measurement values under do(measure action)."""

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, belief: torch.Tensor, action_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action_h.dim() == 3:
            belief = belief[:, None, :].expand(-1, action_h.size(1), -1)
        raw = self.net(torch.cat([belief, action_h], dim=-1))
        mean = raw[..., 0]
        logvar = raw[..., 1].clamp(-6.0, 4.0)
        return mean, logvar


class BeliefTransition(nn.Module):
    """Update a belief after asking an action and receiving a value."""

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(action_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, belief: torch.Tensor, action_h: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        step = self.input_proj(torch.cat([action_h, value.unsqueeze(-1)], dim=-1))
        return self.update(step, belief)


class MeasurementActionBisimulationLens(nn.Module):
    """Sampling-policy robust classifier via canonical measurement-action responses."""

    def __init__(
        self,
        num_vars: int,
        quality_bins: int,
        action_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_canonical_actions: int,
    ):
        super().__init__()
        self.action_embedder = MeasurementActionEmbedder(num_vars, quality_bins, action_dim)
        self.belief = CausalBeliefEncoder(action_dim, hidden_dim)
        self.response = MeasurementResponseHead(hidden_dim, action_dim)
        self.transition = BeliefTransition(hidden_dim, action_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * num_canonical_actions, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def embed_actions(self, batch: dict, prefix: str = "") -> torch.Tensor:
        return self.action_embedder(
            action_var_id=batch[f"{prefix}action_var_id"],
            action_delta_t=batch[f"{prefix}action_delta_t"],
            quality_code=batch[f"{prefix}quality_code"],
            panel_flag=batch[f"{prefix}panel_flag"],
        )

    def response_signature(self, belief: torch.Tensor, canonical_action_h: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.response(belief, canonical_action_h)
        entropy = 0.5 * (1.0 + torch.log(torch.tensor(2.0 * 3.1415926535, device=logvar.device)) + logvar)
        return torch.cat([mean, logvar, entropy], dim=-1).flatten(start_dim=1)

    def forward(self, batch: dict) -> dict:
        action_h = self.embed_actions(batch)
        belief = self.belief(action_h, batch["event_value"], batch["event_mask"])
        canonical_action_h = self.embed_actions(batch, prefix="canonical_")
        signature = self.response_signature(belief["final_belief"], canonical_action_h)
        logits = self.classifier(signature)
        return {
            **belief,
            "action_h": action_h,
            "canonical_action_h": canonical_action_h,
            "signature": signature,
            "logits": logits,
        }

    def observed_action_nll(self, out: dict, batch: dict) -> torch.Tensor:
        mean, logvar = self.response(out["pre_belief"], out["action_h"])
        raw = gaussian_nll(batch["event_value"], mean, logvar)
        return (raw * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

    def action_bellman_closure_loss(self, out: dict, batch: dict) -> torch.Tensor:
        # heldout_index points to observed events used as teacher-forced action probes.
        idx = batch["heldout_event_index"]
        bsz = batch["event_value"].size(0)
        row = torch.arange(bsz, device=batch["event_value"].device)

        pre_b = out["pre_belief"][row, idx]
        action_h = out["action_h"][row, idx]
        value = batch["event_value"][row, idx]
        pred_next_b = self.transition(pre_b, action_h, value)

        pred_sig = self.response_signature(pred_next_b, out["canonical_action_h"])
        true_next_b = out["belief_seq"][row, idx].detach()
        true_sig = self.response_signature(true_next_b, out["canonical_action_h"]).detach()
        return F.smooth_l1_loss(pred_sig, true_sig)

    def response_calibration_loss(self, out: dict, batch: dict) -> torch.Tensor:
        mean, logvar = self.response(out["pre_belief"], out["action_h"])
        eps = (batch["event_value"] - mean) / (0.5 * logvar).exp().clamp_min(1e-3)
        eps_mean = masked_mean(eps, batch["event_mask"], dim=1)
        eps2_mean = masked_mean(eps.pow(2), batch["event_mask"], dim=1)
        return eps_mean.pow(2).mean() + (eps2_mean - 1.0).pow(2).mean()

    def response_distance(self, sig_a: torch.Tensor, sig_b: torch.Tensor) -> torch.Tensor:
        return (sig_a - sig_b).view(sig_a.size(0), -1).pow(2).mean(dim=-1).sqrt()

    def label_lipschitz_loss(self, out: dict, batch: dict, kappa: float = 2.0, margin: float = 0.05) -> torch.Tensor:
        pair_i = batch["response_pair_i"]
        pair_j = batch["response_pair_j"]
        sig_i = out["signature"][pair_i]
        sig_j = out["signature"][pair_j]
        logits_i = out["logits"][pair_i]
        logits_j = out["logits"][pair_j]
        d_resp = self.response_distance(sig_i, sig_j).detach()
        d_logit = (logits_i - logits_j).pow(2).sum(dim=-1).sqrt()
        return F.relu(d_logit - kappa * d_resp - margin).pow(2).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_obs: float = 0.30,
        lambda_dyn: float = 0.20,
        lambda_cal: float = 0.10,
        lambda_lip: float = 0.05,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]
        cls_loss = F.cross_entropy(out["logits"], labels)
        obs_loss = self.observed_action_nll(out, batch)
        dyn_loss = self.action_bellman_closure_loss(out, batch)
        cal_loss = self.response_calibration_loss(out, batch)
        lip_loss = self.label_lipschitz_loss(out, batch)
        total = (
            cls_loss
            + lambda_obs * obs_loss
            + lambda_dyn * dyn_loss
            + lambda_cal * cal_loss
            + lambda_lip * lip_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "observed_action_nll": obs_loss.detach(),
            "action_bellman_closure_loss": dyn_loss.detach(),
            "response_calibration_loss": cal_loss.detach(),
            "label_lipschitz_loss": lip_loss.detach(),
        }


@torch.no_grad()
def build_measurement_action_batch(
    batch: dict,
    num_vars: int,
    quality_bins: int = 3,
    canonical_offsets: tuple[float, ...] = (0.05, 0.25, 0.50),
) -> dict:
    """Create factual actions and a policy-independent canonical action battery."""

    event_time = batch["event_time"]
    event_var_id = batch["event_var_id"]
    event_mask = batch["event_mask"]
    bsz, num_events = event_time.shape
    device = event_time.device

    delta_t = torch.zeros_like(event_time)
    delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)

    # Factual measurement actions parsed from the observation calendar.
    out = dict(batch)
    out["action_var_id"] = event_var_id
    out["action_delta_t"] = delta_t
    out["quality_code"] = torch.zeros_like(event_var_id).clamp_max(quality_bins - 1)
    same_time_gap = delta_t <= delta_t[event_mask > 0].median().clamp_min(1e-6) if (event_mask > 0).any() else torch.zeros_like(delta_t, dtype=torch.bool)
    out["panel_flag"] = same_time_gap.to(event_time.dtype) * event_mask

    # Canonical actions: variable x time-offset probes, independent of factual policy.
    canonical_vars = []
    canonical_dt = []
    canonical_quality = []
    canonical_panel = []
    for var_idx in range(num_vars):
        for offset in canonical_offsets:
            canonical_vars.append(torch.full((bsz,), var_idx, device=device, dtype=event_var_id.dtype))
            canonical_dt.append(torch.full((bsz,), offset, device=device, dtype=event_time.dtype))
            canonical_quality.append(torch.zeros(bsz, device=device, dtype=event_var_id.dtype))
            canonical_panel.append(torch.zeros(bsz, device=device, dtype=event_time.dtype))

    out["canonical_action_var_id"] = torch.stack(canonical_vars, dim=1)
    out["canonical_action_delta_t"] = torch.stack(canonical_dt, dim=1)
    out["canonical_quality_code"] = torch.stack(canonical_quality, dim=1)
    out["canonical_panel_flag"] = torch.stack(canonical_panel, dim=1)

    # Teacher-forced held-out probes for Bellman closure.
    valid_rank = event_mask.cumsum(dim=1)
    valid_count = event_mask.sum(dim=1).long().clamp_min(1)
    sample_pos = (0.5 * valid_count.float()).long().clamp(0, num_events - 1)
    heldout_idx = torch.zeros(bsz, device=device, dtype=torch.long)
    for row in range(bsz):
        candidates = torch.where(event_mask[row] > 0)[0]
        heldout_idx[row] = candidates[min(sample_pos[row].item(), candidates.numel() - 1)]
    out["heldout_event_index"] = heldout_idx

    # Pair indices for response-metric Lipschitz regularization.
    out["response_pair_i"] = torch.arange(bsz, device=device)
    out["response_pair_j"] = torch.arange(bsz, device=device).roll(shifts=1)
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `measurement-action reversal`：训练环境只在高风险时测某变量，测试环境将该变量常规化。
   - `panel-action shift`：训练环境把多个变量作为 panel 问，测试环境拆成单变量异步问。
   - `delayed-followup shift`：训练环境短延迟复测，测试环境长延迟复测。
   - `quality-action shift`：同一变量在不同设备或巡天条件下测量质量系统性变化。
   - `value-pending shift`：借鉴 MILM，测量动作已发生但值尚未返回，测试 response uncertainty 是否合理。

2. **对比方法**
   - 普通 irregular encoder。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - MILM-style value-redacted sampling classifier。
   - MVC-CDE / attentive kernel smoothing baseline。
   - LLMTS 风格 irregular encoder + alignment baseline。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN、RKHS Cubature Debiaser。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - action-response calibration error：事实测量动作下预测分布是否校准。
   - canonical response drift：同一样本在不同事实 policy 下 `R_A*` 的漂移。
   - policy-only shortcut gap：只用事实 action history 分类的性能与 MABL 性能差距。
   - action recommendation fidelity：高敏感 canonical actions 是否对应有价值补测变量。

4. **消融实验**
   - 去掉 canonical action battery，直接用 final belief 分类，验证是否重新依赖事实 policy。
   - 去掉 `L_action_bellman_closure`，检查 response head 是否只是静态重构器。
   - 去掉 observed-action NLL，验证 measurement action semantics 是否消失。
   - 将 canonical actions 替换为事实 actions，验证鲁棒性是否来自标准化追问而非额外参数。
   - 扫描 canonical action 数量，评估动作电池覆盖度与计算开销。

## 5. 预期创新性

1. **从采样机制去偏转向测量动作双模拟**：首次把 sampling-policy shift 表述为 observer measurement-action policy 改变，而不是 mask 分布、危险率、日历、停时、测度或求解轨迹改变。
2. **从事实观测分类转向标准化动作问答分类**：分类器不直接消费事实采样日历，而消费一组 policy-independent canonical measurement responses。
3. **从反事实视图增强转向 action world model closure**：counterfactual intervention 不生成一致性 views，而是生成测量动作 probes，用 Bellman-style closure 训练“问-答-更新 belief”的闭环。
4. **吸收 MVC-CDE 但不复用 solver/path 机制**：可用 kernel-smoothed continuous stem 稳定 belief，但核心鲁棒性来自 action response lens，而不是 path smoothing、NFE trace 或多视图 CDE。
5. **吸收 MILM 的 informative sampling 启发但避免 sampling-only shortcut**：承认“下单/测量动作”有信息，但要求它通过标准动作响应转化为可复现状态证据。
6. **部署诊断直接服务补采样**：若某些 canonical actions 的预测方差高且对分类敏感，模型可以建议下一步最有价值的补测变量/时间，而不是只输出拒识或不确定性。

## 6. 一句话投稿卖点

**MABL 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“观察者测量动作策略改变”问题，并通过 Measurement-Action Bisimulation Lens 将事实采样历史转化为一组 policy-independent canonical measurement responses，使分类器依据标准化测量问答中的状态可辨性决策，而不是依据训练环境中特定的采样频率、panel 下单、复测延迟或 value-pending 日历捷径。**
