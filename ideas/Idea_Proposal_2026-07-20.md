# Title: Triage-Queue Debt Neutralizer：面向采样策略偏移的分诊队列债务中和分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md`，纳入其中记录但当前工作区未检出的历史提案机制。
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
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-06-12.md`
  - `paper_daily_2026-06-25.md`
  - `paper_daily_2026-06-26.md`
  - `paper_daily_2026-07-13.md`
  - `paper_daily_2026-07-19.md`

### 历史核心机制黑名单

为避免与历史提案发生思维重合，本提案明确避开以下机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、hazard-driven resampling、采样 score 零空间正交。
8. 生理流-采样算子交换子、value/policy graph 分离、policy residual sink。
9. additive evidence market、protocol tax、token budget、边际证据审计。
10. posterior quotient dynamics、采样似然后验相除、interventional marginalization。
11. reconstruction error cartography、语义 compiler、HSIC redaction。
12. policy-simplex randomized smoothing、certified radius、Dirichlet / logit-normal do-sampler。
13. Radon-Nikodym density ratio、doubly robust target-measure correction。
14. optional-stopping martingale query、previsible innovation、停时矩控制。
15. censored excursion topology capsules、拓扑审查区间、fragmentation sobriety。
16. policy-gauge horizontal transport、chart span supervision、vertical blindness。
17. policy-only negative film、shadow eraser/stencil。
18. latent packet syndrome code、parity-check、syndrome locator、packet repair。
19. conditional knockoff calendar、soft knockoff-FDR firewall、swap symmetry。
20. observability witness、measurement-action bisimulation、policy-word signature、thermodynamic free-energy、Sklar copula。
21. subjective-logic evidential shield、policy-induced vacuity、class-wise evidence discount。
22. policy lattice meet/join、submodular margins、quality-order loss、shortcut curvature。
23. solver trace front-door neutralization、NFE/roughness/step-size 中介标准化。

本提案选择一个新的正交切入点：**不把采样策略看成概率、缺失、图、拓扑、坐标、证据税、后验因子或数值求解轨迹，而是看成一个资源受限的临床/设备分诊队列。采样政策决定哪些局部细节请求被服务、哪些被积压；分类器只能使用经过队列债务中和的细节证据，从而避免把“某环境愿意为某类样本投入更多测量资源”误当成类别本身。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily_2026-07-19.md` 中的 **Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context-Detail Interaction** 提出 coarse-to-fine 的 context-detail interaction：全局上下文先判断哪里可能有稀疏事件，再由局部细节分支精查边界和事件类型。这个机制非常适合稀疏医疗时序，因为真实诊断事件往往只占很短片段。

但对非规则采样分类来说，context-detail gate 还有一个尚未被充分处理的偏移风险：

- 在训练医院中，全局风险一高，医生立刻追加化验或复测，局部细节被高分辨率服务。
- 在测试医院中，同样风险可能因为资源、班次或流程限制而延迟服务。
- 在设备场景中，报警后是否开启高频采样取决于电量、带宽或本地阈值，而不完全取决于真实状态。
- 因此，“局部细节是否被看见”既可能是状态触发，也可能是服务规则触发。

如果直接复用 adaptive gate，模型会学到：

```text
全局上下文疑似高危 -> 局部细节分支被激活 -> 分类器高置信预测
```

这条链在同一医院有效，但跨采样策略时会退化。真正需要鲁棒的是：

```text
全局上下文产生 detail request，
采样政策作为 service discipline 决定请求是否被服务，
分类器只使用扣除了服务机会差异后的 detail evidence。
```

**Triage-Queue Debt Neutralizer (TQDN)** 的核心直觉是：

> 采样偏移不是简单“看见/没看见”，而是“哪些状态请求获得了测量服务、哪些请求被积压”。若某个类别在训练环境中总能获得更多局部细节服务，模型必须为这种服务机会差异建立队列债务账，而不能把被服务本身当作类别证据。

与历史方案的差异在于：

- 不估计采样概率或危险率；
- 不要求多采样视图 logits 一致；
- 不做策略对抗、density ratio、knockoff、evidential uncertainty 或 certified smoothing；
- 不把采样 token 征税，也不把采样结构修复成 codeword；
- 只把 sampling branch 用来模拟 **detail request arrival / service / backlog debt**。

这样既吸收了 ICLR 2026 adaptive context-detail gate 的前沿机制，又把它推进到 sampling-policy shift 的因果鲁棒版本。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通 Context-Detail Gate 改为 Queue-Debt Context-Detail Encoder

TQDN 保留“全局上下文 + 局部细节”的结构，但在二者之间插入一个可微分诊队列。

1. **Global Context Scout**
   - 输入稀疏事件流 `(value, time, variable, mask, measurement_std)`。
   - 输出每个时间位置的粗粒度上下文 `c_i`。
   - `c_i` 只负责产生 detail request，不直接决定最终类别。

2. **Detail Request Arrivals**
   - 从上下文产生细节请求强度：

```text
a_i = sigmoid(RequestHead(c_i))
```

   - `a_i` 表示“状态本身是否值得进一步精查”，不是“实际是否被采样”。

3. **Policy Service Discipline**
   - sampling branch 只根据时间戳、变量可见性、局部密度和反事实 service recipe 产生服务量 `s_i`。
   - `s_i` 表示当前采样政策愿意服务多少 detail request。
   - 它不进入分类头，也不预测标签。

4. **Queue Debt Recurrence**
   - 用资源队列模拟未服务请求的积压：

```text
debt_i = relu(debt_{i-1} + a_i - s_i)
served_i = min(a_i + debt_{i-1}, s_i)
```

   - 若某类样本在训练医院被密集复测，`served_i` 高但 `debt_i` 低；若测试医院延迟复测，`debt_i` 上升。
   - 分类器不能直接把 `served_i` 当作类别证据，而要看“在当前队列债务下，局部细节是否仍有净状态价值”。

5. **Debt-Neutral Detail Gate**
   - 局部细节分支输出 `d_i`。
   - 最终进入分类的是债务中和后的 detail：

```text
g_i = sigmoid(DetailQuality(d_i) - DebtAdapter(debt_i))
detail_safe_i = g_i * d_i
```

   - 直觉：如果局部细节只是因为训练政策服务充分而出现，`debt_i` 和服务规则改变会暴露它；如果它是真实状态异常，即使服务规则变化，它仍会通过 `DetailQuality` 获得足够净贡献。

### 2.2 改 Loss：从一致性/对抗转向 Queue Work-Conservation + Debt-Neutral Margins

总目标：

```text
L = L_cls
  + lambda_q   * L_queue_conservation
  + lambda_srv * L_service_replay
  + lambda_dn  * L_debt_neutral_margin
  + lambda_reg * L_request_sobriety
```

#### A. 分类损失 `L_cls`

分类器只使用 `global_context + debt-neutral detail`：

```text
L_cls = CE(Classifier(pool(c_i, detail_safe_i)), y)
```

#### B. Queue Conservation `L_queue_conservation`

队列不能退化成任意 gate。要求 backlog recurrence 满足工作守恒：

```text
debt_i ≈ relu(debt_{i-1} + request_i - service_i)
served_i <= request_i + debt_{i-1}
served_i <= service_i
```

这不是采样概率似然，也不是 hazard；它只保证“请求、服务、积压”三者在资源队列语义上自洽。

#### C. Service Replay `L_service_replay`

事实采样下，服务量应能重放观测到的细节可用性，例如局部高频复测、同窗 panel、value-pending 或高质量测量：

```text
L_service_replay = BCE(service_i, observed_detail_service_i)
```

该损失训练 sampling branch 理解“当前政策如何服务请求”，但 `service_i` 不进入分类器。

#### D. Debt-Neutral Margin `L_debt_neutral_margin`

对同一 `request_i` 使用多种反事实 service disciplines：

- `fifo_round`: 固定查房式先来先服务；
- `priority_burst`: 报警后高优先服务；
- `budget_cut`: 每个变量组固定服务预算；
- `delayed_followup`: 高风险请求延迟服务。

不要求各服务规则下 logits 相同，而是约束真实类 margin 对 **队列债务水平** 的直接斜率：

```text
margin_r = logit_y(service_rule=r) - max_{k != y} logit_k(service_rule=r)
L_debt_neutral_margin = slope(margin_r, mean_debt_r)^2
```

这与历史 solver-trace slope 不同：中介不是数值求解轨迹，而是采样资源队列债务；它直接刻画“分类是否依赖某政策服务更勤快/更拖延”。

#### E. Request Sobriety `L_request_sobriety`

避免模型把所有位置都声明为高请求：

```text
L_request_sobriety =
  mean(request_i)
  + relu(min_detail_coverage - served_true_detail_rate)^2
```

第一项防止请求泛滥，第二项防止模型通过关闭所有细节逃避偏移问题。

### 2.3 改 Dataloader：返回 Triage Service Bank，而不是 mask 增强

新增 `TriageServiceCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `observed_detail_service`：哪些事件处在高质量/高频/局部精查服务状态。
3. `service_recipe_bank`：反事实服务规则：
   - `fifo_round`：固定查房窗口服务；
   - `priority_burst`：报警或异常后短窗口优先服务；
   - `variable_budget`：每个变量组固定服务容量；
   - `delayed_followup`：请求产生后延迟服务。
4. `service_capacity_bank`：每个规则下的时间位置服务容量。
5. `detail_visibility_bank`：每个规则下局部 detail inspector 能看到的细节位置。

关键区别：

- 不生成对比正样本；
- 不要求 representation 或 logits 一致；
- 不估计 observation probability、density ratio 或 hazard；
- 不把 service recipe 作为分类特征；
- 只用反事实服务规则训练队列债务中和。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `GlobalContextScout + LocalDetailInspector`。
- 现有 sampling branch 改为 `ServiceDisciplineNet`，输出 service capacity，不进入 classifier。
- 现有 counterfactual intervention 改为 `TriageServiceBank`，生成不同服务规则下的 detail service pattern。
- 推理阶段无需策略标签；从事实观测结构估计服务量与队列债务，并输出：
  - 分类概率；
  - mean queue debt；
  - service reliance gap；
  - 哪些时间窗/变量组的预测依赖高服务机会而非高状态请求。

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


class GlobalContextScout(nn.Module):
    """Coarse context encoder that raises detail requests."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.request_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]
        measurement_std = batch.get("measurement_std", torch.zeros_like(value))

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
        event_h = self.event_proj(event_x) * event_mask.unsqueeze(-1)
        context, _ = self.rnn(event_h)
        request = self.request_head(context).squeeze(-1) * event_mask
        return {"context": context, "request": request, "delta_t": delta_t}


class LocalDetailInspector(nn.Module):
    """Fine local detail branch, activated through queue-debt neutralization."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.detail_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.quality = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict, visibility: torch.Tensor | None = None) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]
        measurement_std = batch.get("measurement_std", torch.zeros_like(value))

        if visibility is None:
            visibility = event_mask
        active = (event_mask * visibility).clamp(0.0, 1.0)

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        local_jump = torch.zeros_like(value)
        local_jump[:, 1:] = value[:, 1:] - value[:, :-1]
        var_h = self.var_embed(var_id.clamp_min(0))
        detail_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                local_jump.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(measurement_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        detail = self.detail_proj(detail_x) * active.unsqueeze(-1)
        quality = self.quality(detail).squeeze(-1) * active
        return {"detail": detail, "detail_quality": quality, "active": active}


class ServiceDisciplineNet(nn.Module):
    """Predict service capacity from sampling coordinates and a service recipe."""

    def __init__(self, num_vars: int, recipe_dim: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.recipe_proj = nn.Linear(recipe_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(num_vars + hidden_dim + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict, recipe: torch.Tensor) -> torch.Tensor:
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]
        bsz, num_events = time.shape

        horizon = time.amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        var_onehot = F.one_hot(var_id.clamp_min(0), self.num_vars).to(time.dtype)
        early = (time_norm <= 0.33).to(time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype)
        late = (time_norm > 0.66).to(time.dtype)
        local_density = 1.0 / (1.0 + delta_t)
        recipe_h = self.recipe_proj(recipe)[:, None, :].expand(bsz, num_events, -1)

        service_x = torch.cat(
            [
                var_onehot,
                recipe_h,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                local_density.unsqueeze(-1),
                early.unsqueeze(-1),
                middle.unsqueeze(-1),
                late.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(service_x).squeeze(-1) * event_mask


def differentiable_triage_queue(
    request: torch.Tensor,
    service: torch.Tensor,
    event_mask: torch.Tensor,
) -> dict:
    """Simulate soft queue debt and served request mass."""

    debts = []
    served = []
    debt_prev = torch.zeros_like(request[:, 0])
    for idx in range(request.size(1)):
        available = request[:, idx] + debt_prev
        service_i = service[:, idx]

        # Smooth approximation of min(available, service_i).
        served_i = available + service_i - torch.sqrt((available - service_i).pow(2) + 1e-4)
        served_i = 0.5 * served_i
        served_i = served_i * event_mask[:, idx]

        debt_i = F.relu(available - served_i) * event_mask[:, idx]
        debts.append(debt_i)
        served.append(served_i)
        debt_prev = debt_i

    return {
        "debt": torch.stack(debts, dim=1),
        "served": torch.stack(served, dim=1),
    }


class DebtNeutralDetailGate(nn.Module):
    """Allow local details through only after accounting for queue debt."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.debt_adapter = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        detail: torch.Tensor,
        detail_quality: torch.Tensor,
        debt: torch.Tensor,
        served: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> dict:
        debt_penalty = self.debt_adapter(torch.log1p(debt).unsqueeze(-1)).squeeze(-1)
        gate = torch.sigmoid(detail_quality - debt_penalty) * served * event_mask
        safe_detail = detail * gate.unsqueeze(-1)
        return {"gate": gate, "safe_detail": safe_detail}


class TriageQueueDebtNeutralizer(nn.Module):
    """Sampling-policy robust context-detail classifier via triage-queue debt."""

    def __init__(
        self,
        num_vars: int,
        recipe_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.context = GlobalContextScout(num_vars, hidden_dim)
        self.detail = LocalDetailInspector(num_vars, hidden_dim)
        self.service = ServiceDisciplineNet(num_vars, recipe_dim, hidden_dim)
        self.debt_gate = DebtNeutralDetailGate(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.recipe_dim = recipe_dim

    def encode_with_recipe(
        self,
        batch: dict,
        recipe: torch.Tensor,
        visibility: torch.Tensor | None = None,
    ) -> dict:
        ctx = self.context(batch)
        det = self.detail(batch, visibility=visibility)
        service = self.service(batch, recipe)
        queue = differentiable_triage_queue(ctx["request"], service, batch["event_mask"])
        gate = self.debt_gate(
            detail=det["detail"],
            detail_quality=det["detail_quality"],
            debt=queue["debt"],
            served=queue["served"],
            event_mask=batch["event_mask"],
        )

        context_pool = masked_mean(ctx["context"], batch["event_mask"], dim=1)
        detail_pool = masked_mean(gate["safe_detail"], batch["event_mask"], dim=1)
        logits = self.classifier(torch.cat([context_pool, detail_pool], dim=-1))

        return {
            "logits": logits,
            **ctx,
            **det,
            **queue,
            **gate,
            "service": service,
        }

    def forward(self, batch: dict) -> dict:
        zero_recipe = torch.zeros(
            batch["event_value"].size(0),
            self.recipe_dim,
            device=batch["event_value"].device,
            dtype=batch["event_value"].dtype,
        )
        return self.encode_with_recipe(batch, zero_recipe)

    def queue_conservation_loss(self, out: dict, event_mask: torch.Tensor) -> torch.Tensor:
        request = out["request"]
        service = out["service"]
        debt = out["debt"]
        served = out["served"]

        debt_prev = torch.zeros_like(debt)
        debt_prev[:, 1:] = debt[:, :-1]
        target_debt = F.relu(debt_prev + request - served) * event_mask
        recur_loss = F.smooth_l1_loss(debt, target_debt.detach())
        cap_loss = F.relu(served - service - 1e-3).pow(2).mean()
        req_loss = F.relu(served - request - debt_prev - 1e-3).pow(2).mean()
        return recur_loss + cap_loss + req_loss

    def debt_neutral_margin_loss(self, batch: dict, labels: torch.Tensor) -> torch.Tensor:
        margins = []
        debts = []
        for recipe, visibility in zip(
            batch["service_recipe_bank"].unbind(dim=1),
            batch["detail_visibility_bank"].unbind(dim=1),
        ):
            out = self.encode_with_recipe(batch, recipe=recipe, visibility=visibility)
            margins.append(true_margin(out["logits"], labels))
            debts.append(masked_mean(out["debt"], batch["event_mask"], dim=1))

        margin = torch.stack(margins, dim=1)
        debt = torch.stack(debts, dim=1)
        debt_c = debt - debt.mean(dim=1, keepdim=True)
        margin_c = margin - margin.mean(dim=1, keepdim=True)
        slope = (debt_c * margin_c).mean(dim=1) / debt_c.pow(2).mean(dim=1).clamp_min(1e-6)
        return slope.pow(2).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_q: float = 0.10,
        lambda_srv: float = 0.15,
        lambda_dn: float = 0.20,
        lambda_reg: float = 0.02,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)

        queue_loss = self.queue_conservation_loss(out, batch["event_mask"])
        service_loss = F.binary_cross_entropy(
            out["service"].clamp(1e-4, 1.0 - 1e-4),
            batch["observed_detail_service"].to(out["service"].dtype),
            reduction="none",
        )
        service_loss = (service_loss * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

        debt_margin_loss = self.debt_neutral_margin_loss(batch, labels)
        request_sobriety = out["request"].mean() + F.relu(0.05 - out["served"].mean()).pow(2)

        total = (
            cls_loss
            + lambda_q * queue_loss
            + lambda_srv * service_loss
            + lambda_dn * debt_margin_loss
            + lambda_reg * request_sobriety
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "queue_conservation_loss": queue_loss.detach(),
            "service_replay_loss": service_loss.detach(),
            "debt_neutral_margin_loss": debt_margin_loss.detach(),
            "request_sobriety_loss": request_sobriety.detach(),
            "mean_queue_debt": out["debt"].mean().detach(),
        }


@torch.no_grad()
def build_triage_service_bank(batch: dict, recipe_dim: int = 4) -> dict:
    """Create counterfactual service disciplines for triage-queue training."""

    time = batch["event_time"]
    var_id = batch["event_var_id"]
    event_mask = batch["event_mask"]
    value = batch["event_value"]
    bsz, num_events = time.shape
    device = time.device

    horizon = time.amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon
    delta_t = torch.zeros_like(time)
    delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

    recipes = []
    visibility = []

    # 1. Fixed routine round: serve early/middle windows.
    routine = ((time_norm <= 0.66).to(event_mask.dtype) * event_mask)
    recipes.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(bsz, -1))
    visibility.append(routine)

    # 2. Priority burst: serve local jumps and dense follow-up windows.
    jump = torch.zeros_like(value)
    jump[:, 1:] = (value[:, 1:] - value[:, :-1]).abs()
    jump_thr = masked_mean(jump, event_mask, dim=1).unsqueeze(-1)
    priority = ((jump >= jump_thr).to(event_mask.dtype) * event_mask)
    recipes.append(torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(bsz, -1))
    visibility.append(priority)

    # 3. Variable budget: keep a small number of observations per variable.
    budget = torch.zeros_like(event_mask)
    for var_idx in torch.unique(var_id[event_mask > 0]).tolist():
        var_hit = ((var_id == int(var_idx)) & (event_mask > 0)).to(event_mask.dtype)
        order = var_hit.cumsum(dim=1)
        budget = torch.maximum(budget, (order <= 3).to(event_mask.dtype) * var_hit)
    recipes.append(torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).expand(bsz, -1))
    visibility.append(budget)

    # 4. Delayed follow-up: serve late windows and larger gaps.
    gap_threshold = masked_mean(delta_t, event_mask, dim=1).unsqueeze(-1)
    delayed = (((time_norm >= 0.50) | (delta_t >= gap_threshold)).to(event_mask.dtype) * event_mask)
    recipes.append(torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(bsz, -1))
    visibility.append(delayed)

    recipe_bank = torch.stack(recipes, dim=1)
    visibility_bank = torch.stack(visibility, dim=1)

    if recipe_dim != 4:
        pad = torch.zeros(bsz, recipe_bank.size(1), max(recipe_dim - 4, 0), device=device)
        recipe_bank = torch.cat([recipe_bank[..., :recipe_dim], pad], dim=-1)[..., :recipe_dim]

    # Factual detail service proxy: high local density or high measurement detail.
    local_density = 1.0 / (1.0 + delta_t)
    density_thr = masked_mean(local_density, event_mask, dim=1).unsqueeze(-1)
    observed_service = ((local_density >= density_thr).to(event_mask.dtype) * event_mask)

    out = dict(batch)
    out["service_recipe_bank"] = recipe_bank
    out["detail_visibility_bank"] = visibility_bank
    out["observed_detail_service"] = observed_service
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `priority-to-routine shift`：训练医院报警后密集复测，测试医院固定查房式采样。
   - `routine-to-delayed shift`：训练环境按固定窗口服务，测试环境延迟 follow-up。
   - `variable-budget shift`：测试环境对某些变量组设置服务容量上限。
   - `detail-service reversal`：训练环境中高风险样本获得更多 detail service，测试环境中服务资源与风险弱相关或反转。

2. **对比方法**
   - 普通 context-detail adaptive gate。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - value-redacted sampling classifier。
   - 以及历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN 等。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - service-reliance gap：事实服务规则 logits 与队列债务中和 logits 的差距。
   - debt-margin slope：真实类 margin 对平均 queue debt 的斜率。
   - request-service mismatch：高请求但低服务样本是否更容易被模型识别为不可靠。
   - detail shortcut score：只用 observed service pattern 预测标签的能力。

4. **消融实验**
   - 去掉 queue debt，只保留普通 adaptive gate，验证是否重新依赖 detail service shortcut。
   - 去掉 `L_debt_neutral_margin`，检查跨服务规则偏移下 margin 是否随服务债务漂移。
   - 去掉 `L_service_replay`，验证 sampling branch 是否失去服务语义。
   - 将 service bank 换成随机 mask，验证收益来自队列服务规则而非普通增强。
   - 扫描 request sobriety 权重，验证模型不是通过关闭全部 detail branch 获得表面鲁棒性。

## 5. 预期创新性

1. **从采样概率转向服务队列**：首次把 sampling-policy shift 表述为 detail request 在不同 service discipline 下的服务机会差异。
2. **从 context-detail gate 转向 debt-neutral gate**：吸收 ICLR 2026 adaptive context-detail interaction，但加入队列债务中和，避免“被精查”本身成为类别捷径。
3. **从反事实增强转向服务规则账本**：counterfactual intervention 不生成一致性视图，而是生成 FIFO、priority、budget、delayed 等服务规则，用于训练队列债务诊断。
4. **从删除采样信息转向资源机会校正**：模型可以承认某些细节确实有价值，但必须区分“状态请求强”与“采样政策服务勤”。
5. **部署解释性强**：TQDN 能报告哪些预测依赖高服务机会、哪些样本处于高 request 但高 debt 的危险区域，从而指导补采样或人工复核。

## 6. 一句话投稿卖点

**TQDN 首次把非规则采样时间序列分类中的 sampling-policy shift 建模为“全局上下文产生局部精查请求，而采样政策作为资源受限分诊队列决定请求是否被服务”的问题，并通过 queue debt recurrence、service discipline bank 与 debt-neutral detail gate，让分类器依赖真实状态触发的细节价值，而不是依赖训练环境中特定医院、设备或流程提供的高分辨率服务机会。**
