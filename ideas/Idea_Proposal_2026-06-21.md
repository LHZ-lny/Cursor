# Title: Optional-Stopping Martingale Queries：面向采样策略偏移的停时鞅查询分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取 `paper_daily.md` 与 `paper_daily_2026-06-12.md`。
- 已读取当前工作区内全部历史提案：
  - `ideas/Idea_Proposal_2026-06-12.md`
  - `ideas/Idea_Proposal_2026-06-13.md`
  - `ideas/Idea_Proposal_2026-06-14.md`
  - `ideas/Idea_Proposal_2026-06-16.md`
  - `ideas/Idea_Proposal_2026-06-19.md`
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案核心机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`

### 历史核心机制黑名单

为避免与历史提案和 `paper_daily.md` 中已有机制发生思维重合，本提案明确避开以下方向作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder。
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
19. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed，或 QuITE 的普通 query token 适配层作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做去偏对抗，不做一致性、平滑认证、后验相除或证据定价，而是把采样策略看成作用在观测流上的停时规则；分类器只能消费相对于历史信息可预测的标准化创新鞅。若表示是真正的状态创新，按 optional stopping theorem，在不同采样停时下其期望漂移不应系统性改变；若模型利用采样政策捷径，则 stopped martingale 会出现可检测漂移。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列中的策略偏移常常表现为“何时停止观察、何时追加检测、何时触发复测”的改变。医疗场景尤其明显：某医院在报警后立刻复测，另一医院只在固定班次复查；某变量在训练医院中一旦连续正常就停止测量，在测试医院中仍按常规测量。此时，采样机制本质上不是一个静态 mask，而是一个依赖历史的 **stopping rule / optional sampling rule**。

近期 `paper_daily.md` 中两个前沿机制给了很好的结合点：

- **QuITE** 提示 query token 可以作为不规则观测到通用 backbone 的轻量适配层。但普通 query 会在注意力中同时吸收观测值和采样时刻，仍可能把策略节奏编码为类别捷径。
- **SDEVI** 提示连续-离散推断应尊重底层连续时间动力学，而不是把离散不规则观测当成补齐网格后的普通序列。但 SDEVI 没有显式处理 observation process 改变时分类器是否依赖采样停时。

Optional-Stopping Martingale Queries (OS-MQ) 的关键思想是：

> 把每个新观测转化为“在看到该观测之前就已经可预测的 query”与“看到该观测后才出现的标准化创新 residual”。分类器聚合的是 stopped martingale signature，而不是原始 mask、时间戳密度或采样概率。

如果某个观测真的携带新的状态信息，它会表现为相对于历史预测的 innovation；如果某个信号主要来自“为什么此刻被测”，它无法在非预期创新中长期保持稳定，因为不同采样策略只是改变停时规则，而不应改变鞅创新的零均值性质。

这与当前“采样解耦/反事实干预”框架的结合方式非常自然：

- value process 学习连续状态的 one-step predictive distribution；
- sampling process 不再输出 hazard、density ratio 或 policy residual，而是输出一组可解释的 stopping recipes；
- counterfactual intervention 不生成一致性增强视图，而是生成不同停时规则，用于检验 stopped innovations 是否仍满足鞅矩条件；
- classifier 只能使用 previsible query 加权后的 innovation 累积量，因此采样节奏本身难以直接成为分类捷径。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通 query embedding 改为 Previsible Martingale Query Encoder

OS-MQ 的 Encoder 在事件级别工作。每个观测事件为 `(t_i, d_i, x_i)`，其中 `d_i` 是变量 id，`x_i` 是观测值。模型显式区分两类信息：

1. **Previsible Query `q_i`**
   - 只由 `history before t_i`、变量 id、时间间隔和当前待查询变量构造。
   - 不能访问当前观测值 `x_i`。
   - 吸收 QuITE 的 query-token 思想，但加上“可预测性约束”：query 必须在当前观测发生前就确定。

2. **Continuous-discrete Innovation `eps_i`**
   - value predictor 根据历史预测当前观测分布：

```text
mu_i, sigma_i = Predictor(history before t_i, d_i, t_i)
eps_i = (x_i - mu_i) / sigma_i
```

   - 这里吸收 SDEVI 的连续-离散思路：不补齐整条路径，而是在离散观测点上预测由连续状态诱导的条件分布。
   - `eps_i` 应近似鞅差分：给定历史，期望为 0。

3. **Martingale Query Token `m_i`**

```text
m_i = q_i * clip(eps_i, -c, c)
```

   - 分类器聚合 `sum_i m_i`、`sum_i m_i^2` 和少量自归一化统计。
   - mask、采样密度、policy id 不进入分类头。

### 2.2 改 Loss：从不变性约束转向 Optional-Stopping Moment Control

总目标：

```text
L = L_cls
  + lambda_innov * L_innovation_calibration
  + lambda_stop  * L_optional_stopping
  + lambda_pred  * L_predictability_barrier
```

#### A. 分类损失 `L_cls`

分类器只接收 stopped martingale signature：

```text
M_tau = sum_{i <= tau} q_i * eps_i
V_tau = sum_{i <= tau} (q_i * eps_i)^2
logits = Classifier([M_tau, log(1 + V_tau)])
L_cls = CE(logits, y)
```

这里 `tau` 是事实观测终点，也可以是由反事实采样模块给出的停时 recipe。重点是分类输入不是完整原始序列表征，而是创新鞅的停止统计。

#### B. Innovation Calibration `L_innovation_calibration`

保证 `eps_i` 真正像条件创新，而不是任意 embedding：

```text
L_innovation_calibration =
  mean_buckets mean(eps)^2
  + mean_buckets (var(eps) - 1)^2
```

bucket 可以按变量、时间窗或预测不确定性分组。该项不是 reconstruction-error pseudo observation，也不做 Wasserstein 对齐；它只要求预测残差在历史条件下零均值、单位尺度。

#### C. Optional-Stopping Moment Loss `L_optional_stopping`

由当前反事实干预模块生成一组 stopping recipes，例如：

- `stop_after_first_abnormal`：某变量第一次异常后停止或密集复测；
- `budgeted_panel_stop`：每个变量组最多保留固定数量观测；
- `late_followup_stop`：只观察报警后的晚期窗口；
- `routine_round_stop`：模拟固定查房时间点截断。

对每个 recipe 得到 stopped martingale `M_tau^r`，不要求不同 recipe 的 representation 或 logits 相同，只要求 stopped martingale 的批量矩条件不发生系统性漂移：

```text
L_optional_stopping = sum_r || mean_batch(M_tau^r) ||_2^2
                    + sum_r || mean_batch(M_tau^r / sqrt(V_tau^r + eps)) ||_2^2
```

直觉：若 `M` 是相对于历史过滤的真鞅，则在合适的有界停时下 `E[M_tau] = E[M_0]`。一旦模型把“被测时刻”本身编码进创新，某些停时 recipe 会让 `M_tau` 出现稳定偏移，该损失会把这条捷径压掉。

#### D. Predictability Barrier `L_predictability_barrier`

防止 query 偷看当前值。实现上有两层：

1. 架构屏障：`q_i` 只能从 `history_state.detach_current()`、`delta_t`、`var_id` 生成。
2. 训练屏障：用一个小 probe 尝试从 `q_i` 预测当前标准化创新 `eps_i`，并惩罚可预测性过强：

```text
L_predictability_barrier = corr(Probe(q_i), stopgrad(eps_i))^2
```

这不是 policy adversarial，因为 probe 不是预测环境或采样策略；它只检查“当前观测值是否提前泄漏进 query”。

### 2.3 改 Dataloader：返回 stopping recipes，而不是 mask 增强或平滑样本

新增 `StoppingRecipeCollator`，每个 batch 返回：

1. `event_value`：按时间排序的观测值。
2. `event_time`：事件时间戳。
3. `event_var_id`：变量 id。
4. `event_mask`：事件 padding mask。
5. `stop_recipe_bank`：一组二值停止矩阵 `[B, R, N]`，表示每个 recipe 保留哪些事件。
6. `history_index`：保证 Encoder 构造 `q_i` 时只能访问 `i` 之前的事件。

与历史提案的关键差异：

- 不估计每个事件的采样概率或危险率。
- 不计算采样测度 density ratio。
- 不要求多个 recipe 下 logits 一致、风险方差小或 certified radius 大。
- 不把 stopping recipe 输入分类器。
- recipe 只用于检验鞅停时矩条件。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `HistoryPredictor`：输出历史状态、当前观测预测均值和尺度。
- 现有 sampling branch 改为 `StoppingRecipeGenerator`：从观测结构构造可解释停时，而不是输出概率、残差槽、税率或测度比。
- 现有 counterfactual intervention 保留为“停时干预”：它决定哪些事件被某种策略看见，但不要求生成完整增强样本。
- 推理阶段只需按事实观测流累积 `M_tau, V_tau` 并分类；如果部署场景存在已知采样规则，可额外报告该规则下的 stopped-moment diagnostic。

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


class EventHistoryEncoder(nn.Module):
    """Causal event encoder for irregular observations."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        value: torch.Tensor,
        event_time: torch.Tensor,
        var_id: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        # value/event_time/var_id/event_mask: [B, N]
        var_h = self.var_embed(var_id)
        event_x = torch.cat(
            [var_h, value.unsqueeze(-1), torch.log1p(event_time).unsqueeze(-1)],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * event_mask.unsqueeze(-1)
        causal_h, _ = self.rnn(event_h)

        # Shift right so the state at i only contains history before event i.
        pre_h = torch.zeros_like(causal_h)
        pre_h[:, 1:] = causal_h[:, :-1]
        return pre_h


class PrevisibleQuery(nn.Module):
    """Build query tokens before seeing the current observation value."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        pre_history: torch.Tensor,
        event_time: torch.Tensor,
        var_id: torch.Tensor,
    ) -> torch.Tensor:
        var_h = self.var_embed(var_id)
        delta_t = torch.zeros_like(event_time)
        delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
        query_x = torch.cat(
            [pre_history, var_h, torch.log1p(delta_t).unsqueeze(-1)],
            dim=-1,
        )
        return self.net(query_x)


class InnovationHead(nn.Module):
    """Predict conditional mean and scale for the current event."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        pre_history: torch.Tensor,
        event_time: torch.Tensor,
        var_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var_h = self.var_embed(var_id)
        pred_x = torch.cat(
            [pre_history, var_h, torch.log1p(event_time).unsqueeze(-1)],
            dim=-1,
        )
        raw = self.net(pred_x)
        mu = raw[..., 0]
        sigma = F.softplus(raw[..., 1]) + 1e-3
        return mu, sigma


def stopped_martingale_signature(
    query: torch.Tensor,
    innovation: torch.Tensor,
    event_mask: torch.Tensor,
    stop_mask: torch.Tensor | None = None,
    clip_value: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stopped martingale M, quadratic variation V, and active mask."""

    # query: [B, N, H], innovation/event_mask: [B, N]
    active = event_mask
    if stop_mask is not None:
        active = active * stop_mask
    eps = innovation.clamp(-clip_value, clip_value) * active
    mart_step = query * eps.unsqueeze(-1)
    m_tau = mart_step.sum(dim=1)
    v_tau = mart_step.pow(2).sum(dim=1)
    return m_tau, v_tau, active


class OptionalStoppingMartingaleQueries(nn.Module):
    """Classify irregular time series through previsible martingale innovations."""

    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.history = EventHistoryEncoder(num_vars, hidden_dim)
        self.query = PrevisibleQuery(num_vars, hidden_dim)
        self.innov = InnovationHead(num_vars, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.query_probe = nn.Linear(hidden_dim, 1)

    def encode_events(self, batch: dict) -> dict:
        pre_h = self.history(
            value=batch["event_value"],
            event_time=batch["event_time"],
            var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
        )
        query = self.query(
            pre_history=pre_h,
            event_time=batch["event_time"],
            var_id=batch["event_var_id"],
        )
        mu, sigma = self.innov(
            pre_history=pre_h,
            event_time=batch["event_time"],
            var_id=batch["event_var_id"],
        )
        innovation = (batch["event_value"] - mu) / sigma
        return {
            "pre_history": pre_h,
            "query": query,
            "mu": mu,
            "sigma": sigma,
            "innovation": innovation,
        }

    def forward(self, batch: dict, stop_mask: torch.Tensor | None = None) -> dict:
        enc = self.encode_events(batch)
        m_tau, v_tau, active = stopped_martingale_signature(
            query=enc["query"],
            innovation=enc["innovation"],
            event_mask=batch["event_mask"],
            stop_mask=stop_mask,
        )
        signature = torch.cat([m_tau, torch.log1p(v_tau)], dim=-1)
        logits = self.classifier(signature)
        enc.update({
            "logits": logits,
            "m_tau": m_tau,
            "v_tau": v_tau,
            "active_mask": active,
        })
        return enc

    def innovation_calibration_loss(self, innovation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        eps_mean = masked_mean(innovation, mask, dim=1)
        eps2_mean = masked_mean(innovation.pow(2), mask, dim=1)
        return eps_mean.pow(2).mean() + (eps2_mean - 1.0).pow(2).mean()

    def optional_stopping_loss(
        self,
        batch: dict,
        encoded: dict,
    ) -> torch.Tensor:
        losses = []
        for stop_mask in batch["stop_recipe_bank"].unbind(dim=1):
            m_tau, v_tau, _ = stopped_martingale_signature(
                query=encoded["query"],
                innovation=encoded["innovation"],
                event_mask=batch["event_mask"],
                stop_mask=stop_mask,
            )
            normalized = m_tau / torch.sqrt(v_tau + 1e-4)
            losses.append(m_tau.mean(dim=0).pow(2).mean())
            losses.append(normalized.mean(dim=0).pow(2).mean())
        return torch.stack(losses).mean()

    def predictability_barrier_loss(self, encoded: dict, mask: torch.Tensor) -> torch.Tensor:
        pred_eps = self.query_probe(encoded["query"]).squeeze(-1)
        eps = encoded["innovation"].detach()

        pred_eps = pred_eps - masked_mean(pred_eps, mask, dim=1).unsqueeze(-1)
        eps = eps - masked_mean(eps, mask, dim=1).unsqueeze(-1)

        cov = masked_mean(pred_eps * eps, mask, dim=1)
        pred_var = masked_mean(pred_eps.pow(2), mask, dim=1)
        eps_var = masked_mean(eps.pow(2), mask, dim=1)
        corr = cov / torch.sqrt(pred_var * eps_var + 1e-6)
        return corr.pow(2).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_innov: float = 0.1,
        lambda_stop: float = 0.2,
        lambda_pred: float = 0.05,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]

        cls_loss = F.cross_entropy(out["logits"], labels)
        innov_loss = self.innovation_calibration_loss(
            innovation=out["innovation"],
            mask=batch["event_mask"],
        )
        stop_loss = self.optional_stopping_loss(batch, out)
        pred_loss = self.predictability_barrier_loss(out, batch["event_mask"])

        total = (
            cls_loss
            + lambda_innov * innov_loss
            + lambda_stop * stop_loss
            + lambda_pred * pred_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "innovation_loss": innov_loss.detach(),
            "optional_stopping_loss": stop_loss.detach(),
            "predictability_barrier_loss": pred_loss.detach(),
        }


@torch.no_grad()
def build_stopping_recipe_bank(
    event_time: torch.Tensor,
    event_var_id: torch.Tensor,
    event_value: torch.Tensor,
    event_mask: torch.Tensor,
    num_vars: int,
) -> torch.Tensor:
    """Create interpretable stopping recipes for optional-stopping moment checks."""

    bsz, num_events = event_time.shape
    recipes = []

    # Recipe 1: budgeted observations per variable.
    budget_mask = torch.zeros_like(event_mask)
    for var_idx in range(num_vars):
        var_event = (event_var_id == var_idx) & (event_mask > 0)
        order = var_event.cumsum(dim=1)
        budget_mask = torch.where(var_event & (order <= 2), torch.ones_like(budget_mask), budget_mask)
    recipes.append(budget_mask)

    # Recipe 2: late-window follow-up.
    horizon = event_time.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    late_mask = ((event_time / horizon) >= 0.66).to(event_mask.dtype) * event_mask
    recipes.append(late_mask)

    # Recipe 3: first large absolute deviation per sample.
    centered = event_value - (event_value * event_mask).sum(dim=1, keepdim=True) / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    abnormal = (centered.abs() >= centered.abs().masked_fill(event_mask == 0, 0.0).mean(dim=1, keepdim=True))
    first_abnormal_rank = (abnormal & (event_mask > 0)).float().cumsum(dim=1)
    abnormal_stop = ((first_abnormal_rank > 0) & (first_abnormal_rank <= 3)).to(event_mask.dtype) * event_mask
    recipes.append(abnormal_stop)

    # Recipe 4: routine round approximation by keeping every other event.
    alternating = ((torch.arange(num_events, device=event_time.device)[None, :] % 2) == 0).to(event_mask.dtype)
    recipes.append(alternating.expand(bsz, -1) * event_mask)

    return torch.stack(recipes, dim=1)
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `alarm-followup shift`：报警后密集复测变为延迟复测。
   - `budgeted variable shift`：某些变量在测试策略下只保留前几次观测。
   - `routine-round shift`：从事件触发采样改为固定查房式采样。
   - `stop-after-normal shift`：连续正常后停止测量的规则在训练/测试中反转。

2. **对比方法**
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy consistency / contrastive augmentation。
   - hazard/null-space baseline。
   - commutator graph surgery baseline。
   - protocol-tax evidence market baseline。
   - posterior quotient dynamics baseline。
   - do-sampler certified smoothing baseline。
   - do-measure doubly robust risk baseline。
   - QuITE 普通 query adapter 与 SDEVI 普通连续-离散分类版本。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - stopped-moment drift：`||mean(M_tau)||`。
   - self-normalized stopped drift：`||mean(M_tau / sqrt(V_tau))||`。
   - innovation calibration error。
   - sampling shortcut score：用停时 recipe 改变后，错误样本的 stopped drift 是否显著升高。

4. **消融实验**
   - 去掉 previsible query shift-right，让 query 可以看到当前值，验证是否出现采样泄漏。
   - 去掉 `L_optional_stopping`，验证停时偏移下鲁棒性下降。
   - 去掉 `L_innovation_calibration`，验证 innovation 是否退化为普通 embedding。
   - 将 stopping recipes 替换成随机 mask，验证收益来自停时结构而非普通增强。
   - 只用 `M_tau` 或只用 `V_tau` 分类，检查鞅均值与二次变差分别提供的稳健信息。

## 5. 预期创新性

1. **从采样概率建模转向停时理论建模**：不问某点为什么被采样，而问在该采样规则作为 stopping time 时，分类创新是否仍满足鞅停时矩条件。
2. **从普通 query token 转向 previsible martingale query**：吸收 QuITE 的轻量 query 适配优势，但用历史可预测性阻断当前观测与采样时刻的直接泄漏。
3. **从连续-离散生成建模转向创新校准**：吸收 SDEVI 对连续状态诱导离散观测分布的重视，但不做后验商、路径补全或采样似然分解。
4. **从反事实增强转向停时诊断**：counterfactual intervention 只产生 stopping recipes，用于 optional-stopping moment control，而不是一致性、风险方差、平滑认证或测度校正。
5. **审稿卖点清晰**：采样策略偏移被形式化为 optional sampling rule shift；方法提供可解释的 stopped-moment drift 指标，能直接说明模型是否把“何时被测/何时停止测”当成类别证据。

## 6. 一句话投稿卖点

**OS-MQ 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样停时污染创新鞅”的问题，并通过 previsible query、连续-离散标准化创新与 optional-stopping moment control，让分类器依赖跨停时稳定的状态创新，而不是依赖训练环境中特定的测量/停止规则。**
