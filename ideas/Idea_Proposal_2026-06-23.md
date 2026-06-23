# Title: Policy-Gauge Horizontal Transport：面向采样策略偏移的规范固定横向传输分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取 `paper_daily.md` 与 `paper_daily_2026-06-12.md`。
- 已读取当前工作区内全部历史提案：
  - `ideas/Idea_Proposal_2026-06-12.md`
  - `ideas/Idea_Proposal_2026-06-13.md`
  - `ideas/Idea_Proposal_2026-06-14.md`
  - `ideas/Idea_Proposal_2026-06-16.md`
  - `ideas/Idea_Proposal_2026-06-19.md`
  - `ideas/Idea_Proposal_2026-06-21.md`
  - `ideas/Idea_Proposal_2026-06-22.md`
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案核心机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`

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
21. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing，或 MedMamba 的 frequency/adaptive graph branch 作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗，不要求多视图表征一致，不做图交换子、后验除法、测度比、停时矩或拓扑审查；而是把采样政策视为同一潜在状态轨迹上的“观测坐标规约”。模型学习一个事件级低秩 gauge frame，将由采样坐标改变诱发的垂直位移从状态传输中投影出去，分类器只消费横向传输 signature。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时序中的偏移常常并不是“观测值变了”，而是“同一条潜在状态轨迹被哪种坐标方式看见了”：

- ICU 医院 A 把报警后的短时间复测压缩成 dense burst，医院 B 只保留轮班时刻；
- 可穿戴设备 A 用电量策略拉长夜间采样间隔，设备 B 保持均匀低频；
- 某实验室 panel 在训练机构中同步出现，在测试机构中被拆成多个异步事件。

这些变化会改变事件密度、跨通道同步机会、局部时间尺度与 token 排列，但不应改变底层状态本身。若模型直接在 token mixing、SSM hidden state 或 learned path 上分类，就可能把这种“坐标规约差异”误当成类别证据。

近期 `paper_daily.md` 中的机制给出三个关键启发：

1. **MTM** 指出 channel-wise asynchrony 会让跨通道 attention 名义存在、实际失效；但其 pivotal token selection 仍可能被采样政策改变的同步窗口牵引。
2. **MedMamba** 说明 SSM 主干适合长医疗时序，并强调 raw/difference 视图处理非平稳性；但 adaptive graph 与 frequency branch 不是为采样政策偏移设计，可能吸收设备或医院协议。
3. **FlowPath** 提醒我们离散观测诱导的连续路径几何很重要；但可学习路径若不区分状态几何与采样坐标几何，仍会混入策略偏差。

Policy-Gauge Horizontal Transport (PGHT) 的核心直觉是：

> 采样政策不是一个要被分类器识别的环境标签，也不是一个要被估计概率的缺失机制；它更像同一潜在状态流形上的坐标规约。训练医院选择了一种事件坐标，测试医院换了另一种事件坐标。我们不让分类器直接消费坐标相关的 event displacement，而是学习一个局部 gauge frame，把“换坐标造成的垂直位移”剥离出去，只保留沿状态流形横向传输的可分类信息。

这与当前“采样解耦/反事实干预”框架的结合方式自然：

- value process 负责把观测值事件提升到 latent event states；
- sampling process 不输出 hazard、density ratio、policy residual、tax、posterior factor 或 censor envelope，而是输出 **chart descriptor**：当前事件坐标的局部密度、同步度、变量偏置和时间伸缩；
- counterfactual intervention 不生成一致性 pair，也不做风险方差，而是生成已知的 **chart moves**，用于告诉模型哪些 latent displacement 是坐标改变引起的；
- classifier 只接收 horizontal transport signature，因此难以直接利用采样密度、事件排列、panel 同步机会或局部复测 burst 作为捷径。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从 token mixing/图分支改为事件级 Gauge-Fixed Horizontal Transport

给定事件流 `(t_i, d_i, x_i)`，先将每个事件提升为 latent event state：

```text
z_i = EventLift(x_i, d_i, log(1 + delta_t_i))
delta z_i = z_i - z_{i-1}
```

PGHT 不直接把 `z_i` 或 `delta z_i` 输入分类头，而是在每个事件处学习一个低秩规范方向：

```text
U_i = ChartConnection(chart_i) in R^{H x K},  K << H
```

其中 `chart_i` 只描述观测坐标，不含标签、不含当前类别预测，例如：

- 局部事件密度；
- 当前变量的近期观测预算；
- 当前事件是否处在 panel-like 同步簇中；
- 前后时间间隔的伸缩比例；
- channel-wise asynchrony score。

`U_i` 的列向量张成 **vertical gauge subspace**，表示“如果只改变采样坐标，这个事件 latent displacement 最可能沿哪些方向移动”。横向投影为：

```text
P_i = I - U_i (U_i^T U_i + eps I)^{-1} U_i^T
h_i = P_i delta z_i
```

分类主干只消费 `h_i`：

```text
s = HorizontalSSM(h_1, h_2, ..., h_N)
logits = Classifier(s)
```

关键差异：

- 不学习 reference points；
- 不做 graph edge 分离；
- 不估计采样概率；
- 不把 missingness pattern 或 policy code 拼入分类器；
- 不要求不同采样视图的 hidden states 相同；
- 不用 frequency branch、prototype、posterior quotient 或 topology capsule。

### 2.2 改 Loss：从一致性/对抗转向 Gauge Chart Span + Vertical Blindness

总目标：

```text
L = L_cls
  + lambda_span  * L_chart_span
  + lambda_blind * L_vertical_blind
  + lambda_frame * L_frame_sobriety
```

#### A. 分类损失 `L_cls`

只使用事实观测坐标下的横向传输：

```text
L_cls = CE(Classifier(HorizontalSSM({P_i delta z_i})), y)
```

#### B. Chart Span Loss `L_chart_span`

当前反事实干预模块生成一组 **已知 chart moves**，例如：

- `burst_split_merge`：把一段高频复测 burst 合并或拆开；
- `panel_batching`：把多个异步变量事件压缩为近同步 panel；
- `channel_offset`：对某些变量施加固定采样延迟；
- `window_dilation`：拉伸早期或晚期观测时间坐标。

这些 chart moves 的目标不是制造正样本，而是提供“哪些变化来自坐标规约”的监督。对同一条样本的事实坐标和反事实 chart 坐标，计算两者 latent displacement 的差：

```text
r_i^c = delta z_i^{chart c} - delta z_i^{factual}
```

如果 `r_i^c` 确实来自采样坐标改变，它应落在 vertical gauge frame 中，横向残差应小：

```text
L_chart_span = mean_c,i || P_i^c r_i^c ||_2^2
```

这不是跨视图 representation consistency：我们不要求 `z^c = z`、`h^c = h` 或 `logits^c = logits`。我们只要求“由已知 chart move 诱发的位移”被 gauge frame 解释为垂直坐标位移，而不是污染横向状态传输。

#### C. Vertical Blindness Loss `L_vertical_blind`

为了防止分类器仍偷偷利用 vertical directions，PGHT 在 latent displacement 层做局部规范扰动：

```text
epsilon_i = U_i a_i,  a_i ~ Normal(0, sigma^2 I)
logits      = f({P_i delta z_i})
logits_vert = f({P_i (delta z_i + epsilon_i)})
```

由于 `epsilon_i` 位于 vertical subspace，经过横向投影后理论上应被消掉：

```text
L_vertical_blind = KL(stopgrad(softmax(logits)) || softmax(logits_vert))
```

这不是 policy adversarial，也不是 hazard-score 梯度正交；它不训练模型预测/欺骗策略标签，只检查分类路径对局部坐标规约方向是否盲视。

#### D. Frame Sobriety `L_frame_sobriety`

若 `U_i` 退化为全空间，模型会把所有信息都投掉；若 `U_i` 塌缩为零，chart span 无法解释采样坐标位移。因此加入低秩规范约束：

```text
L_frame_sobriety =
  || U_i^T U_i - I ||_F^2
  + relu(vertical_energy_ratio - r_max)^2
  + relu(r_min - horizontal_energy_ratio)^2
```

它不是 protocol tax 或 evidence budget；它只约束 gauge frame 的几何条件数与横向信息保留比例，保证 PGHT 不通过删除所有动态来获得虚假鲁棒性。

### 2.3 改 Dataloader：返回 Chart Atlas Bank，而不是停时、审查或平滑样本

新增 `ChartAtlasCollator`，每个 batch 返回：

1. `event_value`：按时间排序的观测值。
2. `event_time`：连续时间戳。
3. `event_var_id`：变量 id。
4. `event_mask`：事件 padding mask。
5. `chart_descriptor`：事实坐标下的局部采样坐标描述。
6. `chart_recipe_bank`：反事实 chart moves 的 one-hot 或连续参数。
7. `chart_visibility_bank`：每个 chart move 后哪些事件被保留、合并或偏移。
8. `chart_descriptor_bank`：每个 chart move 后重新计算的局部坐标描述。

与历史机制区别：

- 不是 hazard resampling；
- 不是 policy-simplex smoothing；
- 不是 stopping recipe；
- 不是 censor recipe；
- 不是 density-ratio target；
- 不是 token 税务账本；
- 不要求 chart views 之间 logits 或 representation 一致。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `EventLift + HorizontalTransportEncoder`。
- 现有 sampling branch 改为 `ChartConnection`：只输出局部 gauge frame `U_i`，不进入分类头。
- 现有 counterfactual intervention 改为 `ChartAtlasBank`：生成已知采样坐标变换，用于训练 `U_i` 跨越坐标诱发位移。
- 推理阶段无需策略标签或反事实视图；只根据事实观测计算 `chart_descriptor`、`U_i` 和横向传输 signature。
- 可解释输出包括：
  - 每个事件的 vertical energy ratio；
  - 每个变量/时间窗的 horizontal contribution；
  - 哪些预测依赖低横向能量、高坐标能量的可疑片段。

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


def event_delta_time(event_time: torch.Tensor) -> torch.Tensor:
    delta_t = torch.zeros_like(event_time)
    delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
    if event_time.size(1) > 1:
        delta_t[:, 0] = delta_t[:, 1]
    return delta_t


class EventLift(nn.Module):
    """Lift irregular observation events into latent event states."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        event_value: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = event_delta_time(event_time)
        var_h = self.var_embed(event_var_id)
        event_x = torch.cat(
            [
                var_h,
                event_value.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(event_x) * event_mask.unsqueeze(-1)


class ChartDescriptorBuilder(nn.Module):
    """Compute local sampling-chart descriptors without using labels."""

    def __init__(self, num_vars: int, descriptor_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.proj = nn.Sequential(
            nn.Linear(num_vars + 5, descriptor_dim),
            nn.SiLU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

    def forward(
        self,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        # All features describe observation coordinates, not observed values.
        delta_t = event_delta_time(event_time)
        horizon = event_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = event_time / horizon

        local_gap = torch.log1p(delta_t)
        local_density = 1.0 / (1.0 + delta_t)
        early = (time_norm <= 0.33).to(event_time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(event_time.dtype)
        late = (time_norm > 0.66).to(event_time.dtype)

        var_onehot = F.one_hot(event_var_id.clamp_min(0), self.num_vars).to(event_time.dtype)
        chart_x = torch.cat(
            [
                var_onehot,
                time_norm.unsqueeze(-1),
                local_gap.unsqueeze(-1),
                local_density.unsqueeze(-1),
                early.unsqueeze(-1) + 0.5 * middle.unsqueeze(-1),
                late.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.proj(chart_x) * event_mask.unsqueeze(-1)


class ChartConnection(nn.Module):
    """Map chart descriptors to a low-rank vertical gauge frame."""

    def __init__(self, descriptor_dim: int, hidden_dim: int, gauge_rank: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gauge_rank = gauge_rank
        self.to_frame = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * gauge_rank),
        )

    def forward(self, chart_descriptor: torch.Tensor) -> torch.Tensor:
        # Return U: [B, N, H, K].
        raw = self.to_frame(chart_descriptor)
        frame = raw.view(*chart_descriptor.shape[:2], self.hidden_dim, self.gauge_rank)

        # Stable local orthonormalization. K is small, so QR is cheap.
        bsz, num_events, hidden_dim, gauge_rank = frame.shape
        flat = frame.reshape(bsz * num_events, hidden_dim, gauge_rank)
        q, _ = torch.linalg.qr(flat, mode="reduced")
        return q.reshape(bsz, num_events, hidden_dim, gauge_rank)


def project_horizontal(delta_z: torch.Tensor, gauge_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project event displacement away from the vertical gauge frame."""

    # delta_z: [B, N, H], gauge_frame: [B, N, H, K]
    coeff = torch.einsum("bnhk,bnh->bnk", gauge_frame, delta_z)
    vertical = torch.einsum("bnhk,bnk->bnh", gauge_frame, coeff)
    horizontal = delta_z - vertical
    return horizontal, vertical


class HorizontalTransportBackbone(nn.Module):
    """A lightweight SSM-style recurrent backbone over horizontal transports."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, horizontal: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        bsz, num_events, hidden_dim = horizontal.shape
        state = torch.zeros(bsz, hidden_dim, device=horizontal.device, dtype=horizontal.dtype)
        for idx in range(num_events):
            step = self.in_proj(horizontal[:, idx])
            mix = self.gate(torch.cat([state, step], dim=-1))
            candidate = self.update(step, state)
            state = torch.where(
                event_mask[:, idx : idx + 1] > 0,
                mix * candidate + (1.0 - mix) * state,
                state,
            )
        return state


class PolicyGaugeHorizontalTransport(nn.Module):
    """Classify irregular time series through gauge-fixed horizontal transport."""

    def __init__(
        self,
        num_vars: int,
        descriptor_dim: int,
        hidden_dim: int,
        gauge_rank: int,
        num_classes: int,
    ):
        super().__init__()
        self.lift = EventLift(num_vars=num_vars, hidden_dim=hidden_dim)
        self.chart = ChartDescriptorBuilder(num_vars=num_vars, descriptor_dim=descriptor_dim)
        self.connection = ChartConnection(
            descriptor_dim=descriptor_dim,
            hidden_dim=hidden_dim,
            gauge_rank=gauge_rank,
        )
        self.backbone = HorizontalTransportBackbone(hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_transport(
        self,
        event_value: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
        chart_descriptor: torch.Tensor | None = None,
    ) -> dict:
        z = self.lift(event_value, event_time, event_var_id, event_mask)
        delta_z = torch.zeros_like(z)
        delta_z[:, 1:] = z[:, 1:] - z[:, :-1]

        if chart_descriptor is None:
            chart_descriptor = self.chart(event_time, event_var_id, event_mask)
        gauge_frame = self.connection(chart_descriptor)
        horizontal, vertical = project_horizontal(delta_z, gauge_frame)
        signature = self.backbone(horizontal, event_mask)

        return {
            "z": z,
            "delta_z": delta_z,
            "chart_descriptor": chart_descriptor,
            "gauge_frame": gauge_frame,
            "horizontal": horizontal,
            "vertical": vertical,
            "signature": signature,
        }

    def forward(self, batch: dict) -> dict:
        enc = self.encode_transport(
            event_value=batch["event_value"],
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
            chart_descriptor=batch.get("chart_descriptor"),
        )
        logits = self.classifier(enc["signature"])
        enc["logits"] = logits
        return enc

    def classify_from_delta(
        self,
        delta_z: torch.Tensor,
        gauge_frame: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        horizontal, _ = project_horizontal(delta_z, gauge_frame)
        signature = self.backbone(horizontal, event_mask)
        return self.classifier(signature)

    def chart_span_loss(self, batch: dict, factual: dict) -> torch.Tensor:
        losses = []
        for cf_value, cf_time, cf_var, cf_mask, cf_chart in zip(
            batch["chart_event_value_bank"].unbind(dim=1),
            batch["chart_event_time_bank"].unbind(dim=1),
            batch["chart_event_var_bank"].unbind(dim=1),
            batch["chart_event_mask_bank"].unbind(dim=1),
            batch["chart_descriptor_bank"].unbind(dim=1),
        ):
            cf = self.encode_transport(cf_value, cf_time, cf_var, cf_mask, cf_chart)

            # Align by padded event index; practical code can replace this with
            # explicit event-id alignment from the collator.
            min_len = min(cf["delta_z"].size(1), factual["delta_z"].size(1))
            chart_residual = cf["delta_z"][:, :min_len] - factual["delta_z"][:, :min_len].detach()
            cf_frame = cf["gauge_frame"][:, :min_len]
            cf_mask = cf_mask[:, :min_len]

            horizontal_residual, _ = project_horizontal(chart_residual, cf_frame)
            raw = horizontal_residual.pow(2).sum(dim=-1)
            losses.append((raw * cf_mask).sum() / cf_mask.sum().clamp_min(1.0))
        return torch.stack(losses).mean()

    def vertical_blindness_loss(
        self,
        factual: dict,
        event_mask: torch.Tensor,
        sigma: float = 0.15,
    ) -> torch.Tensor:
        gauge_frame = factual["gauge_frame"]
        noise_coeff = torch.randn(
            *gauge_frame.shape[:2],
            gauge_frame.size(-1),
            device=gauge_frame.device,
            dtype=gauge_frame.dtype,
        ) * sigma
        vertical_noise = torch.einsum("bnhk,bnk->bnh", gauge_frame, noise_coeff)

        clean_logits = self.classify_from_delta(
            factual["delta_z"],
            gauge_frame,
            event_mask,
        )
        noisy_logits = self.classify_from_delta(
            factual["delta_z"] + vertical_noise,
            gauge_frame,
            event_mask,
        )
        clean_prob = F.softmax(clean_logits.detach(), dim=-1)
        return F.kl_div(
            F.log_softmax(noisy_logits, dim=-1),
            clean_prob,
            reduction="batchmean",
        )

    def frame_sobriety_loss(self, factual: dict, event_mask: torch.Tensor) -> torch.Tensor:
        frame = factual["gauge_frame"]
        gram = torch.einsum("bnhk,bnhl->bnkl", frame, frame)
        eye = torch.eye(frame.size(-1), device=frame.device, dtype=frame.dtype)
        ortho = (gram - eye).pow(2).sum(dim=(-1, -2))
        ortho = (ortho * event_mask).sum() / event_mask.sum().clamp_min(1.0)

        h_energy = factual["horizontal"].pow(2).sum(dim=-1)
        v_energy = factual["vertical"].pow(2).sum(dim=-1)
        total = h_energy + v_energy + 1e-6
        vertical_ratio = (v_energy / total * event_mask).sum() / event_mask.sum().clamp_min(1.0)
        horizontal_ratio = (h_energy / total * event_mask).sum() / event_mask.sum().clamp_min(1.0)

        ratio_loss = F.relu(vertical_ratio - 0.65).pow(2) + F.relu(0.20 - horizontal_ratio).pow(2)
        return ortho + ratio_loss

    def training_loss(
        self,
        batch: dict,
        lambda_span: float = 0.3,
        lambda_blind: float = 0.15,
        lambda_frame: float = 0.05,
    ) -> dict:
        factual = self.forward(batch)
        cls_loss = F.cross_entropy(factual["logits"], batch["labels"])
        span_loss = self.chart_span_loss(batch, factual)
        blind_loss = self.vertical_blindness_loss(factual, batch["event_mask"])
        frame_loss = self.frame_sobriety_loss(factual, batch["event_mask"])

        total = cls_loss + lambda_span * span_loss + lambda_blind * blind_loss + lambda_frame * frame_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "chart_span_loss": span_loss.detach(),
            "vertical_blindness_loss": blind_loss.detach(),
            "frame_sobriety_loss": frame_loss.detach(),
        }


@torch.no_grad()
def build_chart_atlas_bank(
    event_value: torch.Tensor,
    event_time: torch.Tensor,
    event_var_id: torch.Tensor,
    event_mask: torch.Tensor,
    num_vars: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create chart-move views for gauge-frame training.

    This is a sketch. Production code should preserve event ids so chart_span_loss
    can align factual and chart-moved events exactly.
    """

    bsz, num_events = event_time.shape
    device = event_time.device
    values, times, var_ids, masks = [], [], [], []

    # Chart 1: burst split/merge by keeping alternating high-density events.
    alternating = ((torch.arange(num_events, device=device)[None, :] % 2) == 0).to(event_mask.dtype)
    burst_mask = alternating.expand(bsz, -1) * event_mask
    values.append(event_value * burst_mask)
    times.append(event_time)
    var_ids.append(event_var_id)
    masks.append(burst_mask)

    # Chart 2: late-window dilation.
    horizon = event_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = event_time / horizon
    late = (time_norm > 0.66).to(event_time.dtype)
    dilated_time = event_time * (1.0 + 0.25 * late)
    times.append(dilated_time)
    values.append(event_value * event_mask)
    var_ids.append(event_var_id)
    masks.append(event_mask)

    # Chart 3: channel offset for odd variable ids.
    odd_var = (event_var_id % 2 == 1).to(event_time.dtype)
    mean_gap = event_delta_time(event_time).sum(dim=1, keepdim=True) / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    offset_time = event_time + 0.5 * mean_gap * odd_var
    times.append(offset_time)
    values.append(event_value * event_mask)
    var_ids.append(event_var_id)
    masks.append(event_mask)

    # Chart 4: panel batching approximation by snapping nearby times to coarse bins.
    coarse_time = torch.round(time_norm * 8.0) / 8.0 * horizon
    times.append(coarse_time)
    values.append(event_value * event_mask)
    var_ids.append(event_var_id)
    masks.append(event_mask)

    return (
        torch.stack(values, dim=1),
        torch.stack(times, dim=1),
        torch.stack(var_ids, dim=1),
        torch.stack(masks, dim=1),
    )
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `burst-coordinate shift`：训练环境保留报警后 dense burst，测试环境合并 burst。
   - `panel-chart shift`：训练环境变量同步 panel 化，测试环境拆成异步事件。
   - `channel-offset shift`：某些变量在测试设备中系统性延迟。
   - `window-dilation shift`：早期或晚期时间坐标被设备策略拉伸。

2. **对比方法**
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - DHN hazard/null-space baseline。
   - CGS commutator graph surgery baseline。
   - PT-AEM evidence market baseline。
   - PQD posterior quotient baseline。
   - DS-CS certified smoothing baseline。
   - DM-DRR measure-ratio baseline。
   - OS-MQ optional-stopping martingale baseline。
   - CETC censored topology baseline。
   - MTM token mixing 与 MedMamba SSM 主干。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - chart-shift calibration error。
   - vertical energy reliance：错误预测是否依赖高 vertical ratio 片段。
   - horizontal retention ratio：鲁棒提升是否来自保留状态传输，而不是简单删除信息。
   - chart span residual：已知 chart move 是否被 gauge frame 解释。

4. **消融实验**
   - 去掉 `L_chart_span`，检查 gauge frame 是否学不到采样坐标位移。
   - 去掉 `L_vertical_blind`，检查分类头是否重新利用 vertical directions。
   - 将 gauge frame rank 从小到大扫描，验证低秩规范假设。
   - 把 chart moves 替换为随机 mask，验证收益来自坐标规约结构而不是普通增强。
   - 将 horizontal transport 替换为普通 pooled token，验证核心不是额外参数量。

## 5. 预期创新性

1. **从采样机制估计转向观测坐标规范化**：不问某事件为什么被采样，而问这次采样政策把潜在状态轨迹写成了哪种观测坐标。
2. **从 token selection 转向 horizontal transport**：吸收 MTM 对 channel-wise asynchrony 的洞察，但不选择 pivotal tokens；分类证据来自被 gauge-fixing 后的状态传输。
3. **从 SSM/graph/frequency 分支转向 gauge-fixed SSM**：吸收 MedMamba 中 SSM 适合医疗长序列的优势，但不依赖 frequency branch 或 adaptive graph。
4. **从 path geometry 转向 chart geometry**：吸收 FlowPath 对路径几何的重视，但把采样策略造成的坐标几何作为 vertical gauge subspace 处理。
5. **从多视图一致转向已知 chart move 的 span supervision**：反事实干预只告诉模型哪些位移是采样坐标改变导致的，不要求不同视图表示或 logits 相同。
6. **解释性直接服务部署**：vertical energy ratio 可以标记模型是否依赖坐标规约差异；horizontal retention ratio 可以说明鲁棒性是否来自真实状态传输。

## 6. 一句话投稿卖点

**PGHT 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“同一潜在状态轨迹的观测坐标规约改变”，并通过低秩 policy-gauge frame 与 horizontal transport classifier，让分类器只依赖跨采样坐标稳定的状态传输，而不是依赖训练环境中特定的事件密度、同步窗口、panel batching 或时间伸缩捷径。**
