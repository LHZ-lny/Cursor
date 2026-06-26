# Title: Calendar-Knockoff Causal Firewall：面向采样策略偏移的采样日历敲除因果防火墙

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取 `paper_daily.md`、`paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`。
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
- 已读取自动化记忆中记录但当前工作区未检出的历史提案核心机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`
  - `Idea_Proposal_2026-06-24.md`

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
24. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing、MedMamba 的 frequency/adaptive graph branch、MedSpaformer 的普通 token sparsification 或 MILM 的 value-redacted classifier 作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不删除采样信息，不做对抗、平滑、后验除法、拓扑审查、gauge 投影或纠错码；而是把采样时间戳、变量可见性和 panel 共现结构看成一个“采样日历”。当前反事实模块生成与真实日历条件交换的 knockoff 日历，作为统计负控。模型可以使用日历，但任何无法击败 knockoff 负控的日历证据都被视为不可发表的 causal discovery，并通过 soft knockoff-FDR loss 被防火墙拦截。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列中的策略偏移，本质上经常不是“值分布改变”，而是“哪张采样日历被贴在同一条潜在病程上”改变：

- 医院 A 在报警后安排密集复测，医院 B 只在固定查房窗口复测；
- 机构 A 把多个化验作为 panel 同步下单，机构 B 拆成异步事件；
- 设备 A 因电量策略压低夜间采样频率，设备 B 保留更均匀的低频采样；
- 某些检查项目在训练环境中只对高风险病人出现，在测试环境中变成常规筛查。

近期 `paper_daily.md` 的两个最新机制提供了直接启发：

1. **MILM** 的 value-redacted training 证明采样模式本身可能强预测标签。它的价值在于把 informative sampling 暴露出来；风险在于如果直接利用采样模式，就会把医院协议、检查触发规则和 value-pending 机制学成可迁移疾病证据。
2. **MedSpaformer** 的 token sparsification 强调不同长度、通道和数据集中的 token 可见性差异。它提醒我们，模型选择“看哪些 token”会被采样日历牵引；但普通 sparsification 并不能判断某个可见 token 是真实状态信号，还是训练政策制造的可见性优势。

Calendar-Knockoff Causal Firewall (CKCF) 的核心直觉是：

> 如果一张采样日历真的承载了可迁移状态证据，那么它应当能在与其条件交换的 knockoff 日历面前展示出超出随机负控的解释力；如果真实日历与 knockoff 日历对分类贡献难以区分，模型就没有资格把该日历证据当作因果稳定信号。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 仍编码观测值驱动的状态证据；
- sampling process 不输出 hazard、density ratio、policy residual、tax、syndrome 或 gauge frame，而是生成 **conditional knockoff calendar**；
- counterfactual intervention 不再构造一致性正样本，也不要求多个策略视图 logits 相同，而是构造满足交换性负控的日历对；
- classifier 可以通过一个很窄的 calendar adapter 利用时间/变量可见性，但必须接受 knockoff-FDR 防火墙审计。

这样既不粗暴丢弃 informative sampling，也不允许训练环境中特有的 sampling-policy shortcut 无审计地进入决策边界。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：从 do-view 增强改为 Conditional Calendar Knockoff Bank

每条样本的观测结构被抽象成采样日历：

```text
C = {(t_i, d_i, m_i, panel_i, window_i)}_{i=1}^N
```

新增 `CalendarKnockoffCollator`，为每条真实日历 `C` 生成 knockoff 日历 `C_tilde`，要求它保留不应携带标签的边缘结构，但打破训练环境中特定的 label-policy shortcut：

1. **条件交换性约束**
   - 在 batch 内或同一粗环境内匹配样本；
   - 保留总观测数、变量级观测预算、早/中/晚时间窗比例、粗 panel 大小分布；
   - 对事件时间、变量 id、panel id 做受限置换或重采样；
   - 得到 `C_tilde` 后，`(C, C_tilde)` 在这些低阶统计条件下近似可交换。

2. **不触碰观测值**
   - CKCF 不生成伪值、不做插值、不做 reconstruction error 伪观测；
   - knockoff 只作用于日历特征，即时间、变量可见性、panel 共现、局部密度；
   - value path 保持事实观测，用于防止模型因为 knockoff 失真而丢掉真实状态信号。

3. **分组日历原子**
   - 将日历切成 group atoms，例如 `(变量组, 时间窗)`、`panel group`、`高密度复测簇`；
   - 后续 loss 在 group 级别做 knockoff 统计，而不是 token 级买卖或 token 税务。

### 2.2 改 Encoder：增加 Calendar Adapter，但接入 Knockoff Firewall

CKCF 不要求重写现有 irregular encoder。它在已有 value encoder 外增加一个很窄的日历适配层：

```text
h_value = ValueEncoder(values, times, vars, mask)
a_real  = CalendarAdapter(C)
a_ko    = CalendarAdapter(C_tilde)
z_real  = Fuse(h_value, a_real)
z_ko    = Fuse(h_value, a_ko)
```

关键限制：

- `CalendarAdapter` 只输出低秩 residual，不能单独决定 logits；
- 分类主路径仍以 value representation 为主体；
- knockoff 分支与真实分支共享全部参数；
- 日历 residual 的每个 group contribution 都会被 knockoff-FDR loss 审计。

这与历史机制区别明显：

- 不是 missingness pattern encoder 直接分类，因为日历 residual 必须通过 knockoff 负控；
- 不是 policy adversarial，因为没有策略标签、没有 gradient reversal；
- 不是多视图一致性，因为真实日历和 knockoff 日历的 logits 可以不同，目标不是把它们压成一样；
- 不是 protocol tax/evidence market，因为没有证据购买预算或边际税率；
- 不是 smoothing/certification，因为不在 policy simplex 上随机平均预测。

### 2.3 改 Loss：从不变性/对抗转向 Soft Knockoff-FDR Firewall

总目标：

```text
L = L_cls
  + lambda_fdr   * L_soft_knockoff_fdr
  + lambda_swap  * L_swap_symmetry
  + lambda_rank  * L_low_rank_calendar
```

#### A. 分类损失 `L_cls`

事实预测仍使用真实日历：

```text
L_cls = CE(logits_real, y)
```

CKCF 并不禁止模型利用采样日历。若某种 sampling pattern 是跨策略稳定的状态触发信号，它可以通过后续 knockoff 统计保留下来。

#### B. Soft Knockoff-FDR `L_soft_knockoff_fdr`

对每个日历 group `g`，计算真实日历和 knockoff 日历对真实类别 logit margin 的贡献差：

```text
W_g = margin_g(C) - margin_g(C_tilde)
```

直觉：

- 若 `W_g` 大且为正，说明真实日历 group 比 knockoff 负控更能支持分类；
- 若 `W_g` 接近 0 或负，说明该 group 的贡献可能只是可交换日历噪声；
- 若模型在大量 group 上给出正 `W_g`，但负向 knockoff 对照也很多，则说明日历发现的 false discovery 风险高。

使用可微近似的 knockoff-FDR：

```text
soft_pos(t) = sigmoid((W_g - t) / tau)
soft_neg(t) = sigmoid((-W_g - t) / tau)
FDP_hat(t) = (1 + sum_g soft_neg(t)) / (sum_g soft_pos(t) + eps)
L_soft_knockoff_fdr = mean_t relu(FDP_hat(t) - q)^2 * mean_g soft_pos(t)
```

这不是一致性约束：我们不要求 `logits_real == logits_knockoff`。相反，模型允许真实日历胜过 knockoff，但必须在 group-level 统计上控制“看起来有用的日历原子”中有多少可能只是负控噪声。

#### C. Swap Symmetry `L_swap_symmetry`

Knockoff 理论要求真实/knockoff 对交换后统计量符号翻转。训练时随机交换部分 group 的真实和 knockoff 日历，记录 swap sign `s_g in {-1, +1}`：

```text
W_g_swapped ≈ s_g * W_g_original
L_swap_symmetry = || W_swapped - s * stopgrad(W_original) ||_1
```

这项约束保证防火墙不是任意惩罚日历使用，而是在统计负控语义上工作。它不同于 vertical blindness、HSIC redaction、predictability barrier 或 adversarial loss。

#### D. Low-Rank Calendar Residual `L_low_rank_calendar`

为了防止日历适配层绕开 value path，限制其能量与秩：

```text
L_low_rank_calendar = ||A_calendar||_* + relu(||a_real|| / ||h_value|| - r_max)^2
```

它不是 protocol tax，也不是 evidence budget；它只是工程上保证日历 residual 是“可审计补充”，而不是完整分类器。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 保持为分类主干，输出 `h_value`。
- 现有 sampling branch 改为 `CalendarKnockoffGenerator`：从观测结构中生成条件交换的 `C_tilde`。
- 现有 counterfactual intervention 不再产出多个策略视图，而是产出真实/knockoff/swap 三元组。
- 推理阶段默认只使用真实日历，但同时输出：
  - group-level `W_g`；
  - knockoff-FDR estimate；
  - calendar reliance ratio；
  - 哪些时间窗、变量组、panel 共现通过了 knockoff firewall。

### 2.5 为什么与历史提案显著正交

CKCF 的主创新是 **统计负控 + knockoff-FDR 审计**。它不需要估计采样概率，不做平滑认证，不做一致性对齐，不把策略信息除掉，不做停时鞅、拓扑胶囊、gauge frame、syndrome repair 或 policy shadow。它与 MILM 的关系是：MILM 证明 sampling pattern 有预测力；CKCF 进一步问“这种预测力在 knockoff 负控下是否仍像因果稳定证据”。它与 MedSpaformer 的关系是：MedSpaformer 稀疏选择 informative tokens；CKCF 审计“某个可见性/日历 group 被选中是否只是采样政策带来的假发现”。

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


class CalendarAdapter(nn.Module):
    """Low-rank residual adapter for sampling-calendar features."""

    def __init__(self, num_vars: int, hidden_dim: int, rank: int):
        super().__init__()
        self.num_vars = num_vars
        self.rank = rank
        self.atom_proj = nn.Sequential(
            nn.Linear(num_vars + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank),
        )
        self.basis = nn.Parameter(torch.randn(rank, hidden_dim) * 0.02)

    def forward(
        self,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
        panel_id: torch.Tensor,
    ) -> torch.Tensor:
        # Return event-level low-rank calendar residual: [B, N, H].
        horizon = event_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = event_time / horizon
        delta_t = torch.zeros_like(event_time)
        delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)

        early = (time_norm <= 0.33).to(event_time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(event_time.dtype)
        late = (time_norm > 0.66).to(event_time.dtype)
        local_density = 1.0 / (1.0 + delta_t)
        panel_seen = (panel_id >= 0).to(event_time.dtype)

        var_onehot = F.one_hot(event_var_id.clamp_min(0), self.num_vars).to(event_time.dtype)
        atom_x = torch.cat(
            [
                var_onehot,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                local_density.unsqueeze(-1),
                early.unsqueeze(-1),
                middle.unsqueeze(-1),
                late.unsqueeze(-1) + panel_seen.unsqueeze(-1),
            ],
            dim=-1,
        )
        coeff = self.atom_proj(atom_x) * event_mask.unsqueeze(-1)
        return torch.einsum("bnr,rh->bnh", coeff, self.basis)


class CalendarKnockoffFirewall(nn.Module):
    """Wrap a value encoder with a calendar knockoff firewall."""

    def __init__(
        self,
        value_encoder: nn.Module,
        hidden_dim: int,
        num_vars: int,
        num_classes: int,
        calendar_rank: int = 8,
    ):
        super().__init__()
        self.value_encoder = value_encoder
        self.calendar = CalendarAdapter(num_vars, hidden_dim, calendar_rank)
        self.fuse_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_with_calendar(self, batch: dict, prefix: str = "") -> dict:
        h_value = self.value_encoder(
            event_value=batch["event_value"],
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
        )
        # value_encoder may return [B, N, H] or a dict containing event_state.
        if isinstance(h_value, dict):
            h_event = h_value["event_state"]
        else:
            h_event = h_value

        cal = self.calendar(
            event_time=batch[f"{prefix}event_time"],
            event_var_id=batch[f"{prefix}event_var_id"],
            event_mask=batch["event_mask"],
            panel_id=batch[f"{prefix}panel_id"],
        )
        gate = self.fuse_gate(torch.cat([h_event, cal], dim=-1))
        fused_event = h_event + gate * cal
        pooled = masked_mean(fused_event, batch["event_mask"], dim=1)
        logits = self.classifier(pooled)
        return {
            "logits": logits,
            "h_event": h_event,
            "calendar_residual": cal,
            "gate": gate,
            "pooled": pooled,
        }

    def forward(self, batch: dict) -> dict:
        real = self.encode_with_calendar(batch, prefix="")
        knock = self.encode_with_calendar(batch, prefix="knockoff_")
        return {"real": real, "knockoff": knock}

    def group_margin_contributions(
        self,
        out: dict,
        labels: torch.Tensor,
        group_id: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        # Approximate group contribution by pooling gated calendar residual per group
        # and passing through the classifier's last linear layer direction.
        residual = out["calendar_residual"] * out["gate"]
        group_vec = []
        for gid in range(num_groups):
            group_mask = (group_id == gid).to(residual.dtype)
            group_vec.append(masked_mean(residual, group_mask, dim=1))
        group_vec = torch.stack(group_vec, dim=1)  # [B, G, H]

        last = self.classifier[-1]
        class_weight = last.weight  # [C, H]
        true_w = class_weight[labels]  # [B, H]
        rival_w = class_weight[None].expand(labels.size(0), -1, -1).clone()
        rival_w[torch.arange(labels.size(0), device=labels.device), labels] = -1e4
        rival_idx = out["logits"].masked_fill(
            F.one_hot(labels, out["logits"].size(-1)).bool(),
            -1e4,
        ).argmax(dim=-1)
        rival_w = class_weight[rival_idx]
        margin_w = true_w - rival_w
        return torch.einsum("bgh,bh->bg", group_vec, margin_w)

    def training_loss(
        self,
        batch: dict,
        lambda_fdr: float = 0.2,
        lambda_swap: float = 0.1,
        lambda_rank: float = 0.02,
        fdr_q: float = 0.20,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["real"]["logits"], labels)

        num_groups = batch["calendar_swap_sign"].size(1)
        real_margin = self.group_margin_contributions(out["real"], labels, batch["calendar_group_id"], num_groups)
        ko_margin = self.group_margin_contributions(out["knockoff"], labels, batch["calendar_group_id"], num_groups)
        w_stat = real_margin - ko_margin

        fdr_loss = soft_knockoff_fdr_loss(w_stat, q=fdr_q)

        swapped = self.encode_with_calendar(batch, prefix="swapped_")
        swapped_margin = self.group_margin_contributions(swapped, labels, batch["calendar_group_id"], num_groups)
        swap_sign = batch["calendar_swap_sign"].to(w_stat.dtype)
        swap_loss = (swapped_margin - swap_sign * real_margin.detach()).abs().mean()

        cal_norm = out["real"]["calendar_residual"].pow(2).mean().sqrt()
        value_norm = out["real"]["h_event"].pow(2).mean().sqrt().clamp_min(1e-6)
        rank_loss = F.relu(cal_norm / value_norm - 0.35).pow(2)

        total = cls_loss + lambda_fdr * fdr_loss + lambda_swap * swap_loss + lambda_rank * rank_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "knockoff_fdr_loss": fdr_loss.detach(),
            "swap_loss": swap_loss.detach(),
            "rank_loss": rank_loss.detach(),
            "calendar_reliance": (cal_norm / value_norm).detach(),
        }


def soft_knockoff_fdr_loss(
    w_stat: torch.Tensor,
    q: float = 0.20,
    temperature: float = 0.10,
    num_thresholds: int = 8,
) -> torch.Tensor:
    """Differentiable knockoff-FDR firewall over group statistics."""

    max_w = w_stat.detach().abs().amax().clamp_min(1e-3)
    thresholds = torch.linspace(0.0, max_w.item(), num_thresholds, device=w_stat.device, dtype=w_stat.dtype)
    losses = []
    for threshold in thresholds:
        soft_pos = torch.sigmoid((w_stat - threshold) / temperature)
        soft_neg = torch.sigmoid((-w_stat - threshold) / temperature)
        fdp_hat = (1.0 + soft_neg.sum(dim=1)) / soft_pos.sum(dim=1).clamp_min(1.0)
        selected = soft_pos.mean(dim=1)
        losses.append(F.relu(fdp_hat - q).pow(2) * selected)
    return torch.stack(losses, dim=0).mean()


@torch.no_grad()
def build_calendar_knockoff_batch(batch: dict, num_vars: int, num_time_bins: int = 3) -> dict:
    """Create conditional calendar knockoffs by constrained within-batch permutation.

    This sketch preserves low-order calendar statistics while breaking sample-specific
    policy-label shortcuts. Production code can stratify by site, length, and coarse label-free covariates.
    """

    event_time = batch["event_time"]
    event_var_id = batch["event_var_id"]
    event_mask = batch["event_mask"]
    panel_id = batch.get("panel_id", torch.full_like(event_var_id, -1))
    bsz, num_events = event_time.shape
    device = event_time.device

    # Match samples with similar observed length to preserve calendar budget.
    obs_len = event_mask.sum(dim=1)
    order = torch.argsort(obs_len)
    partner = order.roll(shifts=1)
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(bsz, device=device)
    partner = partner[inv_order]

    batch["knockoff_event_time"] = event_time[partner]
    batch["knockoff_event_var_id"] = event_var_id[partner]
    batch["knockoff_panel_id"] = panel_id[partner]

    # Group atoms: variable bucket x coarse time window.
    horizon = event_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_bin = ((event_time / horizon) * num_time_bins).long().clamp(0, num_time_bins - 1)
    var_bucket = event_var_id.clamp(0, num_vars - 1)
    batch["calendar_group_id"] = var_bucket * num_time_bins + time_bin

    # Random group swaps for swap-sign calibration.
    swap_mask = torch.rand(bsz, num_events, device=device) < 0.5
    batch["swapped_event_time"] = torch.where(swap_mask, batch["knockoff_event_time"], event_time)
    batch["swapped_event_var_id"] = torch.where(swap_mask, batch["knockoff_event_var_id"], event_var_id)
    batch["swapped_panel_id"] = torch.where(swap_mask, batch["knockoff_panel_id"], panel_id)

    group_swap = torch.zeros(bsz, num_vars * num_time_bins, device=device)
    for gid in range(num_vars * num_time_bins):
        hit = (batch["calendar_group_id"] == gid) & (event_mask > 0)
        if hit.any():
            group_swap[:, gid] = torch.where((swap_mask & hit).float().sum(dim=1) > 0, -1.0, 1.0)
        else:
            group_swap[:, gid] = 1.0
    batch["calendar_swap_sign"] = group_swap
    return batch
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `panel calendar shift`：训练环境同步 panel，测试环境拆成异步项目。
   - `alarm calendar shift`：训练环境报警后短窗口密集复测，测试环境延迟或稀疏复测。
   - `routine calendar shift`：从事件触发采样换成固定查房采样。
   - `device calendar shift`：夜间或低电量窗口系统性稀疏。

2. **对比方法**
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - MILM-style value-redacted sampling classifier。
   - MedSpaformer-style token sparsification baseline。
   - DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC 等历史方案。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - knockoff-FDR estimate：被模型判定为有用的日历 group 中，负控假发现比例。
   - calendar reliance ratio：日历 residual 能量相对 value representation 的比例。
   - passed-calendar discovery count：通过 knockoff firewall 的时间窗/变量组数量。
   - failed prediction knockoff gap：错误预测是否伴随高 knockoff 假发现。

4. **消融实验**
   - 去掉 knockoff 日历，只使用真实 calendar adapter，验证是否过拟合采样日历。
   - 去掉 `L_soft_knockoff_fdr`，检查 worst-policy accuracy 与 false calendar discoveries。
   - 去掉 `L_swap_symmetry`，验证 knockoff 统计是否失去交换性语义。
   - 将 conditional knockoff 改成完全随机置换，验证保留低阶日历统计的重要性。
   - 扫描 calendar residual rank，验证日历信息作为低秩补充而非主分类器的必要性。

## 5. 预期创新性

1. **从采样去偏转向采样负控审计**：不再试图删除、正交化、征税、除法分解或纠错采样信息，而是用 knockoff 日历检验哪些采样证据经得起统计负控。
2. **从 value-redacted 诊断转向 knockoff firewall**：吸收 MILM 对 informative sampling 的揭示，但不让 sampling-only 能力直接成为分类捷径；采样日历必须击败条件交换的 knockoff。
3. **从 token sparsification 转向日历发现控制**：吸收 MedSpaformer 对 token 可见性和跨数据集迁移的关注，但核心不是选 token，而是控制“日历 group 被发现为有用证据”的 false discovery rate。
4. **从反事实增强转向条件交换负控**：counterfactual intervention 不产生一致性视图、不平滑、不积分、不估计危险率，只生成真实/knockoff/swap 日历三元组。
5. **解释性直接面向部署**：模型能报告哪些变量组、时间窗或 panel 共现通过了 knockoff firewall，哪些只是训练采样政策下的可疑日历假发现。

## 6. 一句话投稿卖点

**CKCF 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“真实采样日历与条件交换 knockoff 日历之间的可审计发现问题”，并通过 calendar adapter、knockoff group statistics 与 soft FDR firewall，让模型只保留能击败统计负控的采样日历证据，从而在不依赖危险率、对抗、一致性、后验商、随机平滑、拓扑、gauge 或纠错码的前提下提升跨采样政策鲁棒性。**
