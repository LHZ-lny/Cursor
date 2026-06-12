# Title: Do-Hazard Nullifier：面向采样策略偏移的危险率得分正交化反事实分类

## 0. 强制去重记录与机制黑名单

### 已读取材料

- `paper_daily_2026-06-12.md`
- 当前仓库未检出 `my_work_summary.md`
- 当前仓库未检出历史 `Idea_Proposal_*.md` 或其他历史提案文件

### 本轮机制黑名单

由于仓库中没有可读取的历史提案，本轮黑名单主要来自今日 paper daily 中已经出现或被明确暗示的机制，后续新想法避免直接复用：

1. learnable reference points / adaptive time encoding 作为主对齐机制。
2. temporal consistency 与 inter-variable consistency regularization。
3. frequency-guided observation encoder 或频域稳定表征。
4. missingness pattern encoder 直接参与分类。
5. prototype-constrained classifier / policy-aware prototypes。
6. 以跨采样增强的一致性或对比学习作为主要创新点。
7. 简单环境对抗去策略分类器。

本提案选择完全不同的切入点：把采样策略建模为连续时间点过程的危险率 nuisance score，并在训练目标中对分类梯度做“采样得分零空间手术”，而不是把时间锚点、缺失模式、频域信息或类别原型作为核心机制。

## 1. Motivation: 为什么它能解决采样偏移问题

非规则采样时间序列的最大风险不是“缺了多少点”，而是“为什么这些点会被采到”。在医疗、可穿戴或工业传感场景中，采样时刻本身常由一个策略决定：病情越重越频繁监测，设备低电量时降低采样率，医生对某些类别患者进行主动复查。若模型把这种策略性观测节奏当成类别证据，一旦测试医院或设备调度规则改变，分类器就会发生 sampling-policy leakage。

今日论文给了两个重要提醒：

- Adaptive Time Encoding 说明固定网格不足，时间处理应该自适应；但如果自适应锚点被标签相关的采样政策牵引，就会吸收策略偏差。
- SPECTRA 说明 missingness 与类别不平衡存在信息耦合；但若缺失模式来自训练环境政策，直接把 missingness pattern 用于分类会放大偏移。

因此，本提案不再问“哪些时间点最适合表达类别”，而是问：

> 如果分类证据的梯度方向能被采样危险率解释，那么这部分证据是否应该从最终决策中被剥离？

我们将采样过程视为 nuisance process，学习其连续时间危险率 score；然后对分类表示施加正交约束，使最终分类器的敏感方向落在采样策略 score 的零空间中。这样既允许模型理解采样机制以完成反事实干预，又阻止采样机制本身成为分类捷径。

## 2. Methodology: 具体修改点

### 2.1 总体框架

命名为 **Do-Hazard Nullifier, DHN**。它由三部分组成：

1. **Value Encoder**：只编码观测值与相对时间间隔，输出轨迹语义表示 `z`。它不接收显式采样策略标签，也不把缺失模式作为分类特征。
2. **Policy Hazard Scorer**：从同一条序列的事件时间、mask 和历史摘要中估计采样危险率 `lambda(t | history)`，得到采样策略的 log-likelihood score。该分支只用于训练正则与反事实重采样，不在推理时参与分类。
3. **Null-Space Classifier**：分类 loss 不仅最小化交叉熵，还要求分类梯度对采样危险率 score 近似正交；若某个表示方向同时能提高类别预测并提高采样策略可预测性，该方向会被惩罚。

### 2.2 改 Encoder

保留你当前“采样解耦/反事实干预”框架中值过程与采样过程分开的思想，但将采样支路改成 **hazard scorer**：

- 输入：`times [B, T]`、`mask [B, T, D]`、观测历史摘要 `h [B, T, H]`。
- 输出：每个时间点的观测危险率 `lambda [B, T, D]`。
- 训练方式：最大化实际观测事件的 point-process likelihood，同时估计未观测区间的 cumulative hazard。

关键区别：该支路不是 missingness pattern encoder，不把缺失表征拼接到分类器；它只提供“采样策略 score 坐标系”。

### 2.3 改 Loss

总目标：

```text
L = L_cls
  + alpha * L_hazard
  + beta  * L_score_null
  + gamma * L_do_risk
```

#### A. 分类损失 `L_cls`

标准交叉熵，使用 Value Encoder 输出的 `z`。

#### B. 采样危险率似然 `L_hazard`

将观测 mask 视为多变量 marked point process 的事件指示：

```text
L_hazard = - sum_{t,d} m_{t,d} log lambda_{t,d}
           + sum_{t,d} lambda_{t,d} * Delta t
```

它学习“当前采样策略为什么在这些时刻观测这些变量”。

#### C. 得分零空间约束 `L_score_null`

分别计算：

- `g_y = d log p(y | z) / dz`：分类器在表示空间中的敏感方向。
- `g_s = d log p(mask, times | z) / dz`：采样策略在表示空间中的 score 方向。

惩罚二者的平方余弦相似度：

```text
L_score_null = mean cosine(g_y, g_s)^2
```

直觉：若一个表示方向强烈影响分类，同时也强烈解释采样政策，它很可能是策略泄漏通道；将其压到零空间中能提升 policy shift 下的稳健性。

#### D. 反事实 do-risk `L_do_risk`

在 dataloader 或 collate_fn 中对同一条连续轨迹构造若干反事实采样政策：

- 高密度政策：上调 hazard 后进行 thinning。
- 低密度政策：下调 hazard 后进行 thinning。
- 变量偏置政策：只改变某些变量的采样危险率。

这些视图不做对比学习，也不强制中间表示完全一致；只要求经过 Null-Space Classifier 后的经验风险稳定：

```text
L_do_risk = Var_k CE(f(x under do(policy=k)), y)
```

这比 logits 一致性更弱，允许不同采样政策下保留可用信息差异，同时压制策略导致的风险振荡。

### 2.4 改 Dataloader

新增 `CounterfactualHazardCollator`：

1. 从原始 `times, values, mask` 估计基础间隔 `Delta t`。
2. 采样一个政策倍率 `rho_d`，可按变量或样本改变。
3. 用 `p_keep = 1 - exp(-rho_d * lambda_d * Delta t)` 生成反事实 mask。
4. 返回 factual view 与若干 `do(policy=rho)` views。

这一步与普通 mask dropout 不同：mask 的生成概率由危险率支路控制，因而模拟的是“采样政策干预”，不是随机缺失增强。

## 3. 预期创新性

1. **从 representation invariance 转向 gradient nuisance orthogonality**：不是要求两个增强视图表征一样，而是要求分类决策方向不落在采样策略 score 方向上。
2. **从 missingness feature 转向 policy score coordinate**：缺失/采样模式不再被直接用于分类，而是作为要被消除的 nuisance 坐标系。
3. **从固定反事实 mask 转向 hazard-driven do intervention**：反事实采样由连续时间危险率生成，更贴近真实采样政策变化。
4. **兼容采样解耦框架**：值过程继续服务分类，采样过程只服务反事实干预和正交约束，不破坏现有主干。

## 4. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class HazardScorer(nn.Module):
    """Estimate sampling-policy hazard without feeding it to the classifier."""

    def __init__(self, hidden_dim: int, num_vars: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_vars),
        )

    def forward(self, history_state: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        # history_state: [B, T, H], delta_t: [B, T]
        x = torch.cat([history_state, delta_t.unsqueeze(-1)], dim=-1)
        return F.softplus(self.net(x)) + 1e-6


def hazard_nll(
    hazard: torch.Tensor,
    mask: torch.Tensor,
    delta_t: torch.Tensor,
) -> torch.Tensor:
    """Point-process negative log likelihood for irregular observations."""

    # hazard, mask: [B, T, D], delta_t: [B, T]
    event_term = mask * torch.log(hazard)
    integral_term = hazard * delta_t.unsqueeze(-1)
    return -(event_term - integral_term).mean()


def score_null_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    hazard_loglik: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Penalize classifier gradients that align with sampling-policy score."""

    cls_logp = F.log_softmax(logits, dim=-1)
    cls_obj = cls_logp.gather(1, labels[:, None]).sum()
    policy_obj = hazard_loglik.sum()

    grad_y = torch.autograd.grad(
        cls_obj,
        z,
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_s = torch.autograd.grad(
        policy_obj,
        z,
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_y = grad_y.flatten(1)
    grad_s = grad_s.flatten(1)
    cos = F.cosine_similarity(grad_y, grad_s, dim=-1)
    return (cos ** 2).mean()


class DoHazardNullifier(nn.Module):
    def __init__(self, value_encoder: nn.Module, hidden_dim: int, num_vars: int, num_classes: int):
        super().__init__()
        self.value_encoder = value_encoder
        self.hazard = HazardScorer(hidden_dim=hidden_dim, num_vars=num_vars)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, values: torch.Tensor, times: torch.Tensor, mask: torch.Tensor):
        # value_encoder should return sequence states and pooled representation.
        seq_state, z = self.value_encoder(values=values, times=times, mask=mask)

        delta_t = torch.zeros_like(times)
        delta_t[:, 1:] = (times[:, 1:] - times[:, :-1]).clamp_min(1e-4)
        delta_t[:, 0] = delta_t[:, 1].detach() if times.size(1) > 1 else 1.0

        hazard = self.hazard(seq_state, delta_t)
        logits = self.classifier(z)
        return logits, z, hazard, delta_t

    def training_loss(
        self,
        batch: dict,
        alpha: float = 0.2,
        beta: float = 0.05,
        gamma: float = 0.1,
    ) -> dict:
        logits, z, hazard, delta_t = self(
            values=batch["values"],
            times=batch["times"],
            mask=batch["mask"],
        )
        labels = batch["labels"]

        cls_loss = F.cross_entropy(logits, labels)
        h_loss = hazard_nll(hazard, batch["mask"], delta_t)

        hazard_loglik = (
            batch["mask"] * torch.log(hazard)
            - hazard * delta_t.unsqueeze(-1)
        )
        null_loss = score_null_loss(logits, labels, hazard_loglik, z)

        do_losses = []
        for view in batch.get("cf_views", []):
            cf_logits, _, _, _ = self(
                values=view["values"],
                times=view["times"],
                mask=view["mask"],
            )
            do_losses.append(F.cross_entropy(cf_logits, labels, reduction="none"))

        if do_losses:
            stacked = torch.stack(do_losses, dim=0)
            do_risk = stacked.var(dim=0, unbiased=False).mean()
        else:
            do_risk = torch.zeros((), device=logits.device)

        total = cls_loss + alpha * h_loss + beta * null_loss + gamma * do_risk
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "hazard_loss": h_loss.detach(),
            "score_null_loss": null_loss.detach(),
            "do_risk": do_risk.detach(),
        }


@torch.no_grad()
def hazard_do_resample(
    values: torch.Tensor,
    mask: torch.Tensor,
    hazard: torch.Tensor,
    delta_t: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a counterfactual sampled view under do(policy=rho)."""

    # rho: [B, 1, D] or [B, T, D]
    keep_prob = 1.0 - torch.exp(-rho * hazard * delta_t.unsqueeze(-1))
    cf_mask = torch.bernoulli(keep_prob).to(mask.dtype) * mask
    cf_values = values * cf_mask
    return cf_values, cf_mask
```

## 5. 实验切入点

1. 在 P12/P19/PAM 上构造三类策略偏移：全局采样率变化、变量选择性采样变化、类别相关采样率反转。
2. 对比普通 mask dropout、采样一致性正则、missingness-aware 分类器、frequency-guided encoder 与 prototype classifier。
3. 重点汇报 worst-policy accuracy、policy-shift calibration error、分类梯度与采样 score 的平均夹角。
4. 消融 `L_hazard`、`L_score_null`、`L_do_risk`，验证不是单纯增强或单纯多任务学习带来的收益。

## 6. 一句话投稿卖点

**DHN 首次把非规则时间序列分类中的采样偏移建模为连续时间危险率 score 泄漏问题，并通过反事实 do-risk 与分类梯度零空间手术，主动切断“采样政策即类别证据”的捷径。**
