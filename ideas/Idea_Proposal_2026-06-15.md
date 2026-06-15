# Title: Policy-Envelope Certified Classifier：面向采样策略偏移的反事实采样包络认证分类器

## 0. 强制去重记录与思维黑名单

### 已读取材料

- `paper_daily.md`
- `paper_daily_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-13.md`
- `ideas/Idea_Proposal_2026-06-14.md`
- 根目录 `README.md`
- `工作素材/README.md`
- `周计划/README.md`
- `周报/README.md`

### 未检出但已确认搜索

- 当前仓库未检出 `my_work_summary.md`。
- 当前仓库未检出任何 `*summary*.md` 或 `*work*.md` 命名的 Markdown 总结文件。

### 历史核心机制黑名单

为了避免与历史提案和 `paper_daily.md` 中已有机制发生思维重合，本提案明确避开以下机制作为主创新：

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
14. additive evidence market、protocol tax、token-level 证据预算或边际证据审计。

本提案选择一个与上述机制正交的切入点：**不估计采样危险率，不要求反事实视图一致，不做对抗去偏，不做算子交换，也不为证据定价；而是把“采样策略偏移”转化为一组可声明的观测包络，并训练分类器对包络内的所有反事实采样轨迹都保持可认证的分类 margin。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中有三条前沿线索值得和当前“采样解耦/反事实干预”框架结合：

- **FlowPath** 提醒我们，不规则采样改变的不只是 mask，而是离散观测诱导出的连续路径几何。
- **GARLIC / SuperMAN** 提醒我们，真实稀疏异步数据中的解释单元可以是 observation-level、signal-level 或 subset-level，而不是固定网格上的 dense token。
- **PYRREGULAR** 提醒我们，方法不应只在某个私有预处理脚本里成立，而应能接入统一 irregular array format，并显式定义跨 policy / cross-policy 评测协议。

历史方案分别从 hazard score、算子交换、协议税市场等角度处理采样偏移，但它们仍有一个共同挑战：训练时看到的反事实采样 policy 永远是有限个。如果测试机构采用一种未枚举过的采样流程，例如把 late-window panel、变量级降采样和设备触发复测混合起来，模型是否仍然稳健很难仅靠若干增强视图证明。

**Policy-Envelope Certified Classifier (PECC)** 的核心直觉是：

> 与其枚举很多 `do(policy)` 视图并希望模型学会泛化，不如把反事实采样干预上升为一个“采样包络”：只要测试时的时间抖动、变量删失、panel 联测变化和观测值保持误差落在包络内，分类器就被训练为拥有可计算的最坏情况 logit margin。

这使采样解耦/反事实干预框架多出一个认证层：

- value branch 仍负责建模观测值驱动的状态证据；
- sampling branch 不输出 hazard、policy label、tax 或 adversarial signal，而是输出**允许的采样扰动半径**；
- counterfactual intervention 不再产生多个正样本或 risk variance，而是用于校准包络是否覆盖真实 policy edit；
- classifier 的目标不是“平均反事实准确”，而是“包络内最坏情况仍有正 margin”。

因此，PECC 直接针对 sampling-policy shift 的核心风险：测试策略可以不同，但只要不同方式落在训练声明的 policy envelope 中，模型的决策边界就不能穿过这个包络。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从单点轨迹编码改为“区间潜变量包络编码”

PECC 保留不规则事件输入 `(time, variable, value, mask)`，但每个样本不再只编码成一个向量 `z`，而是编码成一个 latent interval：

```text
Z(x, E_policy) = [z_lower, z_upper]
```

其中 `E_policy` 是 sampling branch 给出的采样策略包络，包含：

1. **Time Envelope**
   - 每个观测时间允许在 `[t - eps_t, t + eps_t]` 内移动。
   - 该包络用于覆盖不同设备时钟、护士查房窗口或批量化实验室出结果时间。

2. **Visibility Envelope**
   - 每个变量/时间窗允许一定比例的观测被删除或延迟。
   - 不直接把 mask pattern 当分类特征，而是把 mask 的变化范围转成 latent uncertainty。

3. **Value-Hold Envelope**
   - 对 stale observation 的复用引入值域不确定性 `[v - eps_v, v + eps_v]`。
   - 这吸收 GARLIC 中旧观测衰减的启发，但不让“多久未测”直接成为类别捷径。

4. **Subset Envelope**
   - 对 panel 或变量组施加共享扰动半径，覆盖 SuperMAN/GARLIC 讨论的 co-measurement 和 subset-level 稀疏结构。
   - 这里不是学习图边或 additive evidence，而是声明“这个变量组的观测可见性可能一起改变”。

Encoder 使用 interval bound propagation (IBP) 风格的保守传播：

- 事件级输入先变成 `[x_lower, x_upper]`；
- 通过带 Lipschitz 控制的线性层和单调激活传播上下界；
- 通过 masked mean / attention-free set pooling 得到 `[z_lower, z_upper]`；
- 分类头输出每个类别 logit 的上下界 `[ell_lower, ell_upper]`。

关键区别：PECC 不要求两个反事实视图的 representation 一样，也不学习策略残差槽。它只保证分类器在整个策略包络内没有穿越决策边界。

### 2.2 改 Loss：从一致性/对抗/预算转向“最坏情况认证 margin”

总目标：

```text
L = L_cls_center
  + lambda_cert  * L_cert_margin
  + lambda_tube  * L_tube_tightness
  + lambda_cover * L_cf_coverage
```

#### A. 中心轨迹分类损失 `L_cls_center`

使用包络中心点的 logits 做普通分类：

```text
L_cls_center = CE(f(z_center), y)
```

它保证模型仍然对原始观测有判别能力，而不是只学一个过宽的保守包络。

#### B. 认证 margin 损失 `L_cert_margin`

对真实类别 `y`，取其最坏下界；对其他类别，取最坏上界：

```text
cert_margin = logit_lower[y] - max_{k != y} logit_upper[k]
L_cert_margin = relu(margin_target - cert_margin)^2
```

这意味着：只要反事实采样轨迹落在包络中，真实类 logit 的下界仍应高于竞争类上界。它不是 risk variance，也不是 logits consistency；它直接优化一个可审查的 worst-policy certificate。

#### C. 包络紧致损失 `L_tube_tightness`

如果包络无限大，认证会变得过度保守；如果包络过小，又无法覆盖真实 policy shift。因此加入 tightness regularizer：

```text
L_tube_tightness = mean(width(z_upper - z_lower))
```

但该项不能独立收缩包络，而是和 coverage loss 配合使用，避免 sampling branch 把扰动半径压到零。

#### D. 反事实覆盖损失 `L_cf_coverage`

当前框架中已有反事实干预模块。PECC 不把反事实视图拿来做对比学习或一致性，而是只检查它们是否落入包络：

```text
L_cf_coverage = mean relu(z_cf - z_upper)^2 + relu(z_lower - z_cf)^2
```

如果某个 `do(policy)` 视图落在包络之外，sampling branch 必须扩大相应的 policy envelope；如果所有视图都被覆盖，分类器就只需对这个 envelope 做认证训练。

### 2.3 改 Dataloader：生成“策略包络规格”，不是生成增强样本对

新增 `PolicyEnvelopeCollator`，每个 batch 返回：

1. `center_values, center_times, center_mask`：原始不规则观测。
2. `time_radius`：每个时间窗允许的时间抖动。
3. `value_radius`：stale observation 或低置信观测的值域扰动。
4. `visibility_radius`：变量级/时间窗级可见性删失半径。
5. `subset_ids`：panel、变量组或设备通道 id。
6. `cf_views`：少量已有反事实采样视图，仅用于 coverage calibration，不用于一致性或 risk variance。

policy envelope 的半径可以由三种来源合成：

- PYRREGULAR 风格的环境元数据：不同数据集、医院、设备或采样频率分位数。
- 当前 sampling branch 的 policy uncertainty head：预测哪些时间窗/变量组的采样机制更不稳定。
- 反事实干预模块的 edit magnitude：例如 late-window shift、panel co-measurement edit、variable thinning edit 的最大强度。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改造成 `IntervalValueEncoder`，输出中心表示和上下界表示。
- 现有 sampling branch 改造成 `EnvelopeRadiusHead`，只预测采样扰动半径，不输出可被分类器使用的 policy feature。
- 现有 counterfactual intervention 模块保留，但职责改为校准 coverage，而不是产生一致性正样本、hazard resampling 或 risk variance 项。
- 推理阶段输出：
  - center prediction；
  - certified margin；
  - envelope width；
  - per-subset coverage risk，用于诊断模型是否面对超出训练包络的采样政策。

如果测试样本的 envelope width 很大且 certified margin 为负，模型可以触发“不认证”告警。这比盲目给出高置信预测更适合跨机构采样策略变化场景。

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntervalLinear(nn.Module):
    """Linear layer for interval bound propagation."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_pos = self.weight.clamp_min(0.0)
        weight_neg = self.weight.clamp_max(0.0)
        out_lower = F.linear(lower, weight_pos) + F.linear(upper, weight_neg) + self.bias
        out_upper = F.linear(upper, weight_pos) + F.linear(lower, weight_neg) + self.bias
        return out_lower, out_upper


class IntervalMLP(nn.Module):
    """Small monotone-activation MLP that propagates lower/upper bounds."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = IntervalLinear(input_dim, hidden_dim)
        self.fc2 = IntervalLinear(hidden_dim, hidden_dim)

    def forward(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower, upper = self.fc1(lower, upper)
        lower, upper = F.relu(lower), F.relu(upper)
        lower, upper = self.fc2(lower, upper)
        lower, upper = F.relu(lower), F.relu(upper)
        return lower, upper


class EnvelopeRadiusHead(nn.Module):
    """Predict sampling-envelope radii without feeding policy features to logits."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softplus(),
        )

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        delta_t: torch.Tensor,
        var_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # value/time/delta_t: [B, N], var_id: [B, N]
        features = torch.stack([value, time, torch.log1p(delta_t)], dim=-1)
        h = torch.cat([features, self.var_embed(var_id)], dim=-1)
        radius = self.net(h)
        return {
            "value_radius": radius[..., 0],
            "time_radius": radius[..., 1],
            "visibility_radius": radius[..., 2],
        }


class IntervalValueEncoder(nn.Module):
    """Encode irregular event sets into a latent interval envelope."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_mlp = IntervalMLP(input_dim=hidden_dim + 3, hidden_dim=hidden_dim)

    def build_event_bounds(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        delta_t: torch.Tensor,
        var_id: torch.Tensor,
        value_radius: torch.Tensor,
        time_radius: torch.Tensor,
        visibility_radius: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var_h = self.var_embed(var_id)

        center = torch.stack([value, time, torch.log1p(delta_t)], dim=-1)
        radius = torch.stack(
            [
                value_radius,
                time_radius,
                torch.log1p(delta_t + time_radius).sub(torch.log1p(delta_t)).abs(),
            ],
            dim=-1,
        )
        lower = torch.cat([center - radius, var_h], dim=-1)
        upper = torch.cat([center + radius, var_h], dim=-1)

        # Visibility uncertainty widens event features instead of entering the classifier.
        vis_pad = visibility_radius.unsqueeze(-1)
        lower = lower - vis_pad
        upper = upper + vis_pad
        return lower, upper

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        delta_t: torch.Tensor,
        var_id: torch.Tensor,
        event_mask: torch.Tensor,
        envelope: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower, upper = self.build_event_bounds(
            value=value,
            time=time,
            delta_t=delta_t,
            var_id=var_id,
            value_radius=envelope["value_radius"],
            time_radius=envelope["time_radius"],
            visibility_radius=envelope["visibility_radius"],
        )
        lower, upper = self.event_mlp(lower, upper)

        mask = event_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        z_lower = (lower * mask).sum(dim=1) / denom
        z_upper = (upper * mask).sum(dim=1) / denom
        return z_lower, z_upper


class IntervalClassifier(nn.Module):
    """Classifier that returns center logits and certified logit bounds."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.bound_head = IntervalLinear(hidden_dim, num_classes)
        self.center_head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        z_lower: torch.Tensor,
        z_upper: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_center = 0.5 * (z_lower + z_upper)
        logit_lower, logit_upper = self.bound_head(z_lower, z_upper)
        center_logits = self.center_head(z_center)
        return {
            "center_logits": center_logits,
            "logit_lower": logit_lower,
            "logit_upper": logit_upper,
            "z_center": z_center,
        }


class PolicyEnvelopeCertifiedClassifier(nn.Module):
    def __init__(self, num_vars: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.radius = EnvelopeRadiusHead(num_vars=num_vars, hidden_dim=hidden_dim)
        self.encoder = IntervalValueEncoder(num_vars=num_vars, hidden_dim=hidden_dim)
        self.classifier = IntervalClassifier(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        envelope = self.radius(
            value=batch["event_value"],
            time=batch["event_time"],
            delta_t=batch["event_delta_t"],
            var_id=batch["event_var_id"],
        )
        z_lower, z_upper = self.encoder(
            value=batch["event_value"],
            time=batch["event_time"],
            delta_t=batch["event_delta_t"],
            var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
            envelope=envelope,
        )
        out = self.classifier(z_lower, z_upper)
        out.update({
            "z_lower": z_lower,
            "z_upper": z_upper,
            **envelope,
        })
        return out

    def certified_margin(
        self,
        logit_lower: torch.Tensor,
        logit_upper: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        true_lower = logit_lower.gather(1, labels[:, None]).squeeze(1)
        class_mask = F.one_hot(labels, logit_upper.size(-1)).bool()
        rival_upper = logit_upper.masked_fill(class_mask, -1e4).max(dim=-1).values
        return true_lower - rival_upper

    def encode_center_view(self, view: dict) -> torch.Tensor:
        zero_envelope = {
            "value_radius": torch.zeros_like(view["event_value"]),
            "time_radius": torch.zeros_like(view["event_time"]),
            "visibility_radius": torch.zeros_like(view["event_mask"].float()),
        }
        z_lower, z_upper = self.encoder(
            value=view["event_value"],
            time=view["event_time"],
            delta_t=view["event_delta_t"],
            var_id=view["event_var_id"],
            event_mask=view["event_mask"],
            envelope=zero_envelope,
        )
        return 0.5 * (z_lower + z_upper)

    def training_loss(
        self,
        batch: dict,
        lambda_cert: float = 0.5,
        lambda_tube: float = 0.01,
        lambda_cover: float = 0.1,
        margin_target: float = 0.2,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(batch)
        labels = batch["labels"]

        cls_loss = F.cross_entropy(out["center_logits"], labels)
        cert_margin = self.certified_margin(
            out["logit_lower"],
            out["logit_upper"],
            labels,
        )
        cert_loss = F.relu(margin_target - cert_margin).pow(2).mean()
        tube_loss = (out["z_upper"] - out["z_lower"]).abs().mean()

        cover_losses = []
        for view in batch.get("cf_views", []):
            z_cf = self.encode_center_view(view)
            high_violation = F.relu(z_cf - out["z_upper"]).pow(2)
            low_violation = F.relu(out["z_lower"] - z_cf).pow(2)
            cover_losses.append((high_violation + low_violation).mean())

        if cover_losses:
            cover_loss = torch.stack(cover_losses).mean()
        else:
            cover_loss = torch.zeros((), device=labels.device)

        total = (
            cls_loss
            + lambda_cert * cert_loss
            + lambda_tube * tube_loss
            + lambda_cover * cover_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "cert_loss": cert_loss.detach(),
            "tube_loss": tube_loss.detach(),
            "cover_loss": cover_loss.detach(),
            "mean_cert_margin": cert_margin.mean().detach(),
            "mean_envelope_width": (out["z_upper"] - out["z_lower"]).abs().mean().detach(),
        }


@torch.no_grad()
def build_policy_envelope_batch(
    batch: dict,
    base_time_radius: float = 0.05,
    base_value_radius: float = 0.02,
    stale_scale: float = 0.1,
) -> dict:
    """Attach simple policy-envelope priors to an irregular event batch.

    The radii are priors for the learnable EnvelopeRadiusHead, not classifier inputs.
    They can be replaced by PYRREGULAR-style environment metadata or intervention logs.
    """

    delta_t = batch["event_delta_t"]
    batch["time_radius_prior"] = torch.full_like(delta_t, base_time_radius)
    batch["value_radius_prior"] = base_value_radius + stale_scale * torch.log1p(delta_t)
    batch["visibility_radius_prior"] = 1.0 - batch["event_mask"].float()
    return batch
```

## 4. 预期创新性

1. **从“训练时见过的反事实策略”转向“包络内策略认证”**：PECC 不依赖枚举所有 policy，而是训练一个覆盖 policy family 的 latent interval certificate。
2. **从 representation invariance 转向 certified margin**：不要求多视图表征一致，不惩罚 logits 方差，而是直接保证最坏情况下真实类下界超过竞争类上界。
3. **从采样机制估计转向采样不确定性声明**：sampling branch 只输出 envelope radii，不输出 hazard、policy label、tax 或可进入分类器的 mask feature。
4. **自然结合 FlowPath/GARLIC/SuperMAN/PYRREGULAR**：承认采样改变路径几何、旧观测、变量组和 benchmark 环境，但把这些因素统一成可认证的输入包络，而不是图边、证据市场或对抗支路。
5. **面向部署的可审查性**：推理时可以报告 certified margin 与 envelope width；当跨医院采样政策超出包络时，模型明确知道自己没有认证，而不是给出虚假高置信预测。

## 5. 实验切入点

1. 构造四类 policy envelope shift：
   - 时间抖动：观测整体提前/延后或按窗口批量化；
   - 变量删失：某些变量在测试医院系统性少测；
   - panel 联动：实验室 panel 从联测变成按需测，或相反；
   - stale reuse：旧观测在不同策略下被复用更久。
2. 对比：
   - mask dropout；
   - missingness-aware encoder；
   - policy adversarial baseline；
   - hazard/null-space baseline；
   - commutator graph surgery baseline；
   - protocol-tax additive evidence baseline；
   - FlowPath / GARLIC / SuperMAN 风格主干。
3. 新增指标：
   - certified policy accuracy：认证 margin 为正的样本上的准确率；
   - coverage rate：真实反事实视图落入 envelope 的比例；
   - abstention under policy shift：认证失败样本比例；
   - worst-envelope margin；
   - envelope width calibration：错误样本是否具有更宽包络或负认证 margin。
4. 消融：
   - 去掉 `L_cert_margin`，验证是否只剩普通增强；
   - 去掉 `L_cf_coverage`，验证包络是否被 radius head 压得过窄；
   - 去掉 `L_tube_tightness`，验证认证是否过度保守；
   - 只使用固定半径而非 learnable envelope，验证 sampling branch 对策略不确定性的建模价值。

## 6. 一句话投稿卖点

**PECC 首次把非规则采样时间序列中的 sampling-policy shift 表述为“反事实采样包络内的可认证分类问题”：模型不再依赖枚举式增强、危险率去偏、图交换或证据征税，而是对一族时间抖动、变量删失、panel 联测和 stale reuse 策略给出最坏情况 logit margin 证书，从而为跨采样政策部署提供可审查的鲁棒性保证。**
