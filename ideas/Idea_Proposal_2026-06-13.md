# Title: Commutator Graph Surgery：面向采样策略偏移的生理流-采样算子交换子手术

## 0. 强制去重记录与思维黑名单

### 已读取材料

- `paper_daily.md`
- `paper_daily_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-12.md`
- 根目录 `README.md`
- `工作素材/README.md`
- `周计划/README.md`
- `周报/README.md`

### 未检出但已确认搜索

- 当前仓库未检出 `my_work_summary.md`，也未检出任何 `*summary*.md` 或 `*Summary*.md` 文件。

### 历史机制黑名单

为了避免与历史提案和 paper daily 中已有机制发生思维重合，本提案永久避开以下核心机制：

1. learnable reference points / adaptive time encoding 作为主对齐机制。
2. temporal consistency 与 inter-variable consistency regularization。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接参与分类。
5. prototype-constrained classifier / policy-aware prototypes。
6. 跨采样增强的一致性或对比学习作为主要创新点。
7. 简单 environment adversarial / policy adversarial classifier。
8. 连续时间危险率 point-process scorer。
9. 分类梯度与采样 score 的零空间正交化。
10. hazard-driven counterfactual resampling。
11. 多个 do(policy) 视图的 risk variance 约束。

本提案选择一个与上述机制正交的切入点：不估计采样危险率，不做梯度正交，不要求多视图表征一致，也不把缺失模式送入分类器；而是把采样策略看作作用在轨迹图状态上的一个“干预算子”，用它与生理演化算子的非交换残差识别策略污染。

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 paper daily 给出两个非常关键但尚未被充分因果化的观察：

- **FlowPath** 说明，不规则采样分类中真正进入模型的并不是离散点本身，而是由观测点诱导出的连续路径几何。采样策略一变，路径几何也会改变。
- **GSNF** 说明，多变量不规则时序中的变量交互图应进入连续时间演化本身，而不仅是 attention 或后处理。但在 ICU 等场景中，共观测图和生理图经常混在一起：两个变量经常一起被测，可能因为生理耦合，也可能因为医院流程。

采样策略偏移的本质可以被重新表述为一个“算子顺序问题”：

> 若先让潜在生理状态连续演化，再对它施加某个采样政策；与先按该采样政策裁剪观测，再让模型演化，两者在分类相关子空间中是否应该给出同一个生理证据？

对稳定的生理因果信号来说，上述两个顺序应近似可交换。因为真实病理演化不应依赖“医院今天是否多测了 lactate”这一采样流程。相反，策略性观测、联测流程、设备调度会制造明显的非交换残差：先采样再建图会学到流程边，先演化再采样会保留生理边。

因此，本提案提出 **Commutator Graph Surgery (CGS)**：把“生理图流算子”和“采样干预算子”的交换子

```text
[Phi, Pi] h = Phi(Pi(h)) - Pi(Phi(h))
```

作为采样策略污染的结构性证据。分类器只能使用交换子被压小后的 commutative physiological channel；非交换残差被显式收纳到 policy residual sink 中，用于解释采样流程，但不能进入最终分类边界。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从单一路径编码改为“图流算子 + 采样算子”的可交换分解

保留你当前“采样解耦/反事实干预”框架中的 value process 与 sampling process 分工，但把 sampling branch 从“特征支路”升级为“可微干预算子”：

1. **Physio Graph Flow `Phi_theta`**
   - 输入当前 hidden state、观测值、时间间隔。
   - 学习一个 value-driven physiological graph `A_state`，只由观测值变化和变量间动态响应决定。
   - 用图消息传递近似连续时间 one-step flow，吸收 GSNF 的图流思想，但不复制其 reverse-time / ITG 自监督。

2. **Sampling Intervention Operator `Pi_omega`**
   - 输入 hidden state、mask、times 和一个 counterfactual policy recipe。
   - 输出被该采样政策“可见性裁剪”后的 hidden state。
   - 该算子不估计 hazard，不输出采样概率，也不把 mask embedding 拼入分类器；它只定义 `do(policy)` 如何作用在中间状态上。

3. **Commutator Gate**
   - 计算 `delta_comm = Phi(Pi(h)) - Pi(Phi(h))`。
   - 用 `delta_comm` 生成一个软门控，把 hidden state 分成：
     - `h_phys`：交换子较小、可作为稳定生理证据的通道。
     - `h_policy`：交换子较大、解释采样流程和共观测偏差的残差通道。
   - 分类头只接收 `h_phys`。

### 2.2 改 Loss：从“多视图一致”改为“算子交换性 + 残差收纳”

总目标：

```text
L = L_cls
  + lambda_c * L_comm
  + lambda_s * L_sink
  + lambda_g * L_graph_separation
```

#### A. 分类损失 `L_cls`

标准交叉熵，仅使用 `h_phys`：

```text
L_cls = CE(Classifier(h_phys), y)
```

#### B. 交换子损失 `L_comm`

不要求不同采样视图的 representation 或 logits 完全一致，而是要求“生理演化”和“采样干预”的顺序在分类子空间中近似可交换：

```text
L_comm = || W_cls ( Phi(Pi(h)) - Pi(Phi(h)) ) ||_2^2
```

这里 `W_cls` 是分类头前的投影矩阵或一个轻量 projector。它只压制会影响分类的非交换残差，而不是粗暴抹掉所有采样导致的信息差异。

#### C. 残差收纳损失 `L_sink`

为了避免模型把非交换残差偷偷塞回 `h_phys`，新增一个 policy residual sink：

```text
L_sink = BCE(PolicyDecoder(h_policy), policy_recipe)
```

它让 `h_policy` 有能力解释“本次反事实采样是变量偏置、时间段偏置还是联测偏置”，但由于分类器不接收 `h_policy`，这些流程性信号不会直接变成类别捷径。注意这不是 adversarial loss：我们不训练分类表示去骗过策略判别器，而是给策略残差一个专门的吸收槽。

#### D. 图分离损失 `L_graph_separation`

从 FlowPath/GSNF 的启发出发，采样偏移会改变路径几何和共观测图。CGS 同时学习两个图：

- `A_state`：由 value dynamics 诱导的生理图。
- `A_policy`：由 counterfactual policy recipe 诱导的流程图。

二者不要求正交，也不做原型约束，而是约束分类通道中的图交换子：

```text
L_graph_separation = || P_cls (A_state A_policy - A_policy A_state) P_cls^T ||_F^2
```

若某条边只是医院联测流程造成的，它通常会与 value-driven 生理传播不交换；该项会阻止这种流程边进入分类子空间。

### 2.3 改 Dataloader：生成“算子 recipe”，不是生成一致性增强

新增 `CommutatorPolicyCollator`，为每个 batch 返回若干轻量 recipe：

1. `time_window_bias`：只改变早期/晚期时间窗的可见性。
2. `variable_panel_bias`：模拟某组化验变量被联测或不被联测。
3. `co_measurement_bias`：改变变量对之间的共观测关系。

这些 recipe 不直接生成对比学习正样本，也不要求 logits 方差变小。它们只用于定义 `Pi_omega` 的干预动作，从而测量 `Phi` 与 `Pi` 的交换子。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 可作为 `Phi_theta` 的底座。
- 现有反事实采样模块不再输出多视图一致性目标，而是输出 `policy_recipe`，交给 `Pi_omega` 形成可微采样算子。
- 原 sampling branch 不进入 classifier，而是转化为 policy residual sink 的监督源。
- 推理阶段只需要 `Phi_theta + CommutatorGate + Classifier`；无需采样策略标签，也无需生成反事实视图。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysioGraphFlow(nn.Module):
    """One-step value-driven graph flow for irregular multivariate states."""

    def __init__(self, hidden_dim: int, num_vars: int):
        super().__init__()
        self.num_vars = num_vars
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.msg = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, delta_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [B, D, H], delta_t: [B, 1] or [B, D]
        bsz, num_vars, hidden_dim = h.shape
        hi = h[:, :, None, :].expand(bsz, num_vars, num_vars, hidden_dim)
        hj = h[:, None, :, :].expand(bsz, num_vars, num_vars, hidden_dim)

        if delta_t.dim() == 2:
            dt = delta_t.mean(dim=-1, keepdim=True)
        else:
            dt = delta_t
        dt_pair = dt.view(bsz, 1, 1, 1).expand(bsz, num_vars, num_vars, 1)

        edge_input = torch.cat([hi, hj, dt_pair], dim=-1)
        a_state = torch.sigmoid(self.edge_mlp(edge_input).squeeze(-1))
        a_state = a_state * (1.0 - torch.eye(num_vars, device=h.device)[None])

        messages = torch.einsum("bij,bjh->bih", a_state, self.msg(h))
        h_next = self.update(messages.reshape(-1, hidden_dim), h.reshape(-1, hidden_dim))
        h_next = h_next.view(bsz, num_vars, hidden_dim)
        return h_next, a_state


class SamplingInterventionOperator(nn.Module):
    """Differentiable do(policy) operator acting on hidden graph states."""

    def __init__(self, hidden_dim: int, recipe_dim: int):
        super().__init__()
        self.recipe_to_gate = nn.Sequential(
            nn.Linear(recipe_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.policy_edge = nn.Sequential(
            nn.Linear(recipe_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        mask_summary: torch.Tensor,
        recipe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [B, D, H], mask_summary: [B, D, 1], recipe: [B, R]
        gate = self.recipe_to_gate(recipe)[:, None, :]
        visibility = mask_summary.clamp(0.0, 1.0)
        h_policy = h * (0.2 + 0.8 * visibility) * gate

        # A simple policy graph induced by the intervention recipe.
        raw_edge = self.policy_edge(recipe).view(-1, 1, 1)
        a_policy = torch.sigmoid(raw_edge) * torch.matmul(visibility, visibility.transpose(1, 2))
        return h_policy, a_policy


class CommutatorGate(nn.Module):
    """Split hidden state into commutative physiological and policy-residual channels."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, delta_comm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.gate(torch.cat([h, delta_comm.abs()], dim=-1))
        h_phys = h * (1.0 - score)
        h_policy = h * score
        return h_phys, h_policy


class CommutatorGraphSurgery(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_vars: int,
        num_classes: int,
        recipe_dim: int,
    ):
        super().__init__()
        self.value_embed = nn.Linear(input_dim, hidden_dim)
        self.flow = PhysioGraphFlow(hidden_dim=hidden_dim, num_vars=num_vars)
        self.policy_op = SamplingInterventionOperator(hidden_dim=hidden_dim, recipe_dim=recipe_dim)
        self.commutator_gate = CommutatorGate(hidden_dim=hidden_dim)
        self.cls_projector = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.policy_decoder = nn.Linear(hidden_dim, recipe_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, delta_t: torch.Tensor, recipe: torch.Tensor):
        # values: [B, D, input_dim], mask: [B, T, D] or [B, D]
        h0 = self.value_embed(values)
        mask_summary = mask.float().mean(dim=1, keepdim=False).unsqueeze(-1) if mask.dim() == 3 else mask.unsqueeze(-1)

        # Order 1: first intervene on sampling, then run physiological flow.
        pi_h0, a_policy = self.policy_op(h0, mask_summary, recipe)
        phi_after_pi, _ = self.flow(pi_h0, delta_t)

        # Order 2: first run physiological flow, then intervene on sampling.
        phi_h0, a_state = self.flow(h0, delta_t)
        pi_after_phi, _ = self.policy_op(phi_h0, mask_summary, recipe)

        delta_comm = phi_after_pi - pi_after_phi
        h_phys, h_policy = self.commutator_gate(phi_h0, delta_comm)

        pooled_phys = h_phys.mean(dim=1)
        pooled_policy = h_policy.mean(dim=1)
        logits = self.classifier(pooled_phys)
        policy_logits = self.policy_decoder(pooled_policy)

        return {
            "logits": logits,
            "policy_logits": policy_logits,
            "h_phys": h_phys,
            "h_policy": h_policy,
            "delta_comm": delta_comm,
            "a_state": a_state,
            "a_policy": a_policy,
        }

    def training_loss(
        self,
        batch: dict,
        lambda_c: float = 0.1,
        lambda_s: float = 0.05,
        lambda_g: float = 0.02,
    ) -> dict:
        out = self(
            values=batch["values"],
            mask=batch["mask"],
            delta_t=batch["delta_t"],
            recipe=batch["policy_recipe"],
        )

        cls_loss = F.cross_entropy(out["logits"], batch["labels"])

        # Penalize only the non-commuting component visible to the classifier.
        comm_visible = self.cls_projector(out["delta_comm"]).mean(dim=1)
        comm_loss = comm_visible.pow(2).mean()

        # Let the residual channel absorb policy recipe information without feeding it to the classifier.
        sink_loss = F.binary_cross_entropy_with_logits(
            out["policy_logits"],
            batch["policy_recipe"].float(),
        )

        a_state = out["a_state"]
        a_policy = out["a_policy"]
        graph_comm = torch.bmm(a_state, a_policy) - torch.bmm(a_policy, a_state)
        graph_loss = graph_comm.pow(2).mean()

        total = cls_loss + lambda_c * comm_loss + lambda_s * sink_loss + lambda_g * graph_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "comm_loss": comm_loss.detach(),
            "sink_loss": sink_loss.detach(),
            "graph_loss": graph_loss.detach(),
        }


def build_policy_recipe(
    batch_size: int,
    recipe_dim: int = 3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return binary recipes: early/late bias, variable-panel bias, co-measurement bias."""

    recipe = torch.zeros(batch_size, recipe_dim, device=device)
    active = torch.randint(0, recipe_dim, (batch_size,), device=device)
    recipe[torch.arange(batch_size, device=device), active] = 1.0
    return recipe
```

## 4. 预期创新性

1. **从采样分布估计转向算子代数诊断**：不问“当前采样概率是多少”，而问“采样干预与生理演化是否可交换”。
2. **从多视图一致转向顺序不变性**：不是让两个反事实视图输出相同，而是约束 `do(policy)` 与 physiological flow 的先后顺序不应影响分类证据。
3. **从缺失模式特征转向 policy residual sink**：采样流程被允许存在并被解释，但被收纳到不参与分类的残差槽。
4. **自然吸收 FlowPath/GSNF 的前沿启发**：把路径几何偏移和变量图偏移统一为“生理流-采样算子”的非交换问题。

## 5. 实验切入点

1. 构造三类 policy shift：时间窗偏置、变量 panel 偏置、共观测图偏置。
2. 对比 DHN、mask dropout、policy consistency、missingness-aware encoder、FlowPath、GSNF。
3. 新增指标：
   - classification-visible commutator norm；
   - graph commutator norm；
   - worst-policy accuracy；
   - policy-shift calibration error。
4. 消融：
   - 去掉 `L_comm`，观察是否回到采样捷径；
   - 去掉 residual sink，观察非交换残差是否污染分类通道；
   - 去掉 graph commutator，观察变量联测偏置下的鲁棒性下降。

## 6. 一句话投稿卖点

**CGS 首次把非规则时间序列分类中的采样策略偏移表述为“采样干预算子与生理图流算子的非交换污染”，并通过交换子手术把流程性观测结构从分类证据中切除，为 AAAI 2026 的 sampling-policy shift 鲁棒分类提供一个区别于危险率、对比一致性和缺失模式编码的全新机制。**
