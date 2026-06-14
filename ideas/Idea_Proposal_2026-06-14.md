# Title: Protocol-Taxed Additive Evidence Market：面向采样策略偏移的协议征税可加证据市场

## 0. 强制去重记录与思维黑名单

### 已读取材料

- `paper_daily.md`
- `paper_daily_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-13.md`
- 根目录 `README.md`
- `工作素材/README.md`
- `周计划/README.md`
- `周报/README.md`

### 未检出但已确认搜索

- 当前仓库未检出 `my_work_summary.md`。
- 当前仓库未检出任何 `*summary*.md` 或 `*work*.md` 命名的 Markdown 总结文件。

### 历史核心机制黑名单

为了避免与历史提案和 paper daily 中已有机制发生思维重合，本提案明确避开以下机制作为主创新：

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

本提案选择一个与上述机制正交的切入点：**不估计采样概率，不要求跨策略表征一致，不做对抗去偏，不构造交换子，也不把采样残差另设为分类外支路；而是把每个观测证据视为一个可购买的 additive evidence token，并对“更可能由采样协议制造的证据”收取更高协议税。分类器必须在有限税务预算内组合证据，从而降低对策略性观测捷径的依赖。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中的 SuperMAN 和 GARLIC 给出两个很适合与“采样解耦/反事实干预”框架结合的新机制：

- **SuperMAN** 把 temporally sparse heterogeneous data 建成可解释的图/信号集合，并使用 additive networks 让单变量、变量子集和全局证据的贡献可分解。
- **GARLIC** 强调 ICU 不规则时序中 observation-level、signal-level、edge-level 解释的重要性，并用 exponential-decay encoder 与 decoupled optimization 稳定分类和辅助目标。

这两者共同提示：sampling-policy shift 的危险不只是“mask 变了”，而是模型可能把某些**高协议杠杆证据**当成病理证据。例如：

- 某个实验室 panel 在医院 A 中只对危重病人联测，在医院 B 中按常规批量测；
- 某个变量长时间未测在训练医院意味着“稳定”，在测试医院可能只是排班或设备限制；
- 某个滞后边或变量子集的重要性看似可解释，实际解释的是临床流程，而不是稳定生理机制。

传统做法通常有三条路：把 mask 当特征、让不同采样增强保持一致、或学习采样过程后对抗/正交去除。但这些都存在问题：mask 特征容易泄漏；一致性可能抹掉真实信息差异；对抗或零空间约束容易训练不稳定，并且会把“有用但高风险”的证据一刀切掉。

**Protocol-Taxed Additive Evidence Market (PT-AEM)** 的核心直觉是：

> 不禁止模型使用高风险采样证据，而是让它为这类证据付出“协议税”。只有当观测值本身提供足够强的病理增益时，高税证据才值得被分类器购买；如果一个 token 主要因为采样协议而显得重要，它会在有限预算下被自然挤出决策组合。

这与当前“采样解耦/反事实干预”框架天然兼容：采样分支不再输出危险率、对抗标签或多视图一致性目标，而是输出 token-level policy price；反事实干预模块不再服务于风险方差约束，而是用于估计不同采样协议下哪些 token 的出现更依赖流程规则。分类主干仍然只消费 value-derived additive evidence，但证据选择过程受到协议税预算约束。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从整体表征改为“可加证据 token 市场”

保留当前 value process 与 sampling process 解耦思想，但把分类主干改成三层可加证据结构：

1. **Event Evidence Token**
   - 每个观测事件 `(variable, time, value, delta_t)` 生成一个 token。
   - token embedding 只由观测值、变量 id 和相对时间衰减产生，不把 mask pattern 直接拼进分类器。
   - 这里吸收 GARLIC 的 exponential decay 思想：旧观测可以产生衰减后的证据，但“多久未测”不自动等价于类别证据。

2. **Subset Additive Evidence**
   - 借鉴 SuperMAN 的 additive networks，把 token 聚合为单变量证据、变量组证据和全局证据。
   - 每个 token 或 subset 都输出可解释的 class-wise logit contribution `v_i`。
   - 最终 logits 是被购买证据的加权和：

```text
logits = sum_i alpha_i * v_i
```

3. **Protocol Taxed Selector**
   - sampling branch 为每个 token 估计一个协议税 `c_i >= 0`，表示该 token 的出现、联测关系或时间位置对采样协议的依赖程度。
   - selector 根据 value quality 与 protocol tax 共同决定购买权重：

```text
alpha_i = softmax((q_i - eta * c_i) / temperature)
```

关键点：`c_i` 不作为分类特征输入 logit head；它只改变证据市场中的购买成本。模型可以使用高税 token，但必须用足够高的 value quality 抵消税费。

### 2.2 改 Loss：从不变性/对抗/交换子转向“协议预算 + 边际证据审计”

总目标：

```text
L = L_cls
  + lambda_budget * L_tax_budget
  + lambda_audit  * L_marginal_audit
  + lambda_price  * L_price_decoupled
```

#### A. 分类损失 `L_cls`

只用被 selector 购买后的 additive evidence 做分类：

```text
L_cls = CE(sum_i alpha_i * v_i, y)
```

#### B. 协议税预算 `L_tax_budget`

限制单个样本用于分类的总协议税：

```text
tax_spend = sum_i alpha_i * stopgrad(c_i)
L_tax_budget = relu(tax_spend - B_y)^2
```

其中 `B_y` 可以是类别自适应预算。少数类或本来观测更稀疏的类别不应被过度惩罚，因此预算可以用训练集统计或小网络从 value-only difficulty 估计，但不能由 mask pattern 直接决定。

#### C. 边际证据审计 `L_marginal_audit`

为了避免模型把一个高税 token 做成极大的 logit 捷径，对每个 token 的真实类 margin contribution 做上限审计：

```text
margin_i = v_i[y] - max_{k != y} v_i[k]
allowed_i = kappa * value_surprise_i / (1 + c_i)
L_marginal_audit = mean_i relu(|margin_i| - allowed_i)^2
```

`value_surprise_i` 来自 value encoder 的预测残差或局部变化幅度：如果一个观测值真的异常、提供强生理信号，即使协议税较高也能获得更高边际额度；如果它只是因为某个医院流程被频繁测到而显得重要，则 margin 会被压低。

这不是跨视图一致性：我们不要求反事实采样下 logits 一样；也不是对抗学习：我们不训练分类表示去骗过策略判别器。它只是在 additive evidence 层面对“每个被购买证据贡献了多少、付了多少协议税”做可审计约束。

#### D. Decoupled Price Learning `L_price_decoupled`

吸收 GARLIC 的 decoupled optimization 思想，把协议税估计器与分类器交替训练：

1. **Price step**：冻结 value encoder 和 classifier，用反事实 sampling recipes 训练 price estimator 预测 token 是否由某类协议动作引入。
2. **Classifier step**：冻结或 stop-gradient `c_i`，训练 selector 与 additive classifier，使其在协议预算内完成分类。

`L_price_decoupled` 可用多标签 BCE：

```text
L_price_decoupled = BCE(PriceHead(token_context), protocol_action_tag)
```

该损失不进入分类表示反向去偏，只负责给证据市场提供价格。

### 2.3 改 Dataloader：生成“协议拍卖账本”，不是一致性增强样本

新增 `ProtocolAuctionCollator`，每个 batch 返回：

1. `event_tokens`：由原始不规则观测展开得到的事件级 token。
2. `token_group`：变量组、panel、时间窗或滞后桶 id，用于 additive subset aggregation。
3. `protocol_action_tag`：反事实干预模块给出的 token-level 协议动作标签，例如：
   - `late_window_panel`：晚期集中 panel 检测；
   - `co_ordered_lab`：同一医嘱触发的联测；
   - `stale_vital_reuse`：长时间未测但被 decay encoder 反复使用的旧生命体征；
   - `alarm_followup`：报警后短窗口密集复测。
4. `value_surprise`：value-only 预测误差、局部斜率或相邻观测变化幅度。

反事实干预在这里的作用是**标价**而非制造正样本：对同一条轨迹施加不同 sampling recipes，观察哪些 token 或 token-group 的出现对协议动作敏感，由此训练 `PriceHead`。训练分类器时不需要同时输入多个视图，也不惩罚视图间 logits 差异。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 继续负责生成观测值驱动的 event embeddings。
- 现有 sampling branch 改为 `ProtocolPriceEstimator`，输出 token-level tax `c_i`，不进入 classifier。
- 现有反事实采样模块改为生成 `protocol_action_tag` 与 token 价格监督，而不是生成 hazard thinning 视图、risk variance 视图或 contrastive pair。
- 推理阶段可以选择两种模式：
  - **Robust mode**：使用 `PriceHead + TaxedSelector + AdditiveClassifier`，适合跨医院/跨设备部署。
  - **Audit mode**：输出每个 token 的 `alpha_i`、`c_i`、`alpha_i * c_i` 和 class contribution，解释模型是否依赖高协议税证据。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class EventEvidenceEncoder(nn.Module):
    """Encode irregular observations into value-derived evidence tokens."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.value_proj = nn.Linear(2, hidden_dim)
        self.decay_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        value: torch.Tensor,
        delta_t: torch.Tensor,
        var_id: torch.Tensor,
    ) -> torch.Tensor:
        # value, delta_t: [B, N], var_id: [B, N]
        value_input = torch.stack([value, torch.log1p(delta_t)], dim=-1)
        value_feat = self.value_proj(value_input)
        decayed = value_feat * self.decay_gate(delta_t.unsqueeze(-1))
        var_feat = self.var_embed(var_id)
        return self.out(torch.cat([decayed, var_feat], dim=-1))


class ProtocolPriceEstimator(nn.Module):
    """Estimate protocol tax for each token without feeding policy features to logits."""

    def __init__(self, hidden_dim: int, num_actions: int):
        super().__init__()
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.tax_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, token_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # token_h: [B, N, H]
        action_logits = self.action_head(token_h)
        tax = self.tax_head(token_h).squeeze(-1)
        return tax, action_logits


class TaxedAdditiveClassifier(nn.Module):
    """Buy additive evidence under a protocol-tax budget."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.quality = nn.Linear(hidden_dim, 1)
        self.contrib = nn.Linear(hidden_dim, num_classes)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(
        self,
        token_h: torch.Tensor,
        tax: torch.Tensor,
        token_mask: torch.Tensor,
        eta: float = 1.0,
        temperature: float = 0.5,
    ) -> dict:
        # token_h: [B, N, H], tax/token_mask: [B, N]
        quality = self.quality(token_h).squeeze(-1)
        score = (quality - eta * tax.detach()) / temperature
        score = score.masked_fill(token_mask == 0, -1e4)
        alpha = torch.softmax(score, dim=-1)

        contribution = self.contrib(token_h)
        logits = torch.einsum("bn,bnc->bc", alpha, contribution) + self.bias
        tax_spend = (alpha * tax.detach()).sum(dim=-1)
        return {
            "logits": logits,
            "alpha": alpha,
            "contribution": contribution,
            "tax_spend": tax_spend,
        }


class ProtocolTaxedEvidenceMarket(nn.Module):
    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        num_classes: int,
        num_protocol_actions: int,
    ):
        super().__init__()
        self.encoder = EventEvidenceEncoder(num_vars=num_vars, hidden_dim=hidden_dim)
        self.price = ProtocolPriceEstimator(
            hidden_dim=hidden_dim,
            num_actions=num_protocol_actions,
        )
        self.classifier = TaxedAdditiveClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

    def forward(self, batch: dict, eta: float = 1.0) -> dict:
        token_h = self.encoder(
            value=batch["event_value"],
            delta_t=batch["event_delta_t"],
            var_id=batch["event_var_id"],
        )
        tax, action_logits = self.price(token_h)
        out = self.classifier(
            token_h=token_h,
            tax=tax,
            token_mask=batch["event_mask"],
            eta=eta,
        )
        out.update({
            "token_h": token_h,
            "tax": tax,
            "action_logits": action_logits,
        })
        return out

    def training_loss(
        self,
        batch: dict,
        lambda_budget: float = 0.1,
        lambda_audit: float = 0.05,
        lambda_price: float = 0.2,
        tax_budget: float = 0.35,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]

        cls_loss = F.cross_entropy(out["logits"], labels)

        budget_loss = F.relu(out["tax_spend"] - tax_budget).pow(2).mean()

        # Token-level margin contribution audit.
        contrib = out["contribution"]
        true_margin = contrib.gather(
            dim=-1,
            index=labels[:, None, None].expand(-1, contrib.size(1), 1),
        ).squeeze(-1)
        masked_contrib = contrib.masked_fill(
            F.one_hot(labels, contrib.size(-1)).bool()[:, None, :],
            -1e4,
        )
        rival_margin = masked_contrib.max(dim=-1).values
        token_margin = (true_margin - rival_margin).abs()

        value_surprise = batch["value_surprise"].clamp_min(0.0)
        allowed = batch.get("audit_kappa", 1.0) * value_surprise / (1.0 + out["tax"].detach())
        audit_raw = F.relu(token_margin - allowed).pow(2)
        audit_loss = (audit_raw * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

        # Decoupled price supervision from counterfactual protocol-action tags.
        price_loss = F.binary_cross_entropy_with_logits(
            out["action_logits"],
            batch["protocol_action_tag"].float(),
            reduction="none",
        )
        price_loss = (price_loss.mean(dim=-1) * batch["event_mask"]).sum()
        price_loss = price_loss / batch["event_mask"].sum().clamp_min(1.0)

        total = (
            cls_loss
            + lambda_budget * budget_loss
            + lambda_audit * audit_loss
            + lambda_price * price_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "budget_loss": budget_loss.detach(),
            "audit_loss": audit_loss.detach(),
            "price_loss": price_loss.detach(),
            "mean_tax_spend": out["tax_spend"].mean().detach(),
        }


@torch.no_grad()
def build_protocol_action_tags(
    event_time: torch.Tensor,
    event_var_id: torch.Tensor,
    panel_ids: torch.Tensor,
    late_threshold: float,
    num_actions: int = 4,
) -> torch.Tensor:
    """Create token-level protocol-action tags for the price estimator.

    The tags describe which sampling intervention could have introduced a token.
    They are used for pricing only, not as classifier inputs.
    """

    # event_time: [B, N], event_var_id/panel_ids: [B, N]
    tags = torch.zeros(*event_time.shape, num_actions, device=event_time.device)

    late_window = event_time >= late_threshold
    repeated_panel = panel_ids >= 0
    high_frequency_var = event_var_id.unsqueeze(-1) == event_var_id.unsqueeze(-2)
    local_repeat = high_frequency_var.float().sum(dim=-1) > 1

    tags[..., 0] = late_window.float()
    tags[..., 1] = repeated_panel.float()
    tags[..., 2] = (late_window & repeated_panel).float()
    tags[..., 3] = local_repeat.float()
    return tags
```

## 4. 预期创新性

1. **从“去除采样信息”转向“给采样信息定价”**：高协议依赖证据不是被粗暴删除，而是在证据市场中变贵，只有强 value surprise 才能抵消税费。
2. **从整体 representation 鲁棒转向 token-level evidence auditing**：每个观测事件、变量组或时间窗都可解释为“贡献了多少 logit、花了多少协议税”。
3. **从一致性增强转向协议拍卖账本**：反事实采样不再用于构造正样本或风险方差，而是用于标记哪些 token 对协议动作敏感。
4. **自然吸收 SuperMAN 与 GARLIC 的前沿机制**：用 additive evidence 保留可解释性，用 decay token 处理不规则旧观测，用 decoupled optimization 训练价格分支，但不复用它们的图解释或分类目标。
5. **与采样解耦框架高度兼容**：value process 负责证据，sampling process 负责价格，counterfactual intervention 负责价格监督，最终分类器只在预算内购买 value-derived evidence。

## 5. 实验切入点

1. 构造四类 policy shift：
   - panel 联测策略改变；
   - late-window 密集复测策略改变；
   - alarm-followup 高频采样策略改变；
   - stale vital reuse，即旧生命体征长期未更新但被模型反复使用。
2. 对比：
   - mask dropout；
   - missingness-aware encoder；
   - policy adversarial baseline；
   - hazard/null-space baseline；
   - commutator graph surgery baseline；
   - SuperMAN/GARLIC 风格可解释图或 decay encoder。
3. 新增指标：
   - worst-policy accuracy；
   - mean protocol tax spend；
   - high-tax evidence reliance ratio；
   - token-level audit violation rate；
   - prediction-level tax calibration，即错误样本是否具有更高协议税支出。
4. 消融：
   - 去掉 `L_tax_budget`，验证模型是否重新购买高协议税捷径；
   - 去掉 `L_marginal_audit`，验证单个高税 token 是否出现极端 logit 贡献；
   - 去掉 decoupled price step，验证价格估计与分类联合训练是否导致税率塌缩；
   - 把 tax 替换为随机成本，验证收益不是稀疏 attention 本身带来的。

## 6. 一句话投稿卖点

**PT-AEM 首次把非规则采样时间序列中的采样策略偏移表述为“证据市场中的协议税问题”：分类器不再盲目利用被医院流程、设备调度或联测协议制造出的观测捷径，而是在可解释的 additive evidence 层面为每个 token 支付策略依赖成本，从而在保留真实强信号的同时提升跨采样政策鲁棒性。**
