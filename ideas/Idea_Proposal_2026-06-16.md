# Title: Posterior Quotient Dynamics：面向采样策略偏移的后验商动力学指纹

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
- 当前仓库 Markdown 检索中也未发现可替代的 `*work_summary*`、`*summary*` 或中文“总结”类工作总结文件。

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
14. additive evidence market、protocol tax、token-level evidence budget 或边际证据审计。
15. 单纯把 DBGL/GARLIC 式 decay、二部图、time-lagged graph 或 attention 解释作为主创新。

本提案选择一个与上述机制正交的切入点：**不估计采样危险率，不做表示一致性，不做对抗，不做图/算子交换，也不给证据征税；而是把每条不规则序列先提升到“连续时间动力学模型空间”的后验分布，再用贝叶斯自然参数的后验商运算，把可由采样策略解释的似然因子从完整后验中除掉，得到只用于分类的 do-free dynamics signature。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中最适合深化当前“采样解耦/反事实干预”框架的两个前沿信号是：

- **DBGL** 把不规则医疗时序表示为 patient-variable-time 观测结构，说明采样策略不是一个全局缺失率，而是变量级、病人级、事件级的结构化观测算子。
- **Fault Diagnosis of Irregular Sequences by Adjoint Learning in Continuous-Time Model Space** 提醒我们：分类对象可以不是原始观测序列，而是“能解释该序列的连续时间模型”。若模型空间参数真正刻画底层动力学，它会比原始 mask、delta-t 或插值路径更接近可迁移机制。

采样策略偏移的核心问题可以被重新表述为：

> 训练环境中的完整后验 `p(dynamics | values, times, mask)` 往往同时吸收了状态动力学证据与采样政策证据；当医院协议、传感器调度或变量联测规则改变时，这个后验会沿着 policy likelihood 的方向漂移。

过去的思路常试图“压住漂移”：让不同采样视图表征一致、让分类梯度避开采样 score、把非交换残差切出去，或者提高高协议证据的成本。**Posterior Quotient Dynamics (PQD)** 采用完全不同的观点：

```text
完整模型后验 ≈ 状态动力学后验 × 采样似然后验
状态动力学后验 ≈ 完整模型后验 / 采样似然后验
```

也就是说，采样分支不再负责生成增强视图、对抗分类器或输出残差槽；它只学习一个“采样似然因子”在模型空间中的自然参数。分类器看到的是完整后验除以该采样因子之后的 **do-free dynamics signature**。这与当前框架天然契合：value process 给出完整动力学解释，sampling process 给出观测算子的似然解释，counterfactual intervention 用来校准“哪些后验变化应归因于采样算子”，最终分类只依赖后验商。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从序列表征改为“模型空间后验商”

PQD 不直接输出一个 pooled representation，而是输出连续时间动力学参数 `theta` 的高斯后验。`theta` 可以是轻量 continuous-time state-space model 的生成矩阵、低秩 Koopman generator、reservoir readout 参数或变量级响应时间常数。

1. **Full Dynamics Posterior `q_full(theta | x, t, m)`**
   - 输入不规则观测值、时间戳和可见性结构。
   - 输出完整模型空间后验的自然参数 `(eta_full, Lambda_full)`。
   - 它允许使用所有信息来解释“这条序列看起来像什么动力学模型”，因此会包含状态信号和采样信号的混合。

2. **Sampling Likelihood Factor `q_policy(theta | t, m, do_recipe)`**
   - 输入只包含时间戳、变量 id、观测事件结构和反事实采样 recipe。
   - 输出采样似然因子在同一个模型空间中的自然参数 `(eta_policy, Lambda_policy)`。
   - 它不是 missingness pattern classifier，因为它的输出不会进入分类头；它只描述“如果只看采样结构，会把动力学后验往哪个方向推”。

3. **Posterior Quotient Layer**
   - 在高斯自然参数空间中执行可微后验商：

```text
Lambda_state = softplus(Lambda_full - tau * Lambda_policy)
eta_state    = eta_full - tau * eta_policy
q_state(theta) = Normal(Lambda_state^{-1} eta_state, Lambda_state^{-1})
```

   - `tau` 是可学习或调度的 quotient strength，控制从完整后验中除去多少采样似然。
   - 分类器只接收 `q_state` 的均值、方差和少量谱摘要，而不是接收 mask embedding、policy code 或 policy residual。

### 2.2 改 Loss：从一致性/对抗/征税转向“后验因式分解 + 干预积分分类”

总目标：

```text
L = L_cls
  + lambda_rec * L_model_reconstruction
  + lambda_q   * L_quotient_factorization
  + lambda_do  * L_interventional_marginalization
  + lambda_var * L_posterior_sobriety
```

#### A. 分类损失 `L_cls`

分类器只使用后验商得到的状态动力学签名：

```text
theta_state ~ q_state(theta)
logits = Classifier([E(theta_state), Var(theta_state), spectral_summary(theta_state)])
L_cls = CE(logits, y)
```

这里的创新不在于换一个 classifier，而在于分类输入已经是“完整模型后验 / 采样似然后验”的商。

#### B. 模型重构损失 `L_model_reconstruction`

为了让 `theta` 真正成为连续时间模型空间对象，而不是普通 embedding，加入轻量 dynamics decoder：

```text
z(t + delta) = exp(delta * A_theta) z(t)
x_hat(t, d) = Readout_d(z(t))
L_model_reconstruction = NLL(x_observed | theta_full, event_index)
```

该项吸收 model-space learning 的思想：模型要能解释观测值动力学，而不是只为分类拟合一个黑箱向量。DBGL 的启发体现在 `event_index = (patient, variable, time)` 的稀疏观测接口上，但不使用二部图消息传递作为核心机制。

#### C. 后验商因式分解 `L_quotient_factorization`

后验商不能任意相减，否则可能把真实状态信号也除掉。PQD 用一个闭环因式分解约束保证：

```text
q_full(theta | x, t, m)
≈ Normalize(q_state(theta) * q_policy(theta | t, m))
```

在高斯自然参数中，乘法对应自然参数相加，因此：

```text
eta_full_hat    = eta_state + tau * eta_policy
Lambda_full_hat = Lambda_state + tau * Lambda_policy
L_quotient_factorization = KL(q_full || q_full_hat)
```

这不是 representation consistency：它不要求不同采样视图的表示相同；它只要求同一条样本的完整后验可以被“状态因子 × 采样因子”解释。

#### D. 干预积分分类 `L_interventional_marginalization`

当前反事实干预模块不再生成对比正样本，也不做 risk variance。它生成若干 `do(policy=k)` 的采样 recipe，用于近似积分掉采样策略：

```text
p(y | do-free state) = Integral p(y | theta_full / theta_policy_k) p0(k) dk
```

训练时使用 Monte Carlo policy integration：

```text
log p(y) = logmeanexp_k log p(y | q_state^{(k)})
L_interventional_marginalization = - log p(y)
```

关键区别：这里不惩罚各 `k` 的 logits 差异，也不惩罚风险方差；不同干预下的后验商可以不一样，模型只需在把采样策略作为 nuisance 积分掉后仍能给出正确类别概率。

#### E. 后验清醒度 `L_posterior_sobriety`

为了防止模型把 quotient layer 变成无约束放大器，加入一个方差清醒度约束：

```text
L_posterior_sobriety = mean(relu(Var(q_state) - Var(q_full) * r_max))
```

含义是：除去采样似然后，状态后验可以更不确定，但不能无限膨胀。这样能把“采样策略提供的信息被移除后模型不确定性上升”显式量化出来，服务于跨政策部署时的校准。

### 2.3 改 Dataloader：返回“稀疏事件索引 + 干预 recipe”，不是返回增强视图

新增 `PosteriorQuotientCollator`，每个 batch 返回：

1. `event_value`：观测值。
2. `event_time`：连续时间戳。
3. `event_var_id`：变量 id。
4. `event_patient_id` 或 batch 内样本 id。
5. `event_index`：稀疏 `(sample, variable, time_bucket)` 索引，用于 model reconstruction。
6. `policy_recipe_bank`：反事实采样 recipe，例如：
   - `routine_panel_shift`：常规 panel 频率改变；
   - `selective_lab_shift`：特定变量只在疑似风险时观测；
   - `device_budget_shift`：设备预算限制导致某些时间窗稀疏；
   - `followup_protocol_shift`：事件后复测窗口改变。

注意：collator 不需要真的构造 K 份新序列，也不需要做 mask dropout。它只把 recipe 交给 `SamplingLikelihoodFactor`，让后验商层计算“若当前后验中这部分漂移可由该采样算子解释，应从分类签名中除去多少”。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 升级为 `FullDynamicsPosteriorEncoder`，输出模型空间完整后验。
- 现有 sampling branch 升级为 `SamplingLikelihoodFactor`，输出同一模型空间内的采样似然自然参数。
- 现有 counterfactual intervention 模块保留，但从“生成多视图训练目标”改为“提供 policy recipe bank 以近似积分掉采样似然”。
- 推理阶段可以使用两种模式：
  - **Single-policy quotient**：使用观测到的采样结构估计 `q_policy`，快速得到 `q_state`。
  - **Policy-marginal quotient**：对多个标准 recipe 做 Monte Carlo integration，得到更保守的跨政策预测和不确定性。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def positive_precision(raw: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    return F.softplus(raw) + floor


class NaturalGaussian:
    """Diagonal Gaussian represented by natural parameters."""

    def __init__(self, eta: torch.Tensor, precision: torch.Tensor):
        self.eta = eta
        self.precision = precision

    @property
    def mean(self) -> torch.Tensor:
        return self.eta / self.precision.clamp_min(1e-6)

    @property
    def var(self) -> torch.Tensor:
        return 1.0 / self.precision.clamp_min(1e-6)


class FullDynamicsPosteriorEncoder(nn.Module):
    """Amortize q_full(theta | values, times, observed event structure)."""

    def __init__(self, num_vars: int, hidden_dim: int, theta_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.event_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.to_eta = nn.Linear(hidden_dim, theta_dim)
        self.to_precision = nn.Linear(hidden_dim, theta_dim)

    def forward(
        self,
        event_value: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> NaturalGaussian:
        # event_*: [B, N], event_mask: [B, N]
        var_h = self.var_embed(event_var_id)
        event_x = torch.cat(
            [var_h, event_value.unsqueeze(-1), torch.log1p(event_time).unsqueeze(-1)],
            dim=-1,
        )
        event_h = self.event_mlp(event_x) * event_mask.unsqueeze(-1)
        seq_h, _ = self.event_rnn(event_h)

        denom = event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (seq_h * event_mask.unsqueeze(-1)).sum(dim=1) / denom
        return NaturalGaussian(
            eta=self.to_eta(pooled),
            precision=positive_precision(self.to_precision(pooled)),
        )


class SamplingLikelihoodFactor(nn.Module):
    """Estimate the policy likelihood factor in the same model space."""

    def __init__(self, num_vars: int, recipe_dim: int, hidden_dim: int, theta_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.recipe_proj = nn.Linear(recipe_dim, hidden_dim)
        self.policy_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_eta = nn.Linear(hidden_dim, theta_dim)
        self.to_precision = nn.Linear(hidden_dim, theta_dim)

    def forward(
        self,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
        recipe: torch.Tensor,
    ) -> NaturalGaussian:
        # recipe: [B, R]
        var_h = self.var_embed(event_var_id)
        recipe_h = self.recipe_proj(recipe)[:, None, :].expand_as(var_h)
        gap = torch.zeros_like(event_time)
        gap[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)

        policy_x = torch.cat(
            [var_h, recipe_h, torch.log1p(event_time).unsqueeze(-1), torch.log1p(gap).unsqueeze(-1)],
            dim=-1,
        )
        policy_h = self.policy_mlp(policy_x) * event_mask.unsqueeze(-1)
        denom = event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = policy_h.sum(dim=1) / denom
        return NaturalGaussian(
            eta=self.to_eta(pooled),
            precision=positive_precision(self.to_precision(pooled)),
        )


class PosteriorQuotientLayer(nn.Module):
    """Compute q_state(theta) proportional to q_full(theta) / q_policy(theta)^tau."""

    def __init__(self, theta_dim: int):
        super().__init__()
        self.log_tau = nn.Parameter(torch.zeros(theta_dim))

    def forward(self, full: NaturalGaussian, policy: NaturalGaussian) -> NaturalGaussian:
        tau = F.softplus(self.log_tau).view(1, -1)
        state_precision = positive_precision(full.precision - tau * policy.precision)
        state_eta = full.eta - tau * policy.eta
        return NaturalGaussian(eta=state_eta, precision=state_precision)

    def refactorize(self, state: NaturalGaussian, policy: NaturalGaussian) -> NaturalGaussian:
        tau = F.softplus(self.log_tau).view(1, -1)
        return NaturalGaussian(
            eta=state.eta + tau * policy.eta,
            precision=positive_precision(state.precision + tau * policy.precision),
        )


def diagonal_gaussian_kl(q: NaturalGaussian, p: NaturalGaussian) -> torch.Tensor:
    q_mean, p_mean = q.mean, p.mean
    q_var, p_var = q.var, p.var
    kl = 0.5 * (
        torch.log(p_var.clamp_min(1e-6)) - torch.log(q_var.clamp_min(1e-6))
        + (q_var + (q_mean - p_mean).pow(2)) / p_var.clamp_min(1e-6)
        - 1.0
    )
    return kl.sum(dim=-1).mean()


class DynamicsDecoder(nn.Module):
    """Lightweight model-space decoder for observed event values."""

    def __init__(self, theta_dim: int, hidden_dim: int, num_vars: int):
        super().__init__()
        self.init_proj = nn.Linear(theta_dim, hidden_dim)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        theta_mean: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
    ) -> torch.Tensor:
        base = self.init_proj(theta_mean)[:, None, :]
        time_h = self.time_proj(torch.log1p(event_time).unsqueeze(-1))
        var_h = self.var_embed(event_var_id)
        return self.readout(torch.cat([base.expand_as(time_h), time_h, var_h], dim=-1)).squeeze(-1)


class PosteriorQuotientDynamics(nn.Module):
    def __init__(
        self,
        num_vars: int,
        recipe_dim: int,
        hidden_dim: int,
        theta_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.full_encoder = FullDynamicsPosteriorEncoder(num_vars, hidden_dim, theta_dim)
        self.policy_factor = SamplingLikelihoodFactor(num_vars, recipe_dim, hidden_dim, theta_dim)
        self.quotient = PosteriorQuotientLayer(theta_dim)
        self.decoder = DynamicsDecoder(theta_dim, hidden_dim, num_vars)
        self.classifier = nn.Sequential(
            nn.Linear(2 * theta_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def quotient_logits(self, batch: dict, recipe: torch.Tensor) -> dict:
        full = self.full_encoder(
            event_value=batch["event_value"],
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
        )
        policy = self.policy_factor(
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
            recipe=recipe,
        )
        state = self.quotient(full, policy)
        signature = torch.cat([state.mean, torch.log1p(state.var)], dim=-1)
        logits = self.classifier(signature)
        return {"logits": logits, "full": full, "policy": policy, "state": state}

    def training_loss(
        self,
        batch: dict,
        lambda_rec: float = 0.2,
        lambda_q: float = 0.1,
        lambda_do: float = 0.3,
        lambda_var: float = 0.02,
        r_max: float = 4.0,
    ) -> dict:
        labels = batch["labels"]
        factual_recipe = batch["policy_recipe"]
        out = self.quotient_logits(batch, factual_recipe)

        cls_loss = F.cross_entropy(out["logits"], labels)

        recon = self.decoder(
            theta_mean=out["full"].mean,
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
        )
        rec_raw = F.smooth_l1_loss(recon, batch["event_value"], reduction="none")
        rec_loss = (rec_raw * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

        full_hat = self.quotient.refactorize(out["state"], out["policy"])
        quotient_loss = diagonal_gaussian_kl(out["full"], full_hat)

        # Monte Carlo integration over policy recipes, not pairwise consistency.
        log_probs = []
        for recipe in batch["policy_recipe_bank"].unbind(dim=1):
            recipe_out = self.quotient_logits(batch, recipe)
            logp = F.log_softmax(recipe_out["logits"], dim=-1)
            log_probs.append(logp.gather(1, labels[:, None]).squeeze(1))
        do_logp = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0)
        do_logp = do_logp - torch.log(torch.tensor(len(log_probs), device=do_logp.device, dtype=do_logp.dtype))
        do_loss = -do_logp.mean()

        var_limit = out["full"].var.detach() * r_max
        sobriety_loss = F.relu(out["state"].var - var_limit).mean()

        total = (
            cls_loss
            + lambda_rec * rec_loss
            + lambda_q * quotient_loss
            + lambda_do * do_loss
            + lambda_var * sobriety_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "quotient_loss": quotient_loss.detach(),
            "do_loss": do_loss.detach(),
            "sobriety_loss": sobriety_loss.detach(),
        }


def build_policy_recipe_bank(batch_size: int, recipe_dim: int, num_recipes: int, device: torch.device) -> torch.Tensor:
    """Return a recipe bank used for policy marginalization."""

    recipes = torch.zeros(batch_size, num_recipes, recipe_dim, device=device)
    base = torch.arange(num_recipes, device=device) % recipe_dim
    recipes[:, torch.arange(num_recipes, device=device), base] = 1.0
    recipes = recipes + 0.05 * torch.rand_like(recipes)
    return recipes.clamp(0.0, 1.0)
```

## 4. 预期创新性

1. **从“去偏约束”转向“后验商运算”**：不是惩罚模型不要用采样信息，而是在模型空间中显式执行 `完整后验 / 采样似然后验`。
2. **从观测序列表征转向连续时间模型空间指纹**：分类对象是可解释底层动力学的 posterior signature，吸收 adjoint model-space 的优势，但面向 sampling-policy shift 重构其因果语义。
3. **从反事实增强转向干预积分**：counterfactual intervention 只提供 policy recipe bank，用于积分掉 nuisance policy，而不是构造一致性 pair、对比样本或风险方差。
4. **从结构化观测图转向稀疏测量算子**：借鉴 DBGL 对 patient-variable-time 事件结构的重视，但不把二部图/decay/attention 当作核心；观测结构只用于估计采样似然因子。
5. **不确定性天然可解释**：若某个预测高度依赖训练医院的采样流程，除以 policy factor 后 `q_state` 方差会升高，可作为跨医院部署的 abstention 或 calibration 信号。

## 5. 实验切入点

1. 构造四类 policy shift：
   - 变量级 selective lab policy 反转；
   - routine panel 频率改变；
   - device budget 导致晚期时间窗稀疏；
   - follow-up protocol 的复测窗口从短延迟改为长延迟。
2. 对比：
   - missingness-aware encoder；
   - policy adversarial baseline；
   - hazard/null-space baseline；
   - commutator graph surgery baseline；
   - protocol-tax evidence market baseline；
   - DBGL / GARLIC 风格图衰减模型；
   - CT-Res / adjoint model-space 但不做后验商的版本。
3. 新增指标：
   - worst-policy accuracy；
   - policy-marginal negative log likelihood；
   - posterior quotient variance under shift；
   - quotient factorization KL；
   - policy reliance score，即 `||tau * eta_policy|| / ||eta_full||`。
4. 消融：
   - 去掉 posterior quotient，直接用 `q_full` 分类；
   - 去掉 `L_quotient_factorization`，验证后验商是否退化为任意相减；
   - 去掉 policy integration，只用 factual recipe；
   - 将 `q_policy` 替换为随机因子，验证收益不是额外参数量；
   - 将 model-space decoder 替换为普通 pooled encoder，验证连续时间动力学指纹的必要性。

## 6. 一句话投稿卖点

**PQD 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“模型空间后验被采样似然污染”的问题，并通过可微后验商 `q_state ∝ q_full / q_policy` 直接从连续时间动力学指纹中除去策略因子，从而在不依赖对抗、一致性、危险率、交换子或证据征税的前提下提升跨采样政策鲁棒性。**
