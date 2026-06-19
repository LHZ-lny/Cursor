# Title: Do-Sampler Certified Smoothing：采样政策单纯形上的反事实随机平滑认证

## 0. 强制读取记录与思维黑名单

### 已读取材料

- `paper_daily.md`
- `paper_daily_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-13.md`
- `ideas/Idea_Proposal_2026-06-14.md`
- `ideas/Idea_Proposal_2026-06-16.md`
- 根目录 `README.md`
- `工作素材/README.md`
- `周计划/README.md`
- `周报/README.md`
- 自动化记忆中记录的历史提案摘要，尤其是当前工作区未检出的 `Idea_Proposal_2026-06-17.md` 核心机制。

### 未检出但已确认搜索

- 当前仓库未检出 `my_work_summary.md`。
- 当前仓库也未检出 `*summary*.md`、`*Summary*.md` 或中文 `*总结*.md` 形式的可替代工作总结。

### 历史核心机制黑名单

为避免与历史提案和 `paper_daily.md` 中已有机制重合，本提案明确避开以下方向作为主创新：

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
17. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed 作为主机制。

本提案选择一个新的正交切入点：**不试图把采样信息删除、正交化、征税或除法分解，而是把采样政策偏移看成 policy simplex 上的可度量扰动；利用当前框架中的反事实采样器构造随机平滑分布，训练一个在采样政策半径内可认证不变的 smoothed classifier。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列的采样偏移通常不是一个单独的 mask dropout 噪声，而是一整族可部署策略的变化：医院 A 常规联测，医院 B 只在报警后补测；设备 A 高密度采样早期窗口，设备 B 受电量限制而稀疏采样；同一变量在不同机构中的“未测”语义也可能完全不同。

近期 `paper_daily.md` 给了三个关键启发：

- **PYRREGULAR** 强调真实不规则性需要标准化数据接口和跨数据集评测，这提示我们不应只在单一训练策略上报告平均精度，而应报告 policy perturbation 下的可认证鲁棒范围。
- **iTimER** 说明未观测区域的 uncertainty 本身很重要，但误差分布也可能被 sampling policy 污染；因此与其继续建模误差或伪观测，不如直接问：在多大采样政策扰动内，分类标签不会被改变？
- **Record2Vec** 追求跨医院 portable representation，但其 portability 主要是经验结果；我们可以把“跨采样政策可迁移”推进为一个带半径的统计证书。

当前“采样解耦/反事实干预”框架已经具备两个关键资产：一条 value process 分类主干，以及一条 sampling process / counterfactual intervention 模块。过去的想法大多把 sampling branch 用于去偏、对抗、后验拆分或解释残差。**Do-Sampler Certified Smoothing (DS-CS)** 改变 sampling branch 的角色：

> sampling branch 不再告诉分类器“哪些采样信息要删掉”，而是定义一个围绕当前样本采样政策的反事实扰动分布；最终预测来自在这族 `do(policy)` 下的随机平滑平均，并输出一个 policy-simplex 鲁棒半径。

这样做的优势是：

1. 不强迫每个反事实视图给出完全相同的 representation 或 logits，因此不会抹掉真实 informative observation。
2. 不把 mask、采样概率、协议标签或策略残差输入分类头，降低策略捷径。
3. 不依赖危险率估计、图结构解释或模型空间后验，机制上与历史提案保持正交。
4. 可以给审稿人一个清晰卖点：不仅经验上提升 worst-policy accuracy，还能声明“在 policy simplex 中半径 `r` 内的采样策略变化不会改变预测类别”。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：从单一反事实增强改为 policy simplex 扰动核

新增 `PolicySimplexSmoothingCollator`。它不生成对比学习 pair，也不做 risk variance，而是为每个样本生成一个采样政策中心和一组反事实 policy recipes：

1. **Policy center `pi_0`**
   - 从原始 `times, mask` 提取低维采样策略坐标，例如变量组采样密度、早/中/晚时间窗密度、局部复测强度、panel 联测强度。
   - `pi_0` 只用于定义扰动分布，不输入分类 head。

2. **Logit-normal / Dirichlet do-sampler**
   - 在 policy simplex 上采样：

```text
u_k = logit(pi_0) + sigma * epsilon_k
pi_k = softmax(u_k)
```

   - `sigma` 控制希望认证的 policy 半径。
   - 每个 `pi_k` 混合一组基础 recipe，例如 early-window dense、late-window sparse、lab-panel co-measure、single-variable budget cut。

3. **Soft do-mask**
   - 不需要估计 point-process hazard。
   - 用 recipe basis 产生可微 visibility gate：

```text
gate_k(t, d) = sigmoid(sum_r pi_{k,r} * B_r(t, d))
x_k = x * mask * gate_k
```

   - 训练时可用 soft mask 保持可微；评估时可采样 Bernoulli mask 得到真实反事实政策。

### 2.2 改 Encoder：保留 value encoder，外包 policy smoothing wrapper

DS-CS 不要求重写现有主干。任何当前的 irregular time-series classifier 都可以作为 `base_model(values, times, mask)`：

- 如果已有 value encoder / sampling branch 解耦，则分类 logits 仍只来自 value encoder。
- sampling branch 只输出 `policy_center` 或 recipe logits，用于 do-sampler。
- 推理时可选择：
  - **Fast mode**：只采样少量政策扰动，输出 smoothed probability。
  - **Certified mode**：采样较多扰动，估计 top-1 与 top-2 类概率下界/上界，给出鲁棒半径。

### 2.3 改 Loss：从一致性约束转向随机平滑证书

总目标：

```text
L = L_smooth_ce
  + lambda_cert * L_cert_margin
  + lambda_cov  * L_policy_coverage
```

#### A. 平滑分类损失 `L_smooth_ce`

对 `K` 个反事实采样政策得到概率并平均：

```text
p_bar(y | x) = mean_k softmax(f(do(policy_k, x)))
L_smooth_ce = CE(p_bar, y)
```

注意：这不是 pairwise consistency。模型允许某些政策视图下更不确定或甚至短暂犯错；训练目标只要求 policy-smoothed 后的总体预测正确。

#### B. 认证边界损失 `L_cert_margin`

随机平滑理论中，如果 top-1 类概率与 top-2 类概率之间存在足够大间隔，则在扰动半径内预测不变。对 policy-logit 空间的 logit-normal smoothing，可用高斯平滑近似：

```text
r_cert = sigma / 2 * (Phi^{-1}(p_A_lower) - Phi^{-1}(p_B_upper))
```

训练时使用可微 proxy：

```text
L_cert_margin = mean relu(r_target - r_soft)^2
```

其中 `p_A` 是真实类的平滑概率，`p_B` 是最大竞争类概率。这个损失优化“可认证半径”，而不是压缩所有视图 logits 的方差。

#### C. 政策覆盖损失 `L_policy_coverage`

为了避免 do-sampler 总是在很窄区域采样，加入一个轻量 entropy / coverage 项：

```text
L_policy_coverage = - mean entropy(pi_k)
```

它只作用于采样分布，不作用于分类表示；目的是让模型见过足够多的可部署采样策略邻域。

### 2.4 与当前框架的结合方式

- 现有 value process：作为 base classifier，不需要知道 policy recipe。
- 现有 sampling process：从“去偏分支”改为“policy center estimator”，只定义 policy simplex 上的局部扰动核。
- 现有 counterfactual intervention：从“生成一致性增强/风险视图”改为“生成平滑积分样本”。
- 训练输出：除了 accuracy，还汇报 `mean certified radius`、`certified coverage at radius r`、`worst-policy accuracy`。
- 推理输出：`smoothed_prob`、`pred_label`、`r_cert`。这能直接支撑 AAAI 审稿中的“鲁棒性是否可验证”质疑。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SmoothingConfig:
    num_samples: int = 8
    sigma: float = 0.5
    target_radius: float = 0.35
    min_prob: float = 1e-4


class PolicyCenterEstimator(nn.Module):
    """Estimate local policy-simplex coordinates without feeding them to logits."""

    def __init__(self, num_vars: int, num_recipes: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(4 * num_vars, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_recipes),
        )

    def forward(self, times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # times: [B, T], mask: [B, T, D]
        obs_rate = mask.float().mean(dim=1)

        time_norm = times / times[:, -1:].clamp_min(1e-6)
        early = (time_norm <= 0.33).float().unsqueeze(-1)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).float().unsqueeze(-1)
        late = (time_norm > 0.66).float().unsqueeze(-1)

        early_rate = (mask.float() * early).sum(dim=1) / early.sum(dim=1).clamp_min(1.0)
        mid_rate = (mask.float() * middle).sum(dim=1) / middle.sum(dim=1).clamp_min(1.0)
        late_rate = (mask.float() * late).sum(dim=1) / late.sum(dim=1).clamp_min(1.0)

        features = torch.cat([obs_rate, early_rate, mid_rate, late_rate], dim=-1)
        return self.net(features)


class PolicyRecipeBasis(nn.Module):
    """Map recipe weights on a simplex to soft do-masks over time and variables."""

    def __init__(self, num_recipes: int, num_vars: int, num_time_bins: int = 16):
        super().__init__()
        self.num_time_bins = num_time_bins
        self.recipe_basis = nn.Parameter(torch.randn(num_recipes, num_time_bins, num_vars) * 0.02)
        self.recipe_bias = nn.Parameter(torch.zeros(num_recipes, num_time_bins, num_vars))

    def forward(self, recipe_weight: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        # recipe_weight: [B, K, R], times: [B, T]
        bsz, num_samples, _ = recipe_weight.shape
        _, seq_len = times.shape

        time_norm = times / times[:, -1:].clamp_min(1e-6)
        bin_idx = (time_norm * (self.num_time_bins - 1)).long().clamp(0, self.num_time_bins - 1)

        basis = self.recipe_basis + self.recipe_bias
        # [B, T, R, D]
        event_basis = basis[:, bin_idx].permute(1, 2, 0, 3)
        raw_gate = torch.einsum("bkr,btrd->bktd", recipe_weight, event_basis)
        return torch.sigmoid(raw_gate)


def sample_policy_weights(
    center_logits: torch.Tensor,
    num_samples: int,
    sigma: float,
) -> torch.Tensor:
    """Logit-normal samples on the policy simplex."""

    noise = torch.randn(
        center_logits.size(0),
        num_samples,
        center_logits.size(1),
        device=center_logits.device,
        dtype=center_logits.dtype,
    )
    noisy_logits = center_logits[:, None, :] + sigma * noise
    return torch.softmax(noisy_logits, dim=-1)


def smooth_counterfactual_batch(
    values: torch.Tensor,
    mask: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply soft do-masks generated by policy recipes."""

    # values/mask: [B, T, D], gate: [B, K, T, D]
    soft_mask = mask[:, None].float() * gate
    cf_values = values[:, None] * soft_mask
    return cf_values, soft_mask


def normal_icdf(prob: torch.Tensor, eps: float) -> torch.Tensor:
    prob = prob.clamp(eps, 1.0 - eps)
    normal = torch.distributions.Normal(
        torch.zeros((), device=prob.device, dtype=prob.dtype),
        torch.ones((), device=prob.device, dtype=prob.dtype),
    )
    return normal.icdf(prob)


def certified_radius_proxy(
    smooth_prob: torch.Tensor,
    labels: torch.Tensor,
    sigma: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Differentiable Gaussian-smoothing radius proxy in policy-logit space."""

    true_prob = smooth_prob.gather(1, labels[:, None]).squeeze(1)
    rival_prob = smooth_prob.masked_fill(
        F.one_hot(labels, smooth_prob.size(-1)).bool(),
        -1.0,
    ).max(dim=-1).values

    margin = normal_icdf(true_prob, eps) - normal_icdf(rival_prob, eps)
    return 0.5 * sigma * margin


class DoSamplerCertifiedSmoothing(nn.Module):
    """Wrap an irregular TS classifier with counterfactual policy randomized smoothing.

    The base_model must accept values, times and mask, and return logits.
    Sampling-policy coordinates are used only to define do-sampling kernels.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_vars: int,
        num_recipes: int,
        hidden_dim: int,
        config: SmoothingConfig | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.policy_center = PolicyCenterEstimator(num_vars, num_recipes, hidden_dim)
        self.recipe_basis = PolicyRecipeBasis(num_recipes, num_vars)
        self.config = config or SmoothingConfig()

    def forward_smooth(self, values: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> dict:
        center_logits = self.policy_center(times=times, mask=mask)
        recipe_weight = sample_policy_weights(
            center_logits=center_logits,
            num_samples=self.config.num_samples,
            sigma=self.config.sigma,
        )
        gate = self.recipe_basis(recipe_weight=recipe_weight, times=times)
        cf_values, cf_mask = smooth_counterfactual_batch(values, mask, gate)

        logits = []
        for idx in range(self.config.num_samples):
            logits.append(
                self.base_model(
                    values=cf_values[:, idx],
                    times=times,
                    mask=cf_mask[:, idx],
                )
            )
        logits = torch.stack(logits, dim=1)  # [B, K, C]
        prob = torch.softmax(logits, dim=-1)
        smooth_prob = prob.mean(dim=1).clamp_min(self.config.min_prob)
        smooth_prob = smooth_prob / smooth_prob.sum(dim=-1, keepdim=True)

        return {
            "logits": logits,
            "smooth_prob": smooth_prob,
            "recipe_weight": recipe_weight,
            "center_logits": center_logits,
        }

    def training_loss(
        self,
        batch: dict,
        lambda_cert: float = 0.4,
        lambda_cov: float = 0.02,
    ) -> dict:
        out = self.forward_smooth(
            values=batch["values"],
            times=batch["times"],
            mask=batch["mask"],
        )
        labels = batch["labels"]

        smooth_log_prob = out["smooth_prob"].log()
        smooth_ce = F.nll_loss(smooth_log_prob, labels)

        radius = certified_radius_proxy(
            smooth_prob=out["smooth_prob"],
            labels=labels,
            sigma=self.config.sigma,
            eps=self.config.min_prob,
        )
        cert_loss = F.relu(self.config.target_radius - radius).pow(2).mean()

        recipe_weight = out["recipe_weight"].clamp_min(1e-8)
        entropy = -(recipe_weight * recipe_weight.log()).sum(dim=-1).mean()
        coverage_loss = -entropy / math.log(recipe_weight.size(-1))

        total = smooth_ce + lambda_cert * cert_loss + lambda_cov * coverage_loss
        return {
            "loss": total,
            "smooth_ce": smooth_ce.detach(),
            "cert_loss": cert_loss.detach(),
            "coverage_loss": coverage_loss.detach(),
            "mean_radius": radius.mean().detach(),
        }

    @torch.no_grad()
    def predict_certified(self, batch: dict, num_samples: int = 64) -> dict:
        old_samples = self.config.num_samples
        self.config.num_samples = num_samples
        out = self.forward_smooth(batch["values"], batch["times"], batch["mask"])
        self.config.num_samples = old_samples

        pred = out["smooth_prob"].argmax(dim=-1)
        top2 = out["smooth_prob"].topk(2, dim=-1).values
        normal = torch.distributions.Normal(
            torch.zeros((), device=top2.device),
            torch.ones((), device=top2.device),
        )
        radius = 0.5 * self.config.sigma * (
            normal.icdf(top2[:, 0].clamp(self.config.min_prob, 1.0 - self.config.min_prob))
            - normal.icdf(top2[:, 1].clamp(self.config.min_prob, 1.0 - self.config.min_prob))
        )
        return {
            "pred": pred,
            "smooth_prob": out["smooth_prob"],
            "certified_radius": radius,
        }
```

## 4. 实验切入点

1. **Policy simplex 构造**
   - 坐标包括变量级采样率、时间窗采样率、panel 联测强度、报警后复测强度、设备预算稀疏度。
   - 在 P12/P19/PAM/MIMIC-IV 等数据上人工构造训练策略与测试策略的 simplex 偏移。

2. **对比方法**
   - 普通 mask dropout。
   - policy consistency / contrastive augmentation。
   - missingness-aware encoder。
   - hazard/null-space baseline。
   - commutator graph surgery baseline。
   - protocol-tax evidence market baseline。
   - posterior quotient dynamics baseline。
   - iTimER / Record2Vec 风格表示学习或 portability baseline。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - certified accuracy at radius `r`。
   - mean certified policy radius。
   - abstention-aware accuracy：当 `r_cert` 低于部署阈值时拒识。
   - policy simplex calibration error：高置信预测是否也拥有足够政策半径。

4. **消融实验**
   - 去掉 `L_cert_margin`，验证是否只剩普通采样增强。
   - 把 logit-normal policy smoothing 替换为均匀 mask dropout，验证收益来自 policy simplex 结构。
   - 固定 recipe basis 不学习，验证可学习反事实政策核的重要性。
   - 减少 Monte Carlo 样本数，评估认证半径与计算开销的 trade-off。

## 5. 预期创新性

1. **从经验鲁棒转向可认证鲁棒**：不只报告某些 policy shift 下精度更高，而是给出 policy-simplex 半径内预测不变的统计证书。
2. **从去除采样信息转向平滑积分采样信息**：采样机制不进入分类头，也不被强行删除；它定义反事实扰动核，最终预测对该核做随机平滑。
3. **从 pairwise consistency 转向分布级 smoothed decision**：不要求每个采样视图相同，只要求平均预测有足够 top-1/top-2 间隔。
4. **与现有框架低侵入兼容**：保留 value encoder，sampling branch 改为 policy center estimator，counterfactual intervention 改为 randomized do-sampler。
5. **审稿卖点清晰**：面对 AAAI 审稿人对 sampling-policy shift 的质疑，可以同时给出方法、理论半径、认证指标和可部署拒识策略。

## 6. 一句话投稿卖点

**DS-CS 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为 policy simplex 上的可认证扰动问题，并利用反事实 do-sampler 随机平滑训练出带鲁棒半径的分类器，从而在不依赖危险率、交换子、证据税、后验商或语义误差图的前提下，为跨采样政策部署提供可验证的稳定性保证。**
