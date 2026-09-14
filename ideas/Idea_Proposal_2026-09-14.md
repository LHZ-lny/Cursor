# Title: Do-Atlas Antipodal Adapter：面向采样策略偏移的源域反政策近邻契约分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md`，纳入其中记录但当前工作区未完全落盘的历史 proposal 摘要与机制黑名单。
- 已读取 `paper_daily.md` 的开头、最新末段与 `paper_daily_2026-09-13.md`，重点纳入：
  - **INPUTADAPTER**：冻结时间序列预测器，学习 input-space transformation，并在源域 latent space 中检索近邻源样本辅助跨医院适配。
  - **OpenTSLM**：用 Perceiver-style resampler / gated cross-attention 将多变量医疗时序作为原生模态接入语言模型，提示解释接口也会被采样政策偏移污染。
- 已读取当前工作区内全部历史 proposal 文件：
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
  - `ideas/Idea_Proposal_2026-07-28.md`
  - `ideas/Idea_Proposal_2026-07-30.md`
  - `ideas/Idea_Proposal_2026-08-05.md`
  - `ideas/Idea_Proposal_2026-08-06.md`
  - `ideas/Idea_Proposal_2026-08-08.md`
  - `ideas/Idea_Proposal_2026-08-09.md`
  - `ideas/Idea_Proposal_2026-08-22.md`
  - `ideas/Idea_Proposal_2026-08-23.md`
  - `ideas/Idea_Proposal_2026-08-24.md`
  - `ideas/Idea_Proposal_2026-08-25.md`
  - `ideas/Idea_Proposal_2026-08-26.md`
  - `ideas/Idea_Proposal_2026-09-12.md`
  - `ideas/Idea_Proposal_2026-09-13.md`

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下机制作为主创新：

1. learnable reference points / adaptive time encoding、频域掩码修复、prototype 约束、missingness pattern 直接分类、简单 policy adversarial / IRM。
2. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
3. 生理流-采样算子交换子、value/policy graph 分离、policy residual sink。
4. protocol tax / additive evidence market、posterior quotient、density ratio / doubly robust、policy-simplex randomized smoothing。
5. reconstruction error cartography、VQ clauses、optional-stopping martingale、censored topology、policy gauge、syndrome code、knockoff calendar。
6. observability witness、evidential vacuity、information lattice、solver trace front-door、conformal sleeves、IV/control-function、Borda jury。
7. Krylov annihilator、Nystrom volume、tropical route、fixed viva、sequent proof、disease-progress poset clock、feasible hull、IRT-DIF、RG fixed point。
8. bitemporal curtain、clinical tomography、matched risk-set likelihood、Gaussian privacy cloak、CauKer orthogonal synthetic forge、JEPA 双辩裁判。
9. cross-representation PID prism、Noether semantic action、fast/slow meta workflow adapter、test-time workflow adaptation。
10. ORA collider factorization、explain-away responsibility gate、workflow-swap path-specific seal。

本提案选择新的正交切入点：**不重训分类器、不做元学习内循环、不把 workflow 封印成 collider，也不把多视图做一致性、投票、保形、隐私或信息分解；而是在冻结源域预测器前构造一个可查询的 Source Atlas。目标域输入适配器只有在把样本映射到“病理语义相近、采样政策相反或均衡”的源域邻域时才被允许信任。分类依据来自冻结预测器与源域反政策近邻契约，而不是来自目标域采样日历的最近邻捷径。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 `paper_daily_2026-09-13.md` 暴露了一个此前历史方案没有直接占用的现实部署缝隙。

第一，**INPUTADAPTER** 把跨医院 ICU 时序迁移从“改模型参数”转成“适配输入”。这很符合医疗部署：源医院模型可能已经备案、集成进设备或写入临床流程，不能随意重训。问题是，普通 input adapter 只要能让 frozen predictor 在目标域上表现更好，就可能把目标医院的采样政策痕迹翻译成源医院中同样带标签相关性的痕迹。例如目标医院的 `alarm-dense` 复测被 adapter 变换成源医院中“高风险患者特有的 late burst”，冻结模型仍会使用 policy shortcut。

第二，**OpenTSLM** 提醒我们，多变量医疗时序正在变成原生时序-语言融合对象。模型不只给出分类，还可能给出解释、问答或自然语言理由。若采样密度、变量覆盖、panel 同步或 value-pending 被包装进解释文本，系统可能给出“看似医学、实为流程”的理由。因此，适配目标域输入时，不能只让输入进入 frozen predictor 的数值工作区间，还要检查它落入的源域语义邻域是否真正由病理状态支持，而不是由采样政策相似性支持。

历史方案已经覆盖了危险率、后验商、保形、隐私、合成反例、元学习适配、Noether 作用量、PID 棱镜和 collider 封印。本轮换一个 **源域检索契约** 视角：

> 一个目标域样本被适配到源域后，若它的最近邻全部来自与目标样本相同的采样政策、相同 panel 习惯或相同 pending 语法，那么 frozen predictor 的高置信预测很可能只是 policy-nearest shortcut。真正可迁移的病理证据，应当能在源域 atlas 中找到“采样政策相反但病理语义相近”的反政策邻居。

**Do-Atlas Antipodal Adapter (DAAA)** 的核心思想：

1. 冻结已有源域分类器 `F_src`，不改动其权重。
2. 预先构建一个 Source Atlas：每个源域样本存储 frozen feature、标签、采样政策摘要、以及 OpenTSLM/Perceiver 风格的医学语义锚。
3. 目标域 input adapter `A_phi` 只负责把目标不规则事件变换到源域输入坐标。
4. 适配后的样本必须在 Source Atlas 中检索到 **state-similar but policy-antipodal** 的邻居集合：病理语义近，采样政策远或均衡。
5. 分类器仍是 frozen `F_src(A_phi(x))`，但训练损失惩罚“只落入 policy-nearest 邻域”的适配结果，并用源域反政策邻居的标签支持来约束 frozen logit。

这与当前“采样解耦/反事实干预”框架的结合方式自然：

- value process 产生病理语义锚，用于 Source Atlas 的 state key；
- sampling process 产生 policy key，但 policy key 只用于检索平衡与诊断，不进入分类头；
- counterfactual intervention 生成 policy-antipode recipes，测试适配后的样本能否在源域找到跨政策支持；
- frozen predictor 保持不变，最终改动主要在 **Dataloader + Input Adapter + Atlas Loss**。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Source Atlas + Antipodal Retrieval Bank

新增 `SourceAtlasCollator`，每个 batch 不只返回目标样本，还返回一个源域 atlas slice。

每个源域 atlas entry 包含：

1. `src_event_batch`：源域不规则事件输入。
2. `src_frozen_feature`：冻结预测器倒数层特征 `F_src.feature(x_src)`。
3. `src_label`：源域标签。
4. `src_policy_key`：采样政策摘要，包括变量覆盖、时间窗密度、panel-like 共现、pending 率、routine/alarm 节律等。
5. `src_semantic_key`：OpenTSLM/Perceiver 风格时序语义锚，描述观测值对应的医学状态语义，而不是记录流程语义。

目标 batch 额外返回：

- `target_policy_key`：目标采样政策摘要，只用于检索和诊断。
- `counterfactual_policy_key_bank`：由现有反事实模块生成的 policy-antipode 摘要，例如 routine 与 alarm 互换、panel split 与 panel pack 互换、pending 与 returned value 互换。
- `atlas_candidate_mask`：候选源邻居索引，便于大规模训练中做近似检索。

关键区别：

- 不生成 contrastive positive pair。
- 不要求反事实视图 logits 一致。
- 不做 test-time inner-loop adaptation。
- 不把 policy key、center id 或 workflow metadata 输入分类器。
- counterfactual policy 只用于定义“反政策邻域是否存在”，不是用于投票、保形、隐私、证明或 collider 路径干预。

### 2.2 改 Encoder / Adapter：冻结预测器前的 Antipodal Input Adapter

DAAA 包裹一个已经训练好的源域 irregular classifier：

```text
F_src: frozen source predictor
A_phi: trainable target-to-source input adapter
```

`A_phi` 可以是轻量 input-space transformation：

- 数值尺度 adapter：校正变量单位和分布；
- 时间坐标 adapter：把目标采样时间映射到源域模型可读的相对时间尺度；
- mask/value adapter：对 pending、panel 或稀疏变量做保守填充；
- 低幅度 event displacement：允许事件在源域可解释窗口内移动，但不能改写观测值语义。

适配后：

```text
x_adapt = A_phi(x_target)
logits  = F_src(x_adapt)
z_adapt = F_src.feature(x_adapt)
```

同时用一个冻结或弱训练的 `SemanticAnchorEncoder` 提取目标医学语义锚 `k_sem(x_target)`。它借鉴 OpenTSLM 的原生时序 resampler 思想，但不调用 LLM 生成文本理由，也不把 explanation 送入分类头；它只为 atlas retrieval 提供“病理语义是否相近”的度量。

### 2.3 Atlas Antipodal Retrieval

对适配后的目标样本计算两类距离：

```text
d_state(i)  = || z_adapt - src_frozen_feature_i ||_2
            + || k_sem(target) - src_semantic_key_i ||_2

d_policy(i) = cosine_distance(target_policy_key, src_policy_key_i)
```

DAAA 不选择最近的总距离邻居，而选择 **病理近、政策远或政策均衡** 的邻居：

```text
score_i = -d_state(i) + eta * d_policy(i)
```

并构造 two-sided atlas supports：

- `N_near_state`：病理语义最近的源邻居；
- `N_antipodal`：在 `N_near_state` 内采样政策最不同的一组；
- `N_balanced`：覆盖 routine/alarm/panel/pending 等 policy bins 的小集合。

若一个目标样本只有 policy-nearest 邻居而没有 antipodal neighbor，模型不强行输出低置信集合，也不加隐私噪声；它报告 `atlas_contract_violation`，并用训练损失推动 adapter 不要把样本映射到这种 policy-monopoly 区域。

### 2.4 改 Loss：从适配精度转向 Source-Atlas Antipodal Contract

总目标：

```text
L = L_target_cls
  + lambda_src * L_source_identity_contract
  + lambda_atl * L_antipodal_atlas_support
  + lambda_pol * L_policy_monopoly_penalty
  + lambda_sem * L_semantic_anchor_preservation
  + lambda_amp * L_adapter_amplitude_guard
```

#### A. Target / Source Classification `L_target_cls`

若目标域有少量标签，直接用 frozen predictor 输出训练 adapter：

```text
L_target_cls = CE(F_src(A_phi(x_tgt)), y_tgt)
```

若目标域无标签，则用源域 atlas 的 antipodal neighbor label distribution 作为弱监督：

```text
y_atlas = weighted_label_hist(N_antipodal)
L_target_cls = KL(log_softmax(logits), y_atlas)
```

注意这里不是投票裁决或 conformal set。Atlas label 只作为“源域反政策邻域是否支持该预测”的契约标签。

#### B. Source Identity Contract `L_source_identity_contract`

为了避免 adapter 破坏源域已验证模型，对源域样本要求 `A_phi` 近似恒等，且 frozen logits 不变：

```text
L_source_identity =
  || A_phi(x_src) - x_src ||_masked
  + KL(F_src(x_src) || F_src(A_phi(x_src)))
```

这直接服务 INPUTADAPTER 的 frozen predictor 约束：目标域可适配，但不能把源域模型的工作区间任意扭曲。

#### C. Antipodal Atlas Support `L_antipodal_atlas_support`

适配后的目标 feature 应靠近“病理相近且政策不同”的源邻域：

```text
L_antipodal =
  sum_i alpha_i^anti * || z_adapt - z_src_i ||_2^2
  + CE(logits, y_anti_hist)
```

其中 `alpha_i^anti` 只在 state-similar / policy-antipodal 候选上归一化。若模型依赖目标采样政策，它会更靠近 policy-nearest source 而非 antipodal source，该项会变大。

#### D. Policy Monopoly Penalty `L_policy_monopoly_penalty`

定义 top-k atlas 邻域中的 policy concentration：

```text
monopoly = max_policy_bin sum_{i in topk} alpha_i * 1[policy_bin_i]
L_policy_monopoly = relu(monopoly - rho)^2
```

这不是 policy adversarial：我们不训练表示骗过政策判别器。我们只要求“支撑当前预测的源邻域”不能被单一采样政策垄断。

#### E. Semantic Anchor Preservation `L_semantic_anchor_preservation`

输入适配不能把目标值语义改没。用 OpenTSLM-style 时序语义锚检查：

```text
L_semantic =
  SmoothL1(SemAnchor(x_tgt), SemAnchor(A_phi(x_tgt)))
  + CE(SemHead(SemAnchor(A_phi(x_tgt))), pathology_summary)
```

这不是跨策略 representation consistency，也不是 JEPA 目标预测；它是输入变换的医学语义保真约束，防止 adapter 为了贴近源域 policy pattern 而改写病理值。

#### F. Adapter Amplitude Guard `L_adapter_amplitude_guard`

限制 input adapter 的变换幅度：

```text
L_amp =
  || value_delta ||_masked^2
  + || time_delta / horizon ||_masked^2
  + relu(policy_key(A_phi(x)) shift too large)^2
```

幅度护栏保证 DAAA 是“把目标输入翻译到源域可读坐标”，不是生成任意能骗过 frozen predictor 的伪样本。

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def pairwise_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - torch.einsum("bd,bnd->bn", a, b)


class ConservativeInputAdapter(nn.Module):
    """Input-space adapter placed before a frozen source predictor."""

    def __init__(self, num_vars: int, hidden_dim: int, max_time_shift: float = 0.08):
        super().__init__()
        self.num_vars = num_vars
        self.max_time_shift = max_time_shift
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_net = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.value_delta = nn.Linear(hidden_dim, 1)
        self.time_delta = nn.Linear(hidden_dim, 1)
        self.mask_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        event_x = torch.cat(
            [
                self.var_embed(var_id),
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        h = self.event_net(event_x) * mask.unsqueeze(-1)
        dv = 0.10 * torch.tanh(self.value_delta(h).squeeze(-1)) * mask
        dt = self.max_time_shift * torch.tanh(self.time_delta(h).squeeze(-1)) * horizon * mask
        gate = self.mask_gate(h).squeeze(-1)

        adapted = dict(batch)
        adapted["event_value"] = (value + dv) * mask
        adapted["event_time"] = (time + dt).clamp_min(0.0)
        adapted["event_mask"] = (mask * gate).clamp(0.0, 1.0)
        adapted["adapter_value_delta"] = dv
        adapted["adapter_time_delta"] = dt / horizon
        adapted["adapter_mask_gate"] = gate
        return adapted


class SemanticAnchorEncoder(nn.Module):
    """OpenTSLM-inspired time-series semantic anchor used only for atlas retrieval."""

    def __init__(self, num_vars: int, hidden_dim: int, num_latents: int = 8):
        super().__init__()
        self.num_vars = num_vars
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_queries = nn.Parameter(torch.randn(num_latents, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, batch: dict) -> torch.Tensor:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        event_x = torch.cat(
            [
                self.var_embed(var_id),
                value.unsqueeze(-1),
                (time / horizon).unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                quality.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * mask.unsqueeze(-1)
        queries = self.latent_queries[None].expand(value.size(0), -1, -1)
        key_padding_mask = mask <= 0
        latent, _ = self.cross_attn(queries, event_h, event_h, key_padding_mask=key_padding_mask)
        return self.out(latent.mean(dim=1))


class PolicyKeyEncoder(nn.Module):
    """Observation-policy summary used for atlas balancing, not for classification."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        pending = batch.get("value_pending", torch.zeros_like(mask))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_rate = F.one_hot(var_id, self.num_vars).to(time.dtype) * mask.unsqueeze(-1)
        var_rate = var_rate.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        close = (delta_t <= masked_mean(delta_t, mask, dim=1).unsqueeze(-1).clamp_min(1e-6)).to(time.dtype)
        var_change = torch.zeros_like(mask)
        var_change[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(time.dtype)
        panel_like = masked_mean(close * var_change, mask, dim=1).unsqueeze(-1)

        stats = torch.cat(
            [
                mask.mean(dim=1, keepdim=True),
                masked_mean((time_norm <= 0.33).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean(((time_norm > 0.33) & (time_norm <= 0.66)).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean((time_norm > 0.66).to(time.dtype), mask, dim=1).unsqueeze(-1),
                masked_mean(torch.log1p(delta_t), mask, dim=1).unsqueeze(-1),
                panel_like,
                masked_mean(pending, mask, dim=1).unsqueeze(-1),
                mask.sum(dim=1, keepdim=True) / mask.size(1),
            ],
            dim=-1,
        )
        return self.net(torch.cat([var_rate, stats], dim=-1))


def atlas_antipodal_weights(
    z_query: torch.Tensor,
    sem_query: torch.Tensor,
    policy_query: torch.Tensor,
    atlas_feature: torch.Tensor,
    atlas_semantic: torch.Tensor,
    atlas_policy: torch.Tensor,
    state_temperature: float = 0.20,
    policy_bonus: float = 0.35,
) -> dict:
    """Compute source-neighbor weights favoring state similarity and policy distance."""

    state_dist = (z_query[:, None] - atlas_feature).pow(2).mean(dim=-1)
    state_dist = state_dist + (sem_query[:, None] - atlas_semantic).pow(2).mean(dim=-1)
    policy_dist = pairwise_cosine_distance(policy_query, atlas_policy)

    anti_score = -state_dist / state_temperature + policy_bonus * policy_dist
    near_score = -state_dist / state_temperature
    anti_weight = torch.softmax(anti_score, dim=1)
    near_weight = torch.softmax(near_score, dim=1)
    return {
        "anti_weight": anti_weight,
        "near_weight": near_weight,
        "state_dist": state_dist,
        "policy_dist": policy_dist,
    }


def policy_monopoly_loss(weight: torch.Tensor, atlas_policy_bin: torch.Tensor, rho: float = 0.55) -> torch.Tensor:
    """Penalize a retrieval neighborhood dominated by one policy bin."""

    num_bins = int(atlas_policy_bin.max().item()) + 1
    mass = []
    for bin_idx in range(num_bins):
        mass.append((weight * (atlas_policy_bin == bin_idx).to(weight.dtype)).sum(dim=1))
    policy_mass = torch.stack(mass, dim=1)
    monopoly = policy_mass.max(dim=1).values
    return F.relu(monopoly - rho).pow(2).mean()


class DoAtlasAntipodalAdapter(nn.Module):
    """Frozen-predictor input adaptation with source-atlas antipodal support."""

    def __init__(
        self,
        frozen_predictor: nn.Module,
        num_vars: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.frozen_predictor = frozen_predictor
        for param in self.frozen_predictor.parameters():
            param.requires_grad_(False)

        self.adapter = ConservativeInputAdapter(num_vars=num_vars, hidden_dim=hidden_dim)
        self.semantic = SemanticAnchorEncoder(num_vars=num_vars, hidden_dim=hidden_dim)
        self.policy = PolicyKeyEncoder(num_vars=num_vars, hidden_dim=hidden_dim)
        self.num_classes = num_classes

    def frozen_forward(self, batch: dict) -> dict:
        out = self.frozen_predictor(batch)
        if isinstance(out, torch.Tensor):
            out = {"logits": out}
        if "feature" not in out:
            out["feature"] = out["logits"]
        return out

    def forward(self, batch: dict) -> dict:
        adapted = self.adapter(batch)
        frozen = self.frozen_forward(adapted)
        sem = self.semantic(adapted)
        pol = self.policy(batch)
        retrieval = atlas_antipodal_weights(
            z_query=frozen["feature"],
            sem_query=sem,
            policy_query=pol,
            atlas_feature=batch["atlas_feature"],
            atlas_semantic=batch["atlas_semantic"],
            atlas_policy=batch["atlas_policy"],
        )
        return {
            **frozen,
            "adapted_batch": adapted,
            "semantic_key": sem,
            "policy_key": pol,
            **retrieval,
        }

    def training_loss(
        self,
        batch: dict,
        lambda_src: float = 0.20,
        lambda_atl: float = 0.45,
        lambda_pol: float = 0.25,
        lambda_sem: float = 0.15,
        lambda_amp: float = 0.05,
    ) -> dict:
        out = self.forward(batch)
        labels = batch.get("labels")

        atlas_label = F.one_hot(batch["atlas_label"].long(), self.num_classes).to(out["logits"].dtype)
        anti_hist = torch.einsum("bn,bnc->bc", out["anti_weight"], atlas_label)
        anti_hist = anti_hist / anti_hist.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if labels is not None:
            target_loss = F.cross_entropy(out["logits"], labels)
        else:
            target_loss = F.kl_div(
                F.log_softmax(out["logits"], dim=-1),
                anti_hist.detach(),
                reduction="batchmean",
            )

        anti_feature = torch.einsum("bn,bnh->bh", out["anti_weight"], batch["atlas_feature"])
        atlas_feature_loss = F.smooth_l1_loss(out["feature"], anti_feature.detach())
        atlas_label_loss = F.kl_div(
            F.log_softmax(out["logits"], dim=-1),
            anti_hist.detach(),
            reduction="batchmean",
        )
        atlas_loss = atlas_feature_loss + atlas_label_loss

        monopoly_loss = policy_monopoly_loss(out["near_weight"], batch["atlas_policy_bin"])

        raw_sem = self.semantic(batch)
        semantic_loss = F.smooth_l1_loss(out["semantic_key"], raw_sem.detach())

        adapted = out["adapted_batch"]
        amp_loss = (
            adapted["adapter_value_delta"].pow(2).mean()
            + adapted["adapter_time_delta"].pow(2).mean()
            + (adapted["adapter_mask_gate"] - 1.0).pow(2).mean()
        )

        if "source_batch" in batch:
            src_raw = self.frozen_forward(batch["source_batch"])
            src_adapted = self.adapter(batch["source_batch"])
            src_out = self.frozen_forward(src_adapted)
            source_loss = F.kl_div(
                F.log_softmax(src_out["logits"], dim=-1),
                F.softmax(src_raw["logits"].detach(), dim=-1),
                reduction="batchmean",
            )
        else:
            source_loss = torch.zeros((), device=out["logits"].device)

        total = (
            target_loss
            + lambda_src * source_loss
            + lambda_atl * atlas_loss
            + lambda_pol * monopoly_loss
            + lambda_sem * semantic_loss
            + lambda_amp * amp_loss
        )
        return {
            "loss": total,
            "target_cls_or_atlas_loss": target_loss.detach(),
            "source_identity_contract_loss": source_loss.detach(),
            "antipodal_atlas_support_loss": atlas_loss.detach(),
            "policy_monopoly_loss": monopoly_loss.detach(),
            "semantic_anchor_preservation_loss": semantic_loss.detach(),
            "adapter_amplitude_guard_loss": amp_loss.detach(),
            "mean_antipodal_policy_distance": (out["anti_weight"] * out["policy_dist"]).sum(dim=1).mean().detach(),
        }
```

## 4. SourceAtlasCollator 草稿

```python
import torch


@torch.no_grad()
def build_source_atlas_slice(
    target_batch: dict,
    source_memory: dict,
    atlas_size: int = 128,
) -> dict:
    """Attach a source atlas slice to a target batch.

    The atlas is used for retrieval contracts, not for voting ensembles or
    policy labels in the classifier.
    """

    out = dict(target_batch)
    num_source = source_memory["feature"].size(0)
    idx = torch.randint(
        low=0,
        high=num_source,
        size=(target_batch["event_value"].size(0), atlas_size),
        device=target_batch["event_value"].device,
    )
    out["atlas_feature"] = source_memory["feature"][idx]
    out["atlas_semantic"] = source_memory["semantic_key"][idx]
    out["atlas_policy"] = source_memory["policy_key"][idx]
    out["atlas_label"] = source_memory["label"][idx]
    out["atlas_policy_bin"] = source_memory["policy_bin"][idx]
    return out


@torch.no_grad()
def build_policy_antipode_keys(batch: dict, policy_encoder) -> torch.Tensor:
    """Build policy keys for counterfactual antipode diagnostics.

    These views do not supervise logits consistency. They only test whether the
    adapted target sample has support under opposite source policies.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_time, new_var, new_mask, pending=None):
        view = dict(batch)
        view["event_value"] = value * new_mask
        view["event_time"] = new_time
        view["event_var_id"] = new_var
        view["event_mask"] = new_mask
        if pending is not None:
            view["value_pending"] = pending
        return view

    views = []

    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    views.append(clone_with(rounded_time, var_id, mask))

    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_sparse = torch.where(late > 0, mask * alternating, mask)
    views.append(clone_with(time, var_id, alarm_sparse))

    shuffled_var = var_id.roll(shifts=1, dims=0)
    views.append(clone_with(time, shuffled_var, mask))

    pending = mask
    views.append(clone_with(time, var_id, mask, pending=pending))

    odd_var = (var_id % 2 == 1).to(mask.dtype)
    exposure_mask = mask * (1.0 - 0.4 * odd_var)
    views.append(clone_with(time, var_id, exposure_mask))

    return torch.stack([policy_encoder(view) for view in views], dim=1)
```

## 5. 实验切入点

1. **Policy shift 构造**
   - `frozen-source cross-center shift`：源域为 MIMIC-IV，目标域为 eICU / HiRID；冻结源模型，只训练 input adapter。
   - `policy-nearest shortcut shift`：训练源域中 alarm-dense 与高风险相关，测试目标域中 alarm-dense 扩展到普通患者，检查最近邻是否被单一政策垄断。
   - `panel atlas shift`：源域存在同步 panel，目标域将 panel 拆成异步事件，测试反政策邻域能否保持同标签支持。
   - `pending documentation shift`：value-pending 在源域与危重相关，目标域主要由实验室拥堵造成，检查 semantic anchor 是否阻止 adapter 把 pending 翻译成高风险证据。
   - `TSLM explanation shift`：用 OpenTSLM-style 解释接口检查错误样本是否引用采样/记录行为作为医学证据。

2. **对比方法**
   - INPUTADAPTER 原始 input-space adaptation。
   - Frozen predictor without adapter。
   - End-to-end domain adaptation / test-time adaptation。
   - Source kNN retrieval without policy-antipodal constraint。
   - OpenTSLM / time-series-language explanation baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DSPP、DCPD、DIPF、DRG-SFF、DPPC、DCOFF、DD-JEPA、DPSP、DNSA、DMWI、DCST 等。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - frozen cross-center worst AUROC / AUPRC。
   - atlas policy monopoly：top-k 支持邻域中最大 policy bin 占比。
   - antipodal support agreement：反政策近邻标签分布与 frozen prediction 的一致度。
   - semantic-anchor drift：适配前后 OpenTSLM-style semantic key 的漂移。
   - adapter amplitude：输入变换是否过度改写 values / times / masks。
   - policy-nearest error concentration：错误预测是否集中在 policy-nearest、state-far 的邻域。

4. **消融实验**
   - 去掉 `L_policy_monopoly_penalty`，检查 adapter 是否重新落入单一源域采样政策邻域。
   - 去掉 `L_antipodal_atlas_support`，只做普通 frozen input adaptation，检查跨政策鲁棒性是否下降。
   - 去掉 semantic anchor，只用 frozen feature 检索，验证 OpenTSLM-style 原生时序语义锚是否能减少流程性近邻。
   - 让 policy key 进入 classifier 作为反例，验证院内性能可能升高但跨中心退化。
   - 把 antipodal neighbor 替换为随机 neighbor，验证收益来自“病理近 + 政策远”的 Source Atlas 契约。
   - 不冻结源预测器，端到端训练，检查是否破坏已备案模型并重新吸收目标采样 shortcut。

## 6. 预期创新性

1. **从模型重训转向冻结预测器的源域契约适配**：继承 INPUTADAPTER 的现实部署约束，但不是只学习任意输入变换，而是要求变换后的样本落在可审计的源域反政策邻域中。
2. **从 policy-invariant 表示转向 policy-antipodal support**：不要求表示对所有采样视图一致，也不做 adversarial；只要求当前预测能被“采样政策不同但病理语义相近”的源邻居支撑。
3. **从 LLM/TSLM 解释转向语义锚检索**：吸收 OpenTSLM 的原生时序-语言融合接口，但不让自然语言理由参与分类；只用 Perceiver-style semantic anchor 防止 adapter 把记录流程翻译成病理语义。
4. **从元学习 workflow adaptation 转向无内循环 atlas retrieval**：不同于 DMWI 的 fast adapter，DAAA 部署时不需要 support set 内循环，只需一次 input adaptation + atlas retrieval 诊断。
5. **从 collider 封印转向邻域垄断检测**：不同于 DCST 的因果路径封印，DAAA 通过源域邻域的 policy monopoly 发现 frozen predictor 可能正在使用采样捷径。
6. **与采样解耦/反事实干预框架低侵入兼容**：value branch 生成 semantic key，sampling branch 生成 policy key，counterfactual branch 生成 antipode diagnostics，现有 frozen classifier 原封不动。

## 7. 一句话投稿卖点

**DAAA 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“冻结源域预测器在目标输入适配后缺少反政策源邻域支撑”的问题，通过 Source Atlas、OpenTSLM-style semantic anchors、policy-antipodal retrieval、邻域垄断惩罚与源模型恒等契约，让目标域样本只有在病理语义能被跨采样政策的源域近邻共同支持时才被 frozen predictor 信任，从而避免 INPUTADAPTER 式输入变换把目标医院 routine/alarm、panel、pending 或 documentation policy 翻译成源医院中的标签捷径。**
