# Title: Counterfactual Error Cartography Semantic Compiler：面向采样策略偏移的反事实误差地形语义编译器

## 0. 强制去重记录与思维黑名单

### 已读取材料

- `paper_daily.md`
- `paper_daily_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-12.md`
- `ideas/Idea_Proposal_2026-06-13.md`
- `ideas/Idea_Proposal_2026-06-14.md`
- `ideas/Idea_Proposal_2026-06-16.md`

### 未检出但已确认搜索

- 当前仓库未检出 `my_work_summary.md`。
- 当前仓库也未检出 `*summary*.md`、`*Summary*.md` 或中文“总结”命名的工作总结文件。

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
15. 连续时间模型空间后验、采样似然后验因子、posterior quotient layer。
16. 单纯复用 DBGL/GARLIC 式二部图、time-lagged graph、node-specific decay 或 attention 解释。
17. iTimER 原始的 pseudo-observation + Wasserstein alignment + contrastive learning 组合。
18. Record2Vec 原始的“冻结 LLM 摘要后直接分类”的便携 embedding 方案。

本提案选择一个与上述机制正交的切入点：**不估计采样概率，不做跨视图一致性，不做对抗，不做图交换，不给证据征税，也不做后验除法；而是把反事实采样下的重构误差看成一张“误差地形图”，用可微 ANOVA 投影把其中的 state-error 主效应与 policy-error 主效应拆开，再把 state-error 地形编译成可分类的结构化语义子句。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

近期 `paper_daily.md` 中最值得继续挖掘的两个前沿机制是：

- **iTimER** 提醒我们：不规则时序中的 reconstruction error 不只是训练失败信号，而是关于未观测区域、不确定性和采样结构的可学习对象。
- **Record2Vec** 提醒我们：跨医院或跨采样流程迁移时，低层时间戳和 mask 细节往往不便携；更高层的状态语义摘要可能比原始不规则轨迹更稳定。

但两者直接用于 sampling-policy shift 都有风险：

- 如果照搬 iTimER，把重构误差分布对齐或用伪观测增强，模型可能把训练医院的 policy-induced uncertainty 当成状态证据。
- 如果照搬 Record2Vec，把整条 ICU 记录总结成一个 embedding，摘要可能把“某变量被频繁检测”误写成“病情持续恶化”，语义层同样会被采样策略污染。

**Counterfactual Error Cartography Semantic Compiler (CEC-SC)** 的核心直觉是：

> 采样策略偏移首先会改变模型“在哪里重构不好”，而不一定立即改变观测值本身。若同一潜在轨迹在不同 `do(policy)` 下产生不同的误差地形，则这部分误差是 acquisition semantics；若某些误差结构在政策改变后仍作为平均地形存在，则它更可能反映 state uncertainty 或真实动态复杂性。

因此，我们不让分类器直接吃 mask、delta-t、policy code 或 LLM summary；而是先构造反事实误差张量：

```text
E[b, k, t, d] = |x_target[b, t, d] - Recon(x observed under do(policy=k))[b, t, d]|
```

然后在 policy 维度上做可微方差分析式投影：

```text
E_state  = mean_k E[:, k, :, :]
E_policy = E - E_state
```

`E_state` 被编译为“状态语义子句”，例如“乳酸相关区域在晚期窗口存在稳定高不确定性”“心率轨迹的恢复段可重构且低误差”；`E_policy` 被编译为“采集语义子句”，例如“晚期 panel 稀疏导致误差扩张”“某变量只在报警后可见”。分类器只能使用 state clauses；policy clauses 只用于采样过程诊断和校准。

这与当前“采样解耦/反事实干预”框架天然契合：

- value process 负责生成反事实观测下的重构误差场；
- sampling process 负责提供 `do(policy)` recipe，但不进入分类头；
- counterfactual intervention 不再服务于一致性、风险方差或后验商，而是作为“误差地形探针”来定位哪些不确定性来自采样策略。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从轨迹 embedding 改为“误差地形到语义子句”的编译器

CEC-SC 包含四个模块。

#### A. Counterfactual Reconstructor

对同一条样本生成 K 个由 policy recipe 控制的观测版本。重构器只根据每个版本中可见的值和时间，预测一个统一 query grid 或 sparse query set 上的值：

```text
x_hat_k = R_phi(values under do(policy=k), times, observed_mask_k)
E_k     = |x_target - x_hat_k|
```

这里的反事实采样不是 hazard thinning，也不是普通 mask dropout；recipe 可以表示医院流程级动作，例如 routine panel 间隔、报警后复测窗口、设备预算限制或变量联测拆分。

#### B. Error Cartography Layer

把 K 个误差场堆叠成 `E[B, K, T, D]`，执行可微地形分解：

```text
E_state         = mean_k E_k
E_policy_k      = E_k - E_state
E_policy_energy = mean_k |E_policy_k|
```

`E_state` 不是“跨视图表征一致性”，它只是 policy 维度的主效应投影；我们并不要求不同 policy 下的 logits 或 representation 相同。`E_policy_energy` 也不是 risk variance，它度量的是重构误差地形对采样政策的敏感性，而不是分类风险的方差。

#### C. Semantic Clause Compiler

借鉴 Record2Vec 的“语义中介层”思想，但不调用冻结 LLM 直接分类。我们用一个可微向量量化 codebook，把 `E_state` 与 value-derived trend features 编译为有限个 state clauses：

```text
state_clause_j = VQ(MLP([value_trend, E_state, local_slope, variable_id, time_bin]))
```

每个 codebook entry 可以离线映射为中文或英文短语，形成可解释摘要；训练时分类器只看到 clause embedding。这样既保留语义层的可迁移性，又避免 LLM 摘要把采样流程自由写进状态描述。

#### D. Acquisition Clause Compiler

另一个编译器读取 `E_policy_energy` 与 policy recipe，生成 acquisition clauses，用于：

1. 重构每个 recipe 下的 policy-error 地形；
2. 预测或解释采样动作；
3. 输出部署时的 policy sensitivity score。

acquisition clauses 不进入分类器，也不作为 policy residual sink 来吸收分类外残差；它们只承担误差地形归档和偏移诊断。

### 2.2 改 Loss：从一致性/对抗/后验商转向“误差地形可分解 + 语义子句充分性”

总目标：

```text
L = L_cls
  + lambda_rec    * L_reconstruction
  + lambda_state  * L_state_cartography
  + lambda_policy * L_policy_cartography
  + lambda_vq     * L_clause_commitment
  + lambda_hsic   * L_redaction_checksum
```

#### A. 分类损失 `L_cls`

分类器只使用 state clauses：

```text
z_state = Pool(state_clauses)
logits  = Classifier(z_state)
L_cls   = CE(logits, y)
```

它不接收 mask embedding、policy recipe、policy-error map、acquisition clauses 或采样摘要。

#### B. 重构损失 `L_reconstruction`

重构器需要在 factual 与 counterfactual recipe 下都能预测 query target：

```text
L_reconstruction = mean_{k,t,d} mask_target[t,d] * Huber(x_hat_k[t,d], x_target[t,d])
```

该项用于获得可信误差地形，而不是为了插补后直接分类。

#### C. 状态地形充分性 `L_state_cartography`

state clauses 必须能重构 policy 平均后的误差地形：

```text
E_state_hat = StateMapDecoder(state_clauses)
L_state_cartography = ||E_state_hat - stopgrad(E_state)||_1
```

含义：进入分类器的语义子句应当解释“无论采样政策如何改变，仍稳定存在的重构困难在哪里”。这比直接对齐不同 policy 的 representation 更弱，也更可解释。

#### D. 采集地形归档 `L_policy_cartography`

acquisition clauses 负责解释 policy-sensitive error：

```text
E_policy_hat = PolicyMapDecoder(acquisition_clauses, recipe)
L_policy_cartography = ||E_policy_hat - stopgrad(E_policy)||_1
```

这不是对抗去偏，也不是 residual sink 分类隔离；它只是把由采样动作造成的误差地形存入单独账册，供偏移诊断使用。

#### E. 子句量化承诺 `L_clause_commitment`

为了让语义层可解释且稳定，使用 VQ-style commitment loss：

```text
L_clause_commitment = ||sg(h) - e_q||_2^2 + beta ||h - sg(e_q)||_2^2
```

最终可把高频 state clause prototype 命名为临床语义短句，例如“持续高波动但采样无关”“晚期恢复段稳定可重构”“跨变量异常共现但不依赖联测流程”。

#### F. Redaction Checksum `L_redaction_checksum`

为了防止 state clauses 偷偷编码 policy recipe，使用非对抗的 HSIC 独立性校验：

```text
L_redaction_checksum = HSIC(Pool(state_clauses), mean_k recipe_k)
```

这不是训练一个 policy discriminator，也没有梯度反转；它只是对 state semantic summary 与 recipe summary 做核独立性校验。若审稿人担心 HSIC 过强，可以将其作为可选正则或只用于监控指标。

### 2.3 改 Dataloader：返回“反事实误差探针”，不是增强正样本

新增 `ErrorCartographyCollator`，每个 batch 返回：

1. `cf_values`: `[B, K, T, D]`，K 个采样 recipe 下可见的值。
2. `cf_mask`: `[B, K, T, D]`，K 个 recipe 下的观测 mask。
3. `query_times`: `[B, T]`，统一 query grid 或 sparse query set。
4. `target_values`: `[B, T, D]`，用于自监督重构的 held-out / observed target。
5. `target_mask`: `[B, T, D]`，target 是否可用于重构损失。
6. `policy_recipe`: `[B, K, R]`，流程级采样干预 recipe。
7. `labels`: 分类标签。

注意：collator 不生成对比学习 pair，不要求多视图 logits 一致，也不计算 risk variance。它只负责让模型看到“同一底层样本在不同采样政策下哪里更难重构”。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 可替换或扩展为 `CounterfactualReconstructor`，专注于产生可靠的重构误差场。
- 现有 sampling branch 保留为 recipe generator / intervention controller，但不输出分类特征。
- 现有 counterfactual intervention 模块从“构造鲁棒分类视图”改为“构造误差地形探针”。
- 推理阶段可以有两种模式：
  - **Single-policy fast mode**：使用 factual schedule 加若干标准 recipe 估计 `E_state`，输出分类结果。
  - **Audit mode**：同时输出 state clauses、acquisition clauses 与 policy sensitivity score，用于解释预测是否依赖当前采样流程。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A lightweight non-adversarial independence checksum."""

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov = x.transpose(0, 1).matmul(y) / max(x.size(0), 1)
    return cov.pow(2).mean()


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.beta = beta
        nn.init.normal_(self.codebook.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> dict:
        # h: [B, N, C]
        flat = h.reshape(-1, h.size(-1))
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat.matmul(self.codebook.weight.t())
            + self.codebook.weight.pow(2).sum(dim=1)
        )
        code_id = dist.argmin(dim=-1)
        quant = self.codebook(code_id).view_as(h)
        quant_st = h + (quant - h).detach()

        commit = F.mse_loss(h.detach(), quant) + self.beta * F.mse_loss(h, quant.detach())
        return {
            "quantized": quant_st,
            "code_id": code_id.view(h.shape[:-1]),
            "commitment_loss": commit,
        }


class CounterfactualReconstructor(nn.Module):
    """Predict query-grid values from each counterfactual observation schedule."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(3 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_vars),
        )

    def forward(
        self,
        cf_values: torch.Tensor,
        cf_mask: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        # cf_values/cf_mask: [B, K, T, D], query_times: [B, T]
        bsz, num_policy, steps, num_vars = cf_values.shape
        values = cf_values.reshape(bsz * num_policy, steps, num_vars)
        mask = cf_mask.reshape(bsz * num_policy, steps, num_vars)
        times = query_times[:, None, :].expand(bsz, num_policy, steps).reshape(bsz * num_policy, steps)

        var_ids = torch.arange(num_vars, device=cf_values.device)
        var_h = self.var_embed(var_ids).mean(dim=0).view(1, 1, -1)
        observed_mean = (values * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        observed_count = mask.mean(dim=-1, keepdim=True)
        event_x = torch.cat(
            [observed_mean, observed_count, torch.log1p(times).unsqueeze(-1), var_h.expand(values.size(0), steps, -1)],
            dim=-1,
        )
        event_h = self.event_proj(event_x)
        seq_h, _ = self.rnn(event_h)
        pred = self.query_proj(torch.cat([seq_h, torch.log1p(times).unsqueeze(-1)], dim=-1))
        return pred.view(bsz, num_policy, steps, num_vars)


def error_cartography(
    pred: torch.Tensor,
    target_values: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict:
    """Split reconstruction error into state and policy cartography terms."""

    # pred: [B, K, T, D], target_*: [B, T, D]
    target = target_values[:, None, :, :]
    mask = target_mask[:, None, :, :]
    error = (pred - target).abs() * mask
    state_error = error.mean(dim=1)
    policy_error = error - state_error[:, None, :, :]
    policy_energy = policy_error.abs().mean(dim=1)
    return {
        "error": error,
        "state_error": state_error,
        "policy_error": policy_error,
        "policy_energy": policy_energy,
    }


class SemanticClauseCompiler(nn.Module):
    """Compile value/error cartography into discrete semantic clauses."""

    def __init__(self, num_vars: int, hidden_dim: int, num_codes: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vq = VectorQuantizer(num_codes=num_codes, code_dim=hidden_dim)

    def forward(
        self,
        target_values: torch.Tensor,
        target_mask: torch.Tensor,
        state_error: torch.Tensor,
        query_times: torch.Tensor,
    ) -> dict:
        # Build [time, variable] clause tokens.
        bsz, steps, num_vars = target_values.shape
        var_h = self.var_embed(torch.arange(num_vars, device=target_values.device))
        var_h = var_h.view(1, 1, num_vars, -1).expand(bsz, steps, num_vars, -1)

        slope = torch.zeros_like(target_values)
        slope[:, 1:] = target_values[:, 1:] - target_values[:, :-1]
        time_feat = torch.log1p(query_times).view(bsz, steps, 1, 1).expand(bsz, steps, num_vars, 1)

        local = torch.cat(
            [
                target_values.unsqueeze(-1),
                target_mask.unsqueeze(-1),
                state_error.unsqueeze(-1),
                slope.unsqueeze(-1),
                time_feat,
                var_h,
            ],
            dim=-1,
        )
        h = self.local_mlp(local).view(bsz, steps * num_vars, -1)
        return self.vq(h)


class MapDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_vars: int):
        super().__init__()
        self.num_vars = num_vars
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, clause_tokens: torch.Tensor, steps: int, num_vars: int) -> torch.Tensor:
        decoded = self.out(clause_tokens).squeeze(-1)
        return decoded.view(clause_tokens.size(0), steps, num_vars)


class AcquisitionClauseCompiler(nn.Module):
    """Summarize policy-sensitive error for diagnostics, not classification."""

    def __init__(self, recipe_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(recipe_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.recipe_head = nn.Linear(hidden_dim, recipe_dim)

    def forward(self, policy_energy: torch.Tensor, policy_recipe: torch.Tensor) -> dict:
        # policy_energy: [B, T, D], policy_recipe: [B, K, R]
        energy_summary = policy_energy.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        recipe_summary = policy_recipe.mean(dim=1)
        h = self.net(torch.cat([energy_summary, recipe_summary], dim=-1))
        return {
            "acquisition_clause": h,
            "recipe_logits": self.recipe_head(h),
        }


class ErrorCartographySemanticClassifier(nn.Module):
    def __init__(
        self,
        num_vars: int,
        recipe_dim: int,
        hidden_dim: int,
        num_codes: int,
        num_classes: int,
    ):
        super().__init__()
        self.reconstructor = CounterfactualReconstructor(num_vars=num_vars, hidden_dim=hidden_dim)
        self.state_compiler = SemanticClauseCompiler(num_vars=num_vars, hidden_dim=hidden_dim, num_codes=num_codes)
        self.state_decoder = MapDecoder(hidden_dim=hidden_dim, num_vars=num_vars)
        self.acquisition_compiler = AcquisitionClauseCompiler(recipe_dim=recipe_dim, hidden_dim=hidden_dim)
        self.policy_decoder = nn.Sequential(
            nn.Linear(hidden_dim + recipe_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_vars),
            nn.Softplus(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict) -> dict:
        pred = self.reconstructor(
            cf_values=batch["cf_values"],
            cf_mask=batch["cf_mask"],
            query_times=batch["query_times"],
        )
        maps = error_cartography(
            pred=pred,
            target_values=batch["target_values"],
            target_mask=batch["target_mask"],
        )
        state = self.state_compiler(
            target_values=batch["target_values"],
            target_mask=batch["target_mask"],
            state_error=maps["state_error"],
            query_times=batch["query_times"],
        )
        state_tokens = state["quantized"]
        pooled_state = state_tokens.mean(dim=1)
        logits = self.classifier(pooled_state)

        acquisition = self.acquisition_compiler(
            policy_energy=maps["policy_energy"],
            policy_recipe=batch["policy_recipe"],
        )

        return {
            "pred": pred,
            "logits": logits,
            "pooled_state": pooled_state,
            "state_tokens": state_tokens,
            "state_code_id": state["code_id"],
            "vq_loss": state["commitment_loss"],
            **maps,
            **acquisition,
        }

    def training_loss(
        self,
        batch: dict,
        lambda_rec: float = 0.3,
        lambda_state: float = 0.1,
        lambda_policy: float = 0.1,
        lambda_vq: float = 0.02,
        lambda_hsic: float = 0.01,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]

        cls_loss = F.cross_entropy(out["logits"], labels)

        rec_raw = F.smooth_l1_loss(
            out["pred"],
            batch["target_values"][:, None, :, :],
            reduction="none",
        )
        rec_mask = batch["target_mask"][:, None, :, :]
        rec_loss = (rec_raw * rec_mask).sum() / rec_mask.sum().clamp_min(1.0)

        steps, num_vars = batch["target_values"].shape[1:]
        state_hat = self.state_decoder(out["state_tokens"], steps=steps, num_vars=num_vars)
        state_loss = F.l1_loss(state_hat * batch["target_mask"], out["state_error"].detach() * batch["target_mask"])

        recipe_summary = batch["policy_recipe"].mean(dim=1)
        policy_input = torch.cat([out["acquisition_clause"], recipe_summary], dim=-1)
        policy_energy_hat = self.policy_decoder(policy_input).unsqueeze(1).expand(-1, steps, -1)
        policy_loss = F.l1_loss(
            policy_energy_hat * batch["target_mask"],
            out["policy_energy"].detach() * batch["target_mask"],
        )
        recipe_loss = F.binary_cross_entropy_with_logits(out["recipe_logits"], recipe_summary)

        hsic_loss = linear_hsic(out["pooled_state"], recipe_summary)

        total = (
            cls_loss
            + lambda_rec * rec_loss
            + lambda_state * state_loss
            + lambda_policy * (policy_loss + recipe_loss)
            + lambda_vq * out["vq_loss"]
            + lambda_hsic * hsic_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "state_cartography_loss": state_loss.detach(),
            "policy_cartography_loss": policy_loss.detach(),
            "recipe_loss": recipe_loss.detach(),
            "vq_loss": out["vq_loss"].detach(),
            "redaction_checksum": hsic_loss.detach(),
        }
```

## 4. 预期创新性

1. **从“观测值表征”转向“误差地形表征”**：模型不直接把未观测区域补成伪观测用于分类，而是利用反事实采样下的重构误差地图识别哪些不确定性属于状态、哪些属于采样流程。
2. **从“文本摘要直接分类”转向“可微语义子句编译”**：吸收 Record2Vec 的便携语义中介思想，但用 VQ clause grammar 限制摘要内容，避免自由文本把采样政策写进状态描述。
3. **从跨视图一致性转向 policy 维度主效应投影**：不要求不同 `do(policy)` 下的 representation、logits 或 risk 一致，只对误差张量做 ANOVA 式 state/policy cartography。
4. **从策略去除转向误差归档**：policy-sensitive error 不被对抗删除，也不进入 residual sink，而是形成 acquisition clauses，用于偏移诊断、校准和审稿人可读解释。
5. **与采样解耦/反事实干预框架高度兼容**：反事实模块变成误差探针，sampling branch 提供 recipe，value branch 生成可解释 state-error semantics，分类器只消费 state clauses。

## 5. 实验切入点

1. 构造四类 sampling-policy shift：
   - query grid 不变但观测频率改变；
   - 特定变量由 routine measurement 改为 alarm-triggered measurement；
   - panel 联测拆分或合并；
   - 晚期窗口设备预算限制导致系统性稀疏。
2. 对比：
   - iTimER 原始 pseudo-observation / Wasserstein / contrastive 版本；
   - Record2Vec 风格 summarize-then-embed；
   - missingness-aware encoder；
   - policy adversarial baseline；
   - hazard/null-space baseline；
   - commutator graph surgery baseline；
   - protocol-tax evidence market baseline；
   - posterior quotient dynamics baseline。
3. 新增指标：
   - worst-policy accuracy；
   - error cartography policy energy；
   - state-clause policy HSIC；
   - acquisition-clause recipe recoverability；
   - semantic clause stability under deployment shift；
   - high policy-energy abstention calibration。
4. 消融：
   - 去掉 error cartography，直接用重构器 hidden state 分类；
   - 去掉 VQ semantic compiler，改用连续 embedding；
   - 去掉 acquisition cartography，观察 state clauses 是否泄漏 recipe；
   - 去掉 redaction checksum，统计 state-clause policy HSIC 是否上升；
   - 将 `E_state` 替换为单一 factual error，验证反事实误差地形的必要性。

## 6. 一句话投稿卖点

**CEC-SC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“重构误差地形的语义污染”问题：通过反事实采样探针生成 error cartography，用 ANOVA 式主效应投影拆出 state-error 与 policy-error，再把前者编译为可分类的离散语义子句，从而在不依赖危险率、对抗、一致性、交换子、协议税或后验商的前提下提升跨采样策略鲁棒性。**
