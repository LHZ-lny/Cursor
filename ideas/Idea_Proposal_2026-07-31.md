# Title: Krylov Policy Mode Annihilator：面向采样策略偏移的反事实 Krylov 政策模态湮灭器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*work*summary*.md`、`*summary*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化记忆 `MEMORIES.md`，以及记忆中的 `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`、`idea_2026-07-29.md`、`idea_2026-07-30.md`。
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
  - `ideas/Idea_Proposal_2026-06-26.md`
  - `ideas/Idea_Proposal_2026-07-12.md`
  - `ideas/Idea_Proposal_2026-07-13.md`
  - `ideas/Idea_Proposal_2026-07-14.md`
  - `ideas/Idea_Proposal_2026-07-28.md`
  - `ideas/Idea_Proposal_2026-07-30.md`
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`
  - `Idea_Proposal_2026-06-24.md`
  - `Idea_Proposal_2026-06-27.md`
  - `Idea_Proposal_2026-07-15.md`
  - `Idea_Proposal_2026-07-16.md`
  - `Idea_Proposal_2026-07-17.md`
  - `Idea_Proposal_2026-07-18.md`
  - `Idea_Proposal_2026-07-19.md`
  - `Idea_Proposal_2026-07-20.md`
  - `Idea_Proposal_2026-07-21.md`
  - `Idea_Proposal_2026-07-22.md`
  - `Idea_Proposal_2026-07-23.md`
  - `Idea_Proposal_2026-07-24.md`
  - `Idea_Proposal_2026-07-25.md`
  - `Idea_Proposal_2026-07-26.md`
  - `Idea_Proposal_2026-07-27.md`
  - `Idea_Proposal_2026-07-29.md`
- 已读取近期论文记录：`paper_daily.md`。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮永久避开以下机制作为主创新：

1. adaptive time encoding / learnable reference points。
2. 频域掩码修复、跨视图对比学习、普通 temporal/inter-variable consistency。
3. missingness pattern 直接分类、prototype classifier、简单 policy adversarial。
4. hazard scorer、采样 score 零空间、hazard-driven resampling、do-risk variance。
5. 生理流与采样算子交换子、value/policy graph 分离、policy residual sink。
6. additive evidence market、protocol tax、token evidence budget、边际证据审计。
7. posterior quotient、采样似然相除、模型空间后验商、干预积分分类。
8. reconstruction error cartography、ANOVA error projection、VQ semantic clauses、HSIC redaction。
9. policy-simplex randomized smoothing、certified policy radius。
10. 采样测度 density ratio、doubly robust risk、influence-bound、RKHS cubature。
11. optional-stopping martingale query、standardized innovation、停时矩控制。
12. censored excursion topology、policy-gauge horizontal transport、policy shadow negative film。
13. syndrome code / parity repair、calendar knockoff / FDR、observability witness。
14. subjective-logic evidential shield、policy-induced vacuity、class-wise evidence discount。
15. observation-set lattice、meet/join margin、submodular shortcut curvature。
16. solver trace front-door、NFE/roughness mediator、trace standardized readout。
17. measurement-action bisimulation、policy-word signature、thermodynamic free-energy、Sklar copula stripping。
18. triage queue debt、Sinkhorn detail canonicalization、MDL episode transducer、causal sheaf gluing。
19. trigger hysteresis、observer-control barrier certificate、regret escrow、principal-stratum status compiler。
20. counterfactual conformal sleeves、Do-IV structural purifier、Do-Jury rank tribunal。
21. 单纯把 STAR-Set 的 attention bias、VP-GNN 的 variable/patch graph、EHR-SPC 的 status token 或 LLM4EHR 的语义对齐拆成 state-policy 双分支再做一致性。

本提案选择新的正交切入点：**不做一致性、不做对抗、不做后验相除、不做社会选择、不做保形、不拆双图、不估计采样概率；而是把采样政策对 latent state 的影响视为一族低维线性政策模态，用反事实干预学习其 Krylov 子空间，再用可微湮灭多项式把这些政策模态从分类读出前消掉。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 中近期机制给出了四个值得合并的前沿信号：

- **STAR-Set Transformer** 说明 temporal locality bias 与 variable-type affinity 能显著影响异步事件集合的注意力结构。
- **VP-GNN** 说明 variable-wise graph 与 patch-wise graph 会共同决定临床 IMTS 分类边界。
- **EHR-SPC** 用 query-based set prediction 预测未来 status tokens，提示不规则 EHR 可以用集合式 status 表征承载未来病程信息。
- **LLM4EHR** 用医学事件序列语义对齐临床 TS embedding，提示事件语义可以提供比裸 mask 更稳定的状态锚点。

这些机制的共同风险是：采样策略一变，attention bias、变量共现图、patch 可见性、future status set 和事件语义对齐都会发生联动漂移。历史提案已经尝试过把这些漂移解释为图分解、IV 残差、保形校准、陪审团排序、trigger gate、barrier、regret、principal strata 等机制。本轮换一个线性动力学与数值代数视角：

> 对同一条底层病程施加多个反事实采样政策时，模型 latent state 的变化往往集中在少数可重复的方向上：某些方向对应 late dense patch，某些方向对应 variable panel co-observation，某些方向对应 future-status 可见性，某些方向对应事件语义记录习惯。这些方向不是随机噪声，而是一族可学习的 **policy modes**。

如果分类头直接读取原始 latent state，它会把这些 policy modes 当成类别证据。**Krylov Policy Mode Annihilator (KPMA)** 的核心思想是：

1. 用当前反事实采样器生成一组 deployable policy recipes。
2. 对每个 recipe，估计它在 latent state 上诱发的低秩线性位移算子 `A_r`。
3. 由 `h, A_r h, A_r^2 h, ...` 构造 Krylov policy subspace，捕捉“这个采样政策反复作用后会强化的模态”。
4. 学习一个可微湮灭多项式 `p_r(A_r)`，让 policy modes 被压低，而医学 status / value semantics 被保留。
5. 分类器只读取 `p_r(A_r) h` 的聚合状态，避免某种采样制度下的结构模态成为稳定捷径。

这与当前“采样解耦/反事实干预”框架高度兼容：

- value encoder 继续产生观测值驱动的 state embedding。
- sampling branch 不进分类头，只产生 policy recipes 与 policy displacement operator。
- counterfactual intervention 不用于一致性、平滑、校准或投票，而是用于学习 latent policy operator 的作用方向。
- readout 从普通 pooled state 改为 **policy-mode annihilated state**。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：输出 Structural-Status Latent，而不是直接 logits

基础 encoder 可以是 STAR-Set、VP-GNN、EHR-SPC 风格 status encoder 或普通 event Transformer。KPMA 要求它输出三类量：

```text
h_value:        观测值和时间事件编码后的主状态向量
h_struct:       attention bias / variable graph / patch graph 的结构摘要
h_status:       future status / event semantic anchor 的状态摘要
```

组合为：

```text
h0 = LayerNorm(h_value + W_s h_struct + W_u h_status)
```

其中 `h_struct` 与 `h_status` 不作为单独 policy 分支进入分类器；它们只是让 latent state 同时看见近期 paper_daily 中的结构与语义机制。真正的去偏发生在后续 Krylov policy mode annihilation。

### 2.2 改 Sampling Branch：从 policy code 改为 Policy Displacement Operator

对每个反事实采样 recipe `r`，输入同一患者的反事实视图 `x_r`，得到：

```text
h_r = Encoder(x_r)
delta_r = stopgrad(h_r - h0)
```

然后用一个小网络估计低秩政策算子：

```text
A_r = U_r V_r^T,  rank(A_r) << dim(h)
A_r h0 ≈ delta_r
```

这里的 `A_r` 不代表图边、不代表采样概率、不代表后验因子，也不与生理算子做交换子。它只表示“这个采样政策在当前 latent 坐标中最可能把表示推向哪些方向”。

### 2.3 改 Loss：从表示一致转向 Krylov 模态湮灭

总目标：

```text
L = L_cls
  + lambda_fit  * L_operator_fit
  + lambda_ann  * L_policy_annihilation
  + lambda_sem  * L_semantic_retention
  + lambda_cond * L_polynomial_condition
```

#### A. 分类损失 `L_cls`

对每个 policy operator 计算湮灭后的状态：

```text
z_r = p_r(A_r) h0
z_bar = mean_r z_r
logits = Classifier(z_bar)
L_cls = CE(logits, y)
```

与随机平滑或陪审团不同，这里不是平均多个 policy logits，也不是让多个 policy juror 投票；模型先在 latent space 消掉 policy modes，再做一次统一分类。

#### B. Operator Fit Loss `L_operator_fit`

让低秩算子真实刻画反事实采样造成的 latent displacement：

```text
L_operator_fit = mean_r || A_r h0 - stopgrad(h_r - h0) ||_1
```

这一步是 KPMA 的“采样解剖”：采样 policy 不进入分类，但它必须在 latent 空间中留下可解释的线性近似。

#### C. Policy Annihilation Loss `L_policy_annihilation`

由低阶 Krylov 序列构造 policy-only modes：

```text
k_0 = delta_r
k_1 = A_r delta_r
k_2 = A_r^2 delta_r
...
```

湮灭多项式应压低这些 policy modes：

```text
L_policy_annihilation = mean_r,j || p_r(A_r) k_j ||_2^2
```

注意它不要求 `h_r` 与 `h0` 相同，也不要求 logits 一致；它只要求“被算子识别为政策位移的 Krylov 模态”不能被分类读出放大。

#### D. Semantic Retention Loss `L_semantic_retention`

为了避免多项式把所有信息都压掉，利用 EHR-SPC / LLM4EHR 启发加入稳定语义保真：

```text
status_hat = StatusDecoder(z_bar)
L_semantic_retention = CE(status_hat, stable_status_label)
```

`stable_status_label` 可来自：

- EHR-SPC 风格未来 status bucket；
- 临床事件序列中的粗语义标签；
- 无 LLM 场景下的 value-only pseudo status，例如稳定/恶化/恢复/波动。

这不是对比语义对齐，也不是 natural-language summary；它只是保证湮灭后的 state 仍能回答“病程状态是什么”，而不是只学会删除采样痕迹。

#### E. Polynomial Condition Loss `L_polynomial_condition`

湮灭多项式若系数失控，会导致数值不稳定。设：

```text
p_r(A) = I + c_1 A + c_2 A^2 + ... + c_M A^M
```

加入：

```text
L_polynomial_condition =
  ||c||_2^2
  + relu(||p_r(A_r) h0|| / ||h0|| - upper)^2
  + relu(lower - ||p_r(A_r) h0|| / ||h0||)^2
```

该项不是 solver trace，也不是 free-energy；它只是约束湮灭滤波器保留足够状态能量，同时避免多项式爆炸。

### 2.4 改 Dataloader：返回 Policy Operator Bank

新增 `PolicyOperatorBankCollator`，每个 batch 返回：

1. `factual_batch`：原始事件集合。
2. `policy_recipe_bank`：
   - `star_locality_shift`：改变 temporal locality 支持范围；
   - `affinity_panel_shift`：合并或拆分变量联测；
   - `vp_patch_visibility_shift`：改变高权重 patch 的可见预算；
   - `status_set_shift`：改变未来 status event set 的可见性；
   - `semantic_event_shift`：保留数值但稀释某些 EHR event sequence 语义锚点。
3. `cf_batch_bank`：每个 recipe 下的反事实观测视图。
4. `stable_status_label`：用于语义保真的粗状态标签。

关键区别：

- 不构造一致性正样本。
- 不估计 hazard、density ratio、posterior factor 或 conformal threshold。
- 不做 graph split、knockoff、gauge、syndrome repair、rank tribunal。
- 只用反事实采样视图学习 latent policy displacement operator 和其 Krylov policy modes。

### 2.5 推理阶段

给定测试样本：

1. 编码事实视图得到 `h0`。
2. 用部署可接受的标准 policy recipes 生成 operator bank；若不能生成完整反事实值，可只用 sampling coordinates 估计 `A_r`。
3. 计算 `z_bar = mean_r p_r(A_r) h0`。
4. 输出：
   - `pred_label = argmax Classifier(z_bar)`；
   - `policy_mode_energy = mean_r ||(I - p_r(A_r)) h0||`；
   - `annihilation_residual = mean_r ||p_r(A_r) A_r h0||`；
   - `semantic_retention_score`：湮灭后 status 是否仍可恢复。

当 `policy_mode_energy` 高且 `semantic_retention_score` 低时，说明当前预测强依赖采样制度诱发的 latent 模态，应触发补采样或人工复核。

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


class StructuralStatusAdapter(nn.Module):
    """Fuse value state with structural traces and coarse status semantics."""

    def __init__(self, value_dim: int, struct_dim: int, status_dim: int, hidden_dim: int):
        super().__init__()
        self.struct_dim = struct_dim
        self.status_dim = status_dim
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        self.status_proj = nn.Linear(status_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, encoder_out: dict) -> torch.Tensor:
        value_raw = encoder_out["value_state"]
        batch_size = value_raw.size(0)
        device = value_raw.device
        dtype = value_raw.dtype
        struct_raw = encoder_out.get(
            "structure_trace",
            torch.zeros(batch_size, self.struct_dim, device=device, dtype=dtype),
        )
        status_raw = encoder_out.get(
            "status_state",
            torch.zeros(batch_size, self.status_dim, device=device, dtype=dtype),
        )
        value = self.value_proj(value_raw)
        struct = self.struct_proj(struct_raw)
        status = self.status_proj(status_raw)
        return self.norm(value + 0.3 * struct + 0.3 * status)


class LowRankPolicyOperator(nn.Module):
    """Estimate a low-rank latent displacement operator for each policy recipe."""

    def __init__(self, recipe_dim: int, hidden_dim: int, rank: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.net = nn.Sequential(
            nn.Linear(recipe_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim * rank),
        )

    def factors(self, state: torch.Tensor, recipe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([state, recipe], dim=-1))
        left, right = raw.chunk(2, dim=-1)
        left = left.view(state.size(0), self.hidden_dim, self.rank)
        right = right.view(state.size(0), self.hidden_dim, self.rank)
        left = F.normalize(left, p=2, dim=1)
        right = F.normalize(right, p=2, dim=1)
        return left, right

    def apply(self, vector: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # A v = U (V^T v), with U,V: [B, H, R].
        coeff = torch.einsum("bhr,bh->br", right, vector)
        return torch.einsum("bhr,br->bh", left, coeff)


class KrylovAnnihilator(nn.Module):
    """Learn p(A) = I + c1 A + ... + cM A^M for policy-mode removal."""

    def __init__(self, recipe_dim: int, degree: int):
        super().__init__()
        self.degree = degree
        self.coeff_head = nn.Sequential(
            nn.Linear(recipe_dim, 64),
            nn.SiLU(),
            nn.Linear(64, degree),
            nn.Tanh(),
        )

    def polynomial_apply(
        self,
        vector: torch.Tensor,
        recipe: torch.Tensor,
        operator: LowRankPolicyOperator,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coeff = self.coeff_head(recipe)  # [B, degree], bounded for stability.
        out = vector
        power = vector
        for idx in range(self.degree):
            power = operator.apply(power, left, right)
            out = out + coeff[:, idx : idx + 1] * power
        return out, coeff


class KrylovPolicyModeAnnihilator(nn.Module):
    """Wrap any irregular encoder with counterfactual Krylov policy-mode filtering."""

    def __init__(
        self,
        base_encoder: nn.Module,
        value_dim: int,
        struct_dim: int,
        status_dim: int,
        hidden_dim: int,
        recipe_dim: int,
        num_classes: int,
        num_status: int,
        rank: int = 8,
        degree: int = 3,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.adapter = StructuralStatusAdapter(value_dim, struct_dim, status_dim, hidden_dim)
        self.policy_operator = LowRankPolicyOperator(recipe_dim, hidden_dim, rank)
        self.annihilator = KrylovAnnihilator(recipe_dim, degree)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.status_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_status),
        )
        self.degree = degree

    def encode_state(self, batch: dict) -> torch.Tensor:
        return self.adapter(self.base_encoder(batch))

    def apply_recipe(self, h0: torch.Tensor, recipe: torch.Tensor) -> dict:
        left, right = self.policy_operator.factors(h0, recipe)
        filtered, coeff = self.annihilator.polynomial_apply(
            vector=h0,
            recipe=recipe,
            operator=self.policy_operator,
            left=left,
            right=right,
        )
        first_mode = self.policy_operator.apply(h0, left, right)
        residual = self.policy_operator.apply(filtered, left, right)
        return {
            "filtered": filtered,
            "coeff": coeff,
            "left": left,
            "right": right,
            "first_mode": first_mode,
            "annihilation_residual": residual,
        }

    def forward(self, batch: dict) -> dict:
        h0 = self.encode_state(batch)
        recipes = batch["policy_recipe_bank"]  # [B, R, recipe_dim]

        filtered_states = []
        coeffs = []
        first_modes = []
        residuals = []
        for recipe in recipes.unbind(dim=1):
            out_r = self.apply_recipe(h0, recipe)
            filtered_states.append(out_r["filtered"])
            coeffs.append(out_r["coeff"])
            first_modes.append(out_r["first_mode"])
            residuals.append(out_r["annihilation_residual"])

        z = torch.stack(filtered_states, dim=1).mean(dim=1)
        logits = self.classifier(z)
        status_logits = self.status_decoder(z)
        return {
            "logits": logits,
            "status_logits": status_logits,
            "base_state": h0,
            "filtered_state": z,
            "coeff": torch.stack(coeffs, dim=1),
            "first_modes": torch.stack(first_modes, dim=1),
            "annihilation_residuals": torch.stack(residuals, dim=1),
        }

    def operator_fit_loss(self, batch: dict, h0: torch.Tensor) -> torch.Tensor:
        losses = []
        for recipe, cf_batch in zip(
            batch["policy_recipe_bank"].unbind(dim=1),
            batch["cf_batch_bank"],
        ):
            with torch.no_grad():
                h_cf = self.encode_state(cf_batch)
                target_delta = h_cf - h0
            left, right = self.policy_operator.factors(h0, recipe)
            pred_delta = self.policy_operator.apply(h0, left, right)
            losses.append(F.smooth_l1_loss(pred_delta, target_delta))
        return torch.stack(losses).mean()

    def policy_annihilation_loss(self, batch: dict, h0: torch.Tensor) -> torch.Tensor:
        losses = []
        for recipe, cf_batch in zip(
            batch["policy_recipe_bank"].unbind(dim=1),
            batch["cf_batch_bank"],
        ):
            with torch.no_grad():
                h_cf = self.encode_state(cf_batch)
                delta = h_cf - h0

            left, right = self.policy_operator.factors(h0, recipe)
            krylov = delta
            for _ in range(self.degree):
                filtered, _ = self.annihilator.polynomial_apply(
                    vector=krylov,
                    recipe=recipe,
                    operator=self.policy_operator,
                    left=left,
                    right=right,
                )
                losses.append(filtered.pow(2).mean())
                krylov = self.policy_operator.apply(krylov, left, right)
        return torch.stack(losses).mean()

    def polynomial_condition_loss(self, out: dict, lower: float = 0.25, upper: float = 1.75) -> torch.Tensor:
        coeff_loss = out["coeff"].pow(2).mean()
        ratio = out["filtered_state"].norm(dim=-1) / out["base_state"].norm(dim=-1).clamp_min(1e-6)
        ratio_loss = F.relu(ratio - upper).pow(2).mean() + F.relu(lower - ratio).pow(2).mean()
        residual_loss = out["annihilation_residuals"].pow(2).mean()
        return coeff_loss + ratio_loss + residual_loss

    def training_loss(
        self,
        batch: dict,
        lambda_fit: float = 0.25,
        lambda_ann: float = 0.20,
        lambda_sem: float = 0.10,
        lambda_cond: float = 0.05,
    ) -> dict:
        out = self.forward(batch)
        labels = batch["labels"]
        cls_loss = F.cross_entropy(out["logits"], labels)

        fit_loss = self.operator_fit_loss(batch, out["base_state"].detach())
        ann_loss = self.policy_annihilation_loss(batch, out["base_state"].detach())

        if "stable_status_label" in batch:
            sem_loss = F.cross_entropy(out["status_logits"], batch["stable_status_label"])
        else:
            sem_loss = torch.zeros((), device=labels.device, dtype=out["logits"].dtype)

        cond_loss = self.polynomial_condition_loss(out)

        total = (
            cls_loss
            + lambda_fit * fit_loss
            + lambda_ann * ann_loss
            + lambda_sem * sem_loss
            + lambda_cond * cond_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "operator_fit_loss": fit_loss.detach(),
            "policy_annihilation_loss": ann_loss.detach(),
            "semantic_retention_loss": sem_loss.detach(),
            "polynomial_condition_loss": cond_loss.detach(),
            "policy_mode_energy": out["first_modes"].norm(dim=-1).mean(dim=1).detach(),
            "annihilation_residual": out["annihilation_residuals"].norm(dim=-1).mean(dim=1).detach(),
        }


@torch.no_grad()
def build_policy_operator_bank(batch: dict, num_vars: int, recipe_dim: int = 5) -> dict:
    """Sketch counterfactual policy views for Krylov operator learning."""

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    recipes = []
    cf_batches = []

    def clone_with(new_value, new_time, new_var, new_mask):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        return out

    def recipe(index: int):
        r = torch.zeros(bsz, recipe_dim, device=device, dtype=value.dtype)
        r[:, index] = 1.0
        return r

    # 1. STAR-Set locality shift: round timestamps into coarse clinical rounds.
    rounded_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    recipes.append(recipe(0))
    cf_batches.append(clone_with(value * mask, rounded_time, var_id, mask))

    # 2. Variable-affinity panel shift: thin close cross-variable co-observations.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    panel_mask = mask * (1.0 - 0.5 * close * changed_var)
    recipes.append(recipe(1))
    cf_batches.append(clone_with(value * panel_mask, time, var_id, panel_mask))

    # 3. VP-GNN patch visibility shift: keep a tiny fixed event budget per third.
    patch_keep = torch.zeros_like(mask)
    for start, end in [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]:
        in_patch = ((time_norm >= start) & (time_norm < end)).to(mask.dtype) * mask
        rank = in_patch.cumsum(dim=1)
        patch_keep = torch.maximum(patch_keep, (rank <= 2).to(mask.dtype) * in_patch)
    recipes.append(recipe(2))
    cf_batches.append(clone_with(value * patch_keep, time, var_id, patch_keep))

    # 4. Status-set shift: emphasize future/later status events while thinning early routine events.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    status_mask = torch.where(late > 0, mask, mask * alternating)
    recipes.append(recipe(3))
    cf_batches.append(clone_with(value * status_mask, time, var_id, status_mask))

    # 5. Semantic event shift: drop rare variable ids as a proxy for site-specific event semantics.
    rare_proxy = (var_id % 3 == 0).to(mask.dtype)
    semantic_mask = mask * (1.0 - 0.5 * rare_proxy)
    recipes.append(recipe(4))
    cf_batches.append(clone_with(value * semantic_mask, time, var_id, semantic_mask))

    out = dict(batch)
    out["policy_recipe_bank"] = torch.stack(recipes, dim=1)
    out["cf_batch_bank"] = cf_batches
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `locality-bias shift`：训练阶段时间局部邻域与测试阶段查房/报警 cadence 不同。
   - `variable-affinity shift`：训练医院联测 panel，测试医院拆成异步检查。
   - `patch-visibility shift`：关键 patch 在测试政策下预算压缩。
   - `future-status visibility shift`：EHR-SPC 式未来 event set 的可见性变化。
   - `semantic-event shift`：LLM4EHR 式事件序列语义锚点在不同医院记录习惯下改变。

2. **对比方法**
   - STAR-Set / VP-GNN 原模型。
   - EHR-SPC status encoder / LLM4EHR semantic alignment 原模型。
   - 普通反事实增强、mask dropout、policy adversarial baseline。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、D-IVSP、DJRT。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - policy-mode energy：政策算子一阶模态能量。
   - annihilation residual：`||p(A) A h||` 是否足够低。
   - semantic retention：湮灭后 status/semantic label 是否仍可恢复。
   - mode-label leakage：只用 `A_r h` 预测标签的能力，应随训练下降。
   - structure-shift robustness：按 temporal bias、variable affinity、patch visibility 分桶后的性能退化。

4. **消融实验**
   - 去掉 `L_operator_fit`，验证低秩政策算子是否变成任意噪声。
   - 去掉 `L_policy_annihilation`，检查 policy modes 是否重新污染分类。
   - 去掉 `L_semantic_retention`，检查多项式是否过度删除状态信息。
   - 将低秩 operator 换成 MLP residual，验证收益来自 operator/Krylov 结构。
   - 将 counterfactual policy bank 换成随机 mask，验证收益来自可解释 policy recipes。
   - 扫描多项式 degree 与 operator rank，评估鲁棒性、语义保留和计算成本的 trade-off。

## 5. 预期创新性

1. **从 policy view 一致性转向 policy mode 湮灭**：不要求不同采样视图预测相同，而是在 latent space 中识别并消除反复出现的政策模态。
2. **从图/后验/校准/投票转向 Krylov 子空间**：避开历史的图分解、后验商、保形校准和排序陪审团，把 sampling-policy shortcut 表述为低秩线性算子的可湮灭模态。
3. **吸收 STAR-Set / VP-GNN 但不复用其双图拆分**：attention bias、变量图、patch 图只作为结构摘要进入 latent state，真正的鲁棒化由 policy operator 完成。
4. **吸收 EHR-SPC / LLM4EHR 但不做普通语义对齐**：status / semantic label 只用于约束湮灭后仍保留病程语义，避免“删掉一切”的伪鲁棒。
5. **诊断直接可用**：policy-mode energy 与 annihilation residual 可以告诉部署者当前预测是否被某类采样制度模态支配。

## 6. 一句话投稿卖点

**KPMA 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“反事实采样政策在 latent state 上诱发的低秩 Krylov 模态污染”，并通过可微政策位移算子、湮灭多项式与语义保真约束，在不依赖危险率、对抗、一致性、后验商、随机平滑、拓扑、gauge、纠错码、knockoff、证据盾、信息格、求解轨迹、保形、IV 或政策陪审团的前提下，切断 STAR-Set/VP-GNN/EHR-SPC/LLM4EHR 式结构与语义表征中的采样政策模态捷径。**
