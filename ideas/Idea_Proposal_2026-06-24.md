# Title: Policy-Shadow Negative Film：面向采样策略偏移的反事实负片擦影分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已读取自动化记忆 `MEMORIES.md`，纳入其中记录但当前工作区未检出的历史提案核心机制。
- 已读取 `paper_daily.md` 与 `paper_daily_2026-06-12.md`。
- 已读取当前工作区内全部历史提案：
  - `ideas/Idea_Proposal_2026-06-12.md`
  - `ideas/Idea_Proposal_2026-06-13.md`
  - `ideas/Idea_Proposal_2026-06-14.md`
  - `ideas/Idea_Proposal_2026-06-16.md`
  - `ideas/Idea_Proposal_2026-06-19.md`
  - `ideas/Idea_Proposal_2026-06-21.md`
  - `ideas/Idea_Proposal_2026-06-22.md`
  - `ideas/Idea_Proposal_2026-06-23.md`
- 已通过仓库检索确认：当前工作区未发现可替代的 `*summary*.md`、`*work*.md` 或中文“总结”类 Markdown 文件。

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
22. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing，或 MedMamba 的 frequency/adaptive graph branch 作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不做策略对抗，不做跨视图一致，不做图/拓扑/规范几何，也不给证据征税；而是把采样政策看成会在事件胶片上留下“可显影但无状态语义的影子曝光”。反事实干预模块负责制造已知影子负片，模型学习一个 policy-shadow eraser：凡是仅由采样坐标、同步机会或观测预算制造出的可分类痕迹，都必须在进入分类器前被擦影；真正的状态证据则通过值保持约束留下。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列中的 sampling-policy shift 往往不是单个 mask ratio 的变化，而是一种“胶片曝光方式”的改变：

- 医院 A 在报警后产生密集复测 burst，医院 B 只保留固定查房时刻；
- 设备 A 因电量策略降低夜间采样，设备 B 保持低频均匀采样；
- 实验室 panel 在一个机构中同步出现，在另一个机构中被拆成异步事件。

这些策略会在输入事件流上留下很强的 **shadow evidence**：事件密度、变量同窗机会、早晚期观测预算、某些变量是否成组出现。它们在训练环境中可能与标签高度相关，但在测试环境换策略后会迅速失效。

近期 `paper_daily.md` 给出几个关键启发：

1. **MTM** 指出 channel-wise asynchrony 是 IMTS 分类的瓶颈，token mixing 的收益来自让异步通道发生信息交换；但被选中的 pivotal token 可能正是训练策略制造出的同步机会。
2. **MedMamba** 说明 SSM 主干和 raw/difference 视图适合医疗时序的长程/非平稳建模；但 missing-channel robustness 不等于 sampling-policy robustness，自适应图或频域分支仍可能吸收设备/医院策略。
3. **QuITE** 证明轻量输入适配层可以把 IMTS 接到通用 backbone 上；但普通 query 或 adapter 若直接聚合时间戳、变量 id 与观测值，也会把 policy shadow 一并压进主表示。
4. **PYRREGULAR** 提醒我们需要统一表示自然不规则性；这正适合在 dataloader 层生成可控的 shadow-negative recipes，系统检验模型是否只凭采样胶片就能分类。

Policy-Shadow Negative Film (PSNF) 的核心直觉是：

> 如果把观测值擦掉，只保留采样坐标、变量 id、局部密度和同步簇，分类器仍能预测标签，那么模型看到的是 policy shadow，而不是稳定状态。我们不再试图估计这个 shadow 的概率、危险率、测度比或后验因子；我们直接显影一张“采样负片”，训练 eraser 在主胶片中擦掉这些影子曝光。

这与当前“采样解耦/反事实干预”框架的结合方式非常直接：

- value process 仍负责把观测值事件编码成状态候选；
- sampling process 不输出 hazard、policy code、price、quotient、gauge frame 或 stop recipe，而是输出 **shadow stencil**：哪些 latent patch 看起来像采样负片；
- counterfactual intervention 不用于一致性、不用于风险方差、不用于随机平滑，而是制造带有已知 shadow stencil 的 policy-only / policy-spliced 负片；
- classifier 只消费经过 eraser 擦影后的 positive print，因此不能轻易依赖观测密度、变量同窗、panel batching 或某类设备预算造成的捷径。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通事件编码改为 Dual-Exposure Film Encoder

给定事件流 `(t_i, d_i, x_i)`，PSNF 构造两张“胶片”：

1. **Positive value exposure**
   - 输入真实观测值、变量 id、相对时间间隔。
   - 输出 `v_i`，表示观测值携带的状态候选。

2. **Shadow coordinate exposure**
   - 输入时间戳、变量 id、局部采样密度、同窗计数、早/中/晚窗口预算等坐标特征。
   - 不访问当前观测值 `x_i`。
   - 输出 `s_i`，表示仅由采样政策显影出的影子痕迹。

两者经过负片显影器：

```text
z_i = ValueFilm(x_i, d_i, delta_t_i)
s_i = ShadowFilm(t_i, d_i, density_i, co_window_i, budget_i)
e_i = ShadowEraser(z_i, s_i) in [0, 1]^H
print_i = z_i * (1 - e_i)
logits = Classifier(Backbone({print_i}))
```

`e_i` 是维度级 erasure stencil，不是 attention 权重、不是 evidence tax、不是 missingness feature、不是 policy residual sink。它只决定 value latent 中哪些维度与 shadow film 过度共显影，应该在进入分类器前被擦掉。

### 2.2 改 Dataloader：返回 Shadow-Negative Bank，而不是一致性增强视图

新增 `ShadowNegativeCollator`，每个 batch 除事实样本外返回三类负片：

1. **Policy-only negative**
   - 保留 `event_time`、`event_var_id`、局部 density / co-window / budget 特征；
   - 将 `event_value` 替换为样本内均值、变量均值或 learned neutral value；
   - 目的：检验仅采样胶片是否足以分类。

2. **Policy-spliced negative**
   - 保留当前样本的观测值排序与标签；
   - 将某些时间窗或变量组的采样坐标特征替换为另一个策略模板；
   - 同时返回 `shadow_stencil_target`，标记哪些事件/维度被策略模板影响。

3. **Value-preserving blackout**
   - 只遮蔽坐标特征，不遮蔽观测值；
   - 用于训练 eraser 不要把真实状态值一并擦掉。

这些负片不是正样本 pair，不要求 hidden state 或 logits 与事实视图一致；它们只是给 eraser 提供“哪些分类痕迹可以由采样胶片单独产生”的监督。

### 2.3 改 Loss：从对抗/一致性转向 Shadow Nullification + Value Retention

总目标：

```text
L = L_cls
  + lambda_null  * L_shadow_null
  + lambda_edit  * L_stencil_edit
  + lambda_keep  * L_value_retention
  + lambda_sparse * L_eraser_sobriety
```

#### A. 分类损失 `L_cls`

只在擦影后的 positive print 上分类：

```text
L_cls = CE(Classifier(Backbone({z_i * (1 - e_i)})), y)
```

对 policy-spliced negative 也可以使用真实标签做普通 CE，但不做 pairwise consistency：

```text
L_cls_spliced = CE(f(erase(policy_spliced_x)), y)
```

关键点：训练目标不是“spliced logits 必须等于 factual logits”，而是“在擦掉已知采样影子后，剩余状态证据仍应足以预测标签”。

#### B. Shadow Nullification `L_shadow_null`

把 policy-only negative 输入模型时，若擦影器工作正确，分类器应接近无证据状态，即高熵/低置信：

```text
p_shadow = softmax(f(policy_only_negative))
L_shadow_null = KL(Uniform || p_shadow)
```

这不是 adversarial loss：没有策略判别器、没有梯度反转、没有要求主表示骗过环境分类器。它只是直接检查“只看采样胶片时，分类器是否仍能偷看标签”。

#### C. Stencil Edit Supervision `L_stencil_edit`

反事实干预模块知道哪些事件坐标被 policy splice、panel batching、budget shift 或 burst thinning 改动过。PSNF 用这些已知编辑区域监督 eraser：

```text
L_stencil_edit =
  BCE(mean_h(e_i), shadow_stencil_target_i)
```

这不是 topology censor interval、chart span 或 stopping recipe；它不关心持久区间、坐标规范位移或停时矩，只做“采样影子区域在哪里”的负片标注。

#### D. Value Retention `L_value_retention`

为了防止 eraser 把所有信息都擦掉，PSNF 对擦影后的 print 加一个轻量值保持约束：从 `print_i` 预测当前观测值的分位桶或符号化幅度。

```text
q_i = ValueBucketHead(print_i)
L_value_retention = CE(q_i, bucket(x_i))
```

它不是 iTimER 式 reconstruction-error distribution，也不生成 pseudo-observations；它只保证擦影后的 latent 仍保留观测值的状态语义。

#### E. Eraser Sobriety `L_eraser_sobriety`

eraser 不能塌缩为全擦或全留：

```text
L_eraser_sobriety =
  mean(e_i)                         # 保持稀疏擦除
  + relu(min_keep - mean(1 - e_i))^2 # 保留足够状态通道
```

这不是 evidence budget 或 protocol tax，因为它不按 token 购买证据，也不为采样依赖证据定价；它只是防止擦影器退化。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `ValueFilmEncoder`。
- 现有 sampling branch 改为 `ShadowFilmEncoder + ShadowEraser`：只学习采样负片如何污染 value latent，不把 shadow features 输入分类头。
- 现有 counterfactual intervention 改为生成 `shadow_negative_bank` 与 `shadow_stencil_target`，用于负片擦影训练。
- 推理阶段无需策略标签、无需生成反事实视图、无需估计采样概率；只根据事实样本生成 shadow film，擦掉疑似影子曝光后分类。
- 可解释输出包括：
  - 每个事件的 `erase_rate`；
  - 哪些变量/时间窗被判定为 policy shadow；
  - `shadow-only confidence`：仅采样胶片是否足以产生错误高置信。

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def event_delta_time(event_time: torch.Tensor) -> torch.Tensor:
    delta_t = torch.zeros_like(event_time)
    delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
    if event_time.size(1) > 1:
        delta_t[:, 0] = delta_t[:, 1]
    return delta_t


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


class ValueFilmEncoder(nn.Module):
    """Encode observed values into positive film exposure."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        event_value: torch.Tensor,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = event_delta_time(event_time)
        var_h = self.var_embed(event_var_id)
        x = torch.cat(
            [
                var_h,
                event_value.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(x) * event_mask.unsqueeze(-1)


class ShadowFilmEncoder(nn.Module):
    """Encode sampling coordinates into a policy-shadow exposure."""

    def __init__(self, num_vars: int, shadow_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 7, shadow_dim),
            nn.SiLU(),
            nn.Linear(shadow_dim, shadow_dim),
        )

    def forward(
        self,
        event_time: torch.Tensor,
        event_var_id: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = event_delta_time(event_time)
        horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = event_time / horizon

        local_density = 1.0 / (1.0 + delta_t)
        var_onehot = F.one_hot(event_var_id.clamp_min(0), self.num_vars).to(event_time.dtype)

        # Coordinate-only descriptors: no current observed value is used.
        early = (time_norm <= 0.33).to(event_time.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(event_time.dtype)
        late = (time_norm > 0.66).to(event_time.dtype)
        same_as_prev = torch.zeros_like(event_time)
        same_as_prev[:, 1:] = (event_var_id[:, 1:] == event_var_id[:, :-1]).to(event_time.dtype)

        coord = torch.cat(
            [
                var_onehot,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                local_density.unsqueeze(-1),
                early.unsqueeze(-1),
                middle.unsqueeze(-1),
                late.unsqueeze(-1),
                same_as_prev.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(coord) * event_mask.unsqueeze(-1)


class ShadowEraser(nn.Module):
    """Predict which value-film dimensions look like policy-shadow exposure."""

    def __init__(self, hidden_dim: int, shadow_dim: int):
        super().__init__()
        self.to_gate = nn.Sequential(
            nn.Linear(hidden_dim + shadow_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, value_film: torch.Tensor, shadow_film: torch.Tensor) -> torch.Tensor:
        return self.to_gate(torch.cat([value_film, shadow_film], dim=-1))


class FilmBackbone(nn.Module):
    """Small event backbone over erased positive prints."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, print_film: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(print_film * event_mask.unsqueeze(-1))
        pooled = masked_mean(h, event_mask, dim=1)
        return self.norm(pooled)


class PolicyShadowNegativeFilm(nn.Module):
    """Classify irregular time series after erasing sampling-policy shadows."""

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        shadow_dim: int,
        num_classes: int,
        num_value_buckets: int = 16,
    ):
        super().__init__()
        self.value_film = ValueFilmEncoder(num_vars=num_vars, hidden_dim=hidden_dim)
        self.shadow_film = ShadowFilmEncoder(num_vars=num_vars, shadow_dim=shadow_dim)
        self.eraser = ShadowEraser(hidden_dim=hidden_dim, shadow_dim=shadow_dim)
        self.backbone = FilmBackbone(hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.bucket_head = nn.Linear(hidden_dim, num_value_buckets)
        self.num_value_buckets = num_value_buckets

    def encode_print(self, batch: dict) -> dict:
        value = self.value_film(
            event_value=batch["event_value"],
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
        )
        shadow = self.shadow_film(
            event_time=batch["event_time"],
            event_var_id=batch["event_var_id"],
            event_mask=batch["event_mask"],
        )
        erase_gate = self.eraser(value, shadow)
        print_film = value * (1.0 - erase_gate)
        signature = self.backbone(print_film, batch["event_mask"])
        logits = self.classifier(signature)
        return {
            "value_film": value,
            "shadow_film": shadow,
            "erase_gate": erase_gate,
            "print_film": print_film,
            "signature": signature,
            "logits": logits,
        }

    def forward(self, batch: dict) -> dict:
        return self.encode_print(batch)

    def shadow_null_loss(self, shadow_batch: dict) -> torch.Tensor:
        out = self.encode_print(shadow_batch)
        log_prob = F.log_softmax(out["logits"], dim=-1)
        uniform = torch.full_like(log_prob, 1.0 / log_prob.size(-1))
        return F.kl_div(log_prob, uniform, reduction="batchmean")

    def stencil_edit_loss(self, out: dict, batch: dict) -> torch.Tensor:
        target = batch["shadow_stencil_target"].to(out["erase_gate"].dtype)
        event_gate = out["erase_gate"].mean(dim=-1)
        raw = F.binary_cross_entropy(event_gate, target, reduction="none")
        return (raw * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

    def value_retention_loss(self, out: dict, batch: dict) -> torch.Tensor:
        # Bucketize values per batch. Production code should use training-set quantiles.
        value = batch["event_value"]
        lo = value.masked_fill(batch["event_mask"] == 0, 0.0).amin(dim=1, keepdim=True)
        hi = value.masked_fill(batch["event_mask"] == 0, 0.0).amax(dim=1, keepdim=True)
        scaled = (value - lo) / (hi - lo).clamp_min(1e-6)
        bucket = (scaled * (self.num_value_buckets - 1)).long().clamp(0, self.num_value_buckets - 1)

        bucket_logits = self.bucket_head(out["print_film"])
        raw = F.cross_entropy(
            bucket_logits.flatten(0, 1),
            bucket.flatten(),
            reduction="none",
        ).view_as(value)
        return (raw * batch["event_mask"]).sum() / batch["event_mask"].sum().clamp_min(1.0)

    def eraser_sobriety_loss(self, out: dict, event_mask: torch.Tensor, min_keep: float = 0.35) -> torch.Tensor:
        erase = out["erase_gate"].mean(dim=-1)
        erase_rate = (erase * event_mask).sum() / event_mask.sum().clamp_min(1.0)
        keep_rate = ((1.0 - erase) * event_mask).sum() / event_mask.sum().clamp_min(1.0)
        return erase_rate + F.relu(min_keep - keep_rate).pow(2)

    def training_loss(
        self,
        batch: dict,
        lambda_null: float = 0.2,
        lambda_edit: float = 0.15,
        lambda_keep: float = 0.1,
        lambda_sparse: float = 0.02,
    ) -> dict:
        factual = self.forward(batch)
        cls_loss = F.cross_entropy(factual["logits"], batch["labels"])

        # Policy-only negative: sampling coordinates remain, values are neutralized.
        shadow_batch = dict(batch)
        neutral = masked_mean(batch["event_value"], batch["event_mask"], dim=1).unsqueeze(1)
        shadow_batch["event_value"] = neutral.expand_as(batch["event_value"]) * batch["event_mask"]
        null_loss = self.shadow_null_loss(shadow_batch)

        edit_loss = self.stencil_edit_loss(factual, batch)
        keep_loss = self.value_retention_loss(factual, batch)
        sparse_loss = self.eraser_sobriety_loss(factual, batch["event_mask"])

        total = (
            cls_loss
            + lambda_null * null_loss
            + lambda_edit * edit_loss
            + lambda_keep * keep_loss
            + lambda_sparse * sparse_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "shadow_null_loss": null_loss.detach(),
            "stencil_edit_loss": edit_loss.detach(),
            "value_retention_loss": keep_loss.detach(),
            "eraser_sobriety_loss": sparse_loss.detach(),
            "mean_erase_rate": factual["erase_gate"].mean().detach(),
        }


@torch.no_grad()
def build_shadow_stencil_target(
    event_time: torch.Tensor,
    event_var_id: torch.Tensor,
    event_mask: torch.Tensor,
    num_vars: int,
) -> torch.Tensor:
    """Mark event coordinates likely touched by policy-shadow edits."""

    horizon = (event_time * event_mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = event_time / horizon
    delta_t = event_delta_time(event_time)

    dense_burst = (delta_t <= delta_t.masked_fill(event_mask == 0, 0.0).mean(dim=1, keepdim=True)).to(event_time.dtype)
    late_budget = (time_norm > 0.66).to(event_time.dtype)

    repeated_var = torch.zeros_like(event_time)
    repeated_var[:, 1:] = (event_var_id[:, 1:] == event_var_id[:, :-1]).to(event_time.dtype)

    rare_var = torch.zeros_like(event_time)
    for var_idx in range(num_vars):
        var_obs = ((event_var_id == var_idx).to(event_time.dtype) * event_mask)
        var_rate = var_obs.sum(dim=1, keepdim=True) / event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        rare_var = torch.where(
            (event_var_id == var_idx) & (var_rate < (1.0 / max(num_vars, 1))),
            torch.ones_like(rare_var),
            rare_var,
        )

    stencil = torch.maximum(torch.maximum(dense_burst, late_budget), torch.maximum(repeated_var, rare_var))
    return stencil * event_mask
```

## 4. 实验切入点

1. **Policy shadow 构造**
   - `burst-shadow shift`：训练策略保留报警后 dense burst，测试策略合并 burst。
   - `panel-shadow shift`：训练策略同步 panel 化，测试策略拆成异步事件。
   - `budget-shadow shift`：测试设备对晚期或某些变量组施加预算稀疏化。
   - `adapter-shadow shift`：在 QuITE/Transformer/SSM 输入适配层前插入 shadow-only negative，检查 backbone 是否只凭采样胶片分类。

2. **对比方法**
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - DHN hazard/null-space baseline。
   - CGS commutator graph surgery baseline。
   - PT-AEM evidence market baseline。
   - PQD posterior quotient baseline。
   - DS-CS certified smoothing baseline。
   - DM-DRR measure-ratio baseline。
   - OS-MQ optional-stopping martingale baseline。
   - CETC censored topology baseline。
   - PGHT policy-gauge horizontal transport baseline。
   - MTM / QuITE / MedMamba 风格主干加普通 mask augmentation。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - shadow-only confidence：只输入 policy-only negative 时的最大类别概率。
   - shadow leakage gap：事实样本置信度与 policy-only 置信度的相关性。
   - erase localization F1：eraser 与已知 policy edit stencil 的匹配度。
   - value retention accuracy：擦影后 print 对观测值分位桶的预测能力。

4. **消融实验**
   - 去掉 `L_shadow_null`，验证模型是否重新利用采样负片分类。
   - 去掉 `L_stencil_edit`，验证 eraser 是否失去对政策编辑区域的定位能力。
   - 去掉 `L_value_retention`，检查是否出现“擦掉一切”的虚假鲁棒。
   - 将 policy-only negative 替换为随机值噪声，验证收益来自采样胶片负片而非普通噪声增强。
   - 在 MTM/QuITE/MedMamba backbone 前后插入 PSNF，验证它是可插拔的 policy-shadow firewall。

## 5. 预期创新性

1. **从采样机制估计转向采样负片显影**：不问某点为什么被测，也不估计其概率；直接构造“无值但有采样坐标”的 policy-only negative，逼迫模型承认哪些分类痕迹来自采样胶片。
2. **从一致性鲁棒转向无证据高熵原则**：不要求事实视图与反事实视图输出相同；只要求仅由采样影子构成的输入不能给出低熵类别判断。
3. **从 token 选择/征税转向 latent erasure stencil**：不是选择重要 token、购买证据或征收协议税，而是在 value latent 维度上擦掉与 shadow film 共显影的污染。
4. **从图/频域/拓扑/规范几何转向胶片双曝光诊断**：吸收 MTM 对异步同步机会的洞察、MedMamba 对医疗长序列主干的启发、QuITE 的适配层风险，但核心机制是负片擦影，不复用其主干创新。
5. **解释性直接服务部署**：shadow-only confidence 可以作为部署前的策略捷径预警；erase localization 可以指出模型原本依赖的是哪段采样预算、panel 同步或 burst 复测。

## 6. 一句话投稿卖点

**PSNF 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“采样胶片的影子曝光污染”问题，并通过 policy-only negative film、反事实 shadow stencil 与 latent eraser，让分类器在进入主干前擦掉仅由采样坐标显影出的捷径，同时保留观测值驱动的状态证据。**
