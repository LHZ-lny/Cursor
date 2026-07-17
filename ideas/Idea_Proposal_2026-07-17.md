# Title: Policy-Word Signature Renormalizer：面向采样策略偏移的采样词反项签名重整化分类器

## 0. 强制读取记录与思维黑名单

### 已读取与检索的材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md`：其中记录了历史提案机制黑名单，以及多轮任务均未发现 `my_work_summary.md`。
- 已读取近期论文记录：`paper_daily.md` 与 `paper_daily_2026-07-13.md`。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史机制：`2026-06-17`、`2026-06-20`、`2026-06-24`、`2026-06-27`、`2026-07-15`、`2026-07-16`。

### 历史核心机制黑名单

为避免与历史提案和 `paper_daily.md` 中已有机制发生思维重合，本提案明确避开以下方向作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、hazard-driven resampling、分类梯度与采样 score 零空间。
8. 生理流算子与采样算子的交换子手术、图交换或残差 sink。
9. additive evidence market、protocol tax、token-level budget、边际证据审计。
10. posterior quotient、模型空间后验商、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、VQ semantic clauses、HSIC redaction checksum。
12. policy-simplex randomized smoothing、certified policy radius。
13. Radon-Nikodym density ratio、doubly robust target-measure correction、RKHS cubature moment exactness。
14. optional-stopping martingale query、停时矩控制、标准化创新鞅。
15. soft excursion topology、censored persistence interval、拓扑审查 envelope。
16. policy-gauge frame、horizontal transport、chart span supervision。
17. policy-only negative film、shadow eraser/stencil。
18. latent packet codeword、parity-check、syndrome repair。
19. conditional knockoff calendar、soft knockoff-FDR firewall。
20. observability witness、measurement-action bisimulation、canonical measurement response battery。
21. subjective-logic evidential shield、policy-induced vacuity mass。
22. observation-set policy lattice、meet/join、单调/次模边际契约。
23. solver trace front-door、NFE/roughness/step-size 中介标准化。
24. 单纯复用 FlowPath 的可逆路径、MVC-CDE 的普通多视图平滑、GSNF/DBGL/GARLIC 图衰减、iTimER 误差伪观测、Record2Vec/LLM 摘要、QuITE 普通 query、MTM pivotal token mixing、MedMamba frequency/adaptive graph、MedSpaformer token sparsification、MILM value-redacted classifier、StarEmbed foundation embedding。

本提案选择新的正交切入点：**不估计采样概率，不做对抗、不做一致性、不做平滑认证、不做后验除法、不做停时/拓扑/gauge/纠错/knockoff/观测性/证据无知/信息格/solver trace；而是把不规则观测路径写成“值词 value words + 采样词 policy words”的截断路径签名，并在 shuffle algebra 中学习只作用于混合词的 counterterm。分类器只读经过采样词反项重整化后的纯值签名，从代数层面切断采样策略与状态路径几何的混合迭代积分捷径。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

最新 `paper_daily_2026-07-13.md` 中的 **Efficient Neural Controlled Differential Equations via Attentive Kernel Smoothing (MVC-CDE)** 说明：非规则采样分类的关键不只是离散事件本身，而是这些事件诱导出的 control path 几何。采样政策一旦改变，路径的局部粗糙度、平滑带宽偏好、多视图注意力和连续时间驱动都会改变。

历史提案已经从许多层面处理过这种风险：求解轨迹、图结构、后验、拓扑、gauge、证据、信息格等。但还有一个更底层的代数事实没有被利用：

> 连续时间模型实际消费的是路径；路径分类器最容易偷用的不是单个 mask，而是“观测值变化”和“采样坐标变化”在迭代积分中的混合词。

例如，对于一条增强路径

```text
dX_t = [dV_t, dP_t]
```

其中 `V` 是观测值/状态增量，`P` 是采样坐标增量，如时间 gap、变量可见性、panel 共现、局部密度、测量质量。二阶签名中会自然出现：

```text
Integral dV_i dP_j,  Integral dP_j dV_i
```

这些 **value-policy mixed words** 很容易成为 sampling-policy shortcut：

- 同样的乳酸变化，如果总是在训练医院的报警后 dense burst 中出现，`dV x dP` 会把“值变化”和“复测政策”绑定起来。
- 同样的多变量趋势，如果只在训练机构的同步 panel 中被看到，`dP x dV` 会把“panel 日历”和“状态证据”缠在一起。
- MVC-CDE 的 kernel smoothing 虽能降低路径粗糙度，但如果多视图 attention 学到某些采样词与值词的混合迭代积分，它仍可能把采样几何误当成类别几何。

**Policy-Word Signature Renormalizer (PWSR)** 的核心思想是：

> 不在输入层删掉采样信息，也不在表示层要求多视图一致，而是在路径签名代数中把包含采样词的混合迭代积分视为 nuisance ideal。采样分支只负责估计这些混合词对纯值签名造成的 counterterm；分类器只能读取被 counterterm 重整化后的 pure value signature。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 负责构造值路径增量 `dV`；
- sampling process 负责构造采样词增量 `dP`，但 `dP` 不进入分类器；
- counterfactual intervention 不再生成一致性 pair 或 risk view，而是生成可解释的 policy-word edits，用于学习哪些纯值签名漂移应由混合词 counterterm 吸收；
- classifier 只消费 `Renorm(Signature(dV))`，从而减少“值证据 x 采样政策”的高阶缠结。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从普通 CDE/path encoder 改为 Value-Policy Word Signature Tower

PWSR 将不规则事件流 `(t_i, d_i, x_i, m_i)` 转成两个字母表：

1. **Value words `dV`**
   - 由观测值增量、变量 embedding 后的局部变化、measurement-normalized jump 构成。
   - 它承载真实状态路径的可分类几何。

2. **Policy words `dP`**
   - 只由采样坐标构成：`log(1 + delta_t)`、局部观测密度、变量可见性、panel/burst 指示、测量质量、kernel bandwidth preference proxy。
   - 它不输入分类器，只参与签名拆分与 counterterm 估计。

3. **截断签名塔**
   - 构造增强路径 `dX = [dV, dP]` 的 level-1 / level-2 签名。
   - 按词类型拆分：

```text
S_vv: 只含 value letters 的纯值词
S_pp: 只含 policy letters 的纯策略词
S_vp, S_pv: 同时含 value 与 policy letters 的混合词
```

分类器不读取 `S_pp`、`S_vp` 或 `S_pv`。这些词只交给 `CountertermNet`，用来估计采样策略对纯值签名的污染：

```text
C_v = CountertermNet(S_vp, S_pv, S_pp, policy_summary)
S_v^ren = S_v - C_v
logits = Classifier(S_v^ren)
```

这里的 `C_v` 不是 protocol tax、不是 residual sink、不是 posterior factor、不是 syndrome，也不是 uncertainty。它是签名代数意义上的 **采样词反项**：只允许解释“纯值签名中可由 mixed value-policy words 诱发的那部分漂移”。

### 2.2 改 Loss：从一致性/对抗转向 Shuffle-Ideal Renormalization

总目标：

```text
L = L_cls
  + lambda_ct   * L_counterterm_target
  + lambda_shuf * L_shuffle_validity
  + lambda_tri  * L_triangular_sobriety
```

#### A. 分类损失 `L_cls`

事实观测使用重整化后的纯值签名：

```text
L_cls = CE(Classifier(S_v^ren), y)
```

#### B. Counterterm Target `L_counterterm_target`

当前反事实采样模块生成一组 **policy-word edits**，例如：

- `gap_dilation`：只拉伸某些时间段的采样 gap；
- `panel_unbatch`：把同步 panel 拆成异步采样词；
- `burst_thinning`：稀疏化报警后复测 burst；
- `quality_reweight`：改变测量质量或 smoothing bandwidth proxy。

对每个 edit，分别计算事实路径与 policy-edited 路径的签名。若 edit 主要改变采样坐标，纯值签名的漂移中可由混合词解释的部分应由 counterterm 吸收：

```text
Delta S_v = S_v(policy_edit) - S_v(factual)
Delta C_v = C_v(policy_edit) - C_v(factual)
L_counterterm_target = SmoothL1(Delta C_v, stopgrad(Delta S_v))
```

注意这不是多视图一致性：我们不要求 `logits(policy_edit) == logits(factual)`，也不要求 `S_v^ren` 完全相同；我们只训练 counterterm 去解释由 policy-word edit 诱发的纯签名漂移。

#### C. Shuffle Validity `L_shuffle_validity`

反项不能任意修改纯值签名，否则会变成普通黑箱 adapter。路径签名必须满足 shuffle identity。对 level-1 与 level-2 纯值签名加入：

```text
S_i S_j = S_{ij} + S_{ji}
L_shuffle_validity = || outer(S_1^ren, S_1^ren) - (S_2^ren + transpose(S_2^ren)) ||^2
```

这保证重整化后的对象仍像一条合理状态路径的签名，而不是任意 embedding。

#### D. Triangular Sobriety `L_triangular_sobriety`

Counterterm 必须是三角的：低阶混合词只能修正同阶或更高阶纯值词，不能删除所有状态证据。

```text
L_triangular_sobriety =
  relu(||C_v|| / ||S_v|| - r_max)^2
  + relu(r_min - ||S_v^ren|| / ||S_v||)^2
```

这不是 evidence budget 或 protocol tax；它只约束反项规模，防止模型通过“把纯值签名全部减掉”获得虚假鲁棒性。

### 2.3 改 Dataloader：返回 Policy-Word Edit Bank

新增 `PolicyWordSignatureCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `measurement_std` 或质量 proxy。
3. `policy_word_features`：由采样坐标构成的事件级 policy letters。
4. `policy_word_edit_bank`：一组只改变采样词、不直接改标签的 edits：
   - `gap_dilation`；
   - `panel_unbatch`；
   - `burst_thinning`；
   - `bandwidth_quality_shift`。
5. 每个 edit 对应的 `edited_event_time`、`edited_event_mask`、`edited_policy_features`。

关键区别：

- 不估计采样概率、hazard、density ratio 或 posterior factor。
- 不做对比学习、不做 logits 一致、不做 risk variance。
- 不把 policy id、mask pattern、calendar residual、syndrome、uncertainty 或 solver trace 输入分类器。
- 反事实干预只为 mixed-word counterterm 提供代数监督。

### 2.4 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `ValueWordLift + SignatureTower`。
- 现有 sampling branch 改为 `PolicyWordLift`，输出采样词增量 `dP`，不参与分类读出。
- 现有 counterfactual sampler 改为 `PolicyWordEditBank`，生成 policy-word edits 训练 counterterm。
- 推理阶段只需事实路径：
  - 计算 `S_v`、mixed words、policy words；
  - 用 `CountertermNet` 得到 `C_v`；
  - 分类 `S_v^ren = S_v - C_v`。
- 可解释输出包括：
  - mixed-word energy；
  - counterterm norm；
  - 哪些 `value x policy` 词对预测最危险；
  - shuffle validity score，判断重整化后是否仍是可信状态路径。

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


def event_delta_time(event_time: torch.Tensor) -> torch.Tensor:
    delta_t = torch.zeros_like(event_time)
    delta_t[:, 1:] = (event_time[:, 1:] - event_time[:, :-1]).clamp_min(0.0)
    if event_time.size(1) > 1:
        delta_t[:, 0] = delta_t[:, 1]
    return delta_t


class ValuePolicyWordLift(nn.Module):
    """Lift irregular events into value letters and policy letters."""

    def __init__(self, num_vars: int, value_dim: int, policy_dim: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, value_dim),
        )
        self.policy_proj = nn.Sequential(
            nn.Linear(num_vars + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, policy_dim),
        )
        self.num_vars = num_vars

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        meas_std = batch.get("measurement_std", torch.zeros_like(value))
        panel_id = batch.get("panel_id", torch.full_like(var_id, -1))

        delta_t = event_delta_time(time)
        value_jump = torch.zeros_like(value)
        same_var = torch.zeros_like(value)
        same_var[:, 1:] = (var_id[:, 1:] == var_id[:, :-1]).to(value.dtype)
        value_jump[:, 1:] = (value[:, 1:] - value[:, :-1]) * same_var[:, 1:]

        var_h = self.var_embed(var_id.clamp_min(0))
        value_x = torch.cat(
            [
                var_h,
                value.unsqueeze(-1),
                value_jump.unsqueeze(-1),
                torch.log1p(meas_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        d_value = self.value_proj(value_x)

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        local_density = 1.0 / (1.0 + delta_t)
        early = (time_norm <= 0.33).to(value.dtype)
        middle = ((time_norm > 0.33) & (time_norm <= 0.66)).to(value.dtype)
        late = (time_norm > 0.66).to(value.dtype)
        panel_seen = (panel_id >= 0).to(value.dtype)
        var_onehot = F.one_hot(var_id.clamp_min(0), self.num_vars).to(value.dtype)

        policy_x = torch.cat(
            [
                var_onehot,
                torch.log1p(delta_t).unsqueeze(-1),
                local_density.unsqueeze(-1),
                early.unsqueeze(-1),
                middle.unsqueeze(-1),
                late.unsqueeze(-1),
                panel_seen.unsqueeze(-1) + torch.log1p(meas_std).unsqueeze(-1),
            ],
            dim=-1,
        )
        d_policy = self.policy_proj(policy_x)

        return d_value * mask.unsqueeze(-1), d_policy * mask.unsqueeze(-1)


def truncated_signature_level2(increments: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a level-2 signature approximation for piecewise linear increments."""

    dx = increments * mask.unsqueeze(-1)
    level1 = dx.sum(dim=1)

    prefix = dx.cumsum(dim=1) - dx
    cross = torch.einsum("bni,bnj->bij", prefix, dx)
    local = 0.5 * torch.einsum("bni,bnj->bij", dx, dx)
    level2 = cross + local
    return level1, level2


class SignatureSplitter:
    """Split augmented signature into pure value, pure policy and mixed words."""

    def __init__(self, value_dim: int, policy_dim: int):
        self.value_dim = value_dim
        self.policy_dim = policy_dim

    def split(self, level1: torch.Tensor, level2: torch.Tensor) -> dict:
        v = self.value_dim
        value_l1 = level1[:, :v]
        policy_l1 = level1[:, v:]
        vv = level2[:, :v, :v]
        vp = level2[:, :v, v:]
        pv = level2[:, v:, :v]
        pp = level2[:, v:, v:]
        return {
            "value_l1": value_l1,
            "policy_l1": policy_l1,
            "vv": vv,
            "vp": vp,
            "pv": pv,
            "pp": pp,
        }


class PolicyWordCounterterm(nn.Module):
    """Estimate pure-value signature counterterms from mixed policy words."""

    def __init__(self, value_dim: int, policy_dim: int, hidden_dim: int):
        super().__init__()
        self.value_dim = value_dim
        input_dim = (
            policy_dim
            + value_dim * policy_dim
            + policy_dim * value_dim
            + policy_dim * policy_dim
        )
        output_dim = value_dim + value_dim * value_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, parts: dict) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            [
                parts["policy_l1"],
                parts["vp"].flatten(1),
                parts["pv"].flatten(1),
                parts["pp"].flatten(1),
            ],
            dim=-1,
        )
        raw = self.net(features)
        c_l1 = raw[:, : self.value_dim]
        c_l2 = raw[:, self.value_dim :].view(-1, self.value_dim, self.value_dim)
        return c_l1, c_l2


def shuffle_validity_loss(level1: torch.Tensor, level2: torch.Tensor) -> torch.Tensor:
    """Penalize violations of S_i S_j = S_ij + S_ji for pure value words."""

    lhs = torch.einsum("bi,bj->bij", level1, level1)
    rhs = level2 + level2.transpose(1, 2)
    return F.smooth_l1_loss(rhs, lhs.detach())


def flatten_pure_signature(level1: torch.Tensor, level2: torch.Tensor) -> torch.Tensor:
    return torch.cat([level1, level2.flatten(1)], dim=-1)


class PolicyWordSignatureRenormalizer(nn.Module):
    """Classify irregular time series with policy-word signature counterterms."""

    def __init__(
        self,
        num_vars: int,
        value_dim: int,
        policy_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.lift = ValuePolicyWordLift(num_vars, value_dim, policy_dim, hidden_dim)
        self.splitter = SignatureSplitter(value_dim, policy_dim)
        self.counterterm = PolicyWordCounterterm(value_dim, policy_dim, hidden_dim)
        pure_dim = value_dim + value_dim * value_dim
        self.classifier = nn.Sequential(
            nn.Linear(pure_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_signature(self, batch: dict) -> dict:
        d_value, d_policy = self.lift(batch)
        augmented = torch.cat([d_value, d_policy], dim=-1)
        level1, level2 = truncated_signature_level2(augmented, batch["event_mask"])
        parts = self.splitter.split(level1, level2)
        c_l1, c_l2 = self.counterterm(parts)

        ren_l1 = parts["value_l1"] - c_l1
        ren_l2 = parts["vv"] - c_l2
        raw_pure = flatten_pure_signature(parts["value_l1"], parts["vv"])
        ren_pure = flatten_pure_signature(ren_l1, ren_l2)
        logits = self.classifier(ren_pure)
        return {
            **parts,
            "counter_l1": c_l1,
            "counter_l2": c_l2,
            "ren_l1": ren_l1,
            "ren_l2": ren_l2,
            "raw_pure": raw_pure,
            "ren_pure": ren_pure,
            "logits": logits,
        }

    def forward(self, batch: dict) -> dict:
        return self.encode_signature(batch)

    def counterterm_target_loss(self, batch: dict, factual: dict) -> torch.Tensor:
        losses = []
        factual_counter = torch.cat(
            [factual["counter_l1"], factual["counter_l2"].flatten(1)],
            dim=-1,
        )
        factual_raw = factual["raw_pure"].detach()

        for edited_batch in batch["policy_word_edit_bank"]:
            edited = self.encode_signature(edited_batch)
            edited_counter = torch.cat(
                [edited["counter_l1"], edited["counter_l2"].flatten(1)],
                dim=-1,
            )
            delta_counter = edited_counter - factual_counter
            delta_raw = edited["raw_pure"].detach() - factual_raw
            losses.append(F.smooth_l1_loss(delta_counter, delta_raw))
        return torch.stack(losses).mean()

    def triangular_sobriety_loss(
        self,
        factual: dict,
        r_max: float = 0.45,
        r_min: float = 0.35,
    ) -> torch.Tensor:
        counter = torch.cat(
            [factual["counter_l1"], factual["counter_l2"].flatten(1)],
            dim=-1,
        )
        raw_norm = factual["raw_pure"].norm(dim=-1).clamp_min(1e-6)
        counter_ratio = counter.norm(dim=-1) / raw_norm
        retained_ratio = factual["ren_pure"].norm(dim=-1) / raw_norm
        return (
            F.relu(counter_ratio - r_max).pow(2)
            + F.relu(r_min - retained_ratio).pow(2)
        ).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_ct: float = 0.25,
        lambda_shuf: float = 0.10,
        lambda_tri: float = 0.05,
    ) -> dict:
        factual = self.forward(batch)
        labels = batch["labels"]
        cls_loss = F.cross_entropy(factual["logits"], labels)
        ct_loss = self.counterterm_target_loss(batch, factual)
        shuf_loss = shuffle_validity_loss(factual["ren_l1"], factual["ren_l2"])
        tri_loss = self.triangular_sobriety_loss(factual)
        total = cls_loss + lambda_ct * ct_loss + lambda_shuf * shuf_loss + lambda_tri * tri_loss
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "counterterm_loss": ct_loss.detach(),
            "shuffle_validity_loss": shuf_loss.detach(),
            "triangular_sobriety_loss": tri_loss.detach(),
            "mixed_word_energy": (
                factual["vp"].pow(2).mean() + factual["pv"].pow(2).mean()
            ).detach(),
        }


@torch.no_grad()
def build_policy_word_edit_bank(batch: dict) -> list[dict]:
    """Create policy-word edits for counterterm supervision.

    These edits are not consistency views. They only expose how sampling words
    perturb the pure value signature through mixed iterated integrals.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    meas_std = batch.get("measurement_std", torch.zeros_like(value))
    panel_id = batch.get("panel_id", torch.full_like(var_id, -1))
    bsz, num_events = value.shape
    device = value.device

    views = []

    def clone_with(time_new, mask_new, std_new, panel_new):
        out = dict(batch)
        out["event_value"] = value * mask_new
        out["event_time"] = time_new
        out["event_var_id"] = var_id
        out["event_mask"] = mask_new
        out["measurement_std"] = std_new
        out["panel_id"] = panel_new
        return out

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    # Edit 1: gap dilation in late window.
    late = (time_norm > 0.66).to(time.dtype)
    views.append(clone_with(time * (1.0 + 0.20 * late), mask, meas_std, panel_id))

    # Edit 2: burst thinning by removing alternating events.
    alternating = ((torch.arange(num_events, device=device)[None, :] % 2) == 0).to(mask.dtype)
    burst_mask = mask * alternating
    views.append(clone_with(time, burst_mask, meas_std, panel_id))

    # Edit 3: panel unbatch by dropping panel identifiers.
    no_panel = torch.full_like(panel_id, -1)
    views.append(clone_with(time, mask, meas_std, no_panel))

    # Edit 4: quality / bandwidth proxy shift.
    noisier = meas_std + 0.20 * value.detach().abs().mean(dim=1, keepdim=True)
    views.append(clone_with(time, mask, noisier, panel_id))

    return views
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `mixed-word burst shift`：训练环境报警后 dense burst，测试环境稀疏复查。
   - `panel-word shift`：训练环境同步 panel，测试环境拆成异步项目。
   - `gap-word shift`：测试设备在早期/晚期系统性拉伸采样 gap。
   - `quality-word shift`：测量质量或 smoothing bandwidth preference 改变，使 value-policy iterated integrals 分布漂移。

2. **对比方法**
   - 普通 Neural CDE / MVC-CDE。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - policy adversarial baseline。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、DM-DRR、OS-MQ、CETC、PGHT、Policy-Shadow、SCSC、CKCF、Observability-Witness、PIIES、PLSM、ST-FDN、RKHS Cubature、Measurement-Action Bisimulation。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - mixed-word leakage：只用 `S_vp/S_pv` 预测标签的能力。
   - counterterm reliance：`||C_v|| / ||S_v||`。
   - shuffle violation rate：重整化签名是否仍满足路径签名代数约束。
   - policy-word attribution：哪些 value-policy word 对错误预测贡献最大。

4. **消融实验**
   - 去掉 counterterm，直接用纯值签名 `S_v` 分类。
   - 去掉 `L_counterterm_target`，检查 counterterm 是否退化为普通 adapter。
   - 去掉 `L_shuffle_validity`，检查重整化签名是否不再像真实路径。
   - 将 policy-word edits 替换为随机 mask，验证收益来自采样词结构而非普通增强。
   - 只使用 level-1 签名，验证 value-policy mixed iterated integrals 的必要性。

## 5. 预期创新性

1. **从路径平滑转向签名代数重整化**：吸收 MVC-CDE 对 control path 几何的启发，但不依赖多视图平滑、attention 或 solver trace，而是在 signature words 层面切断采样几何混合。
2. **从采样去偏转向 mixed-word counterterm**：不估计采样概率、不删除采样信息、不做对抗或一致性；采样分支只学习混合词对纯值签名的代数反项。
3. **从 token/mask 级快捷方式转向 iterated-integral shortcut**：将 sampling-policy leakage 定位到 `value x policy` 与 `policy x value` 高阶词，能解释为什么模型即使不显式读取 mask 也会受采样策略影响。
4. **反事实干预只服务代数监督**：counterfactual sampler 不生成正样本、不做风险方差、不做平滑、不做 knockoff；它只产生 policy-word edits 来标定 counterterm。
5. **可解释性与有效性同时保留**：mixed-word energy、counterterm norm 和 shuffle violation 能直接诊断预测是否依赖采样策略与状态路径的缠结。

## 6. 一句话投稿卖点

**PWSR 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“路径签名中 value words 与 policy words 的混合迭代积分污染”，并通过 shuffle-algebra counterterm renormalization 从纯值签名中减去采样词诱导漂移，让分类器依赖经过代数重整化的状态路径几何，而不是依赖训练环境中特定的采样 gap、panel、burst 或质量词与观测值变化的高阶缠结。**
