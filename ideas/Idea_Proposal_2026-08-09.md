# Title: Do-RG Scale Fixed-Point Forecaster：面向采样策略偏移的反事实重整化尺度不动点分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取 `paper_daily.md`，其最新追加覆盖 PULSE 与 TCF 等近期机制。
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
  - `ideas/Idea_Proposal_2026-08-05.md`
  - `ideas/Idea_Proposal_2026-08-06.md`
  - `ideas/Idea_Proposal_2026-08-08.md`
- 已读取自动化记忆 `MEMORIES.md` 及其中未完全落盘的历史提案摘要：
  - `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`
  - `idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`
  - `idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-05.md`、`idea_2026-08-06.md`、`idea_2026-08-07.md`、`idea_2026-08-08.md`

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮永久避开以下核心机制作为主创新：

1. learnable reference points / adaptive time encoding。
2. temporal consistency、inter-variable consistency、跨采样视图对比学习。
3. frequency-guided observation encoder 或频域掩码修复。
4. missingness pattern encoder 直接进入分类器。
5. prototype-constrained classifier / policy-aware prototypes。
6. 简单 environment adversarial / policy adversarial classifier。
7. 连续时间危险率 point-process scorer、采样 score 零空间、hazard-driven resampling、do-risk variance。
8. 生理流算子与采样算子交换子、value/policy graph commutator、policy residual sink。
9. additive evidence market、protocol tax、token-level evidence budget、边际证据审计。
10. 模型空间 posterior quotient、采样似然因子相除、干预积分分类。
11. reconstruction error cartography、ANOVA projection、VQ semantic/acquisition clauses、HSIC redaction。
12. policy-simplex randomized smoothing、certified policy radius、logit-normal/Dirichlet do-sampler。
13. 采样测度 density ratio、doubly robust target-measure correction、influence-bound。
14. optional-stopping martingale queries、standardized innovation、stopping recipe moment control。
15. soft excursion topology、censored persistence interval、censor envelope、fragmentation sobriety。
16. policy-gauge frame、horizontal transport、chart span supervision、vertical blindness。
17. policy-only negative film、shadow eraser/stencil、shadow nullification。
18. latent packet codeword、parity-check、syndrome locator、packet repair decoder。
19. conditional knockoff calendar、soft knockoff-FDR、swap symmetry。
20. observability witness、counterfactual observability probe、observability-routed classification。
21. subjective-logic / Dirichlet evidential shield、policy-induced vacuity、class-wise evidence discount。
22. observation-set policy lattice、meet/join masks、monotone/submodular margin、shortcut curvature。
23. solver-trace front-door、NFE/roughness trace mediator、reference trace bank。
24. measurement-action bisimulation、policy-word signature renormalization、thermodynamic free-energy、Sklar copula stripping、triage queue debt、Sinkhorn detail canonicalization、MDL episode transducer、causal sheaf gluing、trigger hysteresis、control barrier certificates、regret escrow、principal-stratum status compiler。
25. counterfactual conformal risk sleeves、counterfactual sampling instruments、Borda / majority rank tribunal、Krylov policy subspace、determinantal / Nystrom volume basis、tropical support routes。
26. fixed clinical viva question bank、pathology sequent proof bank、disease-progress poset clock、pathology feasible hull、IRT latent trait / DIF firewall。

本提案选择新的正交切入点：**不再把采样政策偏移表述为证明资格、测验表单、图结构、校准集合、纠错信道、拓扑审查、后验商、信息格或数值求解中介，而是把不同医院/设备的采样策略看成对同一潜在病程的不同“观测分辨率粗粒化”。模型学习一条可微重整化群（RG）尺度流：分类头只读取从事实观测尺度流向 policy-neutral 固定点后的病理尺度表示；采样分支只解释尺度流中的 irrelevant policy operators，不能直接产生类别证据。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 近期最值得结合当前“采样解耦/反事实干预”框架的两个信号是 **PULSE** 与 **TCF**。

第一，**PULSE** 把 ICU time-series classification 放在 HiRID / MIMIC-IV / eICU 等跨中心设置下比较，揭示真实部署中的 shift 往往不是单点 mask dropout，而是一整套观测分辨率的改变：

- 某中心常规高频测量，另一中心只保留固定查房窗口；
- 某中心把化验 panel 打包同步记录，另一中心把同样项目拆成异步事件；
- 某中心报警后局部密集复测，另一中心受资源约束只做稀疏跟踪；
- 某中心记录 future events 更完整，另一中心存在 value-pending 或延迟入库。

这些变化像“显微镜倍率”变化：同一病程在一个中心被写成高分辨率细事件流，在另一个中心被写成粗粒度摘要流。若分类器直接依赖某个分辨率下才可见的短 burst、panel 同窗或记录密度，它跨中心后就会崩溃。

第二，**TCF** 的 Pathology-Focused Binning、Dual-Calendar RoPE 与 Time-Conditioned Foreseeing 提醒我们：EHR 的时间和数值不应被当成普通 token；模型要理解“在未来某个时间条件下，会出现什么病理事件”。但 TCF 式 future-event likelihood 仍可能混合 patient state 与 care process：未来有没有某项化验，既可能表示病程进展，也可能只是医院的观察流程。

本轮提出 **Do-RG Scale Fixed-Point Forecaster (DRG-SFF)**：

> 将采样政策视为观测粗粒化算子 `C_s`。不同政策对应不同尺度 `s`：高频报警复测是细尺度，固定查房/变量预算是粗尺度，panel batching 是变量-时间联合粗粒化。真正可迁移的病理信号应当在沿尺度流粗粒化后进入一个稳定的 policy-neutral fixed point；训练环境特有的采样细节则应表现为沿尺度流快速衰减的 irrelevant operators。

直觉上，DRG-SFF 不要求每个反事实采样视图的 logits 一样，也不把采样模式当作证据、税费、证明、测验项或校准集合。它只学习一个尺度动力学：

```text
fine event stream --C_s--> coarse event stream
representation h_s --beta(h_s, s)--> h_{s + ds}
fixed-point signature h_* = RG-flow(h_observed)
logits = Classifier(h_*)
```

这样可以解决采样偏移的三个痛点：

1. **跨中心采样密度变化**：不同中心只是从不同初始尺度进入同一 RG 流，分类器读取固定点而非事实分辨率。
2. **panel / burst / pending artifacts**：这些高分辨率观测细节如果不能在粗粒化后保留病理预测力，就会作为 irrelevant operator 衰减。
3. **TCF future-event 混合问题**：DRG-SFF 不直接预测“未来医院会记录什么”，而是在多个尺度下预测 pathology-focused bins，并要求病理 foreseeing 随尺度流保持可解释的 fixed-point 方向。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从单尺度 event encoder 改为 Scale-Conditioned RG Encoder

DRG-SFF 可以包裹当前任意 irregular event encoder，但推荐前端改成三层结构。

#### A. Pathology Scale Atomizer

吸收 TCF 的 Pathology-Focused Binning，将每个事件 `(value, time, variable)` 编译成病理尺度 atom：

```text
a_i = Embed(variable_i, pathology_bin_i, local_delta_t_i, value_i)
```

这里的 bin 是变量特异的病理区间，不是 fixed viva question、IRT item、sequent literal 或 feasible facet。它只是构造尺度流的局部材料。

#### B. Counterfactual Coarse-Graining Ladder

dataloader 生成一组粗粒化尺度 `s = 0, 1, ..., L`：

- `s=0`：事实观测流；
- `routine_round`：将时间吸附到查房窗口；
- `burst_pool`：将报警后短窗口密集复测压成局部均值/极值摘要；
- `panel_pack`：将近同步变量 panel 变成一个 coarse super-event；
- `variable_budget`：每个变量或变量组只保留有限代表事件；
- `pending_collapse`：保留“已施测”但将尚未返回的 value 合并为粗粒度 unknown atom。

注意：这些不是对比学习正样本，不用于 logits consistency，也不构造 meet/join 信息格。它们只是给 RG encoder 一条从细到粗的尺度轨迹。

#### C. RG Beta Flow

对每个尺度得到表示 `h_s`，并学习尺度 beta function：

```text
beta_s = B_phi(h_s, scale_descriptor_s)
h_{s+1} ~= h_s + beta_s
```

同时把 `beta_s` 分解成两类 operator coefficients：

```text
r_s = relevant pathology coefficients
u_s = irrelevant policy-resolution coefficients
```

这里不是 state/policy 双图，也不是 residual sink。`r_s` 和 `u_s` 是重整化语义中的尺度算子系数：`r_s` 表示跨粗粒化仍保留的病理方向，`u_s` 表示只在特定采样分辨率下活跃、沿尺度流应衰减的观测细节。

分类头只读 fixed-point signature：

```text
h_* = h_L + FixedPointExtrapolator({beta_s}_{s=0}^{L-1})
logits = Classifier(h_*)
```

### 2.2 改 Loss：从一致性/证明/校准转向 RG Scale Laws

总目标：

```text
L = L_cls
  + lambda_beta * L_beta_prediction
  + lambda_rg   * L_semigroup_closure
  + lambda_ir   * L_irrelevant_decay
  + lambda_fore * L_scale_foreseeing
  + lambda_obs  * L_observation_process_sidecar
```

#### A. Fixed-Point Classification `L_cls`

事实样本经过完整尺度流后，分类器只读取 `h_*`：

```text
L_cls = CE(Classifier(h_*), y)
```

这与普通 multi-view ensemble 不同：推理不是平均多个采样政策视图，而是把当前事实观测送入学到的 RG flow，外推到 policy-neutral fixed point。

#### B. Beta Prediction `L_beta_prediction`

相邻尺度表示的真实位移由 encoder 直接观测：

```text
delta_s = stopgrad(h_{s+1}) - h_s
L_beta_prediction = SmoothL1(beta_s, delta_s)
```

这让 beta function 学会“粗粒化一步会怎样改变表示”。该项不要求 `h_s == h_{s+1}`，也不要求 logits 一致；它拟合的是尺度动力学。

#### C. Semigroup Closure `L_semigroup_closure`

粗粒化应近似满足半群律：先粗粒化一步再一步，和一次粗粒化两步应由同一个 RG 生成器解释。

```text
h_{s+2}^{two_step} = h_s + beta_s + beta_{s+1}
h_{s+2}^{direct}   = DirectCoarse(h_s, scale_jump=2)
L_semigroup_closure = ||h_{s+2}^{two_step} - h_{s+2}^{direct}||_2^2
```

这不是 commutator surgery：没有生理流/采样算子的交换子，也没有 policy residual sink。它只约束粗粒化尺度流本身像一个可复用的重整化算子，而不是每种医院策略都学一套私有变换。

#### D. Irrelevant Operator Decay `L_irrelevant_decay`

采样分辨率 artifacts 应在粗粒化后衰减。对 irrelevant coefficients `u_s` 施加指数衰减契约：

```text
L_irrelevant_decay =
  mean_s relu(||u_{s+1}||_2 - rho * ||u_s||_2)^2, 0 < rho < 1
```

这不是 evidential vacuity、conformal set 或 hull radius。模型不把不可靠部分转成不确定性集合，而是要求它们在 RG 尺度流中具有“短程/无关”行为。

#### E. Scale-Conditioned Pathology Foreseeing `L_scale_foreseeing`

吸收 TCF 的 future-time conditioned idea，但把“未来日历时间”改成“尺度固定点下的病理查询”：

```text
p_hat(bin, var | h_*, query_time, query_scale) = ForeseeHead(h_*, q_t, q_s)
```

训练目标只预测 pathology-focused bin，不预测 future observation administration：

```text
L_scale_foreseeing = CE(p_hat_pathology, target_pathology_bin)
```

这与 DCPD 的 progress-conditioned foreseeing 不同：DRG-SFF 不学习病程偏序时钟、precedence matrix 或 order-ideal；它学习的是同一时间条件在不同观测分辨率下应收敛到的病理 fixed-point forecast。

#### F. Observation-Process Sidecar `L_observation_process_sidecar`

采样分支保留为 sidecar，只预测粗粒化后会丢失哪些 observation-process 统计：

```text
obs_hat_s = Sidecar(u_s)
L_obs = SmoothL1(obs_hat_s, [event_count_drop, panel_drop, pending_rate, burst_pool_rate])
```

`u_s` 不进入分类头；它只解释哪些观测细节被当成 irrelevant operators 衰减掉。这样保留部署诊断能力，而不把采样细节作为类别证据。

### 2.3 改 Dataloader：返回 RG Scale Ladder，而不是 policy views

新增 `RGLadderCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `pathology_bin_id` 或可微 `bin_prob`。
3. `rg_scale_ladder`：
   - `factual_fine`；
   - `routine_round`；
   - `burst_pool`；
   - `panel_pack`；
   - `variable_budget`；
   - `pending_collapse`。
4. `scale_descriptor_ladder`：
   - 当前尺度的平均 gap、事件数、变量覆盖、panel 率、burst 率、pending 率；
   - 只用于 beta function 与 sidecar，不进入分类头。
5. `direct_jump_ladder`：
   - 用于 semigroup closure，构造 `s -> s+2` 的直接粗粒化结果。
6. `scale_foreseeing_queries`：
   - TCF 风格的 future time query；
   - 额外带 query scale，用于判断 pathology forecast 是否收敛到 fixed-point 语义。

关键区别：

- 不生成 consistency pair。
- 不做 proof / sequent / poset / IRT / feasible hull。
- 不估计 hazard、density ratio、posterior quotient、conformal sleeve 或 policy jury。
- 不构造 topological capsules、gauge frame、syndrome repair、knockoff calendar 或 lattice meet/join。
- 不把 observation-process sidecar 输入分类器。
- 反事实采样只用来形成粗粒化尺度梯子，训练 RG flow 的半群律与 irrelevant operator 衰减。

### 2.4 与 PULSE / TCF 的结合方式

- **来自 PULSE**：HiRID / MIMIC-IV / eICU 的差异被解释为不同观测分辨率初始点。DRG-SFF 可报告每个中心的 `initial_scale_descriptor`、fixed-point drift 和 irrelevant operator energy，解释哪个中心最依赖高分辨率流程。
- **来自 TCF**：保留 pathology-focused bins 和 future-time query，但不把未来事件出现本身当作类别证据；future observation process 被 sidecar 解释，分类只读 fixed-point pathology representation。
- **与采样解耦/反事实干预框架结合**：value process 产生病理 atom；sampling process 产生 coarse-graining ladder 与 scale descriptors；counterfactual intervention 生成可解释尺度变换；classifier 读取 RG fixed point。

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


class PathologyScaleAtomizer(nn.Module):
    """Lift irregular events into pathology-bin atoms for RG scale flow."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.5, 2.5, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))
        self.atom_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_bins + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * mask.unsqueeze(-1)

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        atom_x = torch.cat(
            [
                self.var_embed(var_id),
                bin_prob,
                value.unsqueeze(-1),
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
            ],
            dim=-1,
        )
        atom_h = self.atom_proj(atom_x) * mask.unsqueeze(-1)
        return {
            "atom_h": atom_h,
            "bin_prob": bin_prob,
            "pathology_bin": bin_prob.argmax(dim=-1),
            "event_mask": mask,
        }


class ScalePoolEncoder(nn.Module):
    """Encode one coarse-grained scale of an event stream."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.local = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.pool = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, atom_h: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        active_atom = atom_h * visibility.unsqueeze(-1)
        ctx, _ = self.local(active_atom)
        pooled = masked_mean(ctx, visibility, dim=1)
        return self.pool(pooled)


class RGBetaFunction(nn.Module):
    """Predict RG beta flow and relevant/irrelevant scale coefficients."""

    def __init__(self, hidden_dim: int, scale_dim: int, operator_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(scale_dim, hidden_dim)
        self.beta = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.relevant = nn.Linear(hidden_dim, operator_dim)
        self.irrelevant = nn.Linear(hidden_dim, operator_dim)

    def forward(self, h_s: torch.Tensor, scale_descriptor: torch.Tensor) -> dict:
        scale_h = self.scale_proj(scale_descriptor)
        joint = torch.cat([h_s, scale_h], dim=-1)
        beta = self.beta(joint)
        return {
            "beta": beta,
            "relevant_coeff": self.relevant(beta),
            "irrelevant_coeff": self.irrelevant(beta),
        }


class FixedPointExtrapolator(nn.Module):
    """Map the final coarse representation and beta history to an RG fixed-point signature."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_last: torch.Tensor, beta_stack: torch.Tensor) -> torch.Tensor:
        beta_mean = beta_stack.mean(dim=1)
        beta_tail = beta_stack[:, -1]
        correction = self.refine(torch.cat([beta_mean, beta_tail], dim=-1))
        gate = self.gate(torch.cat([h_last, beta_tail], dim=-1))
        return h_last + gate * correction


class ScaleForeseeHead(nn.Module):
    """Predict pathology-focused bins from fixed-point state under time/scale queries."""

    def __init__(self, hidden_dim: int, num_vars: int, num_bins: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.query = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.var_head = nn.Linear(hidden_dim, num_vars)
        self.bin_head = nn.Linear(hidden_dim, num_bins)

    def forward(self, h_star: torch.Tensor, query_time: torch.Tensor, query_scale: torch.Tensor) -> dict:
        # query_time/query_scale: [B, Q]
        h = h_star[:, None].expand(-1, query_time.size(1), -1)
        q = torch.cat([h, query_time.unsqueeze(-1), query_scale.unsqueeze(-1)], dim=-1)
        qh = self.query(q)
        return {"var_logits": self.var_head(qh), "bin_logits": self.bin_head(qh)}


class ObservationProcessSidecar(nn.Module):
    """Decode observation-process statistics from irrelevant RG operators only."""

    def __init__(self, operator_dim: int, hidden_dim: int, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(operator_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, irrelevant_coeff: torch.Tensor) -> torch.Tensor:
        return self.net(irrelevant_coeff)


class DoRGScaleFixedPointForecaster(nn.Module):
    """Sampling-policy robust classifier via RG scale fixed-point representation."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        hidden_dim: int,
        scale_dim: int,
        operator_dim: int,
        obs_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.atomizer = PathologyScaleAtomizer(num_vars, num_bins, hidden_dim)
        self.scale_encoder = ScalePoolEncoder(hidden_dim)
        self.beta = RGBetaFunction(hidden_dim, scale_dim, operator_dim)
        self.fixed_point = FixedPointExtrapolator(hidden_dim)
        self.foresee = ScaleForeseeHead(hidden_dim, num_vars, num_bins)
        self.sidecar = ObservationProcessSidecar(operator_dim, hidden_dim, obs_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_bins = num_bins
        self.num_vars = num_vars

    def encode_ladder(self, batch: dict) -> dict:
        atom = self.atomizer(batch)
        visibility_ladder = batch["rg_visibility_ladder"]  # [B, S, N]
        scale_desc = batch["scale_descriptor_ladder"]      # [B, S, D]

        h_scales = []
        for scale_idx in range(visibility_ladder.size(1)):
            h_scales.append(self.scale_encoder(atom["atom_h"], visibility_ladder[:, scale_idx]))
        h_scales = torch.stack(h_scales, dim=1)  # [B, S, H]

        beta_list = []
        rel_list = []
        irrel_list = []
        sidecar_list = []
        for scale_idx in range(h_scales.size(1) - 1):
            out = self.beta(h_scales[:, scale_idx], scale_desc[:, scale_idx])
            beta_list.append(out["beta"])
            rel_list.append(out["relevant_coeff"])
            irrel_list.append(out["irrelevant_coeff"])
            sidecar_list.append(self.sidecar(out["irrelevant_coeff"]))

        beta_stack = torch.stack(beta_list, dim=1)
        relevant = torch.stack(rel_list, dim=1)
        irrelevant = torch.stack(irrel_list, dim=1)
        sidecar_pred = torch.stack(sidecar_list, dim=1)
        h_star = self.fixed_point(h_scales[:, -1], beta_stack)
        logits = self.classifier(h_star)

        return {
            **atom,
            "h_scales": h_scales,
            "beta_stack": beta_stack,
            "relevant_coeff": relevant,
            "irrelevant_coeff": irrelevant,
            "sidecar_pred": sidecar_pred,
            "h_star": h_star,
            "logits": logits,
        }

    def beta_prediction_loss(self, out: dict) -> torch.Tensor:
        h_scales = out["h_scales"]
        target_delta = h_scales[:, 1:].detach() - h_scales[:, :-1]
        return F.smooth_l1_loss(out["beta_stack"], target_delta)

    def irrelevant_decay_loss(self, out: dict, rho: float = 0.65) -> torch.Tensor:
        coeff = out["irrelevant_coeff"]
        if coeff.size(1) < 2:
            return torch.zeros((), device=coeff.device)
        norm_now = coeff[:, :-1].norm(dim=-1)
        norm_next = coeff[:, 1:].norm(dim=-1)
        return F.relu(norm_next - rho * norm_now).pow(2).mean()

    def semigroup_closure_loss(self, batch: dict, out: dict) -> torch.Tensor:
        if "direct_jump_visibility_ladder" not in batch:
            return torch.zeros((), device=out["logits"].device)

        atom_h = out["atom_h"]
        direct_vis = batch["direct_jump_visibility_ladder"]  # [B, S-2, N]
        if direct_vis.size(1) == 0:
            return torch.zeros((), device=out["logits"].device)

        losses = []
        h_scales = out["h_scales"]
        beta_stack = out["beta_stack"]
        for idx in range(min(direct_vis.size(1), h_scales.size(1) - 2)):
            direct = self.scale_encoder(atom_h, direct_vis[:, idx])
            two_step = h_scales[:, idx] + beta_stack[:, idx] + beta_stack[:, idx + 1]
            losses.append(F.smooth_l1_loss(two_step, direct.detach()))
        return torch.stack(losses).mean()

    def scale_foreseeing_loss(self, batch: dict, out: dict) -> torch.Tensor:
        if "query_time" not in batch:
            return torch.zeros((), device=out["logits"].device)
        pred = self.foresee(out["h_star"], batch["query_time"], batch["query_scale"])
        target_var = batch["query_target_var"].clamp(0, self.num_vars - 1)
        target_bin = batch["query_target_bin"].clamp(0, self.num_bins - 1)
        var_loss = F.cross_entropy(pred["var_logits"].flatten(0, 1), target_var.flatten())
        bin_loss = F.cross_entropy(pred["bin_logits"].flatten(0, 1), target_bin.flatten())
        return var_loss + bin_loss

    def observation_sidecar_loss(self, batch: dict, out: dict) -> torch.Tensor:
        if "observation_process_target" not in batch:
            return torch.zeros((), device=out["logits"].device)
        return F.smooth_l1_loss(out["sidecar_pred"], batch["observation_process_target"].detach())

    def training_loss(
        self,
        batch: dict,
        lambda_beta: float = 0.25,
        lambda_rg: float = 0.20,
        lambda_ir: float = 0.12,
        lambda_fore: float = 0.20,
        lambda_obs: float = 0.08,
    ) -> dict:
        labels = batch["labels"]
        out = self.encode_ladder(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)
        beta_loss = self.beta_prediction_loss(out)
        rg_loss = self.semigroup_closure_loss(batch, out)
        ir_loss = self.irrelevant_decay_loss(out)
        fore_loss = self.scale_foreseeing_loss(batch, out)
        obs_loss = self.observation_sidecar_loss(batch, out)

        total = (
            cls_loss
            + lambda_beta * beta_loss
            + lambda_rg * rg_loss
            + lambda_ir * ir_loss
            + lambda_fore * fore_loss
            + lambda_obs * obs_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "beta_prediction_loss": beta_loss.detach(),
            "semigroup_closure_loss": rg_loss.detach(),
            "irrelevant_decay_loss": ir_loss.detach(),
            "scale_foreseeing_loss": fore_loss.detach(),
            "observation_sidecar_loss": obs_loss.detach(),
            "mean_irrelevant_energy": out["irrelevant_coeff"].norm(dim=-1).mean().detach(),
        }


@torch.no_grad()
def build_rg_scale_ladder(batch: dict, num_scales: int = 5) -> dict:
    """Create coarse-graining ladders for RG training.

    The ladder describes resolution changes. It is not a set of contrastive
    positives and is not used for logits consistency.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon
    event_idx = torch.arange(num_events, device=device)[None].expand(bsz, -1)

    visibility = []

    # Scale 0: factual fine resolution.
    visibility.append(mask)

    # Scale 1: routine-round thinning.
    visibility.append(mask * ((event_idx % 2) == 0).to(mask.dtype))

    # Scale 2: burst pooling proxy: keep representative events in dense late windows.
    late = (time_norm > 0.66).to(mask.dtype)
    representative = ((event_idx % 3) == 0).to(mask.dtype)
    burst_pool = torch.where(late > 0, representative, torch.ones_like(mask)) * mask
    visibility.append(burst_pool)

    # Scale 3: variable-budget coarse-graining.
    budget = torch.zeros_like(mask)
    for var in torch.unique(var_id[mask > 0]).tolist():
        hit = ((var_id == int(var)) & (mask > 0)).to(mask.dtype)
        rank = hit.cumsum(dim=1)
        budget = torch.maximum(budget, (rank <= 2).to(mask.dtype) * hit)
    visibility.append(budget)

    # Scale 4: pending/value-collapse proxy, retaining only coarse temporal anchors.
    anchor = (((time_norm <= 0.15) | (time_norm >= 0.85)).to(mask.dtype) + ((event_idx % 4) == 0).to(mask.dtype))
    visibility.append((anchor > 0).to(mask.dtype) * mask)

    visibility = torch.stack(visibility[:num_scales], dim=1)

    # Scale descriptors: observation-process coordinates for beta/sidecar only.
    desc = []
    obs_targets = []
    for scale_idx in range(visibility.size(1)):
        vis = visibility[:, scale_idx]
        kept = (vis * mask).sum(dim=1, keepdim=True)
        total = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        keep_rate = kept / total

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        mean_gap = masked_mean(torch.log1p(delta_t), vis, dim=1).unsqueeze(-1)
        early_rate = masked_mean((time_norm <= 0.33).to(mask.dtype), vis, dim=1).unsqueeze(-1)
        late_rate = masked_mean((time_norm > 0.66).to(mask.dtype), vis, dim=1).unsqueeze(-1)
        var_coverage = []
        for var in range(int(var_id.max().item()) + 1):
            var_coverage.append(((var_id == var).to(mask.dtype) * vis).amax(dim=1, keepdim=True))
        if var_coverage:
            var_cov = torch.cat(var_coverage, dim=1).mean(dim=1, keepdim=True)
        else:
            var_cov = torch.zeros_like(keep_rate)
        burst_rate = masked_mean((delta_t <= delta_t.mean(dim=1, keepdim=True)).to(mask.dtype), vis, dim=1).unsqueeze(-1)
        pending_rate = batch.get("value_pending", torch.zeros_like(mask))
        pending_rate = masked_mean(pending_rate, vis, dim=1).unsqueeze(-1)
        desc.append(torch.cat([keep_rate, mean_gap, early_rate, late_rate, var_cov, burst_rate, pending_rate], dim=-1))

        if scale_idx < visibility.size(1) - 1:
            next_vis = visibility[:, scale_idx + 1]
            drop = (vis - next_vis).clamp_min(0.0)
            drop_rate = drop.sum(dim=1, keepdim=True) / total
            panel_drop = drop_rate  # replace with panel-aware stat if panel ids are available
            obs_targets.append(torch.cat([drop_rate, panel_drop, burst_rate, pending_rate], dim=-1))

    out = dict(batch)
    out["rg_visibility_ladder"] = visibility
    out["scale_descriptor_ladder"] = torch.stack(desc, dim=1)
    out["observation_process_target"] = torch.stack(obs_targets, dim=1)

    # Direct jumps for semigroup closure: s -> s+2.
    direct = []
    for scale_idx in range(visibility.size(1) - 2):
        direct.append(torch.minimum(visibility[:, scale_idx], visibility[:, scale_idx + 2]))
    out["direct_jump_visibility_ladder"] = torch.stack(direct, dim=1) if direct else visibility[:, :0]
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cross-center resolution shift`：借鉴 PULSE，在 HiRID / MIMIC-IV / eICU 风格的高频、低频、panel、pending 记录制度之间迁移。
   - `burst-to-routine shift`：训练环境有报警后密集复测，测试环境压缩为固定查房粗粒度记录。
   - `panel-to-asynchronous shift`：训练中多个变量同步 panel，测试中拆成异步事件，观察 irrelevant operator 是否快速衰减。
   - `TCF future-recording shift`：训练中心未来事件记录完整，测试中心 future observation process 延迟或 pending；DRG-SFF 只要求 pathology forecast 稳定，不要求 observation process 稳定。

2. **对比方法**
   - 普通 TCF / EHR foundation model。
   - PULSE 中 LightGBM、传统深度模型与 LLM prompt / hybrid baseline。
   - STAR-Set、VP-GNN、MTM、MVC-CDE、QuITE 等 irregular / asynchronous baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、D-IVSP、KPMA、DVNB、DSPP、DCPD、DFFH、DIPF 等。

3. **核心指标**
   - in-policy accuracy / AUPRC。
   - cross-center worst-policy accuracy。
   - fixed-point drift：不同中心初始尺度流向 `h_*` 后的距离。
   - irrelevant operator half-life：`||u_s||` 沿尺度流衰减到一半所需尺度步数。
   - pathology foreseeing stability：不同观测分辨率下 TCF-style pathology bin forecast 的收敛误差。
   - observation sidecar fidelity：`u_s` 是否能解释事件数下降、panel drop、pending rate、burst pooling，而分类头不读取它。
   - scale shortcut score：错误预测是否伴随高 irrelevant energy 或半群闭包失败。

4. **消融实验**
   - 去掉 `L_irrelevant_decay`，检查模型是否继续依赖高分辨率采样 artifacts。
   - 去掉 `L_semigroup_closure`，检查 RG flow 是否退化成每种 policy 一个私有变换。
   - 去掉 fixed-point extrapolator，直接用事实尺度 `h_0` 分类，验证跨中心退化。
   - 去掉 `L_scale_foreseeing`，验证 TCF pathology query 是否为固定点提供临床语义锚。
   - 让 sidecar 输出进入分类头作为反例，验证 observation-process 统计会带来采样捷径。
   - 将 RG ladder 替换为随机 mask ladder，验证收益来自有结构的粗粒化尺度流，而不是普通增强。

## 5. 预期创新性

1. **从采样偏移转向观测分辨率重整化**：首次把 sampling-policy shift 表述为同一病程在不同观测尺度下的粗粒化流问题。
2. **从多视图一致转向尺度 beta function**：不要求不同采样视图 logits 或 representation 一致，而是学习 `h_s -> h_{s+1}` 的 RG beta 动力学。
3. **从策略去偏转向 irrelevant operator 衰减**：采样 artifacts 不是被删除、征税、纠错、审查或变成 uncertainty，而是在尺度流中表现为应衰减的无关算子。
4. **从 TCF 日历 foreseeing 转向 fixed-point pathology foreseeing**：保留 TCF 的病理分箱与未来时间查询，但将 future observation process 放入 sidecar，分类只读 fixed-point pathology state。
5. **从 PULSE benchmark 转向机制解释**：跨 HiRID / MIMIC-IV / eICU 的差异不只报告性能退化，还能解释为初始分辨率、irrelevant half-life 和 fixed-point drift 的差异。
6. **与现有反事实框架低侵入兼容**：counterfactual sampler 只需生成 coarse-graining ladder；不需要 hazard、density ratio、proof audit、IRT form、conformal calibration、policy jury、knockoff 或图结构改造。

## 6. 一句话投稿卖点

**DRG-SFF 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“不同医院/设备对同一病程使用了不同观测分辨率”的问题，通过 Pathology Scale Atomizer、counterfactual coarse-graining ladder、RG beta function、irrelevant policy-operator decay 与 fixed-point pathology foreseeing，让分类器读取跨分辨率稳定的病理尺度不动点，而不是依赖训练中心特有的高频复测、panel batching、变量预算、value-pending 或 future-recording 流程。**
