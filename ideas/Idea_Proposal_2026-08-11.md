# Title: Do-Kaczmarz Clinical Tomography：面向采样策略偏移的反事实观测射线重建分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取近期论文记录：
  - `paper_daily.md`
  - `paper_daily_2026-08-02.md`
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
- 已读取自动化记忆 `MEMORIES.md` 以及其中保存的额外历史提案摘要：
  - `idea_2026-07-24.md`、`idea_2026-07-25.md`、`idea_2026-07-26.md`、`idea_2026-07-27.md`
  - `idea_2026-07-29.md`、`idea_2026-07-30.md`、`idea_2026-07-31.md`
  - `idea_2026-08-01.md`、`idea_2026-08-04.md`、`idea_2026-08-05.md`、`idea_2026-08-06.md`
  - `idea_2026-08-07.md`、`idea_2026-08-08.md`、`idea_2026-08-09.md`、`idea_2026-08-10.md`

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下机制作为主创新：

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
12. policy-simplex randomized smoothing、certified policy radius、logit-normal / Dirichlet do-sampler。
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
27. observation-resolution RG coarse-graining ladder、scale beta flow、irrelevant operator decay、fixed-point pathology foreseeing。
28. event-time / record-time bitemporal curtain、anti-retrocausal margin、latency sidecar、record-availability foreseeing。

本提案选择新的正交切入点：**不把采样策略建成概率、危险率、日历负控、纠错信道、测验表单、证明系统、尺度流或不确定性护套，而是把不同医院/设备的采样政策看成同一潜在病程的不同“观测射线设计矩阵”。模型先用 TCF 式病理分箱把每个事件变成对潜在临床状态的一条线性/仿射测量，再用可微 Kaczmarz 断层重建恢复 policy-neutral pathology field；分类器只读取重建后的状态场，而不读取观测射线本身。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 最近最值得纳入的前沿机制是 **PULSE** 与 **Time-Conditioned Foreseeing (TCF)**。

**PULSE** 的跨 HiRID / MIMIC-IV / eICU 审计说明，真实 sampling-policy shift 很少只是随机 mask ratio 改变，而是不同医院用不同观测制度“扫描”同一个患者：

- 医院 A 高频测生命体征，医院 B 只在查房或报警后记录；
- 医院 A 把化验打包成 panel，医院 B 按项目异步返回；
- 医院 A 的 value-pending 和 batch-upload 延迟明显，医院 B 实时入库；
- 同一个 sepsis / mortality 终点，在不同中心里被不同变量、时间窗、复测密度所“投影”出来。

这些差异很像断层成像里的投影角度变化：不是身体内部结构换了，而是扫描角度、射线覆盖和探测器噪声换了。若分类器直接依赖“哪条射线被打过”“哪个角度覆盖更密”“哪组变量共线出现”，跨中心后自然会失败。

**TCF** 提供了另一个关键基础：EHR 事件不能只当连续数值 token，而应先转成具有临床语义的 pathology-focused bins，并带有 future-time conditioned query。问题在于，TCF 式 future event likelihood 仍可能混合 patient state 与 observation process：未来是否出现某个事件，既可能来自病程，也可能来自医院扫描制度。

因此，本轮提出 **Do-Kaczmarz Clinical Tomography (DKCT)**：

> 把每个观测事件 `(variable, time, pathology_bin, value)` 看成一条对潜在病理状态场 `theta` 的观测射线；采样政策只决定当前拿到了哪些射线、射线角度如何分布、噪声有多大。模型用可微 Kaczmarz / ART 迭代从这些射线重建 `theta`，分类头只读重建后的病理状态场。反事实采样模块不再制造一致性视图，而是制造观测设计矩阵的覆盖变化，用于训练重建器在射线覆盖偏移下仍保持可解性和低杠杆。

这个思路解决采样偏移的关键在于：

1. **把“是否被测”从类别证据降级为测量设计**
   某个变量被测不再直接支持类别，它只提供一行 measurement equation；如果该行高杠杆、低覆盖或与训练中心特有协议相关，它会影响重建不确定性和残差，而不是直接进入 logits。

2. **把跨中心差异变成可解释的设计矩阵偏移**
   PULSE 中的跨中心差异可表示为 `A_center` 的行覆盖、条件数、leverage 分布和变量/时间 query coverage 差异。模型可以报告“这个中心缺少哪些临床射线角度”，而不是只报告 AUROC 退化。

3. **保留 TCF 的病理语义，但不预测医院未来会记录什么**
   DKCT 使用 TCF 的 pathology-focused bins 作为投影观测值 `b_i`，并用 future-time pathology query 作为未观测虚拟射线；但它预测的是重建状态场在未来临床时间的 pathology response，而不是预测 observation administration。

4. **与采样解耦/反事实干预框架低侵入兼容**
   value process 产生病理射线响应；sampling process 产生观测设计矩阵与噪声/覆盖描述；counterfactual intervention 改变射线覆盖和角度；classifier 仅读取重建后的 `theta`。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从事件池化改为 Observation-Ray Tomography Encoder

DKCT 可以包裹当前任何 irregular event encoder，但推荐把前端改成三层。

#### A. Pathology Ray Atomizer

吸收 TCF 的 Pathology-Focused Binning，将每个事件编译为一条观测射线：

```text
event_i = (x_i, t_i, var_i, mask_i)
bin_i   = PathologyBin(var_i, x_i)
a_i     = RayGeometry(var_i, relative_time_i, delta_t_i, clinical_phase_i)
b_i     = RayResponse(bin_i, value_i)
w_i     = RayReliability(measurement_std_i, value_pending_i, local_density_i)
```

其中：

- `a_i in R^D` 是测量设计矩阵的一行，表示这次观测投影了哪些潜在病理轴；
- `b_i` 是该射线的病理响应，可以是 ordinal bin embedding 或连续 residual；
- `w_i` 是该射线可靠度，只用于重建加权，不进入分类头。

这不是 IRT item、viva question、sequent literal 或 feasible facet。它不问“这是不是一道题/一条规则/一个半空间”，只把观测变成逆问题里的测量行。

#### B. Differentiable Kaczmarz Reconstructor

对潜在病理状态场 `theta`，每条射线给出近似方程：

```text
a_i^T theta ~= b_i
```

使用加权 Kaczmarz / Algebraic Reconstruction Technique 做可微迭代：

```text
residual_i = b_i - a_i^T theta
theta <- theta + step * w_i * residual_i * a_i / (||a_i||^2 + eps)
```

Kaczmarz 的好处是非常适合不规则观测：每个事件本来就是一行测量，事件数量、变量、时间都可变；新增或删除观测只改变行集合，不需要把序列补齐到规则网格。

#### C. Tomographic Classifier

分类器只读取重建后的 `theta_hat` 与少量重建质量摘要：

```text
logits = Classifier(theta_hat, reconstruction_residual_summary)
```

但注意：重建质量摘要只包括 value residual、条件数代理、leverage 分布的截断统计；不包含 policy id、center id、原始 mask pattern 或 observation count 直接特征。其作用是让模型知道“重建是否可信”，而不是让采样政策成为类别捷径。

### 2.2 改 Loss：从一致性/去偏/证明转向 Tomographic Identifiability Discipline

总目标：

```text
L = L_cls
  + lambda_ray  * L_ray_fit
  + lambda_norm * L_normal_equation
  + lambda_lev  * L_leverage_capping
  + lambda_cov  * L_counterfactual_angle_coverage
  + lambda_fore * L_virtual_ray_foreseeing
```

#### A. Fixed Reconstruction Classification `L_cls`

事实观测下，模型先做断层重建，再分类：

```text
theta_hat = Kaczmarz(A_factual, b_factual, w_factual)
L_cls = CE(Classifier(theta_hat), y)
```

与普通 set encoder 的差别是：模型没有自由把 token visibility 学成任意分类特征；它必须先解释每条观测射线如何约束同一个潜在病理状态场。

#### B. Ray Fit `L_ray_fit`

重建出的状态必须能解释观测值：

```text
L_ray_fit = mean_i w_i * (a_i^T theta_hat - b_i)^2
```

这保证 `theta_hat` 不是普通 embedding，而是可被观测方程审计的 reconstruction。

#### C. Normal-Equation Residual `L_normal_equation`

加权最小二乘解应满足：

```text
A^T W (A theta - b) ~= 0
```

训练时加入：

```text
L_normal_equation = || A^T W (A theta_hat - b) ||_2^2
```

该项是 DKCT 的核心稳定器：不同采样政策会改变 `A` 的行分布，但若 `theta_hat` 仍满足同一个逆问题的法方程，它更可能代表状态场而不是采样日历。

#### D. Leverage Capping `L_leverage_capping`

采样政策 shortcut 往往表现为少数高杠杆射线支配分类，例如某中心只在极高风险时测 lactate，导致 `lactate@late` 这行的 leverage 极高。

DKCT 计算近似 leverage：

```text
H_i = a_i^T (A^T W A + ridge I)^(-1) a_i
```

并限制单行过度支配：

```text
L_leverage_capping = mean_i relu(H_i - h_max)^2
```

这不是 evidence tax，也不是 conformal uncertainty。它直接作用在观测设计矩阵的几何杠杆上：高杠杆观测可以用于重建，但不能成为不可审计的单射线诊断。

#### E. Counterfactual Angle Coverage `L_counterfactual_angle_coverage`

反事实采样模块生成不同观测射线覆盖：

- `routine_sparse_angle`：固定查房式低频射线；
- `alarm_dense_angle`：报警后局部密集射线；
- `panel_bundle_angle`：多变量 panel 共线射线；
- `pending_blind_angle`：射线出现但响应 `b_i` 缺失或低可靠；
- `cross_center_schema_angle`：模拟 PULSE 风格变量 schema / center coverage 差异。

DKCT 不要求这些视图的 logits 一致，也不要求 `theta_hat` 相同。它只要求每种设计矩阵的 Gram 覆盖不要塌缩到单一协议角度：

```text
G_r = A_r^T W_r A_r
coverage_r = logdet(G_r + ridge I)
L_counterfactual_angle_coverage = relu(c_min - coverage_r)^2
```

如果某 policy 下覆盖不足，模型应表现为重建质量下降，而不是让分类头从“覆盖不足本身”读标签。

#### F. Virtual-Ray Foreseeing `L_virtual_ray_foreseeing`

吸收 TCF 的 future-time conditioned idea，但改成“虚拟病理射线查询”：

```text
a_q = RayGeometry(var_q, future_time_q, clinical_phase_q)
b_q_hat = a_q^T theta_hat
```

训练目标只预测 pathology-focused bin / response：

```text
L_virtual_ray_foreseeing = CE(BinHead(b_q_hat), target_pathology_bin)
```

这与 TCF 最大区别是：DKCT 不预测未来医院会不会记录该事件；它只问“如果在该未来时间和变量方向打一条虚拟射线，潜在病理状态会给出什么响应”。

### 2.3 改 Dataloader：返回 Ray Design Bank，而不是 policy views

新增 `TomographicRayCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `pathology_bin_id` 或 `bin_prob`。
3. `ray_design`：
   - 每个事件对应一行 `a_i`；
   - 由变量、相对时间、delta-t、clinical phase、TCF-style pathology bin family 生成。
4. `ray_response`：
   - 每条射线的病理响应 `b_i`；
   - 可以是 ordinal bin 的连续嵌入或经过变量标准化的 value。
5. `ray_weight`：
   - 测量可靠度、value-pending、异方差噪声、重复观测稳定性。
6. `counterfactual_ray_bank`：
   - routine / alarm / panel / pending / cross-center schema 的设计矩阵和响应可见性。
7. `virtual_ray_queries`：
   - TCF 式 future-time query；
   - 只用于 pathology response foreseeing，不用于预测 observation process。

关键区别：

- 不生成 consistency pair。
- 不估计 hazard、density ratio、posterior quotient、DIF、RG beta 或 latency curtain。
- 不做 proof、viva、poset、feasible hull、conformal set、jury、knockoff、syndrome repair、topology capsule 或 gauge frame。
- 反事实采样只改变观测设计矩阵 `A` 的射线覆盖，用于训练逆问题可识别性、leverage 控制和 angle coverage。

### 2.4 与 PULSE / TCF 的结合方式

- **来自 PULSE**：HiRID / MIMIC-IV / eICU 被视为不同临床扫描器。DKCT 可以比较各中心的 `logdet(A^TWA)`、leverage tail、变量/时间 angle gap，从机制上解释跨中心退化来自哪些观测射线缺失或过度杠杆。
- **来自 TCF**：Pathology-Focused Binning 提供 `b_i` 的病理语义；future-time conditioned query 被改写为 virtual-ray pathology foreseeing，避免把未来 observation administration 当成可迁移病理。
- **与采样解耦/反事实干预框架结合**：value branch 产生射线响应，sampling branch 产生射线设计与可靠度，counterfactual sampler 产生不同设计矩阵，classifier 仅读取 Kaczmarz 重建后的状态场。

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


class PathologyRayAtomizer(nn.Module):
    """Convert irregular events into tomographic observation rays."""

    def __init__(self, num_vars: int, num_bins: int, ray_dim: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.ray_dim = ray_dim

        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.phase_embed = nn.Embedding(4, hidden_dim)
        self.bin_centers = nn.Parameter(torch.linspace(-2.5, 2.5, num_bins).repeat(num_vars, 1))
        self.bin_width = nn.Parameter(torch.ones(num_vars, num_bins))

        self.ray_row = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_bins + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ray_dim),
        )
        self.response = nn.Sequential(
            nn.Linear(num_bins + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reliability = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict) -> dict:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        measurement_std = batch.get("measurement_std", torch.zeros_like(value))
        value_pending = batch.get("value_pending", torch.zeros_like(value))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        phase = torch.floor(time_norm.clamp(0, 0.999) * 4).long()

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        centers = self.bin_centers[var_id]
        width = F.softplus(self.bin_width[var_id]) + 1e-3
        bin_logits = -((value.unsqueeze(-1) - centers) / width).pow(2)
        bin_prob = torch.softmax(bin_logits, dim=-1) * mask.unsqueeze(-1)

        var_h = self.var_embed(var_id)
        phase_h = self.phase_embed(phase)
        geom = torch.cat(
            [
                var_h,
                phase_h,
                bin_prob,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                value_pending.unsqueeze(-1),
            ],
            dim=-1,
        )
        ray = self.ray_row(geom)
        ray = F.normalize(ray, dim=-1) * mask.unsqueeze(-1)

        response_input = torch.cat(
            [bin_prob, value.unsqueeze(-1), torch.log1p(measurement_std).unsqueeze(-1)],
            dim=-1,
        )
        b = self.response(response_input).squeeze(-1) * mask

        rel_input = torch.cat(
            [
                var_h,
                time_norm.unsqueeze(-1),
                torch.log1p(delta_t).unsqueeze(-1),
                torch.log1p(measurement_std).unsqueeze(-1),
                value_pending.unsqueeze(-1),
            ],
            dim=-1,
        )
        weight = self.reliability(rel_input).squeeze(-1)
        weight = weight * mask * (1.0 - 0.8 * value_pending)

        return {
            "ray_design": ray,
            "ray_response": b,
            "ray_weight": weight.clamp(0.0, 1.0),
            "bin_prob": bin_prob,
            "pathology_bin": bin_prob.argmax(dim=-1),
            "event_mask": mask,
        }


class KaczmarzReconstructor(nn.Module):
    """Differentiable weighted Kaczmarz reconstruction of a latent pathology field."""

    def __init__(self, ray_dim: int, num_steps: int = 8):
        super().__init__()
        self.num_steps = num_steps
        self.init_theta = nn.Parameter(torch.zeros(ray_dim))
        self.step_logit = nn.Parameter(torch.tensor(0.0))
        self.ridge_log = nn.Parameter(torch.tensor(-4.0))

    @property
    def ridge(self) -> torch.Tensor:
        return F.softplus(self.ridge_log) + 1e-5

    def forward(
        self,
        ray_design: torch.Tensor,
        ray_response: torch.Tensor,
        ray_weight: torch.Tensor,
    ) -> dict:
        # ray_design: [B, N, D], ray_response/ray_weight: [B, N]
        bsz, num_rays, ray_dim = ray_design.shape
        theta = self.init_theta[None].expand(bsz, -1)
        step = torch.sigmoid(self.step_logit)

        active_weight = ray_weight.clamp_min(0.0)
        denom = ray_design.pow(2).sum(dim=-1).clamp_min(1e-4)

        # Deterministic cyclic Kaczmarz keeps the draft simple and reproducible.
        for iter_idx in range(self.num_steps):
            ray_idx = iter_idx % max(num_rays, 1)
            a_i = ray_design[:, ray_idx]
            b_i = ray_response[:, ray_idx]
            w_i = active_weight[:, ray_idx]
            residual = b_i - (a_i * theta).sum(dim=-1)
            update = step * w_i.unsqueeze(-1) * residual.unsqueeze(-1) * a_i / denom[:, ray_idx].unsqueeze(-1)
            theta = theta + update

        pred = torch.einsum("bnd,bd->bn", ray_design, theta)
        residual_all = (pred - ray_response) * active_weight
        normal_residual = torch.einsum("bnd,bn->bd", ray_design, residual_all)

        gram = torch.einsum("bnd,bn,bne->bde", ray_design, active_weight, ray_design)
        eye = torch.eye(ray_dim, device=ray_design.device, dtype=ray_design.dtype)
        gram = gram + self.ridge * eye[None]

        return {
            "theta": theta,
            "ray_pred": pred,
            "ray_residual": residual_all,
            "normal_residual": normal_residual,
            "gram": gram,
        }

    def approximate_leverage(
        self,
        gram: torch.Tensor,
        ray_design: torch.Tensor,
        ray_weight: torch.Tensor,
    ) -> torch.Tensor:
        # For a proposal draft, inverse is acceptable. Production can use conjugate gradients.
        gram_inv = torch.linalg.pinv(gram)
        leverage = torch.einsum("bnd,bde,bne->bn", ray_design, gram_inv, ray_design)
        return leverage * ray_weight


class VirtualRayForecaster(nn.Module):
    """Predict pathology bins by querying reconstructed theta with virtual rays."""

    def __init__(self, ray_dim: int, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.query_to_ray = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ray_dim),
        )
        self.bin_head = nn.Sequential(
            nn.Linear(1 + ray_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, theta: torch.Tensor, query_var: torch.Tensor, query_time: torch.Tensor, query_phase: torch.Tensor) -> dict:
        var_h = self.var_embed(query_var.clamp(0, self.num_vars - 1))
        q = torch.cat([var_h, query_time.unsqueeze(-1), query_phase.unsqueeze(-1)], dim=-1)
        a_q = F.normalize(self.query_to_ray(q), dim=-1)
        response = torch.einsum("bqd,bd->bq", a_q, theta)
        bin_logits = self.bin_head(torch.cat([response.unsqueeze(-1), a_q], dim=-1))
        return {"virtual_ray": a_q, "virtual_response": response, "bin_logits": bin_logits}


class DoKaczmarzClinicalTomography(nn.Module):
    """Sampling-policy robust classifier via clinical tomography reconstruction."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        ray_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_kaczmarz_steps: int = 12,
    ):
        super().__init__()
        self.atomizer = PathologyRayAtomizer(num_vars, num_bins, ray_dim, hidden_dim)
        self.reconstructor = KaczmarzReconstructor(ray_dim, num_steps=num_kaczmarz_steps)
        self.forecaster = VirtualRayForecaster(ray_dim, num_vars, num_bins, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(ray_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_bins = num_bins

    def encode_once(self, batch: dict) -> dict:
        ray = self.atomizer(batch)
        recon = self.reconstructor(
            ray_design=ray["ray_design"],
            ray_response=ray["ray_response"],
            ray_weight=ray["ray_weight"],
        )
        leverage = self.reconstructor.approximate_leverage(
            gram=recon["gram"],
            ray_design=ray["ray_design"],
            ray_weight=ray["ray_weight"],
        )

        residual_abs = recon["ray_residual"].abs()
        quality = torch.stack(
            [
                masked_mean(residual_abs, ray["event_mask"], dim=1),
                recon["normal_residual"].pow(2).mean(dim=-1),
                leverage.amax(dim=1),
                ray["ray_weight"].sum(dim=1) / ray["event_mask"].sum(dim=1).clamp_min(1.0),
            ],
            dim=-1,
        )
        logits = self.classifier(torch.cat([recon["theta"], quality], dim=-1))
        return {**ray, **recon, "leverage": leverage, "quality_summary": quality, "logits": logits}

    def forward(self, batch: dict) -> dict:
        return self.encode_once(batch)

    def ray_fit_loss(self, out: dict) -> torch.Tensor:
        raw = out["ray_weight"] * (out["ray_pred"] - out["ray_response"]).pow(2)
        return raw.sum() / out["ray_weight"].sum().clamp_min(1.0)

    def normal_equation_loss(self, out: dict) -> torch.Tensor:
        return out["normal_residual"].pow(2).mean()

    def leverage_capping_loss(self, out: dict, h_max: float = 0.35) -> torch.Tensor:
        return F.relu(out["leverage"] - h_max).pow(2).mean()

    def counterfactual_angle_coverage_loss(self, batch: dict, c_min: float = -8.0) -> torch.Tensor:
        if "counterfactual_ray_bank" not in batch:
            return torch.zeros((), device=batch["event_value"].device)

        losses = []
        for cf_batch in batch["counterfactual_ray_bank"]:
            cf = self.atomizer(cf_batch)
            active = cf["ray_weight"]
            gram = torch.einsum("bnd,bn,bne->bde", cf["ray_design"], active, cf["ray_design"])
            eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype)
            gram = gram + self.reconstructor.ridge.detach() * eye[None]
            sign, logabsdet = torch.linalg.slogdet(gram)
            coverage = torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, c_min - 1.0))
            losses.append(F.relu(c_min - coverage).pow(2).mean())
        return torch.stack(losses).mean()

    def virtual_ray_foreseeing_loss(self, batch: dict, out: dict) -> torch.Tensor:
        if "query_var" not in batch:
            return torch.zeros((), device=out["logits"].device)

        pred = self.forecaster(
            theta=out["theta"],
            query_var=batch["query_var"],
            query_time=batch["query_time"],
            query_phase=batch["query_phase"],
        )
        target_bin = batch["query_target_bin"].clamp(0, self.num_bins - 1)
        return F.cross_entropy(pred["bin_logits"].flatten(0, 1), target_bin.flatten())

    def training_loss(
        self,
        batch: dict,
        lambda_ray: float = 0.30,
        lambda_norm: float = 0.15,
        lambda_lev: float = 0.10,
        lambda_cov: float = 0.08,
        lambda_fore: float = 0.20,
    ) -> dict:
        labels = batch["labels"]
        out = self.forward(batch)
        cls_loss = F.cross_entropy(out["logits"], labels)
        ray_loss = self.ray_fit_loss(out)
        norm_loss = self.normal_equation_loss(out)
        lev_loss = self.leverage_capping_loss(out)
        cov_loss = self.counterfactual_angle_coverage_loss(batch)
        fore_loss = self.virtual_ray_foreseeing_loss(batch, out)

        total = (
            cls_loss
            + lambda_ray * ray_loss
            + lambda_norm * norm_loss
            + lambda_lev * lev_loss
            + lambda_cov * cov_loss
            + lambda_fore * fore_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "ray_fit_loss": ray_loss.detach(),
            "normal_equation_loss": norm_loss.detach(),
            "leverage_capping_loss": lev_loss.detach(),
            "angle_coverage_loss": cov_loss.detach(),
            "virtual_ray_foreseeing_loss": fore_loss.detach(),
            "max_ray_leverage": out["leverage"].amax(dim=1).mean().detach(),
            "mean_abs_ray_residual": out["ray_residual"].abs().mean().detach(),
        }


@torch.no_grad()
def build_counterfactual_ray_bank(batch: dict, num_views: int = 5) -> list[dict]:
    """Create counterfactual observation-ray designs for tomography training.

    These views are not consistency positives. They only change the measurement
    design matrix and reliability pattern used by the inverse problem.
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

    def clone_with(new_value, new_time, new_var, new_mask, pending=None, std=None):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        if pending is not None:
            out["value_pending"] = pending
        if std is not None:
            out["measurement_std"] = std
        out.pop("counterfactual_ray_bank", None)
        return out

    views = []

    # 1. Routine sparse scanner: retain regular coarse-time rays.
    routine_mask = mask * ((event_idx % 2) == 0).to(mask.dtype)
    routine_time = torch.round(time_norm * 6.0) / 6.0 * horizon
    views.append(clone_with(value * routine_mask, routine_time, var_id, routine_mask))

    # 2. Alarm dense scanner: emphasize late rays while thinning early rows.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((event_idx % 2) == 0).to(mask.dtype)
    alarm_mask = torch.where(late > 0, mask, mask * alternating)
    views.append(clone_with(value * alarm_mask, time, var_id, alarm_mask))

    # 3. Panel bundle scanner: snap close cross-variable observations to common times.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    bundled_time = torch.where(close > 0, torch.round(time_norm * 8.0) / 8.0 * horizon, time)
    views.append(clone_with(value * mask, bundled_time, var_id, mask))

    # 4. Pending-blind scanner: ray geometry exists, response reliability is low.
    pending = mask
    pending_std = batch.get("measurement_std", torch.zeros_like(value)) + value.detach().abs()
    views.append(clone_with(torch.zeros_like(value), time, var_id, mask, pending=pending, std=pending_std))

    # 5. Cross-center schema scanner: retain alternating variable families.
    schema_mask = mask * ((var_id % 2) == 0).to(mask.dtype)
    views.append(clone_with(value * schema_mask, time, var_id, schema_mask))

    return views[:num_views]
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cross-center scanner shift`：借鉴 PULSE，在 HiRID / MIMIC-IV / eICU 风格变量 schema、采样密度和入库延迟之间迁移。
   - `angle coverage shift`：训练中心覆盖早期/晚期、生命体征/化验多个角度，测试中心只覆盖少数变量/时间窗。
   - `high-leverage selective lab shift`：训练中心某项化验只在疑似高风险时施测，测试中心变成常规筛查，检查 leverage tail 是否造成 shortcut。
   - `panel collinearity shift`：训练中心同步 panel 造成设计矩阵共线，测试中心拆成异步射线。
   - `TCF virtual-ray shift`：未来病理 query 不再预测事件是否出现，只预测若打该射线应得到的 pathology bin。

2. **对比方法**
   - 普通 TCF / EHR foundation model。
   - PULSE 中传统强基线、深度时序模型与 LLM prompt / hybrid baseline。
   - STAR-Set、VP-GNN、MTM、QuITE、MVC-CDE、SDEVI 等 irregular / asynchronous baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、D-IVSP、KPMA、DVNB、DSPP、DCPD、DFFH、DIPF、DRG-SFF、DBCC 等。

3. **核心指标**
   - in-policy accuracy / AUPRC。
   - cross-center worst-policy accuracy。
   - ray fit residual：`mean_i |a_i^T theta - b_i|`。
   - normal-equation residual：`||A^T W (A theta - b)||`。
   - max leverage / leverage tail：预测是否被少数中心特异射线支配。
   - angle coverage logdet：测试中心设计矩阵是否缺少关键临床角度。
   - virtual-ray foreseeing AUPRC：重建状态对未来 pathology bins 的响应预测质量。
   - shortcut tomography gap：错误高置信样本是否伴随高 leverage、低 coverage 或高法方程残差。

4. **消融实验**
   - 去掉 Kaczmarz reconstruction，改用普通 event pooling，检查采样日历捷径是否回升。
   - 去掉 `L_normal_equation`，检查 `theta` 是否退化成任意 embedding。
   - 去掉 `L_leverage_capping`，观察 selective lab / panel shift 下高杠杆射线是否主导预测。
   - 去掉 `L_counterfactual_angle_coverage`，检查跨中心 coverage gap 是否无法诊断。
   - 去掉 `L_virtual_ray_foreseeing`，验证 TCF-style pathology query 是否为重建状态提供临床语义锚。
   - 将 counterfactual ray bank 替换为随机 mask，验证收益来自观测设计矩阵结构，而不是普通增强。

## 5. 预期创新性

1. **从采样偏移转向临床断层逆问题**：首次把 sampling-policy shift 表述为不同观测射线设计矩阵对同一潜在病理状态场的扫描差异。
2. **从 token / graph / proof / item / scale 转向 Kaczmarz reconstruction**：分类不直接读事件 token、变量图、证明规则、测验项或 RG fixed point，而读可由观测方程审计的重建状态。
3. **从采样去偏转向设计矩阵几何纪律**：用法方程残差、leverage capping 和 angle coverage 控制 sampling shortcut，而不是 hazard、density ratio、对抗、knockoff、conformal、DIF 或 uncertainty。
4. **保留 TCF 的病理语义，剥离 observation administration**：pathology-focused bins 变成射线响应，future-time foreseeing 变成 virtual-ray response query。
5. **对 PULSE 跨中心解释更直接**：模型能解释某中心性能退化是因为射线覆盖不足、设计矩阵共线、高杠杆化验支配，还是重建残差本身过高。
6. **与现有反事实框架低侵入兼容**：counterfactual sampler 只需输出不同采样制度下的观测射线 bank；不需要重写为证明系统、测量学模型、尺度流或双时间幕帘。

## 6. 一句话投稿卖点

**DKCT 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“不同医院/设备用不同观测射线扫描同一潜在病理状态场”的临床断层重建问题，通过 TCF-style pathology ray atomizer、可微 Kaczmarz reconstruction、normal-equation residual、ray leverage capping、counterfactual angle coverage 与 virtual-ray pathology foreseeing，让分类器依赖可由观测方程审计的 policy-neutral 状态重建，而不是依赖训练中心特有的变量可见性、panel 共线、报警复测、value-pending 或未来记录流程。**
