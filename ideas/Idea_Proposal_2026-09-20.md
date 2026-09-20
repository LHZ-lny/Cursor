# Title: Do-Blackwell Semantic Experiment Sieve：面向采样策略偏移的语义实验亏损筛

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md`、`**/*work*summary*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代总结文件。
- 已读取自动化持久记忆 `MEMORIES.md`，并纳入其中记录但当前工作区未完全落盘的历史提案摘要。
- 已读取当前仓库 `ideas/Idea_Proposal_*.md` 的全部历史提案文件，覆盖 2026-06-12 至 2026-09-19 的已落盘提案。
- 已读取近期 `paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`、`paper_daily_2026-09-18.md`，并从 `paper_daily.md` 抽取最新论文索引。重点纳入：
  - **RoMAE**：连续坐标 / axial RoPE 让通用 Transformer 处理不规则时间与通道坐标，但位置表达能力越强，越需要防止采样位置泄漏。
  - **SLAN**：switch layer 只在变量真实观测到时更新局部状态，但 switch 激活本身仍可能是医院/设备政策。
  - **CHARM**：通道语义描述有助于异构传感器迁移，但通道描述、单位和设备配置也可能携带部署政策。
  - **ECG latent ODE**：低采样率下连续形态可恢复性影响类别级鲁棒性，少数类尤其容易受采样率变化伤害。
  - **TimeCHEAT**：局部 channel-dependent、全局 channel-independent 的通道策略说明跨通道借用信息应有尺度边界，但局部共测边可能混入 policy shortcut。

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
11. source atlas / policy-antipodal retrieval、frozen predictor input-adapter contract。
12. DeepFRC-inspired monotone registration、minimal-Schwarzian warp canonicalizer、warp-curvature sidecar。
13. alternating-renewal internal clocks、renewal-normalized reward-rate readout、renewal-balance residual、exposure duty calibration。
14. palimpsest / fixed-capacity pathology memory、write pressure、duplicate fatigue、homeostatic memory slots。
15. typed SSA observation IR、semantic switchboard、dead-code elimination、Dirichlet/Robin boundary condition solver、Green kernel interior field。

本提案选择新的正交切入点：**不估计采样概率，不做密度比，不要求 logits 或 representation 一致，不做 evidence tax / proof / memory / program / boundary / privacy / retrieval；而是把每一种采样政策视为一个统计实验（statistical experiment）。如果一个高覆盖实验能通过随机 garbling 模拟另一个低覆盖实验，那么分类决策在二者之间的差异必须由可度量的 Le Cam deficiency 解释；不能被解释的部分被判为 policy-specific experiment residue。分类器只读取所有可部署实验共同支持的 Blackwell core statistic。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列的核心困难不是“时间戳不规则”本身，而是训练和部署时面对的是不同的 **观测实验**：

- 医院 A 的实验是 alarm-dense：高风险疑似患者会被密集复测；
- 医院 B 的实验是 routine-round：固定查房窗口获得较规则但稀疏的观测；
- 可穿戴设备 A 的实验是 high-rate morphology：高采样率能捕捉 ECG 局部形态；
- 可穿戴设备 B 的实验是 low-rate rhythm：低采样率保留粗节律但丢失细形态；
- TimeCHEAT 式局部跨通道借用在某些中心来自真实生理互补，在另一些中心只是 panel 共测策略。

这些实验之间并不总是“同一个样本的不同增强”。更准确地说，它们具有 **Blackwell order**：

```text
Experiment P is more informative than Q
if observations from Q can be simulated by applying a Markov garbling kernel to observations from P.
```

若高频 ECG 可以随机下采样得到低频 ECG，那么高频实验 Blackwell-dominates 低频实验；若一个完整 panel 记录可以被 garbling 成拆分后的异步记录，那么 panel-pack 实验在某些统计上比 panel-split 更丰富；但如果两个医院记录的是不同变量 schema，则二者可能只具有部分可比性。

历史方案常见目标是“去掉采样信息”或“让多政策预测一致”。这会遇到两个问题：

1. **过度删除**：采样政策有时确实改变了可恢复状态信息，例如 45 Hz ECG 不足以支持少数类心拍形态，不能强行要求与 360 Hz 同样决策。
2. **过度一致**：不同采样实验的信息量不同，低信息实验下的分类风险本来就应该更高。强制 logits 一致会掩盖信息缺失。

**Do-Blackwell Semantic Experiment Sieve (DBSES)** 的核心问题是：

> 一个分类决策在不同采样实验之间的变化，能否被“实验信息亏损”解释？如果能，高信息到低信息的性能下降是合理的；如果不能，模型很可能在使用某个采样政策独有的捷径。

因此，本提案把当前“采样解耦/反事实干预”框架改造成：

- value process 负责学习语义实验统计量；
- sampling process 只生成实验描述与可比性关系，不进入分类头；
- counterfactual intervention 生成一组可部署实验 views，用于学习 garbling kernel 与 Le Cam deficiency；
- classifier 只读取经 garbling sieve 后仍能跨实验传递的 **Blackwell core statistic**。

这样能自然吸收近期论文启发：

- 从 **RoMAE** 借连续坐标表达能力，但不让位置坐标直接成为类别证据；它只参与实验统计量。
- 从 **SLAN** 借 switch/non-imputation 思想，但 switch 激活只定义实验，而不直接分类。
- 从 **CHARM** 借 channel semantics，但通道描述只定义实验 alphabet 与语义可比性。
- 从 **ECG latent ODE** 借“采样率影响形态可恢复性”的诊断，把低采样率性能损失纳入 Blackwell deficiency，而不是误判为模型不鲁棒。
- 从 **TimeCHEAT** 借局部 CD / 全局 CI 的尺度洞察，但通过 garbling kernel 检查局部跨通道边是否能在其他实验中被模拟。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Experiment Panel，而不是 consistency views

新增 `BlackwellExperimentCollator`。每个样本返回一组由当前反事实采样模块生成的 observation experiments：

1. `factual_experiment`
   - 原始不规则事件流。

2. `experiment_view_bank`
   - `high_rate_morphology`：保留高时间分辨率，适合 ECG/传感器局部形态。
   - `low_rate_rhythm`：稀疏下采样，保留粗节律。
   - `local_cd_panel`：短窗口局部跨通道共测，模拟 TimeCHEAT 局部 CD。
   - `global_ci_channel`：按通道独立保留长期轨迹，模拟全局 CI。
   - `semantic_schema_swap`：用 CHARM 式通道语义映射替换本地变量名/单位。
   - `switch_sparse`：模拟 SLAN 式只在真实观测到时更新的稀疏 switch 实验。

3. `experiment_descriptor_bank`
   - 每个 view 的统计实验描述：变量覆盖、时间分辨率、通道语义覆盖、panel 共现强度、采样率、形态可恢复性 proxy。

4. `blackwell_order_matrix`
   - 若实验 `a` 可通过随机 garbling 模拟实验 `b`，则 `B[a,b]=1`。
   - 例如 `high_rate_morphology -> low_rate_rhythm`、`panel_pack -> panel_split`、`full_schema -> sparse_schema`。
   - 若不可比，`B[a,b]=0`，不强行要求它们互相解释。

5. `recoverability_target`
   - 对不同语义通道/类别的可恢复性弱标签，例如 ECG 局部形态、短时尖峰、低频趋势等。

关键区别：

- 这些不是 contrastive positives。
- 不要求多 view logits 一致。
- 不估计采样 hazard、density ratio 或 posterior quotient。
- 不生成 proof、conformal set、memory slots、program IR 或 boundary conditions。
- 反事实采样只用于定义“哪些实验可 garble 到哪些实验”。

### 2.2 改 Encoder：Semantic Experiment Statistic Encoder

每个实验 view 由同一个编码器处理，输出两类量：

```text
s_p      = semantic sufficient statistic under experiment p
e_p      = experiment descriptor
r_p      = policy residue statistic, only for diagnostics
```

#### A. 语义事件编码

输入事件：

```text
(value_i, time_i, variable_i, channel_description_i, quality_i)
```

编码为：

```text
h_i = EventStem(value_i, continuous_time_i, channel_semantic_i, quality_i)
```

这里：

- continuous time 借鉴 RoMAE，但只进入实验统计量，不直接作为分类 shortcut；
- channel semantic 借鉴 CHARM，但只用于通道 alphabet 与跨 schema 可比性；
- switch mask 借鉴 SLAN，但只控制该实验里哪些事件存在。

#### B. Semantic Sufficient Statistic

用轻量 set/sequence encoder 得到实验统计量：

```text
s_p = StatisticEncoder({h_i in experiment p})
```

`s_p` 的目标不是完整 reconstruction，而是能支持分类 decision 的语义统计量。

#### C. Policy Residue

同时记录实验描述 `e_p` 与 residue：

```text
r_p = ResidueHead(s_p, e_p)
```

`r_p` 不进入分类器。它只用于估计该 view 中有多少信息无法通过 Blackwell garbling 与其他实验共享。

### 2.3 改核心模块：Learned Garbling Kernel + Blackwell Core

对任意可比实验 `p -> q`，学习一个 Markov garbling kernel：

```text
G_{p -> q} = GarbleKernel(e_p, e_q)
\hat{s}_{q|p} = G_{p -> q}(s_p)
```

直觉：

- 若高频实验真的比低频实验信息更丰富，高频统计量经过 garbling 后应能模拟低频统计量；
- 若 panel 共测只是一种记录格式，panel-pack 经过 garbling 应能模拟 panel-split；
- 若某个局部跨通道边只存在于训练医院的共测政策，无法被其他实验 garble 解释，它会成为高 deficiency 的 residue。

定义可微 Le Cam deficiency proxy：

```text
delta(p, q) = || \hat{s}_{q|p} - stopgrad(s_q) ||_1
            + MMD( \hat{s}_{q|p}, stopgrad(s_q) )
```

然后构造 Blackwell core：

```text
z_core = CoreAggregator({ GarbleToCore(s_p, e_p) }_p,
                        weights = softmax(-deficiency_to_core))
```

分类器只读取 `z_core`：

```text
logits = Classifier(z_core)
```

### 2.4 改 Loss：从 invariance 转向 Le Cam Deficiency Discipline

总目标：

```text
L = L_cls
  + lambda_garb * L_blackwell_garbling
  + lambda_def  * L_deficiency_risk_envelope
  + lambda_core * L_core_sufficiency
  + lambda_res  * L_residue_nondecision
  + lambda_rec  * L_recoverability_order
```

#### A. Core Classification `L_cls`

事实样本与实验 panel 的 Blackwell core 用于分类：

```text
L_cls = CE(Classifier(z_core), y)
```

不是对每个 view 平均 logits，也不是让每个 view 一样。

#### B. Blackwell Garbling `L_blackwell_garbling`

只在 `B[p,q]=1` 的可比实验对上训练 garbling kernel：

```text
L_blackwell_garbling =
  mean_{p,q: B[p,q]=1} delta(p, q)
```

若两个实验不可比，不施加约束，避免强行把不同信息源对齐。

#### C. Deficiency Risk Envelope `L_deficiency_risk_envelope`

对可比实验 `p -> q`，高信息实验到低信息实验的真实类 margin 差异必须被 deficiency 解释：

```text
margin_p = logit_y(core_p) - max_{k != y} logit_k(core_p)
margin_q = logit_y(core_q) - max_{k != y} logit_k(core_q)

L_deficiency_risk =
  relu( |margin_p - margin_q| - c * delta(p,q) - eps )^2
```

这不是 logits consistency。若低采样率真的丢失 ECG 细形态，`delta(p,q)` 会变大，允许 margin 下降；若 `delta` 很小但 margin 大变，说明模型利用了实验 p 的 policy shortcut。

#### D. Core Sufficiency `L_core_sufficiency`

Blackwell core 必须足够承载病理语义：

```text
L_core_sufficiency = CE(PathologyHead(z_core), pathology_summary)
```

这里的 pathology summary 来自 value bins / channel groups / clinically meaningful coarse labels，不包含 policy id 或采样制度。

#### E. Residue Non-Decision `L_residue_nondecision`

policy residue 可以存在，但不能直接决定类别。用 stop-gradient label probe 审计：

```text
residue_logits = ResidueProbe(stopgrad(r_p))
L_residue_nondecision = relu(label_predictability(residue) - tau)^2
```

它不是 adversarial classifier：不训练主 encoder 去骗环境判别器，只限制被声明为 “experiment residue” 的信息不要成为隐藏分类通道。

#### F. Recoverability Order `L_recoverability_order`

低采样率实验不应声称高可恢复性：

```text
L_recoverability =
  BCE(recover_hat_p, recoverability_target_p)
  + relu(recover_low_rate - recover_high_rate - eps)^2
```

这吸收 ECG latent ODE 的类别级采样率启发，但不做 latent ODE 重构或频域掩码；它只让 deficiency 与可恢复性一致。

### 2.5 推理阶段

部署时若只有一个事实实验：

1. 构造标准 experiment descriptor；
2. 用已学习 `GarbleToCore` 映射到 Blackwell core；
3. 分类器输出类别；
4. 同时报告：
   - `deficiency_to_training_core`：当前实验距离训练核心实验有多远；
   - `policy_residue_energy`：有多少统计量无法被 garbling 解释；
   - `recoverability_profile`：每个通道/类别证据在当前采样率下是否可恢复；
   - `blackwell_warning`：若当前实验与训练核心不可比，提示需要补采样或降级输出。

---

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


def true_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_score = logits.gather(1, labels[:, None]).squeeze(1)
    rival = logits.masked_fill(
        F.one_hot(labels, logits.size(-1)).bool(),
        -1e4,
    ).max(dim=-1).values
    return true_score - rival


class ContinuousSemanticEventStem(nn.Module):
    """Encode value-bearing events under one observation experiment."""

    def __init__(self, num_vars: int, desc_dim: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.desc_proj = nn.Linear(desc_dim, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mix = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(value))
        desc = batch["channel_description"]

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        time_h = self.time_proj(torch.stack([time_norm, torch.log1p(delta_t)], dim=-1))
        value_h = self.value_proj(torch.stack([value, quality], dim=-1))
        var_h = self.var_embed(var_id.clamp_min(0))
        desc_h = self.desc_proj(desc)

        event_h = self.mix(torch.cat([time_h, value_h, var_h, desc_h], dim=-1))
        return event_h * mask.unsqueeze(-1)


class ExperimentDescriptorEncoder(nn.Module):
    """Summarize the observation experiment without feeding it to the classifier."""

    def __init__(self, num_vars: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.net = nn.Sequential(
            nn.Linear(num_vars + 9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        time = batch["event_time"]
        var_id = batch["event_var_id"].clamp(0, self.num_vars - 1)
        mask = batch["event_mask"]
        quality = batch.get("measurement_quality", torch.ones_like(time))
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
                masked_mean(quality, mask, dim=1).unsqueeze(-1),
                masked_mean(pending, mask, dim=1).unsqueeze(-1),
                mask.sum(dim=1, keepdim=True) / mask.size(1),
            ],
            dim=-1,
        )
        return self.net(torch.cat([var_rate, stats], dim=-1))


class SemanticStatisticEncoder(nn.Module):
    """Produce a semantic statistic for one statistical experiment."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.pool = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, event_h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_h, _ = self.context(event_h)
        pooled = masked_mean(seq_h, mask, dim=1)
        return self.pool(pooled)


class GarblingKernel(nn.Module):
    """Learn a Markov-style garbling from one experiment statistic to another."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.kernel_net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.residual = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, stat_src: torch.Tensor, desc_src: torch.Tensor, desc_tgt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([stat_src, desc_src, desc_tgt], dim=-1)
        gate, scale = self.kernel_net(x).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        scale = 0.25 * torch.tanh(scale)
        return stat_src * (1.0 + scale) * gate + self.residual(x) * (1.0 - gate)


class BlackwellCoreAggregator(nn.Module):
    """Map experiment statistics into a common Blackwell core statistic."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.core_desc = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.to_core = GarblingKernel(hidden_dim)
        self.weight = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stats: torch.Tensor, desc: torch.Tensor) -> dict:
        # stats, desc: [B, P, H]
        bsz, num_exp, hidden_dim = stats.shape
        core_desc = self.core_desc.view(1, 1, hidden_dim).expand(bsz, num_exp, -1)
        core_candidates = []
        for idx in range(num_exp):
            core_candidates.append(self.to_core(stats[:, idx], desc[:, idx], core_desc[:, idx]))
        core_candidates = torch.stack(core_candidates, dim=1)

        raw_weight = self.weight(torch.cat([core_candidates, desc], dim=-1)).squeeze(-1)
        alpha = torch.softmax(raw_weight, dim=1)
        core = torch.einsum("bp,bph->bh", alpha, core_candidates)
        return {"core": core, "core_candidates": core_candidates, "core_weight": alpha}


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    xx = torch.exp(-torch.cdist(x, x).pow(2) / (2 * sigma**2)).mean()
    yy = torch.exp(-torch.cdist(y, y).pow(2) / (2 * sigma**2)).mean()
    xy = torch.exp(-torch.cdist(x, y).pow(2) / (2 * sigma**2)).mean()
    return xx + yy - 2 * xy


class DoBlackwellSemanticExperimentSieve(nn.Module):
    """Sampling-policy robust classifier based on Blackwell experiment garbling."""

    def __init__(
        self,
        num_vars: int,
        desc_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_pathology_bins: int,
    ):
        super().__init__()
        self.event = ContinuousSemanticEventStem(num_vars, desc_dim, hidden_dim)
        self.desc = ExperimentDescriptorEncoder(num_vars, hidden_dim)
        self.stat = SemanticStatisticEncoder(hidden_dim)
        self.garble = GarblingKernel(hidden_dim)
        self.core = BlackwellCoreAggregator(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.pathology_head = nn.Linear(hidden_dim, num_pathology_bins)
        self.residue_probe = nn.Linear(hidden_dim, num_classes)
        self.recover_head = nn.Linear(hidden_dim, 1)
        self.num_classes = num_classes

    def encode_one(self, batch: dict) -> dict:
        event_h = self.event(batch)
        desc = self.desc(batch)
        stat = self.stat(event_h, batch["event_mask"])
        recover = torch.sigmoid(self.recover_head(stat)).squeeze(-1)
        return {"stat": stat, "desc": desc, "recover": recover}

    def encode_panel(self, batch: dict) -> dict:
        views = batch.get("experiment_view_bank", [batch])
        stats, descs, recovers = [], [], []
        for view in views:
            out = self.encode_one(view)
            stats.append(out["stat"])
            descs.append(out["desc"])
            recovers.append(out["recover"])
        stats = torch.stack(stats, dim=1)
        descs = torch.stack(descs, dim=1)
        recovers = torch.stack(recovers, dim=1)
        core = self.core(stats, descs)
        logits = self.classifier(core["core"])
        return {**core, "stats": stats, "descs": descs, "recover": recovers, "logits": logits}

    def garbling_deficiency_matrix(self, out: dict, order: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = out["stats"]
        descs = out["descs"]
        bsz, num_exp, _ = stats.shape
        deficiency = stats.new_zeros(bsz, num_exp, num_exp)
        translated_cache = []

        for src in range(num_exp):
            row = []
            for tgt in range(num_exp):
                translated = self.garble(stats[:, src], descs[:, src], descs[:, tgt])
                row.append(translated)
                l1 = (translated - stats[:, tgt].detach()).abs().mean(dim=-1)
                # Batch-level MMD is added as a shared term for stability.
                mmd = mmd_rbf(translated, stats[:, tgt].detach()).detach()
                deficiency[:, src, tgt] = l1 + 0.05 * mmd
            translated_cache.append(row)

        deficiency = deficiency * order.to(deficiency.dtype)
        return deficiency, translated_cache

    def blackwell_garbling_loss(self, out: dict, order: torch.Tensor) -> torch.Tensor:
        deficiency, _ = self.garbling_deficiency_matrix(out, order)
        denom = order.sum().clamp_min(1.0)
        return deficiency.sum() / denom

    def deficiency_risk_envelope_loss(
        self,
        out: dict,
        labels: torch.Tensor,
        order: torch.Tensor,
        bound_scale: float = 2.0,
        eps: float = 0.05,
    ) -> torch.Tensor:
        deficiency, _ = self.garbling_deficiency_matrix(out, order)
        logits_by_view = self.classifier(out["core_candidates"])
        margins = []
        for idx in range(logits_by_view.size(1)):
            margins.append(true_margin(logits_by_view[:, idx], labels))
        margins = torch.stack(margins, dim=1)

        losses = []
        for src in range(margins.size(1)):
            for tgt in range(margins.size(1)):
                if order[src, tgt] <= 0:
                    continue
                gap = (margins[:, src] - margins[:, tgt]).abs()
                allowed = bound_scale * deficiency[:, src, tgt].detach() + eps
                losses.append(F.relu(gap - allowed).pow(2).mean())
        if not losses:
            return torch.zeros((), device=labels.device)
        return torch.stack(losses).mean()

    def core_sufficiency_loss(self, out: dict, batch: dict) -> torch.Tensor:
        if "pathology_summary" not in batch:
            return torch.zeros((), device=out["logits"].device)
        pred = self.pathology_head(out["core"])
        return F.cross_entropy(pred, batch["pathology_summary"].long())

    def residue_nondecision_loss(self, out: dict, labels: torch.Tensor, tau: float = 0.15) -> torch.Tensor:
        residue = out["stats"] - out["core_candidates"].detach()
        residue_logits = self.residue_probe(residue.detach())
        ce = F.cross_entropy(residue_logits.flatten(0, 1), labels[:, None].expand(-1, residue.size(1)).flatten())
        uniform_ce = torch.log(torch.tensor(self.num_classes, device=ce.device, dtype=ce.dtype))
        # Penalize residue if a frozen probe can predict labels much better than uniform.
        return F.relu(uniform_ce - ce - tau).pow(2)

    def recoverability_order_loss(self, out: dict, batch: dict) -> torch.Tensor:
        if "recoverability_target" not in batch:
            return torch.zeros((), device=out["logits"].device)
        target = batch["recoverability_target"].to(out["recover"].dtype)
        bce = F.binary_cross_entropy(out["recover"], target)
        if "recoverability_order" not in batch:
            return bce
        # recoverability_order[a,b]=1 means a should be at least as recoverable as b.
        order = batch["recoverability_order"]
        losses = []
        for src in range(order.size(0)):
            for tgt in range(order.size(1)):
                if order[src, tgt] > 0:
                    losses.append(F.relu(out["recover"][:, tgt] - out["recover"][:, src] - 0.02).pow(2).mean())
        if losses:
            return bce + torch.stack(losses).mean()
        return bce

    def training_loss(
        self,
        batch: dict,
        lambda_garb: float = 0.30,
        lambda_def: float = 0.20,
        lambda_core: float = 0.10,
        lambda_res: float = 0.05,
        lambda_rec: float = 0.10,
    ) -> dict:
        out = self.encode_panel(batch)
        labels = batch["labels"]
        order = batch["blackwell_order_matrix"].to(out["logits"].device)

        cls_loss = F.cross_entropy(out["logits"], labels)
        garb_loss = self.blackwell_garbling_loss(out, order)
        def_loss = self.deficiency_risk_envelope_loss(out, labels, order)
        core_loss = self.core_sufficiency_loss(out, batch)
        residue_loss = self.residue_nondecision_loss(out, labels)
        recover_loss = self.recoverability_order_loss(out, batch)

        total = (
            cls_loss
            + lambda_garb * garb_loss
            + lambda_def * def_loss
            + lambda_core * core_loss
            + lambda_res * residue_loss
            + lambda_rec * recover_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "blackwell_garbling_loss": garb_loss.detach(),
            "deficiency_risk_envelope_loss": def_loss.detach(),
            "core_sufficiency_loss": core_loss.detach(),
            "residue_nondecision_loss": residue_loss.detach(),
            "recoverability_order_loss": recover_loss.detach(),
            "mean_recoverability": out["recover"].mean().detach(),
        }
```

---

## 4. BlackwellExperimentCollator 草稿

```python
import torch


@torch.no_grad()
def build_blackwell_experiment_panel(batch: dict) -> dict:
    """Create experiment views and a Blackwell comparability matrix.

    The views are statistical experiments, not contrastive positives.
    A view pair is constrained only when one experiment can be garbled into the other.
    """

    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    quality = batch.get("measurement_quality", torch.ones_like(value))
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, new_quality):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        out["measurement_quality"] = new_quality
        out.pop("experiment_view_bank", None)
        return out

    views = []

    # 0. Factual experiment.
    views.append(clone_with(value * mask, time, var_id, mask, quality))

    # 1. High-rate morphology: keep dense local information.
    views.append(clone_with(value * mask, time, var_id, mask, quality))

    # 2. Low-rate rhythm: structured thinning preserves coarse temporal coverage.
    low_rate = mask * ((torch.arange(num_events, device=device)[None] % 3) == 0).to(mask.dtype)
    views.append(clone_with(value * low_rate, time, var_id, low_rate, quality * low_rate))

    # 3. Local channel-dependent panel: emphasize near-synchronous cross-channel events.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    panel_mask = torch.maximum(mask * close, mask * 0.5)
    views.append(clone_with(value * panel_mask, time, var_id, panel_mask, quality * panel_mask))

    # 4. Global channel-independent view: keep per-variable sparse representatives.
    ci_mask = torch.zeros_like(mask)
    for var in torch.unique(var_id[mask > 0]).tolist():
        hit = ((var_id == int(var)) & (mask > 0)).to(mask.dtype)
        rank = hit.cumsum(dim=1)
        ci_mask = torch.maximum(ci_mask, (rank <= 3).to(mask.dtype) * hit)
    views.append(clone_with(value * ci_mask, time, var_id, ci_mask, quality * ci_mask))

    # 5. Switch-sparse experiment: only actual high-quality returned values update.
    pending = batch.get("value_pending", torch.zeros_like(mask))
    switch_mask = mask * (quality > quality.mean(dim=1, keepdim=True)).to(mask.dtype) * (1.0 - pending)
    views.append(clone_with(value * switch_mask, time, var_id, switch_mask, quality * switch_mask))

    num_views = len(views)
    order = torch.zeros(num_views, num_views, device=device)
    # Factual/high-rate can be garbled into lower-information experiments.
    order[0, 2:] = 1.0
    order[1, 2:] = 1.0
    # Panel and CI can be garbled into switch-sparse when coverage is reduced.
    order[3, 5] = 1.0
    order[4, 5] = 1.0
    # Self-pairs are useful for stabilizing the learned core map.
    order.fill_diagonal_(1.0)

    out = dict(batch)
    out["experiment_view_bank"] = views
    out["blackwell_order_matrix"] = order

    if "recoverability_target" not in out:
        recover = torch.ones(bsz, num_views, device=device)
        recover[:, 2] = 0.55  # low-rate view: local morphology partly unrecoverable.
        recover[:, 5] = 0.65  # switch-sparse view: depends on returned values.
        out["recoverability_target"] = recover
        rec_order = torch.zeros(num_views, num_views, device=device)
        rec_order[1, 2] = 1.0
        rec_order[1, 5] = 1.0
        rec_order[0, 2] = 1.0
        out["recoverability_order"] = rec_order

    return out
```

---

## 5. 实验切入点

1. **Policy shift 构造**
   - `high-rate -> low-rate morphology`：ECG / wearable 从 360 Hz 或高密度采样降到 90 Hz / 45 Hz / 不均匀低频，报告少数类 F1。
   - `local-CD -> global-CI channel shift`：训练中局部 panel 共测强，测试中全局通道独立采样，或反向。
   - `schema / unit / device swap`：同一语义通道在不同医院或设备中具有不同命名和单位。
   - `switch sparsity shift`：某些变量在测试中心只在真实返回值时更新，训练中心则有 pending / stub / repeated check。
   - `recoverability reversal`：训练中高采样率足以恢复形态，测试中同一类别证据不可恢复。

2. **对比方法**
   - RoMAE / continuous RoPE baseline。
   - SLAN / switch-layer non-imputation baseline。
   - CHARM / channel semantic representation baseline。
   - TimeCHEAT local CD + global CI baseline。
   - ECG latent ODE / continuous-time morphology baseline。
   - 历史方案：DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、DJRT、DSPP、DCPD、DIPF、DRG-SFF、DPPC、DCOFF、DD-JEPA、DPSP、DNSA、DMWI、DCST、DAAA、DREM、Do-Palimpsest、DSSS、DRBI。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - worst-experiment AUROC / AUPRC。
   - Blackwell deficiency matrix：不同采样实验间的可 garble 程度。
   - deficiency-explained margin drift：分类 margin 改变是否可由 deficiency 解释。
   - policy residue label leakage：不可 garble residue 是否仍能预测标签。
   - recoverability calibration：低采样率 / 稀疏 switch 下的类别证据是否被正确降级。
   - experiment incomparability alarm：测试实验与训练实验不可比时的告警质量。

4. **消融实验**
   - 去掉 `L_blackwell_garbling`，检查模型是否重新依赖某个 policy-specific experiment。
   - 去掉 `L_deficiency_risk_envelope`，检查 margin drift 是否超过可解释信息亏损。
   - 用 logits consistency 替代 deficiency envelope，验证强一致性是否损害低采样率场景。
   - 去掉 channel semantic descriptor，验证 CHARM 式语义 alphabet 对跨 schema garbling 的贡献。
   - 去掉 recoverability order，检查低频 ECG / wearable 少数类是否出现高置信错误。
   - 将 experiment descriptor 直接拼入 classifier 作为反例，验证院内性能可能上升但 worst-experiment 下降。

---

## 6. 预期创新性

1. **从采样去偏转向统计实验可比性**：首次把非规则采样策略视为 Blackwell statistical experiments，而不是 nuisance mask、workflow、程序、边界、记忆或隐私属性。
2. **从多视图一致转向 Le Cam deficiency envelope**：不要求不同实验下 logits 相同；只要求决策差异被实验信息亏损解释。
3. **从通道语义增强转向实验 alphabet 对齐**：吸收 CHARM 的 channel descriptions，但用途是定义可 garble 的语义观测字母表，而不是增强分类特征。
4. **从 switch 激活转向实验描述**：吸收 SLAN 的 non-imputation / switch update 思想，但 switch 只定义实验信息量，不进入分类头。
5. **从连续位置编码转向实验统计量**：吸收 RoMAE 的连续坐标能力，但位置只服务 experiment statistic 与 garbling kernel。
6. **从采样率鲁棒转向 recoverability-aware deficiency**：吸收 ECG latent ODE 的低采样率启发，但不重构波形，也不做频域掩码；低采样损失通过 deficiency 和 recoverability 被显式计量。
7. **诊断直接服务部署**：当跨医院失败时，DBSES 能说明是测试实验不可比、deficiency 过大、recoverability 不足，还是 policy residue 仍在泄漏标签。

## 7. 一句话投稿卖点

**DBSES 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“训练和部署处在不同 Blackwell 统计实验中”的问题，通过语义实验统计量、可学习 garbling kernel、Le Cam deficiency risk envelope 与 recoverability-aware Blackwell core，让分类器只依赖跨采样实验可传递的病理语义，而不是依赖高频采样、panel 共测、switch 激活、通道命名或低采样率不可恢复形态造成的政策捷径。**
