# Title: Do-SSA Semantic Switchboard：面向采样策略偏移的观测程序死代码消除分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已扩大搜索 `**/*summary*.md`、`**/*Summary*.md`、`**/*work*.md`、`**/*work*summary*.md` 与中文 `**/*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取当前工作区内全部历史 proposal 的标题、黑名单与核心机制段落，覆盖：
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
  - `ideas/Idea_Proposal_2026-09-14.md`
  - `ideas/Idea_Proposal_2026-09-16.md`
  - `ideas/Idea_Proposal_2026-09-17.md`
- 已读取自动化记忆 `MEMORIES.md` 及历史摘要，额外纳入当前工作区未完整落盘或以记忆形式保留的机制：`idea_2026-07-24.md`、`2026-07-25.md`、`2026-07-26.md`、`2026-07-27.md`、`2026-07-29.md`、`2026-07-31.md`、`2026-08-01.md`、`2026-08-04.md`、`2026-08-07.md`、`2026-08-10.md`、`2026-08-11.md`、`2026-08-21.md` 等。
- 已读取近期 `paper_daily.md` 与最新日期文件 `paper_daily_2026-09-11.md`、`paper_daily_2026-09-13.md`、`paper_daily_2026-09-14.md`、`paper_daily_2026-09-15.md`、`paper_daily_2026-09-16.md`、`paper_daily_2026-09-17.md`，重点纳入：
  - **CHARM / Giving Sensors a Voice**：通道文本描述、channel-aware representation、semantic channel addressing 与跨传感器泛化。
  - **ECG latent ODE classification**：低采样率下的连续形态可恢复性、类别级少数类脆弱性。
  - **RoMAE**：连续坐标 / axial RoPE 能处理不规则时间和通道坐标，但位置表达力越强越需要审计采样位置泄漏。
  - **SLAN**：switch layer 不做插补，只有被真实观测到的变量局部状态更新，但开关激活本身也可能成为 policy shortcut。
  - **Continuum Dropout / BAT / ORA / INPUTADAPTER / OpenTSLM / DeepFRC** 等近期机制均纳入黑名单参考，但不复用其历史 proposal 主方法。

### 历史核心机制黑名单

为避免思维重合，本轮明确避开以下历史主机制：

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

本提案选择新的正交切入点：**不估计采样概率，不要求多视图 logits 或 representation 一致，不做记忆板、暴露率、检索、证明、投票、隐私、后验商、图交换、拓扑、gauge、元适配或合成锻炉；而是把不规则观测流编译成一段带类型的 Static Single Assignment (SSA) 观测程序。采样政策带来的 rename、unit-cast、panel split、repeat check、pending stub、low-rate read 等被显式标注为程序指令。分类器只读取通过语义类型检查和 liveness analysis 后的 canonical physiological registers；纯行政采样指令会被可微 Dead-Code Eliminator 消掉，不能凭出现次数、通道名称或开关激活模式注入类别证据。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

当前非规则采样时间序列分类中，一个被低估的偏移来源是 **观测程序漂移**：

- 同一个乳酸检测，在医院 A 叫 `lactate_blood`，在医院 B 叫 `lactic_acid_plasma`，单位和正常区间可能不同；
- 同一个化验 panel，在一个中心是一次同步记录，在另一个中心被拆成多个异步事件；
- 某个变量的重复复测可能只是 protocol retry，另一个设备的低频读取可能是电池策略，而不是病程稳定；
- 可穿戴 ECG 中 360 Hz、90 Hz、45 Hz 的采样率改变了某些局部形态是否可恢复，少数类最先受损；
- CHARM 提醒我们通道语义和文本描述很重要，但通道描述也可能携带设备/医院配置；SLAN 提醒我们不要插补未观测变量，但 switch 是否打开本身仍可能成为 shortcut。

历史方案大多把采样偏移建模成概率、后验、图、拓扑、证据、校准、隐私、时间尺度、记忆写入或工作流适配。本轮换一个软件系统视角：

> 同一条潜在病程可以被不同医院/设备编译成不同的观测程序。真正可迁移的是程序执行后 canonical physiological registers 中的语义值；不可迁移的是程序里那些只改变记录形式、变量命名、panel 打包、返回延迟或采样频率的行政指令。鲁棒分类器需要像编译器一样做 type checking、SSA def-use analysis 和 dead-code elimination，而不是让每个观测 token 都有资格影响类别边界。

**Do-SSA Semantic Switchboard (DSSS)** 的核心直觉：

1. **把不规则事件变成 typed instructions**：每个事件不只是 `(time, variable, value)`，而是 `READ_VALUE / UNIT_CAST / ALIAS_RENAME / PANEL_SPLIT / REPEAT_CHECK / PENDING_STUB / LOW_RATE_READ` 等程序指令。
2. **用通道语义做类型系统，而不是做分类特征**：借鉴 CHARM 的 channel descriptions，但将其用于 canonical physiological type routing，如 `perfusion.lactate`、`renal.creatinine`、`cardiac.ecg_qrs`，不直接拼进分类头。
3. **用 switch 更新替代插补**：借鉴 SLAN 的非插补思想，只有通过 liveness analysis 的真实 value definition 才能写入对应 canonical register；未测变量不会被伪造输入。
4. **用形态可恢复性保护低采样率场景**：借鉴 ECG latent ODE 的发现，低采样率不是简单缺点，而是某些局部形态不可恢复。DSSS 不让不可恢复的 fine morphology 指令作为 live class evidence。
5. **反事实干预变成 program rewrites**：当前采样解耦/反事实模块不再生成一致性视图，而是生成语义保持或信息破坏的观测程序改写，用于监督哪些指令应当被消除、哪些定义仍可安全写入。

这样可以解决 sampling-policy shift 的三个关键问题：

- **命名/单位/设备漂移**：变量名和单位改变时，通过 type checker 和 unit cast 落到同一 canonical register，而不是让模型记住局部命名。
- **panel / repeat / pending 漂移**：这些采样制度只产生行政指令或低质量 stub，若没有新的 value definition，就被 DCE 标记为 dead code。
- **采样率与形态漂移**：当采样间隔超过某类形态所需分辨率，recoverability gate 会阻止模型从不可恢复细节中制造高置信证据。

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：从 event batch 改为 Observation Program Rewrite Batch

新增 `ObservationProgramCollator`，把每条不规则序列编译为 typed observation IR：

```text
instruction_i = {
    op_code: READ_VALUE / UNIT_CAST / ALIAS_RENAME / PANEL_SPLIT / REPEAT_CHECK / PENDING_STUB / LOW_RATE_READ,
    semantic_desc: channel text / unit / device / clinical group,
    raw_value: observed value or pending placeholder,
    time_coord: continuous timestamp,
    variable_id: local schema id,
    rewrite_group_id: which counterfactual rewrite family produced it,
    admin_label: whether this instruction is sampling-administrative,
    canonical_type_target: optional weak label from ontology / variable mapping,
    recoverability_target: whether required morphology is observable at this sampling resolution
}
```

反事实干预模块产生两类 program rewrites：

1. **语义保持改写**
   - `ALIAS_RENAME`：变量局部命名替换为同义通道；
   - `UNIT_CAST`：mmol/L 与 mg/dL 等单位转换；
   - `PANEL_SPLIT / PANEL_PACK`：同步 panel 拆分或打包；
   - `REPEAT_CHECK`：同值短间隔复测；
   - `PENDING_STUB`：值未返回但记录已存在。

   这些改写不应产生新的 live physiological definition；它们主要测试 DCE 是否能识别记录形式变化。

2. **信息破坏改写**
   - `LOW_RATE_READ`：降低采样率；
   - `WINDOW_THIN`：删除某些局部形态所需的中间点；
   - `DEVICE_DUTY_CYCLE`：用设备策略改变可见窗口。

   这些改写允许降低某些细粒度证据的 recoverability，但不能让模型把“低采样率本身”当作类别证据。

关键区别：

- 不生成 contrastive positive pair；
- 不要求多策略 logits 一致；
- 不构造 source atlas、risk set、conformal set、jury、proof 或 memory slots；
- sampling branch 的输出是 program-op supervision 与 rewrite metadata，不进入分类头。

### 2.2 改 Encoder：Semantic Type Checker + SSA Switchboard

DSSS 的 encoder 分为三步。

#### A. Instruction Embedding

每条观测指令由四类信息组成：

```text
u_i = EmbedValue(x_i)
    + EmbedOp(op_code_i)
    + AxialContinuousCoord(time_i, local_var_id_i)
    + ChannelDescriptionEncoder(desc_i)
```

这里吸收 RoMAE 的连续坐标思想和 CHARM 的通道语义思想，但不把连续位置或通道描述作为直接类别 shortcut。它们只服务于后续类型检查与 def-use analysis。

#### B. Semantic Type Checking

维护一个 canonical physiological type dictionary：

```text
T = {perfusion.lactate, inflammation.crp, renal.creatinine,
     cardiac.qrs_width, respiratory.spo2, ...}
```

Type checker 输出：

```text
type_prob_i = softmax(TypeHead(u_i, T))
unit_ok_i   = sigmoid(UnitHead(u_i))
```

如果某个本地变量名、单位或设备描述无法可信映射到 canonical type，则它不允许直接写入分类寄存器，只进入 policy/program trace 诊断。

#### C. SSA Liveness Switchboard

对每条指令判断是否定义了一个新的 physiological register version：

```text
live_i = sigmoid(LivenessHead(u_i, type_prob_i, unit_ok_i, recoverability_i))
```

写入规则是 switch-style 的：

```text
R_{k}^{(new)} = phi(R_{k}^{(old)}, value_def_i)
```

但只有当 `type_prob_i` 指向 canonical type `k` 且 `live_i` 足够高时，才创建新版本。`ALIAS_RENAME`、`UNIT_CAST`、`PANEL_SPLIT`、`REPEAT_CHECK`、`PENDING_STUB` 等行政指令可以更新 program trace，却不应产生新的 live definition。最终分类器读取：

```text
z = Pool({R_k^last, register_age_k, recoverability_k})
logits = Classifier(z)
```

这与 Do-Palimpsest 的固定容量记忆不同：DSSS 没有记忆压力、重复疲劳或自由 slot assignment；它是一套由语义类型系统约束的 canonical registers，目标是消除采样程序中的 dead code。

### 2.3 改 Loss：从去偏约束转向 Program Liveness Discipline

总目标：

```text
L = L_cls
  + lambda_type * L_semantic_typecheck
  + lambda_dce  * L_deadcode_elimination
  + lambda_phi  * L_ssa_phi_node
  + lambda_rec  * L_morphology_recoverability
  + lambda_prog * L_program_trace_absorption
```

#### A. Register Classification `L_cls`

只使用最后的 canonical physiological registers 分类：

```text
L_cls = CE(Classifier(Pool(R_last)), y)
```

分类头不接收 `op_code`、本地变量名、center id、rewrite group、panel id、pending flag 或 policy descriptor。它只能看到通过类型检查、单位校验和 liveness analysis 后的语义寄存器。

#### B. Semantic Typecheck `L_semantic_typecheck`

利用数据字典、变量描述、单位和弱 ontology 映射监督 type checker：

```text
L_type = CE(type_prob, canonical_type_target)
       + BCE(unit_ok, unit_consistency_target)
```

这把 CHARM 的 channel semantic description 从“更强通道表征”改成“临床类型系统”：通道语义只决定写入哪个 canonical register，不直接产生类别分数。

#### C. Dead-Code Elimination `L_deadcode_elimination`

对反事实 program rewrites 中已知只改变记录形式的行政指令，惩罚其 live credit：

```text
L_dce = mean(admin_label_i * live_i)
```

例如：

- 同一值被 `REPEAT_CHECK` 复制，不应创建新 value definition；
- `PANEL_SPLIT` 只把同一 panel 拆成多个时间戳，不应让每个拆分 token 都独立增加类别证据；
- `PENDING_STUB` 没有真实值返回，只能进入 program trace；
- `ALIAS_RENAME` 和 `UNIT_CAST` 改变命名/单位，但不改变语义寄存器的事实值。

这不是 protocol tax：没有证据预算或购买价格。它也不是 logits consistency：不比较不同采样视图输出，只监督哪些程序指令是死代码。

#### D. SSA Phi-Node Discipline `L_ssa_phi_node`

当多个本地通道是同一 canonical type 的 alias，或 panel split 后多个子事件共同对应一次语义测量，DSSS 使用可微 `phi` 节点合并定义：

```text
R_k^phi = PhiNode({value_def_i | type_i = k and group_i = g})
```

损失约束 `phi` 节点满足两个程序语义：

1. **alias idempotence**：同一语义值经 rename/unit-cast 后只定义一次；
2. **split-pack associativity**：先 pack 再写入与先 split 再 phi 合并应产生同一 register definition。

```text
L_phi = ||R_phi(split(group)) - R_phi(pack(group))||_1
      + relu(num_live_defs(group) - allowed_defs(group))^2
```

这里约束的是 SSA def-use 图的合法性，而不是要求全部样本或全部策略的 patient representation 一致。

#### E. Morphology Recoverability `L_morphology_recoverability`

借鉴 ECG latent ODE 的采样率鲁棒性观察：低频采样可以保留粗节律，却可能丢失少数类需要的局部形态。DSSS 为每条 live candidate 估计 recoverability：

```text
recover_i = sigmoid(RecoverHead(u_i, delta_t_i, semantic_type_i))
```

collator 根据 high-rate reference、局部斜率/曲率代理或临床变量最小采样要求提供弱标签：

```text
L_rec = BCE(recover_i, recoverability_target_i)
```

并将 `recover_i` 作为 liveness 输入，而不是输出不确定性集合：

```text
live_i = live_i * recover_i
```

如果某个低采样率 ECG 片段不足以恢复 QRS 宽度或 P 波细节，模型不能把“设备低采样率”本身当作某个心律失常类别证据；它只能使用仍可恢复的粗节律 register。

#### F. Program Trace Absorption `L_program_trace_absorption`

被 DCE 消掉的行政指令并不浪费：它们进入 program trace head，预测当前采样程序类型，用于诊断和偏移告警：

```text
L_prog = CE(program_op_hat, op_code)
       + CE(rewrite_hat, rewrite_group_id)
```

但该 head 与分类头参数隔离，且不反向给 register classifier。它的作用是解释“为什么当前样本的观测程序与训练中心不同”，不是提供类别证据。

### 2.4 推理阶段

给定一个新医院或新设备的不规则序列：

1. 将原始事件编译为 typed observation instructions；
2. type checker 将本地变量名、单位、设备描述映射到 canonical physiological types；
3. liveness switchboard 只让真实、可恢复、非行政的 value definitions 写入 SSA registers；
4. classifier 读取最终 canonical register bank 输出类别；
5. 同时报告：
   - `dead_code_rate`：多少事件被判为行政采样程序；
   - `type_ambiguity`：哪些变量无法稳定映射到 canonical type；
   - `recoverability_gap`：哪些类别相关形态在当前采样率下不可恢复；
   - `program_shift_score`：新中心观测程序与训练程序差异。

高 `type_ambiguity` 或高 `recoverability_gap` 不是直接拒识集合，也不是 evidential uncertainty；它是数据工程与补采样建议：需要补充单位字典、变量映射或提高特定通道采样率。

## 3. Code Draft: PyTorch 核心模块草稿

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(dtype=x.dtype)
    while weight.dim() < x.dim():
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


@dataclass
class DSSSOutput:
    logits: torch.Tensor
    registers: torch.Tensor
    live_prob: torch.Tensor
    type_prob: torch.Tensor
    unit_ok: torch.Tensor
    recover_prob: torch.Tensor
    program_logits: torch.Tensor


class ContinuousAxialCoord(nn.Module):
    """Small RoMAE-style continuous coordinate embedder for time and local channel id."""

    def __init__(self, hidden_dim: int, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.proj = nn.Linear(4 * num_frequencies, hidden_dim)
        freq = torch.exp(torch.linspace(0.0, 4.0, num_frequencies))
        self.register_buffer("freq", freq)

    def encode_scalar(self, x: torch.Tensor) -> torch.Tensor:
        angles = x.unsqueeze(-1) * self.freq
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, time: torch.Tensor, channel_pos: torch.Tensor) -> torch.Tensor:
        coord = torch.cat(
            [self.encode_scalar(time), self.encode_scalar(channel_pos.float())],
            dim=-1,
        )
        return self.proj(coord)


class TypeDictionary(nn.Module):
    """Canonical physiological type dictionary addressed by channel descriptions."""

    def __init__(self, num_types: int, desc_dim: int, hidden_dim: int):
        super().__init__()
        self.type_embed = nn.Parameter(torch.randn(num_types, hidden_dim) * 0.02)
        self.desc_proj = nn.Linear(desc_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inst_hidden: torch.Tensor, channel_desc: torch.Tensor) -> torch.Tensor:
        desc_key = self.desc_proj(channel_desc)
        query = self.query_proj(inst_hidden + desc_key)
        logits = torch.einsum("bth,kh->btk", query, self.type_embed)
        return logits


class SSASemanticSwitchboard(nn.Module):
    def __init__(
        self,
        num_types: int,
        num_ops: int,
        desc_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.num_types = num_types
        self.hidden_dim = hidden_dim

        self.value_proj = nn.Linear(1, hidden_dim)
        self.op_embed = nn.Embedding(num_ops, hidden_dim)
        self.coord = ContinuousAxialCoord(hidden_dim)
        self.type_dict = TypeDictionary(num_types, desc_dim, hidden_dim)

        self.inst_norm = nn.LayerNorm(hidden_dim)
        self.inst_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.unit_head = nn.Linear(hidden_dim, 1)
        self.recover_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.live_head = nn.Sequential(
            nn.Linear(hidden_dim + num_types + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.write_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.program_head = nn.Linear(hidden_dim, num_ops)

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        local_var: torch.Tensor,
        op_code: torch.Tensor,
        channel_desc: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> DSSSOutput:
        """
        Args:
            value: [B, T] observed value or placeholder.
            time: [B, T] continuous timestamp.
            local_var: [B, T] local schema/channel id.
            op_code: [B, T] observation IR instruction id.
            channel_desc: [B, T, D] text/unit/device description embedding.
            event_mask: [B, T] valid instruction mask.
        """
        value_hidden = self.value_proj(value.unsqueeze(-1))
        op_hidden = self.op_embed(op_code)
        coord_hidden = self.coord(time, local_var)
        desc_hidden = self.type_dict.desc_proj(channel_desc)
        inst = self.inst_norm(value_hidden + op_hidden + coord_hidden + desc_hidden)
        inst = self.inst_mlp(torch.cat([inst, desc_hidden], dim=-1))

        type_logits = self.type_dict(inst, channel_desc)
        type_prob = F.softmax(type_logits, dim=-1)
        unit_ok = torch.sigmoid(self.unit_head(inst)).squeeze(-1)

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        recover_in = torch.cat([inst, torch.log1p(delta_t).unsqueeze(-1)], dim=-1)
        recover_prob = torch.sigmoid(self.recover_head(recover_in)).squeeze(-1)

        live_in = torch.cat(
            [inst, type_prob, unit_ok.unsqueeze(-1), recover_prob.unsqueeze(-1)],
            dim=-1,
        )
        live_prob = torch.sigmoid(self.live_head(live_in)).squeeze(-1)
        live_prob = live_prob * unit_ok * recover_prob * event_mask.float()

        registers = inst.new_zeros(value.size(0), self.num_types, self.hidden_dim)
        for step in range(value.size(1)):
            write = self.write_proj(
                torch.cat([inst[:, step], value[:, step : step + 1]], dim=-1)
            )
            prev = registers.reshape(-1, self.hidden_dim)
            proposal = self.register_gru(
                write.repeat_interleave(self.num_types, dim=0),
                prev,
            ).view_as(registers)

            gate = live_prob[:, step].unsqueeze(-1) * type_prob[:, step]
            gate = gate.unsqueeze(-1)
            registers = registers * (1.0 - gate) + proposal * gate

        register_mask = (registers.abs().sum(dim=-1) > 0).float()
        patient_state = masked_mean(registers, register_mask, dim=1)
        logits = self.classifier(patient_state)

        # Program trace head is diagnostic; training can detach it from classifier loss.
        program_logits = self.program_head(inst.detach())
        return DSSSOutput(
            logits=logits,
            registers=registers,
            live_prob=live_prob,
            type_prob=type_prob,
            unit_ok=unit_ok,
            recover_prob=recover_prob,
            program_logits=program_logits,
        )


def dsss_losses(
    out: DSSSOutput,
    label: torch.Tensor,
    canonical_type_target: torch.Tensor,
    unit_target: torch.Tensor,
    admin_label: torch.Tensor,
    recoverability_target: torch.Tensor,
    op_code: torch.Tensor,
    event_mask: torch.Tensor,
    split_register: torch.Tensor | None = None,
    pack_register: torch.Tensor | None = None,
    weights: Dict[str, float] | None = None,
) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = {
            "type": 0.5,
            "dce": 1.0,
            "phi": 0.5,
            "recover": 0.5,
            "program": 0.2,
        }

    flat_mask = event_mask.reshape(-1).bool()
    loss_cls = F.cross_entropy(out.logits, label)
    loss_type = F.cross_entropy(
        out.type_prob.reshape(-1, out.type_prob.size(-1))[flat_mask].clamp_min(1e-8).log(),
        canonical_type_target.reshape(-1)[flat_mask],
    )
    loss_unit = F.binary_cross_entropy(
        out.unit_ok[flat_mask],
        unit_target.reshape(-1)[flat_mask].float(),
    )

    admin_weight = admin_label.float() * event_mask.float()
    loss_dce = (admin_weight * out.live_prob).sum() / admin_weight.sum().clamp_min(1.0)

    loss_recover = F.binary_cross_entropy(
        out.recover_prob[flat_mask],
        recoverability_target.reshape(-1)[flat_mask].float(),
    )
    loss_program = F.cross_entropy(
        out.program_logits.reshape(-1, out.program_logits.size(-1))[flat_mask],
        op_code.reshape(-1)[flat_mask],
    )

    if split_register is None or pack_register is None:
        loss_phi = out.logits.new_tensor(0.0)
    else:
        loss_phi = F.smooth_l1_loss(split_register, pack_register)

    total = (
        loss_cls
        + weights["type"] * (loss_type + loss_unit)
        + weights["dce"] * loss_dce
        + weights["recover"] * loss_recover
        + weights["program"] * loss_program
        + weights["phi"] * loss_phi
    )
    return {
        "loss": total,
        "loss_cls": loss_cls,
        "loss_type": loss_type,
        "loss_unit": loss_unit,
        "loss_dce": loss_dce,
        "loss_recover": loss_recover,
        "loss_program": loss_program,
        "loss_phi": loss_phi,
    }
```

## 4. 与当前“采样解耦/反事实干预”框架的结合方式

1. **value process**：负责产生 typed value definition，包括数值、变量语义、单位、质量、局部形态 scale。
2. **sampling process**：不输出类别特征，只输出 observation-program rewrite metadata，例如 rename、unit-cast、panel split、repeat、pending、low-rate。
3. **counterfactual intervention**：从 `do(policy)` 视图生成程序改写，而不是生成一致性 pair。它监督 DCE、phi-node 和 recoverability。
4. **classifier**：只读取 canonical SSA registers。若采样政策改变了程序外壳但没有改变 live definitions，分类边界不会被事件数量、开关次数、panel 同步或通道命名牵引。

## 5. 实验切入点

1. **跨医院 ICU**
   - MIMIC-IV -> eICU / HiRID；
   - 人工构造 alias rename、unit-cast、panel split、pending latency 与 repeat-check policy；
   - 指标：AUROC/AUPRC、worst-policy AUPRC、dead-code precision、type ambiguity、program-shift score。

2. **可穿戴 ECG 采样率偏移**
   - MIT-BIH 360 Hz -> 90 Hz / 45 Hz / 不均匀丢包；
   - 报告少数类 S、F 的 F1 和 recoverability calibration；
   - 检验模型是否在低频条件下虚假放大不可恢复形态。

3. **消融**
   - 去掉 DCE：观察 repeat / panel split 是否放大类别 margin；
   - 去掉 type checker：观察跨变量命名与单位漂移；
   - 去掉 recoverability：观察低采样率 ECG 少数类错误；
   - 将 program trace 接入分类头：验证采样程序 shortcut 会显著提升训练域、损害跨政策。

## 6. 预期创新性

1. **把 sampling-policy shift 表述为观测程序漂移**：不是概率缺失、时间坐标漂移或表示不变问题，而是同一病程被不同数据工程/设备流程编译成不同 typed program。
2. **引入可微编译器机制**：type checking、SSA def-use、phi-node、dead-code elimination 都是此前历史提案未使用的架构与损失语言。
3. **通道语义只做类型系统**：吸收 CHARM，但不把 channel description 直接作为分类增强，而是防止命名/单位/设备策略成为 shortcut。
4. **非插补在线更新**：吸收 SLAN 的 switch 思想，但 switch 是否打开不进入分类器；只有 live semantic definition 才写入 register。
5. **采样率鲁棒不靠频域或 ODE 重构**：吸收 ECG latent ODE 的低采样率观察，但只用 recoverability gate 管理“哪些形态可写入”，不做 latent ODE 分类或频域掩码修复。
6. **反事实干预监督程序语义而非 logits**：避免历史上大量多视图一致性、校准、证明、投票、隐私或记忆写入路线。

## 7. 一句话投稿卖点

**DSSS 把非规则采样鲁棒分类从“如何适配不同采样分布”改写为“如何编译并执行不同观测程序”：通过语义类型检查、SSA liveness 与死代码消除，医院/设备采样策略只能改变程序 trace，不能把 rename、panel、repeat、pending 或低采样率伪装成可迁移的病理分类证据。**
