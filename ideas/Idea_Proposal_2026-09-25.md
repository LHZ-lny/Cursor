# Title: Do-FaCuT Pathology Lens：面向采样策略偏移的阶乘累积量去薄化分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前 `/workspace` 未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md`，并纳入其中记录但当前工作区未完全落盘的历史提案摘要。
- 已读取/抽取当前仓库 `ideas/Idea_Proposal_*.md` 的历史提案标题、黑名单、核心机制段与近期完整正文，覆盖当前工作区已落盘的 2026-06-12 至 2026-09-24 提案。
- 已读取近期 `paper_daily_2026-09-24.md` 与 `paper_daily.md` 入口段落，重点纳入：
  - **ReDiTT**：异步 marked event sequence 可在 latent event-token 空间用 conditional diffusion / flow matching 建模，说明未来观测轨迹本身可以作为非参数结构先验；但 retrieval neighbor composition 也可能放大采样政策邻近性。
  - **DynaMamba**：选择性状态空间模型适合长程不规则临床序列，但 selective memory 的写入/遗忘也可能把 mask、delta-t、panel 共测和告警复测长期保存为 policy trace。

### 历史核心机制黑名单

为避免与历史 proposal 发生思维重合，本轮明确避开以下机制作为主创新：

1. hazard point process、采样 score 零空间、hazard-driven resampling、do-risk variance。
2. 生理流-采样算子交换子、state-policy graph 拆分、policy residual sink。
3. protocol tax、additive evidence market、posterior quotient、Radon-Nikodym density ratio、doubly robust correction。
4. reconstruction error cartography、VQ semantic clauses、HSIC redaction、optional-stopping martingale。
5. topology capsule、censored persistence interval、policy gauge horizontal transport、syndrome code、knockoff calendar。
6. observability witness、evidential vacuity、information lattice、solver trace front-door、conformal sleeve、IV/control-function。
7. Borda jury / social-choice tribunal、Krylov annihilator、Nystrom volume、tropical support route、fixed viva、sequent proof、disease-progress poset clock。
8. IRT-DIF、RG fixed point、bitemporal curtain、clinical tomography、matched risk-set likelihood、policy privacy cloak。
9. CauKer-style orthogonal synthetic forge、dialectical JEPA referee、PID semantic prism、Noether semantic action、meta workflow immunization。
10. collider seal、source-atlas antipodal retrieval、DeepFRC/Schwarzian warp canonicalizer、renewal exposure meter、palimpsest memory。
11. typed SSA observation program、Robin boundary field solver、Blackwell experiment order、elicitable functional compass。
12. Lyapunov contraction observer、random-matrix BBP spike sieve、RIP sparse pathology recovery。
13. 单纯 state-policy 双分支、对抗环境分类器、跨视图 logits/representation 一致性、频域掩码对比学习、missingness pattern 直接分类、普通 retrieval-augmented classifier 或普通 dual-SSM classifier。

本提案选择新的正交切入点：**不把采样政策建成概率密度、图边、实验序、边界条件、记忆写入、谱 bulk 或稀疏测量矩阵；而是把不规则观测流看成 marked point counts 的随机薄化、重复簇与 panel 簇叠加。采样政策主要扭曲 count moments；真正稳定的病理共激活应体现在经 thinning/cluster 代数校正后的高阶阶乘累积量中。分类器只读取去薄化后的 pathology factorial cumulants，而不读取事件数、mask 密度、复测次数、panel 同步标记或 Mamba hidden state。**

---

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样分类中的很多 shortcut 并不是来自某个单独 token，而是来自 **事件计数和共现矩的变形**：

- 告警后重复测量会把同一个临床状态复制成一串相似事件，普通 attention / Mamba 会把“更新次数多”当成风险证据。
- lab panel 同步下单会制造跨变量的零延迟共现，普通局部 channel-dependent 模块可能把流程共测误读成病理耦合。
- 可穿戴设备 duty cycle 或医院 routine round 会对所有事件做近似 independent thinning，使一阶频率、二阶共现和三阶 burst 统计整体缩放。
- 低质量/行政性记录可以近似看作 policy-only Poisson clutter：它增加一阶 count，但不应创造稳定的二阶或三阶病理共激活。

近期 `paper_daily_2026-09-24.md` 提供了两个新的启发：

1. **ReDiTT** 说明异步事件流不必被强行网格化，可以在 latent marked-event 空间建模未来事件分布。本提案借用“marked event latent”的思想，但不使用 retrieval memory，也不让相似轨迹邻居直接参与分类，避免把 policy-neighbor 当成 state-neighbor。
2. **DynaMamba** 说明 selective state-space memory 很适合长程 ICU IMTS；但它也提醒我们，隐藏状态可能长期保留采样政策痕迹。本提案只把选择性扫描用于累积阶乘矩的充分统计，不把 SSM hidden state 直接送入分类头。

阶乘累积量给出一个与历史机制正交的数学接口。对计数变量 `N`：

```text
kappa_1 = E[N]
kappa_2 = E[N(N - 1)] - E[N]^2
kappa_3 = E[N(N - 1)(N - 2)] - 3E[N(N - 1)]E[N] + 2E[N]^3
```

它对采样政策有清晰代数行为：

- **independent thinning**：若观测政策以保留率 `r` 随机留下事件，则第 `k` 阶阶乘累积量按 `r^k` 缩放；去薄化后可以回到同一个 latent cumulant。
- **Poisson clutter**：独立行政噪声只改变一阶累积量，高阶阶乘累积量为零；它不应被分类器当成病理共激活。
- **repeat echo**：重复复测主要制造同变量、短延迟、近对角的簇累积量；可被 echo family 从高阶统计中扣除。
- **panel cluster**：同步 panel 产生零延迟跨变量簇，其 cumulant pattern 与真实滞后病理耦合不同；可以作为 policy cluster 被单独建模。

因此 **Do-FaCuT Pathology Lens** 的核心问题是：

> 当前分类 margin 是否来自经 thinning/cluster 代数校正后仍存在的病理高阶共激活？如果某个信号只是在某医院采样政策下由复测、panel pack 或 duty cycle 产生，它会在阶乘累积量方程中表现为可扣除的 policy cumulant，而不能直接成为类别证据。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 把 `(value, time, variable)` 编成 soft marked events；
- sampling process 只提供 thinning / echo / cluster 的代数系数，不进入分类头；
- counterfactual intervention 生成 policy 编辑后的 count process，用于训练 cumulant 方程，而不是做 logits 一致、检索投票、谱筛选、RIP 恢复或 Lyapunov 收缩。

---

## 2. Methodology: 具体修改点

### 2.1 改 Dataloader：Cumulant Intervention Ledger

新增 `FactorialCumulantCollator`。每个样本返回原始事件流以及一组计数过程编辑账本：

```text
event_i = {
    value_i,
    time_i,
    variable_i,
    quality_i,
    mark_family_i,      # lab/vital/medication/device/event type
    policy_recipe_i,    # factual/repeat_echo/panel_cluster/thin/dropout/clutter
}
```

反事实采样模块不再只返回增强视图，而是返回 **Cumulant Intervention Ledger**：

1. `factual_events`
   - 原始非规则观测。

2. `counterfactual_count_bank`
   - `independent_thin_r`：模拟不同医院或设备 duty cycle 的随机薄化。
   - `repeat_echo`：在同一 value bin 周围插入短延迟复测。
   - `panel_cluster`：把一组变量同步打包为 panel。
   - `panel_decluster`：把同步 panel 拆成异步返回。
   - `poisson_clutter`：加入行政性、低质量或 pending-like 记录。
   - `alarm_burst`：在异常窗口复制局部事件，但不改变底层 value trajectory。

3. `intervention_coefficients`
   - `retention_r`：每个 mark family / time bin 的薄化系数。
   - `echo_kernel_id`：重复复测的短延迟核。
   - `cluster_family_id`：panel 或设备簇类型。
   - 这些系数只参与 cumulant 方程，不作为分类特征。

4. `semantic_mark_target`
   - 可由病理分箱、事件类型、异常阈值、弱标签或高质量片段得到的 soft mark 监督。

关键区别：

- 这些 views 不是 contrastive positives；
- 不要求多 view logits / representation 一致；
- 不估计 hazard、density ratio、Blackwell order、RIP matrix 或 BBP bulk edge；
- 不生成 proof、conformal set、program IR、boundary condition、memory slot 或 retrieval atlas；
- 反事实采样只用于检验 **阶乘累积量在 thinning / cluster 代数下是否闭合**。

### 2.2 改 Encoder：Diffusion-Mamba Mark Stem + Factorial Cumulant Lens

#### A. Soft marked-event stem

给定不规则事件：

```text
e_i = (value_i, time_i, variable_i, quality_i)
```

先输出一组 soft pathology marks：

```text
m_i = sigmoid(MarkStem(value_i, time_i, variable_i, quality_i))
```

这里吸收近期机制但改变用途：

- 从 ReDiTT 借 latent marked-event modeling：可增加一个轻量 denoising mark head，学习在异步事件空间补充 mark uncertainty。
- 不使用 ReDiTT 的 retrieval memory bank，避免检索到同政策邻居。
- 从 DynaMamba 借 selective scan 的线性复杂度，用于长序列 mark aggregation。
- 不把 Mamba hidden state 送入分类器，只把它作为高效累计 window counts / projected marks 的工具。

#### B. Windowed factorial counts

把时间轴划成软窗口或临床阶段锚点，得到每个窗口、每个 mark projection 的 soft count：

```text
C_{w,r} = sum_i window_weight(i, w) * projection_r(m_i)
```

为避免高阶张量爆炸，使用 learnable nonnegative random projections `projection_r` 把 mark family 组合压到 `R` 个 cumulant channel。分类器不会看到原始 event count，而只看到去薄化后的 cumulant signature。

#### C. Factorial cumulant lens

在每个样本内部，把多个时间窗口看作同一病程的局部 replicate，计算投影 count 的一阶、二阶、三阶阶乘累积量：

```text
F1 = E_w[C]
F2 = E_w[C(C - 1)]
F3 = E_w[C(C - 1)(C - 2)]

K1 = F1
K2 = F2 - F1^2
K3 = F3 - 3F2F1 + 2F1^3
```

再用 sampling ledger 做代数校正：

```text
K1_state = (K1_obs - clutter_mean) / r
K2_state = (K2_obs - K2_echo - K2_panel) / r^2
K3_state = (K3_obs - K3_echo - K3_panel) / r^3
```

最终分类器只读取：

```text
z = concat(K1_state, K2_state, K3_state, cumulant_quality)
logits = Classifier(z)
```

其中 `cumulant_quality` 是对 cumulant 方程残差的低维诊断，只用于置信度或拒识；若实验设置要求最严格，可以只把 `K*_state` 输入分类器。

### 2.3 改 Loss：从不变性转向 Factorial-Cumulant Algebra Discipline

总目标：

```text
L = L_cls
  + lambda_mark  * L_mark_diffusion
  + lambda_thin  * L_thinning_algebra
  + lambda_clust * L_cluster_cumulant_subtraction
  + lambda_null  * L_poisson_clutter_null
  + lambda_leak  * L_count_shortcut_leakage
```

#### A. Cumulant Classification `L_cls`

只用去薄化后的病理阶乘累积量分类：

```text
L_cls = CE(Classifier(K_state), y)
```

分类头不接收 raw mask、delta-t histogram、event count、policy recipe、center id、panel id、retrieval id 或 Mamba hidden state。

#### B. Mark Diffusion Calibration `L_mark_diffusion`

借 ReDiTT 的异步 latent event 思想，但不用 retrieval：

```text
noise -> noised soft marks
DenoiseMarkHead(noised marks, times, variables) -> clean marks
```

训练 denoising mark head 预测病理 mark / event type / value bin：

```text
L_mark = CE(mark_logits, mark_target) + SmoothL1(time_delta_hat, time_delta_target)
```

它的作用是让软 mark 在稀疏观测下仍有可用语义，而不是生成未来轨迹或检索邻居。

#### C. Thinning Algebra `L_thinning_algebra`

对 `independent_thin_r` 视图，事实 cumulant 与薄化视图必须满足阶数缩放方程：

```text
K1_thin ≈ r * K1_fact
K2_thin ≈ r^2 * K2_fact
K3_thin ≈ r^3 * K3_fact
```

损失：

```text
L_thin =
  |K1_thin - r K1_fact|
  + |K2_thin - r^2 K2_fact|
  + |K3_thin - r^3 K3_fact|
```

这不是多视图 representation consistency：模型不要求两个视图的 hidden state 或 logits 相同，只要求计数过程的已知 cumulant algebra 被满足。

#### D. Cluster Cumulant Subtraction `L_cluster_cumulant_subtraction`

对 `repeat_echo` 与 `panel_cluster`，学习低秩或模板化 cluster cumulant：

```text
K_policy_cluster = ClusterTemplate(cluster_family_id, echo_kernel_id)
K_state = K_obs - K_policy_cluster
```

约束 cluster 模板只能解释短延迟、近对角、同 panel 的 cumulant mass，不能解释跨窗口滞后病理耦合：

```text
L_clust = off_template_mass(K_policy_cluster) + in_template_residual(K_obs - K_state - K_policy_cluster)
```

#### E. Poisson Clutter Null `L_poisson_clutter_null`

对 `poisson_clutter` 视图，独立行政噪声不应创造高阶 cumulant：

```text
L_null = ||K2_clutter||_1 + ||K3_clutter||_1
```

如果加入 pending / administrative token 后分类 margin 变化很大，而高阶 cumulant 方程显示没有新病理共激活，该信号会被判为 count shortcut。

#### F. Count Shortcut Leakage `L_count_shortcut_leakage`

训练一个冻结梯度的审计 probe 尝试用 raw count / mask density 预测标签：

```text
count_logits = CountProbe(stopgrad(raw_count_features))
L_leak = relu(AUC_proxy(count_logits, y) - tau)^2
```

这不是 adversarial 去偏；分类器从结构上不接收 count features。`L_leak` 只用于报告和惩罚数据切分中过强的 count shortcut，帮助审稿人看到偏移风险。

---

## 3. Code Draft: PyTorch 核心模块草稿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_soft_window(times: torch.Tensor, mask: torch.Tensor, num_windows: int, sharpness: float = 30.0) -> torch.Tensor:
    """Softly assign irregular event times in [0, 1] to temporal windows."""
    centers = torch.linspace(0.0, 1.0, num_windows, device=times.device, dtype=times.dtype)
    dist2 = (times.unsqueeze(-1) - centers.view(1, 1, -1)).pow(2)
    weight = torch.softmax(-sharpness * dist2, dim=-1)
    return weight * mask.unsqueeze(-1).to(times.dtype)


def factorial_cumulants(counts: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    counts: [batch, windows, channels], nonnegative soft counts.
    Returns per-sample window-replicate factorial cumulants.
    """
    counts = counts.clamp_min(0.0)
    f1 = counts.mean(dim=1)
    f2 = (counts * (counts - 1.0)).mean(dim=1)
    f3 = (counts * (counts - 1.0) * (counts - 2.0)).mean(dim=1)

    k1 = f1
    k2 = f2 - f1.pow(2)
    k3 = f3 - 3.0 * f2 * f1 + 2.0 * f1.pow(3)
    return k1, k2, k3


class SoftMarkStem(nn.Module):
    def __init__(self, num_vars: int, hidden_dim: int, num_marks: int):
        super().__init__()
        self.var_emb = nn.Embedding(num_vars, hidden_dim)
        self.value_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.quality_net = nn.Linear(1, hidden_dim)
        self.mark_head = nn.Linear(hidden_dim, num_marks)

    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        var_ids: torch.Tensor,
        quality: torch.Tensor,
    ) -> torch.Tensor:
        value_feat = self.value_net(torch.stack([values, times], dim=-1))
        hidden = value_feat + self.var_emb(var_ids) + self.quality_net(quality.unsqueeze(-1))
        return torch.sigmoid(self.mark_head(hidden))


class SelectiveCumulantScan(nn.Module):
    """
    Lightweight DynaMamba-inspired selective gating for accumulating mark counts.
    The recurrent state is not exposed to the classifier; only factorial cumulants are.
    """
    def __init__(self, num_marks: int, num_proj: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(num_marks, num_proj) / num_marks**0.5)
        self.gate = nn.Sequential(
            nn.Linear(num_marks + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_proj),
            nn.Sigmoid(),
        )

    def forward(
        self,
        marks: torch.Tensor,
        times: torch.Tensor,
        quality: torch.Tensor,
        event_mask: torch.Tensor,
        window_weight: torch.Tensor,
    ) -> torch.Tensor:
        proj = F.softplus(self.proj)
        mark_proj = marks @ proj
        gate_in = torch.cat([marks, times.unsqueeze(-1), quality.unsqueeze(-1)], dim=-1)
        gated = mark_proj * self.gate(gate_in)
        gated = gated * event_mask.unsqueeze(-1).to(gated.dtype)
        return torch.einsum("bnw,bnr->bwr", window_weight, gated)


class PolicyCumulantCorrector(nn.Module):
    def __init__(self, num_proj: int, num_cluster_families: int):
        super().__init__()
        self.clutter_head = nn.Sequential(nn.Linear(num_proj, num_proj), nn.Softplus())
        self.cluster_emb = nn.Embedding(num_cluster_families, 2 * num_proj)

    def forward(
        self,
        raw_counts: torch.Tensor,
        retention: torch.Tensor,
        cluster_family_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        k1, k2, k3 = factorial_cumulants(raw_counts)
        retention = retention.clamp_min(0.05)

        cluster = self.cluster_emb(cluster_family_id)
        k2_cluster, k3_cluster = cluster.chunk(2, dim=-1)
        k2_cluster = k2_cluster.tanh()
        k3_cluster = k3_cluster.tanh()

        clutter = self.clutter_head(k1.detach())
        k1_state = (k1 - clutter).div(retention)
        k2_state = (k2 - k2_cluster).div(retention.pow(2))
        k3_state = (k3 - k3_cluster).div(retention.pow(3))

        signature = torch.cat([k1_state, k2_state, k3_state], dim=-1)
        return {
            "raw_k1": k1,
            "raw_k2": k2,
            "raw_k3": k3,
            "k1_state": k1_state,
            "k2_state": k2_state,
            "k3_state": k3_state,
            "signature": signature,
            "cluster_k2": k2_cluster,
            "cluster_k3": k3_cluster,
            "clutter": clutter,
        }


class DoFaCuTClassifier(nn.Module):
    def __init__(
        self,
        num_vars: int,
        num_marks: int,
        num_proj: int,
        hidden_dim: int,
        num_classes: int,
        num_windows: int = 12,
        num_cluster_families: int = 8,
    ):
        super().__init__()
        self.num_windows = num_windows
        self.mark_stem = SoftMarkStem(num_vars, hidden_dim, num_marks)
        self.scan = SelectiveCumulantScan(num_marks, num_proj, hidden_dim)
        self.corrector = PolicyCumulantCorrector(num_proj, num_cluster_families)
        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * num_proj),
            nn.Linear(3 * num_proj, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.mark_denoiser = nn.Sequential(
            nn.Linear(num_marks + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_marks),
        )

    def encode_view(self, batch: dict) -> dict[str, torch.Tensor]:
        marks = self.mark_stem(
            batch["event_value"],
            batch["event_time"],
            batch["event_var_id"],
            batch["event_quality"],
        )
        window_weight = masked_soft_window(
            batch["event_time"],
            batch["event_mask"],
            num_windows=self.num_windows,
        )
        counts = self.scan(
            marks,
            batch["event_time"],
            batch["event_quality"],
            batch["event_mask"],
            window_weight,
        )
        corrected = self.corrector(
            counts,
            batch["retention_factor"],
            batch["cluster_family_id"],
        )
        logits = self.classifier(corrected["signature"])
        corrected.update({"marks": marks, "counts": counts, "logits": logits})
        return corrected

    def mark_diffusion_loss(self, marks: torch.Tensor, batch: dict) -> torch.Tensor:
        noise = torch.randn_like(marks)
        sigma = torch.rand(marks.size(0), marks.size(1), 1, device=marks.device, dtype=marks.dtype)
        noised = (1.0 - sigma) * marks + sigma * noise
        denoise_in = torch.cat([noised, batch["event_time"].unsqueeze(-1), batch["event_quality"].unsqueeze(-1)], dim=-1)
        pred = self.mark_denoiser(denoise_in)
        target = batch["mark_target"].to(pred.dtype)
        mask = batch["event_mask"].unsqueeze(-1).to(pred.dtype)
        return F.binary_cross_entropy_with_logits(pred, target, reduction="none").mul(mask).sum() / mask.sum().clamp_min(1.0)

    def training_loss(self, batch: dict, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        factual = self.encode_view(batch)
        cls_loss = F.cross_entropy(factual["logits"], labels)
        mark_loss = self.mark_diffusion_loss(factual["marks"], batch)

        thin_losses = []
        null_losses = []
        cluster_losses = []
        for view in batch.get("counterfactual_count_bank", []):
            out = self.encode_view(view)
            r = view["retention_factor"].clamp_min(0.05)
            thin_losses.append(
                F.smooth_l1_loss(out["raw_k1"], r * factual["raw_k1"].detach())
                + F.smooth_l1_loss(out["raw_k2"], r.pow(2) * factual["raw_k2"].detach())
                + F.smooth_l1_loss(out["raw_k3"], r.pow(3) * factual["raw_k3"].detach())
            )
            if view.get("recipe", "") == "poisson_clutter":
                null_losses.append(out["raw_k2"].abs().mean() + out["raw_k3"].abs().mean())
            if view.get("recipe", "") in {"repeat_echo", "panel_cluster"}:
                cluster_losses.append(
                    (out["raw_k2"] - out["cluster_k2"]).abs().mean()
                    + (out["raw_k3"] - out["cluster_k3"]).abs().mean()
                )

        zero = cls_loss.new_zeros(())
        thin_loss = torch.stack(thin_losses).mean() if thin_losses else zero
        null_loss = torch.stack(null_losses).mean() if null_losses else zero
        cluster_loss = torch.stack(cluster_losses).mean() if cluster_losses else zero

        total = (
            cls_loss
            + 0.10 * mark_loss
            + 0.20 * thin_loss
            + 0.10 * cluster_loss
            + 0.10 * null_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "mark_diffusion_loss": mark_loss.detach(),
            "thinning_algebra_loss": thin_loss.detach(),
            "cluster_cumulant_loss": cluster_loss.detach(),
            "poisson_null_loss": null_loss.detach(),
        }
```

---

## 4. 实验切入点

1. **数据集**
   - ICU IMTS：P12、P19、MIMIC-III / MIMIC-IV、eICU。
   - 异步事件流：next-event 数据可先做 event-type 或风险标签分类，验证 ReDiTT 场景下的 marked-event cumulant。
   - 可穿戴/ECG：高采样率到低采样率、duty-cycle、dropout、repeat-trigger 采样策略。

2. **偏移协议**
   - `routine -> alarm burst`：训练中风险窗口密集复测，测试中复测规则改变。
   - `panel pack -> panel split`：训练中变量同步联测，测试中异步返回。
   - `high duty -> low duty`：可穿戴设备不同采样预算。
   - `clean -> administrative clutter`：加入 pending、低质量或行政记录。

3. **核心指标**
   - in-policy AUROC / AUPRC。
   - cross-policy AUROC / AUPRC。
   - cumulant algebra residual：`K_thin - r^k K_fact`。
   - policy-created cumulant mass：repeat / panel / clutter 造成的高阶 cumulant 比例。
   - count shortcut audit：raw event count probe 与主分类性能之间的差距。
   - minority-class robustness：特别报告低采样率下少数类下降，呼应 ECG latent ODE 的风险。

4. **消融**
   - 只用 Mamba hidden state 分类。
   - 只用一阶 count。
   - 去掉 thinning algebra。
   - 去掉 cluster subtraction。
   - 去掉 mark diffusion calibration。
   - 用普通 multi-view consistency 替代 cumulant algebra。

---

## 5. 预期创新性

1. **机制正交**：历史 proposal 多数把 sampling shift 表述为概率、图、拓扑、证明、谱、边界、记忆、实验序或恢复矩阵；Do-FaCuT 把问题落到 marked point counts 的阶乘累积量代数，是新的统计结构。
2. **不是多视图一致性**：反事实采样只需满足 thinning / cluster cumulant 方程，不强迫 logits、hidden state 或 embedding 相同。
3. **天然解释 repeat / panel / clutter**：复测、panel 同步、行政噪声分别对应 echo cumulant、cluster cumulant 与 Poisson null，而不是混在一个 policy embedding 里。
4. **吸收 ReDiTT 但避开 retrieval**：使用 latent mark denoising 提升异步事件语义，不把检索邻居作为分类证据。
5. **吸收 DynaMamba 但避开 hidden shortcut**：选择性扫描只作为高效统计累积器，最终分类由去薄化 cumulant signature 完成。

## 6. 一句话投稿卖点

**Do-FaCuT 将非规则采样偏移从“何时测了什么”重写为 marked event count process 的 thinning/cluster 代数问题：分类器只读取经阶乘累积量去薄化后仍存在的病理共激活，从而让复测次数、panel 同步和行政 clutter 无法伪装成稳定类别证据。**
