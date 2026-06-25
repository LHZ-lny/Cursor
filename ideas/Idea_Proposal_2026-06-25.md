# Title: Sampling-Channel Syndrome Codes：面向采样策略偏移的采样信道综合征纠错分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已尝试读取 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
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
- 已纳入自动化记忆中记录但当前工作区未检出的历史提案核心机制：
  - `Idea_Proposal_2026-06-17.md`
  - `Idea_Proposal_2026-06-20.md`
  - `Idea_Proposal_2026-06-24.md`

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
22. policy-only negative film、dual-exposure value/shadow encoders、latent shadow eraser/stencil、shadow nullification high-entropy loss、value retention buckets。
23. 单纯复用 FlowPath 的可逆路径、GSNF/DBGL/GARLIC 的图衰减结构、iTimER 的误差伪观测/Wasserstein 对齐、Record2Vec 的 summarize-then-embed、QuITE 的普通 query token、MTM 的普通 pivotal token mixing，或 MedMamba 的 frequency/adaptive graph branch 作为主机制。

本提案选择新的正交切入点：**不估计采样概率，不删除采样信息，不做多视图一致性，不做图/后验/拓扑/gauge 分解，而是把采样策略偏移看成一个作用在潜在状态包上的结构化通信信道错误。模型学习一组可微 parity-check constraints，把跨策略稳定的状态表示约束为可纠错 codeword；采样分支只输出 syndrome，用于定位 erasure、delay、burst 和 panel split 等采样信道错误，分类器只消费纠错后的 codeword。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

非规则采样时间序列中的 sampling-policy shift 很像通信系统中的信道变化：

- 训练医院在报警后密集复测，相当于 burst duplication channel；
- 测试医院只保留轮班时刻，相当于 structured erasure channel；
- 某些实验室 panel 被同步下单或拆开下单，相当于 packet batching / packet splitting channel；
- 可穿戴设备因电量策略延迟采样，相当于 packet delay channel。

这些变化会改变哪些事件包被看见、何时到达、是否成组到达，却不应改变底层状态本身。历史提案大多把采样政策当作要估计、要剥离、要征税、要平滑或要解释的 nuisance；本提案换一个角度：

> 如果采样策略是信道错误，那么鲁棒分类器不应直接追问“这个错误来自哪个医院协议”，而应学习一种状态码，使得有限的采样错误可以被 syndrome 定位并被 decoder 修复。

近期 `paper_daily.md` 中的机制给了三个关键启发：

1. **MTM** 指出 channel-wise asynchrony 会让变量间信息交换失效；SCSC 不选择 pivotal tokens，而是把异步观测打包成 latent packets，再用 parity checks 让缺失或异步的 packet 可被纠错。
2. **MedMamba** 说明 SSM 适合医疗长序列和非平稳多通道数据；SCSC 使用轻量 SSM-style packet encoder，但不使用其 frequency branch 或 adaptive graph。
3. **PYRREGULAR** 强调真实不规则数据需要统一评测接口；SCSC 可在该接口上显式定义 erasure、delay、burst、panel split 四类采样信道偏移，汇报 correction-before-classification 的鲁棒性。

这与当前“采样解耦/反事实干预”框架天然兼容：

- value process 生成底层状态的 latent packets；
- sampling process 不进入分类头，只预测采样信道 syndrome；
- counterfactual intervention 不构造一致性 pair，而是构造已知信道错误，用于训练 decoder 的纠错能力；
- classifier 只读取纠错后的 codeword，因此不直接利用训练环境中特定的采样密度、联测模式或复测 burst。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从事件表征改为 Latent Packet Codeword

给定事件流 `(t_i, d_i, x_i)`，先按变量组与粗时间窗生成 `K` 个 latent packets。每个 packet 汇总一小段 value-driven 状态证据，但不把 mask pattern 或 policy id 输入分类头：

```text
p_k = PacketSSM({x_i, d_i, delta_t_i}_{i in bin k})
P = [p_1, ..., p_K] in R^{K x H}
```

然后引入可学习 parity-check matrix `H_code`，把稳定状态表示约束到一个低 syndrome 的 code manifold：

```text
syndrome(P) = H_code P
```

直觉上，真实状态 packets 应该满足某些跨变量/跨时间冗余关系；采样政策改变会造成 packet erasure、delay、duplication 或 split，使 parity residual 出现可定位的 syndrome。与 graph、topology、gauge 的区别是：这里不解释变量关系、拓扑持久性或坐标规约，而是学习“哪些 latent packets 在采样信道错误下仍可被冗余校验修复”。

### 2.2 改 Sampling Branch：从策略表征改为 Syndrome Locator

sampling branch 只看观测结构和 counterfactual channel recipe，输出一个 syndrome estimate：

```text
s_hat = SyndromeLocator(times, mask, channel_recipe)
```

它不进入 classifier，也不作为 residual sink。它的作用类似通信接收端的错误定位器：

- `structured erasure`：哪些 packet 因变量预算或时间窗 blackout 不可靠；
- `delay`：哪些 packet 的到达时间被系统性偏移；
- `burst duplication`：哪些 packet 因报警复测被重复强化；
- `panel split / merge`：哪些 packet 因联测或拆单改变了同步关系。

### 2.3 改 Decoder：Syndrome-Guided Packet Repair

decoder 接收 corrupted packets 与 syndrome，输出修复后的 codeword：

```text
P_repair = RepairDecoder(P_corrupt, s_hat)
logits = Classifier(pool(P_repair))
```

分类器永远只接收 `P_repair`，而不是 raw mask、policy code 或 syndrome 本身。这样保留了 informative observation 的价值：若某个观测值真的提供状态证据，它会通过 packet codeword 被保留；若某个信号主要来自采样流程，它更可能表现为 syndrome 而被 decoder 当作信道错误处理。

### 2.4 改 Loss：从一致性/去偏转向纠错码约束

总目标：

```text
L = L_cls
  + lambda_parity * L_codeword_closure
  + lambda_syn    * L_syndrome_target
  + lambda_repair * L_packet_repair
  + lambda_dist   * L_syndrome_disentangle
```

#### A. 分类损失 `L_cls`

只用事实观测下的修复 codeword 做分类：

```text
L_cls = CE(Classifier(P_repair_factual), y)
```

#### B. Codeword Closure `L_codeword_closure`

修复后的 packet 表示应回到低 syndrome 的 code manifold：

```text
L_codeword_closure = || H_code P_repair ||_2^2
```

它不要求反事实视图的表示相同，也不要求 logits 一致；只要求 decoder 输出是一个满足冗余校验的状态码。

#### C. Syndrome Target `L_syndrome_target`

counterfactual intervention 生成已知采样信道错误。对事实 packets `P` 与 corrupted packets `P_c`，可计算真实 parity residual：

```text
s_target = stopgrad(H_code P_c - H_code P)
L_syndrome_target = SmoothL1(s_hat, s_target)
```

这让 sampling branch 学会定位“采样信道如何损坏了状态包”，但 syndrome 不参与分类。

#### D. Packet Repair `L_packet_repair`

只在被 counterfactual channel 明确破坏的 packet 上训练修复：

```text
L_packet_repair = || M_corrupt * (P_repair - stopgrad(P_clean)) ||_1
```

这不是跨策略 representation consistency，因为未损坏 packet 不被强行对齐，logits 也不被要求相同；目标是局部纠错，而不是全局不变。

#### E. Syndrome Disentangle `L_syndrome_disentangle`

为了防止 decoder 把所有状态信息塞进 syndrome，加入一个带宽限制：

```text
L_syndrome_disentangle = mean(||s_hat||_1) + relu(I_proxy(s_hat, label) - epsilon)^2
```

其中 `I_proxy` 用一个冻结小 probe 的可预测性近似实现，只用于限制 syndrome 携带标签信息。它不同于 environment adversarial：probe 不预测医院/策略标签，也不训练分类表示去欺骗；它只限制错误定位信号成为类别捷径。

### 2.5 改 Dataloader：返回 Sampling Channel Bank

新增 `SamplingChannelCollator`，每个 batch 返回：

1. `event_value`、`event_time`、`event_var_id`、`event_mask`。
2. `packet_bin_id`：事件属于哪个 latent packet。
3. `channel_recipe_bank`：反事实采样信道错误：
   - `time_window_erasure`：早期/晚期窗口结构化丢包；
   - `variable_budget_erasure`：某变量组只保留前 `k` 个观测；
   - `burst_duplication`：报警后复测包重复；
   - `panel_split_merge`：同步 panel 被拆开或异步事件被合并。
4. `corrupt_packet_mask`：哪些 packet 被该信道错误影响。
5. `corrupted_event_view`：只用于训练 decoder，不用于分类一致性约束。

### 2.6 与当前“采样解耦/反事实干预”框架的结合方式

- 现有 value encoder 改为 `PacketSSMEncoder`：输出 latent packet codeword。
- 现有 sampling branch 改为 `SyndromeLocator`：定位采样信道错误，不进入分类头。
- 现有 counterfactual intervention 改为 `SamplingChannelBank`：生成可解释的 channel corruptions 和 corrupt packet mask。
- 推理阶段无需知道测试策略标签；可直接用观测结构估计 syndrome 并修复 packets。
- 可解释输出包括：
  - packet-level syndrome norm；
  - repair magnitude；
  - parity residual before/after repair；
  - 哪些变量/时间窗更像采样信道错误而非状态证据。

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


class PacketSSMEncoder(nn.Module):
    """Encode irregular events into value-driven latent packets."""

    def __init__(self, num_vars: int, num_packets: int, hidden_dim: int):
        super().__init__()
        self.num_packets = num_packets
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        # event_*: [B, N], packet_bin_id: [B, N] in [0, K)
        value = batch["event_value"]
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        event_mask = batch["event_mask"]
        packet_id = batch["packet_bin_id"]

        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
        var_h = self.var_embed(var_id)
        event_x = torch.cat([var_h, value.unsqueeze(-1), torch.log1p(delta_t).unsqueeze(-1)], dim=-1)
        event_h = self.event_proj(event_x) * event_mask.unsqueeze(-1)

        bsz, _, hidden_dim = event_h.shape
        packets = torch.zeros(bsz, self.num_packets, hidden_dim, device=value.device, dtype=event_h.dtype)
        for idx in range(event_h.size(1)):
            pid = packet_id[:, idx].clamp(0, self.num_packets - 1)
            old = packets[torch.arange(bsz, device=value.device), pid]
            new = self.update(event_h[:, idx], old)
            active = event_mask[:, idx : idx + 1] > 0
            packets[torch.arange(bsz, device=value.device), pid] = torch.where(active, new, old)
        return packets


class LearnableParityCheck(nn.Module):
    """Compute differentiable syndrome H_code P over latent packets."""

    def __init__(self, num_packets: int, num_checks: int):
        super().__init__()
        self.raw_check = nn.Parameter(torch.randn(num_checks, num_packets) * 0.02)

    def matrix(self) -> torch.Tensor:
        # Row-normalized parity checks keep syndrome scale stable.
        return F.normalize(self.raw_check, p=2, dim=-1)

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        # packets: [B, K, H], syndrome: [B, C, H]
        return torch.einsum("ck,bkh->bch", self.matrix(), packets)


class SyndromeLocator(nn.Module):
    """Locate sampling-channel errors from observation structure only."""

    def __init__(self, num_vars: int, recipe_dim: int, num_checks: int, hidden_dim: int):
        super().__init__()
        self.num_vars = num_vars
        self.recipe_proj = nn.Linear(recipe_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(num_vars + hidden_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_checks),
        )

    def forward(self, batch: dict, recipe: torch.Tensor) -> torch.Tensor:
        # Return [B, C, 1]; broadcast across hidden dimensions in the repair decoder.
        time = batch["event_time"]
        mask = batch["event_mask"]
        var_id = batch["event_var_id"]

        horizon = time.amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        early = (time_norm <= 0.33).to(time.dtype)
        late = (time_norm >= 0.66).to(time.dtype)

        var_obs = F.one_hot(var_id.clamp_min(0), self.num_vars).to(time.dtype) * mask.unsqueeze(-1)
        var_rate = var_obs.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        stats = torch.cat(
            [
                mask.mean(dim=1, keepdim=True),
                (early * mask).mean(dim=1, keepdim=True),
                (late * mask).mean(dim=1, keepdim=True),
                time_norm.diff(dim=1, prepend=time_norm[:, :1]).abs().mean(dim=1, keepdim=True),
                mask.sum(dim=1, keepdim=True) / mask.size(1),
            ],
            dim=-1,
        )
        recipe_h = self.recipe_proj(recipe)
        loc = self.net(torch.cat([var_rate, recipe_h, stats], dim=-1))
        return loc.unsqueeze(-1)


class SyndromeRepairDecoder(nn.Module):
    """Repair corrupted packets using parity syndrome."""

    def __init__(self, hidden_dim: int, num_checks: int):
        super().__init__()
        self.syndrome_proj = nn.Linear(num_checks, hidden_dim)
        self.repair_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.repair_step = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, packets: torch.Tensor, syndrome: torch.Tensor) -> torch.Tensor:
        # packets: [B, K, H], syndrome: [B, C, H] or [B, C, 1]
        if syndrome.size(-1) == 1:
            syndrome_summary = syndrome.squeeze(-1)
        else:
            syndrome_summary = syndrome.mean(dim=-1)
        syn_h = self.syndrome_proj(syndrome_summary)[:, None, :].expand_as(packets)
        gate = self.repair_gate(torch.cat([packets, syn_h], dim=-1))
        delta = self.repair_step(torch.cat([packets, syn_h], dim=-1))
        return packets + gate * delta


class SamplingChannelSyndromeClassifier(nn.Module):
    """Sampling-policy robust classifier based on latent syndrome correction."""

    def __init__(
        self,
        num_vars: int,
        num_packets: int,
        num_checks: int,
        recipe_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = PacketSSMEncoder(num_vars, num_packets, hidden_dim)
        self.parity = LearnableParityCheck(num_packets, num_checks)
        self.locator = SyndromeLocator(num_vars, recipe_dim, num_checks, hidden_dim)
        self.repair = SyndromeRepairDecoder(hidden_dim, num_checks)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.label_probe = nn.Linear(num_checks, num_classes)

    def classify_packets(self, packets: torch.Tensor, syndrome: torch.Tensor) -> dict:
        repaired = self.repair(packets, syndrome)
        pooled = repaired.mean(dim=1)
        logits = self.classifier(pooled)
        return {"logits": logits, "repaired": repaired}

    def forward(self, batch: dict) -> dict:
        packets = self.encoder(batch)
        zero_recipe = torch.zeros(
            packets.size(0),
            batch["channel_recipe_bank"].size(-1),
            device=packets.device,
            dtype=packets.dtype,
        )
        syndrome = self.locator(batch, zero_recipe)
        out = self.classify_packets(packets, syndrome)
        out.update({"packets": packets, "syndrome": syndrome})
        return out

    def training_loss(
        self,
        batch: dict,
        lambda_parity: float = 0.2,
        lambda_syn: float = 0.2,
        lambda_repair: float = 0.3,
        lambda_dist: float = 0.02,
    ) -> dict:
        labels = batch["labels"]
        factual = self.forward(batch)
        cls_loss = F.cross_entropy(factual["logits"], labels)

        # Repaired factual packets should sit near the learned code manifold.
        parity_loss = self.parity(factual["repaired"]).pow(2).mean()

        syndrome_losses = []
        repair_losses = []
        for recipe, corrupt_batch, corrupt_mask in zip(
            batch["channel_recipe_bank"].unbind(dim=1),
            batch["corrupted_event_bank"],
            batch["corrupt_packet_mask"].unbind(dim=1),
        ):
            clean_packets = factual["packets"].detach()
            corrupt_packets = self.encoder(corrupt_batch)

            observed_syndrome = self.parity(corrupt_packets) - self.parity(clean_packets)
            predicted_syndrome = self.locator(corrupt_batch, recipe)
            if predicted_syndrome.size(-1) == 1:
                predicted_syndrome = predicted_syndrome.expand_as(observed_syndrome)
            syndrome_losses.append(F.smooth_l1_loss(predicted_syndrome, observed_syndrome.detach()))

            repaired = self.repair(corrupt_packets, predicted_syndrome)
            packet_weight = corrupt_mask.to(repaired.dtype).unsqueeze(-1)
            repair_raw = (repaired - clean_packets).abs() * packet_weight
            repair_losses.append(repair_raw.sum() / packet_weight.sum().clamp_min(1.0))

        syndrome_loss = torch.stack(syndrome_losses).mean()
        repair_loss = torch.stack(repair_losses).mean()

        # Keep syndrome as low-bandwidth error-location metadata, not label evidence.
        syndrome_summary = factual["syndrome"].squeeze(-1)
        probe_logits = self.label_probe(syndrome_summary.detach())
        probe_prob = F.softmax(probe_logits, dim=-1)
        label_signal = probe_prob.gather(1, labels[:, None]).mean()
        bandwidth = syndrome_summary.abs().mean()
        disentangle_loss = bandwidth + F.relu(label_signal - 0.35).pow(2)

        total = (
            cls_loss
            + lambda_parity * parity_loss
            + lambda_syn * syndrome_loss
            + lambda_repair * repair_loss
            + lambda_dist * disentangle_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "parity_loss": parity_loss.detach(),
            "syndrome_loss": syndrome_loss.detach(),
            "repair_loss": repair_loss.detach(),
            "disentangle_loss": disentangle_loss.detach(),
        }


@torch.no_grad()
def build_sampling_channel_bank(
    event_value: torch.Tensor,
    event_time: torch.Tensor,
    event_var_id: torch.Tensor,
    event_mask: torch.Tensor,
    packet_bin_id: torch.Tensor,
    num_packets: int,
) -> tuple[torch.Tensor, list[dict], torch.Tensor]:
    """Create channel-corrupted event views and packet corruption masks."""

    bsz, num_events = event_value.shape
    device = event_value.device
    recipes = []
    corrupted_batches = []
    corrupt_packet_masks = []

    def make_batch(value, time, var_id, mask):
        return {
            "event_value": value,
            "event_time": time,
            "event_var_id": var_id,
            "event_mask": mask,
            "packet_bin_id": packet_bin_id,
            "channel_recipe_bank": torch.zeros(bsz, 1, 4, device=device),
        }

    # 1. Time-window erasure: remove late events.
    horizon = event_time.amax(dim=1, keepdim=True).clamp_min(1e-6)
    late = (event_time / horizon >= 0.66).to(event_mask.dtype)
    keep = event_mask * (1.0 - late)
    recipes.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(bsz, -1))
    corrupted_batches.append(make_batch(event_value * keep, event_time, event_var_id, keep))
    corrupt_packet_masks.append(_packet_corruption_mask(packet_bin_id, late * event_mask, num_packets))

    # 2. Variable-budget erasure: keep at most two events per variable.
    budget_keep = torch.zeros_like(event_mask)
    for var_idx in torch.unique(event_var_id[event_mask > 0]).tolist():
        var_event = ((event_var_id == int(var_idx)) & (event_mask > 0)).to(event_mask.dtype)
        rank = var_event.cumsum(dim=1)
        budget_keep = torch.where((var_event > 0) & (rank <= 2), torch.ones_like(budget_keep), budget_keep)
    recipes.append(torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(bsz, -1))
    corrupted_batches.append(make_batch(event_value * budget_keep, event_time, event_var_id, budget_keep))
    corrupt_packet_masks.append(_packet_corruption_mask(packet_bin_id, (event_mask - budget_keep).clamp_min(0.0), num_packets))

    # 3. Burst duplication approximation: amplify alternating event values.
    alternating = ((torch.arange(num_events, device=device)[None, :] % 2) == 0).to(event_mask.dtype)
    burst_value = event_value * (1.0 + 0.25 * alternating * event_mask)
    recipes.append(torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).expand(bsz, -1))
    corrupted_batches.append(make_batch(burst_value, event_time, event_var_id, event_mask))
    corrupt_packet_masks.append(_packet_corruption_mask(packet_bin_id, alternating * event_mask, num_packets))

    # 4. Panel split/merge: snap times to coarse bins, changing synchrony but not values.
    coarse_time = torch.round(event_time / horizon * 8.0) / 8.0 * horizon
    recipes.append(torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(bsz, -1))
    corrupted_batches.append(make_batch(event_value, coarse_time, event_var_id, event_mask))
    corrupt_packet_masks.append(_packet_corruption_mask(packet_bin_id, event_mask, num_packets))

    return torch.stack(recipes, dim=1), corrupted_batches, torch.stack(corrupt_packet_masks, dim=1)


def _packet_corruption_mask(packet_bin_id: torch.Tensor, event_corrupt: torch.Tensor, num_packets: int) -> torch.Tensor:
    bsz = packet_bin_id.size(0)
    out = torch.zeros(bsz, num_packets, device=packet_bin_id.device, dtype=event_corrupt.dtype)
    for packet_idx in range(num_packets):
        hit = ((packet_bin_id == packet_idx).to(event_corrupt.dtype) * event_corrupt).amax(dim=1)
        out[:, packet_idx] = hit
    return out
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `time-window erasure`：训练环境晚期窗口密集，测试环境晚期 blackout。
   - `variable-budget erasure`：某些变量组在测试策略中只保留前 `k` 个观测。
   - `burst duplication shift`：训练医院报警后密集复测，测试医院复测稀疏或反之。
   - `panel split/merge shift`：训练环境同步 panel，测试环境拆成异步事件。

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
   - MTM token mixing 与 MedMamba SSM 主干。

3. **核心指标**
   - in-policy accuracy。
   - worst-policy accuracy。
   - parity residual before/after repair。
   - syndrome localization AUC：已知 corrupt packet 是否被 syndrome 成功定位。
   - repair reliance ratio：错误预测是否伴随异常大的 repair magnitude。
   - channel-shift calibration error：高置信预测是否同时有低 parity residual。

4. **消融实验**
   - 去掉 `L_codeword_closure`，验证 latent packets 是否失去可纠错冗余。
   - 去掉 `L_syndrome_target`，验证 syndrome 是否退化成无意义噪声。
   - 去掉 `L_packet_repair`，验证鲁棒性是否只来自额外参数量。
   - 将 channel bank 替换为随机 mask，验证收益来自结构化采样信道建模。
   - 扫描 parity check 数量，验证冗余度与分类信息保留的 trade-off。

## 5. 预期创新性

1. **从采样去偏转向采样纠错**：不再试图删除、正交化、征税或平滑采样信息，而是把采样政策偏移定义为结构化信道错误，并用 syndrome 定位修复。
2. **从 token mixing 转向 packet coding**：吸收 MTM 对 channel-wise asynchrony 的洞察，但不选择 pivotal tokens；变量异步被当作 packet corruption 处理。
3. **从 SSM 表征转向 codeword 表征**：吸收 MedMamba/SSM 的高效长序列编码优势，但核心创新是 parity-constrained latent packets，而不是 frequency branch 或 adaptive graph。
4. **从反事实增强转向信道错误监督**：counterfactual intervention 只产生可解释 corruptions 和 corrupt packet masks，用于 syndrome learning 与 repair learning，不要求多视图 logits 或 representation 一致。
5. **部署解释性清晰**：parity residual 与 repair magnitude 可以直接提示模型是否正在处理采样信道错误，还是仍然把采样流程当成类别证据。

## 6. 一句话投稿卖点

**SCSC 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“潜在状态包经过结构化采样信道后发生 erasure/delay/burst/panel-split 错误”的问题，并通过可微 parity-check、syndrome locator 与 packet repair decoder，让分类器依赖纠错后的状态 codeword，而不是依赖训练环境中特定的采样密度、同步窗口或复测协议。**
