# Title: Do-Calendar Viva Examiner：面向采样策略偏移的反事实临床口试日历分类器

## 0. 强制读取记录与思维黑名单

### 已读取材料

- 已搜索 `my_work_summary.md`：当前工作区未检出该文件。
- 已搜索 `*work*summary*.md`、`*summary*.md`、`*Summary*.md`、`*work*.md` 与中文 `*总结*.md`：当前工作区未发现可替代工作总结文件。
- 已读取自动化记忆 `MEMORIES.md`，并读取记忆中的历史提案摘要：
  - `idea_2026-07-24.md`
  - `idea_2026-07-25.md`
  - `idea_2026-07-26.md`
  - `idea_2026-07-27.md`
  - `idea_2026-07-29.md`
  - `idea_2026-07-30.md`
  - `idea_2026-07-31.md`
  - `idea_2026-08-01.md`
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
- 已读取近期论文记录：
  - `paper_daily.md`
  - 并检出 `paper_daily_2026-06-12.md`、`paper_daily_2026-06-25.md`、`paper_daily_2026-06-26.md`、`paper_daily_2026-07-13.md`、`paper_daily_2026-07-19.md`、`paper_daily_2026-07-26.md`、`paper_daily_2026-07-27.md`、`paper_daily_2026-08-02.md`。

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
11. reconstruction error cartography、ANOVA projection、VQ semantic clauses、HSIC redaction。
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
24. RKHS cubature debiasing、measurement-action bisimulation、policy-word signature renormalization、thermodynamic free-energy、Sklar copula stripping、triage queue debt、Sinkhorn detail canonicalization、MDL episode transducer、causal sheaf gluing、trigger hysteresis、control barrier certificates、regret escrow、principal-stratum status compiler。
25. counterfactual conformal risk sleeves、policy-conditional nonconformity set、leave-policy-out conformal calibration。
26. counterfactual sampling instruments、first-stage structural equation、control-function residualized structure readout、weak-instrument guard。
27. differentiable Borda / pairwise-majority policy jury、no-dictator、structural bribery、minority report。
28. Krylov policy mode subspace、annihilating polynomial、policy-mode energy filtering。
29. determinantal state-volume / Nystrom event basis、policy-overload logdet sobriety。
30. tropical support routes、soft-min bottleneck、route survival / fragility losses。
31. 单纯把 TCF 的 future-event likelihood 拆成 state/policy 双头并做跨策略一致性。
32. 单纯把 PULSE 的 LLM prompt / reasoning workflow 当作自然语言摘要分类器。

本提案选择新的正交切入点：

> **不删除采样信息、不做保形集合、不做 rank tribunal、不做图/路径/后验/格/纠错分解，也不要求多采样视图 logits 一致；而是把每条不规则事件流翻译成一场固定题目的“临床口试”。分类器不能直接看原始采样日历，只能看模型对同一组 policy-neutral 时间问题的 pathology-bin 答案。采样政策可以改变病历来源的缺页方式，但不能改变考官问什么问题。**

## 1. Motivation: 为什么这个结合能解决采样偏移问题

`paper_daily.md` 的最新追加记录中，两个前沿信号尤其适合深化当前“采样解耦/反事实干预”框架：

- **PULSE** 把 ICU time-series classification 放进跨中心 benchmark，显示 HiRID、MIMIC-IV、eICU 等数据源的采样频率、变量定义、护理流程和记录习惯会显著改变模型表现。它提醒我们：sampling-policy shift 不应只在单数据集随机 mask 下测试，而要面向“换医院后同一临床问题如何被重新记录”的场景。
- **Time-Conditioned Foreseeing (TCF)** 针对 EHR event streams 引入 Pathology-Focused Binning、Dual-Calendar RoPE 和 Time-Conditioned Foreseeing objective。它的关键价值不是普通未来预测，而是把 EHR 表征训练成能回答“给定某个临床时间点，未来会出现什么事件/数值语义”的模型。

这些机制暴露了一个新问题：

> 当前多数 robust IMTS 方法仍让分类器直接消费原始观测日历的 representation。即使做了反事实采样、policy branch 或不确定性校准，最终分类边界仍可能绑定“这家医院如何记录病人”。如果测试医院换了记录策略，模型等于拿到另一种考卷格式。

**Do-Calendar Viva Examiner (DCVE)** 的核心思想是把任务改写为“固定临床口试”：

1. 预先定义一个 **policy-neutral viva question bank**：例如在入院后 2h/6h/12h/24h，询问 lactate、WBC、MAP、creatinine 等变量处于哪个 pathology bin；也可以询问某变量组是否进入异常区间、是否改善、是否恶化。
2. 不规则事件 encoder 只负责从事实病历或反事实采样病历中回答这些标准问题。
3. 最终分类器只读取口试答案矩阵 `Answer[question, pathology_bin]` 与答案可信度，而不读取原始 mask、采样密度、变量共现或医院日历。
4. 反事实采样模块不再制造一致性 pair，也不做校准集合；它只生成不同“病历缺页版本”，迫使答题器在同一套口试题上学习从不完整记录中抽取可迁移病程语义。

直觉上，采样策略偏移像“病历记录格式变了”，而 DCVE 要求模型先把任意格式病历翻译成标准化问答表，再做分类：

```text
irregular source chart under policy p
    -> answer fixed viva questions on canonical clinical time
    -> classify from answer sheet
```

因此，训练医院中特有的复测 burst、panel 共现、value-pending 事件、日历密度不再能直接进入分类器。它们只能通过“是否帮助回答固定临床问题”间接发挥作用。

## 2. Methodology: 具体修改点

### 2.1 改 Encoder：从原始事件表示改为 Canonical Viva Answer Sheet

基础 encoder 可以是 event Transformer、STAR-Set、TCF-style EHR Transformer、mTAND 或任意不规则事件主干。DCVE 在其上增加三层：

#### A. Pathology-Focused Soft Binner

借鉴 TCF 的 Pathology-Focused Binning，但不是为了 next-event generation，而是为了把不同医院的数值尺度转成更稳定的临床语义单位：

```text
value x_{i,d} -> soft bin distribution b_{i,d} in {low, normal, mild-high, critical-high, ...}
```

每个变量有一组可学习但单调递增的 cutpoints；可用临床先验初始化。连续值不会被粗暴离散掉，因为使用 soft bin probability。

#### B. Canonical Viva Question Bank

定义固定问题：

```text
q_j = (clinical_time_anchor, variable_or_group, bin_family, trend_window)
```

示例：

- “6 小时时 lactate 最可能处于哪个 pathology bin？”
- “12 小时时 MAP 是否仍在低灌注 bin？”
- “0-24 小时 creatinine 是否从 normal 转向 high？”
- “最后 6 小时炎症变量组是否显示同步恶化？”

这些问题不来自当前样本的采样日历，而来自任务级 canonical clinical time。它类似 PULSE 中跨中心任务定义的统一化，但实现为可微 question tokens。

#### C. Time-Conditioned Answerer

Answerer 用问题 token 对事件 memory 做 cross-attention，输出：

```text
answer_logits[j, bin]
answer_confidence[j]
source_coverage[j]
```

其中 `source_coverage` 只是给 answerer 自身标记“这个问题附近病历是否有证据”，不直接变成类别 logit。分类器读取的是标准化 answer sheet，而不是事件 mask。

关键差异：

- 不是 Record2Vec/LLM summary：没有自然语言摘要，也不依赖冻结 LLM。
- 不是 TCF 原始 future-event likelihood：最终目标不是生成未来 EHR 事件，而是回答固定临床口试题。
- 不是 principal strata：不预测某个 future status under every policy 的潜在可观测矩阵。
- 不是 conformal/evidential abstention：可信度只作为 answer sheet 的质量通道，不构造预测集或 vacuity mass。

### 2.2 改 Loss：从 policy invariance 转向 Standardized Viva Supervision

总目标：

```text
L = L_viva_cls
  + lambda_ans * L_answer_observation
  + lambda_time * L_time_query_foreseeing
  + lambda_src * L_source_erasure_training
  + lambda_comp * L_exam_completeness
```

#### A. Viva Classification Loss `L_viva_cls`

分类器只读取固定问题的答案：

```text
AnswerSheet = concat(softmax(answer_logits), answer_confidence)
logits = Classifier(AnswerSheet)
L_viva_cls = CE(logits, y)
```

这不是 logits averaging，也不是 policy-jury 排序。无论输入事件来自事实政策还是反事实政策，分类头结构都只面对同一张 answer sheet。

#### B. Answer Observation Loss `L_answer_observation`

对能够从观测事件中直接监督的问题，训练 answerer 回答 pathology bin：

```text
target_bin_j = nearest observed value bin around question time
L_answer_observation = CE(answer_logits_j, target_bin_j)
```

如果问题时间附近没有可用观测，则该题不产生 hard target，而只通过分类和 foreseeing 目标学习。这样避免把“未观测”直接当作类别。

#### C. Time-Query Foreseeing Loss `L_time_query_foreseeing`

借鉴 TCF：随机采样一个未来或回看时间条件 `t*`，要求 answerer 预测该时间附近将落入的 pathology bin，而不是预测原始事件是否会被记录：

```text
P(bin_d(t*) | history up to t, query=(t*, d))
```

这让 encoder 学会“病程在标准临床时间上是什么样”，而不是只记住“医院在什么时间记录了什么”。

#### D. Source Erasure Training `L_source_erasure_training`

反事实采样模块生成若干源病历版本：

- routine-round chart：固定查房式记录；
- alarm-dense chart：告警后密集记录；
- panel-split chart：联测 panel 拆成异步事件；
- value-pending chart：记录时间和变量已知，但数值尚未返回；
- cross-center sparse chart：模拟 PULSE 式跨中心低覆盖。

这些视图不做 pairwise representation consistency，也不要求彼此 logits 相同。训练方式是：

```text
for each source chart:
    answer same canonical question bank
    compute CE on label and answer targets available in that chart
```

也就是说，反事实 view 只是“不同缺页版本的病历来源”。模型学会用任意来源回答固定问题，而不是学会让各来源输出同一隐向量。

#### E. Exam Completeness Loss `L_exam_completeness`

为防止分类器只依赖少数被训练医院高频记录的问题，引入轻量 completeness regularizer：

```text
question_use = classifier attention over viva questions
L_exam_completeness = relu(min_required_topics - effective_topic_count)^2
```

这里的 topic 是预定义的变量组/时间段题型，不是采样 policy group；目标是让 classifier 至少参考多个标准临床问题，避免退化为单个医院流程触发题。

它不同于 evidence budget / tax：没有给 token 定价，也不按 policy cost 购买证据；只是让标准口试不塌缩成一道题。

### 2.3 改 Dataloader：返回 Do-Calendar Viva Source Bank

新增 `DoCalendarVivaCollator`，每个 batch 返回：

1. `factual_chart`：原始不规则事件。
2. `source_chart_bank`：反事实源病历版本：
   - `routine_round`
   - `alarm_dense`
   - `panel_split`
   - `value_pending`
   - `cross_center_sparse`
3. `canonical_question_bank`：
   - 固定 clinical time anchors；
   - variable/group ids；
   - pathology bin family ids；
   - trend window ids。
4. `answer_target_bank`：
   - 每个源病历中可监督的问题目标；
   - 没有可监督目标的问题以 mask 跳过 hard answer loss。
5. `source_descriptor`：
   - 只供调试和覆盖统计；
   - 不输入最终 classifier。

与历史机制区别：

- 不生成 contrastive pairs。
- 不做 policy-simplex smoothing 或 certified radius。
- 不做 conformal set / evidential vacuity。
- 不做 graph/state-policy split、IV residual、knockoff negative control、lattice meet/join 或 social-choice voting。
- 不把 sampling policy descriptors 送进分类头。
- 只把反事实采样用于训练“同一套标准临床口试题如何从不同病历来源中作答”。

### 2.4 推理阶段

给定测试样本：

1. encoder 读取原始不规则事件。
2. viva answerer 回答固定 question bank。
3. classifier 从 answer sheet 输出类别。
4. 同时报告：
   - `answer_sheet_entropy`：哪些问题答案不稳定；
   - `source_coverage_by_topic`：哪些标准临床问题缺证据；
   - `viva_attention`：最终分类依赖哪些临床问题；
   - `calendar_dependence_gap`：可选地对几个部署源版本重复答题，检查最终 answer sheet 是否因源病历缺页严重失真。

注意：推理默认不需要生成反事实 views；反事实只用于诊断或压力测试。

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


class PathologySoftBinner(nn.Module):
    """Map scalar clinical values to monotone soft pathology-bin distributions."""

    def __init__(self, num_vars: int, num_bins: int, init_span: float = 2.0):
        super().__init__()
        self.num_vars = num_vars
        self.num_bins = num_bins
        base = torch.linspace(-init_span, init_span, num_bins - 1)
        self.raw_start = nn.Parameter(base[None].repeat(num_vars, 1))
        self.raw_delta = nn.Parameter(torch.zeros(num_vars, num_bins - 1))

    def cutpoints(self) -> torch.Tensor:
        # Monotone cutpoints per variable.
        delta = F.softplus(self.raw_delta) + 1e-3
        centered = self.raw_start[:, :1] + torch.cumsum(delta, dim=-1)
        return centered - centered.mean(dim=-1, keepdim=True)

    def forward(self, value: torch.Tensor, var_id: torch.Tensor) -> torch.Tensor:
        # value/var_id: [B, N]
        cuts = self.cutpoints()[var_id.clamp_min(0)]  # [B, N, K-1]
        left_mass = torch.sigmoid(value.unsqueeze(-1) - cuts)
        cdf = torch.cat(
            [
                torch.ones_like(value[..., None]),
                left_mass,
                torch.zeros_like(value[..., None]),
            ],
            dim=-1,
        )
        # Probability between consecutive cutpoints.
        return (cdf[..., :-1] - cdf[..., 1:]).clamp_min(1e-6)


class IrregularEventMemory(nn.Module):
    """A lightweight event memory; replaceable by STAR-Set, TCF, mTAND, or CDE."""

    def __init__(self, num_vars: int, num_bins: int, hidden_dim: int):
        super().__init__()
        self.var_embed = nn.Embedding(num_vars, hidden_dim)
        self.bin_proj = nn.Linear(num_bins, hidden_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, batch: dict, bin_prob: torch.Tensor) -> dict:
        time = batch["event_time"]
        var_id = batch["event_var_id"]
        mask = batch["event_mask"]
        value_pending = batch.get("value_pending", torch.zeros_like(mask))

        horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
        time_norm = time / horizon
        delta_t = torch.zeros_like(time)
        delta_t[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)

        event_x = torch.cat(
            [
                self.var_embed(var_id.clamp_min(0)),
                self.bin_proj(bin_prob),
                torch.log1p(delta_t).unsqueeze(-1),
                value_pending.unsqueeze(-1),
            ],
            dim=-1,
        )
        event_h = self.event_proj(event_x) * mask.unsqueeze(-1)
        memory, _ = self.rnn(event_h)
        return {"memory": memory, "event_time_norm": time_norm, "event_mask": mask}


class VivaQuestionBank(nn.Module):
    """Canonical policy-neutral clinical questions."""

    def __init__(
        self,
        num_questions: int,
        num_vars: int,
        num_topics: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.num_questions = num_questions
        self.query_time = nn.Parameter(torch.linspace(0.05, 1.0, num_questions))
        self.query_var_logits = nn.Parameter(torch.randn(num_questions, num_vars) * 0.02)
        self.topic_embed = nn.Embedding(num_topics, hidden_dim)
        self.var_embed = nn.Linear(num_vars, hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.topic_id = nn.Parameter(torch.arange(num_questions) % num_topics, requires_grad=False)

    def forward(self, batch_size: int, device: torch.device) -> dict:
        time = self.query_time.sigmoid().to(device)
        var_weight = torch.softmax(self.query_var_logits.to(device), dim=-1)
        topic = self.topic_id.to(device)
        q = (
            self.time_proj(time[:, None])
            + self.var_embed(var_weight)
            + self.topic_embed(topic)
        )
        return {
            "query": q[None].expand(batch_size, -1, -1),
            "query_time": time[None].expand(batch_size, -1),
            "query_var_weight": var_weight[None].expand(batch_size, -1, -1),
            "query_topic": topic,
        }


class TimeConditionedVivaAnswerer(nn.Module):
    """Answer canonical pathology-bin questions from irregular event memory."""

    def __init__(self, hidden_dim: int, num_bins: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_bias = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.answer_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, memory_out: dict, questions: dict) -> dict:
        memory = memory_out["memory"]
        event_mask = memory_out["event_mask"]
        event_time = memory_out["event_time_norm"]
        query = questions["query"]
        query_time = questions["query_time"]

        q = self.q_proj(query)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        scale = q.size(-1) ** -0.5
        attn = torch.einsum("bqh,bnh->bqn", q, k) * scale
        time_gap = (query_time[:, :, None] - event_time[:, None, :]).abs()
        attn = attn + self.time_bias(time_gap.unsqueeze(-1)).squeeze(-1)
        attn = attn.masked_fill(event_mask[:, None] == 0, -1e4)
        weight = torch.softmax(attn, dim=-1)
        context = torch.einsum("bqn,bnh->bqh", weight, v)

        answer_state = torch.cat([query, context], dim=-1)
        answer_logits = self.answer_head(answer_state)
        confidence = torch.sigmoid(self.conf_head(answer_state)).squeeze(-1)
        coverage = torch.einsum("bqn,bn->bq", weight, event_mask.to(weight.dtype))
        return {
            "answer_logits": answer_logits,
            "answer_confidence": confidence,
            "source_coverage": coverage,
            "attention": weight,
        }


class VivaExamClassifier(nn.Module):
    """Classify only from the canonical answer sheet."""

    def __init__(self, num_questions: int, num_bins: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.question_score = nn.Linear(num_bins + 2, 1)
        self.net = nn.Sequential(
            nn.Linear(num_questions * (num_bins + 2), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, answers: dict) -> dict:
        prob = torch.softmax(answers["answer_logits"], dim=-1)
        sheet = torch.cat(
            [
                prob,
                answers["answer_confidence"].unsqueeze(-1),
                answers["source_coverage"].unsqueeze(-1),
            ],
            dim=-1,
        )
        question_weight = torch.softmax(self.question_score(sheet).squeeze(-1), dim=-1)
        logits = self.net(sheet.flatten(start_dim=1))
        return {"logits": logits, "answer_sheet": sheet, "question_weight": question_weight}


class DoCalendarVivaExaminer(nn.Module):
    """Sampling-policy robust classifier via canonical clinical viva questions."""

    def __init__(
        self,
        num_vars: int,
        num_bins: int,
        num_questions: int,
        num_topics: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.binner = PathologySoftBinner(num_vars, num_bins)
        self.memory = IrregularEventMemory(num_vars, num_bins, hidden_dim)
        self.questions = VivaQuestionBank(num_questions, num_vars, num_topics, hidden_dim)
        self.answerer = TimeConditionedVivaAnswerer(hidden_dim, num_bins)
        self.classifier = VivaExamClassifier(num_questions, num_bins, hidden_dim, num_classes)
        self.num_bins = num_bins

    def forward(self, batch: dict) -> dict:
        bin_prob = self.binner(batch["event_value"], batch["event_var_id"])
        memory = self.memory(batch, bin_prob)
        q = self.questions(batch["event_value"].size(0), batch["event_value"].device)
        answers = self.answerer(memory, q)
        pred = self.classifier(answers)
        return {**answers, **pred, "questions": q, "event_bin_prob": bin_prob}

    def answer_loss(self, out: dict, batch: dict) -> torch.Tensor:
        # answer_target: [B, Q], answer_target_mask: [B, Q]
        if "answer_target" not in batch:
            return torch.zeros((), device=out["logits"].device)
        raw = F.cross_entropy(
            out["answer_logits"].transpose(1, 2),
            batch["answer_target"].clamp(0, self.num_bins - 1),
            reduction="none",
        )
        mask = batch["answer_target_mask"].to(raw.dtype)
        return (raw * mask).sum() / mask.sum().clamp_min(1.0)

    def exam_completeness_loss(self, out: dict, min_topics: float = 3.0) -> torch.Tensor:
        # Effective number of attended questions; prevents collapse to one policy-friendly question.
        w = out["question_weight"].clamp_min(1e-8)
        entropy = -(w * w.log()).sum(dim=-1)
        effective_questions = entropy.exp()
        return F.relu(min_topics - effective_questions).pow(2).mean()

    def training_loss(
        self,
        batch: dict,
        lambda_ans: float = 0.35,
        lambda_src: float = 0.40,
        lambda_comp: float = 0.03,
    ) -> dict:
        labels = batch["labels"]
        factual = self.forward(batch)
        cls_loss = F.cross_entropy(factual["logits"], labels)
        answer_loss = self.answer_loss(factual, batch)
        completeness = self.exam_completeness_loss(factual)

        source_losses = []
        for source_batch in batch.get("source_chart_bank", []):
            source_out = self.forward(source_batch)
            # Each source chart is a different chart version answering the same viva,
            # not a pairwise consistency target.
            source_cls = F.cross_entropy(source_out["logits"], labels)
            source_ans = self.answer_loss(source_out, source_batch)
            source_losses.append(source_cls + lambda_ans * source_ans)
        if source_losses:
            source_loss = torch.stack(source_losses).mean()
        else:
            source_loss = torch.zeros((), device=labels.device)

        total = (
            cls_loss
            + lambda_ans * answer_loss
            + lambda_src * source_loss
            + lambda_comp * completeness
        )
        return {
            "loss": total,
            "viva_cls_loss": cls_loss.detach(),
            "answer_loss": answer_loss.detach(),
            "source_erasure_loss": source_loss.detach(),
            "exam_completeness_loss": completeness.detach(),
            "mean_answer_confidence": factual["answer_confidence"].mean().detach(),
        }


@torch.no_grad()
def build_do_calendar_viva_source_bank(batch: dict, num_sources: int = 5) -> list[dict]:
    """Create source chart variants while preserving the canonical question bank."""
    value = batch["event_value"]
    time = batch["event_time"]
    var_id = batch["event_var_id"]
    mask = batch["event_mask"]
    bsz, num_events = time.shape
    device = time.device

    horizon = (time * mask).amax(dim=1, keepdim=True).clamp_min(1e-6)
    time_norm = time / horizon

    def clone_with(new_value, new_time, new_var, new_mask, pending=None):
        out = dict(batch)
        out["event_value"] = new_value
        out["event_time"] = new_time
        out["event_var_id"] = new_var
        out["event_mask"] = new_mask
        if pending is not None:
            out["value_pending"] = pending
        return out

    sources = []

    # 1. Routine-round chart: snap timestamps to standardized clinical rounds.
    rounded_time = torch.round(time_norm * 8.0) / 8.0 * horizon
    sources.append(clone_with(value * mask, rounded_time, var_id, mask))

    # 2. Alarm-dense chart: retain late observations, thin early routine events.
    late = (time_norm > 0.66).to(mask.dtype)
    alternating = ((torch.arange(num_events, device=device)[None] % 2) == 0).to(mask.dtype)
    alarm_mask = torch.where(late > 0, mask, mask * alternating)
    sources.append(clone_with(value * alarm_mask, time, var_id, alarm_mask))

    # 3. Panel-split chart: remove half of locally co-observed cross-variable events.
    gap = torch.zeros_like(time)
    gap[:, 1:] = (time[:, 1:] - time[:, :-1]).clamp_min(0.0)
    mean_gap = (gap * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    close = (gap <= mean_gap.clamp_min(1e-6)).to(mask.dtype)
    changed_var = torch.zeros_like(mask)
    changed_var[:, 1:] = (var_id[:, 1:] != var_id[:, :-1]).to(mask.dtype)
    split_mask = mask * (1.0 - 0.5 * close * changed_var)
    sources.append(clone_with(value * split_mask, time, var_id, split_mask))

    # 4. Value-pending chart: the event calendar is visible, values are not yet returned.
    pending = mask.clone()
    pending_value = torch.zeros_like(value)
    sources.append(clone_with(pending_value, time, var_id, mask, pending=pending))

    # 5. Cross-center sparse chart: fixed event budget per coarse clinical phase.
    sparse = torch.zeros_like(mask)
    for start, end in [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]:
        in_phase = ((time_norm >= start) & (time_norm < end)).to(mask.dtype) * mask
        rank = in_phase.cumsum(dim=1)
        sparse = torch.maximum(sparse, (rank <= 2).to(mask.dtype) * in_phase)
    sources.append(clone_with(value * sparse, time, var_id, sparse))

    return sources[:num_sources]
```

## 4. 实验切入点

1. **Policy shift 构造**
   - `cross-center PULSE split`：以 HiRID/MIMIC/eICU 或类 PULSE 的医院来源作为环境划分，比较 raw encoder 与 viva answer sheet classifier。
   - `TCF calendar shift`：改变绝对/相对时间戳分布，但保持 canonical viva question bank 不变。
   - `panel-split/value-pending shift`：训练时联测或数值及时返回，测试时拆单或值延迟返回。
   - `routine-vs-alarm source shift`：固定查房式记录与告警后密集记录互换。

2. **对比方法**
   - 普通 irregular event Transformer / STAR-Set / TCF classifier。
   - TCF pretraining + direct `[CLS]` classifier。
   - PULSE-style feature engineering / LightGBM / LLM prompt baseline。
   - mask dropout / random missing augmentation。
   - missingness-aware encoder。
   - 历史方案 DHN、CGS、PT-AEM、PQD、DS-CS、OS-MQ、CETC、PGHT、SCSC、CKCF、PIIES、PLSM、ST-FDN、C-CRS、D-IVSP、DJRT、KPMA、DVNB、DTSR。

3. **核心指标**
   - in-policy accuracy / AUPRC。
   - worst-policy accuracy / AUPRC。
   - viva answer calibration：标准问题答案与 held-out observation bin 的校准误差。
   - source-format leakage：只用 answer sheet 的 `source_coverage` 是否能预测医院/政策。
   - question attention stability：跨医院时分类器关注的问题是否仍是同一批临床问题。
   - calendar dependence gap：事实病历与反事实源病历的 answer sheet 变化是否集中在低覆盖题，而不是高权重诊断题。

4. **消融实验**
   - 去掉 canonical viva question bank，直接用 pooled event memory 分类。
   - 去掉 pathology-focused soft binner，改用原始数值回归答案。
   - 去掉 source erasure training，只用事实病历答题。
   - 去掉 exam completeness，让分类器自由塌缩到少数问题。
   - 将 question time anchors 改成样本自适应采样时间，验证 policy-neutral 问题日历的必要性。

## 5. 预期创新性

1. **从“处理采样日历”转向“标准化提问日历”**：不是估计、删除、校准或投票采样政策，而是让所有病历先回答同一套 policy-neutral 临床问题。
2. **从 future-event generation 转向 viva answer sheet**：吸收 TCF 的 time-conditioned foreseeing 与 pathology-focused binning，但最终分类不预测未来 EHR 事件是否被记录，而是预测固定问题下的临床语义答案。
3. **从 LLM prompt benchmark 转向可微临床口试**：吸收 PULSE 对跨中心和 reasoning workflow 的启发，但把“提问-回答-分类”做成端到端 PyTorch 模块，不依赖自然语言摘要或闭源 LLM。
4. **从反事实一致性转向多源病历答题训练**：counterfactual interventions 只改变源病历缺页格式；模型在同一 question bank 上作答，不要求不同源 representation、logits 或风险方差一致。
5. **部署解释直接可读**：输出不是抽象 latent，也不是保形集合或 jury report，而是“哪些标准临床问题被模型如何回答、置信度如何、最终分类依赖哪些问题”。

## 6. 一句话投稿卖点

**DCVE 首次把非规则采样时间序列分类中的 sampling-policy shift 表述为“病历记录格式变化导致模型拿到不同考卷”的问题，并通过 policy-neutral canonical viva question bank、Pathology-Focused soft bin answers 与多源反事实病历答题训练，让分类器只基于标准化临床问题答案决策，而不是直接依赖训练医院特有的采样日历、panel 共现、复测 burst 或 value-pending 流程。**
