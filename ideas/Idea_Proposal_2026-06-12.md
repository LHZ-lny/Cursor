# Title: RaSIB - Rate-Adaptive Spectral Information Bottleneck for Sampling-Shift Robust IRTS Classification

## Motivation

近期 AAAI 2026 的 SPECTRA 强调了一个重要观点：不规则采样时序中的频谱失真、缺失模式与类别判别并不是彼此独立的问题。对我们当前关注的“采样率偏移鲁棒分类”而言，核心风险更进一步：模型在训练采样率下学到的高频能量、谱泄漏、观测间隔分布，可能只是采样策略的副产物，而不是稳定的类别语义。一旦测试阶段采样更稀疏、更密集或采样策略改变，这些伪频域线索会造成分类边界漂移。

RaSIB 的核心想法是把 SPECTRA 的频域稳定建模，与信息瓶颈机制结合：让编码器显式保留对类别有用的频域成分，同时压缩可预测采样率/采样策略的频域成分。换言之，目标不是简单“加入频域信息”，而是学习一个满足下列性质的谱表示：

1. 对标签 `Y` 高信息量；
2. 对采样率签名 `R` 低信息量；
3. 在同一样本的多采样率视图之间保持预测一致；
4. 在不同采样策略导致的混叠和谱泄漏下仍保持稳定。

这使模型从“依赖训练采样率下的固定谱纹理”转向“提取跨采样率保持不变的类别谱语义”，从机制上缓解 sampling-rate shift。

## Methodology

### 1. Dataloader: Multi-Rate Counterfactual Views

对每条不规则时间序列 `x = {(t_i, v_i)}` 构造两个视图：

- `view_anchor`：原始观测或轻微扰动后的观测；
- `view_rate_shift`：通过随机下采样、局部采样间隔拉伸、事件保持式稀疏化等方式生成的反事实采样率视图。

同时输出采样率签名 `r`，例如：

- 平均采样间隔、间隔方差；
- `delta_t` 的分位数直方图；
- 有效观测率；
- 每个变量的观测频次。

该签名不是直接用于分类，而是用于训练阶段约束表示不要泄漏采样策略。

### 2. Encoder: Rate-Conditioned Spectral Mask + Bottleneck

在时间编码器前增加一个频域前端：

1. 将不规则观测投影到统一的时间参考网格，保留 mask；
2. 对每个变量做 `rFFT`，得到幅度谱与相位/实部虚部表示；
3. 用采样率签名 `r` 生成一个 soft spectral mask；
4. mask 的作用不是增强所有频率，而是选择“跨采样率稳定且类别相关”的频段；
5. 将 masked spectrum 还原或直接送入 Transformer/TCN/CDE encoder；
6. 通过 variational information bottleneck 输出 `z = mu + eps * sigma`。

直觉上，采样率变化最容易影响高频、边界频率和泄漏频段；mask generator 学习在不同 `r` 下调整频谱通道，而 bottleneck 防止模型把采样策略本身编码进 `z`。

### 3. Loss: Label-Preserving, Rate-Forgetting Objective

总损失：

```text
L = L_cls
  + beta * KL(q(z|x,r) || N(0,I))
  + lambda_cons * KL(p(y|view_anchor) || p(y|view_rate_shift))
  - lambda_adv * CE(rate_predictor(grad_reverse(z)), r_bin)
  + lambda_mask * spectral_mask_sparsity
```

其中：

- `L_cls` 保证分类性能；
- `KL` 是信息瓶颈，压缩冗余输入信息；
- `consistency loss` 要求同一样本的不同采样率视图预测一致；
- `rate adversarial loss` 让主表示难以预测采样率 bin，从而降低 `I(Z; R)`；
- `mask sparsity` 避免频域分支退化为全频复制。

### 4. Expected Contribution

可以将论文贡献包装为：

- 提出 Sampling-Rate Shift 下的频域伪相关问题定义；
- 提出 Rate-Adaptive Spectral Information Bottleneck，在频域显式执行“保标签、忘采样率”的表示学习；
- 通过多采样率反事实视图训练，提高从 dense-to-sparse、sparse-to-dense、mixed-rate 测试条件下的分类稳定性；
- 与现有不规则时序 encoder 兼容，可作为 GRU-D、Transformer、Neural CDE 或 FlowPath 类路径模型的前端模块。

## Code Draft

下面是核心 PyTorch 草稿，展示如何把频域 mask、信息瓶颈与采样率对抗约束接入现有分类器。真实实验中，`x_grid` 可以由已有 irregular collate/interpolation 模块产生，`rate_sig` 由 dataloader 统计得到。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def grad_reverse(x, scale=1.0):
    return GradReverse.apply(x, scale)


class RateAdaptiveSpectralBottleneck(nn.Module):
    """
    Input:
        x_grid:   [B, C, T], values projected to a common reference grid
        mask_grid:[B, C, T], observation mask on the same grid
        rate_sig: [B, R], sampling-rate signature from the dataloader
    Output:
        z:        [B, latent_dim], rate-invariant bottleneck representation
        aux:      statistics for losses
    """
    def __init__(
        self,
        channels,
        time_steps,
        rate_dim,
        hidden_dim=128,
        latent_dim=128,
        num_classes=2,
        num_rate_bins=4,
    ):
        super().__init__()
        freq_bins = time_steps // 2 + 1
        spectral_dim = channels * freq_bins * 2  # amplitude + observed energy

        self.rate_encoder = nn.Sequential(
            nn.Linear(rate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mask_generator = nn.Sequential(
            nn.Linear(hidden_dim, channels * freq_bins),
            nn.Sigmoid(),
        )
        self.spectral_encoder = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.classifier = nn.Linear(latent_dim, num_classes)
        self.rate_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_rate_bins),
        )

        self.channels = channels
        self.freq_bins = freq_bins

    def forward(self, x_grid, mask_grid, rate_sig, grl_scale=1.0):
        # rFFT exposes frequency components whose stability is sensitive to sampling shifts.
        spectrum = torch.fft.rfft(x_grid * mask_grid, dim=-1)
        amp = torch.log1p(torch.abs(spectrum))

        mask_spectrum = torch.fft.rfft(mask_grid, dim=-1)
        obs_energy = torch.log1p(torch.abs(mask_spectrum))

        rate_feat = self.rate_encoder(rate_sig)
        soft_mask = self.mask_generator(rate_feat).view(
            -1, self.channels, self.freq_bins
        )

        amp = amp * soft_mask
        obs_energy = obs_energy * soft_mask
        spectral_feat = torch.cat(
            [amp.flatten(1), obs_energy.flatten(1)],
            dim=-1,
        )

        h = self.spectral_encoder(spectral_feat)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(min=-8.0, max=8.0)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std if self.training else mu

        logits = self.classifier(z)
        rate_logits = self.rate_predictor(grad_reverse(z, grl_scale))

        aux = {
            "mu": mu,
            "logvar": logvar,
            "rate_logits": rate_logits,
            "soft_mask": soft_mask,
        }
        return logits, z, aux


def rasib_loss(
    anchor_out,
    shifted_out,
    labels,
    rate_bins,
    beta=1e-3,
    lambda_cons=0.5,
    lambda_adv=0.1,
    lambda_mask=1e-3,
):
    logits_a, _, aux_a = anchor_out
    logits_s, _, aux_s = shifted_out

    cls_loss = 0.5 * (
        F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_s, labels)
    )

    def kl_standard_normal(aux):
        mu, logvar = aux["mu"], aux["logvar"]
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    ib_loss = 0.5 * (kl_standard_normal(aux_a) + kl_standard_normal(aux_s))

    prob_a = F.log_softmax(logits_a, dim=-1)
    prob_s = F.softmax(logits_s.detach(), dim=-1)
    cons_loss = F.kl_div(prob_a, prob_s, reduction="batchmean")

    # Because rate logits receive grad reversal, minimizing this CE removes
    # sampling-rate information from the encoder while training the predictor.
    rate_loss = 0.5 * (
        F.cross_entropy(aux_a["rate_logits"], rate_bins)
        + F.cross_entropy(aux_s["rate_logits"], rate_bins)
    )

    mask_loss = 0.5 * (
        aux_a["soft_mask"].mean() + aux_s["soft_mask"].mean()
    )

    total = (
        cls_loss
        + beta * ib_loss
        + lambda_cons * cons_loss
        + lambda_adv * rate_loss
        + lambda_mask * mask_loss
    )
    return total, {
        "cls": cls_loss.detach(),
        "ib": ib_loss.detach(),
        "cons": cons_loss.detach(),
        "rate_adv": rate_loss.detach(),
        "mask": mask_loss.detach(),
    }


def training_step(model, batch, optimizer):
    anchor = model(
        batch["x_anchor_grid"],
        batch["m_anchor_grid"],
        batch["rate_anchor"],
        grl_scale=1.0,
    )
    shifted = model(
        batch["x_shift_grid"],
        batch["m_shift_grid"],
        batch["rate_shift"],
        grl_scale=1.0,
    )
    loss, logs = rasib_loss(
        anchor,
        shifted,
        labels=batch["label"],
        rate_bins=batch["rate_bin"],
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return logs
```

## Experimental Checkpoints

1. **Dense-to-sparse shift**：训练使用较高观测率，测试系统性下采样；
2. **Sparse-to-dense shift**：训练稀疏，测试较密集，检验模型是否过度依赖缺失模式；
3. **Mixed-rate domain split**：按采样率分桶做 domain generalization；
4. **Ablation**：
   - 去掉 spectral mask；
   - 去掉 information bottleneck；
   - 去掉 rate adversarial loss；
   - 去掉 multi-rate consistency。

预期现象：完整 RaSIB 在 ID 测试集上不一定显著压倒最强频域 baseline，但在 OOD sampling-rate shift 下应体现更小的 accuracy/F1 drop 和更稳定的 calibration。
