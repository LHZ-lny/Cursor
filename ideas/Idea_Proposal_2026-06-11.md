# Title: Rate-Shift Invariant Spectral Bottleneck (RSI-SB)

## Motivation

近期 AAAI 2026 的 SPECTRA 提醒我们：不规则采样不仅造成缺失，还会制造稳定的频谱畸变；FlowPath 则强调，输入路径的几何构造会直接影响连续时间分类器的鲁棒性。针对“采样率/采样策略发生偏移时分类性能不稳定”的问题，我建议引入一个更聚焦的前沿机制：**频域信息瓶颈**。

核心直觉是：采样策略偏移会改变观测密度、时间间隔分布和可见频段，从而在频域中产生混叠、泄漏和伪高频模式。普通频域增强可能把这些采样策略伪信号也学进去，导致模型在训练采样率和测试采样率不一致时崩塌。RSI-SB 将频域分解为两类信息：

1. **Label-relevant spectrum**：与真实动态和类别判别有关，应被保留。
2. **Policy-specific spectrum**：由采样率、采样间隔、缺失模式诱导，应被压缩或对抗消除。

因此，RSI-SB 不只是“加入 FFT 分支”，而是在频域特征上加入瓶颈、跨采样率视图一致性和采样策略去识别约束，使分类器学到对采样偏移稳定的频谱表示。

## Methodology

### 1. Encoder: Spectral Bottleneck Encoder

在现有不规则时间序列 Encoder 前后加入一个轻量频域瓶颈分支：

- 将不规则观测通过 mask-aware interpolation / kernel smoothing 投影到统一参考网格。
- 对参考网格信号做 `rFFT`，得到 amplitude 与 phase 或 real/imag 表示。
- 使用可学习的频域门控 `spectral_gate` 选择稳定频段。
- 用变分瓶颈输出 `mu, logvar`，采样得到频域 latent `z_f`。
- 将 `z_f` 与原始时域 latent `z_t` 融合，用于最终分类。

该模块的目标不是重建完整频谱，而是压缩掉与采样策略强相关、但对类别不稳定的频率成分。

### 2. Loss: Label Preservation + Sampling Policy Removal

训练目标由四部分组成：

- `L_cls`：正常分类交叉熵，保证任务性能。
- `L_ib`：频域瓶颈 KL 正则，限制频域 latent 容量，避免记忆采样率伪模式。
- `L_rate_inv`：同一样本不同采样率视图的 supervised / positive contrastive consistency，使表示在采样偏移下靠近。
- `L_adv`：采样策略对抗损失，通过 Gradient Reversal 让频域表示无法预测采样率 bin 或采样策略 ID。

总目标：

```text
L = L_cls + beta * L_ib + lambda_c * L_rate_inv + lambda_adv * L_adv
```

其中 `beta` 控制瓶颈强度，`lambda_adv` 控制去采样策略信息强度。

### 3. Dataloader: Multi-rate Sampling Views

对每条训练序列生成两个或多个采样率视图：

- `view_dense`：保留较多观测点，模拟高频采样设备。
- `view_sparse`：按随机 thinning / block dropout / interval jitter 生成低频或不均匀采样。
- 返回 `sampling_policy_id` 或 `rate_bin`，用于对抗头训练。

这一步非常关键：如果没有显式采样策略扰动，瓶颈只能压缩一般噪声，无法准确学习“哪些频域模式是采样率诱导的”。

### 4. Expected Contribution

论文可以主打以下创新点：

- 从“频域增强”升级为“频域去偏瓶颈”，避免把采样策略伪频谱当作类别证据。
- 将采样率偏移定义为一种可学习的 domain/policy shift，并用对抗信息瓶颈消除。
- 对不规则采样分类提供一个模型无关插件，可接到 Transformer、GRU-D、Neural CDE 或 FlowPath-style encoder 上。

## Code Draft

下面是核心 PyTorch 草稿，展示 Encoder、Gradient Reversal、loss 组合方式。实际接入时，只需要把 `base_encoder` 替换为当前项目中的时域/连续时间编码器。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class SpectralBottleneck(nn.Module):
    """
    Mask-aware spectral bottleneck for rate-shift robust classification.

    x_grid:    [B, T, C] values projected to a reference grid
    mask_grid: [B, T, C] observation mask on the same grid
    """
    def __init__(self, channels, grid_size, latent_dim, hidden_dim=128):
        super().__init__()
        freq_bins = grid_size // 2 + 1
        self.freq_bins = freq_bins

        self.spectral_gate = nn.Sequential(
            nn.Linear(freq_bins * channels * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, freq_bins * channels),
            nn.Sigmoid(),
        )
        self.to_stats = nn.Sequential(
            nn.Linear(freq_bins * channels * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(self, x_grid, mask_grid):
        observed_ratio = mask_grid.sum(dim=1, keepdim=True).clamp_min(1.0)
        x_centered = (x_grid * mask_grid).sum(dim=1, keepdim=True) / observed_ratio
        x_filled = torch.where(mask_grid.bool(), x_grid, x_centered)

        spectrum = torch.fft.rfft(x_filled, dim=1, norm="ortho")
        spec_feat = torch.cat([spectrum.real, spectrum.imag], dim=-1)
        spec_flat = spec_feat.flatten(start_dim=1)

        gate = self.spectral_gate(spec_flat).view(
            x_grid.size(0), self.freq_bins, x_grid.size(-1)
        )
        gated_spectrum = spectrum * gate
        gated_feat = torch.cat([gated_spectrum.real, gated_spectrum.imag], dim=-1)

        mu, logvar = self.to_stats(gated_feat.flatten(start_dim=1)).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_f = mu + eps * std

        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        return z_f, kl, gate


class RateShiftInvariantClassifier(nn.Module):
    def __init__(
        self,
        base_encoder,
        channels,
        grid_size,
        temporal_dim,
        spectral_dim,
        num_classes,
        num_rate_bins,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.spectral = SpectralBottleneck(
            channels=channels,
            grid_size=grid_size,
            latent_dim=spectral_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim + spectral_dim, temporal_dim),
            nn.GELU(),
            nn.Linear(temporal_dim, num_classes),
        )
        self.rate_head = nn.Sequential(
            nn.Linear(spectral_dim, spectral_dim),
            nn.GELU(),
            nn.Linear(spectral_dim, num_rate_bins),
        )

    def forward(self, batch, grl_lambda=1.0):
        # base_encoder consumes the original irregular observations.
        z_t = self.base_encoder(
            values=batch["values"],
            times=batch["times"],
            mask=batch["mask"],
        )
        z_f, kl, gate = self.spectral(batch["x_grid"], batch["mask_grid"])

        logits = self.classifier(torch.cat([z_t, z_f], dim=-1))
        rate_logits = self.rate_head(grad_reverse(z_f, grl_lambda))
        return {
            "logits": logits,
            "rate_logits": rate_logits,
            "z": torch.cat([z_t, z_f], dim=-1),
            "kl": kl,
            "gate": gate,
        }


def positive_view_consistency(z_a, z_b, temperature=0.2):
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.t() / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.t(), labels)
    )


def rsi_sb_loss(model, batch_a, batch_b, beta=1e-3, lambda_c=0.1, lambda_adv=0.2):
    out_a = model(batch_a, grl_lambda=lambda_adv)
    out_b = model(batch_b, grl_lambda=lambda_adv)

    y = batch_a["label"]
    cls_loss = 0.5 * (
        F.cross_entropy(out_a["logits"], y) +
        F.cross_entropy(out_b["logits"], y)
    )
    ib_loss = 0.5 * (out_a["kl"] + out_b["kl"])
    inv_loss = positive_view_consistency(out_a["z"], out_b["z"])

    rate_loss = 0.5 * (
        F.cross_entropy(out_a["rate_logits"], batch_a["rate_bin"]) +
        F.cross_entropy(out_b["rate_logits"], batch_b["rate_bin"])
    )

    total = cls_loss + beta * ib_loss + lambda_c * inv_loss + rate_loss
    return {
        "loss": total,
        "cls": cls_loss.detach(),
        "ib": ib_loss.detach(),
        "rate_inv": inv_loss.detach(),
        "rate_adv": rate_loss.detach(),
    }
```

## Experiment Sketch

- **Train/Test split by sampling policy**：训练使用一种或几种采样率，测试使用未见过的更稀疏、更密集或更 bursty 的采样策略。
- **Ablation**：
  - FFT branch only
  - FFT + multi-rate contrastive
  - FFT + bottleneck
  - Full RSI-SB
- **Metrics**：
  - In-domain accuracy / AUROC
  - Cross-rate accuracy drop
  - Expected calibration error under sampling shift
  - Frequency gate stability across sampling policies

## One-sentence Pitch

RSI-SB 把采样率偏移下的鲁棒不规则时序分类，从“增强频域特征”推进到“学习类别相关、采样策略不变的频域瓶颈表示”。
