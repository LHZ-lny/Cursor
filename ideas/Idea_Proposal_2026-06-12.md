# Idea Proposal - 2026-06-12

## Title

**Receipt Firewall: Counterfactual Sampling-Score Gradient Nulling for Policy-Shift Robust Irregular Time-Series Classification**

## Forced Reading and De-duplication Notes

- Requested historical file `my_work_summary.md`: not present in the current repository.
- Historical proposal files: no `ideas/*.md`, `*Proposal*.md`, `*Idea*.md`, or Chinese `*提案*.md` files were found in the current repository.
- Available recent reading: `paper_daily_2026-06-12.md`.

### Mechanism Blacklist Extracted From Available Records

The new idea must not use the following as its central mechanism:

1. Learnable temporal reference points / adaptive time encoding as the main architecture.
2. Temporal or inter-variable consistency regularization as the main novelty.
3. Missingness pattern encoder directly feeding the classifier.
4. Frequency-guided observation encoder or frequency-domain masking contrastive learning.
5. Prototype-constrained or policy-aware prototype classifiers.
6. Generic mask resampling consistency where two sampled views are simply forced to share logits.
7. Standard environment adversarial learning that only predicts the sampling domain from the latent.

This proposal instead treats the sampling policy as an **audit-only receipt** and removes its gradient-level causal access to the classifier.

## Motivation

Recent work in `paper_daily_2026-06-12.md` suggests two useful but risky directions:

- Adaptive time encoding shows that irregular timestamps need a learnable alignment mechanism, but such alignment can absorb hospital/device-specific sampling rules.
- SPECTRA shows that missingness and class imbalance are coupled, but directly encoding missingness can turn sampling policy into a shortcut.

For sampling-policy shift, the key failure mode is not only that the model "sees masks"; it is that the classifier's decision boundary can become locally sensitive to directions in representation space that are explainable by the sampling policy. A model may look invariant at the logit level under a few augmentations while still keeping a hidden sampling-policy channel that reappears under a new policy.

The proposed combination with the current **sampling decoupling / counterfactual intervention** framework is:

1. Keep the value encoder focused on observed clinical/sensor values.
2. Build a policy receipt from the dataloader: a differentiable summary of how likely the observed schedule is under an estimated or simulated sampling policy.
3. Use counterfactual interventions to produce alternate schedules for the same latent trajectory.
4. Penalize the classifier when its class logit gradient aligns with the sampling-receipt gradient.

Thus, the sampling branch is not a feature provider. It is a firewall inspector: it tells the training objective which latent directions are policy-contaminated, and the classifier is trained to be locally blind to those directions.

## Methodology

### High-level Mechanism

**Counterfactual Sampling-Score Gradient Nulling (CSGN)** introduces a sampling receipt `r` for each observed irregular sequence:

- `r` is an audit vector built from the observation schedule: inter-arrival statistics, censoring indicators, variable-specific observation counts, and estimated log-propensity under the current sampling policy.
- A small `receipt_head` predicts `r` from the learned latent `z`.
- The classifier head predicts labels from `z`.
- The core loss removes the classifier's gradient component along the receipt-explainable direction in latent space.

The novelty is not to make `z` unable to contain every trace of sampling. Some sampling information may be biologically or operationally informative. Instead, the classifier is forbidden from using the part of `z` whose local influence is aligned with the sampling receipt.

### What Changes

#### 1. Dataloader: Add Counterfactual Receipt Generation

For each irregular sequence `(times, values, mask, y)`, return:

- `obs_view`: the original irregular observation set.
- `cf_view`: a counterfactual schedule intervention, e.g. thinning, densifying, or variable-wise swapping of observation times while preserving the same underlying values through interpolation or nearest observed carry.
- `receipt`: schedule-only audit features for `obs_view`.
- `cf_receipt`: schedule-only audit features for `cf_view`.

This is more specific than ordinary mask augmentation: the dataloader records the **sampling policy score** that caused the view, not just the resulting mask.

#### 2. Encoder: Preserve Current Sampling-Decoupled Backbone

Do not replace the existing encoder with learnable reference points, frequency encoders, or missingness encoders. The encoder should still map irregular observations to a value-dominant latent:

```text
z = value_encoder(times, values, mask)
```

The receipt branch is attached only during training:

```text
receipt_pred = receipt_head(stop_or_live(z))
```

Use live `z` for gradient auditing, but do not concatenate `receipt_pred` into the classifier.

#### 3. Loss: Receipt Firewall Gradient Penalty

Let:

- `logit_y` be the logit of the ground-truth class.
- `receipt_energy = || receipt_head(z) - receipt ||^2`.
- `g_cls = d logit_y / d z`.
- `g_rec = d receipt_energy / d z`.

The firewall penalty is:

```text
L_firewall = mean( cosine(g_cls, stopgrad(g_rec))^2 )
```

This directly discourages the classifier from moving along latent directions that are useful for reconstructing the sampling policy receipt.

#### 4. Counterfactual Stabilization Without Collapsing All Policy Signal

For the original and counterfactual views:

```text
L_cf = KL( softmax(logits_obs / tau) || softmax(logits_cf / tau) )
```

But apply it only after receipt-gradient nulling, and optionally weight it by receipt distance:

```text
w = sigmoid(alpha * ||receipt - cf_receipt||)
L_cf = w * KL(...)
```

This focuses invariance pressure on actual policy interventions rather than arbitrary noise.

#### 5. Total Objective

```text
L = L_ce
  + lambda_firewall * L_firewall
  + lambda_cf * L_cf
  + lambda_receipt * L_receipt
```

`L_receipt` trains the audit head to identify policy-sensitive latent directions; `L_firewall` prevents the classifier from exploiting them.

### Why This Is Orthogonal to Prior Ideas

- Unlike missingness-pattern encoding, the receipt is not an input feature for classification.
- Unlike frequency-guided methods, no spectral representation is required.
- Unlike reference-point learning, the method does not depend on a new temporal alignment grid.
- Unlike prototype losses, it does not reshape class geometry through fixed class centers.
- Unlike simple counterfactual consistency, it removes the local gradient channel through which policy leakage enters the classifier.

## Code Draft

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplingReceiptBuilder:
    """Build schedule-only audit features; never include observed values."""

    def __call__(self, times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # times: [B, T], mask: [B, T, C]
        obs_any = mask.float().amax(dim=-1)  # [B, T]
        dt = torch.diff(times, dim=1, prepend=times[:, :1])
        safe_count = obs_any.sum(dim=1).clamp_min(1.0)

        mean_dt = (dt * obs_any).sum(dim=1) / safe_count
        var_dt = (((dt - mean_dt[:, None]) ** 2) * obs_any).sum(dim=1) / safe_count
        density = safe_count / mask.size(1)
        var_density = mask.float().mean(dim=1)  # [B, C]

        # A compact receipt. In practice, append estimated log propensity
        # from the sampler if the counterfactual policy is known.
        return torch.cat(
            [
                density[:, None],
                mean_dt[:, None],
                var_dt[:, None],
                var_density,
            ],
            dim=-1,
        )


class ReceiptFirewallLoss(nn.Module):
    def __init__(
        self,
        receipt_dim: int,
        latent_dim: int,
        lambda_firewall: float = 0.2,
        lambda_receipt: float = 0.05,
        lambda_cf: float = 0.5,
        tau: float = 2.0,
    ) -> None:
        super().__init__()
        self.receipt_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, receipt_dim),
        )
        self.lambda_firewall = lambda_firewall
        self.lambda_receipt = lambda_receipt
        self.lambda_cf = lambda_cf
        self.tau = tau

    def _firewall_penalty(
        self,
        logits: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
        receipt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        receipt_pred = self.receipt_head(z)
        receipt_loss = F.mse_loss(receipt_pred, receipt)

        chosen_logit = logits.gather(1, y[:, None]).sum()
        g_cls = torch.autograd.grad(
            chosen_logit,
            z,
            create_graph=True,
            retain_graph=True,
        )[0]

        receipt_energy = F.mse_loss(receipt_pred, receipt, reduction="sum")
        g_rec = torch.autograd.grad(
            receipt_energy,
            z,
            create_graph=True,
            retain_graph=True,
        )[0].detach()

        cos = F.cosine_similarity(g_cls.flatten(1), g_rec.flatten(1), dim=-1)
        firewall = (cos ** 2).mean()
        return firewall, receipt_loss

    def forward(
        self,
        logits: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
        receipt: torch.Tensor,
        cf_logits: torch.Tensor | None = None,
        cf_receipt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ce = F.cross_entropy(logits, y)
        firewall, receipt_loss = self._firewall_penalty(logits, z, y, receipt)

        if cf_logits is None or cf_receipt is None:
            cf_loss = logits.new_zeros(())
        else:
            p = F.log_softmax(logits / self.tau, dim=-1)
            q = F.softmax(cf_logits.detach() / self.tau, dim=-1)
            receipt_gap = (receipt - cf_receipt).norm(dim=-1).detach()
            weight = torch.sigmoid(receipt_gap).mean()
            cf_loss = weight * F.kl_div(p, q, reduction="batchmean") * (self.tau ** 2)

        total = (
            ce
            + self.lambda_firewall * firewall
            + self.lambda_receipt * receipt_loss
            + self.lambda_cf * cf_loss
        )
        return {
            "loss": total,
            "ce": ce.detach(),
            "receipt_loss": receipt_loss.detach(),
            "firewall": firewall.detach(),
            "cf_loss": cf_loss.detach(),
        }


class PolicyShiftRobustClassifier(nn.Module):
    def __init__(self, value_encoder: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.value_encoder = value_encoder
        self.classifier = classifier

    def forward(
        self,
        times: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.value_encoder(times=times, values=values, mask=mask)
        z.requires_grad_(True)
        logits = self.classifier(z)
        return logits, z
```

### Training Sketch

```python
receipt_builder = SamplingReceiptBuilder()
criterion = ReceiptFirewallLoss(receipt_dim=3 + num_channels, latent_dim=hidden_dim)

for batch in loader:
    logits, z = model(batch["times"], batch["values"], batch["mask"])
    cf_logits, _ = model(batch["cf_times"], batch["cf_values"], batch["cf_mask"])

    receipt = receipt_builder(batch["times"], batch["mask"])
    cf_receipt = receipt_builder(batch["cf_times"], batch["cf_mask"])

    out = criterion(
        logits=logits,
        z=z,
        y=batch["label"],
        receipt=receipt,
        cf_logits=cf_logits,
        cf_receipt=cf_receipt,
    )
    out["loss"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

## Expected Experimental Signature

1. Under in-distribution irregular sampling, accuracy should remain close to the base sampling-decoupled model.
2. Under shifted sampling policies, the method should reduce performance drop because classifier gradients become orthogonal to receipt-explainable latent directions.
3. The strongest diagnostic is not only accuracy: report the cosine similarity between class-logit gradients and receipt gradients before and after training. A successful firewall should sharply reduce this value.

## Suggested Ablations

1. Base sampling-decoupled model.
2. Base model plus ordinary counterfactual logit consistency.
3. Base model plus receipt prediction only.
4. Full Receipt Firewall with gradient nulling.
5. Full model without receipt-distance weighting in the counterfactual KL.

## One-sentence Pitch

Instead of hiding sampling information or using it as a feature, **Receipt Firewall makes the sampler testify against itself**: every counterfactual schedule produces an audit receipt, and the classifier is trained to have zero gradient access to receipt-explainable policy shortcuts.
