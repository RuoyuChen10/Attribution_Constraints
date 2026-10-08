"""Core utilities for subset-attribution human-prior alignment.

The module intentionally contains no model- or dataset-specific code.  It
implements the paper objective and the adaptive log-ratio variant, plus the
construction of masked examples from an already-computed LIMA ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import torch
import torch.nn.functional as F


LossVariant = Literal["paper", "adaptive_log"]


@dataclass(frozen=True)
class AlignmentLossConfig:
    """Configuration shared by deviation and redundancy losses."""

    variant: LossVariant = "paper"
    insertion_weight: float = 2.0
    collaboration_weight: float = 1.0
    adaptive_beta: float = 2.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.variant not in ("paper", "adaptive_log"):
            raise ValueError(f"Unsupported loss variant: {self.variant}")
        if self.insertion_weight < 0 or self.collaboration_weight < 0:
            raise ValueError("Set-function weights must be non-negative")
        if self.normalizer <= 0:
            raise ValueError("At least one set-function weight must be positive")
        if self.adaptive_beta <= 0:
            raise ValueError("adaptive_beta must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @property
    def normalizer(self) -> float:
        return self.insertion_weight + self.collaboration_weight


@dataclass
class DeviationExample:
    insertion: torch.Tensor
    deletion: torch.Tensor
    label: torch.Tensor
    best_human_gain: torch.Tensor
    full_confidence: torch.Tensor


@dataclass
class RedundancyExample:
    insertion_after: torch.Tensor
    deletion_after: torch.Tensor
    insertion_before: torch.Tensor
    deletion_before: torch.Tensor
    label: torch.Tensor
    best_human_gain: torch.Tensor
    full_confidence: torch.Tensor


@dataclass
class SampleAlignmentExamples:
    deviations: list[DeviationExample] = field(default_factory=list)
    redundancies: list[RedundancyExample] = field(default_factory=list)
    prior_consistent: torch.Tensor | None = None
    normalized_gains: torch.Tensor | None = None
    best_human_gain: torch.Tensor | None = None


@dataclass
class AlignmentBatchResult:
    """Per-example losses and diagnostics for one alignment component."""

    loss: torch.Tensor
    bad_gain: torch.Tensor
    best_human_gain: torch.Tensor
    excess: torch.Tensor
    reference: torch.Tensor
    adaptive_weight: torch.Tensor
    satisfied: torch.Tensor


def target_probabilities(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return p(y|x) for every row without materializing Python indices."""

    if logits.ndim != 2:
        raise ValueError(f"Expected [batch, classes] logits, got {tuple(logits.shape)}")
    labels = labels.to(device=logits.device, dtype=torch.long)
    if labels.ndim != 1 or labels.numel() != logits.size(0):
        raise ValueError("labels must have shape [batch]")
    return logits.softmax(dim=-1).gather(1, labels[:, None]).squeeze(1)


def set_utility(
    insertion_logits: torch.Tensor,
    deletion_logits: torch.Tensor,
    labels: torch.Tensor,
    config: AlignmentLossConfig,
) -> torch.Tensor:
    """Compute F(S)=a*p(y|x_S)+b*(1-p(y|x_not_S))."""

    p_insertion = target_probabilities(insertion_logits, labels)
    p_deletion = target_probabilities(deletion_logits, labels)
    return (
        config.insertion_weight * p_insertion
        + config.collaboration_weight * (1.0 - p_deletion)
    )


def normalized_ranking_gains(
    set_scores: torch.Tensor | Sequence[float],
    normalizer: float,
) -> torch.Tensor:
    """Convert prefix F scores into normalized per-rank marginal gains.

    The paper treats the first selected region as F(v_pi1), while ranks r>=2
    use F(prefix_r)-F(prefix_{r-1}).
    """

    scores = torch.as_tensor(set_scores)
    if scores.ndim != 1:
        raise ValueError("set_scores must be one-dimensional")
    if scores.numel() == 0:
        return scores
    gains = torch.empty_like(scores)
    gains[0] = scores[0]
    gains[1:] = scores[1:] - scores[:-1]
    return gains / normalizer


def best_human_gain(
    normalized_gains: torch.Tensor,
    prior_consistent: torch.Tensor,
) -> torch.Tensor:
    """Return the detached maximum positive gain among prior regions."""

    if normalized_gains.ndim != 1 or prior_consistent.ndim != 1:
        raise ValueError("gains and prior_consistent must be one-dimensional")
    if normalized_gains.numel() != prior_consistent.numel():
        raise ValueError("gains and prior_consistent must have equal lengths")
    prior_consistent = prior_consistent.to(
        device=normalized_gains.device, dtype=torch.bool
    )
    candidates = normalized_gains[prior_consistent]
    if candidates.numel() == 0:
        result = normalized_gains.new_zeros(())
    else:
        result = candidates.relu().max()
    return result.detach()


def reference_scale(
    human_gain: torch.Tensor,
    full_confidence: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """r=max(best human gain, full-image uncertainty, epsilon), detached."""

    human_gain, full_confidence = torch.broadcast_tensors(
        human_gain, full_confidence
    )
    uncertainty = 1.0 - full_confidence
    floor = torch.full_like(uncertainty, epsilon)
    return torch.maximum(torch.maximum(human_gain, uncertainty), floor).detach()


def excess_over_human_gain(
    bad_gain: torch.Tensor,
    human_gain: torch.Tensor,
) -> torch.Tensor:
    """Positive evidence gain above the best human-consistent gain."""

    return F.relu(bad_gain - human_gain.detach())


def adaptive_log_penalty(
    excess: torch.Tensor,
    reference: torch.Tensor,
    beta: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Log-ratio penalty 2/beta*log(1+(q/r)^beta).

    The numerator is intentionally not epsilon-shifted, so q=0 produces an
    exact zero loss.  In the full objective, the preceding ReLU stops the gap
    gradient at and below the human-gain boundary.
    """

    if beta <= 0:
        raise ValueError("beta must be positive")
    q = excess.clamp_min(0.0)
    r = reference.detach().clamp_min(epsilon)
    return (2.0 / beta) * torch.log1p(torch.pow(q / r, beta))


def adaptive_gradient_weight(
    excess: torch.Tensor,
    reference: torch.Tensor,
    beta: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return 2*q^beta/(q^beta+r^beta), the log-q gradient coefficient."""

    q_beta = torch.pow(excess.detach().clamp_min(0.0), beta)
    r_beta = torch.pow(reference.detach().clamp_min(epsilon), beta)
    return 2.0 * q_beta / (q_beta + r_beta)


def deviation_loss_from_logits(
    insertion_logits: torch.Tensor,
    deletion_logits: torch.Tensor,
    labels: torch.Tensor,
    human_gain: torch.Tensor,
    full_confidence: torch.Tensor,
    config: AlignmentLossConfig,
) -> AlignmentBatchResult:
    """Compute deviation loss for top-ranked off-prior regions."""

    raw_utility = set_utility(insertion_logits, deletion_logits, labels, config)
    bad_gain = raw_utility / config.normalizer
    human_gain = human_gain.to(device=bad_gain.device, dtype=bad_gain.dtype).detach()
    full_confidence = full_confidence.to(
        device=bad_gain.device, dtype=bad_gain.dtype
    ).detach()
    excess = excess_over_human_gain(bad_gain, human_gain)
    reference = reference_scale(human_gain, full_confidence, config.epsilon)
    weight = adaptive_gradient_weight(
        excess, reference, config.adaptive_beta, config.epsilon
    )
    if config.variant == "paper":
        per_example_loss = raw_utility
    else:
        per_example_loss = adaptive_log_penalty(
            excess, reference, config.adaptive_beta, config.epsilon
        )
    return AlignmentBatchResult(
        loss=per_example_loss,
        bad_gain=bad_gain.detach(),
        best_human_gain=human_gain,
        excess=excess.detach(),
        reference=reference,
        adaptive_weight=weight,
        satisfied=(bad_gain.detach() <= human_gain),
    )


def redundancy_loss_from_logits(
    insertion_after_logits: torch.Tensor,
    deletion_after_logits: torch.Tensor,
    insertion_before_logits: torch.Tensor,
    deletion_before_logits: torch.Tensor,
    labels: torch.Tensor,
    human_gain: torch.Tensor,
    full_confidence: torch.Tensor,
    config: AlignmentLossConfig,
) -> AlignmentBatchResult:
    """Compute redundancy loss with the before-prefix as a detached anchor."""

    after = set_utility(
        insertion_after_logits, deletion_after_logits, labels, config
    )
    before = set_utility(
        insertion_before_logits, deletion_before_logits, labels, config
    ).detach()
    raw_delta = after - before
    bad_gain = raw_delta / config.normalizer
    human_gain = human_gain.to(device=bad_gain.device, dtype=bad_gain.dtype).detach()
    full_confidence = full_confidence.to(
        device=bad_gain.device, dtype=bad_gain.dtype
    ).detach()
    excess = excess_over_human_gain(bad_gain, human_gain)
    reference = reference_scale(human_gain, full_confidence, config.epsilon)
    weight = adaptive_gradient_weight(
        excess, reference, config.adaptive_beta, config.epsilon
    )
    if config.variant == "paper":
        per_example_loss = F.relu(raw_delta)
    else:
        per_example_loss = adaptive_log_penalty(
            excess, reference, config.adaptive_beta, config.epsilon
        )
    return AlignmentBatchResult(
        loss=per_example_loss,
        bad_gain=bad_gain.detach(),
        best_human_gain=human_gain,
        excess=excess.detach(),
        reference=reference,
        adaptive_weight=weight,
        satisfied=(bad_gain.detach() <= human_gain),
    )


def region_prior_consistency(
    ranked_regions: torch.Tensor,
    prior_mask: torch.Tensor,
    overlap_threshold: float,
) -> torch.Tensor:
    """Classify regions using region coverage by the human-prior mask."""

    if ranked_regions.ndim != 3:
        raise ValueError("ranked_regions must have shape [k, height, width]")
    prior_mask = prior_mask.squeeze().to(
        device=ranked_regions.device, dtype=torch.bool
    )
    if prior_mask.shape != ranked_regions.shape[-2:]:
        raise ValueError("prior_mask and ranked_regions must have equal spatial size")
    regions = ranked_regions.to(dtype=torch.bool)
    area = regions.sum(dim=(1, 2)).float()
    overlap = (regions & prior_mask.unsqueeze(0)).sum(dim=(1, 2)).float()
    coverage = overlap / area.clamp_min(1.0)
    return coverage >= overlap_threshold


def build_alignment_examples(
    image: torch.Tensor,
    prior_mask: torch.Tensor,
    ranked_regions: torch.Tensor,
    set_scores: torch.Tensor | Sequence[float],
    label: torch.Tensor,
    full_confidence: torch.Tensor,
    overlap_threshold: float,
    normalizer: float,
) -> SampleAlignmentExamples:
    """Build top-1 deviation and every higher-order redundancy example.

    Deviation and redundancy are deliberately independent: a bad top-ranked
    region does not suppress construction of bad higher-order examples.
    """

    if image.ndim != 3:
        raise ValueError("image must have shape [channels, height, width]")
    regions = ranked_regions.to(device=image.device, dtype=torch.bool)
    if regions.ndim != 3 or regions.shape[-2:] != image.shape[-2:]:
        raise ValueError("ranked_regions must align with image spatial dimensions")
    if regions.size(0) == 0:
        return SampleAlignmentExamples()

    consistent = region_prior_consistency(
        regions, prior_mask, overlap_threshold
    )
    scores = torch.as_tensor(set_scores, device=image.device, dtype=image.dtype)
    if scores.numel() != regions.size(0):
        raise ValueError("set_scores length must match ranked_regions")
    gains = normalized_ranking_gains(scores, normalizer)
    human_gain = best_human_gain(gains, consistent)

    label = label.to(device=image.device, dtype=torch.long).reshape(())
    full_confidence = full_confidence.to(
        device=image.device, dtype=image.dtype
    ).reshape(()).detach()
    examples = SampleAlignmentExamples(
        prior_consistent=consistent.detach(),
        normalized_gains=gains.detach(),
        best_human_gain=human_gain,
    )

    if not bool(consistent[0]):
        first = regions[0].unsqueeze(0)
        examples.deviations.append(
            DeviationExample(
                insertion=image * first,
                deletion=image * (~first),
                label=label,
                best_human_gain=human_gain,
                full_confidence=full_confidence,
            )
        )

    for rank in range(1, regions.size(0)):
        if bool(consistent[rank]):
            continue
        before = regions[:rank].any(dim=0, keepdim=True)
        after = regions[: rank + 1].any(dim=0, keepdim=True)
        examples.redundancies.append(
            RedundancyExample(
                insertion_after=image * after,
                deletion_after=image * (~after),
                insertion_before=image * before,
                deletion_before=image * (~before),
                label=label,
                best_human_gain=human_gain,
                full_confidence=full_confidence,
            )
        )
    return examples


def flatten_examples(
    samples: Iterable[SampleAlignmentExamples],
) -> tuple[list[DeviationExample], list[RedundancyExample]]:
    deviations: list[DeviationExample] = []
    redundancies: list[RedundancyExample] = []
    for sample in samples:
        deviations.extend(sample.deviations)
        redundancies.extend(sample.redundancies)
    return deviations, redundancies
