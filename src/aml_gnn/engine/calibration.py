"""Temperature scaling, so a reported probability means what it says.

Cross-entropy has no reason to stop once the classes are separated: driving the
logits further apart keeps shrinking the loss, and nothing penalises their
magnitude.  On this dataset that produces logit gaps past 3,000, which the
float32 softmax renders as exactly 1.0 -- the model reports certainty on
transactions it gets wrong a quarter of the time.

Temperature scaling is the standard remedy (Guo et al., 2017): divide the logits
by one positive scalar fitted on the validation split.  It is the smallest
possible correction, and deliberately so -- a single monotonic rescaling cannot
reorder anything, so ranking, ROC-AUC and average precision are untouched and
only the numbers change.  The decision threshold does move, because it lives on
the rescaled scale, so it is refitted immediately afterwards.

Fitting on validation matters for the same reason threshold tuning does: a
temperature fitted on test would be a second, hidden training run.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..logging_utils import get_logger

logger = get_logger(__name__)

#: Bounds on the search. A temperature below 1 sharpens rather than softens,
#: which is occasionally right, and the upper bound is far past anything a
#: sanely-trained model needs.
_MIN_TEMPERATURE: float = 0.05
_MAX_TEMPERATURE: float = 1000.0


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Return ``P(illicit)`` from logits divided by ``temperature``."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    scaled: Tensor = torch.softmax(logits.float() / temperature, dim=-1)[:, 1]
    return scaled


def fit_temperature(
    logits: Tensor,
    targets: Tensor,
    *,
    iterations: int = 60,
) -> float:
    """Fit the temperature that minimises validation negative log-likelihood.

    Golden-section search over ``log T`` rather than gradient descent: the
    objective is one-dimensional, smooth and unimodal, so a bracketing search
    converges in a fixed number of steps with no learning rate to choose and no
    chance of diverging on the very large logits that motivate the fix.

    Args:
        logits: ``(n, 2)`` raw model outputs for the fitting split.
        targets: ``(n,)`` 0/1 labels for the same rows.
        iterations: Search steps; 60 pins ``log T`` to well under 1e-6.

    Returns:
        The fitted temperature.  Returns ``1.0`` when there is nothing to fit.

    """
    if logits.numel() == 0 or targets.numel() == 0:
        logger.warning("No validation rows to calibrate on; temperature stays 1.0")
        return 1.0

    values = logits.detach().float().cpu()
    labels = targets.detach().long().cpu()

    def loss_at(log_temperature: float) -> float:
        scaled = values / float(torch.exp(torch.tensor(log_temperature)))
        return float(torch.nn.functional.cross_entropy(scaled, labels))

    low, high = torch.log(torch.tensor(_MIN_TEMPERATURE)), torch.log(
        torch.tensor(_MAX_TEMPERATURE)
    )
    left, right = float(low), float(high)
    phi = (5.0**0.5 - 1.0) / 2.0
    probe_a, probe_b = right - phi * (right - left), left + phi * (right - left)
    loss_a, loss_b = loss_at(probe_a), loss_at(probe_b)

    for _ in range(iterations):
        if loss_a < loss_b:
            right, probe_b, loss_b = probe_b, probe_a, loss_a
            probe_a = right - phi * (right - left)
            loss_a = loss_at(probe_a)
        else:
            left, probe_a, loss_a = probe_a, probe_b, loss_b
            probe_b = left + phi * (right - left)
            loss_b = loss_at(probe_b)

    temperature = float(torch.exp(torch.tensor((left + right) / 2.0)))
    before, after = loss_at(float(torch.tensor(0.0))), loss_at((left + right) / 2.0)
    logger.info(
        "Calibrated temperature %.4f (validation NLL %.4f -> %.4f)",
        temperature, before, after,
    )
    return temperature


def expected_calibration_error(
    probabilities: Tensor,
    targets: Tensor,
    *,
    bins: int = 15,
) -> float:
    """Return the expected calibration error, the gap between claim and outcome.

    Each node is placed in a confidence bin and the bin's mean predicted
    probability is compared with the share that really are illicit; the result
    is the average gap, weighted by bin population.  Zero means the numbers mean
    what they say.

    Args:
        probabilities: ``P(illicit)`` per node.
        targets: 0/1 labels for the same nodes.
        bins: Number of equal-width confidence bins.

    """
    if probabilities.numel() == 0:
        return 0.0
    scores = probabilities.detach().float().cpu()
    labels = targets.detach().float().cpu()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        low, high = float(edges[index]), float(edges[index + 1])
        selected = (scores > low) & (scores <= high) if index else (scores <= high)
        if not selected.any():
            continue
        share = float(selected.float().mean())
        total += share * abs(float(scores[selected].mean()) - float(labels[selected].mean()))
    return total
