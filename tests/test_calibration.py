"""Temperature scaling corrects confidence without touching ranking."""

from __future__ import annotations

import pytest
import torch

from aml_gnn.engine.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from aml_gnn.engine.metrics import evaluate_predictions


@pytest.fixture()
def overconfident() -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits that are ranked well but scaled far too aggressively."""
    generator = torch.Generator().manual_seed(0)
    targets = (torch.rand(600, generator=generator) < 0.3).long()
    signal = targets.float() + 0.6 * torch.randn(600, generator=generator)
    # A 12x gap is enough to push most rows to 0 or 1 under a plain softmax.
    logits = torch.stack([-signal * 12.0, signal * 12.0], dim=1)
    return logits, targets


def test_fitted_temperature_softens_an_overconfident_model(overconfident) -> None:
    """The fitted temperature is greater than one when logits are inflated."""
    logits, targets = overconfident
    assert fit_temperature(logits, targets) > 1.0


def test_calibration_cannot_reorder_predictions(overconfident) -> None:
    """Dividing by a positive scalar preserves every strict ordering."""
    logits, targets = overconfident
    before = torch.softmax(logits, dim=-1)[:, 1]
    after = apply_temperature(logits, fit_temperature(logits, targets))
    order = before.argsort()
    ranked_before, ranked_after = before[order], after[order]
    strictly_increasing = ranked_before[:-1] < ranked_before[1:]
    assert torch.all(ranked_after[:-1][strictly_increasing]
                     <= ranked_after[1:][strictly_increasing])


def test_calibration_can_only_help_ranking(overconfident) -> None:
    """AUC never falls, and rises when saturation had collapsed rows into ties.

    A float32 softmax pins an inflated logit to exactly 1.0, so rows that the
    model actually ranks differently arrive at the metric indistinguishable.
    Rescaling separates them again, which is why calibration is not purely
    cosmetic: it recovers ordering information that overflow had thrown away.
    """
    logits, targets = overconfident
    before = torch.softmax(logits, dim=-1)[:, 1]
    after = apply_temperature(logits, fit_temperature(logits, targets))
    baseline = evaluate_predictions(targets, before, threshold=0.5)
    calibrated = evaluate_predictions(targets, after, threshold=0.5)
    assert calibrated.roc_auc >= baseline.roc_auc
    assert len(before.unique()) < len(after.unique())


def test_calibration_reduces_saturation(overconfident) -> None:
    """Fewer rows pin to the extremes once the logits are rescaled."""
    logits, targets = overconfident
    before = torch.softmax(logits, dim=-1)[:, 1]
    after = apply_temperature(logits, fit_temperature(logits, targets))
    pinned = lambda p: int(((p < 1e-4) | (p > 1 - 1e-4)).sum())  # noqa: E731
    assert pinned(after) < pinned(before)


def test_calibration_improves_the_fitted_split(overconfident) -> None:
    """Calibration error falls on the split the temperature was fitted on."""
    logits, targets = overconfident
    before = torch.softmax(logits, dim=-1)[:, 1]
    after = apply_temperature(logits, fit_temperature(logits, targets))
    assert expected_calibration_error(after, targets) < expected_calibration_error(
        before, targets
    )


def test_temperature_of_one_is_a_no_op(overconfident) -> None:
    """Passing 1.0 reproduces the uncalibrated probabilities exactly."""
    logits, _ = overconfident
    assert torch.allclose(
        apply_temperature(logits, 1.0), torch.softmax(logits.float(), dim=-1)[:, 1]
    )


def test_a_non_positive_temperature_is_rejected(overconfident) -> None:
    """Zero or negative temperatures are a programming error, not a setting."""
    logits, _ = overconfident
    with pytest.raises(ValueError, match="must be positive"):
        apply_temperature(logits, 0.0)


def test_empty_input_falls_back_to_one() -> None:
    """Nothing to fit means no rescaling, rather than a crash."""
    assert fit_temperature(torch.empty(0, 2), torch.empty(0, dtype=torch.long)) == 1.0


def test_perfect_calibration_scores_zero_error() -> None:
    """A predictor whose claims match outcomes exactly has no calibration error."""
    probabilities = torch.cat([torch.zeros(50), torch.ones(50)])
    targets = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    assert expected_calibration_error(probabilities, targets) == pytest.approx(0.0)
