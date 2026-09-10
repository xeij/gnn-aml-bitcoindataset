"""Metric correctness, especially under class imbalance."""

from __future__ import annotations

import pytest
import torch

from aml_gnn.engine.metrics import (
    evaluate_predictions,
    precision_recall_points,
    select_threshold,
)


def test_perfect_predictions_score_one() -> None:
    """A perfect ranking yields F1 = 1 with no false alerts."""
    y = torch.tensor([0, 0, 1, 1])
    probabilities = torch.tensor([0.01, 0.02, 0.98, 0.99])
    metrics = evaluate_predictions(y, probabilities, split="test")
    assert metrics.illicit_f1 == pytest.approx(1.0)
    assert metrics.false_positives == 0 and metrics.false_negatives == 0


def test_majority_class_predictor_is_exposed() -> None:
    """High accuracy with zero recall is exactly what the metrics must reveal."""
    y = torch.cat([torch.zeros(95, dtype=torch.long), torch.ones(5, dtype=torch.long)])
    probabilities = torch.full((100,), 0.01)
    metrics = evaluate_predictions(y, probabilities, split="test")
    assert metrics.accuracy == pytest.approx(0.95)
    assert metrics.illicit_f1 == 0.0
    assert metrics.illicit_recall == 0.0


def test_confusion_counts_sum_to_the_population() -> None:
    """Every evaluated node lands in exactly one confusion cell."""
    y = torch.tensor([0, 1, 0, 1, 1])
    probabilities = torch.tensor([0.1, 0.9, 0.7, 0.2, 0.8])
    metrics = evaluate_predictions(y, probabilities, split="test")
    total = (
        metrics.true_positives + metrics.false_positives
        + metrics.true_negatives + metrics.false_negatives
    )
    assert total == y.numel() == metrics.num_nodes


def test_alert_precision_matches_the_confusion_matrix() -> None:
    """The operational alert-precision view agrees with the raw counts."""
    y = torch.tensor([0, 1, 0, 1])
    probabilities = torch.tensor([0.9, 0.9, 0.1, 0.1])
    metrics = evaluate_predictions(y, probabilities, split="test")
    assert metrics.alerts_raised == 2
    assert metrics.alert_precision == pytest.approx(0.5)


def test_raising_the_threshold_cannot_increase_recall() -> None:
    """Recall is monotonically non-increasing in the threshold."""
    y = torch.tensor([0, 1, 1, 0, 1])
    probabilities = torch.tensor([0.3, 0.6, 0.8, 0.4, 0.55])
    lenient = evaluate_predictions(y, probabilities, split="t", threshold=0.5)
    strict = evaluate_predictions(y, probabilities, split="t", threshold=0.7)
    assert strict.illicit_recall <= lenient.illicit_recall


def test_selected_threshold_beats_the_default() -> None:
    """Tuning a threshold must not be worse than leaving it at 0.5."""
    y = torch.cat([torch.zeros(90, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    probabilities = torch.cat([torch.rand(90) * 0.4, 0.4 + torch.rand(10) * 0.3])
    threshold, f1 = select_threshold(y, probabilities)
    default = evaluate_predictions(y, probabilities, split="t", threshold=0.5)
    assert 0.0 < threshold < 1.0
    assert f1 >= default.illicit_f1 - 1e-9


def test_single_class_split_yields_nan_ranking_metrics() -> None:
    """Ranking metrics are undefined, not zero, when one class is absent."""
    y = torch.zeros(10, dtype=torch.long)
    metrics = evaluate_predictions(y, torch.rand(10), split="t")
    assert metrics.roc_auc != metrics.roc_auc  # NaN


def test_per_time_step_metrics_cover_every_step() -> None:
    """A per-step breakdown reports one value per distinct step."""
    y = torch.tensor([0, 1, 0, 1])
    probabilities = torch.tensor([0.1, 0.9, 0.2, 0.8])
    steps = torch.tensor([1, 1, 2, 2])
    metrics = evaluate_predictions(y, probabilities, split="t", time_step=steps)
    assert set(metrics.per_time_step) == {1, 2}


def test_mismatched_lengths_are_rejected() -> None:
    """Label and score arrays must describe the same nodes."""
    with pytest.raises(ValueError, match="mismatch"):
        evaluate_predictions(torch.tensor([0, 1]), torch.rand(3), split="t")


def test_precision_recall_curve_is_subsampled() -> None:
    """The plotting curve honours its point budget."""
    y = torch.cat([torch.zeros(500, dtype=torch.long), torch.ones(60, dtype=torch.long)])
    recall, precision = precision_recall_points(y, torch.rand(560), max_points=50)
    assert len(recall) == len(precision) <= 50
