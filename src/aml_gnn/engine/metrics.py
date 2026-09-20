"""Evaluation metrics for a heavily imbalanced binary node-classification task.

Accuracy is deliberately reported but never optimised.  With 6.5% illicit
nodes in the test window, a model that predicts "licit" for everything scores
93.5% accuracy and is worthless.  The metric that drives checkpointing,
early stopping and every headline number here is the **illicit-class F1**,
alongside average precision, which summarises the precision/recall curve
without depending on a threshold at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import Tensor

from ..logging_utils import get_logger

logger = get_logger(__name__)

#: Metric keys that may be used as an early-stopping / checkpoint monitor.
MONITORABLE: tuple[str, ...] = (
    "illicit_f1",
    "illicit_precision",
    "illicit_recall",
    "macro_f1",
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "accuracy",
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """A full evaluation of one split at one decision threshold."""

    split: str
    num_nodes: int
    threshold: float
    accuracy: float
    balanced_accuracy: float
    illicit_precision: float
    illicit_recall: float
    illicit_f1: float
    licit_f1: float
    macro_f1: float
    roc_auc: float
    average_precision: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    per_time_step: Mapping[int, float] = field(default_factory=dict)

    @property
    def confusion(self) -> list[list[int]]:
        """Confusion matrix as ``[[TN, FP], [FN, TP]]``."""
        return [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]

    @property
    def alerts_raised(self) -> int:
        """Number of transactions the model would send to an analyst."""
        return self.true_positives + self.false_positives

    @property
    def alert_precision(self) -> float:
        """Share of raised alerts that are genuinely illicit."""
        return self.true_positives / self.alerts_raised if self.alerts_raised else 0.0

    def get(self, key: str) -> float:
        """Return a metric by name, for monitor-driven callbacks."""
        value = getattr(self, key, None)
        if not isinstance(value, (int, float)):
            raise KeyError(f"{key!r} is not a scalar metric. Valid monitors: {MONITORABLE}")
        return float(value)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of every field."""
        payload = asdict(self)
        payload["per_time_step"] = {int(k): float(v) for k, v in self.per_time_step.items()}
        return payload

    def __str__(self) -> str:
        """Return a single-line summary for console logs."""
        return (
            f"{self.split:<5} n={self.num_nodes:>6,} @thr={self.threshold:.2f} | "
            f"illicit P={self.illicit_precision:.4f} R={self.illicit_recall:.4f} "
            f"F1={self.illicit_f1:.4f} | AP={self.average_precision:.4f} "
            f"AUC={self.roc_auc:.4f} | acc={self.accuracy:.4f}"
        )


def _as_numpy(tensor: Tensor) -> np.ndarray:
    """Detach a tensor to a CPU numpy array."""
    return tensor.detach().cpu().numpy()


def evaluate_predictions(
    y_true: Tensor,
    illicit_probability: Tensor,
    *,
    split: str = "eval",
    threshold: float = 0.5,
    time_step: Tensor | None = None,
) -> ClassificationMetrics:
    """Score predicted illicit probabilities against ground-truth labels.

    Args:
        y_true: Long tensor of 0/1 labels for the evaluated nodes only.
        illicit_probability: Float tensor of ``P(illicit)`` for the same nodes.
        split: Label used in logs and plots.
        threshold: Probability above which a node is flagged illicit.
        time_step: Optional per-node time step; when given, illicit F1 is also
            computed within each step so temporal degradation is visible.

    Returns:
        A populated :class:`ClassificationMetrics`.

    """
    if y_true.numel() != illicit_probability.numel():
        raise ValueError(
            f"Label/probability length mismatch: {y_true.numel()} vs "
            f"{illicit_probability.numel()}"
        )
    if y_true.numel() == 0:
        raise ValueError(f"Cannot evaluate split {split!r}: no nodes selected")

    labels = _as_numpy(y_true).astype(int)
    scores = _as_numpy(illicit_probability).astype(float)
    predictions = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()

    both_classes_present = len(np.unique(labels)) > 1
    if both_classes_present:
        roc_auc = float(roc_auc_score(labels, scores))
        average_precision = float(average_precision_score(labels, scores))
        balanced = float(balanced_accuracy_score(labels, predictions))
    else:
        # A single-class window makes ranking metrics undefined rather than zero.
        logger.warning("Split %r contains a single class; ranking metrics set to NaN", split)
        roc_auc = average_precision = balanced = float("nan")

    per_step: dict[int, float] = {}
    if time_step is not None:
        per_step = _per_time_step_f1(labels, predictions, _as_numpy(time_step).astype(int))

    return ClassificationMetrics(
        split=split,
        num_nodes=int(labels.size),
        threshold=float(threshold),
        accuracy=float((predictions == labels).mean()),
        balanced_accuracy=balanced,
        illicit_precision=float(precision[1]),
        illicit_recall=float(recall[1]),
        illicit_f1=float(f1[1]),
        licit_f1=float(f1[0]),
        macro_f1=float(f1.mean()),
        roc_auc=roc_auc,
        average_precision=average_precision,
        true_positives=int(true_positives),
        false_positives=int(false_positives),
        true_negatives=int(true_negatives),
        false_negatives=int(false_negatives),
        per_time_step=per_step,
    )


def _per_time_step_f1(
    labels: np.ndarray, predictions: np.ndarray, steps: np.ndarray
) -> dict[int, float]:
    """Compute illicit F1 within each time step present in ``steps``."""
    results: dict[int, float] = {}
    for step in np.unique(steps):
        selector = steps == step
        if selector.sum() == 0:
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            labels[selector], predictions[selector], labels=[0, 1], zero_division=0
        )
        results[int(step)] = float(f1[1])
    return results


def select_threshold(
    y_true: Tensor,
    illicit_probability: Tensor,
    *,
    objective: str = "f1",
    min_precision: float | None = None,
) -> tuple[float, float]:
    """Choose an operating threshold on a held-out split.

    Fitting the threshold on validation and then freezing it for test keeps the
    reported test number honest; tuning it on test would be a second, hidden
    training run.

    Args:
        y_true: 0/1 labels.
        illicit_probability: ``P(illicit)`` for the same nodes.
        objective: ``"f1"`` maximises illicit F1; ``"recall_at_precision"``
            maximises recall subject to ``min_precision``.
        min_precision: Required precision floor for the constrained objective.

    Returns:
        ``(threshold, achieved_objective_value)``.

    """
    labels = _as_numpy(y_true).astype(int)
    scores = _as_numpy(illicit_probability).astype(float)
    if len(np.unique(labels)) < 2:
        logger.warning("Cannot tune a threshold on a single-class split; defaulting to 0.5")
        return 0.5, 0.0

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]

    if objective == "f1":
        denominator = np.clip(precision + recall, 1e-12, None)
        objective_values = 2 * precision * recall / denominator
    elif objective == "recall_at_precision":
        if min_precision is None:
            raise ValueError("min_precision is required for objective='recall_at_precision'")
        objective_values = np.where(precision >= min_precision, recall, -1.0)
        if objective_values.max() < 0:
            logger.warning(
                "No threshold reaches precision >= %.2f; falling back to max-F1", min_precision
            )
            return select_threshold(y_true, illicit_probability, objective="f1")
    else:
        raise ValueError(f"Unknown threshold objective {objective!r}")

    best = int(np.argmax(objective_values))
    chosen = float(thresholds[best])
    logger.info(
        "Selected threshold %.4f (%s=%.4f, precision=%.4f, recall=%.4f)",
        chosen, objective, objective_values[best], precision[best], recall[best],
    )
    return chosen, float(objective_values[best])


def precision_recall_points(
    y_true: Tensor, illicit_probability: Tensor, *, max_points: int = 300
) -> tuple[list[float], list[float]]:
    """Return a subsampled ``(recall, precision)`` curve for plotting."""
    labels = _as_numpy(y_true).astype(int)
    scores = _as_numpy(illicit_probability).astype(float)
    if len(np.unique(labels)) < 2:
        return [], []
    precision, recall, _ = precision_recall_curve(labels, scores)
    if precision.size > max_points:
        indices = np.linspace(0, precision.size - 1, max_points).astype(int)
        precision, recall = precision[indices], recall[indices]
    return recall.tolist(), precision.tolist()


def masked_evaluate(
    logits: Tensor,
    y: Tensor,
    mask: Tensor,
    *,
    split: str,
    threshold: float = 0.5,
    time_step: Tensor | None = None,
) -> ClassificationMetrics:
    """Evaluate a full-graph logit tensor restricted to one split mask."""
    probability = torch.softmax(logits[mask], dim=-1)[:, 1]
    steps = time_step[mask] if time_step is not None else None
    return evaluate_predictions(
        y[mask], probability, split=split, threshold=threshold, time_step=steps
    )


def format_metric_table(metrics: Sequence[ClassificationMetrics]) -> list[list[str]]:
    """Format several evaluations as rows for the UI results table."""
    rows: list[list[str]] = []
    for metric in metrics:
        rows.append([
            metric.split,
            f"{metric.num_nodes:,}",
            f"{metric.illicit_precision:.4f}",
            f"{metric.illicit_recall:.4f}",
            f"{metric.illicit_f1:.4f}",
            f"{metric.average_precision:.4f}",
            f"{metric.roc_auc:.4f}",
            f"{metric.accuracy:.4f}",
        ])
    return rows


METRIC_TABLE_HEADERS: tuple[str, ...] = (
    "Split", "Nodes", "Precision", "Recall", "F1", "AP", "ROC-AUC", "Accuracy",
)
