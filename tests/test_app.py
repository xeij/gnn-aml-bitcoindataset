"""Figure builders and HTML components render without a browser."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
import torch

from aml_gnn.app import visualisations as viz
from aml_gnn.app.components import metric_headline, status_banner
from aml_gnn.app.theme import normalise_risk
from aml_gnn.engine.metrics import evaluate_predictions


def test_empty_figure_carries_its_message() -> None:
    """The placeholder figure surfaces the reason it is empty."""
    figure = viz.empty_figure("nothing to show")
    assert isinstance(figure, go.Figure)
    assert "nothing to show" in figure.layout.annotations[0].text


def test_risk_ramp_is_anchored_to_the_threshold() -> None:
    """The neutral colour sits exactly on the operating threshold."""
    assert normalise_risk(0.8, 0.8) == pytest.approx(0.5)
    assert normalise_risk(0.0, 0.8) == pytest.approx(0.0)
    assert normalise_risk(1.0, 0.8) == pytest.approx(1.0)


def test_risk_ramp_is_monotonic() -> None:
    """A riskier transaction never maps to a lower ramp position."""
    values = [normalise_risk(p / 20, 0.6) for p in range(21)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_confusion_figure_annotates_every_cell() -> None:
    """All four confusion cells are labelled with their count."""
    metrics = evaluate_predictions(
        torch.tensor([0, 1, 0, 1]), torch.tensor([0.1, 0.9, 0.8, 0.2]), split="test"
    )
    figure = viz.confusion_figure(metrics)
    assert len(figure.layout.annotations) == 4


def test_precision_recall_figure_marks_the_prevalence_baseline() -> None:
    """The curve is drawn against the always-flag reference line."""
    figure = viz.precision_recall_figure([0.0, 0.5, 1.0], [1.0, 0.6, 0.2], baseline=0.065)
    assert len(figure.data) == 1
    assert any(shape.type == "line" for shape in figure.layout.shapes)


def test_temporal_figure_annotates_the_known_shift() -> None:
    """The step-43 market shutdown is called out rather than smoothed over."""
    figure = viz.temporal_f1_figure({40: 0.6, 43: 0.2, 46: 0.1})
    assert any("shutdown" in note.text for note in figure.layout.annotations)


def test_alert_volume_figure_labels_both_series() -> None:
    """Two series means a legend, so identity never rests on colour alone."""
    figure = viz.alert_volume_figure([1, 2], [10, 12], [3, 4])
    assert figure.layout.showlegend
    assert {trace.name for trace in figure.data} == {"alerts raised", "known illicit"}


def test_embedding_figure_separates_the_three_label_classes() -> None:
    """Licit, illicit and unlabelled nodes each get their own labelled series."""
    embeddings = torch.randn(90, 8)
    labels = torch.tensor([0, 1, 2] * 30)
    figure = viz.embedding_figure(embeddings, labels, torch.rand(90))
    assert {trace.name for trace in figure.data} == {"licit", "illicit", "unlabelled"}


def test_attribution_figure_handles_missing_attribution() -> None:
    """A model without attribution degrades to a placeholder, not an error."""
    assert isinstance(viz.feature_attribution_figure({}), go.Figure)


def test_components_emit_html() -> None:
    """The HTML fragments render as markup."""
    assert status_banner("hello", tone="ok").startswith("<div")
    assert "22px" in metric_headline([("Metric", "0.5")])
