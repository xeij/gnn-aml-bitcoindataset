"""Plotly figure builders for the investigation interface.

Every figure follows the same rules: transparent surface, recessive grid and
axis chrome, thin marks, hover detail on every mark, and a legend whenever more
than one series is present so identity never rests on colour alone.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch
from torch import Tensor

from ..data.splits import CLASS_NAMES, ILLICIT_LABEL, LICIT_LABEL, UNKNOWN_LABEL
from ..engine.inference import SubgraphExplanation
from ..engine.metrics import ClassificationMetrics
from ..logging_utils import get_logger
from .theme import (
    EDGE_LINE,
    FOCUS_RING,
    ILLICIT_RED,
    LICIT_BLUE,
    NEUTRAL_GREY,
    RISK_COLORSCALE,
    SEQUENTIAL_BLUE,
    TEXT_MUTED,
    UNKNOWN_GREY,
    axis_style,
    base_layout,
    normalise_risk,
)

logger = get_logger(__name__)

#: Marker sizes: the investigated node is deliberately larger than its context.
_CENTER_MARKER: int = 26
_MIN_MARKER: int = 9
_MAX_MARKER: int = 22


def empty_figure(message: str, *, height: int = 380) -> go.Figure:
    """Return a placeholder figure carrying an explanatory message."""
    figure = go.Figure()
    figure.update_layout(**base_layout(height=height))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 13, "color": TEXT_MUTED},
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    return figure


def _class_name(label: int) -> str:
    """Return a display name for a class code, including ``unknown``."""
    return "unknown" if label == UNKNOWN_LABEL else CLASS_NAMES[label]


def subgraph_figure(
    explanation: SubgraphExplanation,
    *,
    threshold: float,
    layout_seed: int = 7,
) -> go.Figure:
    """Draw a scored k-hop neighbourhood as a force-directed network.

    Node fill encodes risk on the threshold-anchored diverging ramp; node size
    encodes degree within the neighbourhood; the investigated transaction gets
    an amber ring so it is findable without relying on colour.  Edge opacity
    and width encode attention where the architecture provides it, which turns
    the picture into an argument -- the thick edges are the ones the model
    actually used.

    Args:
        explanation: A scored neighbourhood from the inference service.
        threshold: Current operating threshold, used to anchor the colour ramp.
        layout_seed: Seed for the spring layout, so repeated views are stable.
    """
    num_nodes = explanation.num_nodes
    if num_nodes == 0:
        return empty_figure("No neighbourhood to display.")

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    edges = explanation.edge_index.t().tolist()
    graph.add_edges_from((int(u), int(v)) for u, v in edges if u != v)

    # k scales the ideal edge length; loosening it keeps dense hubs readable.
    positions = nx.spring_layout(
        graph, seed=layout_seed, k=1.6 / np.sqrt(max(num_nodes, 1)), iterations=90
    )
    coordinates = np.array([positions[i] for i in range(num_nodes)])

    center_local = int(
        (explanation.node_indices == explanation.center.node_index)
        .nonzero(as_tuple=False)
        .view(-1)[0]
    )

    figure = go.Figure()
    _add_edge_traces(figure, explanation, coordinates, edges)
    _add_node_trace(figure, explanation, coordinates, graph, threshold, center_local)

    centre = explanation.center
    title = (
        f"Transaction {centre.tx_id} · {explanation.num_nodes} nodes · "
        f"{explanation.num_edges} edges · {explanation.num_flagged} above threshold"
    )
    if explanation.truncated:
        title += " · truncated to nearest neighbours"

    layout = base_layout(title, height=520)
    layout["showlegend"] = True
    layout["legend"] = {
        "orientation": "h", "yanchor": "bottom", "y": -0.12,
        "xanchor": "left", "x": 0.0,
        "font": {"size": 11, "color": TEXT_MUTED},
        "bgcolor": "rgba(0,0,0,0)",
    }
    figure.update_layout(**layout)
    figure.update_xaxes(visible=False, showgrid=False)
    figure.update_yaxes(visible=False, showgrid=False, scaleanchor="x", scaleratio=1)
    return figure


def _add_edge_traces(
    figure: go.Figure,
    explanation: SubgraphExplanation,
    coordinates: np.ndarray,
    edges: Sequence[Sequence[int]],
) -> None:
    """Add edges, split into importance bands so width can vary per band.

    Plotly gives a line trace one width, so varying edge weight means bucketing
    rather than drawing thousands of individual traces.
    """
    importance = explanation.edge_importance
    if importance is not None and importance.numel() == len(edges) and importance.numel():
        weights = importance.numpy()
        span = float(weights.max() - weights.min())
        scaled = (weights - weights.min()) / span if span > 1e-9 else np.full_like(weights, 0.5)
    else:
        scaled = np.full(len(edges), 0.5)

    bands = [(0.0, 0.5, 0.8, 0.18), (0.5, 0.85, 1.5, 0.34), (0.85, 1.01, 3.0, 0.65)]
    for low, high, width, opacity in bands:
        xs: list[Optional[float]] = []
        ys: list[Optional[float]] = []
        selected = 0
        for (source, target), weight in zip(edges, scaled):
            if source == target or not (low <= weight < high):
                continue
            selected += 1
            xs.extend([coordinates[source, 0], coordinates[target, 0], None])
            ys.extend([coordinates[source, 1], coordinates[target, 1], None])
        if not selected:
            continue
        figure.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line={"width": width, "color": EDGE_LINE},
                opacity=opacity, hoverinfo="skip", showlegend=False,
            )
        )


def _add_node_trace(
    figure: go.Figure,
    explanation: SubgraphExplanation,
    coordinates: np.ndarray,
    graph: nx.Graph,
    threshold: float,
    center_local: int,
) -> None:
    """Add the node scatter, plus the focus ring on the investigated node."""
    probabilities = explanation.illicit_probability.numpy()
    labels = explanation.labels.numpy()
    tx_ids = explanation.tx_ids.numpy()
    steps = explanation.time_steps.numpy()
    degrees = np.array([graph.degree(i) for i in range(explanation.num_nodes)], dtype=float)

    colour_values = [float(normalise_risk(float(p), threshold)) for p in probabilities]
    span = degrees.max() - degrees.min()
    sizes = (
        np.full_like(degrees, (_MIN_MARKER + _MAX_MARKER) / 2)
        if span < 1e-9
        else _MIN_MARKER + (degrees - degrees.min()) / span * (_MAX_MARKER - _MIN_MARKER)
    )
    sizes[center_local] = _CENTER_MARKER

    hover = [
        f"<b>tx {tx_ids[i]}</b><br>P(illicit) {probabilities[i]:.4f}"
        f"<br>ground truth: {_class_name(int(labels[i]))}"
        f"<br>time step {int(steps[i])} · degree {int(degrees[i])}"
        + ("<br><i>under investigation</i>" if i == center_local else "")
        for i in range(explanation.num_nodes)
    ]

    figure.add_trace(
        go.Scatter(
            x=coordinates[:, 0], y=coordinates[:, 1], mode="markers",
            marker={
                "size": sizes.tolist(),
                "color": colour_values,
                "colorscale": list(RISK_COLORSCALE),
                "cmin": 0.0, "cmax": 1.0,
                # A 2px surface ring keeps overlapping nodes separable.
                "line": {"width": 2, "color": "rgba(255,255,255,0.65)"},
                "colorbar": {
                    "title": {"text": "risk", "font": {"size": 11, "color": TEXT_MUTED}},
                    "tickvals": [0.0, 0.5, 1.0],
                    "ticktext": ["licit", f"thr {threshold:.2f}", "illicit"],
                    "tickfont": {"size": 10, "color": TEXT_MUTED},
                    "thickness": 10, "len": 0.55, "outlinewidth": 0,
                },
            },
            text=hover, hoverinfo="text", showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[coordinates[center_local, 0]], y=[coordinates[center_local, 1]],
            mode="markers",
            marker={
                "size": _CENTER_MARKER + 10, "color": "rgba(0,0,0,0)",
                "line": {"width": 3, "color": FOCUS_RING},
            },
            name="under investigation", hoverinfo="skip", showlegend=True,
        )
    )
    # Legend-only proxies: they name the colour encoding without adding marks.
    for name, colour in (("licit (low risk)", LICIT_BLUE), ("illicit (high risk)", ILLICIT_RED)):
        figure.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker={"size": 10, "color": colour}, name=name, showlegend=True,
            )
        )


def precision_recall_figure(
    recall: Sequence[float],
    precision: Sequence[float],
    *,
    baseline: float,
    label: str = "model",
) -> go.Figure:
    """Plot a precision/recall curve against the always-flag baseline.

    The baseline is the illicit prevalence -- what a model that flags everything
    would achieve -- which is the only fair reference on data this imbalanced.
    """
    if not recall:
        return empty_figure("Not enough labelled data to draw a curve.")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(recall), y=list(precision), mode="lines",
            line={"width": 2, "color": LICIT_BLUE}, name=label,
            hovertemplate="recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=baseline, line={"width": 1.5, "color": NEUTRAL_GREY, "dash": "dot"},
        annotation_text=f"prevalence {baseline:.3f}",
        annotation_position="top right",
        annotation_font={"size": 11, "color": TEXT_MUTED},
    )
    figure.update_layout(**base_layout("Precision / recall on the illicit class", height=340))
    figure.update_xaxes(**axis_style("recall", range=[0, 1]))
    figure.update_yaxes(**axis_style("precision", range=[0, 1]))
    return figure


def temporal_f1_figure(per_time_step: Mapping[int, float]) -> go.Figure:
    """Plot illicit F1 per time step, marking the known distribution shift.

    Around step 43 the dataset's dominant illicit marketplace was shut down.
    Every published model degrades sharply afterwards, and a chart that hides
    that is misleading -- so it is annotated rather than smoothed away.
    """
    if not per_time_step:
        return empty_figure("No per-time-step metrics available.")
    steps = sorted(per_time_step)
    values = [per_time_step[step] for step in steps]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=steps, y=values, mode="lines+markers",
            line={"width": 2, "color": LICIT_BLUE},
            marker={"size": 8, "color": LICIT_BLUE,
                    "line": {"width": 2, "color": "rgba(255,255,255,0.65)"}},
            hovertemplate="time step %{x}<br>illicit F1 %{y:.3f}<extra></extra>",
        )
    )
    if steps and min(steps) <= 43 <= max(steps):
        figure.add_vline(
            x=43, line={"width": 1.5, "color": NEUTRAL_GREY, "dash": "dot"},
            annotation_text="dark-market shutdown",
            annotation_position="top left",
            annotation_font={"size": 11, "color": TEXT_MUTED},
        )
    figure.update_layout(**base_layout("Detection quality over time", height=320))
    figure.update_xaxes(**axis_style("time step"))
    figure.update_yaxes(**axis_style("illicit F1", range=[0, 1]))
    return figure


def alert_volume_figure(
    steps: Sequence[int], alerts: Sequence[int], actual: Sequence[int]
) -> go.Figure:
    """Compare alerts raised against known illicit transactions per time step."""
    if not steps:
        return empty_figure("No temporal data available.")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=list(steps), y=list(alerts), name="alerts raised",
            marker={"color": ILLICIT_RED, "line": {"width": 0}}, opacity=0.85,
            hovertemplate="time step %{x}<br>%{y} alerts<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=list(steps), y=list(actual), mode="lines+markers", name="known illicit",
            line={"width": 2, "color": LICIT_BLUE}, marker={"size": 7},
            hovertemplate="time step %{x}<br>%{y} labelled illicit<extra></extra>",
        )
    )
    layout = base_layout("Alert volume against known illicit activity", height=320)
    layout["showlegend"] = True
    layout["legend"] = {"orientation": "h", "y": 1.12, "x": 0.0,
                        "font": {"size": 11, "color": TEXT_MUTED},
                        "bgcolor": "rgba(0,0,0,0)"}
    figure.update_layout(**layout)
    figure.update_xaxes(**axis_style("time step"))
    figure.update_yaxes(**axis_style("transactions"))
    return figure


def confusion_figure(metrics: ClassificationMetrics) -> go.Figure:
    """Render the 2x2 confusion matrix, shaded by within-class rate.

    Shading uses each cell's share of its *true* class, not its raw count.
    On this dataset the true-negative cell is roughly thirty times every other
    cell combined, so shading by count would drive all three remaining cells to
    the palest step of the ramp and make the errors -- the only cells anyone
    reads a confusion matrix for -- invisible.  Rates are bounded in [0, 1],
    so every cell keeps a legible shade while the counts stay on the labels.
    """
    matrix = metrics.confusion
    rows = ["actually licit", "actually illicit"]
    columns = ["predicted licit", "predicted illicit"]
    rates = [
        [value / total if (total := sum(row)) else 0.0 for value in row] for row in matrix
    ]

    figure = go.Figure(
        go.Heatmap(
            z=rates, x=columns, y=rows,
            colorscale=list(SEQUENTIAL_BLUE), zmin=0.0, zmax=1.0, showscale=False,
            customdata=matrix,
            hovertemplate=(
                "%{y} / %{x}<br>%{customdata:,} transactions"
                "<br>%{z:.1%} of this class<extra></extra>"
            ),
        )
    )
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            rate = rates[row_index][column_index]
            figure.add_annotation(
                x=column_index, y=row_index, showarrow=False,
                text=f"<b>{value:,}</b><br><span style='font-size:11px'>{rate:.1%}</span>",
                font={"size": 15, "color": "#ffffff" if rate > 0.40 else TEXT_MUTED},
            )
    figure.update_layout(
        **base_layout(f"Confusion matrix · {metrics.split} @ {metrics.threshold:.2f}", height=300)
    )
    figure.update_xaxes(**axis_style(showgrid=False, side="top"))
    # Reverse the row order so the matrix reads top-left to bottom-right, the
    # convention every published confusion matrix uses.
    figure.update_yaxes(**axis_style(showgrid=False, autorange="reversed"))
    return figure


def risk_histogram_figure(
    edges: Sequence[float], counts: Sequence[int], *, threshold: float
) -> go.Figure:
    """Plot the distribution of illicit probabilities across the whole graph."""
    if not edges:
        return empty_figure("No scores available.")
    figure = go.Figure(
        go.Bar(
            x=list(edges), y=list(counts),
            marker={"color": LICIT_BLUE, "line": {"width": 0}},
            hovertemplate="P(illicit) ≈ %{x:.2f}<br>%{y:,} transactions<extra></extra>",
        )
    )
    figure.add_vline(
        x=threshold, line={"width": 2, "color": ILLICIT_RED},
        annotation_text=f"threshold {threshold:.2f}",
        annotation_position="top right",
        annotation_font={"size": 11, "color": TEXT_MUTED},
    )
    figure.update_layout(**base_layout("Risk score distribution", height=300))
    figure.update_xaxes(**axis_style("P(illicit)"))
    figure.update_yaxes(**axis_style("transactions (log scale)"), type="log")
    return figure


def feature_attribution_figure(attribution: Mapping[int, float]) -> go.Figure:
    """Plot the input features the illicit logit is locally most sensitive to."""
    if not attribution:
        return empty_figure("Attribution unavailable for this transaction.", height=300)
    ordered = sorted(attribution.items(), key=lambda item: item[1])[-10:]
    labels = [f"feature {index}" for index, _ in ordered]
    values = [value for _, value in ordered]
    figure = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker={"color": LICIT_BLUE, "line": {"width": 0}},
            hovertemplate="%{y}<br>saliency %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(**base_layout("Most influential input features", height=320))
    figure.update_xaxes(**axis_style("|gradient × input|"))
    figure.update_yaxes(**axis_style(showgrid=False))
    return figure


def class_balance_figure(splits: Sequence[object]) -> go.Figure:
    """Show labelled class counts per temporal split as a stacked bar."""
    names = [getattr(split, "name", "?") for split in splits]
    licit = [getattr(split, "num_licit", 0) for split in splits]
    illicit = [getattr(split, "num_illicit", 0) for split in splits]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(x=names, y=licit, name="licit",
               marker={"color": LICIT_BLUE, "line": {"width": 2, "color": "rgba(0,0,0,0)"}},
               hovertemplate="%{x}<br>%{y:,} licit<extra></extra>")
    )
    figure.add_trace(
        go.Bar(x=names, y=illicit, name="illicit",
               marker={"color": ILLICIT_RED, "line": {"width": 2, "color": "rgba(0,0,0,0)"}},
               hovertemplate="%{x}<br>%{y:,} illicit<extra></extra>")
    )
    layout = base_layout("Labelled transactions per split", height=300)
    layout["showlegend"] = True
    layout["barmode"] = "stack"
    layout["legend"] = {"orientation": "h", "y": 1.12, "x": 0.0,
                        "font": {"size": 11, "color": TEXT_MUTED},
                        "bgcolor": "rgba(0,0,0,0)"}
    figure.update_layout(**layout)
    figure.update_xaxes(**axis_style(showgrid=False))
    figure.update_yaxes(**axis_style("transactions"))
    return figure


def embedding_figure(
    embeddings: Tensor,
    labels: Tensor,
    probabilities: Tensor,
    *,
    sample_size: int = 4000,
    seed: int = 11,
) -> go.Figure:
    """Project learned node embeddings to 2-D with PCA and colour by ground truth.

    The point of this view is to show whether the encoder has actually learned a
    separable representation, rather than whether the classifier head happens to
    threshold well.  Unlabelled nodes are drawn in recessive grey as context.
    """
    if embeddings.numel() == 0:
        return empty_figure("No embeddings available.")
    generator = torch.Generator().manual_seed(seed)
    total = embeddings.size(0)
    index = (
        torch.randperm(total, generator=generator)[:sample_size]
        if total > sample_size
        else torch.arange(total)
    )
    sample = embeddings[index].float()
    centred = sample - sample.mean(dim=0, keepdim=True)
    # Economy SVD: the top two right-singular vectors are the PCA axes.
    _, _, components = torch.linalg.svd(centred, full_matrices=False)
    projected = (centred @ components[:2].T).numpy()

    sampled_labels = labels[index].numpy()
    sampled_probabilities = probabilities[index].numpy()

    figure = go.Figure()
    series = (
        (UNKNOWN_LABEL, "unlabelled", UNKNOWN_GREY, 0.28, 4),
        (LICIT_LABEL, "licit", LICIT_BLUE, 0.75, 6),
        (ILLICIT_LABEL, "illicit", ILLICIT_RED, 0.9, 7),
    )
    for code, name, colour, opacity, size in series:
        selector = sampled_labels == code
        if not selector.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=projected[selector, 0], y=projected[selector, 1], mode="markers",
                name=name,
                marker={"size": size, "color": colour, "opacity": opacity,
                        "line": {"width": 0}},
                customdata=sampled_probabilities[selector],
                hovertemplate=f"{name}<br>P(illicit) %{{customdata:.3f}}<extra></extra>",
            )
        )
    layout = base_layout("Learned embedding space (PCA projection)", height=420)
    layout["showlegend"] = True
    layout["margin"]["t"] = 76
    layout["legend"] = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0,
                        "font": {"size": 11, "color": TEXT_MUTED},
                        "bgcolor": "rgba(0,0,0,0)"}
    figure.update_layout(**layout)
    # A handful of extreme nodes otherwise compress the entire population into a
    # few pixels, so the view is clipped to a robust range rather than the full
    # extent. Outliers stay in the data and remain reachable by panning.
    figure.update_xaxes(**axis_style("component 1", range=_robust_range(projected[:, 0])))
    figure.update_yaxes(**axis_style("component 2", range=_robust_range(projected[:, 1])))
    return figure


def _robust_range(values: np.ndarray, *, lower: float = 1.0, upper: float = 99.0,
                  pad: float = 0.06) -> list[float]:
    """Return a padded percentile range, falling back to the full extent."""
    low, high = np.percentile(values, [lower, upper])
    if not np.isfinite([low, high]).all() or high - low < 1e-9:
        low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return [low - 1.0, high + 1.0]
    margin = (high - low) * pad
    return [float(low - margin), float(high + margin)]


def training_history_figure(history: Sequence[object]) -> go.Figure:
    """Plot validation F1 across epochs for a completed run."""
    if not history:
        return empty_figure("No training history recorded.")
    epochs = [getattr(record, "epoch", i) for i, record in enumerate(history)]
    f1 = [getattr(record, "val_illicit_f1", 0.0) for record in history]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=epochs, y=f1, mode="lines", line={"width": 2, "color": LICIT_BLUE},
                   hovertemplate="epoch %{x}<br>val F1 %{y:.4f}<extra></extra>")
    )
    figure.update_layout(**base_layout("Validation illicit F1 per epoch", height=300))
    figure.update_xaxes(**axis_style("epoch"))
    figure.update_yaxes(**axis_style("illicit F1", range=[0, 1]))
    return figure
