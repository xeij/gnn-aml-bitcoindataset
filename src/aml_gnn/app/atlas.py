"""Dataset-scale 3-D views.

The case view in :mod:`.visualisations` answers "why this transaction?".  These
answer "what does the dataset look like?", which is a different rendering
problem: 203,769 nodes and 468,710 edges is far past what a browser will draw
interactively, so every view here is explicit about the budget it works within
and about what was left out.

Three views, all rotatable:

* **Embedding atlas** -- every node in the learned representation, which shows
  whether the encoder separates the classes at all.
* **Temporal drift** -- the same projection with time on the vertical axis, so
  the collapse of the illicit cluster after the marketplace shutdown is visible
  as structure rather than as a line on a chart.
* **Layered structure** -- the real topology, one horizontal layer per time
  step.  Every edge in Elliptic joins two nodes in the same step, so the graph
  is 49 disconnected islands and this is the only view that shows that.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from torch import Tensor

from ..data.splits import ILLICIT_LABEL, LICIT_LABEL, UNKNOWN_LABEL
from ..logging_utils import get_logger
from .theme import (
    CLASS_SYMBOLS,
    EDGE_LINE,
    ILLICIT_TONE,
    LICIT_TONE,
    RISK_COLORSCALE,
    TEXT_MUTED,
    UNKNOWN_TONE,
    base_layout,
    normalise_risk,
    scene_layout,
)
from .visualisations import empty_figure

logger = get_logger(__name__)

#: Point budget beyond which the browser starts to feel the draw. Not a hard
#: cap -- the caller may ask for the whole graph -- but the default sits here.
DEFAULT_POINT_BUDGET: int = 25_000

#: Per-layer node cap for the structural view, and the ceiling on drawn edges.
_LAYER_NODE_CAP: int = 260
_STRUCTURE_EDGE_CAP: int = 16_000

#: Series drawn back-to-front, so labelled nodes land on top of the grey mass.
#: With hue gone, each class carries a distinct marker shape as well as its own
#: step on the lightness ramp, so identity survives in greyscale and in print.
_SERIES: tuple[tuple[int, str, str, float, int, str], ...] = (
    (UNKNOWN_LABEL, "unlabelled", UNKNOWN_TONE, 0.16, 2, CLASS_SYMBOLS["unlabelled"]),
    (LICIT_LABEL, "licit", LICIT_TONE, 0.55, 3, CLASS_SYMBOLS["licit"]),
    (ILLICIT_LABEL, "illicit", ILLICIT_TONE, 0.95, 4, CLASS_SYMBOLS["illicit"]),
)


def sample_indices(total: int, budget: int, *, seed: int = 11) -> Tensor:
    """Return a deterministic uniform sample of node indices.

    Uniform rather than stratified on purpose: over-representing the 2.2% of
    nodes that carry an illicit label would make the picture prettier and its
    apparent class balance a lie.  What the viewer sees is what the split is.

    Args:
        total: Population size.
        budget: Maximum number of indices to return.
        seed: Generator seed, so a redraw returns the same sample.

    """
    if total <= budget:
        return torch.arange(total)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total, generator=generator)[:budget]


def _legend_block() -> dict:
    """Return the shared horizontal legend used by every atlas view."""
    return {
        "orientation": "h",
        "yanchor": "bottom",
        "y": -0.02,
        "xanchor": "left",
        "x": 0.0,
        "font": {"size": 11, "color": TEXT_MUTED},
        "bgcolor": "rgba(0,0,0,0)",
    }


def _hover(
    tx_ids: np.ndarray,
    probabilities: np.ndarray,
    steps: np.ndarray,
    degrees: np.ndarray,
    selector: np.ndarray,
) -> list[str]:
    """Build hover text for one class series."""
    return [
        f"<b>tx {int(tx)}</b><br>P(illicit) {float(p):.4f}"
        f"<br>time step {int(t)} · degree {int(d)}"
        for tx, p, t, d in zip(
            tx_ids[selector], probabilities[selector], steps[selector], degrees[selector],
            strict=True,
        )
    ]


def embedding_atlas_figure(
    projection: Tensor,
    labels: Tensor,
    probabilities: Tensor,
    tx_ids: Tensor,
    time_steps: Tensor,
    degrees: Tensor,
    *,
    budget: int = DEFAULT_POINT_BUDGET,
    colour_by: str = "ground truth",
    threshold: float = 0.5,
    seed: int = 11,
) -> go.Figure:
    """Draw every sampled node in the learned embedding space, in 3-D.

    Colouring by ground truth answers "did the encoder learn a separable
    representation?"; colouring by risk answers "where does the classifier
    actually fire?".  They are different questions and the second one includes
    the 77% of nodes that have no label to check against.

    Args:
        projection: ``(num_nodes, 3)`` PCA coordinates from the service.
        labels: Ground-truth class per node, including the unknown code.
        probabilities: Illicit probability per node.
        tx_ids: Elliptic transaction ids, for hover.
        time_steps: Time step per node, for hover.
        degrees: Undirected degree per node, for hover.
        budget: Maximum points to draw.
        colour_by: ``"ground truth"`` or ``"risk"``.
        threshold: Operating threshold, anchoring the risk ramp.
        seed: Sampling seed.

    """
    if projection.numel() == 0:
        return empty_figure("No embeddings available.")

    total = projection.size(0)
    index = sample_indices(total, budget, seed=seed)
    points = projection[index][:, :3].numpy()
    sampled_labels = labels[index].numpy()
    sampled_probabilities = probabilities[index].numpy()
    sampled_tx = tx_ids[index].numpy()
    sampled_steps = time_steps[index].numpy()
    sampled_degrees = degrees[index].numpy()

    figure = go.Figure()
    node_indices = index.numpy()
    if colour_by == "risk":
        _add_risk_trace(
            figure, points, sampled_probabilities, sampled_tx, sampled_steps,
            sampled_degrees, node_indices, threshold,
        )
    else:
        _add_class_traces(
            figure, points, sampled_labels, sampled_probabilities, sampled_tx,
            sampled_steps, sampled_degrees, node_indices,
        )

    drawn = len(index)
    title = (
        f"Embedding atlas · {drawn:,} of {total:,} nodes"
        f"{'' if drawn == total else ' (uniform sample)'} · coloured by {colour_by}"
    )
    layout = base_layout(title, height=680)
    layout["showlegend"] = colour_by != "risk"
    layout["margin"] = {"l": 0, "r": 0, "t": 44, "b": 0}
    layout["legend"] = _legend_block()
    layout["scene"] = scene_layout("component 1", "component 2", "component 3")
    layout["uirevision"] = "atlas-embedding"
    figure.update_layout(**layout)
    return figure


def _add_class_traces(
    figure: go.Figure,
    points: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    tx_ids: np.ndarray,
    steps: np.ndarray,
    degrees: np.ndarray,
    node_indices: np.ndarray,
) -> None:
    """Add one trace per ground-truth class, unlabelled drawn first."""
    for code, name, colour, opacity, size, symbol in _SERIES:
        selector = labels == code
        if not selector.any():
            continue
        figure.add_trace(
            go.Scatter3d(
                x=points[selector, 0], y=points[selector, 1], z=points[selector, 2],
                mode="markers", name=f"{name} ({int(selector.sum()):,})",
                marker={"size": size, "color": colour, "opacity": opacity,
                        "symbol": symbol, "line": {"width": 0}},
                customdata=node_indices[selector],
                text=_hover(tx_ids, probabilities, steps, degrees, selector),
                hoverinfo="text",
            )
        )


def _add_risk_trace(
    figure: go.Figure,
    points: np.ndarray,
    probabilities: np.ndarray,
    tx_ids: np.ndarray,
    steps: np.ndarray,
    degrees: np.ndarray,
    node_indices: np.ndarray,
    threshold: float,
) -> None:
    """Add a single trace coloured by the threshold-anchored risk ramp."""
    colour_values = [normalise_risk(float(p), threshold) for p in probabilities]
    everything = np.ones(len(probabilities), dtype=bool)
    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker={
                "size": 2.6, "color": colour_values,
                "colorscale": list(RISK_COLORSCALE),
                "cmin": 0.0, "cmax": 1.0, "opacity": 0.7,
                "line": {"width": 0},
                "colorbar": {
                    "title": {"text": "risk", "font": {"size": 11, "color": TEXT_MUTED}},
                    "tickvals": [0.0, 0.5, 1.0],
                    "ticktext": ["licit", f"thr {threshold:.2f}", "illicit"],
                    "tickfont": {"size": 10, "color": TEXT_MUTED},
                    "thickness": 10, "len": 0.55, "outlinewidth": 0,
                },
            },
            customdata=node_indices,
            text=_hover(tx_ids, probabilities, steps, degrees, everything),
            hoverinfo="text", showlegend=False,
        )
    )


def temporal_drift_figure(
    projection: Tensor,
    labels: Tensor,
    probabilities: Tensor,
    tx_ids: Tensor,
    time_steps: Tensor,
    degrees: Tensor,
    *,
    budget: int = DEFAULT_POINT_BUDGET,
    shutdown_step: int = 43,
    seed: int = 11,
) -> go.Figure:
    """Draw the embedding against time, so distribution shift becomes geometry.

    The first two principal components run along x and y; the time step runs up
    z.  Read bottom to top, the illicit band holds together through the training
    and validation windows and then stops: the marketplace that generated most
    of it was seized around step 43.  A per-step F1 line says the same thing,
    but this shows the population it happened to.

    Args:
        projection: ``(num_nodes, >=2)`` PCA coordinates.
        labels: Ground-truth class per node.
        probabilities: Illicit probability per node, for hover.
        tx_ids: Elliptic transaction ids, for hover.
        time_steps: Time step per node -- the vertical axis.
        degrees: Undirected degree per node, for hover.
        budget: Maximum points to draw.
        shutdown_step: Step to mark with a reference plane.
        seed: Sampling seed.

    """
    if projection.numel() == 0:
        return empty_figure("No embeddings available.")

    total = projection.size(0)
    index = sample_indices(total, budget, seed=seed)
    flat = projection[index][:, :2].numpy()
    sampled_labels = labels[index].numpy()
    sampled_probabilities = probabilities[index].numpy()
    sampled_tx = tx_ids[index].numpy()
    sampled_steps = time_steps[index].numpy()
    sampled_degrees = degrees[index].numpy()

    figure = go.Figure()
    for code, name, colour, opacity, size, symbol in _SERIES:
        selector = sampled_labels == code
        if not selector.any():
            continue
        figure.add_trace(
            go.Scatter3d(
                x=flat[selector, 0], y=flat[selector, 1], z=sampled_steps[selector],
                mode="markers", name=f"{name} ({int(selector.sum()):,})",
                marker={"size": size, "color": colour, "opacity": opacity,
                        "symbol": symbol, "line": {"width": 0}},
                customdata=index.numpy()[selector],
                text=_hover(
                    sampled_tx, sampled_probabilities, sampled_steps,
                    sampled_degrees, selector,
                ),
                hoverinfo="text",
            )
        )

    _add_shutdown_plane(figure, flat, shutdown_step)

    drawn = len(index)
    layout = base_layout(
        f"Temporal drift · {drawn:,} of {total:,} nodes · "
        f"illicit signal collapses after step {shutdown_step}",
        height=700,
    )
    layout["showlegend"] = True
    layout["margin"] = {"l": 0, "r": 0, "t": 44, "b": 0}
    layout["legend"] = _legend_block()
    scene = scene_layout("component 1", "component 2", "time step")
    # Time is the point of this view, so the box is stretched along it rather
    # than left cubic, which would compress 49 steps into a flat slab.
    scene["aspectmode"] = "manual"
    scene["aspectratio"] = {"x": 1.0, "y": 1.0, "z": 1.5}
    scene["camera"] = {"eye": {"x": 1.7, "y": 1.7, "z": 0.55}}
    layout["scene"] = scene
    layout["uirevision"] = "atlas-drift"
    figure.update_layout(**layout)
    return figure


def _add_shutdown_plane(figure: go.Figure, flat: np.ndarray, step: int) -> None:
    """Draw a translucent reference plane at the shutdown time step."""
    if flat.size == 0:
        return
    x_low, x_high = float(flat[:, 0].min()), float(flat[:, 0].max())
    y_low, y_high = float(flat[:, 1].min()), float(flat[:, 1].max())
    figure.add_trace(
        go.Surface(
            x=[[x_low, x_high], [x_low, x_high]],
            y=[[y_low, y_low], [y_high, y_high]],
            z=[[step, step], [step, step]],
            showscale=False, opacity=0.13,
            colorscale=[[0.0, TEXT_MUTED], [1.0, TEXT_MUTED]],
            hoverinfo="skip", name=f"step {step}", showlegend=False,
        )
    )


def layered_structure_figure(
    edge_index: Tensor,
    time_steps: Tensor,
    labels: Tensor,
    probabilities: Tensor,
    tx_ids: Tensor,
    degrees: Tensor,
    *,
    per_layer: int = _LAYER_NODE_CAP,
    edge_cap: int = _STRUCTURE_EDGE_CAP,
    seed: int = 11,
) -> go.Figure:
    """Draw the real topology as one horizontal layer per time step.

    Every edge in Elliptic joins two nodes in the same time step, so the graph
    is not one network but 49 disconnected daily snapshots.  Stacking them makes
    that the first thing the viewer sees, and the hub-and-spoke fans that
    dominate individual layers become visible without opening a case.

    Each layer is sampled to its highest-degree nodes plus a random remainder,
    so the hubs survive the budget and the fans stay attached to them.

    Args:
        edge_index: ``(2, num_edges)`` undirected edge list.
        time_steps: Time step per node.
        labels: Ground-truth class per node, for hover.
        probabilities: Illicit probability per node -- drives node colour.
        tx_ids: Elliptic transaction ids, for hover.
        degrees: Undirected degree per node.
        per_layer: Maximum nodes drawn per time step.
        edge_cap: Ceiling on total drawn edges.
        seed: Sampling seed.

    """
    if time_steps.numel() == 0:
        return empty_figure("No graph to display.")

    steps = time_steps.numpy()
    node_degrees = degrees.numpy()
    generator = torch.Generator().manual_seed(seed)

    kept: list[np.ndarray] = []
    for step in np.unique(steps):
        members = np.flatnonzero(steps == step)
        if len(members) <= per_layer:
            kept.append(members)
            continue
        # Hubs first: they are the structure. The rest is a random fill so the
        # layer still reads as a population rather than as a top-N list.
        order = members[np.argsort(-node_degrees[members])]
        hub_count = per_layer // 2
        hubs = order[:hub_count]
        rest = order[hub_count:]
        pick = torch.randperm(len(rest), generator=generator)[: per_layer - hub_count].numpy()
        kept.append(np.concatenate([hubs, rest[pick]]))

    selected = np.concatenate(kept)
    local_of = {int(node): position for position, node in enumerate(selected)}

    coordinates = _layered_coordinates(selected, steps, edge_index, local_of, seed)
    figure = go.Figure()
    drawn_edges, crossing = _add_layer_edges(
        figure, edge_index, local_of, coordinates, edge_cap
    )
    _add_layer_nodes(figure, selected, coordinates, probabilities, labels, tx_ids, degrees)

    # Elliptic's edges are entirely intra-step, which is the point of the view,
    # but that is a property of the data rather than something to assert: any
    # graph with edges spanning steps gets told on instead.
    crossing_note = (
        "no edge crosses a layer"
        if not crossing
        else f"{crossing:,} cross-layer edges omitted"
    )
    layout = base_layout(
        f"Layered structure · {len(selected):,} nodes across {len(kept)} time steps · "
        f"{drawn_edges:,} edges drawn · {crossing_note}",
        height=700,
    )
    layout["showlegend"] = False
    layout["margin"] = {"l": 0, "r": 0, "t": 44, "b": 0}
    scene = scene_layout("", "", "time step")
    scene["aspectmode"] = "manual"
    scene["aspectratio"] = {"x": 1.0, "y": 1.0, "z": 1.8}
    scene["camera"] = {"eye": {"x": 1.8, "y": 1.8, "z": 0.6}}
    scene["xaxis"]["showticklabels"] = False
    scene["yaxis"]["showticklabels"] = False
    layout["scene"] = scene
    layout["uirevision"] = "atlas-layers"
    figure.update_layout(**layout)
    return figure


def _layered_coordinates(
    selected: np.ndarray,
    steps: np.ndarray,
    edge_index: Tensor,
    local_of: dict[int, int],
    seed: int,
) -> np.ndarray:
    """Lay each time step out on its own plane, stacked by step number.

    A force layout per layer would be the faithful choice but costs a second per
    layer at this size.  Instead each layer is laid out radially by degree --
    hubs at the centre, leaves on the rim -- which reproduces the fan shape the
    force layout would find, at a fraction of the cost and deterministically.
    """
    coordinates = np.zeros((len(selected), 3), dtype=float)
    edges = edge_index.numpy()
    local_degree: dict[int, int] = {}
    for source, target in zip(edges[0], edges[1], strict=True):
        if int(source) in local_of and int(target) in local_of:
            local_degree[int(source)] = local_degree.get(int(source), 0) + 1

    rng = np.random.default_rng(seed)
    for step in np.unique(steps[selected]):
        members = [n for n in selected if steps[n] == step]
        ranked = sorted(members, key=lambda n: -local_degree.get(int(n), 0))
        count = len(ranked)
        # Golden-angle spiral: even radial coverage without a force simulation.
        golden = np.pi * (3.0 - np.sqrt(5.0))
        for position, node in enumerate(ranked):
            radius = np.sqrt((position + 0.5) / count)
            angle = position * golden + rng.uniform(-0.05, 0.05)
            local = local_of[int(node)]
            coordinates[local] = (radius * np.cos(angle), radius * np.sin(angle), float(step))
    return coordinates


def _add_layer_edges(
    figure: go.Figure,
    edge_index: Tensor,
    local_of: dict[int, int],
    coordinates: np.ndarray,
    edge_cap: int,
) -> tuple[int, int]:
    """Add intra-layer edges up to the cap.

    Returns:
        The number of edges drawn, and the number skipped for spanning two
        layers.  A skipped edge would render as a diagonal through empty space
        between planes, which reads as depth rather than as connection.

    """
    edges = edge_index.numpy()
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    drawn = 0
    crossing = 0
    for source, target in zip(edges[0], edges[1], strict=True):
        if drawn >= edge_cap:
            break
        head, tail = int(source), int(target)
        if head >= tail or head not in local_of or tail not in local_of:
            continue
        a, b = local_of[head], local_of[tail]
        if coordinates[a, 2] != coordinates[b, 2]:
            crossing += 1
            continue
        xs += [coordinates[a, 0], coordinates[b, 0], None]
        ys += [coordinates[a, 1], coordinates[b, 1], None]
        zs += [coordinates[a, 2], coordinates[b, 2], None]
        drawn += 1
    if drawn:
        figure.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line={"width": 1, "color": EDGE_LINE},
                opacity=0.22, hoverinfo="skip", showlegend=False,
            )
        )
    return drawn, crossing


def _add_layer_nodes(
    figure: go.Figure,
    selected: np.ndarray,
    coordinates: np.ndarray,
    probabilities: Tensor,
    labels: Tensor,
    tx_ids: Tensor,
    degrees: Tensor,
) -> None:
    """Add the node scatter for the layered view, sized by global degree."""
    probability_values = probabilities.numpy()[selected]
    degree_values = degrees.numpy()[selected]
    tx_values = tx_ids.numpy()[selected]
    step_values = coordinates[:, 2]

    # Degree spans three orders of magnitude, so size on a log scale or the
    # hubs swallow the plot and everything else collapses to a dot.
    scaled = np.log1p(degree_values)
    span = float(scaled.max() - scaled.min())
    sizes = 2.0 + (scaled - scaled.min()) / span * 9.0 if span > 1e-9 else np.full_like(scaled, 4.0)

    hover = [
        f"<b>tx {int(tx)}</b><br>P(illicit) {float(p):.4f}"
        f"<br>time step {int(s)} · degree {int(d)}"
        for tx, p, s, d in zip(tx_values, probability_values, step_values, degree_values,
                               strict=True)
    ]
    figure.add_trace(
        go.Scatter3d(
            x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
            mode="markers",
            marker={
                "size": sizes.tolist(),
                "color": probability_values.tolist(),
                "colorscale": list(RISK_COLORSCALE),
                "cmin": 0.0, "cmax": 1.0, "opacity": 0.85,
                "line": {"width": 0},
                "colorbar": {
                    "title": {"text": "P(illicit)", "font": {"size": 11, "color": TEXT_MUTED}},
                    "tickfont": {"size": 10, "color": TEXT_MUTED},
                    "thickness": 10, "len": 0.55, "outlinewidth": 0,
                },
            },
            customdata=selected,
            text=hover, hoverinfo="text", showlegend=False,
        )
    )
