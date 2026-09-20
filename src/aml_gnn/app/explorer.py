"""Flow explorer: a large neighbourhood laid out by how far it propagated.

The case view answers "why this transaction?" on a couple of hundred nodes.
This view answers "where does this money go?" on a few thousand, and the layout
is chosen to make that question legible: a force simulation on 7,000 nodes takes
twenty seconds and produces a hairball, whereas breadth-first distance from the
seed is both instant and the thing the viewer actually wants to read.

So distance from the origin *is* hop distance.  The seed sits at the centre,
its direct counterparties form the first shell, theirs the second, and each
branch keeps its heading as it grows, which turns a payment fan into a visible
limb rather than a knot.  Rotating the result shows depth directly.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from torch import Tensor

from ..data.splits import UNKNOWN_LABEL
from ..logging_utils import get_logger
from .theme import (
    EDGE_LINE,
    FOCUS_RING,
    ILLICIT_TONE,
    LICIT_TONE,
    RISK_COLORSCALE,
    TEXT_MUTED,
    base_layout,
    normalise_risk,
    scene_layout,
)
from .visualisations import empty_figure

logger = get_logger(__name__)

#: Radial distance between consecutive hop shells.
_SHELL_SPACING: float = 1.0

#: Half-angle of the cone a node's children are spread into. Narrow enough that
#: sibling branches stay distinct, wide enough that a big fan does not overlap.
_CONE: float = 0.62


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors spanning the plane perpendicular to ``direction``."""
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first) or 1.0
    second = np.cross(direction, first)
    second /= np.linalg.norm(second) or 1.0
    return first, second


def _sphere_directions(count: int) -> np.ndarray:
    """Return ``count`` roughly equidistant unit vectors, via a golden spiral."""
    if count == 1:
        return np.array([[0.0, 0.0, 1.0]])
    indices = np.arange(count)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = indices * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)


def propagation_coordinates(
    depths: np.ndarray,
    edges: np.ndarray,
    *,
    spacing: float = _SHELL_SPACING,
    seed: int = 11,
) -> np.ndarray:
    """Lay nodes out on concentric shells, one shell per hop from the seed.

    Each node inherits its parent's heading and is offset into a cone around it,
    so a branch grows outward as a recognisable limb.  Cost is linear in the
    number of nodes, which is what makes several thousand of them viable.

    Args:
        depths: Breadth-first depth of each node.
        edges: ``(2, num_edges)`` local edge list, used to find each node's
            parent one shell in.
        spacing: Radial distance between shells.
        seed: Seed for the angular jitter that stops siblings colliding.

    Returns:
        An ``(num_nodes, 3)`` coordinate array.

    """
    count = len(depths)
    coordinates = np.zeros((count, 3), dtype=float)
    if count == 0:
        return coordinates

    adjacency: list[list[int]] = [[] for _ in range(count)]
    for source, target in edges.T:
        adjacency[int(source)].append(int(target))
        adjacency[int(target)].append(int(source))

    rng = np.random.default_rng(seed)
    directions = np.zeros((count, 3), dtype=float)

    roots = np.flatnonzero(depths == 0)
    root_directions = _sphere_directions(max(len(roots), 1))
    for position, node in enumerate(roots):
        directions[node] = root_directions[position]
        # A single seed sits at the origin; several accumulated ones spread onto
        # a small sphere so they stay individually clickable.
        coordinates[node] = root_directions[position] * (0.0 if len(roots) == 1 else 0.45)

    for shell in range(1, int(depths.max()) + 1 if count else 1):
        members = np.flatnonzero(depths == shell)
        if not len(members):
            continue
        children: dict[int, list[int]] = {}
        orphans: list[int] = []
        for node in members:
            parent = next(
                (n for n in adjacency[int(node)] if depths[n] == shell - 1), None
            )
            if parent is None:
                orphans.append(int(node))
            else:
                children.setdefault(int(parent), []).append(int(node))

        for parent, group in children.items():
            heading = directions[parent]
            if not np.any(heading):
                heading = np.array([0.0, 0.0, 1.0])
            first, second = _orthonormal_basis(heading)
            # A wide fan needs a wider cone or its members land on top of one
            # another; a narrow one stays tight to its parent's heading.
            cone = min(_CONE * (1.0 + np.log1p(len(group)) / 3.0), 1.45)
            for position, child in enumerate(group):
                angle = 2.0 * np.pi * position / len(group) + rng.uniform(-0.08, 0.08)
                tilt = cone * (0.55 + 0.45 * rng.random())
                offset = np.cos(angle) * first + np.sin(angle) * second
                heading_out = heading * np.cos(tilt) + offset * np.sin(tilt)
                heading_out /= np.linalg.norm(heading_out) or 1.0
                directions[child] = heading_out
                coordinates[child] = heading_out * (shell * spacing)

        for stray in orphans:
            drift = rng.normal(size=3)
            drift /= np.linalg.norm(drift) or 1.0
            directions[stray] = drift
            coordinates[stray] = drift * (shell * spacing)

    return coordinates


def flow_figure(
    node_indices: Tensor,
    depths: Tensor,
    edges: Tensor,
    probabilities: Tensor,
    labels: Tensor,
    tx_ids: Tensor,
    degrees: Tensor,
    *,
    threshold: float,
    seed_index: int,
    edge_cap: int = 24_000,
) -> go.Figure:
    """Draw an explored neighbourhood as concentric hop shells.

    Args:
        node_indices: Global node index of each drawn node.
        depths: Hop distance from the seed, per node.
        edges: ``(2, num_edges)`` local edge list.
        probabilities: Illicit probability per drawn node.
        labels: Ground-truth class per drawn node.
        tx_ids: Transaction id per drawn node.
        degrees: Global degree per drawn node.
        threshold: Operating threshold, anchoring the colour ramp.
        seed_index: Global index of the node the walk started from.
        edge_cap: Ceiling on drawn edges.

    """
    count = len(node_indices)
    if count == 0:
        return empty_figure("Nothing explored yet.", height=560)

    depth_values = depths.numpy()
    edge_array = edges.numpy()
    coordinates = propagation_coordinates(depth_values, edge_array)

    figure = go.Figure()
    drawn = _add_flow_edges(figure, edge_array, coordinates, edge_cap)
    _add_flow_nodes(
        figure, node_indices, coordinates, depth_values, probabilities,
        labels, tx_ids, degrees, threshold, seed_index,
    )

    reach = int(depth_values.max())
    flagged = int((probabilities.numpy() > threshold).sum())
    title = (
        f"Flow from tx {int(tx_ids[0])} · {count:,} transactions · {drawn:,} edges · "
        f"{reach} hops out · {flagged:,} above threshold"
    )
    if drawn < edge_array.shape[1]:
        title += " · edges capped"

    layout = base_layout(title, height=700)
    layout["showlegend"] = True
    layout["margin"] = {"l": 0, "r": 0, "t": 44, "b": 0}
    layout["legend"] = {
        "orientation": "h", "yanchor": "bottom", "y": -0.02, "xanchor": "left", "x": 0.0,
        "font": {"size": 11, "color": TEXT_MUTED}, "bgcolor": "rgba(0,0,0,0)",
    }
    layout["uirevision"] = "flow-explorer"
    layout["scene"] = scene_layout(eye=(1.6, 1.6, 0.9))
    figure.update_layout(**layout)
    return figure


def _add_flow_edges(
    figure: go.Figure,
    edges: np.ndarray,
    coordinates: np.ndarray,
    edge_cap: int,
) -> int:
    """Add every edge as one trace, returning how many were drawn."""
    total = int(edges.shape[1])
    if not total:
        return 0
    limit = int(min(total, edge_cap))
    head, tail = edges[0, :limit], edges[1, :limit]
    # One interleaved array beats appending per edge: at this size the Python
    # loop is the cost, not the draw.
    xs = np.empty(limit * 3)
    ys = np.empty(limit * 3)
    zs = np.empty(limit * 3)
    for axis, target in enumerate((xs, ys, zs)):
        target[0::3] = coordinates[head, axis]
        target[1::3] = coordinates[tail, axis]
        target[2::3] = np.nan
    figure.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line={"width": 1.2, "color": EDGE_LINE},
            opacity=0.3, hoverinfo="skip", showlegend=False,
        )
    )
    return limit


def _add_flow_nodes(
    figure: go.Figure,
    node_indices: Tensor,
    coordinates: np.ndarray,
    depths: np.ndarray,
    probabilities: Tensor,
    labels: Tensor,
    tx_ids: Tensor,
    degrees: Tensor,
    threshold: float,
    seed_index: int,
) -> None:
    """Add the node scatter, with the seed ringed so it stays findable."""
    probability_values = probabilities.numpy()
    degree_values = degrees.numpy()
    tx_values = tx_ids.numpy()
    label_values = labels.numpy()
    indices = node_indices.numpy()

    colours = [normalise_risk(float(p), threshold) for p in probability_values]
    scaled = np.log1p(degree_values)
    span = float(scaled.max() - scaled.min())
    sizes = (
        3.0 + (scaled - scaled.min()) / span * 9.0
        if span > 1e-9
        else np.full_like(scaled, 5.0)
    )

    names = {0: "licit", 1: "illicit", UNKNOWN_LABEL: "unknown"}
    hover = [
        f"<b>tx {int(tx)}</b><br>P(illicit) {float(p):.4f}"
        f"<br>ground truth: {names.get(int(y), 'unknown')}"
        f"<br>{int(d)} hop{'' if d == 1 else 's'} from seed · degree {int(g)}"
        "<br><i>click to expand from here</i>"
        for tx, p, y, d, g in zip(
            tx_values, probability_values, label_values, depths, degree_values, strict=True
        )
    ]

    figure.add_trace(
        go.Scatter3d(
            x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
            mode="markers",
            marker={
                "size": sizes.tolist(), "color": colours,
                "colorscale": list(RISK_COLORSCALE),
                "cmin": 0.0, "cmax": 1.0, "opacity": 0.88,
                "line": {"width": 0},
                "colorbar": {
                    "title": {"text": "risk", "font": {"size": 11, "color": TEXT_MUTED}},
                    "tickvals": [0.0, 0.5, 1.0],
                    "ticktext": ["licit", f"thr {threshold:.2f}", "illicit"],
                    "tickfont": {"size": 10, "color": TEXT_MUTED},
                    "thickness": 10, "len": 0.55, "outlinewidth": 0,
                },
            },
            customdata=indices,
            text=hover, hoverinfo="text", showlegend=False,
        )
    )

    position = np.flatnonzero(indices == seed_index)
    if len(position):
        spot = int(position[0])
        figure.add_trace(
            go.Scatter3d(
                x=[coordinates[spot, 0]], y=[coordinates[spot, 1]], z=[coordinates[spot, 2]],
                mode="markers",
                marker={"size": float(sizes[spot]) + 12, "color": "rgba(0,0,0,0)",
                        "line": {"width": 4, "color": FOCUS_RING}},
                name="seed transaction", hoverinfo="skip", showlegend=True,
            )
        )
    for name, colour in (("licit", LICIT_TONE), ("illicit", ILLICIT_TONE)):
        figure.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker={"size": 6, "color": colour}, name=name, showlegend=True,
            )
        )
