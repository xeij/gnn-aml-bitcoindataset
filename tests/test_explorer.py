"""Flow-explorer traversal and layout behave on a synthetic graph."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import torch
from torch_geometric.data import Data

from aml_gnn.app import explorer


def _chain_edges(length: int) -> np.ndarray:
    """Return a simple path graph, so hop depth is unambiguous."""
    return np.array([list(range(length - 1)), list(range(1, length))])


def test_depth_becomes_radius() -> None:
    """A node's distance from the origin is its hop count, exactly."""
    depths = np.array([0, 1, 1, 2, 3])
    edges = np.array([[0, 0, 1, 3], [1, 2, 3, 4]])
    coordinates = explorer.propagation_coordinates(depths, edges)
    radii = np.linalg.norm(coordinates, axis=1)
    assert radii[0] == 0.0
    for position, depth in enumerate(depths):
        assert radii[position] == np.float64(depth) or abs(radii[position] - depth) < 1e-9


def test_a_long_chain_marches_outward() -> None:
    """Every step along a path graph lands one shell further out."""
    length = 12
    depths = np.arange(length)
    coordinates = explorer.propagation_coordinates(depths, _chain_edges(length))
    radii = np.linalg.norm(coordinates, axis=1)
    assert all(radii[i] < radii[i + 1] for i in range(length - 1))


def test_layout_is_deterministic() -> None:
    """The same walk drawn twice produces identical coordinates."""
    depths = np.array([0, 1, 1, 1, 2, 2])
    edges = np.array([[0, 0, 0, 1, 2], [1, 2, 3, 4, 5]])
    first = explorer.propagation_coordinates(depths, edges)
    second = explorer.propagation_coordinates(depths, edges)
    assert np.array_equal(first, second)


def test_disconnected_nodes_still_get_a_position() -> None:
    """A node with no parent one shell in is placed rather than left at origin."""
    depths = np.array([0, 2])
    coordinates = explorer.propagation_coordinates(depths, np.empty((2, 0), dtype=int))
    assert np.linalg.norm(coordinates[1]) > 0.0


def test_flow_figure_is_three_dimensional_and_clickable(synthetic_graph: Data) -> None:
    """The figure emits Scatter3d traces carrying node indices for selection."""
    nodes = torch.arange(8)
    depths = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3])
    edges = torch.tensor([[0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7]])
    figure = explorer.flow_figure(
        nodes, depths, edges,
        torch.rand(8), synthetic_graph.y[:8], synthetic_graph.tx_id[:8],
        torch.arange(1, 9),
        threshold=0.5, seed_index=0,
    )
    assert isinstance(figure, go.Figure)
    assert all(isinstance(trace, go.Scatter3d) for trace in figure.data)
    tagged = [t for t in figure.data if t.customdata is not None]
    assert tagged and list(tagged[0].customdata) == nodes.tolist()
    assert "3 hops out" in figure.layout.title.text


def test_flow_figure_caps_its_edges(synthetic_graph: Data) -> None:
    """Exceeding the edge cap is disclosed in the title."""
    nodes = torch.arange(8)
    depths = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3])
    edges = torch.tensor([[0, 0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7]])
    figure = explorer.flow_figure(
        nodes, depths, edges,
        torch.rand(8), synthetic_graph.y[:8], synthetic_graph.tx_id[:8],
        torch.arange(1, 9),
        threshold=0.5, seed_index=0, edge_cap=3,
    )
    assert "edges capped" in figure.layout.title.text


def test_empty_walk_degrades_to_a_placeholder() -> None:
    """Nothing explored yet renders a message, not an exception."""
    figure = explorer.flow_figure(
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.empty(2, 0, dtype=torch.long), torch.empty(0), torch.empty(0),
        torch.empty(0), torch.empty(0), threshold=0.5, seed_index=0,
    )
    assert "Nothing explored" in figure.layout.annotations[0].text
