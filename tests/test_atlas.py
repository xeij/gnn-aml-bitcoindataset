"""Dataset-scale 3-D views build correctly on a synthetic graph."""

from __future__ import annotations

import plotly.graph_objects as go
import torch
from torch_geometric.data import Data

from aml_gnn.app import atlas


def _atlas_inputs(graph: Data) -> dict[str, torch.Tensor]:
    """Bundle the per-node tensors every atlas view consumes."""
    generator = torch.Generator().manual_seed(3)
    num_nodes = graph.num_nodes
    return {
        "labels": graph.y,
        "probabilities": torch.rand(num_nodes, generator=generator),
        "tx_ids": graph.tx_id,
        "time_steps": graph.time_step,
        "degrees": torch.bincount(graph.edge_index[0], minlength=num_nodes),
    }


def test_sample_is_deterministic_and_respects_the_budget() -> None:
    """The same seed returns the same sample, capped at the requested budget."""
    first = atlas.sample_indices(1_000, 120)
    second = atlas.sample_indices(1_000, 120)
    assert len(first) == 120
    assert torch.equal(first, second)


def test_sample_returns_everything_when_the_budget_exceeds_the_population() -> None:
    """A budget larger than the graph draws the whole graph, in order."""
    assert torch.equal(atlas.sample_indices(50, 500), torch.arange(50))


def test_embedding_atlas_separates_the_three_label_classes(synthetic_graph: Data) -> None:
    """Colouring by ground truth yields one trace per class, all in 3-D."""
    inputs = _atlas_inputs(synthetic_graph)
    projection = torch.randn(synthetic_graph.num_nodes, 3, generator=torch.Generator())
    figure = atlas.embedding_atlas_figure(projection, **inputs)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 3
    assert all(isinstance(trace, go.Scatter3d) for trace in figure.data)


def test_embedding_atlas_by_risk_is_one_ramped_trace(synthetic_graph: Data) -> None:
    """Risk colouring collapses to a single trace carrying the diverging ramp."""
    inputs = _atlas_inputs(synthetic_graph)
    projection = torch.randn(synthetic_graph.num_nodes, 3)
    figure = atlas.embedding_atlas_figure(
        projection, **inputs, colour_by="risk", threshold=0.7
    )
    assert len(figure.data) == 1
    assert figure.data[0].marker.colorscale is not None


def test_atlas_title_reports_the_sample_it_drew(synthetic_graph: Data) -> None:
    """A budget below the population is disclosed in the title, not hidden."""
    inputs = _atlas_inputs(synthetic_graph)
    projection = torch.randn(synthetic_graph.num_nodes, 3)
    figure = atlas.embedding_atlas_figure(projection, **inputs, budget=40)
    assert "40 of 200" in figure.layout.title.text
    assert "uniform sample" in figure.layout.title.text


def test_temporal_drift_puts_time_on_the_vertical_axis(synthetic_graph: Data) -> None:
    """Every point's z coordinate is its time step, and the plane is drawn."""
    inputs = _atlas_inputs(synthetic_graph)
    projection = torch.randn(synthetic_graph.num_nodes, 3)
    figure = atlas.temporal_drift_figure(projection, **inputs, shutdown_step=9)
    scatters = [t for t in figure.data if isinstance(t, go.Scatter3d)]
    surfaces = [t for t in figure.data if isinstance(t, go.Surface)]
    assert scatters and len(surfaces) == 1
    steps = set(synthetic_graph.time_step.tolist())
    assert all(set(int(v) for v in trace.z) <= steps for trace in scatters)
    assert figure.layout.scene.zaxis.title.text == "time step"


def test_layered_structure_never_draws_an_edge_across_layers(
    synthetic_graph: Data,
) -> None:
    """Layer height is the time step, so a drawn edge has equal z at both ends."""
    inputs = _atlas_inputs(synthetic_graph)
    figure = atlas.layered_structure_figure(
        synthetic_graph.edge_index,
        inputs["time_steps"], inputs["labels"], inputs["probabilities"],
        inputs["tx_ids"], inputs["degrees"],
    )
    lines = [t for t in figure.data if t.mode == "lines"]
    assert lines, "expected at least one edge trace"
    heights = [z for z in lines[0].z if z is not None]
    # Points arrive in (source, target, None) triples; both ends share a layer.
    assert all(heights[i] == heights[i + 1] for i in range(0, len(heights), 2))


def test_layered_structure_caps_the_edges_it_draws(synthetic_graph: Data) -> None:
    """The edge budget is enforced and reported in the title."""
    inputs = _atlas_inputs(synthetic_graph)
    figure = atlas.layered_structure_figure(
        synthetic_graph.edge_index,
        inputs["time_steps"], inputs["labels"], inputs["probabilities"],
        inputs["tx_ids"], inputs["degrees"],
        edge_cap=5,
    )
    assert "5 edges drawn" in figure.layout.title.text


def test_views_survive_an_empty_graph() -> None:
    """Each view degrades to a placeholder rather than raising."""
    empty = torch.empty(0)
    figure = atlas.embedding_atlas_figure(
        empty, empty, empty, empty, empty, empty
    )
    assert "No embeddings" in figure.layout.annotations[0].text
    structural = atlas.layered_structure_figure(
        torch.empty(2, 0, dtype=torch.long), empty, empty, empty, empty, empty
    )
    assert "No graph" in structural.layout.annotations[0].text
