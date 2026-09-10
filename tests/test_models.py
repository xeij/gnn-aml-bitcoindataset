"""Architecture contracts shared by every registered model."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from aml_gnn.config import ModelConfig
from aml_gnn.models import GAT, available_models, build_model, get_model_class

_SMALL = ModelConfig(hidden_channels=32, num_layers=2, heads=4)


@pytest.mark.parametrize("name", available_models())
def test_forward_shapes(name: str, synthetic_graph: Data) -> None:
    """Every model maps node features to one logit row per node."""
    config = ModelConfig(**{**_SMALL.to_dict(), "name": name})
    model = build_model(config, in_channels=165, out_channels=2)
    model.eval()
    logits = model(synthetic_graph.x, synthetic_graph.edge_index)
    assert logits.shape == (synthetic_graph.num_nodes, 2)


@pytest.mark.parametrize("name", available_models())
def test_encode_matches_declared_embedding_dim(name: str, synthetic_graph: Data) -> None:
    """``embedding_dim`` describes what ``encode`` actually returns."""
    config = ModelConfig(**{**_SMALL.to_dict(), "name": name})
    model = build_model(config, in_channels=165, out_channels=2)
    model.eval()
    embedding = model.encode(synthetic_graph.x, synthetic_graph.edge_index)
    assert embedding.shape == (synthetic_graph.num_nodes, model.embedding_dim)


@pytest.mark.parametrize("name", available_models())
def test_probabilities_are_normalised(name: str, synthetic_graph: Data) -> None:
    """``predict_proba`` returns rows summing to one."""
    config = ModelConfig(**{**_SMALL.to_dict(), "name": name})
    model = build_model(config, in_channels=165, out_channels=2)
    probabilities = model.predict_proba(synthetic_graph.x, synthetic_graph.edge_index)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(synthetic_graph.num_nodes), atol=1e-5)


def test_predict_proba_restores_training_mode(synthetic_graph: Data) -> None:
    """Inference helpers must not silently leave the model in eval mode."""
    model = build_model(_SMALL, in_channels=165, out_channels=2)
    model.train()
    model.predict_proba(synthetic_graph.x, synthetic_graph.edge_index)
    assert model.training


def test_threshold_shifts_the_decision(synthetic_graph: Data) -> None:
    """A higher threshold can only reduce the number of illicit predictions."""
    model = build_model(_SMALL, in_channels=165, out_channels=2)
    lenient = model.predict(synthetic_graph.x, synthetic_graph.edge_index, threshold=0.1)
    strict = model.predict(synthetic_graph.x, synthetic_graph.edge_index, threshold=0.9)
    assert int(strict.sum()) <= int(lenient.sum())


def test_mlp_ignores_topology(synthetic_graph: Data) -> None:
    """The baseline must be genuinely graph-free, or it is not a baseline."""
    model = build_model(
        ModelConfig(**{**_SMALL.to_dict(), "name": "mlp"}), in_channels=165, out_channels=2
    )
    model.eval()
    shuffled = synthetic_graph.edge_index[:, torch.randperm(synthetic_graph.edge_index.size(1))]
    with torch.no_grad():
        assert torch.equal(
            model(synthetic_graph.x, synthetic_graph.edge_index),
            model(synthetic_graph.x, shuffled),
        )


def test_gnn_uses_topology(synthetic_graph: Data) -> None:
    """A graph model's output must change when the edges change."""
    model = build_model(
        ModelConfig(**{**_SMALL.to_dict(), "name": "gcn"}), in_channels=165, out_channels=2
    )
    model.eval()
    with torch.no_grad():
        original = model(synthetic_graph.x, synthetic_graph.edge_index)
        rewired = model(synthetic_graph.x, torch.randint(0, synthetic_graph.num_nodes, (2, 700)))
    assert not torch.allclose(original, rewired)


def test_jumping_knowledge_widens_the_embedding() -> None:
    """Concatenating layer outputs multiplies the embedding width."""
    with_jk = build_model(
        ModelConfig(hidden_channels=32, num_layers=3, jumping_knowledge=True),
        in_channels=165, out_channels=2,
    )
    without_jk = build_model(
        ModelConfig(hidden_channels=32, num_layers=3, jumping_knowledge=False),
        in_channels=165, out_channels=2,
    )
    assert with_jk.embedding_dim == 96
    assert without_jk.embedding_dim == 32


def test_gat_rejects_indivisible_head_count() -> None:
    """Head count must divide the hidden width so the declared shape holds."""
    with pytest.raises(ValueError, match="divisible"):
        build_model(
            ModelConfig(name="gat", hidden_channels=100, heads=3),
            in_channels=165, out_channels=2,
        )


def test_gat_returns_one_attention_weight_per_attended_edge(synthetic_graph: Data) -> None:
    """Attention extraction stays aligned with the edges the layer saw."""
    model = build_model(
        ModelConfig(**{**_SMALL.to_dict(), "name": "gat"}), in_channels=165, out_channels=2
    )
    assert isinstance(model, GAT)
    edges, alpha = model.attention_weights(synthetic_graph.x, synthetic_graph.edge_index)
    assert edges.size(1) == alpha.numel()


def test_registry_rejects_unknown_names() -> None:
    """An unregistered name fails with the list of valid options."""
    with pytest.raises(KeyError, match="Unknown model"):
        get_model_class("transformer")
