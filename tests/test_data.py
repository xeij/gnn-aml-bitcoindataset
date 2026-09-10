"""Split construction, class weighting and feature preprocessing."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from aml_gnn.data.splits import assert_disjoint, class_weights, summarise_split, temporal_mask
from aml_gnn.data.transforms import StandardScaler, sanitise_features, select_feature_block


def test_temporal_mask_excludes_unlabelled_and_other_windows(synthetic_graph: Data) -> None:
    """A window mask selects only labelled nodes inside its own time range."""
    mask = temporal_mask(synthetic_graph.time_step, synthetic_graph.y, (1, 6))
    assert mask.sum() > 0
    assert (synthetic_graph.time_step[mask] <= 6).all()
    assert (synthetic_graph.y[mask] != 2).all()


def test_splits_are_disjoint(synthetic_graph: Data) -> None:
    """No node may appear in more than one split."""
    assert_disjoint([
        synthetic_graph.train_mask, synthetic_graph.val_mask, synthetic_graph.test_mask
    ])


def test_overlapping_splits_are_rejected(synthetic_graph: Data) -> None:
    """An overlap is caught rather than silently inflating scores."""
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint([synthetic_graph.train_mask, synthetic_graph.train_mask])


def test_class_weights_favour_the_minority(synthetic_graph: Data) -> None:
    """The rarer class receives the larger loss weight."""
    weights = class_weights(synthetic_graph.y, synthetic_graph.train_mask)
    counts = torch.bincount(synthetic_graph.y[synthetic_graph.train_mask], minlength=2)
    minority = int(counts.argmin())
    assert weights[minority] == weights.max()
    assert weights.mean() == pytest.approx(1.0, abs=1e-5)


def test_illicit_multiplier_increases_recall_pressure(synthetic_graph: Data) -> None:
    """Raising the multiplier raises the illicit weight relative to licit."""
    base = class_weights(synthetic_graph.y, synthetic_graph.train_mask)
    boosted = class_weights(
        synthetic_graph.y, synthetic_graph.train_mask, illicit_multiplier=3.0
    )
    assert boosted[1] / boosted[0] > base[1] / base[0]


def test_scaler_fits_on_the_masked_subset_only(synthetic_graph: Data) -> None:
    """Standardisation statistics come from the fitting mask, not the full matrix."""
    scaler = StandardScaler.fit(synthetic_graph.x, synthetic_graph.train_mask)
    transformed = scaler.transform(synthetic_graph.x)
    fitted = transformed[synthetic_graph.train_mask]
    assert torch.allclose(fitted.mean(dim=0), torch.zeros(fitted.size(1)), atol=1e-5)
    assert torch.allclose(fitted.std(dim=0), torch.ones(fitted.size(1)), atol=1e-4)


def test_scaler_state_round_trips(synthetic_graph: Data) -> None:
    """A checkpointed scaler reproduces the original transform exactly."""
    scaler = StandardScaler.fit(synthetic_graph.x, synthetic_graph.train_mask)
    restored = StandardScaler.from_state_dict(scaler.state_dict())
    assert torch.equal(scaler.transform(synthetic_graph.x), restored.transform(synthetic_graph.x))


def test_local_feature_block_drops_the_aggregates(synthetic_graph: Data) -> None:
    """The ablation keeps exactly the 93 local columns."""
    assert select_feature_block(synthetic_graph.x, local_only=True).size(1) == 93
    assert select_feature_block(synthetic_graph.x, local_only=False).size(1) == 165


def test_sanitise_replaces_non_finite_values() -> None:
    """NaN and infinity are replaced with zero."""
    x = torch.tensor([[1.0, float("nan")], [float("inf"), 2.0]])
    cleaned = sanitise_features(x)
    assert torch.isfinite(cleaned).all()
    assert cleaned[0, 1] == 0.0 and cleaned[1, 0] == 0.0


def test_split_summary_counts_classes(synthetic_graph: Data) -> None:
    """The summary reports counts consistent with the mask it describes."""
    summary = summarise_split("train", (1, 6), synthetic_graph.y, synthetic_graph.train_mask)
    assert summary.num_nodes == summary.num_licit + summary.num_illicit
    assert 0.0 <= summary.illicit_ratio <= 1.0


def test_sample_node_indices_is_device_agnostic(synthetic_graph: Data) -> None:
    """Sampling works when the graph has been moved off the CPU.

    A serving layer moves the graph to an accelerator, so any mask built here
    has to follow it rather than assuming CPU tensors.
    """
    from aml_gnn.config import DataConfig
    from aml_gnn.data.elliptic import EllipticDataModule

    module = EllipticDataModule(DataConfig())
    # Clone first: ``Data.to`` moves tensors in place, and the fixture is
    # session-scoped, so moving it directly would strand every later test on
    # the accelerator.
    module._data = synthetic_graph.clone().to(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    indices = module.sample_node_indices(label=1, split="test", limit=3)
    assert indices and all(isinstance(i, int) for i in indices)
    assert all(synthetic_graph.y[i] == 1 for i in indices)
