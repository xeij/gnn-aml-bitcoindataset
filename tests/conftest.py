"""Shared fixtures. Most tests run on a synthetic graph so they stay fast."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def synthetic_graph() -> Data:
    """A small imbalanced graph with the same shape contract as Elliptic.

    Time steps 1-6 map to train, 7-8 to validation and 9-12 to test, matching
    the real module's three-way temporal partition on a 200-node graph.
    """
    generator = torch.Generator().manual_seed(0)
    num_nodes, num_features = 200, 165
    x = torch.randn(num_nodes, num_features, generator=generator)
    time_step = torch.randint(1, 13, (num_nodes,), generator=generator)

    y = torch.full((num_nodes,), 2, dtype=torch.long)
    labelled = torch.randperm(num_nodes, generator=generator)[:150]
    y[labelled] = (torch.rand(150, generator=generator) < 0.25).long()
    # Guarantee both classes appear in every window so metrics stay defined.
    for bounds in ((1, 6), (7, 8), (9, 12)):
        in_window = ((time_step >= bounds[0]) & (time_step <= bounds[1])).nonzero().view(-1)
        y[in_window[:8]] = 0
        y[in_window[8:14]] = 1

    edge_index = torch.randint(0, num_nodes, (2, 700), generator=generator)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.time_step = time_step
    data.tx_id = torch.arange(num_nodes) + 100_000
    data.train_mask = (time_step <= 6) & (y != 2)
    data.val_mask = (time_step >= 7) & (time_step <= 8) & (y != 2)
    data.test_mask = (time_step >= 9) & (y != 2)
    return data
