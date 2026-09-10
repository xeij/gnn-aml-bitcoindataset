"""Configuration round-tripping and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from aml_gnn.config import DataConfig, ExperimentConfig, time_step_bounds


def test_round_trips_through_yaml(tmp_path: Path) -> None:
    """A config written to YAML reloads to an equal object."""
    original = ExperimentConfig()
    original.model.hidden_channels = 64
    original.train.epochs = 7
    path = original.to_yaml(tmp_path / "experiment.yaml")

    reloaded = ExperimentConfig.from_yaml(path)
    assert reloaded.model.hidden_channels == 64
    assert reloaded.train.epochs == 7
    assert reloaded.to_dict() == original.to_dict()


def test_rejects_unknown_keys() -> None:
    """A typo in a YAML key raises instead of silently keeping the default."""
    with pytest.raises(ValueError, match="unknown configuration keys"):
        ExperimentConfig.from_dict({"modle": {}})


def test_rejects_overlapping_time_windows() -> None:
    """Overlapping splits are refused, since they would leak across time."""
    config = DataConfig(train_time_steps=(1, 34), val_time_steps=(30, 36))
    with pytest.raises(ValueError, match="overlap"):
        time_step_bounds(config)


def test_accepts_disjoint_time_windows() -> None:
    """The default windows are disjoint and ordered."""
    assert time_step_bounds(DataConfig()) == [(1, 30), (31, 34), (35, 49)]
