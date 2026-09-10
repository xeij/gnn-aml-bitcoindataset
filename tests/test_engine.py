"""End-to-end trainer behaviour and callback logic, on the synthetic graph."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from aml_gnn.config import ExperimentConfig
from aml_gnn.engine.callbacks import EarlyStopping
from aml_gnn.engine.trainer import Trainer, resolve_device, set_seed
from aml_gnn.models import build_model


def _config(tmp_path: Path, **train_overrides: object) -> ExperimentConfig:
    """Build a small, fast experiment config writing into ``tmp_path``."""
    config = ExperimentConfig(name="unit-test")
    config.model.hidden_channels = 16
    config.model.num_layers = 2
    config.train.epochs = 6
    config.train.device = "cpu"
    config.train.log_every = 100
    config.train.checkpoint_dir = tmp_path
    for key, value in train_overrides.items():
        setattr(config.train, key, value)
    return config


def _trainer(synthetic_graph: Data, config: ExperimentConfig) -> Trainer:
    """Assemble a trainer around a freshly built model.

    Seeding precedes model construction, mirroring the CLI: weight
    initialisation is part of what a seed has to pin down.
    """
    set_seed(config.train.seed)
    model = build_model(config.model, in_channels=synthetic_graph.x.size(1), out_channels=2)
    return Trainer(model, synthetic_graph.clone(), config, device=torch.device("cpu"))


def test_fit_produces_a_complete_result(synthetic_graph: Data, tmp_path: Path) -> None:
    """A finished run reports history, a threshold and test metrics."""
    result = _trainer(synthetic_graph, _config(tmp_path)).fit()
    assert len(result.history) == 6
    assert 0.0 < result.tuned_threshold < 1.0
    assert result.test_metrics.split == "test"
    assert result.checkpoint_path is not None and result.checkpoint_path.exists()


def test_checkpoint_is_self_describing(synthetic_graph: Data, tmp_path: Path) -> None:
    """A checkpoint carries everything needed to rebuild the model for serving."""
    result = _trainer(synthetic_graph, _config(tmp_path)).fit()
    assert result.checkpoint_path is not None
    state = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "config", "in_channels", "out_channels", "threshold"):
        assert key in state
    assert ExperimentConfig.from_dict(state["config"]).model.name == "gat"


def test_loss_ignores_nodes_outside_the_training_mask(
    synthetic_graph: Data, tmp_path: Path
) -> None:
    """Corrupting held-out labels must not change the training loss."""
    config = _config(tmp_path)
    trainer = _trainer(synthetic_graph, config)
    set_seed(0)
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(trainer.data.x, trainer.data.edge_index)
        before = float(trainer._loss(logits, trainer.data.train_mask))
        held_out = trainer.data.test_mask | trainer.data.val_mask
        trainer.data.y[held_out] = 1 - trainer.data.y[held_out]
        after = float(trainer._loss(logits, trainer.data.train_mask))
    assert before == pytest.approx(after)


def test_metrics_are_reproducible_under_a_fixed_seed(
    synthetic_graph: Data, tmp_path: Path
) -> None:
    """Two runs with the same seed produce the same test score."""
    first = _trainer(synthetic_graph, _config(tmp_path / "a")).fit()
    second = _trainer(synthetic_graph, _config(tmp_path / "b")).fit()
    assert first.test_metrics.illicit_f1 == pytest.approx(second.test_metrics.illicit_f1)


def test_progress_hook_fires_every_epoch(synthetic_graph: Data, tmp_path: Path) -> None:
    """The UI progress hook receives one call per epoch."""
    seen: list[int] = []
    _trainer(synthetic_graph, _config(tmp_path)).fit(
        progress_hook=lambda epoch, total, metrics: seen.append(epoch)
    )
    assert seen == [1, 2, 3, 4, 5, 6]


def test_class_weighting_can_be_disabled(synthetic_graph: Data, tmp_path: Path) -> None:
    """``class_weight=None`` really does drop the weight tensor."""
    trainer = _trainer(synthetic_graph, _config(tmp_path, class_weight=None))
    assert trainer.criterion_weight is None


def test_early_stopping_triggers_after_patience() -> None:
    """A flat metric stops the run once patience is exhausted."""
    stopper = EarlyStopping(patience=3, monitor="illicit_f1")
    assert stopper.step(0.5, epoch=1)
    for epoch in range(2, 5):
        assert not stopper.step(0.5, epoch=epoch)
    assert stopper.should_stop
    assert stopper.best_epoch == 1


def test_early_stopping_minimises_loss_metrics() -> None:
    """A monitor named ``loss`` is minimised, not maximised."""
    stopper = EarlyStopping(patience=2, monitor="val_loss")
    assert stopper.step(1.0, epoch=1)
    assert stopper.step(0.5, epoch=2)
    assert not stopper.step(0.9, epoch=3)


def test_resolve_device_honours_an_explicit_choice() -> None:
    """An explicit device preference is not overridden by availability."""
    assert resolve_device("cpu").type == "cpu"
