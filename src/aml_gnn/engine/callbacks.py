"""Early stopping and best-checkpoint tracking, decoupled from the training loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from ..logging_utils import get_logger

logger = get_logger(__name__)

#: Metrics where a lower value is better; everything else is maximised.
_MINIMISE = frozenset({"loss", "val_loss", "train_loss"})


@dataclass
class EarlyStopping:
    """Stop training once the monitored metric stops improving.

    Args:
        patience: Epochs to wait after the last improvement.
        monitor: Metric key, used only to infer the optimisation direction.
        min_delta: Minimum change that counts as an improvement.
    """

    patience: int = 40
    monitor: str = "illicit_f1"
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        self.mode: str = "min" if self.monitor in _MINIMISE else "max"
        self.best: float = float("inf") if self.mode == "min" else float("-inf")
        self.best_epoch: int = -1
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False

    def is_improvement(self, value: float) -> bool:
        """Return whether ``value`` beats the best seen by at least ``min_delta``."""
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Record an epoch's metric and return whether it was an improvement."""
        if self.is_improvement(value):
            self.best, self.best_epoch, self.num_bad_epochs = value, epoch, 0
            return True
        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
            logger.info(
                "Early stopping at epoch %d: %s has not improved on %.4f "
                "(epoch %d) for %d epochs",
                epoch, self.monitor, self.best, self.best_epoch, self.patience,
            )
        return False


class ModelCheckpoint:
    """Persists the best model together with everything needed to reload it.

    A checkpoint here is self-describing: weights, the config that built them,
    the fitted feature scaler and the tuned decision threshold all travel in
    one file, so serving never has to guess how the model was trained.
    """

    def __init__(self, directory: Path, *, name: str, monitor: str = "illicit_f1") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.monitor = monitor
        self.best_value: Optional[float] = None
        self.path: Path = self.directory / f"{name}.pt"
        self.metrics_path: Path = self.directory / f"{name}_metrics.json"

    def save(
        self,
        model: nn.Module,
        *,
        epoch: int,
        value: float,
        payload: Mapping[str, Any],
    ) -> Path:
        """Write a checkpoint to disk and return its path."""
        state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "monitor": self.monitor,
            "monitor_value": value,
            **payload,
        }
        torch.save(state, self.path)
        self.best_value = value
        logger.info(
            "Checkpoint saved at epoch %d (%s=%.4f) -> %s", epoch, self.monitor, value, self.path
        )
        return self.path

    def save_metrics(self, metrics: Mapping[str, Any]) -> Path:
        """Write a JSON sidecar with the final evaluation of this run."""
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=str)
        logger.info("Metrics written to %s", self.metrics_path)
        return self.metrics_path
