"""Full-batch transductive training loop for Elliptic node classification.

The whole graph (204k nodes, 469k directed edge entries, 165 features) is
about 135 MB in float32, so it fits comfortably in device memory and there is
no reason to mini-batch: full-batch training gives every node its exact
k-hop receptive field and removes neighbour-sampling variance from the
results.  Supervision is masked -- message passing runs over all nodes,
including the 77% with no label, while the loss is taken only on the
training window.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data

from ..config import ExperimentConfig
from ..data.splits import class_weights
from ..logging_utils import get_logger
from ..models.base import BaseNodeClassifier
from .calibration import apply_temperature, expected_calibration_error, fit_temperature
from .callbacks import EarlyStopping, ModelCheckpoint
from .metrics import ClassificationMetrics, masked_evaluate, select_threshold

logger = get_logger(__name__)

#: Signature of an optional per-epoch progress callback (used by the Gradio app).
ProgressHook = Callable[[int, int, dict[str, float]], None]


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a compute device, preferring CUDA, then Apple MPS, then CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for a reproducible run."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Seeded all RNGs with %d", seed)


@dataclass
class EpochRecord:
    """One row of the training history."""

    epoch: int
    train_loss: float
    val_loss: float
    val_illicit_f1: float
    val_illicit_precision: float
    val_illicit_recall: float
    val_average_precision: float
    learning_rate: float
    seconds: float

    def as_log_line(self) -> str:
        """Render the record for the console log."""
        return (
            f"epoch {self.epoch:>4} | train_loss {self.train_loss:.4f} "
            f"| val_loss {self.val_loss:.4f} | val F1 {self.val_illicit_f1:.4f} "
            f"P {self.val_illicit_precision:.4f} R {self.val_illicit_recall:.4f} "
            f"| AP {self.val_average_precision:.4f} | lr {self.learning_rate:.2e} "
            f"| {self.seconds:.2f}s"
        )


@dataclass
class TrainingResult:
    """Everything a completed run produces."""

    history: list[EpochRecord]
    best_epoch: int
    best_val_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    tuned_threshold: float
    temperature: float
    calibration_error: float
    checkpoint_path: Path | None
    total_seconds: float
    config: ExperimentConfig
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a multi-line summary suitable for the console or the UI."""
        return "\n".join([
            f"Run '{self.config.name}' finished in {self.total_seconds:.1f}s "
            f"({len(self.history)} epochs, best at {self.best_epoch})",
            f"  tuned threshold: {self.tuned_threshold:.4f}, temperature: "
            f"{self.temperature:.3f} (both fitted on validation, "
            f"val ECE {self.calibration_error:.4f})",
            f"  {self.best_val_metrics}",
            f"  {self.test_metrics}",
        ])


class Trainer:
    """Orchestrates optimisation, evaluation, checkpointing and early stopping.

    The trainer holds no architecture-specific logic; any
    :class:`~aml_gnn.models.base.BaseNodeClassifier` can be dropped in.

    Example:
        >>> trainer = Trainer(model, data, config)
        >>> result = trainer.fit()
        >>> result.test_metrics.illicit_f1 > 0.0
        True

    """

    def __init__(
        self,
        model: BaseNodeClassifier,
        data: Data,
        config: ExperimentConfig,
        *,
        device: torch.device | None = None,
        scaler_state: dict[str, Tensor] | None = None,
    ) -> None:
        self.config = config
        self.device = device or resolve_device(config.train.device)
        self.model: BaseNodeClassifier = model.to(self.device)
        self.data: Data = data.to(self.device)
        self._scaler_state = scaler_state

        set_seed(config.train.seed)

        self.criterion_weight: Tensor | None = None
        if config.train.class_weight == "balanced":
            self.criterion_weight = class_weights(
                data.y.cpu(),
                data.train_mask.cpu(),
                illicit_multiplier=config.train.illicit_weight_multiplier,
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=config.train.lr_scheduler_patience,
            min_lr=1e-6,
        )
        self.early_stopping = EarlyStopping(
            patience=config.train.early_stopping_patience, monitor=config.train.monitor
        )
        self.checkpoint = ModelCheckpoint(
            config.train.checkpoint_dir, name=config.name, monitor=config.train.monitor
        )
        self.history: list[EpochRecord] = []

        logger.info("Trainer ready on device '%s'", self.device)
        logger.info("  %s", config.describe())
        if self.criterion_weight is not None:
            logger.info(
                "  loss weights: licit=%.3f illicit=%.3f",
                self.criterion_weight[0], self.criterion_weight[1],
            )

    # --------------------------------------------------------------- steps

    def _loss(self, logits: Tensor, mask: Tensor) -> Tensor:
        """Weighted cross-entropy restricted to labelled nodes in ``mask``."""
        return F.cross_entropy(logits[mask], self.data.y[mask], weight=self.criterion_weight)

    def train_epoch(self) -> float:
        """Run one full-batch optimisation step and return the training loss."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(self.data.x, self.data.edge_index)
        loss = self._loss(logits, self.data.train_mask)
        loss.backward()
        if self.config.train.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)
        self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(
        self, split: str, *, threshold: float = 0.5, logits: Tensor | None = None
    ) -> tuple[ClassificationMetrics, float, Tensor]:
        """Evaluate one split, returning ``(metrics, loss, logits)``.

        Passing ``logits`` in reuses a forward pass across several splits.
        """
        self.model.eval()
        if logits is None:
            logits = self.model(self.data.x, self.data.edge_index)
        mask = getattr(self.data, f"{split}_mask")
        loss = float(self._loss(logits, mask))
        metrics = masked_evaluate(
            logits,
            self.data.y,
            mask,
            split=split,
            threshold=threshold,
            time_step=getattr(self.data, "time_step", None),
        )
        return metrics, loss, logits

    # ----------------------------------------------------------------- fit

    def fit(self, *, progress_hook: ProgressHook | None = None) -> TrainingResult:
        """Train to convergence and return the full result bundle.

        The sequence is: optimise on the train window, monitor the validation
        window each epoch, keep the best checkpoint, then -- once training is
        done -- reload that checkpoint, tune the decision threshold on
        validation, and evaluate the test window exactly once at that frozen
        threshold.
        """
        train_config = self.config.train
        started = time.perf_counter()
        logger.info("Training for up to %d epochs", train_config.epochs)

        for epoch in range(1, train_config.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self.train_epoch()
            val_metrics, val_loss, _ = self.evaluate("val")

            monitored = val_metrics.get(train_config.monitor)
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_illicit_f1=val_metrics.illicit_f1,
                val_illicit_precision=val_metrics.illicit_precision,
                val_illicit_recall=val_metrics.illicit_recall,
                val_average_precision=val_metrics.average_precision,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                seconds=time.perf_counter() - epoch_start,
            )
            self.history.append(record)
            self.scheduler.step(monitored)

            if self.early_stopping.step(monitored, epoch):
                self.checkpoint.save(
                    self.model,
                    epoch=epoch,
                    value=monitored,
                    payload=self._checkpoint_payload(val_metrics),
                )

            if epoch % train_config.log_every == 0 or epoch == 1:
                logger.info(record.as_log_line())
            if progress_hook is not None:
                progress_hook(epoch, train_config.epochs, {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_illicit_f1": val_metrics.illicit_f1,
                    "best": self.early_stopping.best,
                })
            if self.early_stopping.should_stop:
                break

        return self._finalise(started)

    def _finalise(self, started: float) -> TrainingResult:
        """Reload the best weights, tune the threshold, score the test window."""
        best_epoch = self.early_stopping.best_epoch
        if self.checkpoint.path.exists():
            state = torch.load(self.checkpoint.path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state_dict"])
            logger.info("Restored best weights from epoch %d", state["epoch"])
        else:
            logger.warning("No checkpoint was written; evaluating final-epoch weights")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)

        # Calibrate before choosing the threshold, not after: the threshold is a
        # point on the probability scale, so rescaling the scale moves it.
        val_labels = self.data.y[self.data.val_mask]
        temperature = fit_temperature(logits[self.data.val_mask], val_labels)
        probabilities = apply_temperature(logits, temperature)

        threshold, _ = select_threshold(
            val_labels,
            probabilities[self.data.val_mask],
            objective="f1",
        )
        calibration_error = expected_calibration_error(
            probabilities[self.data.val_mask], val_labels
        )
        # ``evaluate`` softmaxes whatever logits it is handed, so scaling them
        # here is what makes its metrics agree with the calibrated threshold.
        scaled = logits / temperature
        val_metrics, _, _ = self.evaluate("val", threshold=threshold, logits=scaled)
        test_metrics, _, _ = self.evaluate("test", threshold=threshold, logits=scaled)

        # Persist threshold and temperature with the weights so serving matches
        # training: both are fitted parameters, and both live in the checkpoint.
        if self.checkpoint.path.exists():
            state = torch.load(self.checkpoint.path, map_location="cpu", weights_only=False)
            state["threshold"] = threshold
            state["temperature"] = temperature
            state["test_metrics"] = test_metrics.to_dict()
            torch.save(state, self.checkpoint.path)

        result = TrainingResult(
            history=self.history,
            best_epoch=best_epoch,
            best_val_metrics=val_metrics,
            test_metrics=test_metrics,
            tuned_threshold=threshold,
            temperature=temperature,
            calibration_error=calibration_error,
            checkpoint_path=self.checkpoint.path if self.checkpoint.path.exists() else None,
            total_seconds=time.perf_counter() - started,
            config=self.config,
        )
        self.checkpoint.save_metrics({
            "config": self.config.to_dict(),
            "best_epoch": best_epoch,
            "num_parameters": self.model.num_parameters(),
            "total_seconds": result.total_seconds,
            "threshold": threshold,
            "temperature": temperature,
            "calibration_error": calibration_error,
            "validation": val_metrics.to_dict(),
            "test": test_metrics.to_dict(),
            "history": [record.__dict__ for record in self.history],
        })
        logger.info("\n%s", result.summary())
        return result

    def _checkpoint_payload(self, val_metrics: ClassificationMetrics) -> dict[str, Any]:
        """Assemble the non-weight contents of a checkpoint."""
        return {
            "config": self.config.to_dict(),
            "model_name": self.config.model.name,
            "in_channels": self.model.in_channels,
            "out_channels": self.model.out_channels,
            "scaler_state": self._scaler_state,
            "val_metrics": val_metrics.to_dict(),
            "threshold": 0.5,
            # Overwritten on finalisation; 1.0 keeps a mid-run checkpoint honest
            # about not having been calibrated yet.
            "temperature": 1.0,
        }
