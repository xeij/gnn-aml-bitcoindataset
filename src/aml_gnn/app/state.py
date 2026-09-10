"""Session state for the Gradio app: checkpoint discovery, loading and caching.

The graph is expensive to prepare and the scored probabilities are expensive to
compute, so both are cached and shared across model switches whenever the
underlying data configuration is identical.  Everything the UI needs is reached
through this object, which keeps the callbacks in ``interface`` thin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ..config import DataConfig, ExperimentConfig
from ..data.elliptic import EllipticDataModule
from ..engine.inference import InferenceService
from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_CHECKPOINT_DIR: Path = Path("artifacts/checkpoints")


def discover_checkpoints(directory: Path = DEFAULT_CHECKPOINT_DIR) -> dict[str, Path]:
    """Return ``{display_name: path}`` for every checkpoint in ``directory``."""
    directory = Path(directory)
    if not directory.exists():
        logger.warning("Checkpoint directory %s does not exist", directory)
        return {}
    found = {path.stem: path for path in sorted(directory.glob("*.pt"))}
    logger.info("Discovered %d checkpoint(s) in %s: %s",
                len(found), directory, ", ".join(found) or "none")
    return found


class AppState:
    """Holds the active inference service and the shared, cached data module.

    Args:
        checkpoint_dir: Directory scanned for ``*.pt`` checkpoints.
        default_checkpoint: Name to activate on startup; falls back to the
            first checkpoint found.
    """

    def __init__(
        self,
        checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
        *,
        default_checkpoint: Optional[str] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints: dict[str, Path] = discover_checkpoints(self.checkpoint_dir)
        self._service: Optional[InferenceService] = None
        self._active_name: Optional[str] = None
        self._data_modules: dict[str, EllipticDataModule] = {}

        preferred = default_checkpoint or self._preferred_default()
        if preferred is not None:
            try:
                self.activate(preferred)
            except Exception:
                logger.exception("Could not load checkpoint %r at startup", preferred)

    # ---------------------------------------------------------------- state

    @property
    def is_ready(self) -> bool:
        """Whether a model is loaded and able to answer queries."""
        return self._service is not None

    @property
    def service(self) -> InferenceService:
        """The active inference service, raising a clear error when absent."""
        if self._service is None:
            raise RuntimeError(
                "No model is loaded. Train one first:\n"
                "  python scripts/train.py --config configs/gat.yaml"
            )
        return self._service

    @property
    def active_name(self) -> Optional[str]:
        """Name of the currently active checkpoint."""
        return self._active_name

    @property
    def available(self) -> list[str]:
        """Names of every discovered checkpoint."""
        return list(self.checkpoints)

    def _preferred_default(self) -> Optional[str]:
        """Prefer a GAT checkpoint, else the first available one."""
        for name in self.checkpoints:
            if "gat" in name.lower():
                return name
        return next(iter(self.checkpoints), None)

    # --------------------------------------------------------------- loading

    def refresh(self) -> list[str]:
        """Rescan the checkpoint directory and return the available names."""
        self.checkpoints = discover_checkpoints(self.checkpoint_dir)
        return self.available

    def activate(self, name: str) -> InferenceService:
        """Load a checkpoint by name and make it the active model."""
        if name not in self.checkpoints:
            self.refresh()
        if name not in self.checkpoints:
            raise KeyError(f"Unknown checkpoint {name!r}. Available: {self.available}")
        if self._active_name == name and self._service is not None:
            return self._service

        path = self.checkpoints[name]
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = ExperimentConfig.from_dict(state["config"])
        module = self._data_module_for(config.data)

        self._service = InferenceService.from_checkpoint(path, data_module=module)
        self._active_name = name
        logger.info("Active model is now %r", name)
        return self._service

    def _data_module_for(self, config: DataConfig) -> EllipticDataModule:
        """Return a prepared data module, reusing one when the config matches.

        Preparing the graph costs several seconds and ~250 MB, so two models
        trained on identical data settings share a single instance.
        """
        key = repr(sorted(config.to_dict().items()))
        module = self._data_modules.get(key)
        if module is None:
            logger.info("Preparing a data module for this configuration")
            module = EllipticDataModule(config)
            module.prepare()
            self._data_modules[key] = module
        else:
            logger.debug("Reusing cached data module")
        return module
