"""Name-to-class registry so architectures are selectable from config or the UI."""

from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

from ..config import ModelConfig
from ..logging_utils import get_logger
from .base import BaseNodeClassifier

logger = get_logger(__name__)

_REGISTRY: Dict[str, Type[BaseNodeClassifier]] = {}

_M = TypeVar("_M", bound=Type[BaseNodeClassifier])


def register_model(name: str) -> Callable[[_M], _M]:
    """Class decorator registering an architecture under ``name``.

    Registration replaces an ``if/elif`` chain in the factory: adding an
    architecture means adding one file, not editing three.
    """
    key = name.lower()

    def decorator(cls: _M) -> _M:
        if key in _REGISTRY:
            raise ValueError(f"Model {key!r} is already registered to {_REGISTRY[key].__name__}")
        cls.model_name = key
        _REGISTRY[key] = cls
        logger.debug("Registered model %r -> %s", key, cls.__name__)
        return cls

    return decorator


def available_models() -> list[str]:
    """Return every registered architecture name, sorted."""
    return sorted(_REGISTRY)


def get_model_class(name: str) -> Type[BaseNodeClassifier]:
    """Look up a registered architecture class by name."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    return _REGISTRY[key]


def build_model(config: ModelConfig, *, in_channels: int, out_channels: int) -> BaseNodeClassifier:
    """Instantiate the architecture named by ``config``.

    Args:
        config: Architecture hyper-parameters, including the registry name.
        in_channels: Input feature dimensionality.
        out_channels: Number of classes.
    """
    cls = get_model_class(config.name)
    model = cls(in_channels=in_channels, out_channels=out_channels, config=config)
    logger.info("Built %s", model.describe())
    return model
