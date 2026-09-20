"""Typed, serialisable configuration objects.

Every knob in the pipeline is declared here as a frozen-by-convention
dataclass.  Configs round-trip through YAML so that an experiment is fully
described by a single file committed alongside its checkpoint.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import yaml

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from _typeshed import DataclassInstance

ModelName = Literal["gcn", "gat", "graphsage", "mlp"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]

_T = TypeVar("_T", bound="_ConfigBase")


def _stringify_paths(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """Dataclass ``dict_factory`` that renders ``Path`` values as strings."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in items}


class _ConfigBase:
    """Mixin providing dict/YAML (de)serialisation for nested dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view, with ``Path`` values stringified."""
        return dataclasses.asdict(
            cast("DataclassInstance", self), dict_factory=_stringify_paths
        )

    @classmethod
    def from_dict(cls: type[_T], payload: Mapping[str, Any]) -> _T:
        """Build an instance from a (possibly partial) mapping.

        Unknown keys raise rather than being silently dropped, so a typo in a
        YAML file fails loudly instead of leaving a default in place.
        """
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(payload) - set(field_types)
        if unknown:
            raise ValueError(
                f"{cls.__name__}: unknown configuration keys {sorted(unknown)}; "
                f"valid keys are {sorted(field_types)}"
            )
        kwargs: dict[str, Any] = {}
        for key, value in payload.items():
            annotation = field_types[key]
            nested = _nested_config_type(annotation)
            if nested is not None and isinstance(value, Mapping):
                kwargs[key] = nested.from_dict(value)
            elif annotation in ("Path", Path) and value is not None:
                kwargs[key] = Path(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


def _nested_config_type(annotation: Any) -> type[_ConfigBase] | None:
    """Resolve a dataclass field annotation to a nested config class, if any."""
    if isinstance(annotation, type) and issubclass(annotation, _ConfigBase):
        return annotation
    if isinstance(annotation, str):
        candidate = globals().get(annotation)
        if isinstance(candidate, type) and issubclass(candidate, _ConfigBase):
            return candidate
    return None


@dataclass
class DataConfig(_ConfigBase):
    """Dataset location, temporal split boundaries and preprocessing switches.

    The Elliptic graph spans 49 discrete time steps.  Splitting on time rather
    than at random is essential: transactions within a time step are densely
    interconnected, so a random split leaks near-duplicate neighbours across
    the boundary and inflates test scores.
    """

    root: Path = Path("data/elliptic")
    train_time_steps: tuple[int, int] = (1, 30)
    val_time_steps: tuple[int, int] = (31, 34)
    test_time_steps: tuple[int, int] = (35, 49)
    #: Drop the 72 aggregated neighbour features, keeping only the 93 local
    #: ones.  The paper reports this ablation; message passing can in principle
    #: rediscover the aggregates itself.
    local_features_only: bool = False
    #: Standardise features using statistics computed on the training split only.
    standardise: bool = True
    #: Treat the directed payment flow graph as undirected.  Illicit signal
    #: propagates in both directions, and most PyG convolutions assume symmetry.
    to_undirected: bool = True
    #: Append a self-loop to every node (in addition to any added by the conv).
    add_self_loops: bool = False
    force_reload: bool = False

    def __post_init__(self) -> None:
        """Coerce fields to their declared types after YAML deserialisation.

        YAML has no tuple type, so a round-tripped config would otherwise come
        back with lists where the annotations promise ``tuple[int, int]``.
        Normalising here keeps two configs describing the same experiment
        comparable, which the checkpoint cache in the app relies on.
        """
        self.root = Path(self.root)
        for field_name in ("train_time_steps", "val_time_steps", "test_time_steps"):
            bounds = tuple(int(value) for value in getattr(self, field_name))
            if len(bounds) != 2:
                raise ValueError(
                    f"{field_name} must hold exactly two time steps, got {bounds}"
                )
            setattr(self, field_name, bounds)


@dataclass
class ModelConfig(_ConfigBase):
    """Architecture hyper-parameters shared by every registered model."""

    name: ModelName = "gat"
    hidden_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    #: GAT only: number of attention heads on every layer but the last.
    heads: int = 4
    #: Concatenate every layer's output before the classifier (jumping knowledge).
    jumping_knowledge: bool = True
    #: Insert a residual connection around each message-passing layer.
    residual: bool = True
    #: Apply batch normalisation after each message-passing layer.
    batch_norm: bool = True


@dataclass
class TrainConfig(_ConfigBase):
    """Optimisation schedule, imbalance handling and early stopping."""

    epochs: int = 300
    learning_rate: float = 5e-3
    weight_decay: float = 5e-4
    #: Inverse-frequency weighting of the illicit class in the loss.  ``None``
    #: disables weighting; ``"balanced"`` derives weights from the train split.
    class_weight: Literal["balanced"] | None = "balanced"
    #: Extra multiplier applied to the illicit class weight.  Values above 1
    #: trade precision for recall, which is what an AML analyst usually wants.
    illicit_weight_multiplier: float = 1.0
    early_stopping_patience: int = 40
    #: Validation metric that drives checkpointing and early stopping.
    monitor: str = "illicit_f1"
    grad_clip_norm: float | None = 1.0
    lr_scheduler_patience: int = 15
    seed: int = 42
    device: DeviceName = "auto"
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    log_every: int = 10


@dataclass
class ExperimentConfig(_ConfigBase):
    """Top-level container binding a data, model and training configuration."""

    name: str = "elliptic-gat"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load an experiment from a YAML file, applying defaults for absent keys."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls.from_dict(payload)

    def to_yaml(self, path: str | Path) -> Path:
        """Serialise the experiment to YAML and return the written path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return path

    def describe(self) -> str:
        """Return a compact single-line summary for log headers."""
        return (
            f"{self.name} | model={self.model.name}"
            f"(h={self.model.hidden_channels}, L={self.model.num_layers})"
            f" | epochs={self.train.epochs} lr={self.train.learning_rate}"
            f" | weighting={self.train.class_weight}"
        )


def time_step_bounds(config: DataConfig) -> Sequence[tuple[int, int]]:
    """Return the (train, val, test) time-step ranges, validating they are disjoint."""
    spans = [config.train_time_steps, config.val_time_steps, config.test_time_steps]
    for low, high in spans:
        if low > high:
            raise ValueError(f"Invalid time-step span ({low}, {high}): low exceeds high")
    ordered = sorted(spans)
    for (_, prev_high), (next_low, _) in zip(ordered, ordered[1:], strict=False):
        if next_low <= prev_high:
            raise ValueError(
                f"Time-step spans overlap: {ordered}. Temporal splits must be "
                "disjoint or the evaluation leaks future information."
            )
    return spans
