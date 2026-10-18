"""Command-line entry point for training a single experiment."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ..config import ExperimentConfig
from ..data.elliptic import EllipticDataModule
from ..engine.trainer import Trainer, set_seed
from ..logging_utils import configure_logging, get_logger
from ..models.registry import available_models, build_model

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``aml-train``."""
    parser = argparse.ArgumentParser(
        prog="aml-train",
        description="Train a GNN illicit-transaction detector on the Elliptic dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML experiment file. Defaults are used when omitted.",
    )
    parser.add_argument(
        "--model", choices=available_models(), default=None,
        help="Override the architecture named in the config.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch budget.")
    parser.add_argument("--lr", type=float, default=None, help="Override the learning rate.")
    parser.add_argument("--hidden", type=int, default=None, help="Override the hidden width.")
    parser.add_argument("--layers", type=int, default=None, help="Override the layer count.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--device", default=None, choices=["auto", "cpu", "cuda", "mps"],
        help="Override device selection.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Override the experiment name, which is also the checkpoint filename.",
    )
    parser.add_argument(
        "--local-features-only", action="store_true",
        help="Drop the 72 aggregated neighbour features (ablation).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Override where the checkpoint and metrics sidecar are written.",
    )
    for split in ("train", "val", "test"):
        parser.add_argument(
            f"--{split}-steps", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None,
            help=f"Override the {split} time-step window.",
        )
    parser.add_argument("--log-file", type=Path, default=None, help="Also log to this file.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level.",
    )
    return parser


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Fold command-line overrides into a loaded configuration."""
    if args.model is not None:
        config.model.name = args.model
        if args.name is None:
            config.name = f"elliptic-{args.model}"
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.lr is not None:
        config.train.learning_rate = args.lr
    if args.hidden is not None:
        config.model.hidden_channels = args.hidden
    if args.layers is not None:
        config.model.num_layers = args.layers
    if args.seed is not None:
        config.train.seed = args.seed
    if args.device is not None:
        config.train.device = args.device
    if args.local_features_only:
        config.data.local_features_only = True
    if args.name is not None:
        config.name = args.name
    if args.checkpoint_dir is not None:
        config.train.checkpoint_dir = args.checkpoint_dir
    for split in ("train", "val", "test"):
        bounds = getattr(args, f"{split}_steps")
        if bounds is not None:
            setattr(config.data, f"{split}_time_steps", tuple(bounds))
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """Run a training experiment and return a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, log_file=args.log_file, force=True)

    config = (
        ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    )
    config = apply_overrides(config, args)

    # Seed before the model exists: weight initialisation is part of the run,
    # so seeding only inside the Trainer would leave results irreproducible.
    set_seed(config.train.seed)

    data_module = EllipticDataModule(config.data)
    data = data_module.prepare()

    model = build_model(
        config.model,
        in_channels=data_module.num_features,
        out_channels=data_module.num_classes,
    )
    scaler = data_module.scaler
    trainer = Trainer(
        model,
        data,
        config,
        scaler_state=scaler.state_dict() if scaler is not None else None,
    )
    trainer.fit()

    # The trainer already logs the run summary on completion, so the CLI only
    # persists the resolved config beside the checkpoint.
    config.to_yaml(Path(config.train.checkpoint_dir) / f"{config.name}_config.yaml")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
