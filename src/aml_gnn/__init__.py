"""Graph-neural-network pipeline for anti-money-laundering detection.

Four modules compose the pipeline:

* :mod:`aml_gnn.data` -- the official Elliptic Bitcoin dataset, temporally split;
* :mod:`aml_gnn.models` -- interchangeable message-passing classifiers;
* :mod:`aml_gnn.engine` -- training, evaluation, checkpointing and inference;
* :mod:`aml_gnn.app` -- the Gradio investigation interface.
"""

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from .logging_utils import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainConfig",
    "__version__",
    "configure_logging",
    "get_logger",
]
