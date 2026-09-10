"""Graph-free baseline used to quantify what message passing actually buys."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .base import BaseNodeClassifier, ClassifierHead
from .registry import register_model


@register_model("mlp")
class MLP(BaseNodeClassifier):
    """Per-node multilayer perceptron that ignores ``edge_index`` entirely.

    Reporting a GNN's F1 without this number is close to meaningless: the
    Elliptic feature vectors already contain 72 hand-engineered neighbourhood
    aggregates, so a plain MLP is a genuinely competitive baseline and the
    honest reference point for any claimed graph gain.
    """

    def __init__(self, in_channels: int, out_channels: int, config: ModelConfig) -> None:
        super().__init__(in_channels, out_channels, config)
        hidden = config.hidden_channels
        layers: list[nn.Module] = []
        width = in_channels
        for _ in range(config.num_layers):
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.BatchNorm1d(hidden) if config.batch_norm else nn.Identity())
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(config.dropout))
            width = hidden
        self.encoder = nn.Sequential(*layers)
        self._embedding_dim = hidden
        self.head = ClassifierHead(hidden, hidden, out_channels, config.dropout)

    @property
    def embedding_dim(self) -> int:
        """Width of the encoder output."""
        return self._embedding_dim

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return per-node embeddings; ``edge_index`` is accepted and ignored."""
        del edge_index  # Signature parity with the graph models is intentional.
        return self.encoder(x)

    def _classify(self, embedding: Tensor) -> Tensor:
        """Map embeddings to class logits."""
        return self.head(embedding)
