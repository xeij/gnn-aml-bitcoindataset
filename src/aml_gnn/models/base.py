"""Abstract base class and shared building blocks for node classifiers.

Every architecture in this package follows the same template: an *encoder*
maps ``(x, edge_index)`` to node embeddings, and a *head* maps embeddings to
class logits.  Keeping that boundary explicit means the Gradio app can request
embeddings for projection plots without knowing which architecture produced
them, and the trainer can stay entirely architecture-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class BaseNodeClassifier(nn.Module, ABC):
    """Interface every AML node classifier implements.

    Subclasses provide :meth:`encode`; ``forward`` and the inference helpers
    are inherited so that behaviour stays identical across architectures.
    """

    #: Registry key, set by :func:`aml_gnn.models.registry.register_model`.
    model_name: str = "base"

    def __init__(self, in_channels: int, out_channels: int, config: ModelConfig) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config

    @abstractmethod
    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return node embeddings of shape ``[num_nodes, embedding_dim]``."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Width of the representation returned by :meth:`encode`."""

    @abstractmethod
    def _classify(self, embedding: Tensor) -> Tensor:
        """Map embeddings to class logits."""

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return class logits of shape ``[num_nodes, out_channels]``."""
        return self._classify(self.encode(x, edge_index))

    @torch.no_grad()
    def predict_proba(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return per-class probabilities in evaluation mode.

        Restores the previous train/eval state so callers can use this mid-training.
        """
        was_training = self.training
        self.eval()
        try:
            return F.softmax(self.forward(x, edge_index), dim=-1)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict(self, x: Tensor, edge_index: Tensor, *, threshold: float = 0.5) -> Tensor:
        """Return hard illicit/licit predictions at a probability ``threshold``.

        The threshold is explicit rather than an implicit argmax: in AML the
        operating point is a business decision about how many false positives
        an investigation team can absorb, not a property of the model.
        """
        probabilities = self.predict_proba(x, edge_index)
        return (probabilities[:, 1] >= threshold).long()

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count parameters, by default only those requiring gradients."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def describe(self) -> str:
        """Return a one-line architecture summary for logs and the UI."""
        return (
            f"{self.__class__.__name__}(in={self.in_channels}, "
            f"hidden={self.config.hidden_channels}, layers={self.config.num_layers}, "
            f"embed={self.embedding_dim}, out={self.out_channels}, "
            f"params={self.num_parameters():,})"
        )


class ClassifierHead(nn.Module):
    """Two-layer MLP mapping node embeddings to class logits."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return logits for each row of ``x``."""
        logits: Tensor = self.net(x)
        return logits


class ResidualProjection(nn.Module):
    """Identity or learned projection used to align residual-connection widths."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection: nn.Linear | None = (
            None
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x`` reshaped to the residual target width."""
        return x if self.projection is None else self.projection(x)
