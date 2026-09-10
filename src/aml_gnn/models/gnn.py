"""Message-passing architectures for illicit-transaction detection.

All three graph models share one template (:class:`MessagePassingClassifier`)
and differ only in the convolution they instantiate.  The template owns the
parts that are easy to get subtly wrong -- residual width alignment,
normalisation placement, jumping-knowledge concatenation -- so a new
architecture is a five-line subclass rather than a copied training stack.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, GCNConv, MessagePassing, SAGEConv

from ..config import ModelConfig
from ..logging_utils import get_logger
from .base import BaseNodeClassifier, ClassifierHead, ResidualProjection
from .registry import register_model

logger = get_logger(__name__)


class MessagePassingClassifier(BaseNodeClassifier):
    """Stacked message-passing encoder with an MLP classification head.

    Each block applies ``conv -> norm -> ReLU -> dropout``, optionally wrapped
    in a residual connection.  With jumping knowledge enabled the outputs of
    every block are concatenated before classification, which lets the head
    weigh 1-hop evidence against k-hop evidence directly.  That matters here:
    some illicit transactions are identifiable from their own features, while
    others are only suspicious because of what they are three hops away from.
    """

    def __init__(self, in_channels: int, out_channels: int, config: ModelConfig) -> None:
        super().__init__(in_channels, out_channels, config)
        if config.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {config.num_layers}")

        hidden = config.hidden_channels
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        widths: Sequence[int] = [in_channels] + [hidden] * config.num_layers
        for layer in range(config.num_layers):
            self.convs.append(self._build_conv(widths[layer], hidden, layer=layer))
            self.norms.append(
                nn.BatchNorm1d(hidden) if config.batch_norm else nn.Identity()
            )
            self.residuals.append(
                ResidualProjection(widths[layer], hidden)
                if config.residual
                else nn.Identity()
            )

        self._embedding_dim = hidden * config.num_layers if config.jumping_knowledge else hidden
        self.head = ClassifierHead(
            in_channels=self._embedding_dim,
            hidden_channels=hidden,
            out_channels=out_channels,
            dropout=config.dropout,
        )
        self.reset_parameters()

    # ------------------------------------------------------- template hooks

    @abstractmethod
    def _build_conv(self, in_channels: int, out_channels: int, *, layer: int) -> MessagePassing:
        """Return the convolution for one block, mapping ``in`` to ``out`` channels."""

    # --------------------------------------------------------------- module

    @property
    def embedding_dim(self) -> int:
        """Width of the encoder output, accounting for jumping knowledge."""
        return self._embedding_dim

    def reset_parameters(self) -> None:
        """Re-initialise every submodule that exposes a reset hook."""
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Run the message-passing stack and return node embeddings."""
        config = self.config
        layer_outputs: list[Tensor] = []
        hidden = x
        for conv, norm, residual in zip(self.convs, self.norms, self.residuals):
            identity = hidden
            hidden = conv(hidden, edge_index)
            hidden = norm(hidden)
            hidden = F.relu(hidden, inplace=True)
            if config.residual:
                hidden = hidden + residual(identity)
            hidden = F.dropout(hidden, p=config.dropout, training=self.training)
            layer_outputs.append(hidden)
        return torch.cat(layer_outputs, dim=-1) if config.jumping_knowledge else hidden

    def _classify(self, embedding: Tensor) -> Tensor:
        """Map encoder output to class logits."""
        return self.head(embedding)


@register_model("gcn")
class GCN(MessagePassingClassifier):
    """Kipf & Welling graph convolution: symmetric-normalised mean aggregation.

    The cheapest of the three and a strong baseline, but it weighs every
    neighbour by degree alone, so a single illicit counterparty in a
    high-degree neighbourhood gets averaged away.
    """

    def _build_conv(self, in_channels: int, out_channels: int, *, layer: int) -> MessagePassing:
        """Return a :class:`GCNConv` with self-loops and cached normalisation off.

        Caching is disabled deliberately: it is only valid for a fixed graph,
        and the app supports scoring induced subgraphs.
        """
        return GCNConv(in_channels, out_channels, add_self_loops=True, cached=False)


@register_model("graphsage")
class GraphSAGE(MessagePassingClassifier):
    """Inductive SAGE convolution with a separate root-node transform.

    Keeping the node's own representation on its own weight matrix means local
    transaction features survive aggregation instead of being blended into the
    neighbourhood mean.
    """

    def _build_conv(self, in_channels: int, out_channels: int, *, layer: int) -> MessagePassing:
        """Return a mean-aggregating :class:`SAGEConv`."""
        return SAGEConv(in_channels, out_channels, aggr="mean", root_weight=True)


@register_model("gat")
class GAT(MessagePassingClassifier):
    """GATv2 attention: learned, asymmetric neighbour weighting.

    This is the default architecture. Attention is the right inductive bias
    for AML because laundering structure is about *which* counterparty a
    transaction touches, not how many; and the learned coefficients double as
    the per-edge attribution the UI renders during an investigation.
    """

    def __init__(self, in_channels: int, out_channels: int, config: ModelConfig) -> None:
        if config.hidden_channels % config.heads != 0:
            raise ValueError(
                f"hidden_channels ({config.hidden_channels}) must be divisible by "
                f"heads ({config.heads}) so concatenated heads keep the declared width"
            )
        super().__init__(in_channels, out_channels, config)

    def _build_conv(self, in_channels: int, out_channels: int, *, layer: int) -> MessagePassing:
        """Return a multi-head :class:`GATv2Conv` whose heads concatenate to ``out_channels``."""
        heads = self.config.heads
        return GATv2Conv(
            in_channels,
            out_channels // heads,
            heads=heads,
            concat=True,
            dropout=self.config.dropout,
            add_self_loops=True,
        )

    @torch.no_grad()
    def attention_weights(self, x: Tensor, edge_index: Tensor, *, layer: int = -1
                          ) -> tuple[Tensor, Tensor]:
        """Return ``(edge_index_with_self_loops, alpha)`` from one attention layer.

        Args:
            x: Node features for the (sub)graph being explained.
            edge_index: Its connectivity.
            layer: Index of the attention layer to read; ``-1`` is the last.

        Returns:
            The edge index the layer actually attended over -- which includes
            the self-loops it inserts -- and the mean attention coefficient
            across heads for each of those edges.
        """
        was_training = self.training
        self.eval()
        try:
            hidden = x
            target = range(len(self.convs))[layer]
            for index, (conv, norm, residual) in enumerate(
                zip(self.convs, self.norms, self.residuals)
            ):
                if index == target:
                    _, (attended_edges, alpha) = conv(
                        hidden, edge_index, return_attention_weights=True
                    )
                    return attended_edges, alpha.mean(dim=-1)
                identity = hidden
                hidden = F.relu(norm(conv(hidden, edge_index)), inplace=True)
                if self.config.residual:
                    hidden = hidden + residual(identity)
        finally:
            self.train(was_training)
        raise RuntimeError(f"Attention layer {layer} was never reached")
