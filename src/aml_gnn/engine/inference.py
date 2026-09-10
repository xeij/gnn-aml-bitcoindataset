"""Serving layer: load a checkpoint and answer investigation-shaped questions.

The Gradio app talks to this class and nothing else.  Keeping serving logic
out of the UI means the same code path can be driven from a notebook, a batch
job or a REST handler without change, and the UI cannot accidentally
reimplement scoring differently from training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph

from ..config import ExperimentConfig
from ..data.elliptic import EllipticDataModule
from ..data.splits import CLASS_NAMES, ILLICIT_LABEL, UNKNOWN_LABEL
from ..logging_utils import get_logger
from ..models.base import BaseNodeClassifier
from ..models.gnn import GAT
from ..models.registry import build_model
from .metrics import ClassificationMetrics, masked_evaluate
from .trainer import resolve_device

logger = get_logger(__name__)

#: Hard ceiling on nodes returned by a neighbourhood query, so one hub
#: transaction cannot lock up the UI with a 40k-node subgraph.
MAX_SUBGRAPH_NODES: int = 220


@dataclass(frozen=True)
class NodeAssessment:
    """Model verdict on a single transaction."""

    node_index: int
    tx_id: int
    time_step: int
    illicit_probability: float
    predicted_label: int
    true_label: int
    threshold: float
    degree: int

    @property
    def predicted_name(self) -> str:
        """Human-readable predicted class."""
        return CLASS_NAMES[self.predicted_label]

    @property
    def true_name(self) -> str:
        """Human-readable ground-truth class, or ``unknown``."""
        return "unknown" if self.true_label == UNKNOWN_LABEL else CLASS_NAMES[self.true_label]

    @property
    def is_correct(self) -> Optional[bool]:
        """Whether the prediction matches ground truth, or ``None`` if unlabelled."""
        if self.true_label == UNKNOWN_LABEL:
            return None
        return self.predicted_label == self.true_label

    @property
    def risk_band(self) -> str:
        """Coarse triage band an analyst can sort a queue by."""
        probability = self.illicit_probability
        if probability >= 0.90:
            return "critical"
        if probability >= self.threshold:
            return "elevated"
        if probability >= self.threshold / 2:
            return "watch"
        return "low"


@dataclass(frozen=True)
class SubgraphExplanation:
    """A k-hop neighbourhood scored and annotated for display."""

    center: NodeAssessment
    node_indices: Tensor
    edge_index: Tensor
    illicit_probability: Tensor
    labels: Tensor
    time_steps: Tensor
    tx_ids: Tensor
    edge_importance: Optional[Tensor] = None
    truncated: bool = False
    feature_attribution: Mapping[int, float] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the returned neighbourhood."""
        return int(self.node_indices.numel())

    @property
    def num_edges(self) -> int:
        """Number of edges in the returned neighbourhood."""
        return int(self.edge_index.size(1))

    @property
    def num_flagged(self) -> int:
        """Neighbours the model scores above the centre node's threshold."""
        return int((self.illicit_probability >= self.center.threshold).sum())


class InferenceService:
    """Scores the Elliptic graph with a trained checkpoint.

    Full-graph probabilities are computed once on construction and cached;
    every subsequent query is a tensor lookup, which keeps the UI responsive.
    """

    def __init__(
        self,
        model: BaseNodeClassifier,
        data_module: EllipticDataModule,
        *,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
        config: Optional[ExperimentConfig] = None,
    ) -> None:
        self.device = device or resolve_device("auto")
        self.model = model.to(self.device).eval()
        self.data_module = data_module
        self.data = data_module.data.to(self.device)
        self.threshold = float(threshold)
        self.config = config
        self._probabilities: Optional[Tensor] = None
        self._embeddings: Optional[Tensor] = None
        self._degrees: Optional[Tensor] = None
        logger.info(
            "InferenceService ready: %s on %s at threshold %.4f",
            self.model.__class__.__name__, self.device, self.threshold,
        )

    # ------------------------------------------------------- construction

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        data_module: Optional[EllipticDataModule] = None,
        device: Optional[torch.device] = None,
    ) -> "InferenceService":
        """Rebuild a service from a checkpoint written by :class:`ModelCheckpoint`.

        The checkpoint carries its own architecture config, so the caller does
        not need to remember which model produced it.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {path}. Train one first: "
                f"`python scripts/train.py --config configs/gat.yaml`"
            )
        device = device or resolve_device("auto")
        state = torch.load(path, map_location=device, weights_only=False)
        config = ExperimentConfig.from_dict(state["config"])
        logger.info("Loading checkpoint %s (epoch %s)", path.name, state.get("epoch"))

        module = data_module or EllipticDataModule(config.data)
        module.prepare()

        model = build_model(
            config.model,
            in_channels=int(state["in_channels"]),
            out_channels=int(state["out_channels"]),
        )
        model.load_state_dict(state["model_state_dict"])
        return cls(
            model,
            module,
            threshold=float(state.get("threshold", 0.5)),
            device=device,
            config=config,
        )

    # ------------------------------------------------------------ scoring

    @property
    def probabilities(self) -> Tensor:
        """Cached ``P(illicit)`` for every node in the graph."""
        if self._probabilities is None:
            with torch.no_grad():
                logits = self.model(self.data.x, self.data.edge_index)
                self._probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu()
            logger.info("Scored all %s nodes", f"{self._probabilities.numel():,}")
        return self._probabilities

    @property
    def embeddings(self) -> Tensor:
        """Cached encoder output for every node, for projection plots."""
        if self._embeddings is None:
            with torch.no_grad():
                self._embeddings = self.model.encode(self.data.x, self.data.edge_index).cpu()
        return self._embeddings

    @property
    def degrees(self) -> Tensor:
        """Cached undirected degree of every node."""
        if self._degrees is None:
            self._degrees = self.data_module.degrees().cpu()
        return self._degrees

    def set_threshold(self, threshold: float) -> None:
        """Change the operating point without rescoring the graph."""
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold must lie in (0, 1), got {threshold}")
        self.threshold = float(threshold)
        logger.info("Operating threshold set to %.4f", threshold)

    def assess(self, node_index: int, *, threshold: Optional[float] = None) -> NodeAssessment:
        """Return the model's verdict on one node index."""
        if not 0 <= node_index < self.data.num_nodes:
            raise IndexError(
                f"Node index {node_index} out of range [0, {self.data.num_nodes})"
            )
        threshold = self.threshold if threshold is None else threshold
        probability = float(self.probabilities[node_index])
        return NodeAssessment(
            node_index=int(node_index),
            tx_id=int(self.data.tx_id[node_index]),
            time_step=int(self.data.time_step[node_index]),
            illicit_probability=probability,
            predicted_label=int(probability >= threshold),
            true_label=int(self.data.y[node_index]),
            threshold=threshold,
            degree=int(self.degrees[node_index]),
        )

    def assess_tx_id(self, tx_id: int, **kwargs: Any) -> NodeAssessment:
        """Return the verdict on an original Elliptic transaction id."""
        index = self.data_module.index_for_tx_id(tx_id)
        if index is None:
            raise KeyError(f"Transaction id {tx_id} is not present in the Elliptic dataset")
        return self.assess(index, **kwargs)

    def evaluate_split(self, split: str, *, threshold: Optional[float] = None
                       ) -> ClassificationMetrics:
        """Score one named split at the current (or given) threshold."""
        mask = getattr(self.data, f"{split}_mask", None)
        if mask is None:
            raise ValueError(f"Unknown split {split!r}")
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
        return masked_evaluate(
            logits.cpu(),
            self.data.y.cpu(),
            mask.cpu(),
            split=split,
            threshold=self.threshold if threshold is None else threshold,
            time_step=self.data.time_step.cpu(),
        )

    def top_alerts(
        self,
        *,
        split: Optional[str] = "test",
        limit: int = 25,
        include_unlabelled: bool = True,
        min_probability: float = 0.0,
    ) -> list[NodeAssessment]:
        """Return the highest-risk transactions, ranked by illicit probability.

        This is the product surface of the whole pipeline: an investigator
        works a queue, so ranking quality matters more than any single
        thresholded decision.
        """
        probabilities = self.probabilities
        candidate = torch.ones(self.data.num_nodes, dtype=torch.bool)
        if split is not None:
            time_step = self.data.time_step.cpu()
            bounds = self._split_bounds(split)
            candidate &= (time_step >= bounds[0]) & (time_step <= bounds[1])
        if not include_unlabelled:
            candidate &= (self.data.y.cpu() != UNKNOWN_LABEL)
        candidate &= probabilities >= min_probability

        indices = candidate.nonzero(as_tuple=False).view(-1)
        if indices.numel() == 0:
            logger.warning("No candidates matched the alert query")
            return []
        ranked = indices[torch.argsort(probabilities[indices], descending=True)][:limit]
        return [self.assess(int(index)) for index in ranked]

    def _split_bounds(self, split: str) -> tuple[int, int]:
        """Return the configured time-step window for a split name."""
        data_config = (self.config or ExperimentConfig()).data
        bounds = {
            "train": data_config.train_time_steps,
            "val": data_config.val_time_steps,
            "test": data_config.test_time_steps,
        }
        if split not in bounds:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(bounds)}")
        return bounds[split]

    # ------------------------------------------------------- explanations

    def explain(
        self,
        node_index: int,
        *,
        num_hops: int = 2,
        max_nodes: int = MAX_SUBGRAPH_NODES,
        with_attention: bool = True,
        with_attribution: bool = True,
    ) -> SubgraphExplanation:
        """Extract and score the k-hop neighbourhood around one transaction.

        Args:
            node_index: Centre of the neighbourhood.
            num_hops: Hops to expand.  Matching the model's layer count shows
                exactly the receptive field that produced the prediction.
            max_nodes: Truncation ceiling; the closest neighbours are kept.
            with_attention: Read GAT attention coefficients as edge importance.
            with_attribution: Compute input-gradient feature attribution.

        Returns:
            A :class:`SubgraphExplanation` with node-level scores and, where
            available, edge importance and per-feature attribution.
        """
        center = self.assess(node_index)
        subset, edge_index, _, _ = k_hop_subgraph(
            node_idx=int(node_index),
            num_hops=num_hops,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes,
        )
        truncated = False
        if subset.numel() > max_nodes:
            subset, edge_index, truncated = self._truncate(node_index, num_hops, max_nodes)

        probabilities = self.probabilities[subset.cpu()]
        edge_importance: Optional[Tensor] = None
        if with_attention and isinstance(self.model, GAT):
            edge_importance = self._attention_importance(subset, edge_index)

        attribution: Mapping[int, float] = {}
        if with_attribution:
            attribution = self._feature_attribution(subset, edge_index, int(node_index))

        return SubgraphExplanation(
            center=center,
            node_indices=subset.cpu(),
            edge_index=edge_index.cpu(),
            illicit_probability=probabilities,
            labels=self.data.y[subset].cpu(),
            time_steps=self.data.time_step[subset].cpu(),
            tx_ids=self.data.tx_id[subset].cpu(),
            edge_importance=edge_importance,
            truncated=truncated,
            feature_attribution=attribution,
        )

    def _truncate(
        self, node_index: int, num_hops: int, max_nodes: int
    ) -> tuple[Tensor, Tensor, bool]:
        """Grow the neighbourhood hop by hop, stopping before ``max_nodes``.

        Truncating by hop distance rather than at random keeps the returned
        subgraph a genuine neighbourhood: the nodes shown are the ones nearest
        the transaction under investigation.
        """
        kept: Tensor = torch.tensor([node_index], device=self.data.edge_index.device)
        for hop in range(1, num_hops + 1):
            candidate, _, _, _ = k_hop_subgraph(
                node_idx=int(node_index),
                num_hops=hop,
                edge_index=self.data.edge_index,
                relabel_nodes=False,
                num_nodes=self.data.num_nodes,
            )
            if candidate.numel() > max_nodes:
                # Keep the previous complete hop, then fill up with this hop's
                # nodes ordered by risk so the most relevant ones survive.
                extras = candidate[~torch.isin(candidate, kept)]
                room = max_nodes - int(kept.numel())
                if room > 0:
                    ranked = extras[
                        torch.argsort(self.probabilities[extras.cpu()], descending=True)
                    ][:room]
                    kept = torch.cat([kept, ranked.to(kept.device)])
                logger.info(
                    "Neighbourhood of node %d truncated to %d nodes at hop %d",
                    node_index, int(kept.numel()), hop,
                )
                break
            kept = candidate
        subset, edge_index, _, _ = k_hop_subgraph(
            node_idx=kept.tolist(),
            num_hops=0,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes,
        )
        return subset, edge_index, True

    def _attention_importance(self, subset: Tensor, edge_index: Tensor) -> Optional[Tensor]:
        """Return mean attention per subgraph edge, aligned to ``edge_index``."""
        model = self.model
        if not isinstance(model, GAT):
            return None
        try:
            x = self.data.x[subset]
            attended_edges, alpha = model.attention_weights(x, edge_index.to(self.device))
            # attention_weights inserts self-loops, so map back by edge identity.
            lookup: dict[tuple[int, int], float] = {}
            for (source, target), weight in zip(attended_edges.t().tolist(), alpha.tolist()):
                lookup[(source, target)] = weight
            values = [
                lookup.get((int(source), int(target)), 0.0)
                for source, target in edge_index.t().tolist()
            ]
            return torch.tensor(values, dtype=torch.float)
        except Exception:  # pragma: no cover - explanation is never load-bearing
            logger.exception("Attention extraction failed; continuing without edge importance")
            return None

    def _feature_attribution(
        self, subset: Tensor, edge_index: Tensor, node_index: int, *, top_k: int = 12
    ) -> dict[int, float]:
        """Input-gradient attribution for the centre node's own features.

        This is a saliency map, not a Shapley value: it says which input
        dimensions the illicit logit is locally most sensitive to, which is
        enough to point an analyst at the right part of the feature vector.
        """
        try:
            local_index = int((subset == node_index).nonzero(as_tuple=False).view(-1)[0])
            x = self.data.x[subset].clone().detach().requires_grad_(True)
            logits = self.model(x, edge_index.to(self.device))
            logits[local_index, ILLICIT_LABEL].backward()
            if x.grad is None:
                return {}
            saliency = (x.grad[local_index] * x[local_index]).detach().abs().cpu()
            top = torch.topk(saliency, k=min(top_k, saliency.numel()))
            return {int(i): float(v) for i, v in zip(top.indices, top.values)}
        except Exception:  # pragma: no cover - explanation is never load-bearing
            logger.exception("Feature attribution failed; continuing without it")
            return {}

    # ------------------------------------------------------------ helpers

    def risk_distribution(self, *, bins: int = 40) -> tuple[list[float], list[int]]:
        """Return a histogram of illicit probabilities across the whole graph."""
        histogram = torch.histc(self.probabilities, bins=bins, min=0.0, max=1.0)
        edges = torch.linspace(0.0, 1.0, bins + 1)[:-1]
        return edges.tolist(), [int(v) for v in histogram]

    def temporal_alert_volume(self, *, threshold: Optional[float] = None
                              ) -> tuple[list[int], list[int], list[int]]:
        """Return per-time-step ``(steps, alerts, true_illicit)`` counts."""
        threshold = self.threshold if threshold is None else threshold
        steps_tensor = self.data.time_step.cpu()
        flagged = self.probabilities >= threshold
        illicit = self.data.y.cpu() == ILLICIT_LABEL
        steps = sorted({int(s) for s in steps_tensor.tolist()})
        alerts = [int((flagged & (steps_tensor == step)).sum()) for step in steps]
        actual = [int((illicit & (steps_tensor == step)).sum()) for step in steps]
        return steps, alerts, actual

    def describe(self) -> Sequence[tuple[str, str]]:
        """Return key/value rows describing the loaded model, for the UI header."""
        config = self.config
        return [
            ("Architecture", self.model.__class__.__name__),
            ("Parameters", f"{self.model.num_parameters():,}"),
            ("Embedding width", str(self.model.embedding_dim)),
            ("Device", str(self.device)),
            ("Operating threshold", f"{self.threshold:.4f}"),
            ("Experiment", config.name if config else "unknown"),
        ]
