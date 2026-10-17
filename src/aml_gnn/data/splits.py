"""Temporal split construction for the Elliptic transaction graph.

The graph is a single connected artefact observed over 49 discrete time steps
(roughly two-week windows).  Nodes are only ever *masked*, never removed, so
training remains transductive: message passing sees the full topology while
the loss is computed on training-window nodes alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from ..logging_utils import get_logger

logger = get_logger(__name__)

#: Label assigned by the upstream dataset to transactions with no ground truth.
UNKNOWN_LABEL: int = 2
LICIT_LABEL: int = 0
ILLICIT_LABEL: int = 1

CLASS_NAMES: tuple[str, str] = ("licit", "illicit")


@dataclass(frozen=True)
class SplitStatistics:
    """Per-split node counts used for logging and for the UI dataset panel."""

    name: str
    time_steps: tuple[int, int]
    num_nodes: int
    num_licit: int
    num_illicit: int

    @property
    def illicit_ratio(self) -> float:
        """Share of labelled nodes in this split that are illicit."""
        return self.num_illicit / self.num_nodes if self.num_nodes else 0.0

    def __str__(self) -> str:
        """Return a single-line summary for console logs."""
        return (
            f"{self.name:<5} t∈[{self.time_steps[0]:>2},{self.time_steps[1]:>2}] "
            f"n={self.num_nodes:>6,} licit={self.num_licit:>6,} "
            f"illicit={self.num_illicit:>5,} ({self.illicit_ratio:.2%})"
        )


def temporal_mask(
    time_step: Tensor,
    y: Tensor,
    bounds: tuple[int, int],
    *,
    labelled_only: bool = True,
) -> Tensor:
    """Build a boolean node mask for an inclusive time-step window.

    Args:
        time_step: Long tensor of shape ``[num_nodes]`` holding 1-indexed steps.
        y: Long tensor of shape ``[num_nodes]`` with 0/1/2 class codes.
        bounds: Inclusive ``(first, last)`` time-step window.
        labelled_only: Exclude ``unknown`` nodes.  Supervised losses and metrics
            require this; a mask used purely for visualisation may not.

    Returns:
        Boolean tensor of shape ``[num_nodes]``.
    """
    first, last = bounds
    mask = (time_step >= first) & (time_step <= last)
    if labelled_only:
        mask &= y != UNKNOWN_LABEL
    return mask


def summarise_split(name: str, bounds: tuple[int, int], y: Tensor, mask: Tensor) -> SplitStatistics:
    """Count labelled nodes of each class inside ``mask``."""
    labels = y[mask]
    return SplitStatistics(
        name=name,
        time_steps=bounds,
        num_nodes=int(mask.sum()),
        num_licit=int((labels == LICIT_LABEL).sum()),
        num_illicit=int((labels == ILLICIT_LABEL).sum()),
    )


def assert_disjoint(masks: Iterable[Tensor]) -> None:
    """Raise if any node appears in more than one split mask.

    A silent overlap here is the single most damaging bug this pipeline could
    ship, so it is checked at every construction rather than trusted.
    """
    stacked = torch.stack(list(masks)).int().sum(dim=0)
    overlap = int((stacked > 1).sum())
    if overlap:
        raise ValueError(
            f"Split masks overlap on {overlap} nodes; temporal evaluation would leak."
        )


def class_weights(y: Tensor, mask: Tensor, *, illicit_multiplier: float = 1.0) -> Tensor:
    """Inverse-frequency class weights computed on a single split.

    With ~2% illicit nodes, an unweighted cross-entropy converges to the
    majority class and reports 98% accuracy while catching nothing.  Weights
    are derived from the *training* mask only, so no test statistics leak into
    the objective.

    Args:
        y: Class codes for every node.
        mask: Nodes to derive frequencies from (normally the train mask).
        illicit_multiplier: Extra gain on the illicit weight.  Above 1.0 buys
            recall at the cost of precision.

    Returns:
        Float tensor ``[w_licit, w_illicit]`` normalised to mean 1.
    """
    labels = y[mask]
    counts = torch.bincount(labels, minlength=2)[:2].float()
    if (counts == 0).any():
        raise ValueError(f"Cannot derive class weights: a class is absent ({counts.tolist()})")
    weights = counts.sum() / (2.0 * counts)
    weights[ILLICIT_LABEL] *= illicit_multiplier
    weights = weights / weights.mean()
    logger.debug(
        "Class weights from %d labelled nodes: licit=%.3f illicit=%.3f",
        int(mask.sum()), weights[LICIT_LABEL], weights[ILLICIT_LABEL],
    )
    return weights
