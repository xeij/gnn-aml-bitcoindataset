"""Feature-space preprocessing applied after loading, before training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

from ..logging_utils import get_logger

logger = get_logger(__name__)

#: The Elliptic feature block is documented as 93 local transaction features
#: followed by 72 features aggregated over one-hop neighbours.
NUM_LOCAL_FEATURES: Final[int] = 93
NUM_AGGREGATE_FEATURES: Final[int] = 72
NUM_TOTAL_FEATURES: Final[int] = NUM_LOCAL_FEATURES + NUM_AGGREGATE_FEATURES


@dataclass
class StandardScaler:
    """Column-wise standardiser whose statistics are fitted on one node subset.

    Fitting on the training mask alone is not a formality: the test window sits
    strictly in the future, and normalising with global statistics would leak
    the distribution of transactions the model is meant to have never seen.
    """

    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, x: Tensor, mask: Tensor, *, eps: float = 1e-8) -> StandardScaler:
        """Compute per-column mean and standard deviation over ``mask`` rows."""
        if int(mask.sum()) < 2:
            raise ValueError("StandardScaler.fit needs at least two rows")
        subset = x[mask]
        mean = subset.mean(dim=0)
        std = subset.std(dim=0).clamp_min(eps)
        degenerate = int((subset.std(dim=0) < eps).sum())
        if degenerate:
            logger.warning(
                "%d/%d feature columns are constant on the fitting subset; "
                "their scale is clamped to eps.", degenerate, x.size(1),
            )
        return cls(mean=mean, std=std)

    def transform(self, x: Tensor) -> Tensor:
        """Apply the fitted standardisation to every row of ``x``."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def state_dict(self) -> dict[str, Tensor]:
        """Return a checkpointable view of the fitted statistics."""
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> StandardScaler:
        """Rebuild a scaler saved by :meth:`state_dict`."""
        return cls(mean=state["mean"], std=state["std"])


def select_feature_block(x: Tensor, *, local_only: bool) -> Tensor:
    """Optionally discard the 72 neighbour-aggregated columns.

    Keeping only local features makes the ablation honest: whatever accuracy
    remains has to come from message passing rather than from hand-built
    neighbourhood summaries already baked into the input.
    """
    if not local_only:
        return x
    if x.size(1) != NUM_TOTAL_FEATURES:
        raise ValueError(
            f"Expected {NUM_TOTAL_FEATURES} feature columns to slice a local block, "
            f"found {x.size(1)}"
        )
    logger.info("Dropping %d aggregated features (local-only ablation)", NUM_AGGREGATE_FEATURES)
    return x[:, :NUM_LOCAL_FEATURES].contiguous()


def sanitise_features(x: Tensor) -> Tensor:
    """Replace non-finite entries with zeros, warning if any were present."""
    non_finite = ~torch.isfinite(x)
    count = int(non_finite.sum())
    if count:
        logger.warning("Replacing %d non-finite feature values with 0.0", count)
        x = torch.where(non_finite, torch.zeros_like(x), x)
    return x
