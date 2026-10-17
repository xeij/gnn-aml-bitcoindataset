"""Loading and preparation of the official Elliptic Bitcoin dataset.

The upstream :class:`torch_geometric.datasets.EllipticBitcoinDataset` gives a
:class:`~torch_geometric.data.Data` object with ``x``, ``edge_index``, ``y``
and a hard-coded train/test mask pair.  Two things it deliberately does not
expose are needed downstream:

* the **time step** of each transaction, which the upstream class reads from
  the raw CSV to build its masks and then discards.  Without it there is no
  validation window and no way to plot performance drift over time;
* the original **transaction id**, which is what an analyst types into the UI.

:class:`EllipticDataModule` recovers both from the raw CSV, caches them next
to the processed tensors, and rebuilds the splits as a three-way temporal
partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.utils import add_remaining_self_loops, degree, to_undirected

from ..config import DataConfig, time_step_bounds
from ..logging_utils import get_logger
from .splits import (
    CLASS_NAMES,
    ILLICIT_LABEL,
    LICIT_LABEL,
    UNKNOWN_LABEL,
    SplitStatistics,
    assert_disjoint,
    summarise_split,
    temporal_mask,
)
from .transforms import StandardScaler, sanitise_features, select_feature_block

logger = get_logger(__name__)

#: Published dataset dimensions; asserted on load so a corrupt or partial
#: download fails here rather than as a confusing shape error during training.
EXPECTED_NUM_NODES: Final[int] = 203_769
EXPECTED_NUM_EDGES: Final[int] = 234_355
EXPECTED_NUM_FEATURES: Final[int] = 165
NUM_TIME_STEPS: Final[int] = 49

_METADATA_CACHE_NAME: Final[str] = "node_metadata.pt"


@dataclass(frozen=True)
class GraphMetadata:
    """Immutable description of a prepared graph, safe to hand to the UI."""

    num_nodes: int
    num_edges: int
    num_features: int
    num_classes: int
    num_time_steps: int
    splits: tuple[SplitStatistics, ...]
    class_names: tuple[str, ...] = CLASS_NAMES

    def as_rows(self) -> list[list[str]]:
        """Return a table-shaped view for display in the Gradio dataset panel."""
        rows = [
            ["Nodes (transactions)", f"{self.num_nodes:,}"],
            ["Edges (payment flows)", f"{self.num_edges:,}"],
            ["Node features", str(self.num_features)],
            ["Time steps", str(self.num_time_steps)],
        ]
        for split in self.splits:
            rows.append(
                [
                    f"{split.name.capitalize()} window "
                    f"(t{split.time_steps[0]}–t{split.time_steps[1]})",
                    f"{split.num_nodes:,} labelled · {split.num_illicit:,} illicit "
                    f"({split.illicit_ratio:.2%})",
                ]
            )
        return rows


class EllipticDataModule:
    """Prepares the Elliptic graph for transductive node classification.

    The module is idempotent and cheap to re-enter: :meth:`prepare` performs
    the download/processing once and caches the resulting :class:`Data` on the
    instance.

    Example:
        >>> module = EllipticDataModule(DataConfig())
        >>> data = module.prepare()
        >>> data.train_mask.sum() > 0
        True
    """

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        self.config: DataConfig = config or DataConfig()
        self._data: Optional[Data] = None
        self._scaler: Optional[StandardScaler] = None
        self._metadata: Optional[GraphMetadata] = None
        self._tx_id_to_index: Optional[dict[int, int]] = None

    # ------------------------------------------------------------------ API

    @property
    def data(self) -> Data:
        """The prepared graph, preparing it on first access."""
        if self._data is None:
            return self.prepare()
        return self._data

    @property
    def metadata(self) -> GraphMetadata:
        """Summary statistics for the prepared graph."""
        if self._metadata is None:
            self.prepare()
        assert self._metadata is not None
        return self._metadata

    @property
    def scaler(self) -> Optional[StandardScaler]:
        """The feature scaler fitted on the training window, if standardising."""
        if self._data is None:
            self.prepare()
        return self._scaler

    @property
    def num_features(self) -> int:
        """Input dimensionality after any feature-block selection."""
        return int(self.data.num_node_features)

    @property
    def num_classes(self) -> int:
        """Number of supervised classes (licit / illicit)."""
        return 2

    def prepare(self) -> Data:
        """Download, validate, transform and split the graph.

        Returns:
            A :class:`Data` object carrying ``x``, ``edge_index``, ``y``,
            ``time_step``, ``tx_id`` and disjoint ``train``/``val``/``test``
            boolean masks over labelled nodes.
        """
        if self._data is not None:
            return self._data

        cfg = self.config
        root = Path(cfg.root)
        logger.info("Loading Elliptic Bitcoin dataset from %s", root.resolve())
        dataset = EllipticBitcoinDataset(root=str(root), force_reload=cfg.force_reload)
        data: Data = dataset[0]
        self._validate_raw(data)

        tx_id, time_step = self._load_node_metadata(root, num_nodes=data.num_nodes)

        x = sanitise_features(data.x.float())
        x = select_feature_block(x, local_only=cfg.local_features_only)
        y = data.y.long()

        edge_index = data.edge_index
        if cfg.to_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
            logger.info(
                "Symmetrised edge index: %d directed → %d undirected entries",
                data.edge_index.size(1), edge_index.size(1),
            )
        if cfg.add_self_loops:
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=data.num_nodes)

        masks = self._build_masks(time_step=time_step, y=y)

        if cfg.standardise:
            self._scaler = StandardScaler.fit(x, masks["train_mask"])
            x = self._scaler.transform(x)
            logger.info("Standardised %d features on the training window only", x.size(1))

        prepared = Data(x=x, edge_index=edge_index, y=y)
        prepared.time_step = time_step
        prepared.tx_id = tx_id
        prepared.labelled_mask = y != UNKNOWN_LABEL
        for name, mask in masks.items():
            setattr(prepared, name, mask)

        self._data = prepared
        self._metadata = self._build_metadata(prepared)
        self._log_summary()
        return prepared

    # ------------------------------------------------------- lookup helpers

    def index_for_tx_id(self, tx_id: int) -> Optional[int]:
        """Map an original Elliptic transaction id to its node index.

        Returns ``None`` when the id is absent from the dataset.
        """
        if self._tx_id_to_index is None:
            ids = self.data.tx_id.tolist()
            self._tx_id_to_index = {int(v): i for i, v in enumerate(ids)}
            logger.debug("Built transaction-id lookup over %d nodes", len(self._tx_id_to_index))
        return self._tx_id_to_index.get(int(tx_id))

    def sample_node_indices(
        self,
        *,
        label: Optional[int] = None,
        split: Optional[str] = None,
        limit: int = 25,
        generator: Optional[torch.Generator] = None,
    ) -> list[int]:
        """Return up to ``limit`` node indices matching a label and/or split.

        Used by the UI to populate its "show me an example" controls without
        the caller needing to know how masks are stored.
        """
        data = self.data
        # The graph may have been moved to an accelerator by a serving layer,
        # so build the mask on whatever device its tensors currently live on.
        device = data.y.device
        mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
        if label is not None:
            mask &= data.y == label
        if split is not None:
            attr = f"{split}_mask"
            if not hasattr(data, attr):
                raise ValueError(f"Unknown split {split!r}; expected train/val/test")
            mask &= getattr(data, attr).to(device)
        candidates = mask.nonzero(as_tuple=False).view(-1).cpu()
        if candidates.numel() == 0:
            return []
        perm = torch.randperm(candidates.numel(), generator=generator)[:limit]
        return candidates[perm].tolist()

    def degrees(self) -> Tensor:
        """Undirected degree of every node, as a float tensor."""
        data = self.data
        node_degrees: Tensor = degree(data.edge_index[0], num_nodes=data.num_nodes)
        return node_degrees

    # ------------------------------------------------------------ internals

    def _validate_raw(self, data: Data) -> None:
        """Fail fast on a graph whose shape does not match the published benchmark."""
        problems: list[str] = []
        if data.num_nodes != EXPECTED_NUM_NODES:
            problems.append(f"nodes {data.num_nodes} != {EXPECTED_NUM_NODES}")
        if data.edge_index.size(1) != EXPECTED_NUM_EDGES:
            problems.append(f"edges {data.edge_index.size(1)} != {EXPECTED_NUM_EDGES}")
        if data.x.size(1) != EXPECTED_NUM_FEATURES:
            problems.append(f"features {data.x.size(1)} != {EXPECTED_NUM_FEATURES}")
        if problems:
            raise RuntimeError(
                "Elliptic dataset does not match the published benchmark ("
                + "; ".join(problems)
                + "). Re-download with force_reload=True."
            )
        logger.debug("Raw dataset shape validated against the published benchmark")

    def _load_node_metadata(self, root: Path, *, num_nodes: int) -> tuple[Tensor, Tensor]:
        """Return ``(tx_id, time_step)`` tensors, reading the raw CSV once and caching.

        Only columns 0 and 1 of the 690 MB features file are parsed, so the
        uncached path costs about a second.
        """
        cache_path = root / "processed" / _METADATA_CACHE_NAME
        if cache_path.exists() and not self.config.force_reload:
            payload = torch.load(cache_path, weights_only=True)
            logger.debug("Loaded cached node metadata from %s", cache_path)
            return payload["tx_id"], payload["time_step"]

        import pandas as pd  # Local import: pandas is only needed on a cache miss.

        raw_features = root / "raw" / "elliptic_txs_features.csv"
        if not raw_features.exists():
            raise FileNotFoundError(
                f"Raw feature file missing at {raw_features}. The processed cache alone "
                "is not enough to recover time steps; delete the processed directory "
                "and re-run to trigger a full download."
            )
        logger.info("Recovering transaction ids and time steps from %s", raw_features.name)
        frame = pd.read_csv(
            raw_features, header=None, usecols=[0, 1], names=["tx_id", "time_step"]
        )
        if len(frame) != num_nodes:
            raise RuntimeError(
                f"Raw feature rows ({len(frame)}) do not match graph nodes ({num_nodes}); "
                "the download is inconsistent."
            )
        self._verify_class_alignment(root, frame["tx_id"].to_numpy())

        tx_id = torch.as_tensor(frame["tx_id"].to_numpy().copy(), dtype=torch.long)
        time_step = torch.as_tensor(frame["time_step"].to_numpy().copy(), dtype=torch.long)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tx_id": tx_id, "time_step": time_step}, cache_path)
        logger.info("Cached node metadata to %s", cache_path)
        return tx_id, time_step

    @staticmethod
    def _verify_class_alignment(root: Path, feature_tx_ids: "np.ndarray") -> None:
        """Check the positional join between the feature and class CSVs.

        Upstream builds ``y`` by taking the class column in file order, which is
        only correct while both files enumerate transactions identically.  That
        holds for the published files; this asserts it instead of assuming it.
        """
        import pandas as pd

        class_path = root / "raw" / "elliptic_txs_classes.csv"
        if not class_path.exists():
            logger.warning("Class file absent at %s; skipping alignment check", class_path)
            return
        class_ids = pd.read_csv(class_path, usecols=["txId"])["txId"].to_numpy()
        if len(class_ids) != len(feature_tx_ids) or not (class_ids == feature_tx_ids).all():
            raise RuntimeError(
                "elliptic_txs_classes.csv is not row-aligned with "
                "elliptic_txs_features.csv, so upstream label assignment is invalid."
            )
        logger.debug("Feature/class row alignment verified across %d rows", len(class_ids))

    def _build_masks(self, *, time_step: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Construct disjoint train/val/test masks from the configured windows."""
        train_bounds, val_bounds, test_bounds = time_step_bounds(self.config)
        masks = {
            "train_mask": temporal_mask(time_step, y, train_bounds),
            "val_mask": temporal_mask(time_step, y, val_bounds),
            "test_mask": temporal_mask(time_step, y, test_bounds),
        }
        assert_disjoint(masks.values())
        for name, mask in masks.items():
            if int(mask.sum()) == 0:
                raise ValueError(f"{name} is empty; check the configured time-step windows")
            present = torch.unique(y[mask]).tolist()
            if ILLICIT_LABEL not in present or LICIT_LABEL not in present:
                raise ValueError(
                    f"{name} contains only classes {present}; both classes are required."
                )
        return masks

    def _build_metadata(self, data: Data) -> GraphMetadata:
        """Assemble the :class:`GraphMetadata` record for a prepared graph."""
        train_bounds, val_bounds, test_bounds = time_step_bounds(self.config)
        splits = tuple(
            summarise_split(name, bounds, data.y, getattr(data, f"{name}_mask"))
            for name, bounds in (
                ("train", train_bounds), ("val", val_bounds), ("test", test_bounds)
            )
        )
        return GraphMetadata(
            num_nodes=int(data.num_nodes),
            num_edges=int(data.edge_index.size(1)),
            num_features=int(data.num_node_features),
            num_classes=self.num_classes,
            num_time_steps=int(data.time_step.max()),
            splits=splits,
        )

    def _log_summary(self) -> None:
        """Emit a human-readable description of the prepared graph."""
        meta = self.metadata
        logger.info(
            "Prepared graph: %s nodes | %s edges | %d features | %d time steps",
            f"{meta.num_nodes:,}", f"{meta.num_edges:,}",
            meta.num_features, meta.num_time_steps,
        )
        for split in meta.splits:
            logger.info("  %s", split)
        unlabelled = int((self.data.y == UNKNOWN_LABEL).sum())
        logger.info(
            "  %s nodes (%.1f%%) are unlabelled and contribute topology but no loss",
            f"{unlabelled:,}", 100.0 * unlabelled / meta.num_nodes,
        )
