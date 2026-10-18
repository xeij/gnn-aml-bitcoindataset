#!/usr/bin/env python3
"""Random-forest baseline on the same temporal split as the GNN runs.

Weber et al. (2019) report that a random forest beats every GNN they tried on
Elliptic. This script reproduces that reference point under this project's
split and threshold protocol, and writes a ``*_metrics.json`` sidecar in the
same shape as the trainer so ``summarise_runs.py`` can place it in the table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml_gnn.config import DataConfig  # noqa: E402
from aml_gnn.data.elliptic import EllipticDataModule  # noqa: E402
from aml_gnn.engine.metrics import evaluate_predictions, select_threshold  # noqa: E402
from aml_gnn.logging_utils import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Fit, tune the threshold on validation, score the test window."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--local-features-only", action="store_true")
    parser.add_argument("--name", default="elliptic-rf")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-steps", type=int, nargs=2, default=None)
    args = parser.parse_args(argv)
    configure_logging("INFO", force=True)

    started = time.perf_counter()
    windows = {
        f"{split}_time_steps": tuple(bounds)
        for split in ("train", "val", "test")
        if (bounds := getattr(args, f"{split}_steps")) is not None
    }
    data_config = DataConfig(
        local_features_only=args.local_features_only, standardise=False, **windows
    )
    module = EllipticDataModule(data_config)
    data = module.prepare()
    x, y = data.x.numpy(), data.y.numpy()

    forest = RandomForestClassifier(
        n_estimators=args.trees, n_jobs=-1, random_state=args.seed
    )
    forest.fit(x[data.train_mask.numpy()], y[data.train_mask.numpy()])
    scores = torch.from_numpy(forest.predict_proba(x)[:, 1].astype(np.float32))

    threshold, _ = select_threshold(data.y[data.val_mask], scores[data.val_mask])
    splits = {}
    for split in ("val", "test"):
        mask = getattr(data, f"{split}_mask")
        splits[split] = evaluate_predictions(
            data.y[mask], scores[mask], split=split, threshold=threshold,
            time_step=data.time_step[mask],
        )
        logger.info("%s", splits[split])

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "name": args.name,
            "data": data_config.to_dict(),
            "model": {"name": "random_forest", "n_estimators": args.trees},
            "train": {"seed": args.seed},
        },
        "best_epoch": 0,
        "num_parameters": int(sum(t.tree_.node_count for t in forest.estimators_)),
        "total_seconds": time.perf_counter() - started,
        "threshold": threshold,
        "temperature": 1.0,
        "calibration_error": float("nan"),
        "validation": splits["val"].to_dict(),
        "test": splits["test"].to_dict(),
        "history": [],
    }
    out = args.checkpoint_dir / f"{args.name}_metrics.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
