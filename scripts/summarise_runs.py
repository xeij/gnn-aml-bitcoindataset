#!/usr/bin/env python3
"""Print a comparison table across every completed run in a checkpoint directory."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_COLUMNS = (
    ("model", 12), ("epochs", 7), ("best", 6), ("thr", 6),
    ("val F1", 8), ("test P", 8), ("test R", 8), ("test F1", 8),
    ("test AP", 8), ("AUC", 8), ("params", 10),
)


def load_runs(directory: Path) -> list[dict[str, Any]]:
    """Read every ``*_metrics.json`` sidecar written by a completed run."""
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*_metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_name"] = path.stem.replace("_metrics", "")
        runs.append(payload)
    return runs


def format_table(runs: Sequence[dict[str, Any]]) -> str:
    """Render the comparison table, sorted by test illicit F1."""
    header = " ".join(name.rjust(width) for name, width in _COLUMNS)
    lines = [header, "-" * len(header)]
    ordered = sorted(runs, key=lambda run: run["test"]["illicit_f1"], reverse=True)
    for run in ordered:
        test, validation = run["test"], run["validation"]
        config = run["config"]
        values = [
            config["model"]["name"],
            str(len(run["history"])),
            str(run["best_epoch"]),
            f"{run['threshold']:.3f}",
            f"{validation['illicit_f1']:.4f}",
            f"{test['illicit_precision']:.4f}",
            f"{test['illicit_recall']:.4f}",
            f"{test['illicit_f1']:.4f}",
            f"{test['average_precision']:.4f}",
            f"{test['roc_auc']:.4f}",
            f"{run.get('num_parameters', '—')}",
        ]
        lines.append(" ".join(v.rjust(w) for v, (_, w) in zip(values, _COLUMNS, strict=True)))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the benchmark table and return a process exit code."""
    parser = argparse.ArgumentParser(description="Compare completed training runs.")
    parser.add_argument("--dir", type=Path, default=Path("artifacts/checkpoints"))
    args = parser.parse_args(argv)

    runs = load_runs(args.dir)
    if not runs:
        print(f"No completed runs found in {args.dir}")
        return 1
    print("\nElliptic Bitcoin — illicit-class results on the held-out test window "
          "(time steps 35–49)\n")
    print(format_table(runs))
    print("\nAll metrics are for the illicit class at the threshold tuned on validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
