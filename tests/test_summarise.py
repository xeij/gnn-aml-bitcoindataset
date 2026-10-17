"""Seed aggregation in the benchmark summary script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "summarise_runs", Path(__file__).resolve().parents[1] / "scripts" / "summarise_runs.py"
)
assert _SPEC is not None and _SPEC.loader is not None
summarise = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summarise)


def _run(model: str, local: bool, f1: float) -> dict[str, object]:
    test = dict.fromkeys(summarise._AGG_KEYS, 0.5)
    test["illicit_f1"] = f1
    return {"config": {"model": {"name": model}, "data": {"local_features_only": local}},
            "test": test}


def test_groups_by_model_and_feature_set() -> None:
    """Seeds of the same model/feature pair collapse to one row with mean and std."""
    rows = summarise.aggregate([
        _run("gat", False, 0.6), _run("gat", False, 0.7), _run("gat", True, 0.4),
    ])
    assert [(r["model"], r["features"], r["seeds"]) for r in rows] == [
        ("gat", "all", 2), ("gat", "local", 1),
    ]
    mean, std = rows[0]["illicit_f1"]
    assert abs(mean - 0.65) < 1e-9 and abs(std - 0.0707) < 1e-3
    assert rows[1]["illicit_f1"] == (0.4, 0.0)
    assert "0.650 ± 0.071" in summarise.format_aggregate(rows)
