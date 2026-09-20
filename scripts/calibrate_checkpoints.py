#!/usr/bin/env python3
"""Fit a temperature onto checkpoints trained before calibration existed.

Retraining is not required to calibrate: the fix needs only the model's logits
on the validation split, which a trained checkpoint can already produce.  This
fits the temperature, refits the decision threshold on the rescaled scale, and
writes both back into the checkpoint and its metrics file.

Ranking is unaffected by construction -- a single positive divisor cannot
reorder anything -- so F1, precision and recall come out unchanged.  Run with
``--dry-run`` to see what would change without touching anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml_gnn.engine.calibration import (  # noqa: E402
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from aml_gnn.engine.inference import InferenceService  # noqa: E402
from aml_gnn.engine.metrics import evaluate_predictions, select_threshold  # noqa: E402
from aml_gnn.logging_utils import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def calibrate(path: Path, *, dry_run: bool = False) -> dict[str, float]:
    """Fit and persist a temperature for one checkpoint, returning a summary."""
    service = InferenceService.from_checkpoint(path)
    data = service.data
    with torch.no_grad():
        logits = service.model(data.x, data.edge_index).cpu()

    labels = data.y.cpu()
    validation = data.val_mask.cpu()
    test = data.test_mask.cpu()

    temperature = fit_temperature(logits[validation], labels[validation])
    before = torch.softmax(logits.float(), dim=-1)[:, 1]
    after = apply_temperature(logits, temperature)
    threshold, _ = select_threshold(
        labels[validation], after[validation], objective="f1"
    )

    old_metrics = evaluate_predictions(
        labels[test], before[test], threshold=service.threshold, split="test"
    )
    new_metrics = evaluate_predictions(
        labels[test], after[test], threshold=threshold, split="test"
    )
    summary = {
        "temperature": temperature,
        "threshold": threshold,
        "previous_threshold": service.threshold,
        "ece_before": expected_calibration_error(before[test], labels[test]),
        "ece_after": expected_calibration_error(after[test], labels[test]),
        "saturated_before": int((before >= 0.9999).sum()),
        "saturated_after": int((after >= 0.9999).sum()),
        "f1_before": old_metrics.illicit_f1,
        "f1_after": new_metrics.illicit_f1,
    }
    if dry_run:
        return summary

    state = torch.load(path, map_location="cpu", weights_only=False)
    state["temperature"] = temperature
    state["threshold"] = threshold
    state["test_metrics"] = new_metrics.to_dict()
    torch.save(state, path)

    metrics_path = path.with_name(f"{path.stem}_metrics.json")
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["temperature"] = temperature
        payload["threshold"] = threshold
        payload["calibration_error"] = summary["ece_after"]
        payload["test"] = new_metrics.to_dict()
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote temperature %.4f into %s", temperature, path.name)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Calibrate every checkpoint in a directory and report what changed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = parser.parse_args(argv)
    configure_logging("WARNING", force=True)

    checkpoints = sorted(args.dir.glob("*.pt"))
    if not checkpoints:
        print(f"No checkpoints found in {args.dir}")
        return 1

    header = f"{'model':<22}{'T':>8}{'threshold':>12}{'ECE':>18}{'saturated':>16}{'F1':>18}"
    print(f"\n{header}\n{'-' * len(header)}")
    for path in checkpoints:
        result = calibrate(path, dry_run=args.dry_run)
        print(
            f"{path.stem:<22}{result['temperature']:>8.3f}"
            f"{result['previous_threshold']:>6.3f}→{result['threshold']:<5.3f}"
            f"{result['ece_before']:>9.4f}→{result['ece_after']:<8.4f}"
            f"{result['saturated_before']:>7,}→{result['saturated_after']:<8,}"
            f"{result['f1_before']:>9.4f}→{result['f1_after']:<8.4f}"
        )
    if args.dry_run:
        print("\nDry run: nothing was written.")
    else:
        print("\nTemperature and refitted threshold written into every checkpoint.")
    print("F1 is unchanged by design — calibration cannot reorder predictions.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
