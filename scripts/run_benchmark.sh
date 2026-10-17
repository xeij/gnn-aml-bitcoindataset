#!/usr/bin/env bash
# Multi-seed benchmark: every architecture plus a random-forest baseline, with
# and without the 72 aggregated neighbour features. Results land in
# artifacts/benchmark/ and never touch the checkpoints the UI serves.
#
#   scripts/run_benchmark.sh            # seeds 0 1 2, full and local features
#   SEEDS="0" FEATURES="full" scripts/run_benchmark.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=artifacts/benchmark
SEEDS=${SEEDS:-"0 1 2"}
FEATURES=${FEATURES:-"full local"}
MODELS=${MODELS:-"mlp gcn graphsage gat"}

for feat in $FEATURES; do
  flag=""; [[ $feat == local ]] && flag="--local-features-only"
  for seed in $SEEDS; do
    name="elliptic-rf-${feat}-s${seed}"
    [[ -f "$OUT/${name}_metrics.json" ]] || \
      $PY scripts/train_rf.py --seed "$seed" --name "$name" --checkpoint-dir "$OUT" $flag
    for model in $MODELS; do
      name="elliptic-${model}-${feat}-s${seed}"
      [[ -f "$OUT/${name}_metrics.json" ]] && continue
      echo "=== $name ==="
      $PY scripts/train.py --config "configs/${model}.yaml" --seed "$seed" \
        --name "$name" --checkpoint-dir "$OUT" --log-level WARNING $flag
    done
  done
done
$PY scripts/summarise_runs.py --dir "$OUT"
