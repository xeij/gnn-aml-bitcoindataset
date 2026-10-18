#!/usr/bin/env bash
# Does retraining on fresher labels recover the post-step-43 collapse?
# Two windows, each scored only on steps after the dark-market shutdown:
#   recent: train 1-38, val 39-42, test 43-49  (more data, still no post-shutdown labels)
#   fresh:  train 1-43, val 44,    test 45-49  (two post-shutdown steps in train/val)
#
#   scripts/run_drift.sh            # both windows, seeds 0 1 2
#   WINDOWS=fresh scripts/run_drift.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SEEDS=${SEEDS:-"0 1 2"}
WINDOWS=${WINDOWS:-"recent fresh"}
MODELS=${MODELS:-"mlp gcn graphsage"}

for window in $WINDOWS; do
  case $window in
    recent) steps="--train-steps 1 38 --val-steps 39 42 --test-steps 43 49" ;;
    fresh)  steps="--train-steps 1 43 --val-steps 44 44 --test-steps 45 49" ;;
    *) echo "unknown window $window" >&2; exit 1 ;;
  esac
  out="artifacts/drift/$window"
  for seed in $SEEDS; do
    name="elliptic-rf-${window}-s${seed}"
    [[ -f "$out/${name}_metrics.json" ]] || \
      $PY scripts/train_rf.py --seed "$seed" --name "$name" --checkpoint-dir "$out" $steps
    for model in $MODELS; do
      name="elliptic-${model}-${window}-s${seed}"
      [[ -f "$out/${name}_metrics.json" ]] && continue
      echo "=== $name ==="
      $PY scripts/train.py --config "configs/${model}.yaml" --seed "$seed" \
        --name "$name" --checkpoint-dir "$out" --log-level WARNING $steps
    done
  done
  $PY scripts/summarise_runs.py --dir "$out"
done
