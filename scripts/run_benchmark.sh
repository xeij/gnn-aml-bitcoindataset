#!/usr/bin/env bash
# Train every registered architecture under identical settings.
set -euo pipefail
cd "$(dirname "$0")/.."
for model in mlp gcn graphsage gat; do
  echo "=== ${model} ==="
  .venv/bin/python scripts/train.py --config "configs/${model}.yaml" --log-level INFO
done
