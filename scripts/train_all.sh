#!/usr/bin/env bash
# Train the three models one after another. Each has the GPU to itself.
# Order: fastest first, so a bug shows up in minutes, not hours.
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p output/logs
for m in segformer3d segresnet swin_unetr; do
  echo "=== $m started $(date)"
  python ml/train.py --config ml/configs/$m.yaml 2>&1 | tee output/logs/$m.log
  echo "=== $m finished $(date)"
done
