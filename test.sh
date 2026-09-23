#!/usr/bin/env bash
set -e

# RUN_DIR="${1:?Usage: bash test.sh <run_dir>}"

python3 val.py \
  --data ./data/anytrek_multiclass.yaml \
  --task test \
  --cfg ./models/yolov5n.yaml \
  --float_model ./checkpoint/32W32F/weights/best-tmp.pt \
  --weights ./checkpoint/4W4F/weights/best-tmp.pt \
  --imgsz 640 \
  --device 0 \
  --batch-size 64 \
  --workers 4 \
  --bit 4 \
  --project "$RUN_DIR" \
  --name . \
  --exist-ok \
  --verbose
