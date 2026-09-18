#!/usr/bin/env bash
set -e

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
  --project ./runs/train/anytrek-yolov5n-4bit2 \
  --name . \
  --exist-ok \
  --verbose
