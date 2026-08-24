#!/usr/bin/env bash

set -euo pipefail

FLOAT_CKPT=./checkpoint/32W32F/weights/best-tmp.pt
DATA_CFG=data/anytrek_multiclass.yaml
SOURCE_VIDEO=/home/alex/Desktop/yolov5n-anytrek-fqat/cargo_cam_video_mask_raw.mp4
IMG_SIZE=640
DEVICE=0

python detect.py \
  --source "$SOURCE_VIDEO" \
  --data "$DATA_CFG" \
  --float_model "$FLOAT_CKPT" \
  --weights "$FLOAT_CKPT" \
  --imgs "$IMG_SIZE" \
  --device "$DEVICE" \
  --view-img \
  --model_only \
  --bit 32
