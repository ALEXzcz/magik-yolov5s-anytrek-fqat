#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Fixed configuration for this batch of CVAT pre-annotations.
SOURCE_DIR="$ROOT_DIR/data/cropped_1920x1080"
DATA_CFG="$ROOT_DIR/data/anytrek_multiclass.yaml"
FLOAT_MODEL="$ROOT_DIR/checkpoint/32W32F/weights/best-tmp.pt"
OUTPUT_DIR="$ROOT_DIR/runs/cvat_prelabels/inference"

export MPLCONFIGDIR="/tmp/yolov5n-anytrek-matplotlib"

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Image directory does not exist: $SOURCE_DIR" >&2
    exit 1
fi
if [[ ! -f "$DATA_CFG" || ! -f "$ROOT_DIR/models/yolov5n.yaml" || ! -f "$FLOAT_MODEL" ]]; then
    echo "Model or data configuration is missing." >&2
    echo "DATA_CFG=$DATA_CFG" >&2
    echo "Model config: $ROOT_DIR/models/yolov5n.yaml" >&2
    echo "FLOAT_MODEL=$FLOAT_MODEL" >&2
    exit 1
fi

echo "Source: $SOURCE_DIR"
echo "Model:  $FLOAT_MODEL"
echo "Device: 0"
echo "Output: $OUTPUT_DIR"
echo "Thresholds: conf=0.5, iou=0.4"

python detect.py \
    --source "$SOURCE_DIR" \
    --data "$DATA_CFG" \
    --cfg "$ROOT_DIR/models/yolov5n.yaml" \
    --float_model "$FLOAT_MODEL" \
    --weights "$FLOAT_MODEL" \
    --imgs 640 \
    --device 0 \
    --model_only \
    --bit 32 \
    --classes 0 1 \
    --conf-thres 0.5 \
    --iou-thres 0.4 \
    --save-txt \
    --save-conf \
    --project "$(dirname "$OUTPUT_DIR")" \
    --name "$(basename "$OUTPUT_DIR")" \
    --exist-ok

python tools/yolo_to_cvat.py \
    --images "$SOURCE_DIR" \
    --labels "$OUTPUT_DIR/labels" \
    --xml "$OUTPUT_DIR/cvat_annotations.xml" \
    --json "$OUTPUT_DIR/detections.json"

(
    cd "$OUTPUT_DIR"
    zip -q -j cvat_annotations.zip cvat_annotations.xml
)

echo
echo "CVAT upload file: $OUTPUT_DIR/cvat_annotations.zip"
echo "Raw YOLO labels:  $OUTPUT_DIR/labels/"
echo "Pixel detections:  $OUTPUT_DIR/detections.json"
