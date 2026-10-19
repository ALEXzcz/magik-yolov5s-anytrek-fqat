set -e

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info

FLOAT_PROJECT=./runs/train/anytrek-yolov5n-float32
FQAT_PROJECT=./runs/train/anytrek-yolov5n-4bit
FLOAT_CKPT=./checkpoint/32W32F/weights/best-tmp.pt
FQAT_CKPT=./checkpoint/4W4F/weights/best-tmp.pt

## float32 baseline:
python3 train.py \
  --data data/anytrek_multiclass.yaml \
  --cfg models/yolov5n.yaml \
  --weights yolov5n.pt \
  --batch-size 50 \
  --workers 16 \
  --hyp data/hyp.scratch-float.yaml \
  --project "$FLOAT_PROJECT" \
  --epochs 300 \
  --device 0 \
  --bit 32

LATEST_FLOAT_BEST="$(find "$FLOAT_PROJECT" -type f -path '*/weights/best.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
cp "$LATEST_FLOAT_BEST" "$FLOAT_CKPT"

# ## 4bit FQAT:
# python3 train.py \
#   --data data/anytrek_multiclass.yaml \
#   --cfg models/yolov5n.yaml \
#   --weights "$FLOAT_CKPT" \
#   --batch-size 50 \
#   --workers 4 \
#   --hyp data/hyp.scratch.yaml \
#   --project "$FQAT_PROJECT" \
#   --epochs 300 \
#   --device 0 \
#   --optimizer Adam \
#   --calibrate-samples 128 \
#   --bit 4

# LATEST_FQAT_BEST="$(find "$FQAT_PROJECT" -type f -path '*/weights/best.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
# cp "$LATEST_FQAT_BEST" "$FQAT_CKPT"

# FQAT_RUN_DIR="$(dirname "$(dirname "$LATEST_FQAT_BEST")")"
# bash test.sh "$FQAT_RUN_DIR"

# watch -n 1 nvidia-smi
