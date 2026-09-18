FLOAT_CKPT=./checkpoint/32W32F/weights/best-tmp.pt
FQAT_CKPT=./checkpoint/4W4F/weights/best-tmp.pt
DATA_CFG=data/anytrek_multiclass.yaml
SOURCE_DIR=data/anytrek_detect/images

# for detect single img
# float
# python detect.py \
#   --source "$SOURCE_DIR" \
#   --data "$DATA_CFG" \
#   --float_model "$FLOAT_CKPT" \
#   --weights "$FLOAT_CKPT" \
#   --imgs 640 \
#   --device 0 \
#   --model_only \
#   --bit 32 \
#   --conf-thres 0.5 \
#   --iou-thres 0.4

# 8bit
# python detect.py \
#   --source "$SOURCE_DIR" \
#   --data "$DATA_CFG" \
#   --float_model "$FLOAT_CKPT" \
#   --weights ./checkpoint/8W8F/weights/best-tmp.pt \
#   --imgs 640 \
#   --device 0 \
#   --view-img \
#   --bit 8

# 4bit
python detect.py \
  --source "$SOURCE_DIR" \
  --data "$DATA_CFG" \
  --float_model "$FLOAT_CKPT" \
  --weights "$FQAT_CKPT" \
  --imgs 640 \
  --device 0 \
  --view-img \
  --bit 4 \
  --conf-thres 0.5 \
  --iou-thres 0.4

#for onnx:
# 8bit
# python detect.py \
#   --source "$SOURCE_DIR" \
#   --data "$DATA_CFG" \
#   --float_model "$FLOAT_CKPT" \
#   --weights ./checkpoint/8W8F/weights/best-tmp.pt \
#   --imgs 640 \
#   --device 0 \
#   --view-img \
#   --onnx \
#   --bit 8

# 4bit
# python detect.py \
#   --source "$SOURCE_DIR" \
#   --data "$DATA_CFG" \
#   --float_model "$FLOAT_CKPT" \
#   --weights "$FQAT_CKPT" \
#   --imgs 640 \
#   --device 0 \
#   --view-img \
#   --onnx \
#   --bit 4
