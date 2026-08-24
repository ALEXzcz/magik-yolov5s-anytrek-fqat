FLOAT_CKPT=./checkpoint/32W32F/weights/best-tmp.pt
FQAT_CKPT=./checkpoint/4W4F/weights/best-tmp.pt
DATA_CFG=data/anytrek_multiclass.yaml

# for float:
python val.py --data "$DATA_CFG" --cfg models/yolov5n.yaml --float_model "$FLOAT_CKPT" --weights "$FLOAT_CKPT" --imgs 640 --device 0 --batch-size 100 --model_only

# for 8bit:
# python val.py --data "$DATA_CFG" --cfg models/yolov5n.yaml --float_model "$FLOAT_CKPT" --weights ./checkpoint/8W8F/weights/best-tmp.pt --imgs 640 --device 0 --batch-size 100 --bit 8

# for 4bit:
# python val.py --data "$DATA_CFG" --cfg models/yolov5n.yaml --float_model "$FLOAT_CKPT" --weights "$FQAT_CKPT" --imgs 640 --device 0 --batch-size 100 --bit 4
