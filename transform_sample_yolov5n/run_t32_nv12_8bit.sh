cd ../
sed -ri 's/(target_device=")[^"]*/\1T32/' models/yolo.py models/common.py
sed -ri 's/(target_device" : ")[^"]*/\1T32/' magik_quantize.py

python detect.py \
  --source data/images/bus.jpg \
  --float_model checkpoint/32W32F/weights/best-tmp.pt \
  --weights checkpoint/8W8F/weights/best-tmp.pt \
  --imgs 640 \
  --device 0 \
  --bit 8 \
  --onnx
cd -

../../../../TransformKit/magik-transform-tools \
  --inputpath ../checkpoint/8W8F/weights/best-tmp.onnx \
  --outputpath ./yolov5n_t32_magik_fqat_nv12.mk.h \
  --config cfg/magik_t32_nv12.cfg

cp -r compile_result/glibc/ uranus_sample_yolov5n-anytrek-fqat/model_t32/
cp -r compile_result/uclibc/ uranus_sample_yolov5n-anytrek-fqat/model_t32/
rm -r compile_result magik_transform_event_*.log
