set -e

cd ../
sed -ri 's/(target_device=")[^"]*/\1T33/' models/yolo.py models/common.py
sed -ri 's/(target_device" : ")[^"]*/\1T33/' magik_quantize.py

python detect.py \
  --source data/images/bus.jpg \
  --float_model checkpoint/32W32F/weights/best-tmp.pt \
  --weights checkpoint/4W4F/weights/best-tmp.pt \
  --imgs 640 \
  --device 0 \
  --bit 4 \
  --onnx
cd -

../magik-transform-tools \
  --inputpath ../checkpoint/4W4F/weights/best-tmp.onnx \
  --outputpath ./yolov5n_t33_magik_fqat_nv12.mk.h \
  --config cfg/magik_t33_nv12.cfg

MODEL_NAME="yolov5n_t33_magik_fqat_nv12.mgk"
MODEL_ROOT="uranus_sample_yolov5n-anytrek-fqat/model_t33"

for ABI in glibc uclibc; do
  MODEL_DIR="${MODEL_ROOT}/${ABI}"
  mkdir -p "${MODEL_DIR}"
  cp -f "compile_result/${ABI}/${MODEL_NAME}" "${MODEL_DIR}/${MODEL_NAME}"
  bash ./write_model_metadata.sh "${MODEL_DIR}" "${MODEL_NAME}"
done

rm -rf compile_result
rm -f magik_transform_event_*.log
