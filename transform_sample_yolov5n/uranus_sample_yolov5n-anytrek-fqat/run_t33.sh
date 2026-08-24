##### SOC T32 #####
export LD_LIBRARY_PATH=$(pwd)/../InferenceKit/uranus/tnna/540/lib/uclibc:${LD_LIBRARY_PATH}

MODEL_DIR=model_t33/uclibc
MODEL_NAME=$(find "${MODEL_DIR}" -maxdepth 1 -type f -name 'yolov5n_t33_magik_fqat_nv12_*.mgk' | sort | tail -n 1)

if [ -z "${MODEL_NAME}" ]; then
  echo "No timestamped mgk model found in ${MODEL_DIR}" >&2
  exit 1
fi

echo "Using model: ${MODEL_NAME}"

# for nv12 model
./load_mgk_uclibc_nv12 "${MODEL_NAME}" test_image/bus_640x640_nv12.yuv 640 640
