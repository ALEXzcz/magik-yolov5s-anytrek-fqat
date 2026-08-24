#!/usr/bin/env bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <model_dir> <model_name>" >&2
  exit 1
fi

MODEL_DIR="$1"
MODEL_NAME="$2"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Model file not found: ${MODEL_PATH}" >&2
  exit 1
fi

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
MODEL_BASENAME="${MODEL_NAME%.mgk}"
STAMPED_MODEL_NAME="${MODEL_BASENAME}_${TIMESTAMP}.mgk"
STAMPED_MODEL_PATH="${MODEL_DIR}/${STAMPED_MODEL_NAME}"

mv "${MODEL_PATH}" "${STAMPED_MODEL_PATH}"

echo "Generated:"
echo "  ${STAMPED_MODEL_PATH}"
