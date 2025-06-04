#!/bin/bash
set -e

LOG_FILE="/output/${OUTPUT_FILE%.*}.txt"

python3 image_stitch_batch.py \
  --path /data/ \
  --output /output/"${OUTPUT_FILE}" \
  --matching-algorithm "${MATCHING_ALGORITHM}" \
  --vram-size "${VRAM_SIZE}" \
  2>&1 | tee "$LOG_FILE"
