#!/bin/bash
#
# Usage:
#   sh denoise.sh <checkpoint> <input> <output> [config] [tile]
#
#   tile=0 : full-image inference (use with model trained at matching resolution)
#   tile=64: tiled inference with 64x64 tiles (use with model trained at patch_size=64)
#
# Examples:
#   sh denoise.sh ckpt.pth ./noisy.tif ./denoised.tif                    # full-image
#   sh denoise.sh ckpt.pth ./noisy.tif ./denoised.tif tbsn_confocal.json 64  # tiled
#

set -e

CHECKPOINT="${1:?missing checkpoint path}"
INPUT="${2:?missing input path}"
OUTPUT="${3:?missing output path}"
CONFIG="${4:-tbsn_confocal.json}"
TILE="${5:-0}"

python validate/inference.py \
    --config "option/${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --tile "${TILE}"
