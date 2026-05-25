#!/bin/sh
#
# Usage:
#   sh denoise.sh <checkpoint> <input> <output> [config] [tile]
#
#   tile=0 : full-image inference (use with model trained at matching resolution)
#   tile=64: tiled inference with 64x64 tiles (use with model trained at patch_size=64)
#
#   <input> and <output> can be files or directories.
#   If <input> is a directory, all image files in it will be processed
#   and saved to the <output> directory with the same filenames.
#
# Examples:
#   sh denoise.sh ckpt.pth ./noisy.tif ./denoised.tif                          # single file
#   sh denoise.sh ckpt.pth ./noisy.tif ./denoised.tif tbsn_confocal.json 64    # tiled
#   sh denoise.sh ckpt.pth ./noisy_images/ ./denoised_images/                   # folder
#   sh denoise.sh ckpt.pth ./noisy_images/ ./denoised_images/ tbsn_confocal.json 64  # folder + tiled
#

set -e

CHECKPOINT="${1:?missing checkpoint path}"
INPUT="${2:?missing input path}"
OUTPUT="${3:?missing output path}"
CONFIG="${4:-tbsn_confocal.json}"
TILE="${5:-0}"

run_inference() {
    python validate/inference.py \
        --config "option/${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --input "$1" \
        --output "$2" \
        --tile "${TILE}"
}

if [ -f "${INPUT}" ]; then
    run_inference "${INPUT}" "${OUTPUT}"
elif [ -d "${INPUT}" ]; then
    mkdir -p "${OUTPUT}"
    for img in "${INPUT}"/*; do
        [ -f "${img}" ] || continue
        case "${img}" in
            *.tif|*.tiff|*.png|*.jpg|*.jpeg|*.bmp|*.TIF|*.TIFF|*.PNG|*.JPG|*.JPEG|*.BMP) ;;
            *) continue ;;
        esac
        fname=$(basename "${img}")
        run_inference "${img}" "${OUTPUT}/${fname}"
    done
else
    echo "Error: ${INPUT} is not a valid file or directory"
    exit 1
fi
