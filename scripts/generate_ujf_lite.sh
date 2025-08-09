#!/bin/bash
export UNICODE_VARIANT="ujf"
export UNICODE_DIR="unicode"
export PIXEL_VARIANT="ujf"
export PACK_NAME="ujf_lite"

python3 conv_unifont.py \
    --list-file unifont_list/$UNICODE_VARIANT.json \
    --hex-dir unifont_hex \
    --out-dir pixel/$UNICODE_DIR

python3 merge_glyph.py \
    --config merge_config/$PIXEL_VARIANT.json \
    --overrides unifont_width_overrides.json \
    --unicode-dir pixel/$UNICODE_DIR \
    --out-dir ../$PACK_NAME/subpacks/ujf/font \
    --verbose

: '
python3 merge_default8.py \
    --config merge_config/$PIXEL_VARIANT.json \
    --out ../$PACK_NAME/subpacks/ujf/font/default8.png
'
cp default8_improved.png ../$PACK_NAME/subpacks/ujf/font/default8.png
