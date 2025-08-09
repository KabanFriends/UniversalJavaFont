#!/bin/bash
export UNICODE_VARIANT="ujf_jp"
export UNICODE_DIR="unicode_jp"
export PIXEL_VARIANT="ujf_symbols"
export PACK_NAME="ujf_symbols_jp"

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

python3 make_smooth_font_entries.py \
    --base font_entry_base.json \
    --unicode-dir pixel/$UNICODE_DIR \
    --pixel-prefix $UNICODE_DIR \
    --out font_entries/$PACK_NAME.json

python3 make_smooth_config.py \
    --template config_template/$PIXEL_VARIANT.json \
    --smooth font_entries/$PACK_NAME.json \
    --out config/$PACK_NAME.json

cp config/$PACK_NAME.json config.json
java -Xms1G -Xmx1G -jar BESmoothFontGen.jar
rm -r ../$PACK_NAME/subpacks/ujf/font/smooth/
mv smooth/ ../$PACK_NAME/subpacks/ujf/font/smooth/
