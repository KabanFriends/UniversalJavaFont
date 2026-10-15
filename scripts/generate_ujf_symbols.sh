#!/bin/bash
set -e
PYTHON="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )/.venv/bin/python3"

./check_deps.sh

export UNICODE_VARIANT="ujf"
export UNICODE_DIR="unicode"
export PIXEL_VARIANT="ujf_symbols"
export PACK_NAME="ujf_symbols"

"$PYTHON" conv_unifont.py \
    --list-file unifont_list/$UNICODE_VARIANT.json \
    --hex-dir unifont_hex \
    --out-dir pixel/$UNICODE_DIR

"$PYTHON" merge_glyph.py \
    --config merge_config/$PIXEL_VARIANT.json \
    --overrides unifont_width_overrides.json \
    --unicode-dir pixel/$UNICODE_DIR \
    --out-dir ../$PACK_NAME/font \
    --verbose

: '
"$PYTHON" merge_default8.py \
    --config merge_config/$PIXEL_VARIANT.json \
    --out ../$PACK_NAME/font/default8.png
'
cp default8_improved.png ../$PACK_NAME/font/default8.png

"$PYTHON" make_smooth_font_entries.py \
    --base font_entry_base.json \
    --unicode-dir pixel/$UNICODE_DIR \
    --pixel-prefix $UNICODE_DIR \
    --out font_entries/$PACK_NAME.json

"$PYTHON" make_smooth_config.py \
    --template config_template/$PIXEL_VARIANT.json \
    --smooth font_entries/$PACK_NAME.json \
    --out config/$PACK_NAME.json

# Share identical glyphs and move characters from pages the client never
# loads into the freed texture cells (writes cellMapping into the config)
"$PYTHON" make_cell_mapping.py \
    --config config/$PACK_NAME.json \
    --priority symbol_chars.json \
    --fill-rest \
    --out config/${PACK_NAME}_cells.json \
    --report config/${PACK_NAME}_cells_report.json

cp config/$PACK_NAME.json config.json
java -Djava.awt.headless=true -Xms1G -Xmx1G -jar BESmoothFontGen.jar
rm -rf ../$PACK_NAME/font/smooth/
mv smooth/ ../$PACK_NAME/font/smooth/
