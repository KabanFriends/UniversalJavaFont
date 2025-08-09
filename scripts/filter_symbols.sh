#!/bin/bash
python3 filter_symbols.py \
    pixel/accented.json \
    --symbols symbol_chars.json \
    --out pixel/accented_symbols.json

python3 filter_symbols.py \
    pixel/nonlatin_european.json \
    --symbols symbol_chars.json \
    --out pixel/nonlatin_european_symbols.json
