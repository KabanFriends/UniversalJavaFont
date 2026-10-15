#!/bin/bash
PYTHON="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )/.venv/bin/python3"

"$PYTHON" filter_symbols.py \
    pixel/accented.json \
    --symbols symbol_chars.json \
    --out pixel/accented_symbols.json

"$PYTHON" filter_symbols.py \
    pixel/nonlatin_european.json \
    --symbols symbol_chars.json \
    --out pixel/nonlatin_european_symbols.json
