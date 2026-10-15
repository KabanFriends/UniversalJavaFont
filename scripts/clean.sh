#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT="$DIR/.."

for font_dir in "$ROOT"/ujf*/font; do
    [ -d "$font_dir" ] || continue
    echo "Cleaning ${font_dir#"$ROOT"/}"
    find "$font_dir" -mindepth 1 -delete
done
