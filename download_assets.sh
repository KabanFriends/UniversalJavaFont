#!/bin/bash
set -e

VERSION="26.3"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_URL="https://assets.mcasset.cloud/$VERSION/assets/minecraft"
TEXTURE_DIR="$DIR/scripts/texture_sources"
HEX_DIR="$DIR/scripts/unifont_hex"

TEXTURES=(ascii accented nonlatin_european)
UNIFONT_ZIPS=(unifont unifont_jp unifont_pua)

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$TEXTURE_DIR" "$HEX_DIR"

echo "* Downloading textures ($VERSION)"
for name in "${TEXTURES[@]}"; do
    echo "  $name.png"
    curl -fsSL "$BASE_URL/textures/font/$name.png" -o "$TEXTURE_DIR/$name.png"
done

echo "* Downloading unifont hex files ($VERSION)"
for name in "${UNIFONT_ZIPS[@]}"; do
    echo "  $name.zip"
    curl -fsSL "$BASE_URL/font/$name.zip" -o "$TMP_DIR/$name.zip"
    python3 -I -m zipfile -e "$TMP_DIR/$name.zip" "$TMP_DIR/$name"
    mv "$TMP_DIR/$name"/*.hex "$HEX_DIR/"
done

echo "* Downloaded"
