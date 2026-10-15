#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BUILD_DIR="$DIR/.build/msdfgen"

echo "* Building msdfgen"
cmake -S "$DIR/msdfgen" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMSDFGEN_USE_VCPKG=OFF \
    -DMSDFGEN_USE_SKIA=OFF \
    -DMSDFGEN_DISABLE_SVG=ON
cmake --build "$BUILD_DIR" --parallel

echo "* Installing artifacts"
mkdir -p "$DIR/scripts/msdfgen"
cp "$BUILD_DIR/msdfgen" "$DIR/scripts/msdfgen/msdfgen"

echo "* Installed"
