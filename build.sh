#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "* Cleaning old generated files"
(cd $DIR/scripts && ./clean.sh)

echo "* Installing texture sources"
(cd $DIR/scripts && ./install_textures.sh)

echo "* Building packs"
(cd $DIR/scripts && ./generate_all.sh)
