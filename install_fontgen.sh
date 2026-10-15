#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "* Building BESmoothFontGen"
chmod +x $DIR/BESmoothFontGen/gradlew
(cd $DIR/BESmoothFontGen && ./gradlew clean build)

echo "* Installing artifacts"
cp $DIR/BESmoothFontGen/build/libs/BESmoothFontGen-*-all.jar $DIR/scripts/BESmoothFontGen.jar

echo "* Installed"
