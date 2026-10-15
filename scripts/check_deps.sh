#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

check() {
    if [ ! -f "$1" ]; then
        echo "ERROR: $2 was not found ($1). Run $3 in the repository root first." >&2
        exit 1
    fi
}

check "$DIR/../.venv/bin/python3" "Python virtual environment" ./setup_venv.sh
check "$DIR/BESmoothFontGen.jar" "BESmoothFontGen.jar" ./install_fontgen.sh
check "$DIR/msdfgen/msdfgen" "msdfgen binary" ./install_msdfgen.sh
