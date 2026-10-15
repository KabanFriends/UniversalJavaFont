#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "* Creating virtual environment"
python3 -m venv "$DIR/.venv"

echo "* Installing packages"
"$DIR/.venv/bin/pip" install -r "$DIR/requirements.txt"

echo "* Done"
