#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, isort)"

source activate glidertools-dev

echo "[black]"
black --exclude flo_functions --exclude __init__ --check -S -l 79 glidertools

echo "[isort]"
isort --recursive --check-only -w 79 glidertools

# Leave this commented, very strict
echo "[doc8]"
doc8 --ignore-path docs/_generated docs/
doc8 *.rst
