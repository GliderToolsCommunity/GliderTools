#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

source activate glidertools-dev

echo "[flake8]"
flake8 glidertools --exclude=__init__.py --max-line-length=79 --ignore=C901,W605,W503,F722

echo "[black]"
black --check -S -l 79 glidertools

echo "[isort]"
isort --recursive --check-only -w 79 glidertools

# Leave this commented, very strict
echo "[doc8]"
# doc8 docs/source
doc8 *.rst
