#!/bin/bash
set -euo pipefail; IFS=$'\n\t'

NAME=$( python3 setup.py --name )
VER=$( python3 setup.py --version )

echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

git tag v$VER
git push origin v$VER

echo "========================================================================"
echo "Releasing $NAME v$VER on PyPI"
echo "========================================================================"

python3 -m build
python3 -m twine upload dist/*
rm -r dist/ *.egg-info
