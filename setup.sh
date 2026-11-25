#!/bin/bash
# BlockN setup for Linux and macOS.

set -e

echo "Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is not installed."
    exit 1
fi

python3 - << 'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10 or newer is required (found {sys.version.split()[0]})")
print(f"Found Python {sys.version.split()[0]}")
PY

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[gui,dev]"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example."
fi

mkdir -p exports saves

python -m pytest
echo
echo "Setup complete."
echo "  source .venv/bin/activate"
echo "  python main.py"
echo "  blockn generate --classical-order Doric --seed 7 --width 10 --depth 8 --summary"
