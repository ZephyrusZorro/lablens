#!/usr/bin/env bash
# LabLens one-time setup for Linux / macOS.
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for v in python3.11 python3.12 python3.10 python3; do
  if command -v "$v" >/dev/null && "$v" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'; then
    PY="$v"; break
  fi
done
[ -n "$PY" ] || { echo "LabLens needs Python 3.10-3.12"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
python scripts/setup_models.py
python -m lablens.doctor
echo "Setup complete. Run: source .venv/bin/activate && python -m lablens.live"
