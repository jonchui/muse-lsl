#!/usr/bin/env bash
# Record an existing Muse LSL stream plus typed event markers.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

DURATION_SECONDS=3600
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  DURATION_SECONDS="$1"
  shift
fi

python3 scripts/record_session.py --duration "$DURATION_SECONDS" "$@"
