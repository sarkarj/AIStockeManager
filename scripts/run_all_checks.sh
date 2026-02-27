#!/usr/bin/env bash
set -euo pipefail

pytest -q
python3 scripts/verify_drl_integrity.py
