#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

LOG_FILE="artifacts/basic_scenario.log"

{
  echo "=== Basic CI scenario ==="
  date
  echo

  echo "Python version:"
  python --version
  echo

  echo "Working directory:"
  pwd
  echo

  echo "Available raw batches:"
  ls -1 data/raw_batches || true
  echo

  echo "Starting model training on first 2 raw batches..."
  python -u run.py --mode train --max-batches 2
  echo

  echo "=== Basic CI scenario finished successfully ==="
  date
} 2>&1 | tee "$LOG_FILE"