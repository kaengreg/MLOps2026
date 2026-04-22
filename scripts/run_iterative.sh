#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts \
  state \
  dev/models \
  dev/metrics \
  dev/reports \
  dev/predictions

LOG_FILE="artifacts/iterative_scenario.log"

N_ITERATIONS="${N_ITERATIONS:-2}"
INITIAL_BATCHES="${INITIAL_BATCHES:-2}"

{
  echo "=== Iterative CI scenario ==="
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

  echo "Configured update iterations: ${N_ITERATIONS}"
  echo "Initial training batch count: ${INITIAL_BATCHES}"
  echo

  if ! [[ "$N_ITERATIONS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: N_ITERATIONS must be a non-negative integer"
    exit 1
  fi

  if ! [[ "$INITIAL_BATCHES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: INITIAL_BATCHES must be a positive integer"
    exit 1
  fi

  if [ "$INITIAL_BATCHES" -lt 2 ]; then
    echo "ERROR: INITIAL_BATCHES must be at least 2"
    exit 1
  fi

  echo "Resetting state and old artifacts for a clean CI run..."
  rm -f state/pipeline_state.json
  rm -f dev/models/*
  rm -f dev/metrics/*
  rm -f dev/reports/*
  rm -f dev/predictions/*
  echo "State reset completed."
  echo

  echo "----------------------------------------"
  echo "Initial training"
  echo "Training with max_batches=${INITIAL_BATCHES}"
  echo "----------------------------------------"
  python -u run.py --mode train --max-batches "${INITIAL_BATCHES}"
  echo

  for ((i=1; i<=N_ITERATIONS; i++)); do
    echo "----------------------------------------"
    echo "Update iteration ${i}/${N_ITERATIONS}"
    echo "----------------------------------------"
    python -u run.py --mode update
    echo
  done

  echo "=== Iterative CI scenario finished successfully ==="
  date
} 2>&1 | tee "$LOG_FILE"