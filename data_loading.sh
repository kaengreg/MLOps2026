#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$PROJECT_ROOT/data/source"
ARCHIVE_PATH="$SOURCE_DIR/nyc-yellow-taxi-trip-data.zip"

mkdir -p "$SOURCE_DIR"

cd "$SOURCE_DIR"

kaggle datasets download elemento/nyc-yellow-taxi-trip-data -p "$SOURCE_DIR"

if [ -f "$ARCHIVE_PATH" ]; then
    unzip -o "$ARCHIVE_PATH" -d "$SOURCE_DIR"
    echo "Archive extracted to: $SOURCE_DIR"
else
    echo "Downloaded archive not found: $ARCHIVE_PATH"
    exit 1
fi