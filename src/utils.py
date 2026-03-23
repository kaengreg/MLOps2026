import json
from typing import Any

from config import PIPELINE_STATE_FILE


def load_pipeline_state() -> dict[str, Any]:
    if not PIPELINE_STATE_FILE.exists():
        return {"last_processed_batch": 0, "initialized": False}

    with open(PIPELINE_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pipeline_state(state: dict[str, Any]) -> None:
    PIPELINE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(PIPELINE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)