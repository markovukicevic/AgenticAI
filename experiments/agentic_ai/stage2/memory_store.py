import json
from pathlib import Path
from typing import Any, Dict


MEMORY_FILE = Path(__file__).parent / "memory.json"


def load_memory() -> Dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {}

    try:
        with MEMORY_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def save_memory(
    memory: Dict[str, Any],
) -> None:
    with MEMORY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            memory,
            file,
            indent=2,
            ensure_ascii=False,
        )