"""Read and write the experiment's JSON files consistently.

Manifests, reports, prepared datasets, and resumable evaluation results all use
UTF-8 JSON. These helpers keep formatting and JSONL serialization in one place
without coupling generic file operations to the dataset-writing pipeline.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            write_jsonl_line(handle, record)


def write_jsonl_line(handle: TextIO, value: Any) -> None:
    handle.write(compact_json(value) + "\n")
