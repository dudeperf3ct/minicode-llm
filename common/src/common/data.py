"""Shared data records, hashing, checksums, and batching."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple


class Stratum(NamedTuple):
    """Joint code-task sampling fields."""

    gpt_difficulty: str
    subset: str
    style: str


@dataclass(frozen=True)
class Candidate:
    """A code task eligible for deterministic selection."""

    question_id: str
    source_index: int
    stratum: Stratum
    normalized_question: str


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def stable_hash(seed: int, namespace: str, value: str) -> str:
    payload = f"{seed}\0{namespace}\0{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {label}")
    if path.stat().st_size != expected["bytes"]:
        raise RuntimeError(f"Size mismatch for {label}")
    if sha256_file(path) != expected["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {label}")


def batched(values: list[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("Batch size must be positive")
    for index in range(0, len(values), size):
        yield values[index : index + size]


def candidate_ids(candidates: list[Candidate]) -> list[str]:
    return [candidate.question_id for candidate in candidates]
