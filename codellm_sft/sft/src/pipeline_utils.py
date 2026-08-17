"""Shared local primitives for the matched KodCode preparation pipeline.

The preparation scripts use these helpers to keep hashing, normalization,
manifest verification, batching, and Qwen chat-template rendering consistent
across source filtering, decontamination, statistics, and uploads.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from json_utils import read_json

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

PROJECT_DIR = Path(__file__).resolve().parents[1]
REVISIONS_PATH = PROJECT_DIR / "manifests" / "revisions.json"
DEFAULT_TRAIN_SIZE = 10_000
VALIDATION_SIZE = 500
TEST_SIZE = 500
PILOT_SIZE = 1_000
OVERFIT_SIZE = 32
STRATIFICATION_FIELDS = ("gpt_difficulty", "subset", "style")
SOURCE_COLUMNS = [
    "style",
    "subset",
    "question_id",
    "question",
    "test",
    "test_info",
    "gpt_difficulty",
    "r1_correctness",
    "r1_solution",
    "conversations",
]

type Messages = list[dict[str, str]]


class Stratum(NamedTuple):
    """Joint sampling fields with tuple-compatible ordering and hashing."""

    gpt_difficulty: str
    subset: str
    style: str


@dataclass(frozen=True)
class Candidate:
    """A source row that remains eligible for deterministic selection."""

    question_id: str
    source_index: int
    stratum: Stratum
    normalized_question: str


def normalize_text(text: str) -> str:
    """Apply the experiment's locked question normalization."""

    return " ".join(text.lower().strip().split())


def stable_hash(seed: int, namespace: str, value: str) -> str:
    """Return a deterministic namespaced rank for sampling and ordering."""

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
    for index in range(0, len(values), size):
        yield values[index : index + size]


def render_batch(tokenizer: PreTrainedTokenizerBase, conversations: list[Messages]) -> list[str]:
    """Render conversations with the pinned model's native chat template."""

    rendered = tokenizer.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=False
    )
    if isinstance(rendered, str):
        if len(conversations) != 1:
            raise RuntimeError("Tokenizer returned one string for a conversation batch")
        return [rendered]
    if len(rendered) != len(conversations):
        raise RuntimeError("Tokenizer returned the wrong number of rendered conversations")
    return rendered


def token_lengths(tokenizer: PreTrainedTokenizerBase, rendered: list[str]) -> list[int]:
    encoded = tokenizer(
        rendered, add_special_tokens=False, padding=False, truncation=False, return_length=True
    )
    lengths = [int(length) for length in encoded["length"]]
    if len(lengths) != len(rendered):
        raise RuntimeError("Tokenizer returned the wrong number of sequence lengths")
    return lengths


def load_revisions() -> dict[str, Any]:
    revisions = read_json(REVISIONS_PATH)
    required = {
        "base_model",
        "benchmarks",
        "decontamination_reference",
        "source_dataset",
        "tokenizer",
    }
    missing = required - revisions.keys()
    if missing:
        raise ValueError(f"Missing revision entries: {sorted(missing)}")
    return revisions


def candidate_ids(candidates: list[Candidate]) -> list[str]:
    return [candidate.question_id for candidate in candidates]
