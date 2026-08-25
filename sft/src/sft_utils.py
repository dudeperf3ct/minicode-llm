"""SFT-specific paths, constants, and tokenizer helpers.

Generic data, I/O, Hub, sampling, and decontamination helpers live in the
repository-level ``common`` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from common.io import read_json

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
