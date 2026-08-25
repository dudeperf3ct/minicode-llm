"""Core source filtering, sampling, and rendering operations.

This module transforms pinned KodCode train rows into one matched candidate
pool, filters reasoning sequences with the Qwen chat template, selects the
jointly stratified train/validation/test and nested subsets, and validates
split invariants. File serialization lives in ``data_writer``.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

from common.data import Candidate, Stratum, batched, candidate_ids, normalize_text, stable_hash
from common.hub import list_files
from common.sampling import Allocation, SplitMap
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerBase

import sft_utils


@dataclass(frozen=True)
class SourcePool:
    """Validated source rows and their deduplicated candidate index."""

    source: Dataset
    candidates: list[Candidate]
    counts: dict[str, int]
    duplicate_ids: list[str]


@dataclass(frozen=True)
class SplitSelection:
    """All final and nested splits plus their allocation audit data."""

    train: list[Candidate]
    validation: list[Candidate]
    test: list[Candidate]
    pilot: list[Candidate]
    overfit: list[Candidate]
    selected_allocation: Allocation
    split_allocations: dict[str, Allocation]
    pilot_allocation: Allocation
    overfit_allocation: Allocation

    def all_named_splits(self) -> SplitMap:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "pilot": self.pilot,
            "overfit": self.overfit,
        }


def load_and_deduplicate_source(revisions: dict[str, Any], seed: int) -> SourcePool:
    """Load only pinned train files, validate rows, and deduplicate questions."""

    source_revision = revisions["source_dataset"]
    train_files = _download_train_files(source_revision)
    source = _load_train_dataset(train_files, source_revision["split"])

    keepers: dict[str, Candidate] = {}
    seen_ids: set[str] = set()
    removed_duplicate_ids: list[str] = []
    counts: Counter[str] = Counter(
        source_examples=len(source), source_train_parquet_files=len(train_files)
    )

    print("Validating and deduplicating source rows...", flush=True)
    for source_index, row in enumerate(source):
        error = _source_validation_error(row)
        if error:
            counts[error] += 1
            continue
        candidate = _candidate_from_row(row, source_index)
        if candidate.question_id in seen_ids:
            raise RuntimeError(f"Duplicate source question_id: {candidate.question_id}")
        seen_ids.add(candidate.question_id)
        _keep_best_duplicate(keepers, removed_duplicate_ids, candidate, seed)

    candidates = sorted(keepers.values(), key=lambda item: item.source_index)
    counts["structurally_valid"] = len(candidates) + len(removed_duplicate_ids)
    counts["deduplicated_questions"] = len(candidates)
    counts["duplicates_removed"] = len(removed_duplicate_ids)
    return SourcePool(
        source=source,
        candidates=candidates,
        counts=dict(sorted(counts.items())),
        duplicate_ids=sorted(removed_duplicate_ids),
    )


def filter_by_reasoning_length(
    source: Dataset,
    candidates: list[Candidate],
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int,
    batch_size: int,
) -> tuple[list[Candidate], list[str]]:
    """Remove IDs whose fully rendered reasoning conversation is too long."""

    eligible: list[Candidate] = []
    removed_ids: list[str] = []
    processed = 0
    next_report = 25_000

    print("Rendering and length-filtering reasoning conversations...", flush=True)
    for batch in batched(candidates, batch_size):
        rows = [source[candidate.source_index] for candidate in batch]
        conversations = [reasoning_messages(row) for row in rows]
        lengths = sft_utils.token_lengths(
            tokenizer, sft_utils.render_batch(tokenizer, conversations)
        )
        for candidate, length in zip(batch, lengths, strict=True):
            if length > sequence_length:
                removed_ids.append(candidate.question_id)
                continue
            eligible.append(candidate)

        processed += len(batch)
        if processed >= next_report or processed == len(candidates):
            print(f"Length-filtered {processed:,}/{len(candidates):,} rows", flush=True)
            next_report += 25_000
    return eligible, sorted(removed_ids)


def assert_prepared_data(
    splits: dict[str, list[Candidate]],
    pilot: list[Candidate],
    overfit: list[Candidate],
    train_size: int,
    validation_size: int,
    test_size: int,
) -> None:
    expected = {"train": train_size, "validation": validation_size, "test": test_size}
    for split, size in expected.items():
        if len(splits[split]) != size:
            raise RuntimeError(f"{split} has {len(splits[split])} rows, expected {size}")

    split_sets = {name: set(candidate_ids(items)) for name, items in splits.items()}
    all_ids = [
        candidate.question_id
        for split in ("train", "validation", "test")
        for candidate in splits[split]
    ]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Final split IDs are not unique")
    if not set(candidate_ids(pilot)) <= split_sets["train"]:
        raise RuntimeError("Pilot is not nested within train")
    if not set(candidate_ids(overfit)) <= set(candidate_ids(pilot)):
        raise RuntimeError("Overfit is not nested within pilot")

    normalized = [candidate.normalized_question for items in splits.values() for candidate in items]
    if len(normalized) != len(set(normalized)):
        raise RuntimeError("Normalized questions are not unique across final splits")


def _download_train_files(source_revision: dict[str, Any]) -> list[str]:
    train_files = sorted(
        path
        for path in list_files(source_revision["repo_id"], source_revision["revision"])
        if path.startswith("data/train-") and path.endswith(".parquet")
    )
    if len(train_files) != 11:
        raise RuntimeError(f"Expected 11 pinned KodCode train files, found {len(train_files)}")
    return [
        hf_hub_download(
            repo_id=source_revision["repo_id"],
            filename=file_path,
            repo_type="dataset",
            revision=source_revision["revision"],
        )
        for file_path in train_files
    ]


def _load_train_dataset(train_files: list[str], split: str) -> Dataset:
    print("Loading only the 11 pinned KodCode train Parquet files...", flush=True)
    source = load_dataset(
        "parquet",
        data_files={"train": train_files},
        split=split,
        columns=sft_utils.SOURCE_COLUMNS,
    )
    missing_columns = set(sft_utils.SOURCE_COLUMNS) - set(source.column_names)
    if missing_columns:
        raise RuntimeError(f"KodCode schema is missing columns: {sorted(missing_columns)}")
    return source


def _source_validation_error(row: dict[str, Any]) -> str | None:
    if row["r1_correctness"] != "True":
        return "r1_correctness_not_true"
    for field in (
        "question_id",
        "question",
        "r1_solution",
        *sft_utils.STRATIFICATION_FIELDS,
    ):
        if not isinstance(row[field], str) or not row[field].strip():
            return f"missing_{field}"

    conversations = row["conversations"]
    if not isinstance(conversations, list) or len(conversations) != 2:
        return "invalid_conversation_length"
    first, second = conversations
    if not isinstance(first, dict) or not isinstance(second, dict):
        return "invalid_conversation_items"
    if first.get("from") != "human" or second.get("from") != "gpt":
        return "invalid_conversation_roles"
    if not isinstance(first.get("value"), str) or not first["value"].strip():
        return "missing_conversation_question"
    if not isinstance(second.get("value"), str) or not second["value"].strip():
        return "missing_conversation_assistant"
    if _malformed_reasoning(second["value"]):
        return "malformed_reasoning"
    if "<think>" in row["r1_solution"] or "</think>" in row["r1_solution"]:
        return "reasoning_in_direct_solution"
    return None


def _malformed_reasoning(reasoning: str) -> bool:
    reasoning = reasoning.strip()
    return (
        not reasoning.startswith("<think>")
        or reasoning.count("<think>") != 1
        or reasoning.count("</think>") != 1
        or not reasoning.split("</think>", 1)[1].strip()
    )


def _candidate_from_row(row: dict[str, Any], source_index: int) -> Candidate:
    return Candidate(
        question_id=row["question_id"].strip(),
        source_index=source_index,
        stratum=Stratum(*(row[field].strip() for field in sft_utils.STRATIFICATION_FIELDS)),
        normalized_question=normalize_text(row["question"]),
    )


def _keep_best_duplicate(
    keepers: dict[str, Candidate],
    removed_ids: list[str],
    candidate: Candidate,
    seed: int,
) -> None:
    existing = keepers.get(candidate.normalized_question)
    if existing is None:
        keepers[candidate.normalized_question] = candidate
        return

    candidate_rank = stable_hash(seed, "deduplicate", candidate.question_id)
    existing_rank = stable_hash(seed, "deduplicate", existing.question_id)
    if candidate_rank < existing_rank:
        removed_ids.append(existing.question_id)
        keepers[candidate.normalized_question] = candidate
    else:
        removed_ids.append(candidate.question_id)


def direct_messages(row: dict[str, Any]) -> sft_utils.Messages:
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["r1_solution"]},
    ]


def reasoning_messages(row: dict[str, Any]) -> sft_utils.Messages:
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["conversations"][1]["value"]},
    ]
