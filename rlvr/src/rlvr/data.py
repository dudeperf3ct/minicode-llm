"""Prepare a deterministic 1K KodCode prompt set for GRPO."""

import argparse
import ast
import shutil
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.data import Candidate, Stratum, batched, normalize_text, sha256_file, stable_hash
from common.decontamination import DecontaminationResult, NgramDecontaminator
from common.io import read_json, write_json, write_jsonl
from common.sampling import Allocation, select_stratified
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

PROJECT_DIR = Path(__file__).resolve().parents[2]
REVISIONS_PATH = PROJECT_DIR / "manifests" / "revisions.json"
SOURCE_COLUMNS = [
    "style",
    "subset",
    "question_id",
    "question",
    "test",
    "gpt_difficulty",
    "r1_correctness",
]
ALLOWED_TEST_MODULES = frozenset(sys.stdlib_module_names) | {"pytest", "solution"}


@dataclass(frozen=True)
class SourcePool:
    source: Dataset
    candidates: list[Candidate]
    counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=PROJECT_DIR / "manifests" / "data_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=1_000)
    parser.add_argument("--context-length", type=int, default=8_192)
    parser.add_argument("--max-completion-length", type=int, default=2_048)
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_revisions(path: Path = REVISIONS_PATH) -> dict[str, Any]:
    revisions = read_json(path)
    required = {
        "benchmarks",
        "decontamination_reference",
        "sft_dataset",
        "sft_private_test",
        "source_dataset",
        "starting_model",
        "tokenizer",
    }
    missing = required - revisions.keys()
    if missing:
        raise ValueError(f"Missing revision entries: {sorted(missing)}")
    return revisions


def load_sft_exclusions(revisions: dict[str, Any]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    questions: set[str] = set()

    public = revisions["sft_dataset"]
    for split in public["splits"]:
        rows = load_dataset(
            public["repo_id"], public["config"], revision=public["revision"], split=split
        ).select_columns(["id", "messages"])
        _add_exclusions(rows, ids, questions)

    private = revisions["sft_private_test"]
    rows = load_dataset(
        private["repo_id"], revision=private["revision"], split=private["split"]
    ).select_columns(["id", "messages"])
    _add_exclusions(rows, ids, questions)
    if len(ids) != 11_000:
        raise RuntimeError(f"Expected 11,000 unique SFT IDs, found {len(ids):,}")
    return ids, questions


def _add_exclusions(rows: Dataset, ids: set[str], questions: set[str]) -> None:
    for row in rows:
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise RuntimeError(f"SFT exclusion row has invalid messages: {row['id']}")
        question = messages[0]
        if not isinstance(question, dict) or question.get("role") != "user":
            raise RuntimeError(f"SFT exclusion row has invalid first message: {row['id']}")
        content = question.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"SFT exclusion row has an empty question: {row['id']}")
        ids.add(str(row["id"]))
        questions.add(normalize_text(content))


def load_source(revisions: dict[str, Any]) -> Dataset:
    source = revisions["source_dataset"]
    dataset = load_dataset(
        source["repo_id"],
        source["config"],
        revision=source["revision"],
        split=source["split"],
        columns=SOURCE_COLUMNS,
    )
    missing = set(SOURCE_COLUMNS) - set(dataset.column_names)
    if missing:
        raise RuntimeError(f"KodCode schema is missing columns: {sorted(missing)}")
    return dataset


def build_source_pool(
    source: Dataset, excluded_ids: set[str], excluded_questions: set[str], seed: int
) -> SourcePool:
    keepers: dict[str, Candidate] = {}
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter(source_examples=len(source))

    for source_index, row in enumerate(source):
        error = _validation_error(row)
        if error:
            counts[error] += 1
            continue

        question_id = row["question_id"].strip()
        if question_id in seen_ids:
            raise RuntimeError(f"Duplicate source question_id: {question_id}")
        seen_ids.add(question_id)
        normalized_question = normalize_text(row["question"])
        if question_id in excluded_ids or normalized_question in excluded_questions:
            counts["removed_sft_overlap"] += 1
            continue

        candidate = Candidate(
            question_id=question_id,
            source_index=source_index,
            stratum=Stratum(
                row["gpt_difficulty"].strip(), row["subset"].strip(), row["style"].strip()
            ),
            normalized_question=normalized_question,
        )
        existing = keepers.get(normalized_question)
        if existing is None:
            keepers[normalized_question] = candidate
            continue

        counts["removed_normalized_duplicate"] += 1
        if stable_hash(seed, "rlvr:deduplicate", candidate.question_id) < stable_hash(
            seed, "rlvr:deduplicate", existing.question_id
        ):
            keepers[normalized_question] = candidate

    candidates = sorted(keepers.values(), key=lambda item: item.source_index)
    counts["eligible_before_decontamination"] = len(candidates)
    return SourcePool(source=source, candidates=candidates, counts=dict(sorted(counts.items())))


def _validation_error(row: dict[str, Any]) -> str | None:
    if row["r1_correctness"] != "True":
        return "r1_correctness_not_true"
    for field in ("question_id", "question", "test", "gpt_difficulty", "subset", "style"):
        if not isinstance(row[field], str) or not row[field].strip():
            return f"missing_{field}"
    if row["style"].strip() != "instruct":
        return "unsupported_style"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tests = ast.parse(row["test"])
    except SyntaxError:
        return "invalid_test_syntax"
    imports = set()
    for node in ast.walk(tests):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    if imports - ALLOWED_TEST_MODULES:
        return "unsupported_test_dependency"
    return None


def load_tokenizer(revisions: dict[str, Any]) -> PreTrainedTokenizerBase:
    tokenizer = revisions["tokenizer"]
    return AutoTokenizer.from_pretrained(tokenizer["repo_id"], revision=tokenizer["revision"])


def filter_by_context(
    source: Dataset,
    candidates: list[Candidate],
    tokenizer: PreTrainedTokenizerBase,
    context_length: int,
    max_completion_length: int,
    batch_size: int,
) -> tuple[list[Candidate], dict[str, int], list[str]]:
    eligible: list[Candidate] = []
    prompt_lengths: dict[str, int] = {}
    removed: list[str] = []

    for batch in batched(candidates, batch_size):
        conversations = [
            [{"role": "user", "content": source[candidate.source_index]["question"]}]
            for candidate in batch
        ]
        rendered = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(rendered, str):
            rendered = [rendered]
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        lengths = [int(length) for length in encoded["length"]]
        if len(lengths) != len(batch):
            raise RuntimeError("Tokenizer returned the wrong number of prompt lengths")
        for candidate, length in zip(batch, lengths, strict=True):
            if length + max_completion_length > context_length:
                removed.append(candidate.question_id)
                continue
            eligible.append(candidate)
            prompt_lengths[candidate.question_id] = length
    return eligible, prompt_lengths, sorted(removed)


def write_training_data(
    source: Dataset,
    selected: list[Candidate],
    prompt_lengths: dict[str, int],
    allocation: Allocation,
    output_dir: Path,
) -> None:
    rows = []
    for candidate in selected:
        row = source[candidate.source_index]
        rows.append(
            {
                "gpt_difficulty": candidate.stratum.gpt_difficulty,
                "prompt": [{"role": "user", "content": row["question"]}],
                "prompt_tokens": prompt_lengths[candidate.question_id],
                "question_id": candidate.question_id,
                "style": candidate.stratum.style,
                "subset": candidate.stratum.subset,
                "test": row["test"],
            }
        )
    write_jsonl(output_dir / "train.jsonl", rows)
    write_json(
        output_dir / "selected_ids.json",
        {
            "allocation": [
                {
                    "gpt_difficulty": stratum.gpt_difficulty,
                    "selected": count,
                    "style": stratum.style,
                    "subset": stratum.subset,
                }
                for stratum, count in sorted(allocation.items())
            ],
            "ids": [candidate.question_id for candidate in selected],
        },
    )


def write_manifest(
    path: Path,
    output_dir: Path,
    revisions: dict[str, Any],
    args: argparse.Namespace,
    source_counts: dict[str, int],
    decontamination: DecontaminationResult,
    context_removed: list[str],
    eligible_count: int,
) -> None:
    files = {}
    for file_path in sorted(output_dir.rglob("*")):
        if file_path.is_file():
            relative = file_path.relative_to(PROJECT_DIR).as_posix()
            files[relative] = {
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
    write_json(
        path,
        {
            "command": {
                "context_length": args.context_length,
                "max_completion_length": args.max_completion_length,
                "ngram_size": args.ngram_size,
                "seed": args.seed,
                "train_size": args.train_size,
            },
            "counts": {
                "context_length_removed": len(context_removed),
                "decontamination": decontamination.counts,
                "eligible_for_selection": eligible_count,
                "selected": args.train_size,
                "source": source_counts,
            },
            "files": files,
            "revisions": revisions,
        },
    )


def reset_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is not empty; pass --overwrite to regenerate it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.train_size <= 0:
        raise ValueError("--train-size must be positive")
    if args.max_completion_length >= args.context_length:
        raise ValueError("--max-completion-length must be smaller than --context-length")

    output_dir = args.output_dir.resolve()
    reset_output_dir(output_dir, args.overwrite)
    revisions = load_revisions()
    print("Loading pinned SFT exclusions...", flush=True)
    excluded_ids, excluded_questions = load_sft_exclusions(revisions)
    print("Loading and filtering pinned KodCode RL data...", flush=True)
    pool = build_source_pool(load_source(revisions), excluded_ids, excluded_questions, args.seed)
    decontamination = NgramDecontaminator(revisions, args.ngram_size).run(
        pool.candidates, output_dir / "decontamination_report.jsonl"
    )
    print("Filtering prompts by the configured context budget...", flush=True)
    eligible, prompt_lengths, context_removed = filter_by_context(
        pool.source,
        decontamination.candidates,
        load_tokenizer(revisions),
        args.context_length,
        args.max_completion_length,
        args.tokenizer_batch_size,
    )
    if len(eligible) < args.train_size:
        raise RuntimeError(f"Only {len(eligible):,} examples remain; {args.train_size:,} required")

    selected, allocation = select_stratified(eligible, args.train_size, args.seed, "rlvr:train")
    write_training_data(pool.source, selected, prompt_lengths, allocation, output_dir)
    write_manifest(
        args.data_manifest.resolve(),
        output_dir,
        revisions,
        args,
        pool.counts,
        decontamination,
        context_removed,
        len(eligible),
    )
    print(f"Prepared {len(selected):,} RLVR prompts under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
