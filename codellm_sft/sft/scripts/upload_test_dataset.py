"""Publish the untouched test questions and private tests to a private Hub dataset.

The local question and test files are verified against the data manifest, joined
by ID into one test split, and staged only in a temporary directory. The upload
is allowed only when the destination dataset repository is private. The
resulting immutable Hub commit is saved in manifests/evaluation.json for the
evaluation script.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

from hub_utils import ensure_private_dataset, upload_dataset_folder
from json_utils import read_json, write_json, write_jsonl
from pipeline_utils import PROJECT_DIR, verify_file

DEFAULT_REPO_ID = "dudeperf3ct/qwen35-kodcode-sft-private-test"
QUESTIONS_PATH = PROJECT_DIR / "data/test/questions.jsonl"
TESTS_PATH = PROJECT_DIR / "data/test/private_tests.jsonl"
EVALUATION_MANIFEST = PROJECT_DIR / "manifests/evaluation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_sources()
    rows = pair_test_rows()

    api = HfApi()
    ensure_private_dataset(api, args.repo_id)

    with tempfile.TemporaryDirectory(prefix="kodcode-private-test-") as directory:
        staging_dir = Path(directory)
        write_jsonl(staging_dir / "test.jsonl", rows)
        (staging_dir / "README.md").write_text(dataset_card(len(rows)), encoding="utf-8")
        commit = upload_dataset_folder(
            api,
            staging_dir,
            args.repo_id,
            "Upload pinned KodCode private test split",
        )

    evaluation = {
        "test_dataset": {"repo_id": args.repo_id, "revision": commit.oid, "split": "test"}
    }
    write_json(EVALUATION_MANIFEST, evaluation)
    print(f"Uploaded {len(rows)} test examples")
    print(f"Dataset revision: {commit.oid}")
    print(f"Saved: {EVALUATION_MANIFEST}")


def verify_sources() -> None:
    manifest_path = PROJECT_DIR / "manifests/data_manifest.json"
    manifest = read_json(manifest_path)
    for path in (QUESTIONS_PATH, TESTS_PATH):
        relative = str(path.relative_to(PROJECT_DIR))
        expected = manifest["files"][relative]
        verify_file(path, expected, relative)


def pair_test_rows() -> list[dict[str, Any]]:
    questions = load_dataset("json", data_files=str(QUESTIONS_PATH), split="train")
    private_tests = load_dataset("json", data_files=str(TESTS_PATH), split="train")
    tests = {row["id"]: row["test"] for row in private_tests}
    rows = [{**row, "test": tests[row["id"]]} for row in questions]
    if len(rows) != len(tests):
        raise ValueError("Question and private-test files contain different IDs")
    return rows


def dataset_card(examples: int) -> str:
    return f"""---
license: cc-by-nc-4.0
pretty_name: Qwen3.5 KodCode private test split
configs:
  - config_name: default
    data_files:
      - split: test
        path: test.jsonl
---

# Qwen3.5 KodCode private test split

This private repository contains the {examples} untouched test prompts and their
paired executable tests for the Qwen3.5 KodCode SFT experiment. It must not be
used for training, validation, or hyperparameter selection.
"""


if __name__ == "__main__":
    main()
