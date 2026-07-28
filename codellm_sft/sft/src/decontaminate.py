"""Remove benchmark-contaminated KodCode questions with word n-grams.

This module reproduces the simple Open-R1 decontamination strategy selected for
the experiment: load pinned HumanEval, full MBPP, and LiveCodeBench release_v5
prompts; normalize them; build word-level 8-gram indices; remove any KodCode
question sharing a benchmark n-gram; and write an auditable removal report.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfFileSystem

import pipeline_utils as utils
from hub_utils import assert_revision, list_files
from json_utils import compact_json, write_jsonl

type BenchmarkLookups = dict[str, dict[str, str]]


@dataclass(frozen=True)
class BenchmarkPrompt:
    benchmark: str
    task_id: str
    prompt: str


@dataclass(frozen=True)
class DecontaminationResult:
    candidates: list[utils.Candidate]
    counts: dict[str, int]
    benchmark_audit: dict[str, Any]
    short_benchmark_prompts: dict[str, int]


class NgramDecontaminator:
    """Own the pinned benchmark indices and apply one deterministic pass."""

    def __init__(self, revisions: dict[str, Any], ngram_size: int) -> None:
        self.revisions = revisions
        self.ngram_size = ngram_size

    def run(self, candidates: list[utils.Candidate], report_path: Path) -> DecontaminationResult:
        prompts, benchmark_audit = self._load_benchmark_prompts()
        lookups, short_prompts = self._build_lookups(prompts)
        clean_candidates, counts = self._filter_candidates(candidates, lookups, report_path)
        return DecontaminationResult(
            candidates=clean_candidates,
            counts=counts,
            benchmark_audit=benchmark_audit,
            short_benchmark_prompts=short_prompts,
        )

    def _load_benchmark_prompts(self) -> tuple[list[BenchmarkPrompt], dict[str, Any]]:
        prompts: list[BenchmarkPrompt] = []
        audit: dict[str, Any] = {}
        prompts.extend(self._load_humaneval(audit))
        prompts.extend(self._load_mbpp(audit))
        prompts.extend(self._load_livecodebench(audit))
        audit["normalized_prompt_snapshot_sha256"] = self._snapshot_hash(prompts)
        return prompts, audit

    def _load_humaneval(self, audit: dict[str, Any]) -> list[BenchmarkPrompt]:
        config = self.revisions["benchmarks"]["humaneval"]
        assert_revision(config["repo_id"], config["revision"])
        rows = load_dataset(
            config["repo_id"],
            config["config"],
            split="test",
            revision=config["revision"],
            columns=["task_id", "prompt"],
        )
        audit["humaneval"] = {"examples": len(rows)}
        return [BenchmarkPrompt("humaneval", str(row["task_id"]), row["prompt"]) for row in rows]

    def _load_mbpp(self, audit: dict[str, Any]) -> list[BenchmarkPrompt]:
        config = self.revisions["benchmarks"]["mbpp"]
        assert_revision(config["repo_id"], config["revision"])
        rows = load_dataset(
            config["repo_id"],
            config["config"],
            revision=config["revision"],
            columns=["task_id", "text"],
        )
        if not isinstance(rows, DatasetDict):
            raise TypeError("Expected MBPP to load as a DatasetDict")

        prompts = [
            BenchmarkPrompt("mbpp", f"{split}/{row['task_id']}", row["text"])
            for split in config["splits"]
            for row in rows[split]
        ]
        audit["mbpp"] = {"examples": len(prompts), "splits": config["splits"]}
        return prompts

    def _load_livecodebench(self, audit: dict[str, Any]) -> list[BenchmarkPrompt]:
        config = self.revisions["benchmarks"]["livecodebench"]
        assert_revision(config["repo_id"], config["revision"])
        mirror = config["materialized_mirror"]

        prefix = f"{config['config']}/"
        files = sorted(
            path
            for path in list_files(mirror["repo_id"], mirror["revision"])
            if path.startswith(prefix) and path.endswith(".parquet")
        )
        prompts: list[BenchmarkPrompt] = []
        filesystem = HfFileSystem()
        for file_path in files:
            hub_path = f"datasets/{mirror['repo_id']}@{mirror['revision']}/{file_path}"
            with filesystem.open(hub_path, "rb") as handle:
                table = pq.read_table(handle, columns=["question_id", "question_content"])
            prompts.extend(
                BenchmarkPrompt("livecodebench", str(row["question_id"]), row["question_content"])
                for row in table.to_pylist()
            )

        if len(prompts) != 880:
            raise RuntimeError(
                f"Expected 880 LiveCodeBench release_v5 prompts, found {len(prompts)}"
            )
        audit["livecodebench"] = {
            "examples": len(prompts),
            "files": files,
            "materialized_mirror": mirror,
            "official_revision": config["revision"],
        }
        return prompts

    def _build_lookups(
        self, prompts: list[BenchmarkPrompt]
    ) -> tuple[BenchmarkLookups, dict[str, int]]:
        lookups: BenchmarkLookups = defaultdict(dict)
        short_counts: Counter[str] = Counter()
        for prompt in prompts:
            ngrams = self._word_ngrams(prompt.prompt)
            if not ngrams:
                short_counts[prompt.benchmark] += 1
            for ngram in ngrams:
                benchmark_lookup = lookups[prompt.benchmark]
                existing_task = benchmark_lookup.get(ngram)
                # Reports need one deterministic representative task per match.
                if existing_task is None or prompt.task_id < existing_task:
                    benchmark_lookup[ngram] = prompt.task_id

        return dict(lookups), dict(sorted(short_counts.items()))

    def _filter_candidates(
        self, candidates: list[utils.Candidate], lookups: BenchmarkLookups, report_path: Path
    ) -> tuple[list[utils.Candidate], dict[str, int]]:
        clean: list[utils.Candidate] = []
        reports: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        benchmark_revisions = {
            name: value["revision"] for name, value in self.revisions["benchmarks"].items()
        }

        print(f"Running one-stage {self.ngram_size}-word n-gram decontamination...", flush=True)
        for candidate in candidates:
            ngrams = self._word_ngrams(candidate.normalized_question)
            if not ngrams:
                counts["source_questions_shorter_than_ngram"] += 1
            matches = self._find_matches(ngrams, lookups, benchmark_revisions)
            if matches:
                reports.append(self._report_row(candidate, matches))
                counts["removed_any_benchmark"] += 1
                for match in matches:
                    counts[f"removed_{match['benchmark']}"] += 1
            else:
                clean.append(candidate)

        reports.sort(key=lambda row: row["question_id"])
        write_jsonl(report_path, reports)
        counts["clean_after_decontamination"] = len(clean)
        return clean, dict(sorted(counts.items()))

    def _find_matches(
        self, ngrams: list[str], lookups: BenchmarkLookups, benchmark_revisions: dict[str, str]
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for benchmark in sorted(lookups):
            lookup = lookups[benchmark]
            matching_ngram = next((ngram for ngram in ngrams if ngram in lookup), None)
            if matching_ngram is None:
                continue
            matches.append(
                {
                    "benchmark": benchmark,
                    "benchmark_revision": benchmark_revisions[benchmark],
                    "benchmark_task_id": lookup[matching_ngram],
                    "matching_ngram": matching_ngram,
                }
            )
        return matches

    def _word_ngrams(self, text: str) -> list[str]:
        words = utils.normalize_text(text).split()
        return [
            " ".join(words[index : index + self.ngram_size])
            for index in range(len(words) - self.ngram_size + 1)
        ]

    @staticmethod
    def _report_row(candidate: utils.Candidate, matches: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "matches": matches,
            "normalized_question_sha256": utils.sha256_text(candidate.normalized_question),
            "question_id": candidate.question_id,
            "removed": True,
        }

    @staticmethod
    def _snapshot_hash(prompts: list[BenchmarkPrompt]) -> str:
        lines = [
            compact_json(
                {
                    "benchmark": prompt.benchmark,
                    "prompt": utils.normalize_text(prompt.prompt),
                    "task_id": prompt.task_id,
                }
            )
            for prompt in prompts
        ]
        return utils.sha256_text("\n".join(lines) + "\n")
