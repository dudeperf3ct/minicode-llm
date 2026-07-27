"""Write deterministic selection, statistics, and provenance reports.

Prepared JSONL is intentionally separate from its audit trail. This module
serializes ordered IDs, stratum allocations, filtering counts, source/model
revisions, tokenizer-template identity, package versions, and checksums without
mixing reporting concerns into the preparation command.
"""

from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

import data_pipeline
import data_statistics
import data_writer
import pipeline_utils as utils
from transformers import PreTrainedTokenizerBase


def write_selected_ids(
    *,
    output_dir: Path,
    revisions: dict[str, Any],
    candidates: list[utils.Candidate],
    selection: data_pipeline.SplitSelection,
    duplicate_ids: list[str],
    overlength_ids: list[str],
    seed: int,
) -> None:
    allocations = {
        "selected": selection.selected_allocation,
        **selection.split_allocations,
    }
    ordered_ids = {
        "overfit_32": utils.candidate_ids(selection.overfit),
        "pilot_1000": utils.candidate_ids(selection.pilot),
        "test": utils.candidate_ids(selection.test),
        "train": utils.candidate_ids(selection.train),
        "validation": utils.candidate_ids(selection.validation),
    }
    data_writer.write_json(
        output_dir / "selected_ids.json",
        {
            "allocations": _serialize_allocation(_stratum_capacities(candidates), allocations),
            "deduplication_removed_ids": duplicate_ids,
            "normalization": "lowercase, strip, collapse all whitespace",
            "ordered_ids": ordered_ids,
            "overlength_removed_ids": overlength_ids,
            "revisions": revisions,
            "seed": seed,
            "stratification_fields": list(utils.STRATIFICATION_FIELDS),
        },
    )


def write_split_statistics(
    *,
    output_dir: Path,
    selection: data_pipeline.SplitSelection,
    benchmark_audit: dict[str, Any],
    short_benchmark_prompts: dict[str, int],
    decontamination_counts: dict[str, int],
    source_counts: dict[str, int],
    ngram_size: int,
) -> None:
    named_splits = selection.all_named_splits()
    split_statistics = {
        split: {"examples": len(items), "strata": data_statistics.stratum_counts(items)}
        for split, items in named_splits.items()
    }
    nested_allocations = {
        "overfit": _serialize_allocation(
            _stratum_capacities(selection.pilot), {"overfit": selection.overfit_allocation}
        ),
        "pilot": _serialize_allocation(
            _stratum_capacities(selection.train), {"pilot": selection.pilot_allocation}
        ),
    }
    data_writer.write_json(
        output_dir / "split_statistics.json",
        {
            "benchmark_audit": benchmark_audit,
            "benchmark_prompts_shorter_than_ngram": short_benchmark_prompts,
            "decontamination": decontamination_counts,
            "ngram_size": ngram_size,
            "source_filtering": source_counts,
            "splits": split_statistics,
            "stratified_allocations": nested_allocations,
        },
    )


def write_data_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    revisions: dict[str, Any],
    source_counts: dict[str, int],
    decontamination_counts: dict[str, int],
    candidates: list[utils.Candidate],
    selection: data_pipeline.SplitSelection,
    overlength_ids: list[str],
    tokenizer: PreTrainedTokenizerBase,
    command: dict[str, int],
) -> None:
    generated_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    counts = {
        "decontamination": decontamination_counts,
        "eligible_after_length_filter": len(candidates),
        "final_splits": {
            split: len(items) for split, items in selection.all_named_splits().items()
        },
        "overlength_removed": len(overlength_ids),
        "source_filtering": source_counts,
    }
    files = {
        str(Path("data") / path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": utils.sha256_file(path),
        }
        for path in generated_files
    }
    packages = {
        package: metadata.version(package)
        for package in ("datasets", "jinja2", "pyarrow", "transformers")
    }
    data_writer.write_json(
        manifest_path,
        {
            "command": command,
            "counts": counts,
            "files": files,
            "packages": packages,
            "revisions": revisions,
            "tokenizer_chat_template_sha256": utils.sha256_text(tokenizer.chat_template),
        },
    )


def _stratum_capacities(candidates: list[utils.Candidate]) -> data_pipeline.Allocation:
    return dict(Counter(candidate.stratum for candidate in candidates))


def _serialize_allocation(
    capacities: data_pipeline.Allocation, allocations: dict[str, data_pipeline.Allocation]
) -> list[dict[str, Any]]:
    rows = []
    for stratum in sorted(capacities):
        row: dict[str, Any] = {
            "eligible": capacities[stratum],
            "gpt_difficulty": stratum.gpt_difficulty,
            "subset": stratum.subset,
            "style": stratum.style,
        }
        for name, allocation in allocations.items():
            row[name] = allocation.get(stratum, 0)
        rows.append(row)
    return rows
