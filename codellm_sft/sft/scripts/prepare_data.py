"""Build the reusable matched KodCode dataset for every SFT phase.

The command loads only pinned verified train files, validates and deduplicates
questions, removes benchmark overlaps, filters reasoning examples with the
Qwen3.5 chat template, performs deterministic joint-stratified selection,
writes matched direct/reasoning data, isolates the untouched test material, and
records complete statistics and checksums.
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import data_pipeline
import data_writer
import decontaminate
import pipeline_reports
import pipeline_utils as utils
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass(frozen=True)
class PreparedSource:
    """Filtered source data and metadata needed to write final splits."""

    source: Dataset
    candidates: list[utils.Candidate]
    source_counts: dict[str, int]
    duplicate_ids: list[str]
    decontamination: decontaminate.DecontaminationResult
    overlength_ids: list[str]
    tokenizer: PreTrainedTokenizerBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare matched direct and reasoning KodCode SFT datasets."
    )
    parser.add_argument("--output-dir", type=Path, default=utils.PROJECT_DIR / "data")
    parser.add_argument(
        "--data-manifest", type=Path, default=utils.PROJECT_DIR / "manifests" / "data_manifest.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=16_384)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--pilot-size", type=int, default=1_000)
    parser.add_argument("--overfit-size", type=int, default=32)
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--tokenizer-batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_source(
    args: argparse.Namespace, revisions: dict[str, Any], output_dir: Path
) -> PreparedSource:
    """Run filtering stages that must occur before deterministic selection."""

    source_pool = data_pipeline.load_and_deduplicate_source(revisions, args.seed)
    decontamination = decontaminate.NgramDecontaminator(revisions, args.ngram_size).run(
        source_pool.candidates, output_dir / "decontamination_report.jsonl"
    )
    tokenizer = load_tokenizer(revisions)
    candidates, overlength_ids = data_pipeline.filter_by_reasoning_length(
        source=source_pool.source,
        candidates=decontamination.candidates,
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        batch_size=args.tokenizer_batch_size,
    )
    required = args.train_size + args.validation_size + args.test_size
    if len(candidates) < required:
        raise RuntimeError(f"Only {len(candidates)} examples remain; {required} are required")
    return PreparedSource(
        source=source_pool.source,
        candidates=candidates,
        source_counts=source_pool.counts,
        duplicate_ids=source_pool.duplicate_ids,
        decontamination=decontamination,
        overlength_ids=overlength_ids,
        tokenizer=tokenizer,
    )


def load_tokenizer(revisions: dict[str, Any]) -> PreTrainedTokenizerBase:
    tokenizer_revision = revisions["tokenizer"]
    print("Loading pinned Qwen3.5 tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_revision["repo_id"], revision=tokenizer_revision["revision"]
    )
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned tokenizer has no chat template")
    return tokenizer


def build_split_selection(
    candidates: list[utils.Candidate], args: argparse.Namespace
) -> data_pipeline.SplitSelection:
    total_size = args.train_size + args.validation_size + args.test_size
    selected, selected_allocation = data_pipeline.select_stratified(
        candidates=candidates, size=total_size, seed=args.seed, namespace="main"
    )
    splits, split_allocations = data_pipeline.partition_selected(
        selected=selected,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    pilot, pilot_allocation = data_pipeline.select_stratified(
        candidates=splits["train"], size=args.pilot_size, seed=args.seed, namespace="pilot"
    )
    overfit, overfit_allocation = data_pipeline.select_stratified(
        candidates=pilot, size=args.overfit_size, seed=args.seed, namespace="overfit"
    )
    data_pipeline.assert_prepared_data(
        splits=splits,
        pilot=pilot,
        overfit=overfit,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    return data_pipeline.SplitSelection(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        pilot=pilot,
        overfit=overfit,
        selected_allocation=selected_allocation,
        split_allocations=split_allocations,
        pilot_allocation=pilot_allocation,
        overfit_allocation=overfit_allocation,
    )


def manifest_parameters(args: argparse.Namespace) -> dict[str, int]:
    return {
        "ngram_size": args.ngram_size,
        "overfit_size": args.overfit_size,
        "pilot_size": args.pilot_size,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "test_size": args.test_size,
        "train_size": args.train_size,
        "validation_size": args.validation_size,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.pilot_size > args.train_size or args.overfit_size > args.pilot_size:
        raise ValueError("Nested subset sizes are inconsistent")


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to regenerate it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    reset_output_dir(output_dir, args.overwrite)

    revisions = utils.load_revisions()
    prepared = prepare_source(args, revisions, output_dir)
    selection = build_split_selection(prepared.candidates, args)
    data_writer.write_training_datasets(
        source=prepared.source,
        selection=selection,
        tokenizer=prepared.tokenizer,
        output_dir=output_dir,
        batch_size=args.tokenizer_batch_size,
        sequence_length=args.sequence_length,
        overlength_count=len(prepared.overlength_ids),
    )
    pipeline_reports.write_selected_ids(
        output_dir=output_dir,
        revisions=revisions,
        candidates=prepared.candidates,
        selection=selection,
        duplicate_ids=prepared.duplicate_ids,
        overlength_ids=prepared.overlength_ids,
        seed=args.seed,
    )
    pipeline_reports.write_split_statistics(
        output_dir=output_dir,
        selection=selection,
        benchmark_audit=prepared.decontamination.benchmark_audit,
        short_benchmark_prompts=prepared.decontamination.short_benchmark_prompts,
        decontamination_counts=prepared.decontamination.counts,
        source_counts=prepared.source_counts,
        ngram_size=args.ngram_size,
    )
    pipeline_reports.write_data_manifest(
        manifest_path=args.data_manifest.resolve(),
        output_dir=output_dir,
        revisions=revisions,
        source_counts=prepared.source_counts,
        decontamination_counts=prepared.decontamination.counts,
        candidates=prepared.candidates,
        selection=selection,
        overlength_ids=prepared.overlength_ids,
        tokenizer=prepared.tokenizer,
        command=manifest_parameters(args),
    )

    total_size = args.train_size + args.validation_size + args.test_size
    print(f"Prepared {total_size} matched examples under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
