"""Upload the prepared training payload to a Hugging Face dataset repository.

The command verifies every local source file against the generated manifest,
then uploads direct/reasoning train, validation, pilot, and overfit JSONL plus
provenance. Untouched test questions, private tests, selected test IDs, and the
detailed decontamination report are intentionally never staged for upload.
"""

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from hub_utils import upload_dataset_folder
from json_utils import read_json
from pipeline_utils import PROJECT_DIR, verify_file

PAYLOAD_FILES = {
    "direct/train.jsonl": "data/kodcode_direct.jsonl",
    "direct/validation.jsonl": "data/validation/kodcode_direct.jsonl",
    "direct/pilot.jsonl": "data/subsets/train-1000/kodcode_direct.jsonl",
    "direct/overfit.jsonl": "data/subsets/train-32/kodcode_direct.jsonl",
    "reasoning/train.jsonl": "data/kodcode_reasoning.jsonl",
    "reasoning/validation.jsonl": "data/validation/kodcode_reasoning.jsonl",
    "reasoning/pilot.jsonl": "data/subsets/train-1000/kodcode_reasoning.jsonl",
    "reasoning/overfit.jsonl": "data/subsets/train-32/kodcode_reasoning.jsonl",
    "metadata/token_statistics.json": "data/token_statistics.json",
    "metadata/split_statistics.json": "data/split_statistics.json",
    "metadata/data_manifest.json": "manifests/data_manifest.json",
    "metadata/revisions.json": "manifests/revisions.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload prepared SFT data to an existing Hugging Face dataset repo."
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo, for example user/name.")
    parser.add_argument("--revision", default="main", help="Destination branch.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify and stage the payload without any Hub mutation.",
    )
    return parser.parse_args()


def verify_payload(manifest: dict[str, Any]) -> None:
    """Fail before network mutation when any generated file has changed."""

    manifest_files = manifest["files"]
    for source in PAYLOAD_FILES.values():
        path = PROJECT_DIR / source
        if source.startswith("data/"):
            expected = manifest_files.get(source)
            if expected is None:
                raise RuntimeError(f"{source} is absent from the data manifest")
            verify_file(path, expected, source)
        elif not path.is_file():
            raise FileNotFoundError(f"Missing upload source: {path}")


def stage_payload(staging_dir: Path, manifest: dict[str, Any]) -> None:
    for destination, source in PAYLOAD_FILES.items():
        target = staging_dir / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_DIR / source, target)
    (staging_dir / "README.md").write_text(dataset_card(manifest), encoding="utf-8")


def dataset_card(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    revisions = manifest["revisions"]
    source = revisions["source_dataset"]
    model = revisions["base_model"]
    return f"""---
license: cc-by-nc-4.0
pretty_name: Qwen3.5 KodCode matched SFT experiment
configs:
  - config_name: direct
    data_files:
      - split: train
        path: direct/train.jsonl
      - split: validation
        path: direct/validation.jsonl
      - split: pilot
        path: direct/pilot.jsonl
      - split: overfit
        path: direct/overfit.jsonl
  - config_name: reasoning
    data_files:
      - split: train
        path: reasoning/train.jsonl
      - split: validation
        path: reasoning/validation.jsonl
      - split: pilot
        path: reasoning/pilot.jsonl
      - split: overfit
        path: reasoning/overfit.jsonl
---

# Qwen3.5 KodCode matched SFT experiment

Prepared from [`{source["repo_id"]}`](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1) at revision `{source["revision"]}` for
`{model["repo_id"]}` at revision `{model["revision"]}`.

- Train: {counts["final_splits"]["train"]:,}
- Validation: {counts["final_splits"]["validation"]:,}
- Pilot: {counts["final_splits"]["pilot"]:,}
- Overfit: {counts["final_splits"]["overfit"]:,}
- Maximum rendered reasoning length: {manifest["command"]["sequence_length"]:,}
- Seed: {manifest["command"]["seed"]}

The `direct` and `reasoning` configurations contain exactly matched IDs in the same order. Only assistant targets differ.
"""  # noqa: E501


def main() -> None:
    args = parse_args()
    manifest = read_json(PROJECT_DIR / "manifests" / "data_manifest.json")
    verify_payload(manifest)

    with tempfile.TemporaryDirectory(prefix="codellm-sft-upload-") as temporary:
        staging_dir = Path(temporary)
        stage_payload(staging_dir, manifest)
        staged_bytes = sum(path.stat().st_size for path in staging_dir.rglob("*") if path.is_file())
        print(f"Verified and staged {len(PAYLOAD_FILES) + 1} files ({staged_bytes:,} bytes)")
        if args.dry_run:
            print("Dry run complete; no Hugging Face repository was changed.")
            return
        commit = upload_dataset_folder(
            HfApi(),
            staging_dir,
            args.repo_id,
            "Upload matched Qwen3.5 KodCode SFT data",
            args.revision,
        )
        print(f"Uploaded: {commit.commit_url}")
        print(f"Dataset revision: {commit.oid}")


if __name__ == "__main__":
    main()
