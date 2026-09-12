"""Verify and upload the prepared RLVR dataset to Hugging Face."""

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

from common.data import verify_file
from common.hub import upload_dataset_folder
from common.io import read_json
from huggingface_hub import HfApi

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "dudeperf3ct/qwen35-kodcode-rlvr-data"
DEFAULT_REVISION = "main"
PAYLOAD_FILES = {
    "train.jsonl": "data/train.jsonl",
    "metadata/data_manifest.json": "manifests/data_manifest.json",
    "metadata/decontamination_report.jsonl": "data/decontamination_report.jsonl",
    "metadata/revisions.json": "manifests/revisions.json",
    "metadata/selected_ids.json": "data/selected_ids.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def verify_payload(manifest: dict[str, Any]) -> None:
    for source in PAYLOAD_FILES.values():
        path = PROJECT_DIR / source
        if source.startswith("data/"):
            expected = manifest["files"].get(source)
            if expected is None:
                raise RuntimeError(f"{source} is absent from the data manifest")
            verify_file(path, expected, source)
        elif not path.is_file():
            raise FileNotFoundError(f"Missing upload source: {path}")


def stage_payload(path: Path, manifest: dict[str, Any]) -> None:
    for destination, source in PAYLOAD_FILES.items():
        target = path / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_DIR / source, target)
    (path / "README.md").write_text(dataset_card(manifest), encoding="utf-8")


def dataset_card(manifest: dict[str, Any]) -> str:
    source = manifest["revisions"]["source_dataset"]
    return f"""---
license: cc-by-nc-4.0
pretty_name: Qwen3.5 KodCode RLVR 1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
---

# Qwen3.5 KodCode RLVR 1K

Deterministically sampled from `{source["repo_id"]}` at revision
`{source["revision"]}` after removing SFT and public benchmark overlap.

- Prompts: {manifest["counts"]["selected"]:,}
- Seed: {manifest["command"]["seed"]}
- Maximum completion length: {manifest["command"]["max_completion_length"]:,}

The payload contains prompts, metadata, and public tests. It does not contain reference solutions.
"""


def main() -> None:
    args = parse_args()
    manifest_path = PROJECT_DIR / "manifests" / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; run `uv run rlvr-prepare` first")
    manifest = read_json(manifest_path)
    verify_payload(manifest)

    with tempfile.TemporaryDirectory(prefix="rlvr-upload-") as temporary:
        staging_dir = Path(temporary)
        stage_payload(staging_dir, manifest)
        staged_bytes = sum(path.stat().st_size for path in staging_dir.rglob("*") if path.is_file())
        print(f"Verified and staged {len(PAYLOAD_FILES) + 1} files ({staged_bytes:,} bytes)")
        if args.dry_run:
            print("Dry run complete; no Hugging Face repository was changed.")
            return

        api = HfApi()
        api.create_repo(DEFAULT_REPO_ID, repo_type="dataset", exist_ok=True)
        commit = upload_dataset_folder(
            api, staging_dir, DEFAULT_REPO_ID, "Upload Qwen3.5 KodCode RLVR data", DEFAULT_REVISION
        )
        print(f"Uploaded: {commit.commit_url}")
        print(f"Dataset revision: {commit.oid}")


if __name__ == "__main__":
    main()
