"""Shared Hugging Face Hub operations."""

from pathlib import Path

from huggingface_hub import CommitInfo, HfApi


def assert_revision(repo_id: str, revision: str, repo_type: str = "dataset") -> None:
    info = HfApi().repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
    if info.sha != revision:
        raise RuntimeError(f"{repo_id} resolved to {info.sha}, expected {revision}")


def list_files(repo_id: str, revision: str, repo_type: str = "dataset") -> list[str]:
    assert_revision(repo_id, revision, repo_type)
    return HfApi().list_repo_files(repo_id, repo_type=repo_type, revision=revision)


def ensure_private_dataset(api: HfApi, repo_id: str) -> None:
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.dataset_info(repo_id).private:
        raise RuntimeError(f"{repo_id} must remain private")


def upload_dataset_folder(
    api: HfApi, folder: Path, repo_id: str, commit_message: str, revision: str | None = None
) -> CommitInfo:
    return api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message=commit_message,
    )
