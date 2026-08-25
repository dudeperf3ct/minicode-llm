"""Common models for API."""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple


class DownloadStatus(Enum):
    """Download status types."""

    SUCCESS = "success"
    SKIPPED = "skipped_already_exists"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    MAX_RETRIES = "max_retries_exceeded"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class FileMetadata:
    """Metadata for the files."""

    sha1_git: str
    repo_url: str
    file_path: str

    @classmethod
    def from_dict(cls, data: dict) -> "FileMetadata":
        """Create from dictionary."""
        return cls(
            sha1_git=data["sha1_git"],
            repo_url=data.get("repo_url", "unknown"),
            file_path=data.get("path", "unknown"),
        )


class DownloadResult(NamedTuple):
    """Download data."""

    sha1_git: str
    local_path: str | None
    success: bool
    size: int
    status: DownloadStatus
    error_message: str | None = None
