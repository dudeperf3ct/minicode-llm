"""Configuration loader for CodeLLM data processing pipeline."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent


class S3DatasetConfig(BaseModel):
    """S3 dataset download configuration."""

    s3_url: str = Field(
        default="s3://softwareheritage/graph/2019-01-28-popular-3k-python/parquet/",
        description="S3 URL for SWH dataset",
    )
    dataset_name: str = Field(
        default="2019-01-28-popular-3k-python", description="Name of data folder"
    )
    parallelism: int = Field(default=5, description="Number of parallel download threads")
    auto_download: bool = Field(
        default=True, description="Automatically download dataset if not present"
    )


class RayConfig(BaseModel):
    """Ray processing configuration."""

    batch_size: int = Field(default=4, description="Number of repositories per batch")


class SWHAPIConfig(BaseModel):
    """Software Heritage API configuration."""

    base_url: str = Field(
        default="https://archive.softwareheritage.org/api/1",
        description="Base URL for SWH API",
    )
    max_concurrent: int = Field(default=10, description="Maximum concurrent HTTP connections")
    requests_per_hour: int = Field(
        default=120,
        description=(
            "Rate limit: maximum API requests per hour "
            "(120 anonymous, 1200 with authentication per SWH docs)"
        ),
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    bearer_token: str | None = Field(
        default_factory=lambda: os.getenv("SWH_BEARER_TOKEN"),
        description="Bearer token for authentication",
    )
    max_files: int | None = Field(
        default=None, description="Maximum number of files to download from API."
    )

    @property
    def headers(self) -> dict:
        """Get headers with authentication."""
        headers = {"Accept": "application/json", "User-Agent": "SWH-Dataset-Downloader/1.0"}

        if self.bearer_token:
            headers["Authorisation"] = f"Bearer {self.bearer_token}"

        return headers


class ProcessingConfig(BaseModel):
    """Data processing configuration."""

    file_extensions: list[str] = Field(
        default=[".py"], description="File extensions to filter (e.g., '.py', '.js')"
    )
    file_patterns: list[str] = Field(
        default=[],
        description="Regex patterns to match filenames (e.g., '(?i)^license' for LICENSE files)",
    )


class Config(BaseModel):
    """Main configuration model."""

    # Path strings from YAML (relative or absolute)
    data_dir: str = Field(default="data")

    # Configuration sections
    s3_dataset: S3DatasetConfig = Field(default_factory=S3DatasetConfig)
    ray: RayConfig = Field(default_factory=RayConfig)
    swh_api: SWHAPIConfig = Field(default_factory=SWHAPIConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    def get_path(self, path_str: str) -> Path:
        """Convert path string to absolute Path.

        Args:
            path_str: Path string from config (relative or absolute)

        Returns:
            Absolute Path object
        """
        path = Path(path_str)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def data_path(self) -> Path:
        """Get absolute data directory path."""
        return self.get_path(self.data_dir)

    @property
    def raw_dataset_path(self) -> Path:
        """Get path to raw dataset directory."""
        return self.data_path / "raw" / self.s3_dataset.dataset_name

    @property
    def processed_path(self) -> Path:
        """Get path to processed data directory."""
        return self.data_path / "processed"

    @property
    def content_path(self) -> Path:
        """Get path to downloaded content directory."""
        return self.data_path / "content"

    @property
    def log_path(self) -> Path:
        """Get path to logs directory."""
        return self.data_path / "logs"

    @property
    def file_metadata_file(self) -> Path:
        """Get path to file_metadata.parquet file."""
        return self.processed_path / "file_metadata.parquet"

    @property
    def download_stats_file(self) -> Path:
        """Get path to download_stats.parquet file."""
        return self.processed_path / "download_stats.parquet"


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file (default: config/config.yaml)

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / "codellm_data" / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        config_dict = yaml.safe_load(f) or {}

    return Config(**config_dict)
