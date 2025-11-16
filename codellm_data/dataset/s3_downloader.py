"""Downloader for SWH datasets from S3."""

from pathlib import Path

from loguru import logger
from swh.datasets.download import DatasetDownloader


class SWHDatasetDownloader:
    """Download SWH datasets from S3 bucket."""

    @staticmethod
    def is_dataset_present(dataset_path: Path) -> bool:
        """Check if dataset is already downloaded.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            True if dataset exists and has content, False otherwise
        """
        if not dataset_path.exists():
            return False

        # Check if directory has parquet files
        return any(dataset_path.rglob("*.parquet"))

    @staticmethod
    def download_dataset(
        local_path: Path, s3_url: str, key_name: str, parallelism: int = 5
    ) -> None:
        """Download dataset from Software Archive S3 bucket.

        Args:
            local_path: Path to download the dataset locally
            s3_url: S3 url pointing to the dataset bucket.
                More information can be found here:
                https://docs.softwareheritage.org/devel/swh-export/graph/dataset.html
            key_name: Object key name in the S3 bucket
            parallelism: Number of threads to use for parallel download

        Raises:
            ValueError: If s3_url is invalid.
        """
        if not s3_url.startswith("s3://"):
            raise ValueError(f"s3 url is not a valid url, got {s3_url}")

        logger.info(f"Downloading dataset from {s3_url}/{key_name}")
        logger.info(f"Destination: {local_path}")
        logger.info(f"Parallelism: {parallelism}")

        DatasetDownloader(
            local_path=local_path, s3_url=s3_url, prefix=key_name, parallelism=parallelism
        )

        logger.info("Dataset download complete")

    @classmethod
    def ensure_dataset(
        cls,
        local_path: Path,
        s3_url: str,
        key_name: str,
        parallelism: int = 5,
        auto_download: bool = True,
    ) -> bool:
        """Ensure dataset is present, download if needed.

        Args:
            local_path: Path to download the dataset locally
            s3_url: S3 url pointing to the dataset bucket
            key_name: Object key name in the S3 bucket
            parallelism: Number of threads to use for parallel download
            auto_download: Whether to automatically download if not present

        Returns:
            True if dataset is ready to use, False otherwise

        Raises:
            ValueError: If s3_url is invalid
            FileNotFoundError: If dataset not found and auto_download is False
        """
        if cls.is_dataset_present(local_path):
            logger.info(f"Dataset already present at {local_path}")
            return True

        if not auto_download:
            raise FileNotFoundError(
                f"Dataset not found at {local_path} and auto_download is disabled"
            )

        logger.warning(f"Dataset not found at {local_path}")
        logger.info("Auto-download enabled, starting download...")

        # Create directory if it doesn't exist
        local_path.mkdir(parents=True, exist_ok=True)

        cls.download_dataset(
            local_path=local_path,
            s3_url=s3_url,
            key_name=key_name,
            parallelism=parallelism,
        )

        return True
