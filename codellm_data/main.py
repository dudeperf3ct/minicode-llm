"""Prepare dataset for CodeLLM training."""

import argparse
from pathlib import Path
from typing import NamedTuple

import polars as pl
import ray
from loguru import logger

from codellm_data.config.config import Config, load_config
from codellm_data.content.downloader import SWHContentDownloader
from codellm_data.dataset.parser import SWHDatasetParser
from codellm_data.utils.logging import setup_logging
from codellm_data.utils.timer import timer


class RepoInfo(NamedTuple):
    """Repository information for processing."""

    directory_id: bytes
    repo_url: str


@ray.remote
class SWHDatasetActor:
    """Ray actor that loads dataset once and processes multiple batches.

    This actor maintains state across method calls, avoiding redundant
    dataset loading for each batch of repositories.

    Note: Calling logging in Actor class results in pickling issue.
    """

    def __init__(self, dataset_path: Path):
        """Initialise actor with pre-loaded dataset.

        Args:
            dataset_path: Path to dataset
        """
        self.swh_dataset = SWHDatasetParser(data_path=dataset_path)
        self.swh_dataset.load_dataset()

    def process_batch(self, rows: list[RepoInfo]) -> list[dict]:
        """Process a batch of repositories using pre-loaded dataset.

        Args:
            rows: List of RepoInfo dataclass

        Returns:
            List of all files from all repos in batch
        """
        all_files = []
        for row in rows:
            files = self.swh_dataset.get_directory_files(
                directory_id=row.directory_id, repo_url=row.repo_url
            )
            all_files.extend(files)
        return all_files


def prepare_dataset(config: Config):
    """Download metadata and prepare data.

    Args:
        config: Application configuration
    """
    # Create directories if needed
    config.raw_dataset_path.mkdir(exist_ok=True, parents=True)
    config.processed_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Raw dataset path: {config.raw_dataset_path}")
    logger.info(f"Processed data path: {config.processed_path}")
    logger.info(f"Configuration: {config.model_dump()}")

    ray.init()  # pyrefly: ignore[missing-argument]

    logger.info("Loading dataset for preprocessing...")
    swh_dataset = SWHDatasetParser(data_path=config.raw_dataset_path, s3_config=config.s3_dataset)
    swh_dataset.load_dataset()
    directories = swh_dataset.get_main_revision()
    repos = swh_dataset.add_repo_urls(directories)

    # # Limit for testing
    # repos = repos[:50]

    # Prepare batches
    rows = [
        RepoInfo(directory_id=row["root_directory_id"], repo_url=row["repo_url"])
        for row in repos.iter_rows(named=True)
    ]
    batches = [
        rows[i : i + config.ray.batch_size] for i in range(0, len(rows), config.ray.batch_size)
    ]

    # Create Ray actor pool
    logger.info(f"Creating {len(batches)} actors for parallel processing...")
    actors = [
        SWHDatasetActor.remote(config.raw_dataset_path)  # type: ignore[attr-defined]
        for _ in batches
    ]

    # Process batches in parallel
    futures = [
        actor.process_batch.remote(batch) for actor, batch in zip(actors, batches, strict=True)
    ]

    # Collect results
    all_files_nested = ray.get(futures)

    # Flatten results
    all_files = []
    for files in all_files_nested:
        all_files.extend(files)

    ray.shutdown()  # pyrefly: ignore[missing-argument]

    logger.info(f"Total files collected: {len(all_files)}")

    # Save to parquet
    files_df = pl.DataFrame(all_files)
    files_df.write_parquet(config.file_metadata_file)
    logger.info(f"Saved repository files info to {config.file_metadata_file}")


def download_data(config: Config) -> None:
    """Download contents using Software Heritage API.

    Args:
        config: Application configuration.
    """
    # Create directories if needed
    config.content_path.mkdir(exist_ok=True, parents=True)
    config.processed_path.mkdir(exist_ok=True, parents=True)

    # Load and filter data
    if not config.file_metadata_file.exists():
        logger.warning(f"Metadata file not found: {config.file_metadata_file}")
        logger.info("Run prepare_dataset first to generate file metadata")
        return

    files_df = pl.read_parquet(config.file_metadata_file)

    # Convert binary sha1_git to hex string - required for API calls
    files_df = files_df.with_columns(
        pl.col("sha1_git").map_elements(lambda x: x.hex(), return_dtype=pl.String).alias("sha1_git")
    )

    # Build all filters (extensions + patterns)
    filters = [pl.col("name").str.ends_with(ext) for ext in config.processing.file_extensions] + [
        pl.col("name").str.contains(pattern) for pattern in config.processing.file_patterns
    ]

    if filters:
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter = combined_filter | f
        filtered_files = files_df.filter(combined_filter)
    else:
        filtered_files = files_df

    logger.info(f"Total files after filtering: {len(filtered_files)}")
    logger.info(f"Filtered by extensions: {config.processing.file_extensions}")
    logger.info(f"Filtered by patterns: {config.processing.file_patterns}")

    swh_downloader = SWHContentDownloader(config=config.swh_api)
    results = swh_downloader.download_batch_sync(
        files_df=filtered_files, max_files=config.swh_api.max_files, output_dir=config.content_path
    )

    results_dicts = []
    for r in results:
        result_dict = r._asdict()
        result_dict["status"] = r.status.value
        results_dicts.append(result_dict)
    results_df = pl.DataFrame(results_dicts)
    results_df.write_parquet(config.download_stats_file)
    logger.info(f"Download statistics saved: {config.download_stats_file}")


def main(args: argparse.Namespace) -> None:
    """Main entry point for the script."""
    config = load_config()
    setup_logging(config.log_path)

    if args.command == "parse":
        with timer("Parsing dataset"):
            prepare_dataset(config)

    elif args.command == "download":
        with timer("Downloading data"):
            download_data(config)


if __name__ == "__main__":
    config = load_config()

    parser = argparse.ArgumentParser(description="Prepare and download dataset for CodeLLM")
    parser.add_argument(
        "command",
        choices=["parse", "download"],
        help="Command to execute: 'parse' to parse dataset, 'download' to download content",
    )
    args = parser.parse_args()

    main(args)
