"""Parse SWH dataset from local storage."""

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from codellm_data.config.config import S3DatasetConfig
from codellm_data.dataset.s3_downloader import SWHDatasetDownloader
from codellm_data.dataset.schema import SWHSchema


class SWHDatasetParser(SWHDatasetDownloader):
    """Parse SWH dataset from local parquet files.

    Database schema and relations: https://docs.softwareheritage.org/_images/db-schema.svg
    """

    def __init__(self, data_path: Path, s3_config: S3DatasetConfig | None = None):
        """Init.

        Args:
            data_path: Path to the dataset directory
            s3_config: S3DatasetConfig for auto-download
        """
        self.data_path = data_path
        self.s3_config = s3_config
        self.schema: SWHSchema = SWHSchema()
        self._tables: dict[str, pl.LazyFrame] = {}

        # Auto-download dataset if configured
        if s3_config and s3_config.auto_download:
            self.ensure_dataset_available()

    def ensure_dataset_available(self) -> None:
        """Ensure dataset is downloaded and ready to use."""
        if not self.s3_config:
            logger.warning("No S3 config provided, skipping auto-download check")
            return

        SWHDatasetDownloader.ensure_dataset(
            local_path=self.data_path,
            s3_url=self.s3_config.s3_url,
            key_name=self.s3_config.dataset_name,
            parallelism=self.s3_config.parallelism,
            auto_download=self.s3_config.auto_download,
        )

    def read_parquet_files(self, table_name: str) -> pl.LazyFrame:
        """Read parquet files for a specific table.

        Args:
            table_name: Name of the table (e.g., 'origin', 'content')

        Returns:
            Polars LazyFrame for efficient processing
        """
        table_path = self.data_path / table_name
        if not table_path.exists():
            raise FileNotFoundError(f"Table directory not found: {table_path}")

        logger.info(f"Reading table: {table_name}")
        return pl.scan_parquet(table_path / "*.parquet")

    def load_dataset(self):
        """Load all tables from the dataset."""
        logger.info("Loading SWH dataset tables")

        tables = {}
        for table_name in [
            self.schema.origin,
            self.schema.snapshot_branch,
            self.schema.revision,
            self.schema.directory,
            self.schema.directory_entry_file,
            self.schema.directory_entry_dir,
            self.schema.directory_entry_rev,
            self.schema.content,
            self.schema.skipped_content,
        ]:
            try:
                tables[table_name] = self.read_parquet_files(table_name)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {table_name}: {e}")

        self._tables = tables
        logger.info(f"Loaded {len(tables)} tables")

    def get_main_revision(self) -> pl.DataFrame:
        """Extract main/master branch revisions for each origin.

        Returns:
            DataFrame with columns: origin_id, revision_id, root_directory_id
        """
        logger.info("Filter main/master branches only")

        # Branch names in snapshot_branch use full git refs format
        refs_main = b"refs/heads/main"
        refs_master = b"refs/heads/master"

        # Filter for main/master branches that point to revisions
        branch_heads = (
            self._tables[self.schema.snapshot_branch]
            .filter((pl.col("name") == refs_main) | (pl.col("name") == refs_master))
            .filter(pl.col("target_type") == "revision")
            .select(
                [
                    pl.col("object_id").alias("snapshot_id"),
                    pl.col("name").alias("branch_name"),
                    pl.col("target").alias("revision_id"),
                ]
            )
        )

        # Join with revisions to get directory info
        heads_with_metadata = (
            branch_heads.join(
                self._tables[self.schema.revision].select(
                    [
                        pl.col("id").alias("revision_id"),
                        pl.col("directory").alias("root_directory_id"),
                    ]
                ),
                on="revision_id",
                how="inner",
            )
            .select(
                [
                    "snapshot_id",
                    "revision_id",
                    "root_directory_id",
                    "branch_name",
                ]
            )
            .collect()
        )
        logger.info(f"Found {len(heads_with_metadata)} main/master branch revisions")
        return heads_with_metadata

    def get_directory_files(
        self,
        directory_id: bytes,
        recursive: bool = True,
        current_path: str = "",
        repo_url: str = "",
    ) -> list[dict[str, Any]]:
        """Get all files in a directory (optionally recursive).

        Args:
            directory_id: Directory ID (bytes/hash)
            recursive: Whether to traverse subdirectories
            current_path: Current path string for tracking
            repo_url: Path to the git or gitlab or pypi or debian package

        Returns:
            List of file dictionaries with metadata
        """
        # Get the directory entry to find file_entries and dir_entries lists
        dir_row = (
            self._tables[self.schema.directory]
            .filter(pl.col("id") == directory_id)
            .select(["file_entries", "dir_entries"])
            .collect()
        )

        if len(dir_row) == 0:
            logger.debug("No files found")
            return []

        files = []
        file_entry_ids = dir_row["file_entries"][0]
        dir_entry_ids = dir_row["dir_entries"][0]

        # Get file entries if they exist
        if file_entry_ids is not None and len(file_entry_ids) > 0:
            files_df = (
                self._tables[self.schema.directory_entry_file]
                .filter(pl.col("id").is_in(file_entry_ids))
                .join(
                    self._tables[self.schema.content].select(
                        [pl.col("sha1_git").alias("target"), pl.col("sha1")]
                    ),
                    on="target",
                    how="left",
                )
                .collect()
            )

            # Convert to list of dicts with path info
            for row in files_df.iter_rows(named=True):
                file_name = (
                    row["name"].decode("utf-8", errors="replace")
                    if isinstance(row["name"], bytes)
                    else row["name"]
                )
                file_path = f"{current_path}/{file_name}" if current_path else file_name

                files.append(
                    {
                        "path": file_path,
                        "name": file_name,
                        "sha1_git": row["target"],
                        "repo_url": repo_url,
                    }
                )

        # Recursively process subdirectories if requested
        if recursive and dir_entry_ids is not None and len(dir_entry_ids) > 0:
            subdirs_df = (
                self._tables[self.schema.directory_entry_dir]
                .filter(pl.col("id").is_in(dir_entry_ids))
                .select(["target", "name"])
                .collect()
            )

            for row in subdirs_df.iter_rows(named=True):
                subdir_name = (
                    row["name"].decode("utf-8", errors="replace")
                    if isinstance(row["name"], bytes)
                    else row["name"]
                )
                subdir_path = f"{current_path}/{subdir_name}" if current_path else subdir_name

                subdir_files = self.get_directory_files(
                    row["target"], recursive=True, current_path=subdir_path, repo_url=repo_url
                )
                files.extend(subdir_files)

        return files

    def add_repo_urls(self, revisions_df: pl.DataFrame) -> pl.DataFrame:
        """Add repository URLs to revisions DataFrame.

        Args:
            revisions_df: DataFrame with 'snapshot_id' column

        Returns:
            DataFrame with added 'origin_id' and 'repo_url' columns
        """
        logger.info("Adding repository URLs")

        # Load required tables
        if "snapshot_branches" not in self._tables:
            self._tables["snapshot_branches"] = self.read_parquet_files("snapshot_branches")
        if "snapshot" not in self._tables:
            self._tables["snapshot"] = self.read_parquet_files("snapshot")
        if "origin_visit" not in self._tables:
            self._tables["origin_visit"] = self.read_parquet_files("origin_visit")

        # Chain: branch_id → snapshot_branches.branch_id → snapshot_branches.snapshot_id →
        #        snapshot.object_id → origin_visit.snapshot_id → origin.id
        result = (
            revisions_df.lazy()
            # Join with snapshot_branches to get actual snapshot_id
            .join(
                self._tables["snapshot_branches"].select(
                    [pl.col("branch_id"), pl.col("snapshot_id").alias("actual_snapshot_id")]
                ),
                left_on="snapshot_id",  # This is actually branch_id from snapshot_branch.object_id
                right_on="branch_id",
                how="left",
            )
            # Join with snapshot to verify (optional, can skip)
            .join(
                self._tables["snapshot"].select(
                    [
                        pl.col("object_id").alias("snap_obj_id"),
                    ]
                ),
                left_on="actual_snapshot_id",
                right_on="snap_obj_id",
                how="left",
            )
            # Join with origin_visit to get origin_id
            .join(
                self._tables["origin_visit"].select(
                    [
                        pl.col("snapshot_id").alias("ov_snapshot_id"),
                        pl.col("origin").alias("origin_id"),
                    ]
                ),
                left_on="actual_snapshot_id",
                right_on="ov_snapshot_id",
                how="left",
            )
            # Join with origin to get URL
            .join(
                self._tables[self.schema.origin].select(
                    [
                        pl.col("id").alias("origin_id"),
                        pl.col("url").alias("repo_url"),
                        pl.col("type").alias("origin_type"),
                    ]
                ),
                on="origin_id",
                how="left",
            )
            .unique(subset=["root_directory_id", "snapshot_id"])
            .collect()
        )

        # Count how many have URLs
        url_count = result.filter(pl.col("repo_url").is_not_null()).shape[0]
        logger.info(f"Revisions with repo URLs: {url_count}/{len(result)}")
        return result
