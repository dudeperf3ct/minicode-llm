"""Download content of files from SWH API."""

import asyncio
import hashlib
import time
from asyncio import Semaphore
from pathlib import Path

import httpx
import polars as pl
from loguru import logger
from tqdm.asyncio import tqdm

from codellm_data.content.models import DownloadResult, DownloadStatus, FileMetadata

# Constants for rate limit thresholds
RATE_LIMIT_CRITICAL_RATIO = 0.1  # Less than 10% remaining
RATE_LIMIT_WARNING_RATIO = 0.3  # Less than 30% remaining
RATE_LIMIT_CRITICAL_MULTIPLIER = 3  # Multiply base delay by 3
RATE_LIMIT_WARNING_MULTIPLIER = 2  # Multiply base delay by 2

# HTTP status codes
HTTP_NOT_FOUND = 404
HTTP_RATE_LIMITED = 429


class SWHContentDownloader:
    """Download content for files using Software Heritage API."""

    def __init__(self, config) -> None:
        """Init."""
        self.config = config
        self._semaphore = Semaphore(config.max_concurrent)
        self.rate_limiter = asyncio.Lock()
        self.last_request_time = 0.0

        # Track rate limit info from headers
        self.rate_limit_remaining: int | None = None
        self.rate_limit_reset: int | None = None
        self.adaptive_delay: float = config.rate_limit_delay

    async def _rate_limit(self):
        async with self.rate_limiter:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_request_time

            # Use adaptive delay if we're getting close to rate limits
            delay = self.adaptive_delay
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
            self.last_request_time = asyncio.get_event_loop().time()

    def _update_rate_limit_info(self, headers: httpx.Headers):
        """Update rate limit info from response headers.

        Docs: https://archive.softwareheritage.org/api/#rate-limiting
        SWH API returns:
        - X-RateLimit-Limit: requests allowed per hour
        - X-RateLimit-Remaining: requests remaining in current window
        - X-RateLimit-Reset: Unix timestamp when limit resets
        """
        if "X-RateLimit-Remaining" in headers:
            self.rate_limit_remaining = int(headers["X-RateLimit-Remaining"])

        if "X-RateLimit-Reset" in headers:
            self.rate_limit_reset = int(headers["X-RateLimit-Reset"])

        if "X-RateLimit-Limit" in headers:
            limit = int(headers["X-RateLimit-Limit"])

            # Adaptive rate limiting: slow down when approaching limit
            if self.rate_limit_remaining is not None:
                remaining_ratio = self.rate_limit_remaining / limit

                if remaining_ratio < RATE_LIMIT_CRITICAL_RATIO:
                    # Increase delay significantly
                    self.adaptive_delay = (
                        self.config.rate_limit_delay * RATE_LIMIT_CRITICAL_MULTIPLIER
                    )
                    reset_time = (
                        time.ctime(self.rate_limit_reset) if self.rate_limit_reset else "unknown"
                    )
                    logger.warning(
                        f"Rate limit low: {self.rate_limit_remaining}/{limit} remaining, "
                        f"resets at {reset_time}, "
                        f"increasing delay to {self.adaptive_delay:.2f}s"
                    )
                elif remaining_ratio < RATE_LIMIT_WARNING_RATIO:
                    # Increase delay moderately
                    self.adaptive_delay = (
                        self.config.rate_limit_delay * RATE_LIMIT_WARNING_MULTIPLIER
                    )
                else:
                    # Reset to normal delay
                    self.adaptive_delay = self.config.rate_limit_delay

    async def download_file(  # noqa: PLR0911
        self,
        file_meta: FileMetadata,
        client: httpx.AsyncClient,
        output_path: Path,
        skip_existing: bool = True,
    ) -> DownloadResult:
        """Download a single file and save it to the disk.

        Args:
            file_meta: File metadata
            client: HTTP client
            output_path: Path to save the file
            skip_existing: If True, skip download if file already exists (default: True)
        """
        url = f"{self.config.base_url}/content/sha1_git:{file_meta.sha1_git}/raw/"

        def _make_result(
            success: bool,
            size: int = 0,
            status: DownloadStatus = DownloadStatus.UNKNOWN_ERROR,
            msg: str | None = None,
        ) -> DownloadResult:
            return DownloadResult(
                sha1_git=file_meta.sha1_git,
                local_path=str(output_path) if success else None,
                success=success,
                size=size,
                status=status,
                error_message=msg,
            )

        # Check if file already exists
        if skip_existing and output_path.exists():
            file_size = output_path.stat().st_size
            logger.debug(f"Skipping {file_meta.file_path} - already exists ({file_size} bytes)")
            return _make_result(True, file_size, DownloadStatus.SKIPPED, "File already exists")

        for attempt in range(self.config.max_retries):
            await self._rate_limit()

            async with self._semaphore:
                try:
                    response = await client.get(url)
                    self._update_rate_limit_info(response.headers)
                    response.raise_for_status()

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(output_path.write_bytes, response.content)

                    logger.debug(
                        f"Downloaded {file_meta.file_path} ({len(response.content)} bytes) "
                        f"from {file_meta.repo_url}"
                    )
                    return _make_result(True, len(response.content), DownloadStatus.SUCCESS)

                except httpx.HTTPStatusError as e:
                    code = e.response.status_code

                    if code == HTTP_NOT_FOUND:
                        logger.debug(f"File not found (404): {file_meta.file_path} - {url}")
                        return _make_result(False, status=DownloadStatus.NOT_FOUND, msg="Not found")

                    if code == HTTP_RATE_LIMITED:
                        # Use Retry-After header if present, otherwise exponential backoff
                        # Start with minimum 2 seconds, then 4, 8, 16, etc.
                        retry_after = e.response.headers.get("Retry-After")
                        wait_time = float(retry_after) if retry_after else (2 ** (attempt + 1))
                        logger.warning(
                            f"Rate limited (429) on attempt {attempt + 1}/"
                            f"{self.config.max_retries} for {file_meta.file_path}, "
                            f"waiting {wait_time:.1f}s"
                        )
                        await asyncio.sleep(wait_time)
                        continue  # retry

                except httpx.TimeoutException:
                    if attempt == self.config.max_retries - 1:
                        logger.warning(
                            f"Timeout after {self.config.max_retries} attempts: "
                            f"{file_meta.file_path}"
                        )
                        return _make_result(False, status=DownloadStatus.TIMEOUT, msg="Timeout")
                    logger.debug(
                        f"Timeout on attempt {attempt + 1}/{self.config.max_retries} "
                        f"for {file_meta.file_path}, retrying in {2**attempt}s"
                    )
                    await asyncio.sleep(2**attempt)

                except httpx.RequestError as e:
                    if attempt == self.config.max_retries - 1:
                        logger.warning(
                            f"Network error after {self.config.max_retries} attempts "
                            f"for {file_meta.file_path}: {e}"
                        )
                        return _make_result(False, status=DownloadStatus.NETWORK_ERROR, msg=str(e))
                    logger.debug(
                        f"Network error on attempt {attempt + 1}/{self.config.max_retries} "
                        f"for {file_meta.file_path}, retrying in {2**attempt}s"
                    )
                    await asyncio.sleep(2**attempt)

                except Exception as e:
                    logger.error(f"Unexpected error downloading {file_meta.file_path}: {e}")
                    return _make_result(False, msg=str(e))

        logger.warning(f"Max retries exceeded for {file_meta.file_path}")
        return _make_result(False, status=DownloadStatus.MAX_RETRIES, msg="Max retries exceeded")

    def _get_output_path(self, file_meta: FileMetadata, base_dir: Path) -> Path:
        """Generate output path for a file."""
        repo_hash = hashlib.sha256(file_meta.repo_url.encode("utf-8")).hexdigest()[:8]
        folder_dir = base_dir.joinpath(repo_hash)
        folder_dir.mkdir(exist_ok=True, parents=True)
        return folder_dir.joinpath(file_meta.file_path)

    async def download_files_batch(
        self,
        files_df: pl.DataFrame,
        output_dir: Path,
        max_files: int | None = None,
        skip_existing: bool = True,
    ) -> list[DownloadResult]:
        """Downloads files in a batch.

        Args:
            files_df: DataFrame containing file metadata
            output_dir: Directory to save downloaded files
            max_files: Maximum number of files to download (None = all)
            skip_existing: If True, skip files that already exist (default: True)
        """
        files_list = [FileMetadata.from_dict(file) for file in files_df.to_dicts()]

        if max_files:
            files_list = files_list[:max_files]

        total_files = len(files_list)
        logger.info(f"Starting download of {total_files} files to {output_dir}")
        logger.info(f"Max concurrent downloads: {self.config.max_concurrent}")
        logger.info(f"Rate limit delay: {self.config.rate_limit_delay}s between requests")
        logger.info(f"Skip existing files: {skip_existing}")

        limits = httpx.Limits(
            max_connections=self.config.max_concurrent * 2,
            max_keepalive_connections=self.config.max_concurrent,
        )

        async with httpx.AsyncClient(
            timeout=self.config.timeout, limits=limits, headers=self.config.headers
        ) as client:
            # create tasks
            tasks = [
                self.download_file(
                    file_meta, client, self._get_output_path(file_meta, output_dir), skip_existing
                )
                for file_meta in files_list
            ]

            results = await tqdm.gather(*tasks, desc="Downloading files", unit="file")

            # Log summary statistics
            total = len(results)
            skipped = sum(1 for r in results if r.status == DownloadStatus.SKIPPED)
            successful = sum(1 for r in results if r.status == DownloadStatus.SUCCESS)
            failed = sum(1 for r in results if not r.success)

            logger.info(
                f"Download complete: {total} total, "
                f"{successful} downloaded, {skipped} skipped, {failed} failed"
            )

            return results

    def download_batch_sync(
        self,
        files_df: pl.DataFrame,
        output_dir: Path,
        max_files: int | None = None,
        skip_existing: bool = True,
    ):
        """Synchronous wrapper for download_batch.

        Args:
            files_df: DataFrame containing file metadata
            output_dir: Directory to save downloaded files
            max_files: Maximum number of files to download (None = all)
            skip_existing: If True, skip files that already exist (default: True)
        """
        return asyncio.run(
            self.download_files_batch(files_df, output_dir, max_files, skip_existing)
        )
