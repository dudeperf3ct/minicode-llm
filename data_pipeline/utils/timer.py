"""Time tracking utilities for performance monitoring."""

from collections.abc import Generator
from contextlib import contextmanager
from time import perf_counter

from loguru import logger


@contextmanager
def timer(operation_name: str) -> Generator[None, None, None]:
    """Context manager to track and log execution time of operations.

    Args:
        operation_name: Name of the operation being timed

    Yields:
        None

    Example:
        >>> with timer("data processing"):
        ...     process_data()
        [INFO] Time for data processing: 12.34 sec
    """
    start_time = perf_counter()
    try:
        yield
    finally:
        elapsed_time = perf_counter() - start_time
        logger.info(f"Time for {operation_name}: {elapsed_time:.2f} sec")
