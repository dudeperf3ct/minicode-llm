"""Logging configuration."""

from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path) -> None:
    """Configure loguru to log to both console and file.

    Args:
        log_dir: Directory to store log files
    """
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / "codellm_data.log"

    logger.remove()

    # Add console handler with colour
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",  # noqa: E501
        level="INFO",
        colorize=True,
    )

    # Add file handler with rotation
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
    )

    logger.info(f"Logging to file: {log_file}")
