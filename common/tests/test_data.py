from pathlib import Path

import pytest

from common.data import batched, normalize_text, sha256_file, verify_file


def test_normalize_and_batch() -> None:
    assert normalize_text("  Mixed\n  CASE  ") == "mixed case"
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_batched_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        list(batched([1], 0))


def test_verify_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("content", encoding="utf-8")

    verify_file(path, {"bytes": 7, "sha256": sha256_file(path)}, "artifact")
