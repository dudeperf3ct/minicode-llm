"""Token-length and stratum statistics for prepared SFT datasets.

The pipeline records prompt, assistant-turn, and complete rendered lengths for
each direct and reasoning split. These helpers only aggregate measurements;
chat rendering and record writing remain in the core dataset pipeline.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from common.data import Candidate


@dataclass
class LengthAccumulator:
    """Collect token lengths for one dataset variant."""

    prompt_lengths: list[int] = field(default_factory=list)
    total_lengths: list[int] = field(default_factory=list)

    def add(self, prompt_length: int, total_length: int) -> None:
        self.prompt_lengths.append(prompt_length)
        self.total_lengths.append(total_length)

    def summarize(self) -> dict[str, int | float]:
        if not self.total_lengths:
            raise ValueError("Cannot summarize an empty split")
        return {
            "assistant_tokens": sum(self.total_lengths) - sum(self.prompt_lengths),
            "examples": len(self.total_lengths),
            "prompt_tokens": sum(self.prompt_lengths),
            "total_tokens": sum(self.total_lengths),
        } | summarize_lengths(self.total_lengths, suffix="_length")


def summarize_lengths(lengths: list[int], *, suffix: str = "") -> dict[str, int | float]:
    """Summarize one non-empty collection of token lengths."""

    if not lengths:
        raise ValueError("Cannot summarize empty token lengths")
    ordered = sorted(lengths)
    return {
        f"median{suffix}": median(ordered),
        f"p90{suffix}": _percentile(ordered, 0.90),
        f"p95{suffix}": _percentile(ordered, 0.95),
        f"p99{suffix}": _percentile(ordered, 0.99),
        f"maximum{suffix}": ordered[-1],
    }


def add_reasoning_ratios(token_statistics: dict[str, Any]) -> None:
    """Add the supervised-token ratio used when interpreting experiment cost."""

    for statistics in token_statistics["splits"].values():
        direct_tokens = statistics["direct"]["assistant_tokens"]
        reasoning_tokens = statistics["reasoning"]["assistant_tokens"]
        statistics["reasoning_to_direct_assistant_token_ratio"] = reasoning_tokens / direct_tokens


def stratum_counts(candidates: list[Candidate]) -> list[dict[str, Any]]:
    counts = Counter(candidate.stratum for candidate in candidates)
    return [
        {
            "examples": count,
            "gpt_difficulty": stratum.gpt_difficulty,
            "style": stratum.style,
            "subset": stratum.subset,
        }
        for stratum, count in sorted(counts.items())
    ]


def _percentile(ordered_values: list[int], probability: float) -> int:
    return ordered_values[math.ceil(probability * len(ordered_values)) - 1]
