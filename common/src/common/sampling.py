"""Deterministic stratified selection and partitioning."""

import math
from collections import defaultdict

from common.data import Candidate, Stratum, stable_hash

type Allocation = dict[Stratum, int]
type SplitMap = dict[str, list[Candidate]]


def select_stratified(
    candidates: list[Candidate], size: int, seed: int, namespace: str
) -> tuple[list[Candidate], Allocation]:
    grouped = _group_candidates(candidates)
    allocation = largest_remainder(
        {stratum: len(items) for stratum, items in grouped.items()}, size
    )
    selected: list[Candidate] = []
    for stratum, items in grouped.items():
        ordered = _stable_order(items, seed, f"{namespace}:select")
        selected.extend(ordered[: allocation[stratum]])
    return _stable_order(selected, seed, f"{namespace}:order"), allocation


def partition_selected(
    selected: list[Candidate],
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> tuple[SplitMap, dict[str, Allocation]]:
    if len(selected) != train_size + validation_size + test_size:
        raise ValueError("Selected pool size does not equal requested split sizes")

    grouped = _group_candidates(selected)
    capacities = {stratum: len(items) for stratum, items in grouped.items()}
    validation_allocation = largest_remainder(capacities, validation_size)
    remaining = {
        stratum: capacities[stratum] - validation_allocation[stratum] for stratum in capacities
    }
    test_allocation = largest_remainder(remaining, test_size)
    train_allocation = {
        stratum: remaining[stratum] - test_allocation[stratum] for stratum in remaining
    }

    splits: SplitMap = {"train": [], "validation": [], "test": []}
    for stratum, items in grouped.items():
        ordered = _stable_order(items, seed, "partition")
        validation_end = validation_allocation[stratum]
        test_end = validation_end + test_allocation[stratum]
        splits["validation"].extend(ordered[:validation_end])
        splits["test"].extend(ordered[validation_end:test_end])
        splits["train"].extend(ordered[test_end:])

    for split, items in splits.items():
        splits[split] = _stable_order(items, seed, f"{split}:order")
    return splits, {
        "train": train_allocation,
        "validation": validation_allocation,
        "test": test_allocation,
    }


def largest_remainder(capacities: dict[Stratum, int], size: int) -> Allocation:
    total = sum(capacities.values())
    if size < 0 or size > total:
        raise ValueError(f"Cannot allocate {size} examples from capacity {total}")
    if size == 0:
        return {stratum: 0 for stratum in capacities}

    exact = {stratum: capacity * size / total for stratum, capacity in capacities.items()}
    allocation = {stratum: math.floor(quota) for stratum, quota in exact.items()}
    remaining = size - sum(allocation.values())
    ranked = sorted(
        capacities,
        key=lambda stratum: (-(exact[stratum] - allocation[stratum]), stratum),
    )
    for stratum in ranked[:remaining]:
        allocation[stratum] += 1
    if sum(allocation.values()) != size:
        raise RuntimeError("Largest-remainder allocation did not reach requested size")
    if any(allocation[stratum] > capacities[stratum] for stratum in capacities):
        raise RuntimeError("Largest-remainder allocation exceeded a stratum capacity")
    return allocation


def _group_candidates(candidates: list[Candidate]) -> dict[Stratum, list[Candidate]]:
    grouped: dict[Stratum, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.stratum].append(candidate)
    return dict(grouped)


def _stable_order(candidates: list[Candidate], seed: int, namespace: str) -> list[Candidate]:
    return sorted(
        candidates, key=lambda candidate: stable_hash(seed, namespace, candidate.question_id)
    )
