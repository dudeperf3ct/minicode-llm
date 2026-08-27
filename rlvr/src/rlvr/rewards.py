"""Axolotl-compatible binary code reward."""

import ast
from functools import cache
from typing import Any

from evals.code import extract_code

from rlvr.verifier import ModalVerifier, VerificationRequest, Verifier


@cache
def default_verifier() -> ModalVerifier:
    return ModalVerifier()


def final_answer(content: str) -> str:
    """Return the answer after a Qwen reasoning trace, when present."""

    return content.rsplit("</think>", 1)[-1]


def code_reward(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    test: list[str],
    question_id: list[str],
    style: list[str],
    **kwargs: Any,
) -> list[float]:
    """Return one when public tests pass and zero for model-caused failures."""

    del prompts, kwargs
    return score_completions(completions, test, question_id, style, verifier=default_verifier())


def score_completions(
    completions: list[list[dict[str, str]]],
    tests: list[str],
    question_ids: list[str],
    styles: list[str],
    *,
    verifier: Verifier,
) -> list[float]:
    size = len(completions)
    if not (len(tests) == len(question_ids) == len(styles) == size):
        raise ValueError("Completion metadata lengths do not match")

    rewards = [0.0] * size
    requests: list[VerificationRequest] = []
    request_indices: list[int] = []

    for index, (completion, tests_for_prompt, current_id, current_style) in enumerate(
        zip(completions, tests, question_ids, styles, strict=True)
    ):
        if current_style != "instruct":
            raise ValueError(f"Unsupported RLVR style: {current_style}")
        if len(completion) != 1 or completion[0].get("role") != "assistant":
            raise ValueError("Each completion must contain exactly one assistant message")
        content = final_answer(completion[0].get("content", ""))
        code = extract_code(content, current_style)
        if not code:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        requests.append(VerificationRequest(current_id, code, tests_for_prompt))
        request_indices.append(index)

    results = verifier.verify_batch(requests)
    if len(results) != len(request_indices):
        raise RuntimeError("Verifier returned the wrong number of results")

    for index, request, result in zip(request_indices, requests, results, strict=True):
        if result.question_id != request.question_id:
            raise RuntimeError("Verifier result order does not match request order")
        rewards[index] = 1.0 if result.passed else 0.0
    return rewards
