from rlvr.rewards import score_completions, score_reasoning_format
from rlvr.verifier import VerificationRequest, VerificationResult


class FakeVerifier:
    def __init__(self, outcomes: dict[str, bool]) -> None:
        self.outcomes = outcomes
        self.requests: list[VerificationRequest] = []

    def verify_batch(self, requests: list[VerificationRequest]) -> list[VerificationResult]:
        self.requests = requests
        return [
            VerificationResult(
                question_id=request.question_id,
                status="passed" if self.outcomes[request.question_id] else "test_failed",
                passed=self.outcomes[request.question_id],
                duration_seconds=0.0,
            )
            for request in requests
        ]


def completion(content: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": content}]


def test_reasoning_format_requires_closed_trace_and_valid_code() -> None:
    completions = [
        completion("<think>reasoning</think>\n```python\ndef solve():\n    return 1\n```"),
        completion("<think>reasoning\n```python\ndef solve():\n    return 1\n```"),
        completion("<think>reasoning</think>\n```python\ndef solve(:\n```"),
        completion("<think>reasoning</think>\nNo executable answer."),
    ]

    assert score_reasoning_format(completions, ["instruct"] * 4) == [1.0, 0.0, 0.0, 0.0]


def test_code_reward_only_verifies_extractable_valid_code() -> None:
    verifier = FakeVerifier({"passes": True, "fails": False})
    completions = [
        completion("```python\ndef add(left, right):\n    return left + right\n```"),
        completion("<think>reasoning</think>\ndef add(left, right):\n    return left - right"),
        completion("def add(:"),
    ]

    rewards = score_completions(
        completions,
        ["tests", "tests", "tests"],
        ["passes", "fails", "invalid"],
        ["instruct"] * 3,
        verifier=verifier,
    )

    assert rewards == [1.0, 0.0, 0.0]
    assert [request.question_id for request in verifier.requests] == ["passes", "fails"]
