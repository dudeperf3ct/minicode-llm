from common.data import Candidate, Stratum
from common.sampling import partition_selected, select_stratified


def test_sampling_matches_sft_regression_fixture() -> None:
    candidates = [
        Candidate(
            question_id=f"q{index:02d}",
            source_index=index,
            stratum=Stratum(
                "easy" if index < 8 else "medium" if index < 16 else "hard",
                "a" if index % 2 == 0 else "b",
                "instruct" if index % 3 else "complete",
            ),
            normalized_question=f"question {index}",
        )
        for index in range(20)
    ]

    selected, _ = select_stratified(candidates, 16, 42, "regression")
    splits, _ = partition_selected(selected, 10, 3, 3, 42)

    assert [item.question_id for item in selected] == [
        "q01",
        "q10",
        "q17",
        "q03",
        "q05",
        "q13",
        "q18",
        "q02",
        "q00",
        "q09",
        "q19",
        "q14",
        "q06",
        "q16",
        "q12",
        "q04",
    ]
    assert {name: [item.question_id for item in rows] for name, rows in splits.items()} == {
        "train": ["q17", "q16", "q09", "q13", "q18", "q04", "q05", "q03", "q12", "q10"],
        "validation": ["q00", "q02", "q01"],
        "test": ["q06", "q19", "q14"],
    }


def test_sampling_is_deterministic() -> None:
    candidates = [
        Candidate(str(index), index, Stratum("easy", "subset", "instruct"), str(index))
        for index in range(10)
    ]

    first, _ = select_stratified(candidates, 5, 42, "train")
    second, _ = select_stratified(candidates, 5, 42, "train")

    assert first == second
