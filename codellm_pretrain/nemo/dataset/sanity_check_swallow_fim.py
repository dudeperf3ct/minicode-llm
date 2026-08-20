"""Sanity checker for the SwallowCode streaming FIM pipeline."""

import argparse
import json
import random
from math import sqrt

from datasets import load_dataset
from transformers import AutoTokenizer

from dataset.swallow_fim_iterable_dataset import (
    END_OF_TEXT,
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIMConfig,
    FIMTelemetry,
    FIMTransform,
)


def _shorten(text: str, max_chars: int) -> str:
    cleaned = text.replace("\n", "\\n")
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}..."


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sanity checks for SwallowCode FIM transform")
    parser.add_argument("--hf-path", default="tokyotech-llm/swallow-code-v2")
    parser.add_argument("--hf-subset", default="swallowcode-v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="improved_code")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--fim-rate", type=float, default=0.8)
    parser.add_argument("--psm-prob", type=float, default=0.5)
    parser.add_argument("--spm-prob", type=float, default=0.5)
    parser.add_argument("--min-code-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-previews", type=int, default=3)
    parser.add_argument("--preview-max-chars", type=int, default=180)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--validate", action="store_true")
    return parser


def _index_or_neg1(text: str, token: str) -> int:
    return text.find(token)


def _validate_string_layout(mode: str, text: str) -> bool:
    prefix_idx = _index_or_neg1(text, FIM_PREFIX)
    suffix_idx = _index_or_neg1(text, FIM_SUFFIX)
    middle_idx = _index_or_neg1(text, FIM_MIDDLE)
    eot_idx = _index_or_neg1(text, END_OF_TEXT)

    if mode == "psm":
        if min(prefix_idx, suffix_idx, middle_idx, eot_idx) < 0:
            return False
        return prefix_idx < suffix_idx < middle_idx < eot_idx

    if mode == "spm":
        if min(prefix_idx, suffix_idx, middle_idx, eot_idx) < 0:
            return False
        return suffix_idx < prefix_idx < middle_idx < eot_idx

    if mode == "none":
        return prefix_idx < 0 and suffix_idx < 0 and middle_idx < 0 and text.endswith(END_OF_TEXT)

    return False


def _build_checks(
    *,
    fim_config: FIMConfig,
    telemetry: FIMTelemetry,
    num_samples: int,
    format_error_count: int,
    tokenizer_checks: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    t = telemetry.to_dict()

    observed_rate = float(t["effective_fim_rate"])
    target_rate = fim_config.fim_rate
    n = max(1, num_samples)
    sigma = sqrt(max(1e-12, target_rate * (1.0 - target_rate) / n))
    rate_tol = max(0.05, 3.0 * sigma)
    report["fim_rate"] = {
        "target": target_rate,
        "observed": observed_rate,
        "tolerance": rate_tol,
        "pass": abs(observed_rate - target_rate) <= rate_tol,
        "note": "Literature typically uses high FIM rates (0.5-0.9) with mixed AR/FIM.",
    }

    fim_applied = int(t["fim_applied"])
    if fim_applied > 0:
        observed_psm_share = int(t["psm_count"]) / fim_applied
        target_psm_share = fim_config.psm_prob
        sigma_psm = sqrt(max(1e-12, target_psm_share * (1.0 - target_psm_share) / fim_applied))
        psm_tol = max(0.05, 3.0 * sigma_psm)
        psm_pass = abs(observed_psm_share - target_psm_share) <= psm_tol
    else:
        observed_psm_share = 0.0
        target_psm_share = fim_config.psm_prob
        psm_tol = 1.0
        psm_pass = False

    report["psm_spm_balance"] = {
        "target_psm_share": target_psm_share,
        "observed_psm_share": observed_psm_share,
        "tolerance": psm_tol,
        "pass": psm_pass,
        "note": "PSM/SPM should stay close to configured mix; common recommendation is 50/50.",
    }

    report["format_layout"] = {
        "format_error_count": format_error_count,
        "pass": format_error_count == 0,
        "note": "Checks strict token order for PSM/SPM and no FIM markers for non-FIM samples.",
    }

    if tokenizer_checks is not None:
        report["tokenizer_special_tokens"] = tokenizer_checks

    return report


def main() -> None:
    args = build_arg_parser().parse_args()

    fim_config = FIMConfig(
        fim_rate=args.fim_rate,
        psm_prob=args.psm_prob,
        spm_prob=args.spm_prob,
        min_code_length=args.min_code_length,
    )
    transform = FIMTransform(config=fim_config)
    telemetry = FIMTelemetry()
    previews: list[dict[str, str | int]] = []

    dataset = load_dataset(
        args.hf_path,
        args.hf_subset,
        split=args.split,
        streaming=True,
        revision=args.revision,
    )
    if args.shuffle_buffer_size > 0 and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    rng = random.Random(args.seed)
    processed = 0
    format_error_count = 0

    tokenizer = None
    special_token_ids: dict[str, int | None] | None = None
    tokenizer_counter = {
        "fim_samples_missing_special_tokens": 0,
        "non_fim_samples_with_special_markers": 0,
    }
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        special_token_ids = {
            "fim_prefix_id": tokenizer.convert_tokens_to_ids(FIM_PREFIX),
            "fim_suffix_id": tokenizer.convert_tokens_to_ids(FIM_SUFFIX),
            "fim_middle_id": tokenizer.convert_tokens_to_ids(FIM_MIDDLE),
            "endoftext_id": tokenizer.convert_tokens_to_ids(END_OF_TEXT),
        }

    for row in dataset:
        if processed >= args.num_samples:
            break

        text = row.get(args.text_field)
        if not isinstance(text, str) or not text:
            telemetry.mark_missing_text()
            continue

        result = transform.apply(text, rng)
        if not _validate_string_layout(result.mode, result.text):
            format_error_count += 1

        if tokenizer is not None and special_token_ids is not None:
            encoded = tokenizer.encode(result.text, add_special_tokens=False)
            counts = {
                name: (encoded.count(token_id) if token_id is not None else 0)
                for name, token_id in special_token_ids.items()
            }
            if result.mode in {"psm", "spm"}:
                if not (
                    counts["fim_prefix_id"] >= 1
                    and counts["fim_suffix_id"] >= 1
                    and counts["fim_middle_id"] >= 1
                    and counts["endoftext_id"] >= 1
                ):
                    tokenizer_counter["fim_samples_missing_special_tokens"] += 1
            elif result.mode == "none":
                if (
                    counts["fim_prefix_id"] > 0
                    or counts["fim_suffix_id"] > 0
                    or counts["fim_middle_id"] > 0
                ):
                    tokenizer_counter["non_fim_samples_with_special_markers"] += 1

        telemetry.update(result)
        processed += 1

        if len(previews) < args.num_previews:
            previews.append(
                {
                    "mode": result.mode,
                    "middle_chars": result.middle_chars,
                    "preview": _shorten(result.text, args.preview_max_chars),
                }
            )

    report: dict[str, object] = {
        "config": {
            "hf_path": args.hf_path,
            "hf_subset": args.hf_subset,
            "split": args.split,
            "text_field": args.text_field,
            "revision": args.revision,
            "fim_rate": fim_config.fim_rate,
            "psm_prob": fim_config.psm_prob,
            "spm_prob": fim_config.spm_prob,
            "min_code_length": fim_config.min_code_length,
            "seed": args.seed,
            "num_samples": args.num_samples,
        },
        "telemetry": telemetry.to_dict(),
        "previews": previews,
        "sentinel_strings": {
            "fim_prefix": FIM_PREFIX,
            "fim_suffix": FIM_SUFFIX,
            "fim_middle": FIM_MIDDLE,
            "endoftext": END_OF_TEXT,
        },
    }

    tokenizer_checks = None
    if tokenizer is not None and special_token_ids is not None:
        unknown_hits = {
            name: (tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id)
            for name, token_id in special_token_ids.items()
        }
        tokenizer_checks = {
            "pass": (
                not any(unknown_hits.values())
                and tokenizer_counter["fim_samples_missing_special_tokens"] == 0
            ),
            "unknown_hits": unknown_hits,
            "fim_samples_missing_special_tokens": tokenizer_counter[
                "fim_samples_missing_special_tokens"
            ],
            "non_fim_samples_with_special_markers": tokenizer_counter[
                "non_fim_samples_with_special_markers"
            ],
            "note": "All transformed FIM samples should include all special marker token IDs; non-FIM marker occurrences are informational.",
        }

        report["tokenizer"] = {
            "name_or_path": args.tokenizer,
            "special_token_ids": {
                **special_token_ids,
                "unk_token_id": tokenizer.unk_token_id,
            },
            "unknown_hits": unknown_hits,
        }

    if args.validate:
        report["checks"] = _build_checks(
            fim_config=fim_config,
            telemetry=telemetry,
            num_samples=processed,
            format_error_count=format_error_count,
            tokenizer_checks=tokenizer_checks,
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
