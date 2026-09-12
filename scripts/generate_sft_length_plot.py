# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers==5.15.0",
# ]
# ///

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

MODEL_LABELS = {
    "base": "Base",
    "direct_lora": "Direct SFT (LoRA)",
    "direct_fft": "Direct SFT (full FT)",
    "reasoning_lora": "Reasoning SFT (LoRA)",
    "reasoning_fft": "Reasoning SFT (full FT)",
    "post_direct": "Post-trained (non-thinking)",
    "post_thinking": "Post-trained (thinking)",
}

MODEL_COLORS = {
    "base": "#64748b",
    "direct_lora": "#2563eb",
    "direct_fft": "#06b6d4",
    "reasoning_lora": "#7c3aed",
    "reasoning_fft": "#db2777",
    "post_direct": "#ea580c",
    "post_thinking": "#16a34a",
}

DATASET_LABELS = {
    "kodcode": "KodCode held-out",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "livecodebench_easy": "LiveCodeBench Easy",
    "livecodebench_medium": "LiveCodeBench Medium",
    "livecodebench_hard": "LiveCodeBench Hard",
}

HELD_OUT_RUNS = {
    "base": "baseline-qwen35-4b-base-pinned",
    "direct_lora": "main-10k-direct-lora-seed42",
    "direct_fft": "main-10k-direct-fft-seed42",
    "reasoning_lora": "main-10k-reasoning-lora-seed42",
    "reasoning_fft": "main-10k-reasoning-fft-seed42",
    "post_direct": "reference-qwen35-4b-post-trained",
    "post_thinking": "reference-qwen35-4b-post-trained-thinking",
}

EVALPLUS_RUNS = {
    "base": "qwen3.5-4b-base/evalplus",
    "direct_lora": "qwen3.5-4b-direct-lora/evalplus",
    "direct_fft": "qwen3.5-4b-direct-fft/evalplus",
    "reasoning_lora": "qwen3.5-4b-reasoning-lora/thinking/evalplus",
    "reasoning_fft": "qwen3.5-4b-reasoning-fft/thinking/evalplus",
    "post_direct": "qwen3.5-4b-post-trained-nonthinking/evalplus",
    "post_thinking": "qwen3.5-4b-post-trained-thinking/evalplus",
}

LIVECODEBENCH_RUNS = {
    "base": "qwen3.5-4b-base/livecodebench",
    "direct_lora": "qwen3.5-4b-direct-lora/livecodebench",
    "direct_fft": "qwen3.5-4b-direct-fft/livecodebench",
    "reasoning_lora": "qwen3.5-4b-reasoning-lora/thinking/livecodebench",
    "reasoning_fft": "qwen3.5-4b-reasoning-fft/thinking/livecodebench",
    "post_direct": "qwen3.5-4b-post-trained/direct/livecodebench",
    "post_thinking": "qwen3.5-4b-post-trained/thinking/livecodebench",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the interactive SFT response-length distribution plot."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("codellm_sft"),
        help="Path to the codellm_sft directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("images/sft_output_length_distribution.html"),
        help="HTML fragment written for the Hugo plotly shortcode.",
    )
    return parser.parse_args()


COMPLETION_TOKENS_PATTERN = re.compile(rb'"completion_tokens"\s*:\s*(\d+)')


def jsonl_rows(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def one_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one file matching {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def held_out_lengths(source_root: Path) -> dict[str, list[int]]:
    reports_root = source_root / "sft/reports/test"
    result: dict[str, list[int]] = {}
    for model, run_name in HELD_OUT_RUNS.items():
        result[model] = [
            int(row["output_tokens"])
            for row in jsonl_rows(reports_root / run_name / "results.jsonl")
            if isinstance(row.get("output_tokens"), int) and row["output_tokens"] > 0
        ]
    return result


def evalplus_lengths(
    source_root: Path, tokenizer: Any
) -> dict[str, dict[str, list[int]]]:
    results_root = source_root / "eval/results"
    result = {"humaneval": {}, "mbpp": {}}
    for model, relative_path in EVALPLUS_RUNS.items():
        for dataset, model_lengths in result.items():
            raw_path = one_file(results_root / relative_path / dataset, "*.raw.jsonl")
            lengths = []
            for row in jsonl_rows(raw_path):
                token_ids = tokenizer.encode(
                    row["solution"], add_special_tokens=False, verbose=False
                )
                lengths.append(len(token_ids))
            model_lengths[model] = lengths
    return result


def completion_token_lengths(path: Path) -> list[int]:
    lengths = []
    overlap = 128
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            data = tail + chunk
            boundary = max(0, len(data) - overlap)
            for match in COMPLETION_TOKENS_PATTERN.finditer(data):
                if match.start() >= boundary:
                    break
                lengths.append(int(match.group(1)))
            tail = data[boundary:]
    lengths.extend(
        int(match.group(1)) for match in COMPLETION_TOKENS_PATTERN.finditer(tail)
    )
    return lengths


def livecodebench_lengths(source_root: Path) -> dict[str, dict[str, list[int]]]:
    results_root = source_root / "eval/results"
    result = {
        "livecodebench_easy": {},
        "livecodebench_medium": {},
        "livecodebench_hard": {},
    }
    for model, relative_path in LIVECODEBENCH_RUNS.items():
        run_root = results_root / relative_path
        summaries = sorted(run_root.glob("*/summary.json"))
        if len(summaries) != 3:
            raise RuntimeError(
                f"Expected three LiveCodeBench summaries in {run_root}, found {len(summaries)}"
            )
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text())
            dataset = summary["configuration"]["task"]["name"]
            result[dataset][model] = completion_token_lengths(
                summary_path.parent / "results.json"
            )
    return result


def load_tokenizer(source_root: Path) -> Any:
    revisions_path = source_root / "sft/manifests/revisions.json"
    tokenizer_revision = json.loads(revisions_path.read_text())["tokenizer"]
    return AutoTokenizer.from_pretrained(
        tokenizer_revision["repo_id"], revision=tokenizer_revision["revision"]
    )


def plot_html(datasets: dict[str, dict[str, list[int]]]) -> str:
    payload = json.dumps(datasets, separators=(",", ":"))
    labels = json.dumps(DATASET_LABELS, separators=(",", ":"))
    models = json.dumps(MODEL_LABELS, separators=(",", ":"))
    colors = json.dumps(MODEL_COLORS, separators=(",", ":"))
    return f"""<div id="sft-output-length-distribution" class="plotly-graph-div" style="height:680px;width:100%;"></div>
<script src="https://cdn.plot.ly/plotly-3.3.0.min.js" charset="utf-8"></script>
<script>
(() => {{
  const datasets = {payload};
  const datasetLabels = {labels};
  const modelLabels = {models};
  const modelColors = {colors};
  const datasetKeys = Object.keys(datasetLabels);
  const modelKeys = Object.keys(modelLabels);
  const traces = [];

  datasetKeys.forEach((datasetKey, datasetIndex) => {{
    modelKeys.forEach((modelKey) => {{
      const lengths = datasets[datasetKey][modelKey];
      traces.push({{
        type: "violin",
        name: modelLabels[modelKey],
        y: lengths,
        visible: datasetIndex === 0,
        points: false,
        box: {{ visible: true }},
        meanline: {{ visible: true }},
        spanmode: "hard",
        line: {{ color: modelColors[modelKey], width: 1.5 }},
        fillcolor: modelColors[modelKey],
        opacity: 0.72,
        hovertemplate: `${{modelLabels[modelKey]}}<br>Completion tokens: %{{y:,}}<extra></extra>`
      }});
    }});
  }});

  const buttons = datasetKeys.map((datasetKey, datasetIndex) => ({{
    label: datasetLabels[datasetKey],
    method: "update",
    args: [
      {{
        visible: datasetKeys.flatMap((_, index) =>
          modelKeys.map(() => index === datasetIndex)
        )
      }},
      {{ title: {{ text: `Generated response length — ${{datasetLabels[datasetKey]}}` }} }}
    ]
  }}));

  const plot = document.getElementById("sft-output-length-distribution");
  const isDark = () => document.documentElement.dataset.theme === "dark";
  const theme = () => ({{
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {{ color: isDark() ? "#d1d5db" : "#334155" }},
    xaxis: {{ gridcolor: isDark() ? "#374151" : "#e2e8f0" }},
    yaxis: {{ gridcolor: isDark() ? "#374151" : "#e2e8f0" }}
  }});

  Plotly.newPlot(plot, traces, {{
    ...theme(),
    title: {{ text: `Generated response length — ${{datasetLabels[datasetKeys[0]]}}` }},
    violingap: 0.12,
    violinmode: "group",
    showlegend: false,
    margin: {{ l: 72, r: 24, t: 120, b: 145 }},
    xaxis: {{ ...theme().xaxis, tickangle: -28, automargin: true }},
    yaxis: {{
      ...theme().yaxis,
      title: {{ text: "Completion tokens (log scale)" }},
      type: "log",
      tickvals: [10, 30, 100, 300, 1000, 3000, 10000, 30000],
      ticktext: ["10", "30", "100", "300", "1k", "3k", "10k", "30k"]
    }},
    updatemenus: [{{
      type: "dropdown",
      direction: "down",
      x: 0,
      xanchor: "left",
      y: 1.18,
      yanchor: "top",
      active: 0,
      buttons
    }}],
    annotations: [{{
      text: "Violin width shows density; the inner box shows the median and interquartile range.",
      x: 0,
      xref: "paper",
      xanchor: "left",
      y: -0.34,
      yref: "paper",
      yanchor: "top",
      showarrow: false,
      font: {{ size: 11 }}
    }}]
  }}, {{ responsive: true, displaylogo: false }});

  new MutationObserver(() => Plotly.relayout(plot, {{
    "font.color": isDark() ? "#d1d5db" : "#334155",
    "xaxis.gridcolor": isDark() ? "#374151" : "#e2e8f0",
    "yaxis.gridcolor": isDark() ? "#374151" : "#e2e8f0"
  }})).observe(
    document.documentElement,
    {{ attributes: true, attributeFilter: ["data-theme"] }}
  );
}})();
</script>
"""


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.source_root)
    datasets: dict[str, dict[str, list[int]]] = {
        "kodcode": held_out_lengths(args.source_root)
    }
    datasets.update(evalplus_lengths(args.source_root, tokenizer))
    datasets.update(livecodebench_lengths(args.source_root))

    for dataset, models in datasets.items():
        missing = set(MODEL_LABELS) - set(models)
        if missing:
            raise RuntimeError(f"{dataset} is missing models: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(plot_html(datasets))
    print(f"Wrote {args.output}")
    for dataset, models in datasets.items():
        counts = ", ".join(
            f"{model}={len(lengths)}" for model, lengths in models.items()
        )
        print(f"{dataset}: {counts}")


if __name__ == "__main__":
    main()
