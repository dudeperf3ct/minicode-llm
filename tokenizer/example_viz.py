"""Programmatic tokenizer visualisation examples."""

from pathlib import Path

import yaml
from tokenizers import Tokenizer

from config import TokenizerConfig
from visualise_tokenizer import apply_high_contrast, generate_html, visualise


def load_config(path: Path) -> TokenizerConfig:
    """Load config from YAML."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TokenizerConfig.from_dict(raw)


def slugify(label: str) -> str:
    """Create a filesystem-friendly name."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_") or "example"


def main() -> None:
    """Run a few sample visualisations."""
    cfg = load_config(Path("config.yaml"))
    tokenizer_path = Path(cfg.output.output_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        msg = f"Tokenizer not found at {tokenizer_path}; train it first."
        raise FileNotFoundError(msg)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    samples = {
        "function_def": "def add_numbers(x: int, y: int) -> int:\n    return x + y",
        "json_snippet": '{"name": "Codellm", "version": "0.1.0", "enabled": true}',
        "indent_vs_tab": "for i in range(3):\n    print(i)\n\tprint('tab branch')",
        "fim_example": "<|fim_prefix|>def foo():<|fim_suffix|>",
    }

    output_dir = Path(cfg.output.output_dir) / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, text in samples.items():
        print(f"\n=== {label} ===")
        visualise(tokenizer, text, show_pre=False)
        html = generate_html(tokenizer, text)
        html = apply_high_contrast(html)
        out_path = output_dir / f"{slugify(label)}.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
