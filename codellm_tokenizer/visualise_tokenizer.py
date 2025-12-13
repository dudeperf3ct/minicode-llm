"""CLI helper to visualise how the tokenizer splits text."""

import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from pathlib import Path

import yaml
from tokenizers import Tokenizer
from tokenizers.tools.visualizer import EncodingVisualizer

from config import TokenizerConfig


def build_arg_parser() -> ArgumentParser:
    """Build CLI arguments."""
    parser = ArgumentParser(description="Visualise tokenization for sample text.")
    parser.add_argument(
        "--tokenizer",
        dest="tokenizer_path",
        help="Path to tokenizer.json. Defaults to output_dir/tokenizer.json from config.yaml.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file to resolve default tokenizer path (default: config.yaml).",
    )
    parser.add_argument(
        "--text",
        help="Literal text to tokenize. If omitted, read from --file or stdin.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to a file whose contents will be tokenized.",
    )
    parser.add_argument(
        "--show-pre-tokenizer",
        action="store_true",
        help="Show pre-tokenizer splits before BPE merges.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip printing the token table (useful when only HTML output is needed).",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        help="Write the interactive HTML visualizer to this path (uses tokenizers.tools.visualizer).",
    )
    parser.add_argument(
        "--high-contrast",
        action="store_true",
        help="Inject brighter token colours (overrides default visualizer CSS).",
    )
    return parser


def load_config(path: Path) -> TokenizerConfig:
    """Load project config to resolve the default tokenizer path."""
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return TokenizerConfig.from_dict(raw)


def resolve_tokenizer_path(args: Namespace) -> Path:
    """Resolve which tokenizer.json to load."""
    if args.tokenizer_path:
        return Path(args.tokenizer_path)

    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            cfg = load_config(cfg_path)
            return Path(cfg.output.output_dir) / "tokenizer.json"
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not read {cfg_path}: {exc}", file=sys.stderr)

    return Path("artifacts/tokenizer.json")


def read_text(args: Namespace) -> str:
    """Return text to tokenize."""
    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    return text


def printable_token(token: str) -> str:
    """Make whitespace visible in token prints."""
    return token.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "<sp>")


def print_table(rows: Iterable[tuple[int, int, str, tuple[int, int]]]) -> None:
    """Pretty print token details as a table."""
    header = f"{'idx':>4} {'id':>6} {'token':<30} offsets"
    print(header)
    print("-" * len(header))
    for idx, token_id, token, (start, end) in rows:
        display = printable_token(token)
        print(f"{idx:>4} {token_id:>6} {display:<30} [{start},{end})")


def show_pre_tokenizer(tokenizer: Tokenizer, text: str) -> None:
    """Show pre-tokenizer splits if configured."""
    if tokenizer.pre_tokenizer is None:
        print("Pre-tokenizer: <none configured>")
        return

    print("\nPre-tokenizer splits:")
    splits = tokenizer.pre_tokenizer.pre_tokenize_str(text)
    rows = []
    for idx, (token, (start, end)) in enumerate(splits):
        rows.append((idx, -1, token, (start, end)))
    print_table(rows)


def visualise(tokenizer: Tokenizer, text: str, show_pre: bool, show_table: bool = True):
    """Encode text and optionally print tokens with ids and offsets."""
    if show_pre:
        show_pre_tokenizer(tokenizer, text)

    encoding = tokenizer.encode(text)
    if show_table:
        print("\nFinal tokens:")
        rows = []
        for idx, (token_id, token, offsets) in enumerate(
            zip(encoding.ids, encoding.tokens, encoding.offsets)
        ):
            rows.append((idx, token_id, token, offsets))
        print_table(rows)
        print(f"\nSummary: {len(text)} characters -> {len(encoding.tokens)} tokens")
    return encoding


def generate_html(tokenizer: Tokenizer, text: str) -> str:
    """Build interactive HTML using tokenizers' visualizer utilities."""
    visualizer = EncodingVisualizer(tokenizer, default_to_notebook=False)
    return visualizer(text, default_to_notebook=False)


def apply_high_contrast(html: str) -> str:
    """Override visualizer colours to be more distinct."""
    override = """
    <style>
    .even-token{background:#dbeafe !important; border:1px solid #93c5fd !important;}
    .odd-token{background:#fef3c7 !important; border:1px solid #fcd34d !important;}
    .special-token:empty::before{background:#f97316 !important;}
    </style>
    """
    if "</head>" in html:
        return html.replace("</head>", f"{override}\\n</head>", 1)
    return override + html


def main() -> None:
    """Entrypoint for CLI visualizer."""
    args = build_arg_parser().parse_args()
    tokenizer_path = resolve_tokenizer_path(args)

    if not tokenizer_path.exists():
        msg = f"Tokenizer file not found: {tokenizer_path}"
        raise FileNotFoundError(msg)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    text = read_text(args)

    visualise(tokenizer, text, show_pre=args.show_pre_tokenizer, show_table=not args.no_table)

    html_out: Path | None = args.html_out
    if html_out is None:
        html_out = Path("tokenizer_viz.html")

    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_text = generate_html(tokenizer, text)
        if args.high_contrast:
            html_text = apply_high_contrast(html_text)
        html_out.write_text(html_text, encoding="utf-8")
        print(f"HTML visualisation written to {html_out}")


if __name__ == "__main__":
    main()
