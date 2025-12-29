# template/main.py.jinja
"""Main entry point for codellm_pretrain."""

import sys
from pathlib import Path


def main() -> int:
    """Run the application.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    print("Hello from codellm_pretrain!")
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
