#!/usr/bin/env python3
"""Fail the commit when non-test Python files exceed the configured line budget."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MAX_LINES = 500


def is_test_file(path: Path) -> bool:
    """Return True when the path refers to a test module."""
    normalized_name = path.name.lower()
    if normalized_name.startswith("test_") or normalized_name.endswith("_test.py"):
        return True

    lowered_parts = {part.lower() for part in path.parts}
    return "tests" in lowered_parts or "test" in lowered_parts


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Ensure non-test Python files do not exceed the maximum line count.",
    )
    parser.add_argument("paths", nargs="*", help="Files supplied by pre-commit.")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help=f"Maximum allowed lines per file (default: {DEFAULT_MAX_LINES}).",
    )

    args = parser.parse_args(argv)
    failures: list[tuple[Path, int]] = []

    for raw_path in args.paths:
        path = Path(raw_path)

        if not path.exists() or path.suffix != ".py" or is_test_file(path):
            continue

        line_count = count_lines(path)
        if line_count > args.max_lines:
            failures.append((path, line_count))

    if failures:
        for path, line_count in failures:
            print(
                f"{path}: {line_count} lines (limit {args.max_lines})",
                file=sys.stderr,
            )
        print("Split the file or rethink the abstraction to stay under the limit.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
