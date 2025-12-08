#!/usr/bin/env python3
"""Remove simulated conversation JSON files that have no mistakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def filter_dir(data_dir: Path) -> None:
    files = sorted(data_dir.glob("*.json"))
    removed = 0
    kept = 0
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"Skipping {path}: failed to read ({exc})")
            continue
        mistakes = payload.get("mistakes")
        if isinstance(mistakes, list) and len(mistakes) == 0:
            path.unlink()
            removed += 1
        else:
            kept += 1

    print(
        f"Filtered directory {data_dir}: removed {removed} file(s) with empty mistakes; {kept} file(s) remain."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete conversation JSON files whose 'mistakes' list is empty.")
    parser.add_argument("data_dir", help="Directory containing simulated conversation JSON files")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    filter_dir(data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
