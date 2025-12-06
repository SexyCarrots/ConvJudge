#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Tuple

# Folder containing the JSON files
TARGET_DIR = Path("/Users/jingbo.yang/Desktop/chat-ai/notebooks/llm-as-a-judge/ConvJudge/dump/simulated_conv_refine")

# How many oldest files to keep
KEEP_COUNT = 800

def file_birth_or_mtime(p: Path) -> float:
    """
    Prefer creation time (birth time) when available (macOS),
    otherwise fall back to modification time.
    """
    st = os.stat(p)
    # macOS / some BSDs expose st_birthtime
    birth = getattr(st, "st_birthtime", None)
    return float(birth if birth is not None else st.st_mtime)

def main() -> None:
    if not TARGET_DIR.is_dir():
        raise SystemExit(f"Not a directory: {TARGET_DIR}")

    json_files: List[Path] = [p for p in TARGET_DIR.glob("*.json") if p.is_file()]

    total = len(json_files)
    if total <= KEEP_COUNT:
        print(f"Found {total} JSON files. KEEP_COUNT={KEEP_COUNT}. Nothing to delete.")
        return

    # Sort by creation time (oldest first); tie-break by name for stability
    json_files.sort(key=lambda p: (file_birth_or_mtime(p), p.name))

    keep: List[Path] = json_files[:KEEP_COUNT]
    delete: List[Path] = json_files[KEEP_COUNT:]

    print(f"Total JSON files: {total}")
    print(f"Keeping oldest {len(keep)} files, deleting {len(delete)} newer files.\n")

    # Uncomment the next two lines if you want to preview before deleting:
    # for f in delete:
    #     print(f"[DRY-RUN] Would delete: {f}")

    # Delete the newer files
    errors: List[Tuple[Path, Exception]] = []
    for f in delete:
        try:
            f.unlink()
            # print(f"Deleted: {f}")  # optional logging
        except Exception as e:
            errors.append((f, e))

    if errors:
        print(f"\nCompleted with {len(errors)} errors:")
        for f, e in errors:
            print(f"  Failed to delete {f}: {e}")
    else:
        print("Deletion complete without errors.")

if __name__ == "__main__":
    main()
