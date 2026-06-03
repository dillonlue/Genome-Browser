#!/usr/bin/env python3
import os
import shlex
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    python_bin = os.environ.get("BPNET_PYTHON", os.environ.get("PYTHON_BIN", "python"))
    script = repo_root / "07_simplified_tracker" / "simplified_tracker.py"

    args = [
        "--region-config",
        "05_genome_browser_BPNet/shared/intermediate/03_top_regions.json",
        "--track-config",
        "05_genome_browser_BPNet/configs/01_bpnet_paper_tracks.json",
        "--output-dir",
        "07_simplified_tracker/output",
        "--window-span",
        "1000",
        "--output-span",
        "1000",
        "--title",
        "Simplified genome tracker (top regions)",
    ]

    cmd = [python_bin, "-m", "pdb", str(script), *args]
    print("Running:")
    print(" ".join(shlex.quote(part) for part in cmd))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
