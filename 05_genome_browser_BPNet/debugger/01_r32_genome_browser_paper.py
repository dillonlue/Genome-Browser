#!/usr/bin/env python3
import os
import shlex
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    python_bin = os.environ.get("BPNET_PYTHON", os.environ.get("PYTHON_BIN", "python"))
    script = repo_root / "03_naive_motif_removal" / "scripts" / "09_ctcf_genome_browser.py"

    args = [
        "--region-config",
        "05_genome_browser_BPNet/shared/intermediate/30_paper_regions.json",
        "--track-config",
        "05_genome_browser_BPNet/configs/30_bpnet_paper_bw_tracks.json",
        "--output-dir",
        "05_genome_browser_BPNet/output/30_genome_browser",
        "--num-workers",
        "1",
        "--title",
        "BPNet paper genome browser (paper regions)",
        "--window-span",
        "1000",
        "--output-span",
        "1000",
        "--main-html",
        "05_genome_browser_BPNet/output/30_main.html",
        "--indv-plots-dir",
        "05_genome_browser_BPNet/output/30_indv_plots",
        "--summary-tsv",
        "05_genome_browser_BPNet/output/30_summary_stats.tsv",
    ]

    cmd = [python_bin, "-m", "pdb", str(script), *args]
    print("Running:")
    print(" ".join(shlex.quote(part) for part in cmd))
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
