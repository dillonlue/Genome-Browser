#!/usr/bin/env python3
"""
Select the top-N overlap regions ranked by mean CTCF signal.

This utility scans a BED file containing input windows (2114 bp) and computes
the average CTCF ChIP-seq signal over each window using the provided bigWig.
The highest-scoring regions are written to a JSON region spec consumed by the
genome tracker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pyBigWig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select top CTCF overlap regions.")
    parser.add_argument("--input-bed", required=True, type=Path, help="BED file with candidate windows.")
    parser.add_argument("--ctcf-bw", required=True, type=Path, help="CTCF ChIP-seq bigWig.")
    parser.add_argument("--count", type=int, default=100, help="Number of regions to retain (default: 100).")
    parser.add_argument("--output", required=True, type=Path, help="Destination JSON region file.")
    return parser.parse_args()


def load_windows(path: Path) -> List[Tuple[str, int, int]]:
    windows: List[Tuple[str, int, int]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, *_ = line.rstrip("\n").split("\t")
            windows.append((chrom, int(start), int(end)))
    return windows


def rank_by_ctcf(
    windows: List[Tuple[str, int, int]],
    bw_path: Path,
) -> List[Tuple[str, int, int]]:
    scored: List[Tuple[float, str, int, int]] = []
    with pyBigWig.open(str(bw_path)) as bw:
        for chrom, start, end in windows:
            mean_val = bw.stats(chrom, start, end, nBins=1, type="mean")[0]
            score = 0.0 if mean_val is None else float(mean_val)
            scored.append((score, chrom, start, end))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [(chrom, start, end) for _, chrom, start, end in scored]


def write_region_json(windows: List[Tuple[str, int, int]], output: Path) -> None:
    payload = {
        "regions": [
            {
                "chr": chrom,
                "start": start,
                "region_name": f"ctcf_overlap_top100_rank{idx + 1}",
            }
            for idx, (chrom, start, _end) in enumerate(windows)
        ]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    windows = load_windows(args.input_bed)
    if not windows:
        raise ValueError("No windows found in the provided BED file.")
    ranked = rank_by_ctcf(windows, args.ctcf_bw)
    top_n = ranked[: max(0, int(args.count))]
    write_region_json(top_n, args.output)


if __name__ == "__main__":
    main()
