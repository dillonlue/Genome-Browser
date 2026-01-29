#!/usr/bin/env python3
"""
Select representative genome browser regions based on signal intensity.

This utility scans a BED file restricted to test chromosomes and identifies:
  - The region with the highest mean ATAC signal in the merged bigWig.
  - The region with the highest mean CTCF ChIP-seq signal.

Outputs a JSON region specification consumed by the genome tracker plotter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pyBigWig

WINDOW_SPAN = 2114  # ChromBPNet input length (see Snakefile_preprocess)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select genome browser regions.")
    parser.add_argument("--peaks-bed", required=True, type=Path)
    parser.add_argument("--test-folds", required=True, type=Path)
    parser.add_argument("--atac-bw", required=True, type=Path)
    parser.add_argument("--dnase-bw", required=True, type=Path)
    parser.add_argument("--ctcf-bw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_test_chromosomes(path: Path) -> Sequence[str]:
    with path.open() as handle:
        payload = json.load(handle)
    return [str(chrom) for chrom in payload["test"]]


def load_regions(path: Path, allowed_chroms: Iterable[str]) -> List[Tuple[str, int, int]]:
    allowed = set(allowed_chroms)
    regions: List[Tuple[str, int, int]] = []
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom not in allowed:
                continue
            start = int(parts[1])
            end = int(parts[2])
            regions.append((chrom, start, end))
    return regions


def adjusted_start(start: int, end: int, chrom_length: int) -> int:
    centre = (start + end) // 2
    half_span = WINDOW_SPAN // 2
    window_start = centre - half_span
    if window_start < 0:
        window_start = 0
    max_start = max(0, chrom_length - WINDOW_SPAN)
    if window_start > max_start:
        window_start = max_start
    return int(window_start)


def rank_top_regions(
    regions: Sequence[Tuple[str, int, int]],
    bw_path: Path,
    prefix: str,
    top_k: int,
    exclude: set,
) -> List[dict]:
    scores: List[Tuple[float, str, int]] = []

    with pyBigWig.open(str(bw_path)) as bw:
        chrom_lengths = bw.chroms()
        for chrom, start, end in regions:
            length = chrom_lengths.get(chrom, WINDOW_SPAN)
            window_start = adjusted_start(start, end, length)
            window_end = window_start + WINDOW_SPAN
            stats = bw.stats(chrom, window_start, window_end, nBins=1, type="mean")[0]
            score = 0.0 if stats is None else float(stats)
            scores.append((score, chrom, window_start))

    scores.sort(key=lambda item: item[0], reverse=True)
    selected: List[dict] = []
    seen = set()
    for score, chrom, start in scores:
        key = (chrom, start)
        if key in seen or key in exclude:
            continue
        seen.add(key)
        exclude.add(key)
        selected.append(
            {
                "chr": chrom,
                "start": int(start),
                "region_name": f"{prefix}_rank{len(selected) + 1}",
            }
        )
        if len(selected) >= top_k:
            break

    return selected


def main() -> None:
    args = parse_args()
    test_chroms = load_test_chromosomes(args.test_folds)
    regions = load_regions(args.peaks_bed, test_chroms)

    used = set()
    atac_regions = rank_top_regions(regions, args.atac_bw, "atac", 5, used)
    dnase_regions = rank_top_regions(regions, args.dnase_bw, "dnase", 5, used)
    ctcf_regions = rank_top_regions(regions, args.ctcf_bw, "ctcf", 5, used)

    payload = {"regions": atac_regions + dnase_regions + ctcf_regions}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
