#!/usr/bin/env python3
"""
Select the highest CTCF-overlap region from the centred test windows.

This helper scans the ATAC/DNase/CTCF intersection BED restricted to the test
chromosomes, computes the mean CTCF signal for every window, and writes a JSON
region specification containing the top-scoring locus. The resulting JSON feeds
the genome browser minimal-overlap demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple

import pyBigWig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the highest CTCF overlap region.")
    parser.add_argument("--input-bed", required=True, type=Path)
    parser.add_argument("--ctcf-bw", required=True, type=Path)
    parser.add_argument("--folds-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_test_chromosomes(path: Path) -> Sequence[str]:
    with path.open() as handle:
        payload = json.load(handle)
    return [str(chrom) for chrom in payload["test"]]


def iter_allowed_windows(
    path: Path,
    allowed: Iterable[str],
) -> Iterator[Tuple[str, int, int]]:
    accepted = set(allowed)
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom = parts[0]
            if chrom not in accepted:
                continue
            start = int(parts[1])
            end = int(parts[2])
            yield chrom, start, end


def select_highest_mean(
    regions: Iterable[Tuple[str, int, int]],
    bw_path: Path,
) -> Tuple[str, int, float]:
    best_score = float("-inf")
    best_chrom = ""
    best_start = 0
    with pyBigWig.open(str(bw_path)) as bw:
        for chrom, start, end in regions:
            mean_value = bw.stats(chrom, start, end, nBins=1, type="mean")[0]
            score = 0.0 if mean_value is None else float(mean_value)
            if score > best_score:
                best_score = score
                best_chrom = chrom
                best_start = start
    return best_chrom, best_start, best_score


def main() -> None:
    args = parse_args()
    test_chroms = load_test_chromosomes(args.folds_json)
    regions = list(iter_allowed_windows(args.input_bed, test_chroms))
    if not regions:
        raise ValueError("No regions available after filtering to test chromosomes.")
    chrom, start, score = select_highest_mean(regions, args.ctcf_bw)
    payload = {
        "regions": [
            {
                "chr": chrom,
                "start": int(start),
                "region_name": "ctcf_overlap_rank1",
                "ctcf_mean_signal": float(score),
            }
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
