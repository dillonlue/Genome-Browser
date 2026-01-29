#!/usr/bin/env python3
"""
Combine high-signal regions with additional random windows (start-only output).

The script samples candidate peaks on test chromosomes, converts them to the
ChromBPNet input window length (2,114 bp), and appends a fixed number of random
regions to an existing high-signal configuration.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

WINDOW_SPAN = 2114  # ChromBPNet input length (Snakefile_preprocess constant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create high-signal + random region config.")
    parser.add_argument("--peaks-bed", required=True, type=Path)
    parser.add_argument("--test-folds", required=True, type=Path)
    parser.add_argument("--chromsizes", required=True, type=Path)
    parser.add_argument("--base-regions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--random-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def load_test_chromosomes(path: Path) -> Sequence[str]:
    with path.open() as handle:
        payload = json.load(handle)
    return [str(chrom) for chrom in payload["test"]]


def load_chromsizes(path: Path) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with path.open() as handle:
        for line in handle:
            chrom, size = line.rstrip("\n").split("\t")
            sizes[chrom] = int(size)
    return sizes


def adjust_start(start: int, chrom: str, chromsizes: Dict[str, int]) -> int:
    length = chromsizes[chrom]
    half_span = WINDOW_SPAN // 2
    window_start = start - half_span
    if window_start < 0:
        window_start = 0
    max_start = max(0, length - WINDOW_SPAN)
    if window_start > max_start:
        window_start = max_start
    return int(window_start)


def load_candidate_starts(
    peaks_path: Path,
    allowed_chroms: Iterable[str],
    chromsizes: Dict[str, int],
) -> List[Tuple[str, int]]:
    allowed = set(allowed_chroms)
    regions: List[Tuple[str, int]] = []
    with peaks_path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom not in allowed or chrom not in chromsizes:
                continue
            start = int(parts[1])
            end = int(parts[2])
            centre = (start + end) // 2
            window_start = adjust_start(centre, chrom, chromsizes)
            regions.append((chrom, window_start))
    return regions


def load_base_regions(path: Path) -> List[Dict[str, object]]:
    with path.open() as handle:
        payload = json.load(handle)
    return list(payload["regions"])


def main() -> None:
    args = parse_args()
    test_chroms = load_test_chromosomes(args.test_folds)
    chromsizes = load_chromsizes(args.chromsizes)
    candidates = load_candidate_starts(args.peaks_bed, test_chroms, chromsizes)
    base_regions = load_base_regions(args.base_regions)

    base_set = {(str(r["chr"]), int(r["start"])) for r in base_regions}

    unique_candidates = [region for region in candidates if region not in base_set]
    rng = random.Random(args.seed)
    rng.shuffle(unique_candidates)

    random_regions: List[Dict[str, object]] = []
    for idx, (chrom, start) in enumerate(unique_candidates[: args.random_count], start=1):
        random_regions.append(
            {
                "chr": chrom,
                "start": int(start),
                "region_name": f"random_region_{idx}",
            }
        )

    payload = {"regions": base_regions + random_regions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
