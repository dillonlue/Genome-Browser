#!/usr/bin/env python3
import argparse
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _read_motif_hits(path: Path) -> Dict[str, List[List[str]]]:
    hits: Dict[str, List[List[str]]] = {}
    with path.open("r") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom = parts[0]
            hits.setdefault(chrom, []).append(parts)
    for chrom in hits:
        hits[chrom].sort(key=lambda row: int(row[1]))
    return hits


def _load_regions(table_path: Path, window_span: int):
    regions = pd.read_csv(table_path, sep="\t")
    if "window_start" in regions.columns and "window_end" in regions.columns:
        regions = regions.copy()
    else:
        half_span = window_span // 2
        mids = (regions["start"].astype(int) + regions["end"].astype(int)) // 2
        regions = regions.copy()
        regions["window_start"] = mids - half_span
        regions["window_end"] = regions["window_start"] + window_span
    return regions


def _build_region_index(regions: pd.DataFrame):
    per_chrom = {}
    for _, row in regions.iterrows():
        chrom = str(row["chrom"])
        per_chrom.setdefault(chrom, []).append(
            (
                int(row["window_start"]),
                int(row["window_end"]),
                int(row["peak_index"]),
            )
        )
    for chrom in per_chrom:
        per_chrom[chrom].sort(key=lambda item: item[0])
    return per_chrom


def _motif_center(parts: List[str]) -> int:
    start = int(parts[1])
    end = int(parts[2])
    return (start + end) // 2


def _window_sum_at_center(base_contrib: np.ndarray, center: int, width: int) -> float:
    seq_len = int(base_contrib.shape[0])
    half = width // 2
    start = center - half
    end = start + width
    if start < 0 or end > seq_len:
        return 0.0
    return float(base_contrib[start:end].sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter motif hits by contribution window-sum values."
    )
    parser.add_argument("--motif-bed", required=True)
    parser.add_argument("--regions-tsv", required=True)
    parser.add_argument("--attr-npz", required=True)
    parser.add_argument("--ohe-npz", required=True)
    parser.add_argument("--output-bed", required=True)
    parser.add_argument("--window-span", type=int, default=2114)
    parser.add_argument("--window-sum-span", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    regions = _load_regions(Path(args.regions_tsv), args.window_span)
    region_index = _build_region_index(regions)

    attr_payload = np.load(args.attr_npz, allow_pickle=True)
    ohe_payload = np.load(args.ohe_npz, allow_pickle=True)
    attr = attr_payload[attr_payload.files[0]]
    ohe = ohe_payload[ohe_payload.files[0]]

    hits = _read_motif_hits(Path(args.motif_bed))

    cache: Dict[int, np.ndarray] = {}
    kept_lines: List[str] = []

    for chrom, rows in hits.items():
        regions_for_chrom = region_index.get(chrom, [])
        if not regions_for_chrom:
            continue
        starts = [item[0] for item in regions_for_chrom]
        for parts in rows:
            center = _motif_center(parts)
            idx = bisect_right(starts, center) - 1
            if idx < 0:
                continue
            window_start, window_end, peak_index = regions_for_chrom[idx]
            if center >= window_end:
                continue
            rel_center = center - window_start
            if peak_index not in cache:
                combined = attr[peak_index] * ohe[peak_index]
                cache[peak_index] = combined.sum(axis=0)
            base_contrib = cache[peak_index]
            score = _window_sum_at_center(base_contrib, rel_center, args.window_sum_span)
            if score > args.threshold:
                kept_lines.append("\t".join(parts))

    output_path = Path(args.output_bed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for line in kept_lines:
            handle.write(line + "\n")


if __name__ == "__main__":
    main()
