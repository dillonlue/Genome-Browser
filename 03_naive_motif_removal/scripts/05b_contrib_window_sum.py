#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyBigWig



def _read_chrom_sizes(path: Path) -> List[Tuple[str, int]]:
    sizes = []
    with path.open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, size_str = line.rstrip("\n").split("\t")[:2]
            sizes.append((chrom, int(size_str)))
    return sizes


def _write_bigwig(output_path: Path, entries, chrom_sizes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bw = pyBigWig.open(str(output_path), "w")
    bw.addHeader(chrom_sizes)
    if entries:
        chrom_order = {chrom: idx for idx, (chrom, _) in enumerate(chrom_sizes)}
        entries.sort(key=lambda row: (chrom_order.get(row[0], 1_000_000), row[1]))
        chroms, starts, ends, values = zip(*entries)
        bw.addEntries(list(chroms), list(starts), ends=list(ends), values=list(values))
    bw.close()


def _window_sum_track(contrib: np.ndarray, width: int) -> np.ndarray:
    seq_len = contrib.shape[0]
    scores = np.zeros(seq_len, dtype=np.float32)
    if width <= 0 or width > seq_len:
        return scores
    center_offset = width // 2
    for offset in range(seq_len - width + 1):
        window_sum = float(contrib[offset : offset + width].sum())
        center_idx = offset + center_offset
        if 0 <= center_idx < seq_len:
            scores[center_idx] = window_sum
    return scores


def _collect_entries(
    top: pd.DataFrame,
    per_region_scores: List[np.ndarray],
    chrom_sizes,
):
    size_map = {chrom: size for chrom, size in chrom_sizes}
    aggregate = {}
    for idx, row in enumerate(top.itertuples(index=False)):
        chrom = str(row.chrom)
        chrom_size = size_map.get(chrom)
        if chrom_size is None:
            continue
        start = int(row.start)
        end = int(row.end)
        mid = (start + end) // 2
        scores = per_region_scores[idx]
        half_span = scores.shape[0] // 2
        window_start = mid - half_span
        for offset, value in enumerate(scores):
            pos = window_start + offset
            if pos < 0 or pos >= chrom_size:
                continue
            key = (chrom, pos)
            if key in aggregate:
                total, count = aggregate[key]
                aggregate[key] = (total + float(value), count + 1)
            else:
                aggregate[key] = (float(value), 1)
    entries = [
        (chrom, pos, pos + 1, total / count)
        for (chrom, pos), (total, count) in aggregate.items()
    ]
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build centered contribution window-sum bigWig for top windows."
    )
    parser.add_argument("--attr-npz", required=True)
    parser.add_argument("--ohe-npz", required=True)
    parser.add_argument("--top-tsv", required=True)
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--output-bw", required=True)
    parser.add_argument("--window-span", type=int, default=50)
    args = parser.parse_args()

    top = pd.read_csv(args.top_tsv, sep="\t").sort_values("rank")
    attr_payload = np.load(args.attr_npz, allow_pickle=True)
    ohe_payload = np.load(args.ohe_npz, allow_pickle=True)
    attr = attr_payload[attr_payload.files[0]]
    ohe = ohe_payload[ohe_payload.files[0]]
    combined = attr * ohe

    width = int(args.window_span)

    per_region_scores: List[np.ndarray] = []
    for idx in range(combined.shape[0]):
        base_contrib = combined[idx].sum(axis=0)
        per_region_scores.append(_window_sum_track(base_contrib, width))

    chrom_sizes = _read_chrom_sizes(Path(args.chrom_sizes))
    entries = _collect_entries(top, per_region_scores, chrom_sizes)
    _write_bigwig(Path(args.output_bw), entries, chrom_sizes)


if __name__ == "__main__":
    main()
