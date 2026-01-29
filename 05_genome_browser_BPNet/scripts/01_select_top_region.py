#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEFAULT_TEST_CHROMS = ("chr1", "chr8", "chr9")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top regions by total ChIP-nexus counts from a DeepLIFT H5."
    )
    parser.add_argument("--deeplift-h5", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--random-n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--test-chroms", nargs="+", default=list(DEFAULT_TEST_CHROMS))
    args = parser.parse_args()

    h5_path = Path(args.deeplift_h5)
    with h5py.File(h5_path, "r") as handle:
        chroms = handle["metadata/range/chr"][:]
        starts = handle["metadata/range/start"][:]
        ends = handle["metadata/range/end"][:]

        total_counts = None
        for tf in ("Oct4", "Sox2", "Nanog", "Klf4"):
            dataset = f"targets/{tf}/profile"
            if dataset not in handle:
                raise KeyError(f"Missing dataset '{dataset}' in {h5_path}")
            counts = handle[dataset][:].sum(axis=(1, 2))
            total_counts = counts if total_counts is None else total_counts + counts

    if total_counts is None:
        raise ValueError("No target profiles found to compute counts.")

    chroms = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in chroms]
    chroms_arr = np.array(chroms)
    all_idx = np.arange(chroms_arr.size)
    if args.test_chroms:
        test_set = set(args.test_chroms)
        mask = np.array([c in test_set for c in chroms_arr], dtype=bool)
        if not mask.any():
            raise ValueError("No regions found on requested test chromosomes.")
        chroms_arr = chroms_arr[mask]
        starts = starts[mask]
        ends = ends[mask]
        total_counts = total_counts[mask]
        peak_indices = all_idx[mask]
    else:
        peak_indices = all_idx

    order = np.argsort(total_counts)[::-1]
    top_idx = order[: args.top_n]
    remaining = order[args.top_n :]
    rng = np.random.default_rng(args.seed)
    if args.random_n > 0:
        if args.random_n > remaining.shape[0]:
            raise ValueError("random-n exceeds available remaining regions.")
        random_idx = rng.choice(remaining, size=args.random_n, replace=False)
    else:
        random_idx = np.array([], dtype=order.dtype)

    top_table = pd.DataFrame(
        {
            "chrom": chroms_arr[top_idx],
            "start": starts[top_idx],
            "end": ends[top_idx],
            "total_counts": total_counts[top_idx],
            "peak_index": peak_indices[top_idx],
        }
    ).reset_index(drop=True)
    top_table["subset"] = "top"
    top_table["subset_rank"] = top_table.index + 1

    random_table = pd.DataFrame(
        {
            "chrom": chroms_arr[random_idx],
            "start": starts[random_idx],
            "end": ends[random_idx],
            "total_counts": total_counts[random_idx],
            "peak_index": peak_indices[random_idx],
        }
    ).reset_index(drop=True)
    random_table["subset"] = "random"
    random_table["subset_rank"] = random_table.index + 1

    table = pd.concat([top_table, random_table], ignore_index=True)
    table["rank"] = table.index + 1

    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table[
        [
            "rank",
            "subset",
            "subset_rank",
            "chrom",
            "start",
            "end",
            "total_counts",
            "peak_index",
        ]
    ].to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
