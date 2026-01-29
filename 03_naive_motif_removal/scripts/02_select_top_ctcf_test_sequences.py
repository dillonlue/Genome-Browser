#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tangermeme.io import extract_loci


def _resolve_path(path_str, base_dir):
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top and random test-set sequences by total binding counts."
    )
    parser.add_argument(
        "--evaluate-json",
        required=True,
        help="BPNet evaluate JSON containing loci, signals, controls, and chroms.",
    )
    parser.add_argument(
        "--chroms-json",
        help="Optional JSON file with a 'chroms' list to override evaluate chroms.",
    )
    parser.add_argument(
        "--attr-idxs",
        help="Optional attribution idxs file to align extracted loci with attributions.",
    )
    parser.add_argument(
        "--output-tsv",
        required=True,
        help="Output TSV of top sequences and their coordinates.",
    )
    parser.add_argument(
        "--output-npz",
        required=True,
        help="Output NPZ with sequences, controls, signals, and metadata.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--random-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    evaluate_path = Path(args.evaluate_json)
    with evaluate_path.open("r") as handle:
        params = json.load(handle)

    base_dir = evaluate_path.parent.resolve()
    validation_chroms = params.get("validation_chroms", params["chroms"])
    if args.chroms_json:
        override_payload = json.loads(Path(args.chroms_json).read_text())
        validation_chroms = override_payload.get("chroms", validation_chroms)
    loci_paths = [_resolve_path(path, base_dir) for path in params["loci"]]
    signals = [_resolve_path(path, base_dir) for path in params["signals"]]
    controls = [_resolve_path(path, base_dir) for path in params["controls"]]
    sequences = _resolve_path(params["sequences"], base_dir)

    examples = extract_loci(
        sequences=str(sequences),
        signals=[str(path) for path in signals],
        in_signals=[str(path) for path in controls],
        loci=[str(path) for path in loci_paths],
        chroms=validation_chroms,
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=0,
        exclusion_lists=params["exclusion_lists"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        return_mask=True,
        verbose=True,
    )

    X, y, X_ctl, mask = examples

    bed = pd.read_csv(loci_paths[0], sep="\t", header=None, comment="#")
    bed = bed[bed[0].isin(validation_chroms)].reset_index(drop=True)
    mask_array = np.asarray(mask)
    if mask_array.dtype == bool:
        bed = bed[mask_array]
    else:
        bed = bed.iloc[mask_array]
    bed = bed.reset_index(drop=True)

    if args.attr_idxs:
        idxs = np.load(args.attr_idxs, allow_pickle=False)
        if idxs.dtype == bool:
            if idxs.shape[0] != X.shape[0]:
                raise ValueError("attr idx mask length does not match extracted loci.")
            keep = idxs
        else:
            keep = np.zeros(X.shape[0], dtype=bool)
            keep[idxs] = True
        X = X[keep]
        y = y[keep]
        X_ctl = X_ctl[keep]
        bed = bed.iloc[keep].reset_index(drop=True)

    total_counts = y.sum(dim=(1, 2)).numpy()
    table = bed[[0, 1, 2]].copy()
    table.columns = ["chrom", "start", "end"]
    table["total_counts"] = total_counts
    table["peak_index"] = np.arange(len(table))
    table = table.sort_values("total_counts", ascending=False)
    table = table.drop_duplicates(subset=["chrom", "start", "end"], keep="first")
    top_bed = table.head(args.top_n).copy()
    top_bed["subset"] = "top"
    top_bed["subset_rank"] = np.arange(1, len(top_bed) + 1)

    remaining = table.iloc[args.top_n:].copy()
    if args.random_n > 0:
        random_bed = remaining.sample(n=args.random_n, random_state=args.seed)
    else:
        random_bed = remaining.head(0).copy()
    random_bed["subset"] = "random"
    random_bed["subset_rank"] = np.arange(1, len(random_bed) + 1)

    combined = pd.concat([top_bed, random_bed], ignore_index=True)
    combined["rank"] = np.arange(1, len(combined) + 1)

    output_tsv = Path(args.output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    combined[
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
    ].to_csv(output_tsv, sep="\t", index=False)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        sequences=X[combined["peak_index"].to_numpy()].numpy(),
        controls=X_ctl[combined["peak_index"].to_numpy()].numpy(),
        signals=y[combined["peak_index"].to_numpy()].numpy(),
        chroms=combined["chrom"].to_numpy(),
        starts=combined["start"].to_numpy(),
        ends=combined["end"].to_numpy(),
        total_counts=combined["total_counts"].to_numpy(),
        peak_indices=combined["peak_index"].to_numpy(),
        subset=combined["subset"].to_numpy(),
        subset_ranks=combined["subset_rank"].to_numpy(),
    )


if __name__ == "__main__":
    main()
