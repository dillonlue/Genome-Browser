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
        description="Extract all test sequences and build 2114bp windows."
    )
    parser.add_argument("--evaluate-json", required=True)
    parser.add_argument(
        "--chroms-json",
        help="Optional JSON file with a 'chroms' list to override evaluate chroms.",
    )
    parser.add_argument(
        "--attr-idxs",
        help="Optional attribution idxs file to align extracted loci with attributions.",
    )
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-bed", required=True)
    parser.add_argument("--window-span", type=int, default=2114)
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
    table = table.reset_index(drop=True)
    table["sequence_rank"] = np.arange(1, len(table) + 1)

    window_span = int(args.window_span)
    half_span = window_span // 2
    mids = (table["start"].astype(int) + table["end"].astype(int)) // 2
    table["window_start"] = mids - half_span
    table["window_end"] = table["window_start"] + window_span

    output_tsv = Path(args.output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    table[
        [
            "sequence_rank",
            "chrom",
            "start",
            "end",
            "window_start",
            "window_end",
            "total_counts",
            "peak_index",
        ]
    ].to_csv(output_tsv, sep="\t", index=False)

    output_bed = Path(args.output_bed)
    output_bed.parent.mkdir(parents=True, exist_ok=True)
    with output_bed.open("w") as handle:
        for _, row in table.iterrows():
            name = f"test_rank{int(row['sequence_rank'])}"
            handle.write(
                f"{row['chrom']}\t{int(row['window_start'])}\t"
                f"{int(row['window_end'])}\t{name}\n"
            )

    peak_indices = table["peak_index"].to_numpy()
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        sequences=X[peak_indices].numpy(),
        controls=X_ctl[peak_indices].numpy(),
        signals=y[peak_indices].numpy(),
        chroms=table["chrom"].to_numpy(),
        starts=table["start"].to_numpy(),
        ends=table["end"].to_numpy(),
        window_starts=table["window_start"].to_numpy(),
        window_ends=table["window_end"].to_numpy(),
        total_counts=table["total_counts"].to_numpy(),
        peak_indices=table["peak_index"].to_numpy(),
        sequence_ranks=table["sequence_rank"].to_numpy(),
    )


if __name__ == "__main__":
    main()
