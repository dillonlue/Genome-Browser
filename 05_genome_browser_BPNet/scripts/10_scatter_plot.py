#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import plotly.express as px

DEFAULT_TEST_CHROMS = ("chr1", "chr8", "chr9")


def _sum_bigwig(bw, chrom: str, start: int, end: int) -> float:
    stats = bw.stats(chrom, start, end, type="sum")
    if not stats or stats[0] is None:
        return 0.0
    return float(stats[0])


def _load_peak_regions(peak_paths, window_span: int, test_chroms) -> pd.DataFrame:
    frames = []
    for path in peak_paths:
        peaks = pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
        )
        summit = ((peaks["start"] + peaks["end"]) // 2).astype(int)
        region_start = summit - (window_span // 2)
        region_end = region_start + window_span
        frames.append(
            pd.DataFrame(
                {
                    "chrom": peaks["chrom"],
                    "start": region_start,
                    "end": region_end,
                }
            )
        )
    regions = pd.concat(frames, ignore_index=True)
    regions["start"] = regions["start"].clip(lower=0)
    regions = regions.drop_duplicates(ignore_index=True)
    if test_chroms:
        regions = regions.loc[regions["chrom"].isin(test_chroms)].reset_index(drop=True)
    return regions


def _common_chrom_lengths(bigwigs) -> dict:
    common = None
    for bw in bigwigs:
        chroms = bw.chroms()
        if common is None:
            common = dict(chroms)
            continue
        next_common = {}
        for chrom, length in common.items():
            if chrom in chroms:
                next_common[chrom] = min(length, chroms[chrom])
        common = next_common
    return common or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot total counts versus loss for positive regions."
    )
    parser.add_argument("--peaks-beds", nargs="+", required=True)
    parser.add_argument("--observed-bws", nargs="+", required=True)
    parser.add_argument("--predicted-bws", nargs="+", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--window-span", type=int, default=1000)
    parser.add_argument("--test-chroms", nargs="+", default=list(DEFAULT_TEST_CHROMS))
    args = parser.parse_args()

    raw = []
    for entry in args.test_chroms:
        raw.extend([part for part in entry.split(",") if part])
    test_chroms = {
        chrom if chrom.startswith("chr") else f"chr{chrom}"
        for chrom in raw
    }

    regions = _load_peak_regions(args.peaks_beds, args.window_span, test_chroms)
    if regions.empty:
        raise ValueError("No regions remain after test-chrom filtering.")

    observed_handles = [pyBigWig.open(path) for path in args.observed_bws]
    predicted_handles = [pyBigWig.open(path) for path in args.predicted_bws]

    try:
        chrom_sizes = _common_chrom_lengths(observed_handles + predicted_handles)
        x_values = []
        loss_values = []
        for _, row in regions.iterrows():
            chrom = str(row["chrom"])
            if chrom not in chrom_sizes:
                continue
            chrom_size = int(chrom_sizes[chrom])
            start = max(0, int(row["start"]))
            end = min(chrom_size, int(row["end"]))
            if end <= start:
                continue

            observed_total = sum(
                _sum_bigwig(bw, chrom, start, end) for bw in observed_handles
            )
            predicted_total = sum(
                _sum_bigwig(bw, chrom, start, end) for bw in predicted_handles
            )

            observed_total = max(0.0, observed_total)
            predicted_total = max(0.0, predicted_total)
            loss = (np.log1p(predicted_total) - np.log1p(observed_total)) ** 2

            x_values.append(observed_total)
            loss_values.append(loss)
    finally:
        for bw in observed_handles + predicted_handles:
            bw.close()

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_plot = np.asarray(x_values, dtype=float)
    y_plot = np.asarray(loss_values, dtype=float)
    mask = x_plot > 0
    plot_df = pd.DataFrame(
        {
            "total_count": x_plot[mask],
            "loss": y_plot[mask],
        }
    )

    fig = px.scatter(
        plot_df,
        x="total_count",
        y="loss",
        title=f"Positive regions: {len(plot_df)}",
        log_x=True,
        labels={
            "total_count": "Total count in region",
            "loss": "Loss",
        },
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
