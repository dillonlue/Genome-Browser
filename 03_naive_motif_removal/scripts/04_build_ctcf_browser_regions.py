#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build genome browser region configs for top sequences."
    )
    parser.add_argument("--top-tsv", required=True)
    parser.add_argument("--output-bed", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--window-span", type=int, default=2114)
    parser.add_argument("--region-prefix", default="bpnet_top_rank")
    args = parser.parse_args()

    top = pd.read_csv(args.top_tsv, sep="\t").sort_values("rank")
    window_span = int(args.window_span)
    half_span = window_span // 2

    bed_rows = []
    regions = []
    has_subset = "subset" in top.columns
    has_subset_rank = "subset_rank" in top.columns
    for _, row in top.iterrows():
        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])
        mid = (start + end) // 2
        window_start = mid - half_span
        window_end = window_start + window_span
        rank = int(row["rank"])
        if has_subset:
            subset = str(row["subset"]).lower()
            subset_rank = int(row["subset_rank"]) if has_subset_rank else rank
            name = f"bpnet_{subset}_rank{subset_rank}"
        else:
            name = f"{args.region_prefix}{rank}"
        bed_rows.append((chrom, window_start, window_end, name))
        regions.append(
            {
                "chr": chrom,
                "start": window_start,
                "end": window_end,
                "region_name": name,
            }
        )

    bed_path = Path(args.output_bed)
    bed_path.parent.mkdir(parents=True, exist_ok=True)
    with bed_path.open("w") as handle:
        for chrom, start, end, name in bed_rows:
            handle.write(f"{chrom}\t{start}\t{end}\t{name}\n")

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"window_span": window_span, "regions": regions}
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
