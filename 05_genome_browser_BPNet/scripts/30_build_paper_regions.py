#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def _pick_name(row, fallback: str) -> str:
    for key in ("region_name", "name", "label"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build genome browser regions centered on paper loci."
    )
    parser.add_argument("--regions-tsv", required=True)
    parser.add_argument("--output-bed", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--window-span", type=int, default=1000)
    args = parser.parse_args()

    regions = pd.read_csv(args.regions_tsv, sep="\t")
    window_span = int(args.window_span)
    half_span = window_span // 2

    bed_rows = []
    json_regions = []
    for idx, row in regions.iterrows():
        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])
        mid = (start + end) // 2
        window_start = max(0, mid - half_span)
        window_end = window_start + window_span
        fallback_name = f"paper_region_{idx + 1}"
        name = _pick_name(row, fallback_name)
        bed_rows.append((chrom, window_start, window_end, name))
        json_regions.append(
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
    with json_path.open("w") as handle:
        json.dump({"window_span": window_span, "regions": json_regions}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
