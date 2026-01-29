#!/usr/bin/env python3
"""
Convert a BED file of regions into a genome browser region config JSON.

Each BED entry becomes an element in the ``regions`` list containing the
chromosome, start coordinate, and an auto-generated ``region_name`` that
combines a user-provided prefix with the 1-based index of the region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bed", required=True, type=Path, help="BED file to convert.")
    parser.add_argument("--output", required=True, type=Path, help="Destination JSON region config.")
    parser.add_argument(
        "--region-prefix",
        default="region",
        help="Prefix for auto-generated region names (default: region).",
    )
    return parser.parse_args()


def load_bed_entries(path: Path) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, *_ = line.rstrip("\n").split("\t")
            entries.append((chrom, int(start)))
    return entries


def main() -> None:
    args = parse_args()
    entries = load_bed_entries(args.input_bed)
    regions = [
        {
            "chr": chrom,
            "start": start,
            "region_name": f"{args.region_prefix}_{idx + 1}",
        }
        for idx, (chrom, start) in enumerate(entries)
    ]
    payload = {"regions": regions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
