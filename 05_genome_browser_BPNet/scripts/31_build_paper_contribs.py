#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np

TASKS = ("Oct4", "Sox2", "Nanog", "Klf4")
PROFILE_DATASET = "profile/wn"
COUNTS_DATASET = "counts/pre-act"


def _decode_chroms(raw):
    return [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in raw]


def _read_bed(path: Path):
    regions = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, *_ = line.rstrip("\n").split("\t")
            regions.append((chrom, int(start), int(end)))
    return regions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined BPNet contribution arrays for paper regions."
    )
    parser.add_argument("--deeplift-h5", required=True)
    parser.add_argument("--regions-bed", required=True)
    parser.add_argument("--output-ohe-npz", required=True)
    parser.add_argument("--output-agg-attr-npz", required=True)
    parser.add_argument("--output-attr-template", required=True)
    args = parser.parse_args()

    bed_path = Path(args.regions_bed)
    regions = _read_bed(bed_path)
    if not regions:
        raise ValueError(f"No regions found in {bed_path}")

    deeplift_path = Path(args.deeplift_h5)
    with h5py.File(deeplift_path, "r") as handle:
        chroms = _decode_chroms(handle["metadata/range/chr"][:])
        starts = handle["metadata/range/start"][:]
        ends = handle["metadata/range/end"][:]
        index_map = {
            (chrom, int(start), int(end)): idx
            for idx, (chrom, start, end) in enumerate(zip(chroms, starts, ends))
        }

        window_span = regions[0][2] - regions[0][1]
        for chrom, start, end in regions:
            if end - start != window_span:
                raise ValueError("All regions must have the same window span.")

        agg = np.zeros((len(regions), 4, window_span), dtype=np.float32)
        per_task = {
            task: np.zeros((len(regions), 4, window_span), dtype=np.float32) for task in TASKS
        }
        ohe = np.zeros((len(regions), 4, window_span), dtype=np.float32)

        for region_idx, (chrom, start, end) in enumerate(regions):
            key = (chrom, start, end)
            h5_idx = index_map.get(key)
            if h5_idx is None:
                print(f"[paper_contribs] Missing region in H5: {chrom}:{start}-{end}")
                continue
            seq = handle["inputs/seq"][h5_idx].astype(np.float32)
            ohe[region_idx] = np.transpose(seq, (1, 0))
            region_sum = np.zeros((4, window_span), dtype=np.float32)
            for task in TASKS:
                profile_key = f"hyp_imp/{task}/{PROFILE_DATASET}"
                counts_key = f"hyp_imp/{task}/{COUNTS_DATASET}"
                profile = handle[profile_key][h5_idx].astype(np.float32)
                counts = handle[counts_key][h5_idx].astype(np.float32)
                combined = profile + counts
                combined_t = np.transpose(combined, (1, 0))
                per_task[task][region_idx] = combined_t
                region_sum += combined_t
            agg[region_idx] = region_sum

    output_ohe = Path(args.output_ohe_npz)
    output_ohe.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_ohe, ohe)

    output_agg = Path(args.output_agg_attr_npz)
    output_agg.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_agg, agg)

    template = args.output_attr_template
    if "{task}" not in template:
        raise ValueError("--output-attr-template must include '{task}'.")
    for task in TASKS:
        output_path = Path(template.format(task=task))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, per_task[task])


if __name__ == "__main__":
    main()
