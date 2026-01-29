#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np


def _write_header(handle) -> None:
    handle.write("MEME version 4\n\n")
    handle.write("ALPHABET= ACGT\n\n")
    handle.write("strands: + -\n\n")
    handle.write("Background letter frequencies:\n")
    handle.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")


def _iter_patterns(handle):
    for group_name in ("pos_patterns", "neg_patterns"):
        if group_name not in handle:
            continue
        group = handle[group_name]
        for pattern_name in group.keys():
            yield group_name, pattern_name, group[pattern_name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TF-MoDISco motifs into a MEME file."
    )
    parser.add_argument("--modisco-h5", required=True)
    parser.add_argument("--output-meme", required=True)
    args = parser.parse_args()

    output_meme = Path(args.output_meme)
    output_meme.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.modisco_h5, "r") as handle, output_meme.open("w") as out:
        _write_header(out)
        for group_name, pattern_name, pattern in _iter_patterns(handle):
            seq = np.asarray(pattern["sequence"], dtype=float)
            motif_id = f"{group_name}.{pattern_name}"
            if "seqlets" in pattern and "start" in pattern["seqlets"]:
                nsites = int(pattern["seqlets"]["start"].shape[0])
            else:
                nsites = int(seq.shape[0])
            out.write(f"MOTIF {motif_id}\n")
            out.write(
                "letter-probability matrix: alength= 4 "
                f"w= {seq.shape[0]} nsites= {nsites} E= 0\n"
            )
            for row in seq:
                out.write(" ".join(f"{val:.6f}" for val in row) + "\n")
            out.write("\n")


if __name__ == "__main__":
    main()
