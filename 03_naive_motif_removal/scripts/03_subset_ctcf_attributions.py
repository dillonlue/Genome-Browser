#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subset DeepSHAP arrays for the top sequences."
    )
    parser.add_argument("--top-tsv", required=True)
    parser.add_argument("--attr-npz", required=True)
    parser.add_argument("--ohe-npz", required=True)
    parser.add_argument("--output-attr-npz", required=True)
    parser.add_argument("--output-ohe-npz", required=True)
    args = parser.parse_args()

    top = pd.read_csv(args.top_tsv, sep="\t").sort_values("rank")
    peak_indices = top["peak_index"].to_numpy()

    attr_payload = np.load(args.attr_npz, allow_pickle=True)
    ohe_payload = np.load(args.ohe_npz, allow_pickle=True)
    attr = attr_payload[attr_payload.files[0]]
    ohe = ohe_payload[ohe_payload.files[0]]

    selected_attr = attr[peak_indices]
    selected_ohe = ohe[peak_indices]

    output_attr = Path(args.output_attr_npz)
    output_attr.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_attr, selected_attr)

    output_ohe = Path(args.output_ohe_npz)
    output_ohe.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_ohe, selected_ohe)


if __name__ == "__main__":
    main()
