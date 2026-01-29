#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEFAULT_TEST_CHROMS = ("chr1", "chr8", "chr9")

PROFILE_DATASET = "profile/wn"
COUNTS_DATASET = "counts/pre-act"
TASKS = ("Oct4", "Sox2", "Nanog", "Klf4")


def _load_tasks(
    handle: h5py.File, require_profile: bool, require_counts: bool
) -> list[str]:
    missing = []
    for name in TASKS:
        profile_key = f"hyp_imp/{name}/{PROFILE_DATASET}"
        counts_key = f"hyp_imp/{name}/{COUNTS_DATASET}"
        has_profile = profile_key in handle
        has_counts = counts_key in handle
        if require_profile and not has_profile:
            missing.append(profile_key)
        if require_counts and not has_counts:
            missing.append(counts_key)
    if missing:
        missing_fmt = ", ".join(missing)
        raise KeyError(f"Missing required hyp_imp datasets: {missing_fmt}")
    return list(TASKS)


def _format_template(template: str, task: str) -> Path:
    if "{task}" not in template:
        raise ValueError("Template must include '{task}' placeholder.")
    return Path(template.format(task=task))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subset DeepLIFT attributions for selected regions."
    )
    parser.add_argument("--deeplift-h5", required=True)
    parser.add_argument("--top-tsv", required=True)
    parser.add_argument(
        "--output-attr-npz",
        help="(deprecated) Use --output-attr-profile-npz instead.",
    )
    parser.add_argument("--output-attr-profile-npz")
    parser.add_argument("--output-attr-counts-npz")
    parser.add_argument("--output-attr-profile-template")
    parser.add_argument("--output-attr-counts-template")
    parser.add_argument("--output-ohe-npz", required=True)
    parser.add_argument("--test-chroms", nargs="+", default=list(DEFAULT_TEST_CHROMS))
    args = parser.parse_args()

    profile_out = args.output_attr_profile_npz or args.output_attr_npz
    counts_out = args.output_attr_counts_npz
    profile_template = args.output_attr_profile_template
    counts_template = args.output_attr_counts_template
    if (
        profile_out is None
        and counts_out is None
        and profile_template is None
        and counts_template is None
    ):
        raise ValueError(
            "Provide at least one of --output-attr-profile-npz, --output-attr-counts-npz, "
            "--output-attr-profile-template, or --output-attr-counts-template."
        )

    top = pd.read_csv(args.top_tsv, sep="\t")
    if args.test_chroms:
        top = top[top["chrom"].isin(args.test_chroms)]
        if top.empty:
            raise ValueError("No regions remain after test-chrom filtering.")
    top = top.sort_values("rank")
    peak_indices = top["peak_index"].to_numpy()

    need_profile = profile_out is not None or profile_template is not None
    need_counts = counts_out is not None or counts_template is not None

    h5_path = Path(args.deeplift_h5)
    with h5py.File(h5_path, "r") as handle:
        tasks = _load_tasks(handle, require_profile=need_profile, require_counts=need_counts)
        profile_sum = None
        counts_sum = None
        for task in tasks:
            if need_profile:
                profile_key = f"hyp_imp/{task}/{PROFILE_DATASET}"
                profile = handle[profile_key][:].astype(np.float32)
                selected_profile = profile[peak_indices]
                if profile_sum is None:
                    profile_sum = selected_profile.copy()
                else:
                    profile_sum += selected_profile
                if profile_template is not None:
                    output_profile = _format_template(profile_template, task)
                    output_profile.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(output_profile, np.transpose(selected_profile, (0, 2, 1)))
            if need_counts:
                counts_key = f"hyp_imp/{task}/{COUNTS_DATASET}"
                counts = handle[counts_key][:].astype(np.float32)
                selected_counts = counts[peak_indices]
                if counts_sum is None:
                    counts_sum = selected_counts.copy()
                else:
                    counts_sum += selected_counts
                if counts_template is not None:
                    output_counts = _format_template(counts_template, task)
                    output_counts.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(output_counts, np.transpose(selected_counts, (0, 2, 1)))

        ohe = handle["inputs/seq"][:].astype(np.float32)

    selected_profile = None
    if profile_sum is not None and profile_out is not None:
        selected_profile = np.transpose(profile_sum, (0, 2, 1))

    selected_counts = None
    if counts_sum is not None and counts_out is not None:
        selected_counts = np.transpose(counts_sum, (0, 2, 1))

    selected_ohe = ohe[peak_indices]
    selected_ohe = np.transpose(selected_ohe, (0, 2, 1))

    if selected_profile is not None and profile_out is not None:
        output_profile = Path(profile_out)
        output_profile.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_profile, selected_profile)

    if selected_counts is not None and counts_out is not None:
        output_counts = Path(counts_out)
        output_counts.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_counts, selected_counts)

    output_ohe = Path(args.output_ohe_npz)
    output_ohe.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_ohe, selected_ohe)


if __name__ == "__main__":
    main()
