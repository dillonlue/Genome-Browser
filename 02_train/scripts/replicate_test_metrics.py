#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from bpnetlite import performance
from tangermeme.io import extract_loci


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / path).resolve()


def _load_evaluate_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def _extract_dataset(params: dict, base_dir: Path, chroms_override: Path | None):
    chroms = params.get("validation_chroms", params["chroms"])
    if chroms_override is not None:
        override_payload = json.loads(chroms_override.read_text())
        chroms = override_payload.get("chroms", chroms)

    loci_paths = [_resolve_path(path, base_dir) for path in params["loci"]]
    signals = [_resolve_path(path, base_dir) for path in params["signals"]]
    controls = [_resolve_path(path, base_dir) for path in params["controls"]]
    sequences = _resolve_path(params["sequences"], base_dir)

    X, y, X_ctl, _ = extract_loci(
        sequences=str(sequences),
        signals=[str(path) for path in signals],
        in_signals=[str(path) for path in controls],
        loci=[str(path) for path in loci_paths],
        chroms=chroms,
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=0,
        exclusion_lists=params["exclusion_lists"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        return_mask=True,
        verbose=True,
    )
    return X, y, X_ctl


def _format_results(metrics: dict) -> str:
    headers = [
        "profile_mnll",
        "profile_jsd",
        "profile_pearson",
        "profile_spearman",
        "count_pearson",
        "count_spearman",
        "count_mse",
    ]
    header_line = "\t".join(headers)
    values = [metrics[name] for name in headers]
    value_line = "\t".join(f"{value:.15g}" for value in values)
    return f"{header_line}\n{value_line}\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute BPNet performance metrics on the test set."
    )
    parser.add_argument("--evaluate-json", required=True)
    parser.add_argument("--chroms-json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kernel-sigma", type=int, default=7)
    parser.add_argument("--kernel-width", type=int, default=81)
    parser.add_argument("--smooth-true", action="store_true")
    parser.add_argument("--smooth-predictions", action="store_true")
    parser.add_argument("--output-tsv")
    args = parser.parse_args()

    evaluate_path = Path(args.evaluate_json)
    params = _load_evaluate_json(evaluate_path)
    base_dir = evaluate_path.parent.resolve()

    X, y, X_ctl = _extract_dataset(
        params, base_dir, Path(args.chroms_json) if args.chroms_json else None
    )

    model_path = _resolve_path(params["model"], base_dir)
    device = torch.device(args.device)
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    profile_sums = {
        "profile_mnll": 0.0,
        "profile_jsd": 0.0,
        "profile_pearson": 0.0,
        "profile_spearman": 0.0,
    }
    profile_count = 0

    pred_logcounts_all = []
    true_logcounts_all = []

    with torch.no_grad():
        for offset in range(0, X.shape[0], args.batch_size):
            batch_seq = X[offset : offset + args.batch_size].float().to(device)
            batch_true = y[offset : offset + args.batch_size].to(device)
            if X_ctl.numel() > 0:
                batch_ctl = X_ctl[offset : offset + args.batch_size].float().to(device)
                logits, logcounts = model(batch_seq, batch_ctl)
            else:
                logits, logcounts = model(batch_seq)

            measures = performance.calculate_performance_measures(
                logits,
                batch_true,
                logcounts,
                kernel_sigma=args.kernel_sigma,
                kernel_width=args.kernel_width,
                smooth_true=args.smooth_true,
                smooth_predictions=args.smooth_predictions,
                measures=list(profile_sums.keys()),
            )

            batch_size = batch_true.shape[0]
            for key in profile_sums:
                profile_sums[key] += measures[key].sum().item()
            profile_count += batch_size

            true_logcounts = torch.log(batch_true.sum(dim=(-1, -2)) + 1).cpu()
            pred_logcounts_all.append(logcounts.cpu())
            true_logcounts_all.append(true_logcounts)

    pred_logcounts = torch.cat(pred_logcounts_all, dim=0)
    true_logcounts = torch.cat(true_logcounts_all, dim=0)

    count_pearson = performance.pearson_corr(pred_logcounts.T, true_logcounts.T)
    count_spearman = performance.spearman_corr(pred_logcounts.T, true_logcounts.T)
    count_mse = performance.mean_squared_error(pred_logcounts.T, true_logcounts.T)

    metrics = {
        "profile_mnll": profile_sums["profile_mnll"] / profile_count,
        "profile_jsd": profile_sums["profile_jsd"] / profile_count,
        "profile_pearson": profile_sums["profile_pearson"] / profile_count,
        "profile_spearman": profile_sums["profile_spearman"] / profile_count,
        "count_pearson": count_pearson.item(),
        "count_spearman": count_spearman.item(),
        "count_mse": count_mse.item(),
    }

    output_text = _format_results(metrics)
    if args.output_tsv:
        output_path = Path(args.output_tsv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text)
    else:
        print(output_text, end="")


if __name__ == "__main__":
    main()
