#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def _plot_histogram(
    values,
    png_path: Path,
    svg_path: Path,
    xlabel: str,
    bins,
    xlim,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.hist(values, bins=bins)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)


def _hist_limits(values, step: int = 10):
    lower = float(np.percentile(values, 2.5))
    upper = float(np.percentile(values, 97.5))
    start = step * np.floor(lower / step)
    end = step * np.ceil(upper / step)
    if start == end:
        start -= step
        end += step
    bins = np.arange(start, end + step, step)
    return bins, (start, end)


def _summary_rows(df: pd.DataFrame, subset: str):
    rows = []
    for metric in ["delta_total_loss", "delta_profile_loss", "delta_count_loss"]:
        values = df[metric].to_numpy()
        rows.append(
            {
                "subset": subset,
                "metric": metric,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render delta-loss histograms for motif removal analysis."
    )
    parser.add_argument("--loss-tsv", required=True)
    parser.add_argument("--main-html", required=True)
    parser.add_argument("--summary-tsv", required=True)
    parser.add_argument("--table-tsv", required=True)
    parser.add_argument("--indv-plots-dir", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    losses = pd.read_csv(args.loss_tsv, sep="\t")
    losses["delta_total_loss"] = losses["motif_total_loss"] - losses["baseline_total_loss"]
    losses["delta_profile_loss"] = (
        losses["motif_profile_loss"] - losses["baseline_profile_loss"]
    )
    losses["delta_profile_loss_plus"] = (
        losses["motif_profile_loss_plus"] - losses["baseline_profile_loss_plus"]
    )
    losses["delta_profile_loss_minus"] = (
        losses["motif_profile_loss_minus"] - losses["baseline_profile_loss_minus"]
    )
    losses["delta_count_loss"] = losses["motif_count_loss"] - losses["baseline_count_loss"]

    losses = losses.sort_values("total_counts", ascending=False)
    top = losses.head(args.top_n)
    top_set = set(top["sequence_rank"].tolist())
    losses["subset"] = np.where(
        losses["sequence_rank"].isin(top_set),
        f"top_{args.top_n}",
        "all",
    )

    indv_dir = Path(args.indv_plots_dir)
    indv_dir.mkdir(parents=True, exist_ok=True)
    all_png = indv_dir / "delta_total_loss_all.png"
    all_svg = indv_dir / "delta_total_loss_all.svg"
    top_png = indv_dir / f"delta_total_loss_top{args.top_n}.png"
    top_svg = indv_dir / f"delta_total_loss_top{args.top_n}.svg"

    bins, xlim = _hist_limits(losses["delta_total_loss"].to_numpy())
    _plot_histogram(
        losses["delta_total_loss"].to_numpy(),
        all_png,
        all_svg,
        "Delta total loss",
        bins,
        xlim,
    )
    _plot_histogram(
        top["delta_total_loss"].to_numpy(),
        top_png,
        top_svg,
        "Delta total loss",
        bins,
        xlim,
    )

    summary_rows = []
    summary_rows.extend(_summary_rows(losses, "all"))
    summary_rows.extend(_summary_rows(top, f"top_{args.top_n}"))
    summary_path = Path(args.summary_tsv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)

    table_path = Path(args.table_tsv)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    losses.to_csv(table_path, sep="\t", index=False)

    main_html = Path(args.main_html)
    main_html.parent.mkdir(parents=True, exist_ok=True)
    all_png_rel = os.path.relpath(all_png, start=main_html.parent)
    all_svg_rel = os.path.relpath(all_svg, start=main_html.parent)
    top_png_rel = os.path.relpath(top_png, start=main_html.parent)
    top_svg_rel = os.path.relpath(top_svg, start=main_html.parent)

    html_rows = []
    for _, row in losses.iterrows():
        html_rows.append(
            "<tr>"
            f"<td>{row['chrom']}:{int(row['start'])}-{int(row['end'])}</td>"
            f"<td>{row['total_counts']:.2f}</td>"
            f"<td>{row['delta_total_loss']:.4f}</td>"
            f"<td>{row['delta_profile_loss']:.4f}</td>"
            f"<td>{row['delta_count_loss']:.4f}</td>"
            f"<td>{int(row['motif_hit_count'])}</td>"
            f"<td>{int(row['motif_bases_kept'])}</td>"
            f"<td>{row['kept_fraction']:.3f}</td>"
            f"<td>{row['subset']}</td>"
            "</tr>"
        )

    main_html.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<title>Motif-only delta histograms</title>",
                "</head>",
                "<body>",
                "<h1>Motif-only delta histograms</h1>",
                "<h2>Delta total loss (all regions)</h2>",
                f"<div><img src=\"{all_png_rel}\" alt=\"Delta total loss (all)\" "
                "style=\"max-width: 100%; height: auto;\"></div>",
                f"<div><a href=\"{all_svg_rel}\">SVG version</a></div>",
                f"<h2>Delta total loss (top {args.top_n} regions)</h2>",
                f"<div><img src=\"{top_png_rel}\" alt=\"Delta total loss (top)\" "
                "style=\"max-width: 100%; height: auto;\"></div>",
                f"<div><a href=\"{top_svg_rel}\">SVG version</a></div>",
                "<h2>Delta table</h2>",
                "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">",
                "<tr>",
                "<th>Region</th>",
                "<th>Total counts</th>",
                "<th>Delta loss</th>",
                "<th>Delta profile loss</th>",
                "<th>Delta count loss</th>",
                "<th>Motif hits</th>",
                "<th>Bases kept</th>",
                "<th>Kept fraction</th>",
                "<th>Subset</th>",
                "</tr>",
                *html_rows,
                "</table>",
                "</body>",
                "</html>",
            ]
        )
    )


if __name__ == "__main__":
    main()
