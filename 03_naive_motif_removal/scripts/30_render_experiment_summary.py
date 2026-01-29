#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def _read_experiment_defs(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open("r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def _plot_violin(values, labels, png_path: Path, svg_path: Path, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 4.5))
    positions = np.arange(1, len(values) + 1)
    ax.violinplot(values, positions=positions, showmeans=False, showextrema=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("Experiment", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render aggregate summary for motif-removal experiments."
    )
    parser.add_argument("--experiment-defs", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    experiment_defs = _read_experiment_defs(Path(args.experiment_defs))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indv_dir = output_dir / "indv_plots"
    indv_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_violin_values = []
    top_violin_values = []
    labels = []

    for exp in experiment_defs:
        exp_id = exp["experiment_id"]
        label = exp["label"]
        exp_dir = Path(args.experiment_root) / exp_id
        delta_table = exp_dir / "output" / "23_delta_histograms" / "delta_table.tsv"

        losses = pd.read_csv(delta_table, sep="\t")
        delta_values = losses["delta_total_loss"].to_numpy()
        mean_delta = float(np.mean(delta_values))
        median_delta = float(np.median(delta_values))
        better_mask = delta_values < 0
        better_pct = float(np.mean(better_mask) * 100.0) if better_mask.size else float("nan")

        top_label = f"top_{args.top_n}"
        top_mask = losses["subset"] == top_label
        top_values = losses.loc[top_mask, "delta_total_loss"].to_numpy()
        top_better_mask = top_values < 0
        top_better_pct = (
            float(np.mean(top_better_mask) * 100.0) if top_better_mask.size else float("nan")
        )

        example_html = exp_dir / "output" / "10_main.html"
        histogram_html = exp_dir / "output" / "23_delta_histograms" / "23_main.html"

        summary_rows.append(
            {
                "experiment_id": exp_id,
                "label": label,
                "mean_delta_total_loss": mean_delta,
                "median_delta_total_loss": median_delta,
                "motif_only_better_pct": better_pct,
                "motif_only_better_pct_top": top_better_pct,
                "example_html": str(example_html),
                "histogram_html": str(histogram_html),
            }
        )

        all_violin_values.append(delta_values)
        top_violin_values.append(top_values)
        labels.append(label)

    summary_path = output_dir / "summary_stats.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)

    all_png = indv_dir / "delta_total_loss_violin_all.png"
    all_svg = indv_dir / "delta_total_loss_violin_all.svg"
    top_png = indv_dir / f"delta_total_loss_violin_top{args.top_n}.png"
    top_svg = indv_dir / f"delta_total_loss_violin_top{args.top_n}.svg"

    _plot_violin(all_violin_values, labels, all_png, all_svg, "Delta total loss")
    _plot_violin(top_violin_values, labels, top_png, top_svg, "Delta total loss")

    summary_df = pd.DataFrame(summary_rows)
    html_rows = []
    for _, row in summary_df.iterrows():
        example_rel = os.path.relpath(row["example_html"], start=output_dir)
        histogram_rel = os.path.relpath(row["histogram_html"], start=output_dir)
        html_rows.append(
            "<tr>"
            f"<td>{row['label']}</td>"
            f"<td>{row['mean_delta_total_loss']:.4f}</td>"
            f"<td>{row['median_delta_total_loss']:.4f}</td>"
            f"<td>{row['motif_only_better_pct']:.2f}%</td>"
            f"<td>{row['motif_only_better_pct_top']:.2f}%</td>"
            f"<td><a href=\"{example_rel}\">Example report</a></td>"
            f"<td><a href=\"{histogram_rel}\">Delta histograms</a></td>"
            "</tr>"
        )

    main_html = output_dir / "main.html"
    all_png_rel = os.path.relpath(all_png, start=output_dir)
    all_svg_rel = os.path.relpath(all_svg, start=output_dir)
    top_png_rel = os.path.relpath(top_png, start=output_dir)
    top_svg_rel = os.path.relpath(top_svg, start=output_dir)

    main_html.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<title>Motif removal experiment summary</title>",
                "</head>",
                "<body>",
                "<h1>Motif removal experiment summary</h1>",
                "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">",
                "<tr>",
                "<th>Experiment</th>",
                "<th>Mean delta total loss</th>",
                "<th>Median delta total loss</th>",
                "<th>Motif-only better (%)</th>",
                f"<th>Motif-only better Top {args.top_n} regions (%)</th>",
                "<th>Example report</th>",
                "<th>Delta histograms</th>",
                "</tr>",
                *html_rows,
                "</table>",
                "<h2>Delta total loss (all regions)</h2>",
                f"<div><img src=\"{all_png_rel}\" alt=\"Delta total loss (all)\" "
                "style=\"max-width: 100%; height: auto;\"></div>",
                f"<div><a href=\"{all_svg_rel}\">SVG version</a></div>",
                f"<h2>Delta total loss (top {args.top_n} regions)</h2>",
                f"<div><img src=\"{top_png_rel}\" alt=\"Delta total loss (top)\" "
                "style=\"max-width: 100%; height: auto;\"></div>",
                f"<div><a href=\"{top_svg_rel}\">SVG version</a></div>",
                "</body>",
                "</html>",
            ]
        )
    )


if __name__ == "__main__":
    main()
