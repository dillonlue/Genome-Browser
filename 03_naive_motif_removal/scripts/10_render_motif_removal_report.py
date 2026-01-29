#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def _sanitize_identifier(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).lower()


def _browser_png_name(region, window_span: int) -> str:
    chrom = region["chr"]
    start = int(region["start"])
    end = start + window_span
    safe_name = _sanitize_identifier(region["region_name"])
    return f"{chrom}_{start}_{end}_{safe_name}.png"


def _nav_links(rank: int, total: int, main_rel: str) -> str:
    links = []
    if rank > 1:
        links.append(f"<a href=\"sequence_{rank - 1}.html\">Previous</a>")
    if rank < total:
        links.append(f"<a href=\"sequence_{rank + 1}.html\">Next</a>")
    links.append(f"<a href=\"{main_rel}\">Back to summary</a>")
    return "<div>" + " | ".join(links) + "</div>"


def _equation_block(row: pd.Series, count_weight: float) -> str:
    label_width = 24

    def _num(val: float) -> str:
        return f"{val:>10.4f}"

    def _signed(val: float) -> str:
        return f"{val:>+10.4f}"

    def _label(name: str) -> str:
        return f"{name:<{label_width}}"

    baseline_total = float(row["baseline_total_loss"])
    baseline_profile = float(row["baseline_profile_loss"])
    baseline_count = float(row["baseline_count_loss"])
    baseline_profile_plus = float(row["baseline_profile_loss_plus"])
    baseline_profile_minus = float(row["baseline_profile_loss_minus"])

    motif_total = float(row["motif_total_loss"])
    motif_profile = float(row["motif_profile_loss"])
    motif_count = float(row["motif_count_loss"])
    motif_profile_plus = float(row["motif_profile_loss_plus"])
    motif_profile_minus = float(row["motif_profile_loss_minus"])

    delta_total = float(row["delta_total_loss"])
    delta_profile = float(row["delta_profile_loss"])
    delta_count = float(row["delta_count_loss"])
    delta_profile_plus = float(row["delta_profile_loss_plus"])
    delta_profile_minus = float(row["delta_profile_loss_minus"])

    lines = [
        (
            f"{_label('delta_total_loss')} = "
            f"delta_profile_loss ({_signed(delta_profile)}) + "
            f"{count_weight:.4f} * delta_count_loss ({_signed(delta_count)}) = "
            f"{_signed(delta_total)}"
        ),
        (
            f"{_label('delta_profile_loss')} = "
            f"delta_profile_loss_plus ({_signed(delta_profile_plus)}) + "
            f"delta_profile_loss_minus ({_signed(delta_profile_minus)})"
        ),
        (
            f"{_label('delta_count_loss')} = "
            f"motif_count_loss ({_num(motif_count)}) - "
            f"baseline_count_loss ({_num(baseline_count)}) = "
            f"{_signed(delta_count)}"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render HTML report for motif-only loss analysis."
    )
    parser.add_argument("--loss-tsv", required=True)
    parser.add_argument("--top-tsv", required=True)
    parser.add_argument("--region-json", required=True)
    parser.add_argument("--browser-plots-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--main-html", required=True)
    parser.add_argument("--summary-tsv", required=True)
    parser.add_argument("--indv-plots-dir", required=True)
    parser.add_argument("--window-span", type=int, default=2114)
    parser.add_argument("--count-loss-weight", type=float, required=True)
    args = parser.parse_args()

    losses = pd.read_csv(args.loss_tsv, sep="\t")
    top = pd.read_csv(args.top_tsv, sep="\t")
    losses = losses.sort_values("sequence_rank")
    top = top.sort_values("rank")

    merged = losses.merge(
        top[["rank", "total_counts"]],
        left_on="sequence_rank",
        right_on="rank",
        how="left",
    )
    merged["delta_total_loss"] = (
        merged["motif_total_loss"] - merged["baseline_total_loss"]
    )
    merged["delta_profile_loss"] = (
        merged["motif_profile_loss"] - merged["baseline_profile_loss"]
    )
    merged["delta_profile_loss_plus"] = (
        merged["motif_profile_loss_plus"] - merged["baseline_profile_loss_plus"]
    )
    merged["delta_profile_loss_minus"] = (
        merged["motif_profile_loss_minus"] - merged["baseline_profile_loss_minus"]
    )
    merged["delta_count_loss"] = (
        merged["motif_count_loss"] - merged["baseline_count_loss"]
    )

    main_html = Path(args.main_html)
    main_dir = main_html.parent
    indv_dir = Path(args.indv_plots_dir)
    indv_dir.mkdir(parents=True, exist_ok=True)

    with open(args.region_json, "r") as handle:
        region_payload = json.load(handle)
    regions = region_payload["regions"]

    browser_dir = Path(args.browser_plots_dir)
    total = merged.shape[0]

    summary_rows = []
    html_rows = []
    for idx, row in merged.iterrows():
        rank = int(row["sequence_rank"])
        region = regions[rank - 1]
        browser_png = browser_dir / _browser_png_name(region, args.window_span)
        browser_rel_main = os.path.relpath(browser_png, start=main_dir)
        detail_rel = os.path.relpath(
            indv_dir / f"sequence_{rank}.html",
            start=main_dir,
        )

        summary_rows.append(
            {
                "sequence_rank": rank,
                "chrom": row["chrom"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "total_counts": float(row["total_counts"]),
                "baseline_total_loss": float(row["baseline_total_loss"]),
                "motif_total_loss": float(row["motif_total_loss"]),
                "delta_total_loss": float(row["delta_total_loss"]),
                "baseline_profile_loss": float(row["baseline_profile_loss"]),
                "motif_profile_loss": float(row["motif_profile_loss"]),
                "delta_profile_loss": float(row["delta_profile_loss"]),
                "baseline_profile_loss_plus": float(row["baseline_profile_loss_plus"]),
                "motif_profile_loss_plus": float(row["motif_profile_loss_plus"]),
                "delta_profile_loss_plus": float(row["delta_profile_loss_plus"]),
                "baseline_profile_loss_minus": float(row["baseline_profile_loss_minus"]),
                "motif_profile_loss_minus": float(row["motif_profile_loss_minus"]),
                "delta_profile_loss_minus": float(row["delta_profile_loss_minus"]),
                "baseline_count_loss": float(row["baseline_count_loss"]),
                "motif_count_loss": float(row["motif_count_loss"]),
                "delta_count_loss": float(row["delta_count_loss"]),
                "motif_hit_count": int(row["motif_hit_count"]),
                "motif_bases_kept": int(row["motif_bases_kept"]),
                "kept_fraction": float(row["kept_fraction"]),
            }
        )

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
            f"<td><a href=\"{detail_rel}\">Details</a></td>"
            f"<td><a href=\"{browser_rel_main}\">Browser PNG</a></td>"
            "</tr>"
        )

        main_rel = os.path.relpath(main_html, start=indv_dir)
        nav = _nav_links(rank, total, main_rel)
        detail_path = indv_dir / f"sequence_{rank}.html"
        browser_rel = os.path.relpath(browser_png, start=indv_dir)
        equation_block = _equation_block(row, args.count_loss_weight)
        detail_path.write_text(
            "\n".join(
                [
                    "<!DOCTYPE html>",
                    "<html>",
                    "<head>",
                    "<meta charset=\"utf-8\">",
                    "<title>Motif-only loss</title>",
                    "</head>",
                    "<body>",
                    nav,
                    "<h2>Motif-only loss summary</h2>",
                    "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">",
                    "<tr><th>Metric</th><th>Value</th></tr>",
                    f"<tr><td>Region</td><td>{row['chrom']}:{int(row['start'])}-{int(row['end'])}</td></tr>",
                    f"<tr><td>Total counts</td><td>{row['total_counts']:.2f}</td></tr>",
                    f"<tr><td>Baseline total loss</td><td>{row['baseline_total_loss']:.4f}</td></tr>",
                    f"<tr><td>Motif-only total loss</td><td>{row['motif_total_loss']:.4f}</td></tr>",
                    f"<tr><td>Delta total loss</td><td>{row['delta_total_loss']:.4f}</td></tr>",
                    f"<tr><td>Baseline profile loss</td><td>{row['baseline_profile_loss']:.4f}</td></tr>",
                    f"<tr><td>Motif-only profile loss</td><td>{row['motif_profile_loss']:.4f}</td></tr>",
                    f"<tr><td>Delta profile loss</td><td>{row['delta_profile_loss']:.4f}</td></tr>",
                    f"<tr><td>Baseline profile loss (+)</td><td>{row['baseline_profile_loss_plus']:.4f}</td></tr>",
                    f"<tr><td>Motif-only profile loss (+)</td><td>{row['motif_profile_loss_plus']:.4f}</td></tr>",
                    f"<tr><td>Delta profile loss (+)</td><td>{row['delta_profile_loss_plus']:.4f}</td></tr>",
                    f"<tr><td>Baseline profile loss (-)</td><td>{row['baseline_profile_loss_minus']:.4f}</td></tr>",
                    f"<tr><td>Motif-only profile loss (-)</td><td>{row['motif_profile_loss_minus']:.4f}</td></tr>",
                    f"<tr><td>Delta profile loss (-)</td><td>{row['delta_profile_loss_minus']:.4f}</td></tr>",
                    f"<tr><td>Baseline count loss</td><td>{row['baseline_count_loss']:.4f}</td></tr>",
                    f"<tr><td>Motif-only count loss</td><td>{row['motif_count_loss']:.4f}</td></tr>",
                    f"<tr><td>Delta count loss</td><td>{row['delta_count_loss']:.4f}</td></tr>",
                    f"<tr><td>Motif hits</td><td>{int(row['motif_hit_count'])}</td></tr>",
                    f"<tr><td>Motif bases kept</td><td>{int(row['motif_bases_kept'])}</td></tr>",
                    f"<tr><td>Kept fraction</td><td>{row['kept_fraction']:.3f}</td></tr>",
                    "</table>",
                    "<h2>Loss decomposition</h2>",
                    "<pre>",
                    equation_block,
                    "</pre>",
                    "<h2>Genome browser</h2>",
                    f"<div><img src=\"{browser_rel}\" alt=\"Genome browser\" style=\"max-width: 100%; height: auto;\"></div>",
                    "</body>",
                    "</html>",
                ]
            )
        )

    summary_path = Path(args.summary_tsv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)

    main_html.parent.mkdir(parents=True, exist_ok=True)
    main_html.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<title>Motif-only loss summary</title>",
                "</head>",
                "<body>",
                "<h1>Motif-only loss summary</h1>",
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
                "<th>Details</th>",
                "<th>Genome browser</th>",
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
