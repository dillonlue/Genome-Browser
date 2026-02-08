#!/usr/bin/env python3
"""Simplified genome tracker wrapper.

Renders genome browser plots for a region/track config and writes a compact
HTML overview plus a copy of plots in a stable naming scheme.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import shutil
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

from genome_browser import main as genome_tracker

DEFAULT_WINDOW_SPAN = 2114
DEFAULT_SEED_INDV_PLOTS = Path("05_genome_browser_BPNet/output/05_indv_plots")


def _infer_window_span(payload: dict) -> int | None:
    if "window_span" in payload:
        return int(payload["window_span"])
    spans = set()
    for region in payload.get("regions", []):
        if "end" in region and "start" in region:
            spans.add(int(region["end"]) - int(region["start"]))
    if not spans:
        return None
    if len(spans) > 1:
        raise ValueError(f"Region config contains multiple window spans: {sorted(spans)}")
    return spans.pop()


def _resolve_window_span(payload: dict, cli_window_span: int | None) -> int:
    inferred = _infer_window_span(payload)
    if cli_window_span is None:
        return inferred if inferred is not None else DEFAULT_WINDOW_SPAN
    if inferred is not None and inferred != cli_window_span:
        raise ValueError(
            "Region config window span does not match --window-span "
            f"({inferred} vs {cli_window_span})."
        )
    return cli_window_span


def _sanitize_lower(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).lower()


def _sanitize_preserve_case(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _region_end(region: dict, window_span: int) -> int:
    if "end" in region and region["end"] is not None:
        return int(region["end"])
    return int(region["start"]) + window_span


def _region_plot_name(region: dict, window_span: int) -> str:
    end = _region_end(region, window_span)
    safe_name = _sanitize_lower(region["region_name"])
    return f"{region['chr']}_{region['start']}_{end}_{safe_name}.png"


def _region_plot_name_preserve_case(region: dict, window_span: int) -> str:
    end = _region_end(region, window_span)
    safe_name = _sanitize_preserve_case(region["region_name"])
    return f"{region['chr']}_{region['start']}_{end}_{safe_name}.png"


def _write_summary(payload: dict, output_tsv: Path, window_span: int) -> None:
    rows = []
    for region in payload["regions"]:
        chrom = region["chr"]
        start = int(region["start"])
        end = _region_end(region, window_span)
        rows.append((region["region_name"], chrom, start, end))

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w") as handle:
        handle.write("region_name\tchrom\tstart\tend\n")
        for name, chrom, start, end in rows:
            handle.write(f"{name}\t{chrom}\t{start}\t{end}\n")


def _write_simple_html(output_path: Path, title: str, payload: dict, window_span: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{title}</title>",
        "<meta charset=\"utf-8\">",
        "<style>body{font-family:Arial, sans-serif; margin:20px;} h2{margin-top:40px;} img{max-width:100%; height:auto; border:1px solid #ccc;} .motif-pair{display:flex; gap:20px; flex-wrap:wrap;} .motif-pair img{width:45%; min-width:280px;} .bar-plots{border:1px solid #ddd; padding:12px; margin:16px 0;} .bar-plots img{max-width:60%; min-width:280px;} .text-tracks{border:1px solid #ddd; padding:12px; margin:16px 0;} .text-track-block{margin-bottom:12px;} .text-track-block pre{background:#f7f7f7; padding:8px; white-space:pre-wrap;}</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p><strong>Generated:</strong> {now}</p>",
    ]

    for region in payload["regions"]:
        chrom = region["chr"]
        start = int(region["start"])
        end = _region_end(region, window_span)
        name = region["region_name"]
        figure_name = _region_plot_name(region, window_span)
        lines.append(f"<h2>{name} ({chrom}:{start:,}-{end:,})</h2>")
        lines.append(
            f"<img src=\"indv_plots/{figure_name}\" alt=\"{name} ({chrom}:{start:,}-{end:,})\">"
        )

    lines.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(lines))


def _write_index_html(index_path: Path, main_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if main_path.exists():
        index_path.write_text(main_path.read_text())


def _sync_plot_copies(payload: dict, output_dir: Path, window_span: int) -> None:
    plots_dir = output_dir / "plots"
    indv_dir = output_dir / "indv_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for region in payload["regions"]:
        src = indv_dir / _region_plot_name(region, window_span)
        if not src.exists():
            continue
        dest = plots_dir / _region_plot_name_preserve_case(region, window_span)
        if dest.exists():
            continue
        shutil.copy2(src, dest)


def _seed_indv_plots(payload: dict, output_dir: Path, window_span: int) -> None:
    seed_dir = DEFAULT_SEED_INDV_PLOTS
    if not seed_dir.exists():
        return
    indv_dir = output_dir / "indv_plots"
    indv_dir.mkdir(parents=True, exist_ok=True)
    for region in payload["regions"]:
        filename = _region_plot_name(region, window_span)
        dest = indv_dir / filename
        if dest.exists():
            continue
        src = seed_dir / filename
        if src.exists():
            shutil.copy2(src, dest)


def _expected_plot_paths(payload: dict, output_dir: Path, window_span: int) -> list[Path]:
    indv_dir = output_dir / "indv_plots"
    return [indv_dir / _region_plot_name(region, window_span) for region in payload["regions"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render simplified genome tracker outputs.")
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--track-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-span", type=int, default=None)
    parser.add_argument("--output-span", type=int, default=None)
    parser.add_argument("--title", default="Simplified genome tracker")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.region_config).read_text())
    if "regions" not in payload or not payload["regions"]:
        raise ValueError("Region config contains no regions")
    window_span = _resolve_window_span(payload, args.window_span)

    _seed_indv_plots(payload, output_dir, window_span)

    expected_plots = _expected_plot_paths(payload, output_dir, window_span)
    missing_plots = [path for path in expected_plots if not path.exists()]
    if missing_plots:
        sys.argv = [
            "genome_browser",
            "--region-config",
            args.region_config,
            "--track-config",
            args.track_config,
            "--output-dir",
            str(output_dir),
            "--num-workers",
            str(args.num_workers),
            "--window-span",
            str(window_span),
        ]
        if args.output_span is not None:
            sys.argv.extend(["--output-span", str(args.output_span)])
        if args.title:
            sys.argv.extend(["--title", args.title])

        genome_tracker.main()

    main_html = output_dir / "main.html"
    _write_simple_html(main_html, args.title, payload, window_span)
    _write_index_html(output_dir / "index.html", main_html)
    _write_summary(payload, output_dir / "summary_stats.tsv", window_span)
    _sync_plot_copies(payload, output_dir, window_span)


if __name__ == "__main__":
    main()
