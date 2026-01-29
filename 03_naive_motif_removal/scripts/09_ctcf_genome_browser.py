#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).resolve().parents[2]))

from genome_browser import main as genome_tracker

DEFAULT_WINDOW_SPAN = 2114


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
        raise ValueError(
            f"Region config contains multiple window spans: {sorted(spans)}"
        )
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


def _write_summary(payload: dict, output_tsv: Path, window_span: int) -> None:
    rows = []
    for region in payload["regions"]:
        chrom = region["chr"]
        start = int(region["start"])
        end = int(region.get("end", start + window_span))
        rows.append((region["region_name"], chrom, start, end))

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w") as handle:
        handle.write("region_name\tchrom\tstart\tend\n")
        for name, chrom, start, end in rows:
            handle.write(f"{name}\t{chrom}\t{start}\t{end}\n")


def _rewrite_main_html(main_html: Path, indv_plots_dir: Path) -> None:
    rel_indv = os.path.relpath(indv_plots_dir, start=main_html.parent)
    html_text = main_html.read_text()
    html_text = html_text.replace("indv_plots/", f"{rel_indv}/")
    main_html.write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render genome browser outputs for motif removal analysis."
    )
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--track-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--main-html", required=True)
    parser.add_argument("--indv-plots-dir", required=True)
    parser.add_argument("--summary-tsv", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--title")
    parser.add_argument("--window-span", type=int, default=None)
    parser.add_argument("--output-span", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region_payload = json.loads(Path(args.region_config).read_text())
    window_span = _resolve_window_span(region_payload, args.window_span)

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
    ]
    if args.title:
        sys.argv.extend(["--title", args.title])
    sys.argv.extend(["--window-span", str(window_span)])
    if args.output_span is not None:
        sys.argv.extend(["--output-span", str(args.output_span)])
    genome_tracker.main()

    source_main = output_dir / "main.html"
    source_indv = output_dir / "indv_plots"

    target_main = Path(args.main_html)
    target_main.parent.mkdir(parents=True, exist_ok=True)
    if source_main != target_main:
        source_main.replace(target_main)

    target_indv = Path(args.indv_plots_dir)
    target_indv.parent.mkdir(parents=True, exist_ok=True)
    if source_indv != target_indv:
        if target_indv.exists():
            shutil.rmtree(target_indv)
        source_indv.rename(target_indv)

    _rewrite_main_html(target_main, target_indv)
    _write_summary(region_payload, Path(args.summary_tsv), window_span)


if __name__ == "__main__":
    main()
