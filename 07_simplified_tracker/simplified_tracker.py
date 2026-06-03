#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig

DEFAULT_WINDOW_SPAN = 2114
DEFAULT_OUTPUT_SPAN = 1000

def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).lower()

def region_end(region: dict, window_span: int) -> int:
    return int(region.get("end") or (int(region["start"]) + window_span))

def region_plot_name(region: dict, window_span: int) -> str:
    end = region_end(region, window_span)
    return f"{region['chr']}_{region['start']}_{end}_{sanitize(region['region_name'])}.png"

def infer_window_span(payload: dict) -> int | None:
    if "window_span" in payload:
        return int(payload["window_span"])
    spans = {int(r["end"]) - int(r["start"]) for r in payload.get("regions", []) if "end" in r and "start" in r}
    if not spans:
        return None
    if len(spans) > 1:
        raise ValueError(f"Region config contains multiple window spans: {sorted(spans)}")
    return spans.pop()

def resolve_window_span(payload: dict, cli_window_span: int | None) -> int:
    inferred = infer_window_span(payload)
    if cli_window_span is None:
        return inferred if inferred is not None else DEFAULT_WINDOW_SPAN
    if inferred is not None and inferred != cli_window_span:
        raise ValueError("Region config window span does not match --window-span " f"({inferred} vs {cli_window_span}).")
    return cli_window_span

def read_bigwig(path: Path, chrom: str, start: int, end: int) -> np.ndarray:
    with pyBigWig.open(str(path)) as bw:
        values = bw.values(chrom, start, end, numpy=True)
    array = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

def load_bed_index(path: Path) -> Dict[Tuple[str, int, int], int]:
    mapping: Dict[Tuple[str, int, int], int] = {}
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, *_ = line.strip().split("\t")
            mapping[(chrom, int(start), int(end))] = idx
    return mapping

def load_tracks(track_config: Path) -> List[dict]:
    payload = json.loads(track_config.read_text())
    tracks: List[dict] = []
    for entry in payload.get("tracks", []):
        ftype = entry.get("file_type")
        if ftype == "bw":
            path = Path(entry.get("file_location", ""))
            if str(path).startswith("01_download_data/"):
                tracks.append({"type": "bw", "label": entry.get("track_name", path.name), "path": path})
        elif ftype == "bpnet_lite_contribution":
            attr = Path(entry.get("precomputed_attr_npz", "")); ohe = Path(entry.get("precomputed_ohe_npz", ""))
            bed = Path(entry.get("bed_path", ""))
            if not (attr.exists() and ohe.exists() and bed.exists()):
                continue
            contrib = np.load(attr)["arr_0"] * np.load(ohe)["arr_0"]
            tracks.append({"type": "contrib", "label": entry.get("track_name", attr.name), "contrib": contrib, "index": load_bed_index(bed)})
    if not tracks:
        raise ValueError("Track config has no supported tracks")
    return tracks

def plot_region(chrom: str, start: int, name: str, window_span: int, output_span: int, tracks: List[dict], output_path: Path) -> None:
    end = start + window_span
    x = np.arange(window_span)
    fig, axes = plt.subplots(len(tracks), 1, sharex=True, figsize=(22, max(6, 1.2 * len(tracks))))
    if len(tracks) == 1:
        axes = [axes]
    offset = max(0, (window_span - output_span) // 2)
    for idx, track in enumerate(tracks):
        ax = axes[idx]
        if track["type"] == "bw":
            path = track["path"]
            if not path.exists():
                ax.text(0.5, 0.5, f"Missing: {path}", ha="center", va="center", fontsize=8); ax.set_axis_off(); continue
            values = read_bigwig(path, chrom, start, end)
            ax.plot(x, values, linewidth=0.8); ax.fill_between(x, 0, values, alpha=0.2)
        else:
            key = (chrom, start + offset, start + offset + output_span)
            row = track["index"].get(key)
            if row is None:
                ax.text(0.5, 0.5, "Missing attribution for region", ha="center", va="center", fontsize=8); ax.set_axis_off(); continue
            contrib = track["contrib"][row]
            full = np.zeros((4, window_span), dtype=np.float32)
            full[:, offset:offset + min(output_span, contrib.shape[1])] = contrib[:, :output_span]
            logomaker.Logo(pd.DataFrame(full.T, columns=["A", "C", "G", "T"]), ax=ax, color_scheme="classic", vpad=0.1)
            ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(track["label"], rotation=0, ha="right", va="center", labelpad=50, fontsize=8)
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if output_span < window_span:
            ax.axvline(offset, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.axvline(offset + output_span, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ticks = np.linspace(0, window_span - 1, 6, dtype=int)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f"{start + int(pos):,}" for pos in ticks], rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel(f"{chrom} coordinate")
    fig.suptitle(f"{name} ({chrom}:{start:,}-{end:,})", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def write_summary(payload: dict, output_tsv: Path, window_span: int) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w") as handle:
        handle.write("region_name\tchrom\tstart\tend\n")
        for region in payload["regions"]:
            chrom = region["chr"]; start = int(region["start"]); end = region_end(region, window_span)
            handle.write(f"{region['region_name']}\t{chrom}\t{start}\t{end}\n")

def write_html(output_path: Path, title: str, payload: dict, window_span: int) -> None:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "<!DOCTYPE html>", "<html>", "<head>", f"<title>{title}</title>", '<meta charset="utf-8">',
        "<style>body{font-family:Arial, sans-serif; margin:20px;} h2{margin-top:40px;} img{max-width:100%; height:auto; border:1px solid #ccc;}</style>",
        "</head>", "<body>", f"<h1>{title}</h1>", f"<p><strong>Generated:</strong> {now}</p>",
    ]
    for region in payload["regions"]:
        chrom = region["chr"]; start = int(region["start"]); end = region_end(region, window_span)
        name = region["region_name"]; figure_name = region_plot_name(region, window_span)
        lines.append(f"<h2>{name} ({chrom}:{start:,}-{end:,})</h2>")
        lines.append(f"<img src=\"indv_plots/{figure_name}\" alt=\"{name} ({chrom}:{start:,}-{end:,})\">")
    lines.extend(["</body>", "</html>"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

def main() -> None:
    parser = argparse.ArgumentParser(description="Render simplified genome tracker outputs.")
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--track-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-span", type=int, default=None)
    parser.add_argument("--output-span", type=int, default=None)
    parser.add_argument("--title", default="Simplified genome tracker")
    args = parser.parse_args()
    payload = json.loads(Path(args.region_config).read_text())
    if "regions" not in payload or not payload["regions"]:
        raise ValueError("Region config contains no regions")
    window_span = resolve_window_span(payload, args.window_span)
    output_span = args.output_span or min(window_span, DEFAULT_OUTPUT_SPAN)
    tracks = load_tracks(Path(args.track_config))
    output_dir = Path(args.output_dir); indv_dir = output_dir / "indv_plots"
    for region in payload["regions"]:
        chrom = region["chr"]; start = int(region["start"]); name = region["region_name"]
        out_path = indv_dir / region_plot_name(region, window_span)
        plot_region(chrom, start, name, window_span, output_span, tracks, out_path)
    write_html(output_dir / "main.html", args.title, payload, window_span)
    (output_dir / "index.html").write_text((output_dir / "main.html").read_text())
    write_summary(payload, output_dir / "summary_stats.tsv", window_span)

if __name__ == "__main__":
    main()
