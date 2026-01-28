#!/usr/bin/env python
"""Render PWM sequence logos from BPNet PFM files."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_pfm(path: Path, pseudocount: float = 1e-4) -> pd.DataFrame:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = [float(x) for x in stripped.split()]
            if len(values) != 4:
                raise ValueError(f"Unexpected column count in {path}: {len(values)}")
            rows.append(values)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    matrix = np.asarray(rows, dtype=float)
    matrix = matrix + pseudocount
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return pd.DataFrame(matrix, columns=["A", "C", "G", "T"])


def render_logo(df: pd.DataFrame, title: str, output: Path) -> None:
    info = logomaker.transform_matrix(df.copy(), from_type="probability", to_type="information")
    width = max(4.0, df.shape[0] * 0.4)
    fig, ax = plt.subplots(figsize=(width, 2.4))
    logomaker.Logo(info, ax=ax, color_scheme="classic")
    ax.set_ylabel("bits")
    ax.set_xlabel("position")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 2.2)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_index(index_path: Path, entries: list[tuple[str, str]]) -> None:
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8' />",
        "<title>BPNet PWM Logos</title>",
        "<style>body{font-family:Arial, sans-serif; margin:24px;} .grid{display:flex; flex-wrap:wrap; gap:24px;} .card{border:1px solid #ddd; padding:12px; border-radius:8px;} img{max-width:100%; height:auto;}</style>",
        "</head>",
        "<body>",
        "<h1>BPNet PWM Logos</h1>",
        "<div class='grid'>",
    ]
    for motif_name, rel_path in entries:
        lines.append("<div class='card'>")
        lines.append(f"<h3>{motif_name}</h3>")
        lines.append(f"<img src='{rel_path}' alt='{motif_name} logo' />")
        lines.append("</div>")
    lines.extend(["</div>", "</body>", "</html>"])
    index_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PWM logos from PFM files.")
    parser.add_argument("--pfm-glob", required=True, help="Glob pattern for PFM files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for PNG logos.")
    parser.add_argument("--done-file", required=True, help="Touch file written when done.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfm_files = sorted(Path(p) for p in glob.glob(args.pfm_glob))
    if not pfm_files:
        raise SystemExit(f"No PFM files found for pattern: {args.pfm_glob}")

    entries: list[tuple[str, str]] = []
    for pfm_path in pfm_files:
        motif_name = pfm_path.parent.name
        df = load_pfm(pfm_path)
        out_name = f"01_{motif_name}_pwm_logo.png"
        out_path = out_dir / out_name
        render_logo(df, motif_name, out_path)
        entries.append((motif_name, out_name))

    index_path = out_dir / "01_index.html"
    write_index(index_path, entries)

    done_path = Path(args.done_file)
    done_path.write_text(
        f"Rendered {len(entries)} PWM logos on {dt.datetime.now().isoformat()}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
