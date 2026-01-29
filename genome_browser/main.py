#!/usr/bin/env python3
"""
Genome tracker plotter driven by JSON region and track specifications.

The region configuration lists labelled windows (``chr``, ``start``,
``region_name``) while the track configuration enumerates the available
datasets. Each track entry needs a ``track_name``, ``file_location``, and
``file_type``. Supported types are ``bw`` (bigWig signal), ``keras_contribution``,
``bpnet_lite_contribution``, and ``motif_pwm`` (on-the-fly motif scanning). The
script renders individual plots for every region and assembles a simple HTML
overview referencing the generated figures.
"""

from __future__ import annotations

import os

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from types import SimpleNamespace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import gzip
import html
import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import h5py
from scipy import stats
from submitit import AutoExecutor
from tqdm.auto import tqdm
from bisect import bisect_right
from shutil import which

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _get_motif_scan():
    try:
        from scripts_preprocessing import scan_motif_hits as motif_scan  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Motif PWM tracks require scripts_preprocessing to be available. "
            "Either add scripts_preprocessing to the repo or avoid motif_pwm tracks."
        ) from exc
    return motif_scan

DEFAULT_WINDOW_SPAN = 2114
DEFAULT_OUTPUT_SPAN = 1000

WINDOW_SPAN = DEFAULT_WINDOW_SPAN
OUTPUT_SPAN = DEFAULT_OUTPUT_SPAN
OUTPUT_START = (WINDOW_SPAN - OUTPUT_SPAN) // 2
OUTPUT_END = OUTPUT_START + OUTPUT_SPAN
_MOTIF_PWM_CACHE: Dict[Tuple[str, str], Tuple[List[List[float]], List[List[float]], float, float]] = {}
_BED_INTERVAL_CACHE: Dict[str, Dict[str, List["IntervalSegment"]]] = {}
_FOCUS_BED_CACHE: Dict[str, Dict[str, Tuple[List[int], List[int]]]] = {}
_PRECOMPUTED_ARRAY_CACHE: Dict[str, np.ndarray] = {}
_PRECOMPUTED_REGION_DATA_CACHE: Dict[
    str, Tuple[np.ndarray, Dict[Tuple[str, int, int], int]]
] = {}
_SEQLET_WINDOW_CACHE: Dict[str, List[Tuple[str, int, int]]] = {}
_MODISCO_SEQLET_CACHE: Dict[str, Dict[Tuple[str, int], List["IntervalSegment"]]] = {}
DEFAULT_SEQLET_WINDOW = 500
DEFAULT_GENOME_FASTA = Path("data/k562/downloads/hg38.fa")
DEFAULT_INTERVAL_COLOR = "#000000"
DEFAULT_INTERSECTION_BED = Path(
    "data/k562/peak_intersections/atac_dnase_ctcf_centered_input_windows_test_chrs.bed"
)


@dataclass
class TrackSpec:
    name: str
    file_type: str
    location: Path | None = None
    smoothing_sigma: float = 0.0
    track_id: str | None = None
    model_path: Path | None = None
    shared_y_axis_scale_group: str | None = None
    motif_meme: Path | None = None
    motif_id: str | None = None
    motif_min_score: float | None = None
    bed_path: Path | None = None
    motif_logo_files: Tuple[Path, Path] | None = None
    source_track_id: str | None = None
    percentile: float = 99.0
    threshold: float | None = None
    max_gap: int = 5
    min_len: int = 10
    _motif_logo_dfs: Tuple[pd.DataFrame, pd.DataFrame] | None = None
    precomputed_contrib_h5: Path | None = None
    precomputed_attr_npz: Path | None = None
    precomputed_ohe_npz: Path | None = None
    per_region_bed_dir: Path | None = None
    track_label: str = ""
    modisco_h5: Path | None = None
    interpreted_regions_bed: Path | None = None
    seqlet_window: int | None = None
    focus_bed: Path | None = None
    focus_mode: str | None = None
    bar_metric: str | None = None
    bar_output_subdir: str | None = None
    bar_comparisons: List[Dict[str, str]] | None = None
    use_full_window: bool = False
    positive_only: bool = False

    def has_precomputed(self) -> bool:
        return self.precomputed_contrib_h5 is not None or (
            self.precomputed_attr_npz is not None and self.precomputed_ohe_npz is not None
        )

    def precomputed_signature(self) -> str | None:
        if self.precomputed_contrib_h5 is not None:
            path = self.precomputed_contrib_h5.resolve()
            if not path.exists():
                raise FileNotFoundError(f"Precomputed contribution file not found: {path}")
            mtime = path.stat().st_mtime
            return f"h5|{path}|{mtime:.6f}"
        if self.precomputed_attr_npz is not None and self.precomputed_ohe_npz is not None:
            attr_path = self.precomputed_attr_npz.resolve()
            ohe_path = self.precomputed_ohe_npz.resolve()
            if not attr_path.exists():
                raise FileNotFoundError(f"Precomputed attr file not found: {attr_path}")
            if not ohe_path.exists():
                raise FileNotFoundError(f"Precomputed ohe file not found: {ohe_path}")
            attr_mtime = attr_path.stat().st_mtime
            ohe_mtime = ohe_path.stat().st_mtime
            return f"npz|{attr_path}|{attr_mtime:.6f}|{ohe_path}|{ohe_mtime:.6f}"
        return None


@dataclass
class RegionSpec:
    chrom: str
    start: int
    name: str


@dataclass
class IntervalSegment:
    chrom: str
    start: int
    end: int
    score: float
    strand: str
    label: str | None = None


def sanitize_identifier(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).lower()


def region_filename(region: RegionSpec) -> str:
    end = region.start + WINDOW_SPAN
    safe_name = sanitize_identifier(region.name)
    return f"{region.chrom}_{region.start}_{end}_{safe_name}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render genome tracker figures.")
    parser.add_argument(
        "--region-config",
        required=True,
        type=Path,
        help="Path to JSON file describing the regions to plot.",
    )
    parser.add_argument(
        "--track-config",
        required=True,
        type=Path,
        help="Path to JSON file listing the raw signal tracks to render.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where plots and the overview page will be written.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of worker processes for contribution computation (default: 20).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title for the overview page.",
    )
    parser.add_argument(
        "--window-span",
        type=int,
        default=None,
        help="Input window length for each region (default: 2114).",
    )
    parser.add_argument(
        "--output-span",
        type=int,
        default=None,
        help="Output span within the window (default: min(1000, window span)).",
    )
    parser.add_argument(
        "--force-parallel",
        action="store_true",
        help="Process regions in parallel locally (128 workers) when Slurm is unavailable.",
    )
    return parser.parse_args()


def _set_window_params(window_span: int, output_span: int | None) -> None:
    global WINDOW_SPAN, OUTPUT_SPAN, OUTPUT_START, OUTPUT_END
    if output_span is None:
        output_span = min(DEFAULT_OUTPUT_SPAN, window_span)
    if output_span > window_span:
        raise ValueError(
            f"Output span {output_span} cannot exceed window span {window_span}."
        )
    WINDOW_SPAN = int(window_span)
    OUTPUT_SPAN = int(output_span)
    OUTPUT_START = (WINDOW_SPAN - OUTPUT_SPAN) // 2
    OUTPUT_END = OUTPUT_START + OUTPUT_SPAN


def _load_track_thresholds(path: Path) -> Dict[str, float]:
    if not path.exists():
        print(f"[genome_tracker] Warning: Thresholds file not found: {path}")
        return {}
    thresholds: Dict[str, float] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            raw = row.get("threshold")
            if raw is None or raw == "":
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            track_name = row.get("track_name") or row.get("track") or row.get("name")
            if track_name:
                thresholds[str(track_name)] = value
            track_id = row.get("track_id")
            if track_id:
                thresholds[str(track_id)] = value
            motif = row.get("motif")
            if motif:
                motif_name = str(motif)
                thresholds.setdefault(motif_name, value)
                thresholds.setdefault(f"CWM score: {motif_name}", value)
    return thresholds


def load_track_specs(path: Path) -> List[TrackSpec]:
    print("[genome_tracker] Loading track specs...")
    with path.open() as handle:
        payload = json.load(handle)

    requires_precomputed = bool(payload.get("precomputed_contributions", False))
    default_global_percentile = payload.get("global_threshold_motif_contrib")
    if default_global_percentile is not None:
        default_global_percentile = float(default_global_percentile)

    threshold_map: Dict[str, float] = {}
    thresholds_path = payload.get("track_thresholds_tsv")
    if thresholds_path:
        threshold_map = _load_track_thresholds(Path(thresholds_path))

    tracks: List[TrackSpec] = []
    for entry in payload["tracks"]:
        file_type = str(entry["file_type"]).strip().lower().replace(" ", "_")
        location = Path(entry["file_location"]) if "file_location" in entry else None
        model_path = Path(entry["model_path"]) if "model_path" in entry else None
        motif_meme = Path(entry["motif_meme"]) if "motif_meme" in entry else None
        motif_id = entry.get("motif_id")
        motif_min_score = None
        bed_path = Path(entry["bed_path"]) if "bed_path" in entry else None
        source_track_id = entry.get("source_track_id")
        modisco_h5_path = Path(entry["modisco_h5"]) if "modisco_h5" in entry else None
        interpreted_regions_bed = (
            Path(entry["interpreted_regions_bed"]) if "interpreted_regions_bed" in entry else None
        )
        seqlet_window = entry.get("seqlet_window")
        seqlet_window_value = int(seqlet_window) if seqlet_window is not None else None
        track_name = str(entry["track_name"])
        track_id = entry.get("track_id")
        if track_id is None:
            track_id = sanitize_identifier(track_name)

        percentile_value = entry.get("percentile")
        if (
            percentile_value is None
            and file_type == "contrib_motifs"
            and default_global_percentile is not None
        ):
            percentile_value = default_global_percentile
        percentile = float(percentile_value) if percentile_value is not None else 99.0
        threshold_raw = entry.get("threshold")
        threshold = None if threshold_raw is None else float(threshold_raw)
        if threshold is None and threshold_map:
            threshold = threshold_map.get(track_name)
            if threshold is None and track_id is not None:
                threshold = threshold_map.get(str(track_id))
        max_gap = int(entry.get("max_gap", 5))
        min_len = int(entry.get("min_len", 10))
        precomputed_h5 = entry.get("precomputed_contrib_h5")
        precomputed_attr = entry.get("precomputed_attr_npz")
        precomputed_ohe = entry.get("precomputed_ohe_npz")
        precomputed_h5_path = Path(precomputed_h5) if precomputed_h5 else None
        precomputed_attr_path = Path(precomputed_attr) if precomputed_attr else None
        precomputed_ohe_path = Path(precomputed_ohe) if precomputed_ohe else None
        use_full_window = bool(entry.get("use_full_window", False))
        positive_only = bool(entry.get("positive_only", False))
        shared_group = entry.get("shared_y_axis_scale_group")
        if shared_group is not None:
            shared_group = str(shared_group)
        focus_bed = Path(entry["focus_bed"]) if entry.get("focus_bed") else None
        focus_mode = entry.get("focus_mode")
        bar_metric = entry.get("metric")
        bar_output_subdir = entry.get("output_subdir")
        bar_comparisons = entry.get("comparisons")
        if file_type == "motif_pwm":
            if motif_meme is None or motif_id is None:
                raise ValueError(
                    f"Motif track '{entry['track_name']}' requires motif_meme and motif_id."
                )
            raw_score = entry.get("motif_min_score", entry.get("min_score"))
            motif_min_score = float(raw_score)
        if file_type == "bed" and bed_path is None:
            raise ValueError(f"Bed track '{entry['track_name']}' requires bed_path.")
        if file_type == "contrib_motifs":
            if source_track_id is None:
                raise ValueError(
                    f"Derived contribution track '{entry['track_name']}' requires source_track_id."
                )
        if file_type == "seqlet_modisco":
            if modisco_h5_path is None or interpreted_regions_bed is None:
                raise ValueError(
                    f"Seqlet track '{entry['track_name']}' requires modisco_h5 and interpreted_regions_bed."
                )
        if file_type == "bar_correlation":
            if not isinstance(bar_comparisons, list) or not bar_comparisons:
                raise ValueError(
                    f"Bar correlation track '{entry['track_name']}' requires a non-empty comparisons list."
                )
            for comp in bar_comparisons:
                if "track1" not in comp or "track2" not in comp:
                    raise ValueError(
                        f"Bar correlation track '{entry['track_name']}' comparison missing track1 or track2."
                    )

        bar_metric_value = str(bar_metric).strip().lower() if bar_metric else None
        if file_type == "bar_correlation" and bar_metric_value is None:
            bar_metric_value = "pearson"
        bar_output_dir_value = str(bar_output_subdir) if bar_output_subdir else "barplots"
        track = TrackSpec(
            name=track_name,
            file_type=file_type,
            location=location,
            smoothing_sigma=float(entry.get("smoothing_sigma", 0.0)),
            track_id=str(track_id),
            model_path=model_path,
            shared_y_axis_scale_group=shared_group,
            motif_meme=motif_meme,
            motif_id=motif_id,
            motif_min_score=motif_min_score,
            bed_path=bed_path,
            source_track_id=source_track_id,
            percentile=percentile,
            threshold=threshold,
            max_gap=max_gap,
            min_len=min_len,
            precomputed_contrib_h5=precomputed_h5_path,
            precomputed_attr_npz=precomputed_attr_path,
            precomputed_ohe_npz=precomputed_ohe_path,
            track_label=sanitize_identifier(track_name),
            modisco_h5=modisco_h5_path,
            interpreted_regions_bed=interpreted_regions_bed,
            seqlet_window=seqlet_window_value,
            focus_bed=focus_bed,
            focus_mode=str(focus_mode) if focus_mode else None,
            bar_metric=bar_metric_value,
            bar_output_subdir=bar_output_dir_value,
            bar_comparisons=bar_comparisons if isinstance(bar_comparisons, list) else None,
            use_full_window=use_full_window,
            positive_only=positive_only,
        )
        tracks.append(track)

    if requires_precomputed:
        for spec in tracks:
            if spec.file_type in {"keras_contribution", "bpnet_lite_contribution"}:
                if not spec.has_precomputed():
                    raise ValueError(
                        f"Track '{spec.name}' requires precomputed contributions but none were provided."
                    )

    print(f"[genome_tracker] Loaded {len(tracks)} track specs from {path}")
    return tracks


def load_region_specs(path: Path) -> List[RegionSpec]:
    print("[genome_tracker] Loading region specs...")
    with path.open() as handle:
        payload = json.load(handle)

    regions: List[RegionSpec] = []
    for entry in payload["regions"]:
        regions.append(
            RegionSpec(
                chrom=str(entry["chr"]),
                start=int(entry["start"]),
                name=str(entry["region_name"]),
            )
        )
    print(f"[genome_tracker] Loaded {len(regions)} regions from {path}")
    return regions


def categorize_specs(
    specs: Sequence[TrackSpec],
) -> Tuple[
    List[TrackSpec],
    List[TrackSpec],
    List[TrackSpec],
    List[TrackSpec],
    List[TrackSpec],
    List[TrackSpec],
    List[TrackSpec],
]:
    raw = [spec for spec in specs if spec.file_type == "bw"]
    bed = [spec for spec in specs if spec.file_type == "bed"]
    motif = [spec for spec in specs if spec.file_type == "motif_pwm"]
    contrib = [
        spec
        for spec in specs
        if spec.file_type in {"keras_contribution", "bpnet_lite_contribution"}
    ]
    seqlet = [spec for spec in specs if spec.file_type == "seqlet_modisco"]
    derived = [spec for spec in specs if spec.file_type == "contrib_motifs"]
    bar = [spec for spec in specs if spec.file_type == "bar_correlation"]
    return raw, bed, motif, contrib, seqlet, derived, bar


def _load_motif_pwm(spec: TrackSpec) -> Tuple[List[List[float]], List[List[float]], float, float]:
    if spec.motif_meme is None or spec.motif_id is None:
        raise ValueError(f"Motif track {spec.name} missing motif_meme or motif_id")
    key = (str(spec.motif_meme.resolve()), spec.motif_id)
    cached = _MOTIF_PWM_CACHE.get(key)
    if cached is None:
        motif_scan = _get_motif_scan()
        matrix = motif_scan.load_motif_matrix(spec.motif_meme, spec.motif_id)
        log_matrix, min_score, max_score = motif_scan.build_log_matrix(matrix)
        rev_matrix = motif_scan.reverse_complement_matrix(log_matrix)
        cached = (log_matrix, rev_matrix, min_score, max_score)
        _MOTIF_PWM_CACHE[key] = cached
        print(f"[genome_tracker] Cached PWM for {spec.name} ({spec.motif_id})")
    return cached


def _fetch_window_sequence(chrom: str, start: int, end: int) -> str:
    with pyfaidx.Fasta(str(DEFAULT_GENOME_FASTA)) as fasta:
        seq = fasta[chrom][start:end].seq.upper()
    return seq


def compute_motif_hits_for_region(spec: TrackSpec, region: RegionSpec) -> List[IntervalSegment]:
    print(f"[genome_tracker] Computing motif hits for '{spec.name}' in region '{region.name}'")
    start = region.start
    end = start + WINDOW_SPAN
    sequence = _fetch_window_sequence(region.chrom, start, end)
    log_matrix, rev_matrix, min_score, max_score = _load_motif_pwm(spec)
    scale = 1000
    hits: List[IntervalSegment] = []
    motif_scan = _get_motif_scan()
    forward_hits = motif_scan.scan_sequence(
        sequence,
        start,
        region.chrom,
        log_matrix,
        min_score,
        max_score,
        spec.motif_min_score,
        None,
        scale,
        spec.name,
        "+",
    )
    reverse_hits = motif_scan.scan_sequence(
        sequence,
        start,
        region.chrom,
        rev_matrix,
        min_score,
        max_score,
        spec.motif_min_score,
        None,
        scale,
        spec.name,
        "-",
    )
    for chrom, hit_start, hit_end, _label, raw_score, strand in forward_hits + reverse_hits:
        hits.append(
            IntervalSegment(
                chrom=chrom,
                start=hit_start,
                end=hit_end,
                score=float(raw_score) / float(scale),
                strand=strand,
                label=spec.name,
            )
        )
    hits.sort(key=lambda h: (h.start, h.end, h.strand))
    print(
        f"[genome_tracker] Motif track '{spec.name}' yielded {len(hits)} hits in region '{region.name}'"
    )
    return hits


def get_motif_logo_dfs(spec: TrackSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cached = getattr(spec, "_motif_logo_matrix", None)
    if cached is not None:
        return cached

    if spec.motif_meme is None or spec.motif_id is None:
        raise ValueError(f"Motif track {spec.name} missing motif information.")
    motif_scan = _get_motif_scan()
    matrix = motif_scan.load_motif_matrix(spec.motif_meme, spec.motif_id)
    forward_df = pd.DataFrame(matrix, columns=["A", "C", "G", "T"])

    reverse_matrix = []
    for row in reversed(matrix):
        reverse_matrix.append([row[3], row[2], row[1], row[0]])
    reverse_df = pd.DataFrame(reverse_matrix, columns=["A", "C", "G", "T"])

    cache_tuple = (forward_df, reverse_df)
    setattr(spec, "_motif_logo_matrix", cache_tuple)
    return cache_tuple


def ensure_motif_logo_images(spec: TrackSpec, output_dir: Path) -> Tuple[Path, Path] | None:
    print(f"[genome_tracker] Ensuring motif logos for '{spec.name}'")
    if spec.motif_logo_files is not None:
        return spec.motif_logo_files
    if spec.motif_meme is None or spec.motif_id is None:
        return None

    forward_df, reverse_df = get_motif_logo_dfs(spec)
    info_forward = logomaker.transform_matrix(
        forward_df.copy(), from_type="probability", to_type="information"
    )
    info_reverse = logomaker.transform_matrix(
        reverse_df.copy(), from_type="probability", to_type="information"
    )

    logo_dir = output_dir / "indv_plots" / "motif_logos"
    logo_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_identifier(spec.name)
    forward_path = logo_dir / f"{safe_name}_forward.png"
    reverse_path = logo_dir / f"{safe_name}_reverse.png"

    def _save_logo(df: pd.DataFrame, path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 2))
        logomaker.Logo(df, ax=ax, color_scheme="classic")
        ax.set_ylabel("bits")
        ax.set_xlabel("position")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 2.2)
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    _save_logo(info_forward, forward_path, f"{spec.name} (+)")
    _save_logo(info_reverse, reverse_path, f"{spec.name} (-)")

    spec.motif_logo_files = (forward_path, reverse_path)
    print(
        f"[genome_tracker] Generated motif logos for '{spec.name}' at {forward_path} and {reverse_path}"
    )
    return spec.motif_logo_files


def write_motif_hits_bed(
    output_root: Path,
    region: RegionSpec,
    spec: TrackSpec,
    hits: Sequence[IntervalSegment],
) -> Path:
    region_dir = (
        output_root
        / "intermediate_files"
        / "derived_beds"
        / sanitize_identifier(region.name)
    )
    region_dir.mkdir(parents=True, exist_ok=True)
    bed_path = region_dir / f"{sanitize_identifier(spec.name)}.bed"
    with bed_path.open("w") as handle:
        for hit in hits:
            handle.write(
                f"{hit.chrom}\t{hit.start}\t{hit.end}\t{spec.name}\t{hit.score:.6f}\t{hit.strand}\n"
            )
    print(
        f"[genome_tracker] Saved {len(hits)} motif intervals for '{spec.name}' in region '{region.name}' to {bed_path}"
    )
    return bed_path


def _load_bed_interval_map(path: Path) -> Dict[str, List[IntervalSegment]]:
    key = str(path.resolve())
    cached = _BED_INTERVAL_CACHE.get(key)
    if cached is not None:
        print(f"[genome_tracker] Reusing cached BED intervals for {path}")
        return cached

    interval_map: Dict[str, List[IntervalSegment]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3] if len(parts) > 3 else "."
            score = float(parts[4]) if len(parts) > 4 else 0.0
            strand = parts[5] if len(parts) > 5 else "."
            segment = IntervalSegment(
                chrom=chrom,
                start=start,
                end=end,
                score=score,
                strand=strand,
                label=name,
            )
            interval_map.setdefault(chrom, []).append(segment)

    for chrom_segments in interval_map.values():
        chrom_segments.sort(key=lambda seg: seg.start)

    _BED_INTERVAL_CACHE[key] = interval_map
    total_segments = sum(len(v) for v in interval_map.values())
    print(f"[genome_tracker] Cached {total_segments} intervals from BED {path}")
    return interval_map


def _load_ordered_interpreted_windows(path: Path) -> List[Tuple[str, int, int]]:
    key = str(path.resolve())
    cached = _SEQLET_WINDOW_CACHE.get(key)
    if cached is not None:
        return cached
    opener = gzip.open if path.suffix == ".gz" else open
    windows: List[Tuple[str, int, int]] = []
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            windows.append((chrom, start, end))
    _SEQLET_WINDOW_CACHE[key] = windows
    print(f"[genome_tracker] Loaded {len(windows)} interpreted windows from {path}")
    return windows


def _seqlet_window_offset(seqlet_window: int) -> int:
    seqlet_window = max(0, min(seqlet_window, WINDOW_SPAN))
    if seqlet_window <= 0 or seqlet_window >= WINDOW_SPAN:
        return 0
    return (WINDOW_SPAN - seqlet_window) // 2


def _load_modisco_seqlet_map(spec: TrackSpec) -> Dict[Tuple[str, int], List[IntervalSegment]]:
    if spec.modisco_h5 is None or spec.interpreted_regions_bed is None:
        raise ValueError(f"Seqlet track '{spec.name}' missing modisco_h5 or interpreted_regions_bed.")
    seqlet_window = spec.seqlet_window or DEFAULT_SEQLET_WINDOW
    cache_key = (
        f"{spec.modisco_h5.resolve()}|{spec.interpreted_regions_bed.resolve()}|{seqlet_window}"
    )
    cached = _MODISCO_SEQLET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    windows = _load_ordered_interpreted_windows(spec.interpreted_regions_bed)
    if not windows:
        raise ValueError(f"No interpreted windows found for seqlet track '{spec.name}'.")
    offset = _seqlet_window_offset(seqlet_window)
    seqlet_map: Dict[Tuple[str, int], List[IntervalSegment]] = defaultdict(list)
    with h5py.File(spec.modisco_h5, "r") as handle:
        for cluster_key in ("pos_patterns", "neg_patterns"):
            cluster = handle.get(cluster_key)
            if cluster is None:
                continue
            for pattern_name in cluster.keys():
                seq_group = cluster[pattern_name].get("seqlets")
                if seq_group is None:
                    continue
                total_seqlets = seq_group["start"].shape[0]
                if total_seqlets == 0:
                    continue
                starts = seq_group["start"][:]
                ends = seq_group["end"][:]
                example_idx = seq_group["example_idx"][:]
                is_rev = seq_group["is_revcomp"][:]
                contrib = seq_group["contrib_scores"][:]
                for idx in range(total_seqlets):
                    ex_idx = int(example_idx[idx])
                    if ex_idx < 0 or ex_idx >= len(windows):
                        continue
                    chrom, window_start, window_end = windows[ex_idx]
                    seq_start = int(starts[idx])
                    seq_end = int(ends[idx])
                    genome_start = window_start + offset + seq_start
                    genome_end = window_start + offset + seq_end
                    strand = "-" if bool(is_rev[idx]) else "+"
                    score = float(np.sum(np.abs(contrib[idx])))
                    label = f"{cluster_key}.{pattern_name}"
                    seqlet_map[(chrom, window_start)].append(
                        IntervalSegment(
                            chrom=chrom,
                            start=genome_start,
                            end=genome_end,
                            score=score,
                            strand=strand,
                            label=label,
                        )
                    )
    for intervals in seqlet_map.values():
        intervals.sort(key=lambda seg: seg.score, reverse=True)
    _MODISCO_SEQLET_CACHE[cache_key] = seqlet_map
    print(
        f"[genome_tracker] Cached {sum(len(v) for v in seqlet_map.values())} seqlets "
        f"from {spec.modisco_h5} for track '{spec.name}'"
    )
    return seqlet_map


def seqlet_segments_for_region(spec: TrackSpec, region: RegionSpec) -> List[IntervalSegment]:
    seqlet_map = _load_modisco_seqlet_map(spec)
    key = (region.chrom, region.start)
    segments = seqlet_map.get(key, [])
    print(
        f"[genome_tracker] Track '{spec.name}' contributes {len(segments)} seqlets to region '{region.name}'"
    )
    return segments


def bed_segments_for_region(spec: TrackSpec, region: RegionSpec) -> List[IntervalSegment]:
    bed_path = spec.bed_path
    if spec.per_region_bed_dir and spec.track_label:
        candidate = (
            spec.per_region_bed_dir
            / sanitize_identifier(region.name)
            / f"{spec.track_label}.bed"
        )
        if candidate.exists():
            bed_path = candidate
    if bed_path is None:
        raise ValueError(f"Bed track {spec.name} missing bed_path")
    bed_map = _load_bed_interval_map(bed_path)
    chrom_segments = bed_map.get(region.chrom, [])
    region_end = region.start + WINDOW_SPAN
    selected: List[IntervalSegment] = []
    for segment in chrom_segments:
        if segment.end <= region.start:
            continue
        if segment.start >= region_end:
            break
        selected.append(segment)
    print(
        f"[genome_tracker] Track '{spec.name}' contributes {len(selected)} preset intervals to region '{region.name}'"
    )
    return selected


def _per_base_magnitude(contrib_profile: np.ndarray) -> np.ndarray:
    array = np.asarray(contrib_profile, dtype=np.float32)
    if array.ndim == 2:
        return np.max(array, axis=0)
    return array


def _flatten_region_map(region_map: Mapping[str, np.ndarray]) -> np.ndarray:
    segments: List[np.ndarray] = []
    for contrib in region_map.values():
        segments.append(_per_base_magnitude(contrib))
    if not segments:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(segments)

def _load_threshold_cache(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    cache: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            cache[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return cache


def _save_threshold_cache(path: Path, cache: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


def _load_projected_h5(path: Path) -> np.ndarray:
    try:
        import deepdish as dd  # optional dependency for .h5 contribution bundles
    except ImportError as exc:
        raise ImportError(
            "deepdish is required to load precomputed .h5 contributions."
        ) from exc
    payload = dd.io.load(str(path))
    data = np.asarray(payload["projected_shap"]["seq"], dtype=np.float32)
    return data


def _load_attr_ohe(attr_path: Path, ohe_path: Path) -> np.ndarray:
    with np.load(attr_path) as attr_file:
        attr = np.asarray(attr_file["arr_0"], dtype=np.float32)
    with np.load(ohe_path) as ohe_file:
        ohe = np.asarray(ohe_file["arr_0"], dtype=np.float32)
    return attr * ohe


def _load_precomputed_array(spec: TrackSpec) -> Tuple[str, np.ndarray]:
    signature = spec.precomputed_signature()
    if signature is None:
        raise ValueError(
            f"Track '{spec.name}' requires precomputed contributions but none were provided."
        )
    cached = _PRECOMPUTED_ARRAY_CACHE.get(signature)
    if cached is not None:
        return signature, cached
    if spec.precomputed_contrib_h5 is not None:
        array = _load_projected_h5(spec.precomputed_contrib_h5)
    elif spec.precomputed_attr_npz is not None and spec.precomputed_ohe_npz is not None:
        array = _load_attr_ohe(spec.precomputed_attr_npz, spec.precomputed_ohe_npz)
    else:
        raise ValueError(
            f"Track '{spec.name}' is missing precomputed contribution file definitions."
        )
    _PRECOMPUTED_ARRAY_CACHE[signature] = array
    return signature, array


def _read_bed_coordinates(path: Path) -> List[Tuple[str, int, int]]:
    coords: List[Tuple[str, int, int]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start_str, end_str, *_ = line.rstrip("\n").split("\t")
            coords.append((chrom, int(start_str), int(end_str)))
    return coords


def _infer_bed_from_h5(h5_path: Path) -> Optional[Path]:
    name = h5_path.name
    if name.endswith(".profile_scores.h5"):
        bed_candidate = h5_path.with_name(name.replace(".profile_scores.h5", ".interpreted_regions.bed"))
        if bed_candidate.exists():
            return bed_candidate
        args_candidate = h5_path.with_name(name.replace(".profile_scores.h5", ".interpret.args.json"))
        if args_candidate.exists():
            with args_candidate.open() as handle:
                payload = json.load(handle)
            regions_value = payload.get("regions")
            if regions_value:
                candidate = Path(regions_value).expanduser()
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                if candidate.exists():
                    return candidate
    beds = sorted(h5_path.parent.glob("*.interpreted_regions.bed"))
    if len(beds) == 1:
        return beds[0]
    return None


def _infer_regions_bed(spec: TrackSpec) -> Path:
    if spec.precomputed_contrib_h5 is not None:
        bed_path = _infer_bed_from_h5(spec.precomputed_contrib_h5)
        if bed_path is not None:
            return bed_path
    if spec.bed_path is not None:
        return spec.bed_path
    if DEFAULT_INTERSECTION_BED.exists():
        return DEFAULT_INTERSECTION_BED
    raise ValueError(f"Unable to determine regions BED for track '{spec.name}'.")


def _load_precomputed_indices(
    spec: TrackSpec, total_regions: int, rows: int
) -> np.ndarray:
    if spec.precomputed_attr_npz is None:
        raise ValueError(
            f"Track '{spec.name}' is missing attr NPZ path for index alignment."
        )
    attr_path = spec.precomputed_attr_npz
    name = attr_path.name
    if name.endswith(".attr.npz"):
        idx_name = name.replace(".attr.npz", ".idx.npy")
    else:
        idx_name = f"{name}.idx.npy"
    idx_path = attr_path.with_name(idx_name)
    indices = np.load(idx_path, allow_pickle=False)
    if indices.dtype == np.bool_:
        selected = np.flatnonzero(indices)
    else:
        selected = np.asarray(indices, dtype=np.int64)
    if selected.size != rows:
        raise ValueError(
            f"Index map for track '{spec.name}' incompatible with contribution array."
        )
    return selected


def _load_precomputed_contribution_bundle(
    spec: TrackSpec,
) -> Tuple[str, np.ndarray, Dict[Tuple[str, int, int], int]]:
    signature = spec.precomputed_signature()
    if signature is None:
        raise ValueError(
            f"Track '{spec.name}' requires precomputed contributions but none were provided."
        )
    cached = _PRECOMPUTED_REGION_DATA_CACHE.get(signature)
    if cached is not None:
        return signature, cached[0], cached[1]

    signature, raw_array = _load_precomputed_array(spec)
    bed_path = _infer_regions_bed(spec)
    coords = _read_bed_coordinates(bed_path)
    total_regions = len(coords)

    if raw_array.shape[0] == total_regions:
        aligned = raw_array
    else:
        aligned = np.zeros((total_regions, raw_array.shape[1], raw_array.shape[2]), dtype=np.float32)
        indices = _load_precomputed_indices(spec, total_regions, raw_array.shape[0])
        for row_idx, region_idx in enumerate(indices):
            aligned[int(region_idx)] = raw_array[row_idx]

    region_lookup = {
        (chrom, start, end): idx for idx, (chrom, start, end) in enumerate(coords)
    }
    _PRECOMPUTED_REGION_DATA_CACHE[signature] = (aligned, region_lookup)
    return signature, aligned, region_lookup


def _matrix_from_precomputed(
    spec: TrackSpec, regions: Sequence[RegionSpec]
) -> Optional[np.ndarray]:
    if not spec.has_precomputed():
        return None
    _, aligned, region_lookup = _load_precomputed_contribution_bundle(spec)
    matrix = np.zeros((len(regions), aligned.shape[1], aligned.shape[2]), dtype=np.float32)
    for idx, region in enumerate(regions):
        key = (region.chrom, region.start, region.start + WINDOW_SPAN)
        region_idx = region_lookup.get(key)
        if region_idx is None:
            raise KeyError(
                f"Region {key} missing from precomputed contributions for track '{spec.name}'."
            )
        matrix[idx] = aligned[region_idx]
    print(f"[genome_tracker] Using precomputed contributions for '{spec.name}'")
    return matrix


def _flatten_precomputed_array(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected 3D contribution array, got shape {array.shape}")
    magnitudes = np.max(array, axis=1)
    return magnitudes.ravel()


def _threshold_cache_key(track_identifier: str, percentile: float, signature: str) -> str:
    return f"{track_identifier}:{percentile:.6f}:{signature}"


def _significant_intervals_from_contrib(
    contrib_profile: np.ndarray,
    percentile: float = 99.0,
    max_gap: int = 5,
    min_len: int = 10,
    threshold: float | None = None,
) -> List[Tuple[int, int]]:
    if contrib_profile.size == 0:
        return []

    per_base = _per_base_magnitude(contrib_profile)

    if not np.any(per_base):
        return []

    cutoff = threshold
    if cutoff is None:
        cutoff = float(np.percentile(per_base, percentile))
    if cutoff <= 0:
        return []

    sig = per_base >= cutoff
    intervals: List[Tuple[int, int]] = []
    idx = 0
    n = len(sig)

    while idx < n:
        if not sig[idx]:
            idx += 1
            continue

        region_start = idx
        last_sig_idx = idx
        gap = 0
        idx += 1

        while idx < n:
            if sig[idx]:
                gap = 0
                last_sig_idx = idx
                idx += 1
            else:
                gap += 1
                if gap < max_gap:
                    idx += 1
                else:
                    idx = last_sig_idx + 1
                    break

        region_end = idx
        if region_end - region_start >= min_len:
            intervals.append((region_start, region_end))

    print(
        f"[genome_tracker] Identified {len(intervals)} significant intervals using "
        f"threshold {cutoff:.6f} (percentile={percentile})"
    )
    return intervals


def build_contrib_region_spec(
    spec: TrackSpec,
    regions: Sequence[RegionSpec],
    contrib_map: Mapping[str, np.ndarray],
    output_dir: Path,
    percentile: float = 99.0,
    max_gap: int = 5,
    min_len: int = 10,
    threshold: float | None = None,
) -> TrackSpec | None:
    if not contrib_map:
        return None

    intermediate_dir = output_dir / "intermediate_files"
    derived_root = intermediate_dir / "derived_beds"
    derived_root.mkdir(parents=True, exist_ok=True)
    track_label = sanitize_identifier(spec.name)

    print(
        f"[genome_tracker] Building derived intervals for '{spec.name}' "
        f"(percentile={percentile}, threshold={threshold})"
    )
    for region in regions:
        print(f"[genome_tracker] Processing region '{region.name}'")
        region_contrib = contrib_map.get(region.name)
        print(f"[genome_tracker] Analyzing contribution profile for region '{region.name}'")
        intervals = _significant_intervals_from_contrib(
            region_contrib,
            percentile=percentile,
            max_gap=max_gap,
            min_len=min_len,
            threshold=threshold,
        )
        print(f"[genome_tracker] Found {len(intervals)} significant intervals in region '{region.name}'")
        region_dir = derived_root / sanitize_identifier(region.name)
        region_dir.mkdir(parents=True, exist_ok=True)
        region_file = region_dir / f"{track_label}.bed"
        with region_file.open("w") as region_bed:
            for start_idx, end_idx in intervals:
                chrom_start = region.start + start_idx
                chrom_end = region.start + end_idx
                region_bed.write(
                    f"{region.chrom}\t{chrom_start}\t{chrom_end}\t{track_label}\t0\t.\n"
                )

    print(
        f"[genome_tracker] Wrote derived intervals for '{spec.name}' across {len(regions)} regions"
    )
    return TrackSpec(
        name=f"{spec.name} significant (top1%)",
        file_type="bed",
        track_id=spec.track_id,
        per_region_bed_dir=derived_root,
        track_label=track_label,
    )


def load_bigwig(path: Path, chrom: str, start: int, end: int) -> np.ndarray:
    with pyBigWig.open(str(path)) as bw:
        values = bw.values(chrom, start, end, numpy=True)
    array = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def apply_gaussian_smoothing(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values

    radius = max(1, int(round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel_sum = kernel.sum()
    if kernel_sum == 0.0:
        return values
    kernel /= kernel_sum

    padded = np.pad(values, (radius,), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    trimmed = smoothed[radius:-radius]
    return trimmed.astype(np.float32, copy=False)


def fetch_bigwig_tracks(
    specs: Iterable[TrackSpec], chrom: str, start: int, end: int
) -> List[Tuple[TrackSpec, np.ndarray]]:
    loaded: List[Tuple[TrackSpec, np.ndarray]] = []
    for spec in specs:
        if spec.file_type != "bw":
            raise ValueError(f"Unsupported file type '{spec.file_type}' for track {spec.name}")
        if spec.location is None:
            raise ValueError(f"Track {spec.name} missing file_location")
        values = load_bigwig(spec.location, chrom, start, end)
        smoothed = apply_gaussian_smoothing(values, spec.smoothing_sigma)
        loaded.append((spec, smoothed))
    return loaded


def render_interval_segments(
    ax: plt.Axes,
    region: RegionSpec,
    segments: Sequence[IntervalSegment],
    label: str,
    pwm_forward: pd.DataFrame | None = None,
    pwm_reverse: pd.DataFrame | None = None,
) -> None:
    region_start = region.start
    region_end = region.start + WINDOW_SPAN
    has_segments = False
    label_idx = 1
    for segment in segments:
        rel_start = max(0.0, float(segment.start - region_start))
        rel_end = min(float(WINDOW_SPAN), float(segment.end - region_start))
        if rel_end <= rel_start:
            continue
        if segment.end <= region_start or segment.start >= region_end:
            continue
        has_segments = True
        ax.hlines(
            y=0.0,
            xmin=rel_start,
            xmax=rel_end,
            colors=DEFAULT_INTERVAL_COLOR,
            linewidth=4.0,
        )
        text_x = (rel_start + rel_end) / 2
        ax.text(
            text_x,
            0.02,
            str(label_idx),
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )
        label_idx += 1

    if not has_segments:
        ax.text(
            0.5,
            0.5,
            "No regions",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="gray",
        )

    ax.set_ylim(-0.05, 0.05)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=45)
    ax.set_xlim(0, WINDOW_SPAN - 1)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    boundary_left = OUTPUT_START
    boundary_right = OUTPUT_END
    ax.axvline(boundary_left, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.axvline(boundary_right, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    if pwm_forward is not None and pwm_reverse is not None:
        logo_width = "32%"
        logo_height = "50%"
        forward_ax = inset_axes(ax, width=logo_width, height=logo_height, loc="upper right", borderpad=1.0)
        logomaker.Logo(pwm_forward, ax=forward_ax, color_scheme="classic")
        forward_ax.set_xticks([])
        forward_ax.set_yticks([])
        forward_ax.set_title("+", fontsize=8)
        forward_ax.spines["top"].set_visible(False)
        forward_ax.spines["right"].set_visible(False)
        forward_ax.spines["bottom"].set_visible(False)
        forward_ax.spines["left"].set_visible(False)

        reverse_ax = inset_axes(ax, width=logo_width, height=logo_height, loc="center right", borderpad=1.0)
        logomaker.Logo(pwm_reverse, ax=reverse_ax, color_scheme="classic")
        reverse_ax.set_xticks([])
        reverse_ax.set_yticks([])
        reverse_ax.set_title("-", fontsize=8)
        reverse_ax.spines["top"].set_visible(False)
        reverse_ax.spines["right"].set_visible(False)
        reverse_ax.spines["bottom"].set_visible(False)
        reverse_ax.spines["left"].set_visible(False)





def _compute_track_contributions(args: Tuple[int, TrackSpec, Sequence[RegionSpec]]) -> Tuple[int, np.ndarray]:
    track_idx, spec, regions = args
    if spec.model_path is None:
        raise ValueError(f"Contribution track {spec.name} missing model_path")
    from chrombpnet.evaluation.interpret import input_utils
    from scripts_postprocessing.utils_shap import compute_region_projected_shap
    ns = SimpleNamespace(model_h5=str(spec.model_path))
    model = input_utils.load_model_wrapper(ns)
    track_matrix = np.zeros((len(regions), 4, WINDOW_SPAN), dtype=np.float32)
    for region_idx, region in enumerate(regions):
        chrom = region.chrom
        start = region.start
        end = start + WINDOW_SPAN
        contribution = compute_region_projected_shap(model, chrom, start, end)
        array = np.asarray(contribution, dtype=np.float32)
        if array.shape == (WINDOW_SPAN, 4):
            array = array.T
        elif array.shape != (4, WINDOW_SPAN):
            raise ValueError(
                f"Unexpected contribution shape {array.shape} for track {spec.name}"
            )
        track_matrix[region_idx] = array
    return track_idx, track_matrix


def prepare_contribution_data(
    specs: Sequence[TrackSpec],
    regions: Sequence[RegionSpec],
    cache_path: Path,
    num_workers: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[genome_tracker] Preparing contribution data cache at {cache_path}")
    if cache_path.exists():
        print(f"[genome_tracker] Loading contribution data from cache")
        payload = np.load(cache_path, allow_pickle=True)
        cached_regions = [str(name) for name in payload["region_names"]]
        cached_tracks = [str(name) for name in payload["track_names"]]
        if cached_regions == [region.name for region in regions] and cached_tracks == [spec.name for spec in specs]:
            bundle = np.asarray(payload["contributions"], dtype=np.float32)
            prepared: Dict[str, Dict[str, np.ndarray]] = {}
            for track_idx, spec in enumerate(specs):
                track_map = {
                    region.name: bundle[i, track_idx] for i, region in enumerate(regions)
                }
                prepared[spec.name] = track_map
            print(f"[genome_tracker] Cache hit: using contributions for {len(specs)} tracks across {len(regions)} regions")
            return prepared
        else:
            print("[genome_tracker] Cache mismatch (regions/tracks differ); recomputing")
    print("[genome_tracker] Computing contribution data afresh")

    contributions = np.zeros((len(regions), len(specs), 4, WINDOW_SPAN), dtype=np.float32)
    track_names = [spec.name for spec in specs]
    region_names = [region.name for region in regions]

    satisfied_tracks: Set[int] = set()
    for track_idx, spec in enumerate(specs):
        matrix = _matrix_from_precomputed(spec, regions)
        if matrix is None:
            continue
        contributions[:, track_idx] = matrix
        satisfied_tracks.add(track_idx)

    keras_specs = [
        (idx, spec)
        for idx, spec in enumerate(specs)
        if spec.file_type == "keras_contribution" and idx not in satisfied_tracks
    ]
    if keras_specs:
        max_workers = max(1, min(len(keras_specs), int(num_workers)))
        args = [(idx, spec, regions) for idx, spec in keras_specs]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for track_idx, track_matrix in executor.map(_compute_track_contributions, args):
                contributions[:, track_idx] = track_matrix
                satisfied_tracks.add(track_idx)

    for track_idx, spec in enumerate(specs):
        print(f"[genome_tracker] Processing contribution track '{spec.name}' ({track_idx + 1}/{len(specs)})")
        if track_idx in satisfied_tracks:
            continue
        if spec.file_type != "bpnet_lite_contribution":
            continue
        if spec.model_path is None:
            raise ValueError(f"Contribution track {spec.name} missing model_path")
        from scripts_postprocessing.utils_shap import compute_bpnet_lite_contributions
        region_payload = [
            (region.chrom, region.start, region.start + WINDOW_SPAN) for region in regions
        ]
        contrib_array = compute_bpnet_lite_contributions(
            spec.model_path,
            region_payload,
            window_span=WINDOW_SPAN,
        )
        if contrib_array.shape != (len(regions), 4, WINDOW_SPAN):
            raise ValueError(
                f"Unexpected bpnet-lite contribution shape {contrib_array.shape} "
                f"for track {spec.name}"
            )
        contributions[:, track_idx] = np.asarray(contrib_array, dtype=np.float32)
        satisfied_tracks.add(track_idx)

    np.savez_compressed(
        cache_path,
        region_names=np.array(region_names, dtype=object),
        track_names=np.array(track_names, dtype=object),
        contributions=contributions,
    )

    prepared: Dict[str, Dict[str, np.ndarray]] = {}
    for track_idx, spec in enumerate(specs):
        prepared[spec.name] = {
            region.name: contributions[region_idx, track_idx]
            for region_idx, region in enumerate(regions)
        }
    return prepared


def render_region(
    region: RegionSpec,
    render_items: Sequence[Tuple[str, TrackSpec, object]],
    output_path: Path,
    title: str,
) -> None:
    total_tracks = len(render_items)
    if total_tracks == 0:
        return

    height = max(3.5, total_tracks * 1.6)
    fig, axes = plt.subplots(
        total_tracks,
        1,
        sharex=True,
        figsize=(40, height),
        constrained_layout=False,
    )

    if total_tracks == 1:
        axes = [axes]  # type: ignore[assignment]

    positions = np.arange(WINDOW_SPAN)
    output_region = slice(OUTPUT_START, OUTPUT_END)
    boundary_left = OUTPUT_START
    boundary_right = OUTPUT_END

    group_bounds: Dict[str, Tuple[float, float]] = {}
    track_bounds: Dict[str, Tuple[float, float]] = {}
    for track_type, spec, data in render_items:
        if track_type != "contrib":
            continue
        flattened = np.asarray(data, dtype=np.float32).ravel()
        if flattened.size == 0:
            min_val = 0.0
            max_val = 0.0
        else:
            min_val = float(flattened.min())
            max_val = float(flattened.max())
        key = spec.shared_y_axis_scale_group
        if key:
            current = group_bounds.get(key)
            if current is None:
                group_bounds[key] = (min_val, max_val)
            else:
                group_bounds[key] = (min(current[0], min_val), max(current[1], max_val))
        else:
            track_bounds[spec.track_id] = (min_val, max_val)

    def _padded_limits(bounds: Tuple[float, float]) -> Tuple[float, float]:
        lower, upper = bounds
        span = upper - lower
        pad = 0.05 * span if span > 0 else 0.1
        lower -= pad
        upper += pad
        if lower == upper:
            epsilon = 0.1 if lower == 0 else abs(lower) * 0.05
            lower -= epsilon
            upper += epsilon
        return lower, upper

    group_limits = {key: _padded_limits(bounds) for key, bounds in group_bounds.items()}
    track_limits = {key: _padded_limits(bounds) for key, bounds in track_bounds.items()}

    axis_idx = 0
    for track_type, spec, payload in render_items:
        ax = axes[axis_idx]
        axis_idx += 1

        if track_type == "raw":
            values = np.asarray(payload, dtype=np.float32)
            if spec.positive_only:
                values = np.maximum(values, 0.0)
            if spec.use_full_window:
                masked = values
            else:
                masked = np.full_like(values, np.nan, dtype=np.float32)
                masked[output_region] = values[output_region]

            line = ax.plot(positions, masked, linewidth=1.0)
            color = line[0].get_color()
            valid = ~np.isnan(masked)
            if np.any(valid):
                ax.fill_between(positions, 0, np.nan_to_num(masked), where=valid, color=color, alpha=0.25)

            limit_values = masked if spec.use_full_window else masked[output_region]
            if limit_values.size > 0 and np.isfinite(limit_values).any():
                max_val = float(np.nanmax(limit_values))
                min_val = float(np.nanmin(limit_values))
                if spec.threshold is not None and np.isfinite(spec.threshold):
                    max_val = max(max_val, float(spec.threshold))
                    min_val = min(min_val, float(spec.threshold))
                upper = max_val * 1.1 if max_val >= 0 else max_val * 1.1
                lower = 0.0 if min_val >= 0 else min_val * 1.1
                if lower == upper:
                    epsilon = 0.1 if lower == 0 else abs(lower) * 0.05
                    lower -= epsilon
                    upper += epsilon
                ax.set_ylim(lower, upper)

            if spec.threshold is not None and np.isfinite(spec.threshold):
                ax.axhline(
                    spec.threshold,
                    color=color,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.9,
                )

            ax.set_ylabel(spec.name, rotation=0, ha="right", va="center", labelpad=45)
            ax.set_xlim(0, WINDOW_SPAN - 1)
            ax.tick_params(axis="y", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
            if not spec.use_full_window:
                ax.axvline(boundary_left, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
                ax.axvline(boundary_right, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

        elif track_type == "interval":
            segments = payload  # type: ignore[assignment]
            render_interval_segments(ax, region, segments, spec.name, None, None)

        elif track_type == "contrib":
            data = np.asarray(payload, dtype=np.float32)
            df = pd.DataFrame(data.T, columns=["A", "C", "G", "T"])
            logomaker.Logo(df, ax=ax, color_scheme="classic", vpad=0.1)
            ax.set_ylabel(spec.name, rotation=0, ha="right", va="center", labelpad=45)
            ax.set_xlim(0, WINDOW_SPAN - 1)
            if spec.shared_y_axis_scale_group and spec.shared_y_axis_scale_group in group_limits:
                ax.set_ylim(group_limits[spec.shared_y_axis_scale_group])
            else:
                limit_key = spec.track_id
                if limit_key in track_limits:
                    ax.set_ylim(track_limits[limit_key])
            ax.tick_params(axis="y", labelsize=8)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
            ax.axvline(boundary_left, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.axvline(boundary_right, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

        if axis_idx != total_tracks:
            ax.set_xticklabels([])

    tick_positions = np.linspace(0, WINDOW_SPAN - 1, 11, dtype=int)
    tick_labels = [f"{region.start + int(pos):,}" for pos in tick_positions]

    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel(f"{region.chrom} coordinate")

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_region_job(
    region: RegionSpec,
    render_plan: Sequence[Tuple[str, TrackSpec]],
    contrib_arrays: Dict[str, np.ndarray],
    plots_dir: Path,
    output_root: Path,
    bar_specs: Sequence[TrackSpec],
    track_lookup: Mapping[str, str],
) -> Tuple[str, List[Tuple[str, Path]]]:
    print(f"[genome_tracker] Rendering region '{region.name}'")
    region_end = region.start + WINDOW_SPAN
    figure_path = plots_dir / region_filename(region)
    render_items: List[Tuple[str, TrackSpec, object]] = []
    for track_kind, spec in render_plan:
        if track_kind == "raw":
            if spec.location is None:
                continue
            values = load_bigwig(spec.location, region.chrom, region.start, region_end)
            values = apply_gaussian_smoothing(values, spec.smoothing_sigma)
            render_items.append(("raw", spec, values))
        elif track_kind == "interval":
            segments = bed_segments_for_region(spec, region)
            render_items.append(("interval", spec, segments))
        elif track_kind == "motif":
            hits = compute_motif_hits_for_region(spec, region)
            write_motif_hits_bed(output_root, region, spec, hits)
            render_items.append(("interval", spec, hits))
        elif track_kind == "contrib":
            data = contrib_arrays.get(spec.name)
            if data is None:
                continue
            render_items.append(("contrib", spec, data))
        elif track_kind == "seqlet":
            segments = seqlet_segments_for_region(spec, region)
            render_items.append(("interval", spec, segments))
    figure_title = f"{region.name} ({region.chrom}:{region.start:,}-{region_end:,})"
    render_region(region, render_items, figure_path, figure_title)
    bar_entries = _generate_bar_plot_entries(
        region,
        bar_specs,
        contrib_arrays,
        plots_dir,
        track_lookup,
    )
    return figure_path.name, bar_entries


def write_overview_html(
    output_path: Path,
    title: str | None,
    entries: Sequence[Tuple[RegionSpec, Path, List[Tuple[str, Path]]]],
    motif_entries: Sequence[Tuple[str, Path, Path]],
) -> None:
    print(f"[genome_tracker] Writing overview HTML to {output_path}")
    heading = title if title else "Genome tracker overview"
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{heading}</title>",
        '<meta charset="utf-8">',
        '<style>body{font-family:Arial, sans-serif; margin:20px;} h2{margin-top:40px;} img{max-width:100%; height:auto; border:1px solid #ccc;} .motif-pair{display:flex; gap:20px; flex-wrap:wrap;} .motif-pair img{width:45%; min-width:280px;} .bar-plots{border:1px solid #ddd; padding:12px; margin:16px 0;} .bar-plots img{max-width:60%; min-width:280px;} .text-tracks{border:1px solid #ddd; padding:12px; margin:16px 0;} .text-track-block{margin-bottom:12px;} .text-track-block pre{background:#f7f7f7; padding:8px; white-space:pre-wrap;}</style>',
        "</head>",
        "<body>",
        f"<h1>{heading}</h1>",
        f"<p><strong>Generated:</strong> {timestamp}</p>",
    ]

    if motif_entries:
        lines.append("<h2>Motif Logos</h2>")
        for motif_name, forward_path, reverse_path in motif_entries:
            lines.append(f"<h3>{motif_name}</h3>")
            lines.append('<div class="motif-pair">')
            lines.append(
                f'<div><p>Forward</p><img src="{forward_path.as_posix()}" alt="{motif_name} forward"></div>'
            )
            lines.append(
                f'<div><p>Reverse</p><img src="{reverse_path.as_posix()}" alt="{motif_name} reverse"></div>'
            )
            lines.append("</div>")

    for region, relative_path, bar_entries in entries:
        region_end = region.start + WINDOW_SPAN
        display_name = f"{region.name} ({region.chrom}:{region.start:,}-{region_end:,})"
        rel_str = relative_path.as_posix()
        lines.append(f"<h2>{display_name}</h2>")
        lines.append(f'<img src="{rel_str}" alt="{display_name}">')
        if bar_entries:
            lines.append('<div class="bar-plots">')
            lines.append("<h3>Supplementary Bar Plots</h3>")
            for track_label, rel_bar_path in bar_entries:
                rel_bar_str = rel_bar_path.as_posix()
                escaped_label = html.escape(track_label)
                lines.append(f"<div><strong>{escaped_label}</strong><br>")
                lines.append(f'<img src="{rel_bar_str}" alt="{escaped_label}"></div>')
            lines.append("</div>")

    lines.append("</body>")
    lines.append("</html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write("\n".join(lines))


def _collect_contrib_arrays(
    region: RegionSpec,
    contrib_specs: Sequence[TrackSpec],
    contrib_cache: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for spec in contrib_specs:
        region_map = contrib_cache.get(spec.name)
        if region_map is None or region.name not in region_map:
            raise KeyError(
                f"Missing contributions for track '{spec.name}' and region '{region.name}'"
            )
        value = region_map[region.name]
        arrays[spec.name] = value
        if spec.track_id:
            arrays[spec.track_id] = value
    return arrays


def _collapse_contribution_trace(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return arr.sum(axis=0)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_center = a - a.mean()
    b_center = b - b.mean()
    denom = np.linalg.norm(a_center) * np.linalg.norm(b_center)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a_center, b_center) / denom)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    ranks_a = stats.rankdata(a)
    ranks_b = stats.rankdata(b)
    return _safe_pearson(ranks_a, ranks_b)


def _bar_plot_rel_path(spec: TrackSpec, region: RegionSpec) -> Path:
    subdir = spec.bar_output_subdir or "barplots"
    track_dir = spec.track_id or sanitize_identifier(spec.name)
    filename = f"{sanitize_identifier(region.name)}.png"
    return Path("indv_plots") / subdir / track_dir / filename


def _generate_bar_plot_entries(
    region: RegionSpec,
    bar_specs: Sequence[TrackSpec],
    contrib_arrays: Mapping[str, np.ndarray],
    plots_dir: Path,
    track_lookup: Mapping[str, str],
) -> List[Tuple[str, Path]]:
    if not bar_specs:
        return []
    entries: List[Tuple[str, Path]] = []
    output_dir = plots_dir.parent
    for spec in bar_specs:
        comparisons = spec.bar_comparisons or []
        if not comparisons:
            continue
        metric = (spec.bar_metric or "pearson").lower()
        focus_slice = _resolve_focus_slice(region, spec)
        values: List[float] = []
        labels: List[str] = []
        annotations: List[str] = []
        for comp in comparisons:
            track1_id = comp.get("track1")
            track2_id = comp.get("track2")
            if track1_id is None or track2_id is None:
                continue
            arr1 = contrib_arrays.get(track1_id)
            if arr1 is None:
                mapped = track_lookup.get(track1_id or "")
                if mapped:
                    arr1 = contrib_arrays.get(mapped)
            arr2 = contrib_arrays.get(track2_id)
            if arr2 is None:
                mapped = track_lookup.get(track2_id or "")
                if mapped:
                    arr2 = contrib_arrays.get(mapped)
            if arr1 is None or arr2 is None:
                print(
                    f"[genome_tracker] Warning: Missing contributions for bar track '{spec.name}' "
                    f"({track1_id}, {track2_id}) in region '{region.name}'"
                )
                value = float("nan")
            else:
                trace1 = _collapse_contribution_trace(arr1)
                trace2 = _collapse_contribution_trace(arr2)
                if focus_slice:
                    start_idx, end_idx = focus_slice
                    end_idx = min(end_idx, trace1.size, trace2.size)
                    if end_idx <= start_idx:
                        value = float("nan")
                        labels.append(label)
                        values.append(value)
                        annotations.append("NA")
                        continue
                    trace1 = trace1[start_idx:end_idx]
                    trace2 = trace2[start_idx:end_idx]
                if metric == "spearman":
                    value = _safe_spearman(trace1, trace2)
                else:
                    value = _safe_pearson(trace1, trace2)
            label = comp.get("label") or f"{track1_id} vs {track2_id}"
            labels.append(label)
            values.append(value)
            annotations.append("NA" if not np.isfinite(value) else f"{value:.2f}")

        if not labels:
            continue
        rel_path = _bar_plot_rel_path(spec, region)
        abs_path = output_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(labels))
        plot_vals = [v if np.isfinite(v) else 0.0 for v in values]
        ax.bar(x, plot_vals, color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ylabel = "Pearson correlation" if metric == "pearson" else "Spearman correlation"
        ax.set_ylabel(ylabel)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for xpos, annot in enumerate(annotations):
            ax.text(xpos, plot_vals[xpos], annot, ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(abs_path, dpi=150)
        plt.close(fig)
        entries.append((spec.name, rel_path))
    return entries


def _existing_bar_entries(
    region: RegionSpec,
    bar_specs: Sequence[TrackSpec],
    output_dir: Path,
) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    for spec in bar_specs:
        rel_path = _bar_plot_rel_path(spec, region)
        abs_path = output_dir / rel_path
        if abs_path.exists():
            entries.append((spec.name, rel_path))
    return entries


def _load_focus_bed(path: Path) -> Dict[str, Tuple[List[int], List[int]]]:
    key = str(path.resolve())
    cached = _FOCUS_BED_CACHE.get(key)
    if cached is not None:
        return cached
    intervals: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            intervals[chrom].append((start, end))
    processed: Dict[str, Tuple[List[int], List[int]]] = {}
    for chrom, spans in intervals.items():
        spans.sort()
        starts = [entry[0] for entry in spans]
        ends = [entry[1] for entry in spans]
        processed[chrom] = (starts, ends)
    _FOCUS_BED_CACHE[key] = processed
    return processed


def _center_peak_slice(
    region: RegionSpec,
    focus_path: Path,
) -> Tuple[int, int] | None:
    records = _load_focus_bed(focus_path)
    chrom_entries = records.get(region.chrom)
    if chrom_entries is None:
        return None
    starts, ends = chrom_entries
    center = region.start + WINDOW_SPAN // 2
    idx = bisect_right(starts, center) - 1
    if idx < 0:
        return None
    if center >= ends[idx]:
        return None
    start = max(region.start, starts[idx])
    end = min(region.start + WINDOW_SPAN, ends[idx])
    if end <= start:
        return None
    rel_start = start - region.start
    rel_end = end - region.start
    return rel_start, rel_end


def _resolve_focus_slice(region: RegionSpec, spec: TrackSpec) -> Tuple[int, int] | None:
    if spec.focus_bed is None or spec.focus_mode is None:
        return None
    focus_mode = spec.focus_mode.lower()
    if focus_mode == "center_peak":
        return _center_peak_slice(region, spec.focus_bed)
    return None


def main() -> None:
    cli_args = parse_args()
    span = cli_args.window_span if cli_args.window_span is not None else DEFAULT_WINDOW_SPAN
    _set_window_params(span, cli_args.output_span)
    regions = load_region_specs(cli_args.region_config)
    all_specs = load_track_specs(cli_args.track_config)

    raw_specs, interval_specs, motif_specs, contrib_specs, seqlet_specs, _, bar_specs = categorize_specs(all_specs)
    derived_bed_specs: List[TrackSpec] = []
    output_dir = cli_args.output_dir
    plots_dir = output_dir / "indv_plots"
    intermediate_dir = output_dir / "intermediate_files"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    cache_path = intermediate_dir / "cached_contributions.npz"
    overview_path = output_dir / "main.html"
    save_path = output_dir / "processing.log"
    threshold_cache_path = intermediate_dir / "global_threshold_cache.json"
    motif_logo_entries: List[Tuple[str, Path, Path]] = []
    for spec in motif_specs:
        files = ensure_motif_logo_images(spec, output_dir)
        if files:
            forward_rel = Path("indv_plots") / "motif_logos" / files[0].name
            reverse_rel = Path("indv_plots") / "motif_logos" / files[1].name
            motif_logo_entries.append((spec.name, forward_rel, reverse_rel))

    selected_regions = list(regions)
    if not selected_regions:
        raise ValueError("No regions available for rendering")

    regions_to_render = selected_regions
    expected_plots = [plots_dir / region_filename(region) for region in regions_to_render]
    figures_exist = all(path.exists() for path in expected_plots)
    save_file = save_path.exists() and figures_exist
    if save_file and bar_specs:
        for region in regions_to_render:
            entries = _existing_bar_entries(region, bar_specs, output_dir)
            if len(entries) < len(bar_specs):
                save_file = False
                break

    overview_entries: List[Tuple[RegionSpec, Path, List[Tuple[str, Path]]]] = []

    if save_file:
        print("[genome_tracker] Skipping render; outputs already exist.")
        print(f"[genome_tracker] Using cached outputs in {output_dir}")
        summary_text = save_path.read_text().strip()
        if summary_text:
            print(summary_text)
        overview_entries = [
            (
                region,
                Path("indv_plots") / region_filename(region),
                _existing_bar_entries(region, bar_specs, output_dir),
            )
            for region in regions_to_render
        ]
    else:
        plots_dir.mkdir(parents=True, exist_ok=True)

        contrib_cache: Dict[str, Dict[str, np.ndarray]] = {}
        if contrib_specs:
            print("Preparing contribution data...")
            contrib_cache = prepare_contribution_data(
                contrib_specs, regions, cache_path, cli_args.num_workers
            )

        contrib_mapping: Mapping[str, Mapping[str, np.ndarray]] = contrib_cache
        threshold_cache = _load_threshold_cache(threshold_cache_path)
        threshold_cache_dirty = False
        contrib_lookup = {spec.track_id: spec for spec in contrib_specs if spec.track_id}
        auto_threshold_specs: List[Tuple[TrackSpec, TrackSpec]] = []
        for member in all_specs:
            if member.file_type != "contrib_motifs":
                continue
            if member.threshold is not None:
                continue
            base_spec = contrib_lookup.get(member.source_track_id)
            if base_spec is None:
                print(
                    f"[genome_tracker] Warning: Derived track '{member.name}' has no contribution source '{member.source_track_id}'."
                )
                continue
            if not base_spec.has_precomputed():
                continue
            auto_threshold_specs.append((member, base_spec))

        if auto_threshold_specs:
            print("[genome_tracker] Computing global thresholds for derived contribution tracks...")
        for member, base_spec in auto_threshold_specs:
            signature, array = _load_precomputed_array(base_spec)
            cache_id = member.track_id or member.name
            key = _threshold_cache_key(cache_id, member.percentile, signature)
            cached_value = threshold_cache.get(key)
            if cached_value is not None:
                threshold_value = cached_value
                print(
                    f"[genome_tracker] Threshold for '{member.name}' "
                    f"(percentile={member.percentile}) -> {threshold_value:.6f} (cache hit)"
                )
            else:
                flattened = _flatten_precomputed_array(array)
                if flattened.size == 0:
                    threshold_value = 0.0
                else:
                    threshold_value = float(np.percentile(flattened, member.percentile))
                threshold_cache[key] = threshold_value
                threshold_cache_dirty = True
                print(
                    f"[genome_tracker] Threshold for '{member.name}' "
                    f"(percentile={member.percentile}) -> {threshold_value:.6f}"
                )
            member.threshold = threshold_value
        if threshold_cache_dirty:
            _save_threshold_cache(threshold_cache_path, threshold_cache)
        resolved_specs: List[TrackSpec] = []
        print("Building derived contribution BED tracks...")
        for spec in all_specs:
            if spec.file_type == "contrib_motifs":
                base_spec = contrib_lookup.get(spec.source_track_id)
                if base_spec is None:
                    print(
                        f"[genome_tracker] Warning: No contribution track '{spec.source_track_id}' for derived track '{spec.name}'"
                    )
                    continue
                region_map = contrib_mapping.get(base_spec.name)
                if not region_map:
                    print(
                        f"[genome_tracker] Warning: No contribution data for '{base_spec.name}' when building '{spec.name}'"
                    )
                    continue
                derived_bed = build_contrib_region_spec(
                    base_spec,
                    regions_to_render,
                    region_map,
                    output_dir,
                    percentile=spec.percentile,
                    threshold=spec.threshold,
                    max_gap=spec.max_gap,
                    min_len=spec.min_len,
                )
                if derived_bed:
                    derived_bed.name = spec.name
                    derived_bed.file_type = "bed"
                    derived_bed_specs.append(derived_bed)
                    resolved_specs.append(derived_bed)
            else:
                resolved_specs.append(spec)
        if derived_bed_specs:
            print(
                f"[genome_tracker] Generated {len(derived_bed_specs)} derived BED tracks from contributions"
            )
        all_specs = resolved_specs
        raw_specs, interval_specs, motif_specs, contrib_specs, seqlet_specs, _, bar_specs = categorize_specs(all_specs)
        render_plan = []
        for spec in all_specs:
            if spec.file_type == "bw":
                render_plan.append(("raw", spec))
            elif spec.file_type == "bed":
                render_plan.append(("interval", spec))
            elif spec.file_type == "motif_pwm":
                render_plan.append(("motif", spec))
            elif spec.file_type in {"keras_contribution", "bpnet_lite_contribution"}:
                render_plan.append(("contrib", spec))
            elif spec.file_type == "seqlet_modisco":
                render_plan.append(("seqlet", spec))
        track_lookup = {spec.track_id: spec.name for spec in all_specs if spec.track_id}
        job_specs: List[Tuple] = []
        for region in regions_to_render:
            if contrib_specs:
                arrays = _collect_contrib_arrays(region, contrib_specs, contrib_mapping)
            else:
                arrays = {}
            job_specs.append(
                (
                    _render_region_job,
                    (
                        region,
                        tuple(render_plan),
                        arrays,
                        plots_dir,
                        output_dir,
                        tuple(bar_specs),
                        track_lookup,
                    ),
                    {},
                )
            )

        print(f"[genome_tracker] Queued {len(job_specs)} region jobs")
        print("Rendering regions...")

        sbatch_available = which("sbatch") is not None
        force_parallel = bool(cli_args.force_parallel)

        if sbatch_available:
            submitit_dir = Path("logs/submitit")
            submitit_dir.mkdir(parents=True, exist_ok=True)
            executor = AutoExecutor(folder=str(submitit_dir), cluster="slurm")
            executor.update_parameters(
                timeout_min=60,
                slurm_job_name="parallel_genome_browser",
                cpus_per_task=1,
                mem_gb=36,
                slurm_partition="interactive,main,pritykinlab",
            )
            jobs = [
                executor.submit(job_fn, *job_args, **job_kwargs)
                for job_fn, job_args, job_kwargs in job_specs
            ]
            job_results: List[Tuple[str, List[Tuple[str, Path]]]] = []
            for job in tqdm(jobs, desc="Submitit jobs: One per region"):
                result = job.result()
                job_results.append(result)
        elif force_parallel and job_specs:
            max_workers = min(128, len(job_specs))
            print(f"[genome_tracker] sbatch unavailable; running locally with {max_workers} workers (--force-parallel).")
            job_results: List[Tuple[str, List[Tuple[str, Path]]]] = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(job_fn, *job_args, **job_kwargs)
                    for job_fn, job_args, job_kwargs in job_specs
                ]
                for future in tqdm(futures, desc="Local parallel jobs: One per region"):
                    job_results.append(future.result())
        else:
            print("[genome_tracker] sbatch unavailable; running jobs sequentially.")
            job_results: List[Tuple[str, List[Tuple[str, Path]]]] = []
            for job_fn, job_args, job_kwargs in tqdm(job_specs, desc="Local jobs: One per region"):
                result = job_fn(*job_args, **job_kwargs)
                job_results.append(result)

        for (job_fn, job_args, job_kwargs), result in zip(job_specs, job_results):
            region = job_args[0]
            figure_name, bar_entries = result
            overview_entries.append((region, Path("indv_plots") / figure_name, bar_entries))

        summary_lines = [
            f"regions: {', '.join(region.name for region in regions_to_render)}",
            f"raw_tracks: {', '.join(spec.name for spec in raw_specs)}",
            f"contrib_tracks: {', '.join(spec.name for spec in contrib_specs)}",
            f"motif_tracks: {', '.join(spec.name for spec in motif_specs)}",
            f"bed_tracks: {', '.join(spec.name for spec in interval_specs)}",
            f"bar_tracks: {', '.join(spec.name for spec in bar_specs)}",
        ]
        save_path.write_text("\n".join(summary_lines) + "\n")

    write_overview_html(overview_path, cli_args.title, overview_entries, motif_logo_entries)
    print(f"[genome_tracker] Completed genome tracker run in {output_dir}")


if __name__ == "__main__":
    main()
