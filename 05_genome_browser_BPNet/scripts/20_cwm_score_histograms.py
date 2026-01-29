#!/usr/bin/env python3
import argparse
import base64
import csv
import html
import json
import os
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numpy.lib.stride_tricks import sliding_window_view

DEFAULT_TEST_CHROMS = ("chr1", "chr8", "chr9")


class PatternRef(NamedTuple):
    modisco_path: Path
    metacluster: str
    pattern: str
    task: str
    cwm: np.ndarray

def _load_tasks(handle: h5py.File) -> list[str]:
    tasks = []
    for name in ("Oct4", "Sox2", "Nanog", "Klf4"):
        if f"hyp_imp/{name}/profile/wn" in handle:
            tasks.append(name)
    if not tasks:
        raise KeyError("No hyp_imp profile/wn datasets found in DeepLIFT file.")
    return tasks


def _load_attributions(
    h5_path: Path, test_chroms: set[str] | None
) -> tuple[np.ndarray, dict[int, int]]:
    with h5py.File(h5_path, "r") as handle:
        chroms = handle["metadata/range/chr"][:]
        chroms = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in chroms]
        tasks = _load_tasks(handle)
        attr_sum = None
        for task in tasks:
            dataset = f"hyp_imp/{task}/profile/wn"
            data = handle[dataset][:].astype(np.float32)
            if attr_sum is None:
                attr_sum = data
            else:
                attr_sum += data
    if attr_sum is None:
        raise RuntimeError("No attribution data loaded from DeepLIFT file.")
    index_map = {idx: idx for idx in range(attr_sum.shape[0])}
    if test_chroms:
        chroms_arr = np.array(chroms)
        mask = np.array([c in test_chroms for c in chroms_arr], dtype=bool)
        if not mask.any():
            raise ValueError("No regions found on requested test chromosomes.")
        selected = np.flatnonzero(mask)
        attr_sum = attr_sum[mask]
        index_map = {int(orig_idx): new_idx for new_idx, orig_idx in enumerate(selected)}
    return attr_sum, index_map


def _reverse_complement_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] != 4:
        raise ValueError("CWM matrix must have 4 columns (A,C,G,T).")
    return matrix[::-1, :][:, [3, 2, 1, 0]]


def _scan_cwm(
    contribs: np.ndarray, cwm: np.ndarray, chunk_size: int
) -> np.ndarray:
    num_regions, seq_len, channels = contribs.shape
    if channels != 4:
        raise ValueError("Contribution matrices must have 4 channels.")
    kernel = int(cwm.shape[0])
    scan_len = seq_len - kernel + 1
    if scan_len <= 0:
        return np.zeros((num_regions, 0), dtype=np.float32)
    scores = np.empty((num_regions, scan_len), dtype=np.float32)
    for start in range(0, num_regions, chunk_size):
        end = min(num_regions, start + chunk_size)
        chunk = contribs[start:end]
        windows = sliding_window_view(chunk, window_shape=kernel, axis=1)
        # windows shape: (batch, scan_len, channels, kernel)
        chunk_scores = np.tensordot(windows, cwm, axes=([2, 3], [1, 0]))
        scores[start:end] = chunk_scores.astype(np.float32, copy=False)
    return scores


def _bins_for_scores(scores: np.ndarray, bins: int) -> np.ndarray:
    if scores.size == 0:
        return np.linspace(-1.0, 1.0, num=bins + 1)
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if min_score == max_score:
        min_score -= 1.0
        max_score += 1.0
    return np.linspace(min_score, max_score, num=bins + 1)


def _score_range(scores: np.ndarray) -> tuple[float, float]:
    if scores.size == 0:
        return -1.0, 1.0
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if min_score == max_score:
        min_score -= 1.0
        max_score += 1.0
    return min_score, max_score


def _encode_base64_float32(values: np.ndarray) -> str:
    if values.size == 0:
        return ""
    arr = np.asarray(values, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _write_interactive_histogram(
    scores: np.ndarray,
    seqlet_scores: np.ndarray,
    html_path: Path,
    title: str,
    bins: int,
) -> None:
    if scores.size and seqlet_scores.size:
        combined = np.concatenate([scores, seqlet_scores], axis=0)
    else:
        combined = scores if scores.size else seqlet_scores
    range_min, range_max = _score_range(combined)

    scores_b64 = _encode_base64_float32(scores)
    seqlet_b64 = _encode_base64_float32(seqlet_scores)

    title_text = html.escape(title)
    title_json = json.dumps(title)
    scores_json = json.dumps(scores_b64)
    seqlets_json = json.dumps(seqlet_b64)
    range_json = json.dumps([range_min, range_max])
    bins_json = json.dumps(int(bins))

    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        f"<title>{title_text}</title>",
        "<script charset=\"utf-8\" src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 16px; }",
        "#histogram { width: 100%; height: 520px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title_text}</h1>",
        "<p>Scroll or box-zoom to zoom. Double-click to reset. "
        "Bins are recomputed for the visible range.</p>",
        "<div id=\"histogram\"></div>",
        "<script>",
        f"const TITLE = {title_json};",
        f"const SCORE_B64 = {scores_json};",
        f"const SEQLET_B64 = {seqlets_json};",
        f"const FULL_RANGE = {range_json};",
        f"const NBINS = {bins_json};",
        "function b64ToFloat32Array(b64) {",
        "  if (!b64) {",
        "    return new Float32Array();",
        "  }",
        "  const binary = atob(b64);",
        "  const len = binary.length;",
        "  const bytes = new Uint8Array(len);",
        "  for (let i = 0; i < len; i += 1) {",
        "    bytes[i] = binary.charCodeAt(i);",
        "  }",
        "  return new Float32Array(bytes.buffer);",
        "}",
        "function computeHistogram(values, range, bins) {",
        "  let min = range[0];",
        "  let max = range[1];",
        "  if (!Number.isFinite(min) || !Number.isFinite(max)) {",
        "    return { centers: [], densities: [], n: 0, width: 0 };",
        "  }",
        "  if (min === max) {",
        "    min -= 1.0;",
        "    max += 1.0;",
        "  }",
        "  const width = (max - min) / bins;",
        "  const counts = new Array(bins).fill(0);",
        "  let n = 0;",
        "  for (let i = 0; i < values.length; i += 1) {",
        "    const v = values[i];",
        "    if (!Number.isFinite(v)) {",
        "      continue;",
        "    }",
        "    if (v < min || v > max) {",
        "      continue;",
        "    }",
        "    let idx = Math.floor((v - min) / width);",
        "    if (idx >= bins) {",
        "      idx = bins - 1;",
        "    }",
        "    counts[idx] += 1;",
        "    n += 1;",
        "  }",
        "  const centers = counts.map((_, idx) => min + (idx + 0.5) * width);",
        "  const densities = counts.map((count) => (n > 0 ? count / (n * width) : 0));",
        "  return { centers, densities, n, width };",
        "}",
        "const scores = b64ToFloat32Array(SCORE_B64);",
        "const seqlets = b64ToFloat32Array(SEQLET_B64);",
        "const container = document.getElementById('histogram');",
        "function buildTraces(range) {",
        "  const allHist = computeHistogram(scores, range, NBINS);",
        "  const traces = [",
        "    {",
        "      type: 'bar',",
        "      x: allHist.centers,",
        "      y: allHist.densities,",
        "      name: `All windows (n=${allHist.n})`,",
        "      opacity: 0.6,",
        "      marker: { color: '#1f77b4' },",
        "      hovertemplate: 'Score=%{x:.6f}<br>Density=%{y:.6f}<extra></extra>'",
        "    }",
        "  ];",
        "  if (seqlets.length > 0) {",
        "    const seqHist = computeHistogram(seqlets, range, NBINS);",
        "    traces.push({",
        "      type: 'bar',",
        "      x: seqHist.centers,",
        "      y: seqHist.densities,",
        "      name: `Seqlets (n=${seqHist.n})`,",
        "      opacity: 0.6,",
        "      marker: { color: '#ff7f0e' },",
        "      hovertemplate: 'Score=%{x:.6f}<br>Density=%{y:.6f}<extra></extra>'",
        "    });",
        "  }",
        "  return traces;",
        "}",
        "function render(range) {",
        "  const traces = buildTraces(range);",
        "  const layout = {",
        "    title: TITLE,",
        "    barmode: 'overlay',",
        "    bargap: 0.05,",
        "    xaxis: { title: 'CWM score', range: range },",
        "    yaxis: { title: 'Density', autorange: true }",
        "  };",
        "  return Plotly.react(container, traces, layout, {",
        "    responsive: true,",
        "    scrollZoom: true",
        "  });",
        "}",
        "let isUpdating = false;",
        "render(FULL_RANGE).then(() => {",
        "  container.on('plotly_relayout', (eventData) => {",
        "    if (isUpdating) {",
        "      return;",
        "    }",
        "    let nextRange = null;",
        "    if (eventData && eventData['xaxis.autorange']) {",
        "      nextRange = FULL_RANGE;",
        "    } else if (eventData && eventData['xaxis.range[0]'] !== undefined) {",
        "      const start = parseFloat(eventData['xaxis.range[0]']);",
        "      const end = parseFloat(eventData['xaxis.range[1]']);",
        "      if (Number.isFinite(start) && Number.isFinite(end)) {",
        "        nextRange = [start, end];",
        "      }",
        "    }",
        "    if (!nextRange) {",
        "      return;",
        "    }",
        "    if (nextRange[0] === nextRange[1]) {",
        "      nextRange = [nextRange[0] - 1.0, nextRange[1] + 1.0];",
        "    }",
        "    isUpdating = true;",
        "    render(nextRange).then(() => {",
        "      isUpdating = false;",
        "    });",
        "  });",
        "});",
        "</script>",
        "</body>",
        "</html>",
    ]
    html_path.write_text("\n".join(html_lines))


def _plot_histogram(
    scores: np.ndarray,
    seqlet_scores: np.ndarray,
    png_path: Path,
    title: str,
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if scores.size and seqlet_scores.size:
        combined = np.concatenate([scores, seqlet_scores], axis=0)
    else:
        combined = scores if scores.size else seqlet_scores
    hist_bins = _bins_for_scores(combined, bins)
    if scores.size:
        ax.hist(
            scores,
            bins=hist_bins,
            density=True,
            alpha=0.6,
            label=f"All windows (n={scores.size})",
        )
    if seqlet_scores.size:
        ax.hist(
            seqlet_scores,
            bins=hist_bins,
            density=True,
            alpha=0.6,
            label=f"Seqlets (n={seqlet_scores.size})",
        )
    ax.set_xlabel("CWM score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    if scores.size or seqlet_scores.size:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def _discover_modisco_paths(deeplift_path: Path) -> list[Path]:
    base = deeplift_path.parent
    paths = sorted(base.glob("deeplift/*/out/profile/wn/modisco.h5"))
    return paths


def _collect_patterns(modisco_paths: list[Path]) -> list[PatternRef]:
    patterns: list[PatternRef] = []
    base = "metacluster_idx_to_submetacluster_results"
    for path in modisco_paths:
        with h5py.File(path, "r") as handle:
            if base not in handle:
                continue
            for metacluster in handle[base].keys():
                patterns_group = (
                    f"{base}/{metacluster}/seqlets_to_patterns_result/patterns"
                )
                if patterns_group not in handle:
                    continue
                for pattern_name in handle[patterns_group].keys():
                    if not pattern_name.startswith("pattern_"):
                        continue
                    pattern_group = handle[f"{patterns_group}/{pattern_name}"]
                    if not isinstance(pattern_group, h5py.Group):
                        continue
                    task_names = [
                        key
                        for key in pattern_group.keys()
                        if key not in {"seqlets_and_alnmts", "sequence"}
                    ]
                    if not task_names:
                        continue
                    task_name = task_names[0]
                    cwm_path = (
                        f"{patterns_group}/{pattern_name}/{task_name}"
                        "/profile/wn_contrib_scores/fwd"
                    )
                    if cwm_path not in handle:
                        continue
                    cwm = handle[cwm_path][:].astype(np.float32)
                    if cwm.ndim != 2 or cwm.shape[1] != 4:
                        continue
                    patterns.append(
                        PatternRef(
                            modisco_path=path,
                            metacluster=metacluster,
                            pattern=pattern_name,
                            task=task_name,
                            cwm=cwm,
                        )
                    )
    return patterns


def _max_similarity(query: np.ndarray, target: np.ndarray) -> float:
    if query.size == 0 or target.size == 0:
        return float("-inf")
    if query.shape[0] > target.shape[0]:
        query, target = target, query
    if query.shape[0] > target.shape[0]:
        return float("-inf")
    windows = sliding_window_view(target, window_shape=query.shape[0], axis=0)
    if windows.shape[0] == 0:
        return float("-inf")
    # Align window order to (window_len, 4) to match query layout.
    windows = np.swapaxes(windows, 1, 2)
    q = query.reshape(-1)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return float("-inf")
    w_flat = windows.reshape(windows.shape[0], -1)
    w_norm = np.linalg.norm(w_flat, axis=1)
    denom = q_norm * w_norm
    dots = w_flat @ q
    sims = np.where(denom > 0, dots / denom, float("-inf"))
    return float(np.max(sims))


def _best_pattern_match(motif_cwm: np.ndarray, patterns: list[PatternRef]) -> PatternRef | None:
    if not patterns:
        return None
    rc_cwm = _reverse_complement_matrix(motif_cwm)
    best_score = float("-inf")
    best_pattern: PatternRef | None = None
    for pattern in patterns:
        score = max(
            _max_similarity(motif_cwm, pattern.cwm),
            _max_similarity(rc_cwm, pattern.cwm),
        )
        if score > best_score:
            best_score = score
            best_pattern = pattern
    return best_pattern


def _parse_seqlet_record(raw: bytes | str) -> tuple[int, int, int, bool]:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode()
    parts = {}
    for item in raw.split(","):
        if not item:
            continue
        key, value = item.split(":", 1)
        parts[key.strip()] = value.strip()
    return (
        int(parts["example"]),
        int(parts["start"]),
        int(parts["end"]),
        parts.get("rc", "False").lower() == "true",
    )


def _seqlet_scores_for_pattern(
    contribs: np.ndarray,
    motif_cwm: np.ndarray,
    pattern: PatternRef,
    index_map: dict[int, int],
) -> np.ndarray:
    seq_len = contribs.shape[1]
    motif_len = motif_cwm.shape[0]
    rc_cwm = _reverse_complement_matrix(motif_cwm)
    seqlet_scores: list[float] = []
    base = "metacluster_idx_to_submetacluster_results"
    seqlets_path = (
        f"{base}/{pattern.metacluster}/seqlets_to_patterns_result/"
        f"patterns/{pattern.pattern}/seqlets_and_alnmts/seqlets"
    )
    with h5py.File(pattern.modisco_path, "r") as handle:
        if seqlets_path not in handle:
            return np.array([], dtype=np.float32)
        seqlets_ds = handle[seqlets_path]
        for raw in seqlets_ds:
            example_idx, start, end, _ = _parse_seqlet_record(raw)
            mapped_idx = index_map.get(example_idx)
            if mapped_idx is None:
                continue
            if start < 0 or end > seq_len:
                continue
            seqlet = contribs[mapped_idx, start:end, :]
            if seqlet.shape[0] < motif_len:
                continue
            windows = sliding_window_view(seqlet, window_shape=motif_len, axis=0)
            if windows.shape[0] == 0:
                continue
            fwd_scores = np.tensordot(windows, motif_cwm, axes=([1, 2], [1, 0]))
            rev_scores = np.tensordot(windows, rc_cwm, axes=([1, 2], [1, 0]))
            max_score = max(float(np.max(fwd_scores)), float(np.max(rev_scores)))
            if np.isfinite(max_score):
                seqlet_scores.append(max_score)
    return np.array(seqlet_scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan contribution weight matrices (CWMs) across positive regions "
            "and render score histograms."
        )
    )
    parser.add_argument("--deeplift-h5", required=True)
    parser.add_argument("--cwm-dir", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--summary-tsv", required=True)
    parser.add_argument("--indv-plots-dir", required=True)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--test-chroms", nargs="+", default=list(DEFAULT_TEST_CHROMS))
    args = parser.parse_args()

    raw = []
    for entry in args.test_chroms:
        raw.extend([part for part in entry.split(",") if part])
    test_chroms = {
        chrom if chrom.startswith("chr") else f"chr{chrom}"
        for chrom in raw
    }

    contribs, index_map = _load_attributions(Path(args.deeplift_h5), test_chroms)
    num_regions, seq_len, _ = contribs.shape

    cwm_dir = Path(args.cwm_dir)
    indv_dir = Path(args.indv_plots_dir)
    indv_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    html_rows = []

    modisco_paths = _discover_modisco_paths(Path(args.deeplift_h5))
    pattern_refs = _collect_patterns(modisco_paths)

    motif_dirs = sorted([p for p in cwm_dir.iterdir() if p.is_dir()])
    for motif_dir in motif_dirs:
        cwm_path = motif_dir / "CWM.txt"
        if not cwm_path.exists():
            continue
        motif_name = motif_dir.name
        try:
            cwm = np.loadtxt(cwm_path, dtype=np.float32)
        except ValueError:
            continue
        if cwm.ndim == 1:
            cwm = cwm.reshape(1, -1)
        if cwm.shape[1] != 4:
            continue

        kernel = int(cwm.shape[0])
        scan_len = seq_len - kernel + 1

        if scan_len <= 0:
            scores = np.array([], dtype=np.float32)
        else:
            forward_scores = _scan_cwm(contribs, cwm, args.chunk_size)
            rc_cwm = _reverse_complement_matrix(cwm)
            reverse_scores = _scan_cwm(contribs, rc_cwm, args.chunk_size)
            scores = np.concatenate(
                [forward_scores.ravel(), reverse_scores.ravel()], axis=0
            )
            scores = scores[np.isfinite(scores)]

        seqlet_scores = np.array([], dtype=np.float32)
        pattern_match = _best_pattern_match(cwm, pattern_refs)
        if pattern_match is not None:
            seqlet_scores = _seqlet_scores_for_pattern(
                contribs, cwm, pattern_match, index_map
            )

        png_path = indv_dir / f"{motif_name}_hist.png"
        html_path = indv_dir / f"{motif_name}_hist.html"
        title = (
            f"{motif_name} (kernel={kernel}, n={scores.size}, "
            f"seqlets={seqlet_scores.size})"
        )
        _plot_histogram(scores, seqlet_scores, png_path, title, args.bins)
        _write_interactive_histogram(scores, seqlet_scores, html_path, title, args.bins)

        summary_row = {
            "motif": motif_name,
            "kernel_size": kernel,
            "num_regions": num_regions,
            "scan_len": scan_len if scan_len > 0 else 0,
            "num_scores": int(scores.size),
            "score_min": float(np.min(scores)) if scores.size else float("nan"),
            "score_max": float(np.max(scores)) if scores.size else float("nan"),
            "score_mean": float(np.mean(scores)) if scores.size else float("nan"),
            "score_median": float(np.median(scores)) if scores.size else float("nan"),
            "score_std": float(np.std(scores)) if scores.size else float("nan"),
            "hist_png": str(png_path),
        }
        summary_rows.append(summary_row)

        html_rows.append(
            {
                "motif": motif_name,
                "kernel_size": kernel,
                "num_scores": int(scores.size),
                "png_path": png_path,
                "html_path": html_path,
            }
        )

    summary_path = Path(args.summary_tsv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_path.write_text(
            "motif\tkernel_size\tnum_regions\tscan_len\tnum_scores\tscore_min\t"
            "score_max\tscore_mean\tscore_median\tscore_std\thist_png\n"
        )

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>CWM score histograms</title>",
        "</head>",
        "<body>",
        "<h1>CWM score histograms (positive regions)</h1>",
        f"<p>Regions: {num_regions}. Window span: {seq_len}.</p>",
        "<p>Click a histogram to open the interactive view. "
        "Zooming recomputes bins within the visible range.</p>",
        "<table border=\"1\" cellspacing=\"0\" cellpadding=\"4\">",
        "<tr>",
        "<th>Motif</th>",
        "<th>Kernel size</th>",
        "<th>Scores</th>",
        "<th>Histogram (interactive link)</th>",
        "</tr>",
    ]

    for row in html_rows:
        img_rel = os.path.relpath(row["png_path"], start=output_html.parent)
        html_rel = os.path.relpath(row["html_path"], start=output_html.parent)
        html_lines.append(
            "<tr>"
            f"<td>{row['motif']}</td>"
            f"<td>{row['kernel_size']}</td>"
            f"<td>{row['num_scores']}</td>"
            f"<td><a href=\"{html_rel}\">"
            f"<img src=\"{img_rel}\" alt=\"{row['motif']}\" "
            "style=\"max-width: 900px; height: auto;\"></a>"
            f"<br><a href=\"{html_rel}\">interactive</a></td>"
            "</tr>"
        )

    html_lines.extend(["</table>", "</body>", "</html>"])
    output_html.write_text("\n".join(html_lines))


if __name__ == "__main__":
    main()
