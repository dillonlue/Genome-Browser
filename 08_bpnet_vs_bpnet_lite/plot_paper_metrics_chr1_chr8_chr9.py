import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pyBigWig
from plotly.subplots import make_subplots

# Inputs
REGIONS_BED = Path(
    "01_download_data/raw_data/01_zenodo_3371216_data/data/merged_set/unique.training.bed.gz"
)
TRUE_BW_ROOT = Path("01_download_data/raw_data/01_zenodo_3371216_data/data/chip-nexus")
BPNET_PRED_BW_ROOT = Path("01_download_data/raw_data/02_zenodo_4294904_files")
DEFAULT_BPNET_LITE_PRED_BW_ROOT = Path("08_bpnet_vs_bpnet_lite/bpnet_lite_predictions")

CHROMS = {"chr1", "chr8", "chr9"}
MAX_REGIONS_PER_CHROM = 1000
TASKS = ["Oct4", "Sox2", "Nanog", "Klf4"]
EPS = 1e-3

TRUE_BWS = {
    task: {
        "pos": TRUE_BW_ROOT / task / "counts.pos.bw",
        "neg": TRUE_BW_ROOT / task / "counts.neg.bw",
    }
    for task in TASKS
}

BPNET_PRED_BWS = {
    task: {
        "pos": BPNET_PRED_BW_ROOT / f"{task}.preds.pos.bw",
        "neg": BPNET_PRED_BW_ROOT / f"{task}.preds.neg.bw",
    }
    for task in TASKS
}

OUTPUT_HTML = Path("08_bpnet_vs_bpnet_lite/observed_predicted_total_counts_spearman_chr1_chr8_chr9.html")
COLORS = {
    "Oct4": "#1f77b4",
    "Sox2": "#ff7f0e",
    "Nanog": "#2ca02c",
    "Klf4": "#d62728",
}


def safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    a = a[mask]
    b = b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    a = a[mask]
    b = b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    a_ranks = np.argsort(np.argsort(a)).astype(np.float64)
    b_ranks = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(a_ranks, b_ranks)[0, 1])


def load_regions():
    regions = []
    per_chrom_counts = {chrom: 0 for chrom in CHROMS}
    import gzip

    with gzip.open(REGIONS_BED, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, start, end, *_ = line.rstrip().split("\t")
            if chrom not in CHROMS or per_chrom_counts[chrom] >= MAX_REGIONS_PER_CHROM:
                continue
            start = int(start)
            end = int(end)
            if end <= start:
                continue
            regions.append((chrom, start, end))
            per_chrom_counts[chrom] += 1
            if all(per_chrom_counts[c] >= MAX_REGIONS_PER_CHROM for c in CHROMS):
                break
    return regions, per_chrom_counts


def total_counts(handle_pos, handle_neg, chrom, start, end):
    pos = np.nan_to_num(handle_pos.values(chrom, start, end, numpy=True), nan=0.0)
    neg = np.nan_to_num(handle_neg.values(chrom, start, end, numpy=True), nan=0.0)
    return float(np.sum(pos) + np.sum(neg))


def build_pred_bws(root: Path):
    return {
        task: {
            "pos": root / f"{task}.preds.pos.bw",
            "neg": root / f"{task}.preds.neg.bw",
        }
        for task in TASKS
    }


def require_paths(model_name, bw_paths):
    missing = []
    for task in TASKS:
        for strand in ("pos", "neg"):
            path = bw_paths[task][strand]
            if not path.exists():
                missing.append(str(path))
    if missing:
        missing_text = "\n".join(missing)
        raise SystemExit(f"Missing {model_name} prediction BigWigs:\n{missing_text}")


def collect_counts(regions, pred_bws):
    bw_handles = {}
    for task in TASKS:
        bw_handles[(task, "true", "pos")] = pyBigWig.open(str(TRUE_BWS[task]["pos"]))
        bw_handles[(task, "true", "neg")] = pyBigWig.open(str(TRUE_BWS[task]["neg"]))
        bw_handles[(task, "pred", "pos")] = pyBigWig.open(str(pred_bws[task]["pos"]))
        bw_handles[(task, "pred", "neg")] = pyBigWig.open(str(pred_bws[task]["neg"]))

    count_true = {task: [] for task in TASKS}
    count_pred = {task: [] for task in TASKS}

    try:
        for chrom, start, end in regions:
            for task in TASKS:
                obs = total_counts(
                    bw_handles[(task, "true", "pos")],
                    bw_handles[(task, "true", "neg")],
                    chrom,
                    start,
                    end,
                )
                pred = total_counts(
                    bw_handles[(task, "pred", "pos")],
                    bw_handles[(task, "pred", "neg")],
                    chrom,
                    start,
                    end,
                )
                count_true[task].append(obs)
                count_pred[task].append(max(pred, 0.0))
    finally:
        for handle in bw_handles.values():
            handle.close()

    return count_true, count_pred


def add_model_row(fig, row, model_name, count_true, count_pred):
    per_task_spearman = {
        task: safe_spearman(count_true[task], count_pred[task]) for task in TASKS
    }
    max_count = max(
        max(max(values) for values in count_true.values()),
        max(max(values) for values in count_pred.values()),
    )

    for col, task in enumerate(TASKS, start=1):
        x = np.asarray(count_pred[task], dtype=np.float64) + EPS
        y = np.asarray(count_true[task], dtype=np.float64) + EPS
        spearman = per_task_spearman[task]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=task,
                legendgroup=task,
                showlegend=row == 1,
                marker=dict(size=5, opacity=0.55, color=COLORS[task]),
                hovertemplate=(
                    f"{model_name} | {task}<br>"
                    "Predicted: %{x:.2f}<br>"
                    "Observed: %{y:.2f}<br>"
                    f"Spearman r: {spearman:.3f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=[EPS, max_count + EPS],
                y=[EPS, max_count + EPS],
                mode="lines",
                line=dict(color="#666666", dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        fig.add_annotation(
            x=0.99,
            y=0.03,
            text=f"rho={spearman:.3f}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color=COLORS[task], size=12),
            bgcolor="rgba(255,255,255,0.75)",
            row=row,
            col=col,
        )

        fig.update_xaxes(
            title_text="Predicted total read counts",
            type="log",
            exponentformat="power",
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="Observed total read counts",
            type="log",
            exponentformat="power",
            row=row,
            col=col,
        )

    return per_task_spearman


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot observed versus predicted total counts for BPNet and BPNet-lite on chr1/8/9."
    )
    parser.add_argument("--output-html", default=str(OUTPUT_HTML))
    parser.add_argument(
        "--bpnet-only",
        action="store_true",
        help="Only plot the BPNet row and skip BPNet-lite inputs.",
    )
    parser.add_argument(
        "--bpnet-lite-pred-root",
        default=str(DEFAULT_BPNET_LITE_PRED_BW_ROOT),
        help="Directory containing BPNet-lite prediction BigWigs named <TF>.preds.pos.bw and <TF>.preds.neg.bw.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    regions, per_chrom_counts = load_regions()
    if not regions:
        raise SystemExit("No regions found for chr1/chr8/chr9")

    require_paths("BPNet", BPNET_PRED_BWS)
    bpnet_true, bpnet_pred = collect_counts(regions, BPNET_PRED_BWS)
    if args.bpnet_only:
        fig = make_subplots(
            rows=1,
            cols=len(TASKS),
            subplot_titles=tuple(f"BPNet {task}" for task in TASKS),
            specs=[[{"type": "scatter"} for _ in TASKS]],
            horizontal_spacing=0.05,
        )
        bpnet_spearman = add_model_row(fig, 1, "BPNet", bpnet_true, bpnet_pred)
        bpnet_lite_spearman = None
        fig.update_layout(
            height=600,
            width=1800,
            title_text="BPNet observed and predicted total read counts with Spearman correlation across chr1, chr8, and chr9",
            legend_title_text="Transcription factor",
        )
    else:
        bpnet_lite_pred_bws = build_pred_bws(Path(args.bpnet_lite_pred_root))
        require_paths("BPNet-lite", bpnet_lite_pred_bws)
        bpnet_lite_true, bpnet_lite_pred = collect_counts(regions, bpnet_lite_pred_bws)

        fig = make_subplots(
            rows=2,
            cols=len(TASKS),
            subplot_titles=tuple(
                f"{model} {task}"
                for model in ("BPNet", "BPNet-lite")
                for task in TASKS
            ),
            specs=[[{"type": "scatter"} for _ in TASKS] for _ in range(2)],
            horizontal_spacing=0.05,
            vertical_spacing=0.12,
        )

        bpnet_spearman = add_model_row(fig, 1, "BPNet", bpnet_true, bpnet_pred)
        bpnet_lite_spearman = add_model_row(fig, 2, "BPNet-lite", bpnet_lite_true, bpnet_lite_pred)

        fig.update_layout(
            height=1100,
            width=1800,
            title_text="Observed and predicted total read counts with Spearman correlation across chr1, chr8, and chr9",
            legend_title_text="Transcription factor",
        )

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")

    print("Wrote", output_html)
    print("Regions used:", len(regions), "per chrom", per_chrom_counts)
    for task in TASKS:
        print(
            "BPNet",
            task,
            "spearman_r=",
            f"{bpnet_spearman[task]:.4f}",
        )
    if bpnet_lite_spearman is not None:
        for task in TASKS:
            print(
                "BPNet-lite",
                task,
                "spearman_r=",
                f"{bpnet_lite_spearman[task]:.4f}",
            )


if __name__ == "__main__":
    main()
