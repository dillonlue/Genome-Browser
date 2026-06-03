#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from bpnetlite import performance
from bpnetlite.bpnet import _mixture_loss
from tangermeme.io import extract_loci


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / path).resolve()


def _load_evaluate_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def _extract_dataset(params: dict, base_dir: Path, chroms_override: Path | None):
    chroms = params.get("validation_chroms", params["chroms"])
    if chroms_override is not None:
        override_payload = json.loads(chroms_override.read_text())
        chroms = override_payload.get("chroms", chroms)

    loci_paths = [_resolve_path(path, base_dir) for path in params["loci"]]
    signals = [_resolve_path(path, base_dir) for path in params["signals"]]
    controls = [_resolve_path(path, base_dir) for path in params["controls"]]
    sequences = _resolve_path(params["sequences"], base_dir)

    X, y, X_ctl, _ = extract_loci(
        sequences=str(sequences),
        signals=[str(path) for path in signals],
        in_signals=[str(path) for path in controls],
        loci=[str(path) for path in loci_paths],
        chroms=chroms,
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=0,
        exclusion_lists=params["exclusion_lists"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        return_mask=True,
        verbose=True,
    )
    return X, y, X_ctl


def _evaluate_model(
    model_path: Path,
    X: torch.Tensor,
    y: torch.Tensor,
    X_ctl: torch.Tensor,
    count_loss_weight: float,
    device: torch.device,
    batch_size: int,
    kernel_sigma: int,
    kernel_width: int,
    smooth_true: bool,
    smooth_predictions: bool,
):
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    profile_sums = {
        "profile_mnll": 0.0,
        "profile_jsd": 0.0,
        "profile_pearson": 0.0,
        "profile_spearman": 0.0,
    }
    profile_count = 0

    pred_logcounts_all = []
    true_logcounts_all = []

    profile_loss_sum = 0.0
    count_loss_sum = 0.0
    total_loss_sum = 0.0

    with torch.no_grad():
        for offset in range(0, X.shape[0], batch_size):
            batch_seq = X[offset : offset + batch_size].float().to(device)
            batch_true = y[offset : offset + batch_size].to(device)
            if X_ctl.numel() > 0:
                batch_ctl = X_ctl[offset : offset + batch_size].float().to(device)
                logits, logcounts = model(batch_seq, batch_ctl)
            else:
                logits, logcounts = model(batch_seq)

            measures = performance.calculate_performance_measures(
                logits,
                batch_true,
                logcounts,
                kernel_sigma=kernel_sigma,
                kernel_width=kernel_width,
                smooth_true=smooth_true,
                smooth_predictions=smooth_predictions,
                measures=list(profile_sums.keys()),
            )

            batch_size_actual = batch_true.shape[0]
            for key in profile_sums:
                profile_sums[key] += measures[key].sum().item()
            profile_count += batch_size_actual

            profile_loss, count_loss, total_loss = _mixture_loss(
                batch_true, logits, logcounts, count_loss_weight
            )
            profile_loss_sum += profile_loss.item() * batch_size_actual
            count_loss_sum += count_loss.item() * batch_size_actual
            total_loss_sum += total_loss.item() * batch_size_actual

            true_logcounts = torch.log(batch_true.sum(dim=(-1, -2)) + 1).cpu()
            pred_logcounts_all.append(logcounts.cpu())
            true_logcounts_all.append(true_logcounts)

    pred_logcounts = torch.cat(pred_logcounts_all, dim=0)
    true_logcounts = torch.cat(true_logcounts_all, dim=0)

    count_pearson = performance.pearson_corr(pred_logcounts.T, true_logcounts.T)
    count_spearman = performance.spearman_corr(pred_logcounts.T, true_logcounts.T)
    count_mse = performance.mean_squared_error(pred_logcounts.T, true_logcounts.T)

    metrics = {
        "profile_mnll": profile_sums["profile_mnll"] / profile_count,
        "profile_jsd": profile_sums["profile_jsd"] / profile_count,
        "profile_pearson": profile_sums["profile_pearson"] / profile_count,
        "profile_spearman": profile_sums["profile_spearman"] / profile_count,
        "count_pearson": count_pearson.item(),
        "count_spearman": count_spearman.item(),
        "count_mse": count_mse.item(),
        "profile_loss": profile_loss_sum / profile_count,
        "count_loss": count_loss_sum / profile_count,
        "total_loss": total_loss_sum / profile_count,
    }
    return metrics


def _collect_plot_data(
    model_path: Path,
    X: torch.Tensor,
    y: torch.Tensor,
    X_ctl: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_residual_samples: int,
) -> Dict[str, object]:
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    profile_corr: List[float] = []
    count_true: List[float] = []
    count_pred: List[float] = []
    residual_samples: List[float] = []

    with torch.no_grad():
        for offset in range(0, X.shape[0], batch_size):
            batch_seq = X[offset : offset + batch_size].float().to(device)
            batch_true = y[offset : offset + batch_size].to(device)
            if X_ctl.numel() > 0:
                batch_ctl = X_ctl[offset : offset + batch_size].float().to(device)
                logits, logcounts = model(batch_seq, batch_ctl)
            else:
                logits, logcounts = model(batch_seq)

            # Expected shapes: logits (B, T, L, S), logcounts (B, T)
            # Convert to predicted counts per base by applying softmax across length.
            probs = torch.softmax(logits, dim=2)
            total_counts = torch.exp(logcounts) - 1.0
            if total_counts.dim() == 3 and total_counts.shape[-1] == 1:
                total_counts = total_counts.squeeze(-1)
            total_counts = total_counts[..., None, None]
            pred_profile = probs * total_counts

            true_profile = batch_true
            if true_profile.dim() != pred_profile.dim():
                raise ValueError("Unexpected shape mismatch between true and predicted profiles.")

            # Sum over strands for per-region comparisons.
            true_sum = true_profile.sum(dim=-1)
            pred_sum = pred_profile.sum(dim=-1)

            # Per-region profile Pearson correlations.
            for i in range(true_sum.shape[0]):
                t = true_sum[i].flatten().cpu().numpy()
                p = pred_sum[i].flatten().cpu().numpy()
                if t.size < 2 or p.size < 2:
                    continue
                if t.std() == 0 or p.std() == 0:
                    continue
                profile_corr.append(float(np.corrcoef(t, p)[0, 1]))

            # Counts per region (sum over length + strands).
            count_true.extend(true_profile.sum(dim=(-1, -2)).cpu().numpy().tolist())
            count_pred.extend(pred_profile.sum(dim=(-1, -2)).cpu().numpy().tolist())

            # Pearson residuals per bin (cap for plotting).
            obs = true_sum.flatten().cpu().numpy()
            exp = pred_sum.flatten().cpu().numpy()
            exp = exp.clip(min=0.0)
            eps = 1e-3
            resid = (obs - exp) / (exp + eps) ** 0.5
            if len(residual_samples) < max_residual_samples:
                remaining = max_residual_samples - len(residual_samples)
                if resid.size > remaining:
                    idx = torch.randperm(resid.size)[:remaining].numpy()
                    resid = resid[idx]
                residual_samples.extend(resid.tolist())

    return {
        "profile_corr": profile_corr,
        "count_true": count_true,
        "count_pred": count_pred,
        "residual_samples": residual_samples,
    }


def _write_html(
    output_path: Path,
    label_a: str,
    label_b: str,
    data_a: Dict[str, object],
    data_b: Dict[str, object],
    title: str,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Per-region profile Pearson correlation",
            "Counts: predicted vs true (per region)",
            "Pearson residuals distribution (profile bins)",
            "Per-model count Pearson correlation",
        ),
        specs=[[{"type": "box"}, {"type": "scatter"}], [{"type": "histogram"}, {"type": "bar"}]],
    )

    fig.add_trace(go.Box(y=data_a["profile_corr"], name=label_a, boxmean=True), row=1, col=1)
    fig.add_trace(go.Box(y=data_b["profile_corr"], name=label_b, boxmean=True), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=data_a["count_true"],
            y=data_a["count_pred"],
            mode="markers",
            name=label_a,
            marker=dict(size=4, opacity=0.6),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=data_b["count_true"],
            y=data_b["count_pred"],
            mode="markers",
            name=label_b,
            marker=dict(size=4, opacity=0.6),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Histogram(x=data_a["residual_samples"], name=f"{label_a} residuals", nbinsx=80),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=data_b["residual_samples"], name=f"{label_b} residuals", nbinsx=80),
        row=2,
        col=1,
    )

    def _corr(x, y):
        if len(x) < 2 or len(y) < 2:
            return float("nan")
        x = torch.tensor(x)
        y = torch.tensor(y)
        if x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(torch.corrcoef(torch.stack([x, y]))[0, 1])

    fig.add_trace(
        go.Bar(
            x=[label_a, label_b],
            y=[_corr(data_a["count_true"], data_a["count_pred"]), _corr(data_b["count_true"], data_b["count_pred"])],
            name="count pearson",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=900, width=1200, title_text=title)
    fig.update_xaxes(title_text="True counts", row=1, col=2)
    fig.update_yaxes(title_text="Predicted counts", row=1, col=2)
    fig.update_xaxes(title_text="Pearson residual", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Pearson r", row=2, col=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def _format_results(rows):
    headers = [
        "model",
        "profile_mnll",
        "profile_jsd",
        "profile_pearson",
        "profile_spearman",
        "count_pearson",
        "count_spearman",
        "count_mse",
        "profile_loss",
        "count_loss",
        "total_loss",
    ]
    lines = ["\t".join(headers)]
    for name, metrics in rows:
        values = [name] + [metrics[key] for key in headers[1:]]
        line = "\t".join(
            [values[0]] + [f"{value:.15g}" for value in values[1:]]
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BPNet-style models on the same test set."
    )
    parser.add_argument("--evaluate-json", required=True)
    parser.add_argument("--chroms-json")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--label-a", default="model_a")
    parser.add_argument("--label-b", default="model_b")
    parser.add_argument("--count-loss-weight", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kernel-sigma", type=int, default=7)
    parser.add_argument("--kernel-width", type=int, default=81)
    parser.add_argument("--smooth-true", action="store_true")
    parser.add_argument("--smooth-predictions", action="store_true")
    parser.add_argument("--output-tsv")
    parser.add_argument("--output-html")
    parser.add_argument("--html-title", default="BPNet comparison: Pearson correlations and residuals")
    parser.add_argument("--max-residual-samples", type=int, default=200000)
    args = parser.parse_args()

    evaluate_path = Path(args.evaluate_json)
    params = _load_evaluate_json(evaluate_path)
    base_dir = evaluate_path.parent.resolve()

    count_loss_weight = (
        args.count_loss_weight
        if args.count_loss_weight is not None
        else float(params.get("count_loss_weight", 1.0))
    )

    X, y, X_ctl = _extract_dataset(
        params, base_dir, Path(args.chroms_json) if args.chroms_json else None
    )

    device = torch.device(args.device)
    model_a_path = _resolve_path(args.model_a, base_dir)
    model_b_path = _resolve_path(args.model_b, base_dir)

    rows = []
    rows.append(
        (
            args.label_a,
            _evaluate_model(
                model_a_path,
                X,
                y,
                X_ctl,
                count_loss_weight,
                device,
                args.batch_size,
                args.kernel_sigma,
                args.kernel_width,
                args.smooth_true,
                args.smooth_predictions,
            ),
        )
    )
    rows.append(
        (
            args.label_b,
            _evaluate_model(
                model_b_path,
                X,
                y,
                X_ctl,
                count_loss_weight,
                device,
                args.batch_size,
                args.kernel_sigma,
                args.kernel_width,
                args.smooth_true,
                args.smooth_predictions,
            ),
        )
    )

    output_text = _format_results(rows)
    if args.output_tsv:
        output_path = Path(args.output_tsv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text)
    else:
        print(output_text, end="")

    if args.output_html:
        data_a = _collect_plot_data(
            model_a_path,
            X,
            y,
            X_ctl,
            device,
            args.batch_size,
            args.max_residual_samples,
        )
        data_b = _collect_plot_data(
            model_b_path,
            X,
            y,
            X_ctl,
            device,
            args.batch_size,
            args.max_residual_samples,
        )
        _write_html(
            Path(args.output_html),
            args.label_a,
            args.label_b,
            data_a,
            data_b,
            args.html_title,
        )


if __name__ == "__main__":
    main()
