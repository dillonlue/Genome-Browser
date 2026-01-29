#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pyBigWig

from bpnetlite.losses import MNLLLoss, log1pMSELoss


def _load_window_table(top_tsv: str, window_span: int) -> pd.DataFrame:
    top = pd.read_csv(top_tsv, sep="\t").sort_values("rank")
    half_span = window_span // 2
    window_starts = []
    window_ends = []
    for _, row in top.iterrows():
        start = int(row["start"])
        end = int(row["end"])
        mid = (start + end) // 2
        window_start = mid - half_span
        window_end = window_start + window_span
        window_starts.append(window_start)
        window_ends.append(window_end)
    top = top.copy()
    top["window_start"] = window_starts
    top["window_end"] = window_ends
    return top


def _load_motif_hits(bed_path: str):
    hits = {}
    with open(bed_path, "r") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            hits.setdefault(chrom, []).append((start, end))
    for chrom in hits:
        hits[chrom].sort()
    return hits


def _mask_for_window(hits, chrom, window_start, window_end, seq_len):
    mask = np.zeros(seq_len, dtype=bool)
    intervals = hits.get(chrom, [])
    motif_hits = 0
    for start, end in intervals:
        if end <= window_start:
            continue
        if start >= window_end:
            break
        rel_start = max(start, window_start) - window_start
        rel_end = min(end, window_end) - window_start
        mask[rel_start:rel_end] = True
        motif_hits += 1
    return mask, motif_hits


def _dinucleotide_shuffle_indices(base_indices, rng):
    seq_len = int(base_indices.shape[0])
    if seq_len <= 1:
        return base_indices.copy()
    next_indices = [[] for _ in range(4)]
    for pos in range(seq_len - 1):
        current = int(base_indices[pos])
        next_indices[current].append(pos + 1)
    for base in range(4):
        n_edges = len(next_indices[base])
        if n_edges > 1:
            order = np.arange(n_edges)
            order[:-1] = rng.permutation(n_edges - 1)
            next_indices[base] = [next_indices[base][i] for i in order]
    shuffled = np.empty_like(base_indices)
    idx = 0
    shuffled[0] = base_indices[idx]
    counters = [0] * 4
    for pos in range(1, seq_len):
        current = int(base_indices[idx])
        edge_idx = counters[current]
        idx = next_indices[current][edge_idx]
        counters[current] += 1
        shuffled[pos] = base_indices[idx]
    return shuffled


def _randomized_batch(base_indices, keep_mask, n_random, rng, shuffle_mode):
    batch = np.tile(base_indices[None, :], (n_random, 1))
    if shuffle_mode == "mono":
        scramble_positions = np.where(~keep_mask)[0]
        if scramble_positions.size > 0:
            original = base_indices[scramble_positions]
            for idx in range(n_random):
                batch[idx, scramble_positions] = rng.permutation(original)
    else:
        for idx in range(n_random):
            shuffled = _dinucleotide_shuffle_indices(base_indices, rng)
            shuffled[keep_mask] = base_indices[keep_mask]
            batch[idx] = shuffled
    batch = torch.from_numpy(batch)
    batch = F.one_hot(batch, num_classes=4).permute(0, 2, 1).float()
    return batch


def _read_chrom_sizes(path: str) -> List[Tuple[str, int]]:
    sizes = []
    with open(path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, size_str = line.rstrip("\n").split("\t")[:2]
            sizes.append((chrom, int(size_str)))
    return sizes


def _write_bigwig(output_path: str, entries, chrom_sizes) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bw = pyBigWig.open(str(output), "w")
    bw.addHeader(chrom_sizes)
    if entries:
        chrom_order = {chrom: idx for idx, (chrom, _) in enumerate(chrom_sizes)}
        entries.sort(key=lambda row: (chrom_order.get(row[0], 1_000_000), row[1]))
        chroms, starts, ends, values = zip(*entries)
        bw.addEntries(list(chroms), list(starts), ends=list(ends), values=list(values))
    bw.close()


def _collect_entries(
    top: pd.DataFrame, profiles: np.ndarray, output_span: int, chrom_sizes
):
    size_map = {chrom: size for chrom, size in chrom_sizes}
    aggregate = {}
    half_span = output_span // 2
    for idx, row in enumerate(top.itertuples(index=False)):
        chrom = str(row.chrom)
        chrom_size = size_map.get(chrom)
        if chrom_size is None:
            continue
        start = int(row.start)
        end = int(row.end)
        mid = (start + end) // 2
        window_start = mid - half_span
        profile = profiles[idx]
        for offset, value in enumerate(profile):
            pos = window_start + offset
            if pos < 0 or pos >= chrom_size:
                continue
            key = (chrom, pos)
            if key in aggregate:
                total, count = aggregate[key]
                aggregate[key] = (total + float(value), count + 1)
            else:
                aggregate[key] = (float(value), 1)
    entries = [
        (chrom, pos, pos + 1, total / count)
        for (chrom, pos), (total, count) in aggregate.items()
    ]
    return entries


def _profiles_from_logits(logits: torch.Tensor, logcounts: torch.Tensor) -> torch.Tensor:
    if logcounts.ndim == 1:
        logcounts = logcounts[:, None]
    batch, channels, length = logits.shape

    if logcounts.shape[1] == 1:
        flat = logits.reshape(batch, -1)
        log_probs = F.log_softmax(flat, dim=-1)
        probs = torch.exp(log_probs).reshape(batch, channels, length)
        counts = torch.exp(logcounts).reshape(batch, 1, 1)
        return probs * counts

    if logcounts.shape[1] * 2 == channels:
        predicted = torch.zeros_like(logits)
        for idx in range(logcounts.shape[1]):
            pair = logits[:, 2 * idx : 2 * idx + 2, :]
            flat = pair.reshape(batch, -1)
            log_probs = F.log_softmax(flat, dim=-1)
            probs = torch.exp(log_probs).reshape(batch, 2, length)
            counts = torch.exp(logcounts[:, idx]).reshape(batch, 1, 1)
            predicted[:, 2 * idx : 2 * idx + 2, :] = probs * counts
        return predicted

    if logcounts.shape[1] == channels:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        counts = torch.exp(logcounts).reshape(batch, channels, 1)
        return probs * counts

    flat = logits.reshape(batch, -1)
    log_probs = F.log_softmax(flat, dim=-1)
    probs = torch.exp(log_probs).reshape(batch, channels, length)
    counts = torch.exp(logcounts[:, :1]).reshape(batch, 1, 1)
    return probs * counts


def _strand_indices(num_channels: int):
    plus = list(range(0, num_channels, 2))
    minus = list(range(1, num_channels, 2))
    return plus, minus


def _profile_loss_for_indices(logits, y, indices):
    if not indices:
        return torch.tensor(0.0, device=logits.device)
    batch = logits.shape[0]
    subset_logits = logits[:, indices, :]
    if y.ndim == 2:
        subset_y = y[indices, :]
    else:
        subset_y = y[:, indices, :]
    log_probs = F.log_softmax(subset_logits.reshape(batch, -1), dim=-1)
    y_flat = subset_y.reshape(1, -1).expand(batch, -1)
    return MNLLLoss(log_probs, y_flat)


def _count_targets(y, logcounts):
    if y.ndim == 3:
        y = y[0]
    counts_per_channel = y.sum(dim=-1)
    if logcounts.ndim == 1:
        logcounts = logcounts[:, None]
    if logcounts.shape[1] == 1:
        total = counts_per_channel.sum()
        return total.reshape(1, 1)
    if logcounts.shape[1] == counts_per_channel.shape[0]:
        return counts_per_channel.reshape(1, -1)
    if logcounts.shape[1] * 2 == counts_per_channel.shape[0]:
        grouped = counts_per_channel.view(-1, 2).sum(dim=1)
        return grouped.reshape(1, -1)
    total = counts_per_channel.sum()
    return total.reshape(1, 1)


def _evaluate_loss(
    model,
    base_indices,
    keep_mask,
    control,
    y,
    count_loss_weight,
    device,
    n_random,
    rng,
    shuffle_mode,
):
    with torch.no_grad():
        X = _randomized_batch(base_indices, keep_mask, n_random, rng, shuffle_mode).to(
            device
        )
        if control is not None:
            X_ctl_batch = control.expand(n_random, -1, -1)
            logits, logcounts = model(X, X_ctl_batch)
        else:
            logits, logcounts = model(X)

        batch = logits.shape[0]
        log_probs = F.log_softmax(logits.reshape(batch, -1), dim=-1)
        y_flat = y.reshape(1, -1).expand(batch, -1)
        profile_loss = MNLLLoss(log_probs, y_flat)

        plus_idx, minus_idx = _strand_indices(logits.shape[1])
        profile_loss_plus = _profile_loss_for_indices(logits, y, plus_idx)
        profile_loss_minus = _profile_loss_for_indices(logits, y, minus_idx)

        count_targets = _count_targets(y, logcounts)
        count_batch = count_targets.expand(batch, -1)
        count_loss = log1pMSELoss(logcounts, count_batch)

        total_loss = profile_loss + count_loss_weight * count_loss

    profile_losses = profile_loss.cpu().numpy()
    profile_losses_plus = profile_loss_plus.cpu().numpy()
    profile_losses_minus = profile_loss_minus.cpu().numpy()
    count_losses = count_loss.cpu().numpy()
    total_losses = total_loss.cpu().numpy()

    return {
        "profile_loss_mean": float(profile_losses.mean()),
        "profile_loss_plus_mean": float(profile_losses_plus.mean()),
        "profile_loss_minus_mean": float(profile_losses_minus.mean()),
        "count_loss_mean": float(count_losses.mean()),
        "total_loss_mean": float(total_losses.mean()),
        "total_loss_std": float(total_losses.std(ddof=1)) if n_random > 1 else 0.0,
    }


def _predict_profiles(
    model,
    base_indices,
    keep_mask,
    control,
    device,
    n_random,
    rng,
    shuffle_mode,
):
    with torch.no_grad():
        X = _randomized_batch(base_indices, keep_mask, n_random, rng, shuffle_mode).to(
            device
        )
        if control is not None:
            X_ctl_batch = control.expand(n_random, -1, -1)
            logits, logcounts = model(X, X_ctl_batch)
        else:
            logits, logcounts = model(X)

        predicted = _profiles_from_logits(logits, logcounts)

    return predicted.mean(dim=0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute motif-only loss by scrambling non-motif positions."
    )
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--top-tsv", required=True)
    parser.add_argument("--motif-bed", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--count-loss-weight", type=float, required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--output-bws", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--shuffle-mode", choices=["mono", "dinuc"], default="mono")
    parser.add_argument("--window-span", type=int, default=2114)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.input_npz, allow_pickle=True)
    sequences = data["sequences"]
    controls = data["controls"]
    signals = data["signals"]
    chroms = data["chroms"]
    starts = data["starts"]
    ends = data["ends"]

    top = _load_window_table(args.top_tsv, args.window_span)
    motif_hits = _load_motif_hits(args.motif_bed)

    device = torch.device(args.device)
    model = torch.load(args.model, weights_only=False, map_location=device)
    model.eval()

    all_rows = []
    motif_preds: List[np.ndarray] = []
    for idx in tqdm(range(sequences.shape[0]), desc="Sequences"):
        chrom = str(chroms[idx])
        window_start = int(top.iloc[idx]["window_start"])
        window_end = int(top.iloc[idx]["window_end"])
        keep_mask, motif_hit_count = _mask_for_window(
            motif_hits,
            chrom,
            window_start,
            window_end,
            sequences[idx].shape[-1],
        )
        kept_bases = int(keep_mask.sum())
        kept_fraction = kept_bases / sequences[idx].shape[-1]

        base_indices = sequences[idx].argmax(axis=0)
        y = torch.from_numpy(signals[idx]).to(device)
        if controls.size > 0:
            control = torch.from_numpy(controls[idx]).to(device).unsqueeze(0)
        else:
            control = None

        baseline = _evaluate_loss(
            model,
            base_indices,
            np.ones_like(keep_mask, dtype=bool),
            control,
            y,
            args.count_loss_weight,
            device,
            1,
            rng,
            args.shuffle_mode,
        )

        motif_only = _evaluate_loss(
            model,
            base_indices,
            keep_mask,
            control,
            y,
            args.count_loss_weight,
            device,
            args.n_random,
            rng,
            args.shuffle_mode,
        )

        motif_only_pred = _predict_profiles(
            model,
            base_indices,
            keep_mask,
            control,
            device,
            args.n_random,
            rng,
            args.shuffle_mode,
        )
        motif_preds.append(motif_only_pred)

        all_rows.append(
            {
                "sequence_rank": idx + 1,
                "chrom": chrom,
                "start": int(starts[idx]),
                "end": int(ends[idx]),
                "window_start": window_start,
                "window_end": window_end,
                "baseline_profile_loss": baseline["profile_loss_mean"],
                "baseline_profile_loss_plus": baseline["profile_loss_plus_mean"],
                "baseline_profile_loss_minus": baseline["profile_loss_minus_mean"],
                "baseline_count_loss": baseline["count_loss_mean"],
                "baseline_total_loss": baseline["total_loss_mean"],
                "motif_profile_loss": motif_only["profile_loss_mean"],
                "motif_profile_loss_plus": motif_only["profile_loss_plus_mean"],
                "motif_profile_loss_minus": motif_only["profile_loss_minus_mean"],
                "motif_count_loss": motif_only["count_loss_mean"],
                "motif_total_loss": motif_only["total_loss_mean"],
                "motif_total_loss_std": motif_only["total_loss_std"],
                "motif_hit_count": motif_hit_count,
                "motif_bases_kept": kept_bases,
                "kept_fraction": kept_fraction,
            }
        )

    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_path, sep="\t", index=False)

    motif_array = np.stack(motif_preds, axis=0)
    output_span = int(motif_array.shape[2])
    chrom_sizes = _read_chrom_sizes(args.chrom_sizes)

    output_bws = list(args.output_bws)
    if len(output_bws) != motif_array.shape[1]:
        raise ValueError(
            f"Expected {motif_array.shape[1]} output bigWigs, got {len(output_bws)}."
        )

    for idx, output_bw in enumerate(output_bws):
        entries = _collect_entries(
            top, motif_array[:, idx, :], output_span, chrom_sizes
        )
        _write_bigwig(output_bw, entries, chrom_sizes)


if __name__ == "__main__":
    main()
