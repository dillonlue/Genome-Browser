#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyBigWig
import torch
import torch.nn.functional as F


def _read_chrom_sizes(path: Path) -> List[Tuple[str, int]]:
    sizes = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, size_str = line.rstrip("\n").split("\t")[:2]
            sizes.append((chrom, int(size_str)))
    return sizes


def _write_bigwig(output_path: Path, entries, chrom_sizes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bw = pyBigWig.open(str(output_path), "w")
    bw.addHeader(chrom_sizes)
    if entries:
        chrom_order = {chrom: idx for idx, (chrom, _) in enumerate(chrom_sizes)}
        entries.sort(key=lambda row: (chrom_order.get(row[0], 1_000_000), row[1]))
        chroms, starts, ends, values = zip(*entries)
        bw.addEntries(list(chroms), list(starts), ends=list(ends), values=list(values))
    bw.close()


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


def _collect_entries(
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    profiles: np.ndarray,
    output_span: int,
    chrom_sizes,
):
    size_map = {chrom: size for chrom, size in chrom_sizes}
    aggregate = {}
    half_span = output_span // 2
    for idx in range(profiles.shape[0]):
        chrom = str(chroms[idx])
        chrom_size = size_map.get(chrom)
        if chrom_size is None:
            continue
        start = int(starts[idx])
        end = int(ends[idx])
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build predicted bigWigs for top windows from a BPNet model."
    )
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--output-bws", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    sequences = data["sequences"]
    controls = data["controls"]
    chroms = data["chroms"]
    starts = data["starts"]
    ends = data["ends"]

    device = torch.device(args.device)
    model = torch.load(args.model, weights_only=False, map_location=device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for offset in range(0, sequences.shape[0], args.batch_size):
            batch_seq = torch.from_numpy(sequences[offset : offset + args.batch_size]).float()
            batch_seq = batch_seq.to(device)
            if controls.size > 0:
                batch_ctl = torch.from_numpy(
                    controls[offset : offset + args.batch_size]
                ).float().to(device)
                logits, logcounts = model(batch_seq, batch_ctl)
            else:
                logits, logcounts = model(batch_seq)
            predicted = _profiles_from_logits(logits, logcounts)
            outputs.append(predicted.cpu().numpy())

    predicted = np.concatenate(outputs, axis=0)
    channel_count = predicted.shape[1]
    output_bws = [Path(path) for path in args.output_bws]
    if len(output_bws) != channel_count:
        raise ValueError(
            f"Expected {channel_count} output bigWigs, got {len(output_bws)}."
        )

    chrom_sizes = _read_chrom_sizes(Path(args.chrom_sizes))
    output_span = predicted.shape[2]

    for idx, output_path in enumerate(output_bws):
        entries = _collect_entries(
            chroms, starts, ends, predicted[:, idx, :], output_span, chrom_sizes
        )
        _write_bigwig(output_path, entries, chrom_sizes)


if __name__ == "__main__":
    main()
