#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pyBigWig
import pyfaidx

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
INDEX_TO_BASE = ("A", "C", "G", "T")
DEFAULT_BACKGROUND = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}


def _parse_background(lines: List[str]) -> Dict[str, float]:
    for idx, line in enumerate(lines):
        if not line.startswith("Background letter frequencies"):
            continue
        for next_line in lines[idx + 1 :]:
            if not next_line.strip():
                continue
            tokens = next_line.split()
            background: Dict[str, float] = {}
            for pos in range(0, len(tokens) - 1, 2):
                base = tokens[pos].upper()
                try:
                    value = float(tokens[pos + 1])
                except ValueError:
                    continue
                if base in DEFAULT_BACKGROUND:
                    background[base] = value
            if background:
                return background
        break
    return DEFAULT_BACKGROUND.copy()


def _parse_motif_matrix(lines: List[str], motif_id: str) -> List[List[float]]:
    in_motif = False
    in_matrix = False
    matrix: List[List[float]] = []
    for line in lines:
        if line.startswith("MOTIF "):
            parts = line.split()
            current_id = parts[1] if len(parts) > 1 else ""
            in_motif = current_id == motif_id
            in_matrix = False
            continue
        if in_motif and "letter-probability matrix" in line:
            in_matrix = True
            continue
        if in_matrix:
            if not line.strip():
                break
            tokens = line.strip().split()
            try:
                row = [float(tok) for tok in tokens[:4]]
            except ValueError:
                break
            if len(row) < 4:
                break
            matrix.append(row)
    if not matrix:
        raise ValueError(f"Motif {motif_id} not found in MEME file.")
    return matrix


def load_motif_matrix(meme_path: Path, motif_id: str) -> Tuple[List[List[float]], Dict[str, float]]:
    lines = meme_path.read_text().splitlines()
    background = _parse_background(lines)
    matrix = _parse_motif_matrix(lines, motif_id)
    return matrix, background


def build_log_matrix(
    motif_matrix: List[List[float]],
    background: Dict[str, float],
    pseudocount: float = 1e-6,
) -> Tuple[List[List[float]], float, float]:
    log_matrix: List[List[float]] = []
    min_score = 0.0
    max_score = 0.0
    for row in motif_matrix:
        log_row = []
        row_min = None
        row_max = None
        for idx, prob in enumerate(row[:4]):
            base = INDEX_TO_BASE[idx]
            bg = background.get(base, DEFAULT_BACKGROUND[base])
            value = max(prob, pseudocount) / bg
            score = float(math.log2(value))
            log_row.append(score)
            row_min = score if row_min is None else min(row_min, score)
            row_max = score if row_max is None else max(row_max, score)
        log_matrix.append(log_row)
        if row_min is not None:
            min_score += row_min
        if row_max is not None:
            max_score += row_max
    return log_matrix, min_score, max_score


def reverse_complement_matrix(log_matrix: List[List[float]]) -> List[List[float]]:
    complement = {0: 3, 1: 2, 2: 1, 3: 0}
    reversed_rows = []
    for row in reversed(log_matrix):
        reversed_rows.append([row[complement[idx]] for idx in range(4)])
    return reversed_rows


def _read_bed(path: Path) -> Iterable[Tuple[str, int, int, str]]:
    with path.open("r") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3] if len(parts) > 3 else f"region_{idx}"
            yield chrom, start, end, name


def _read_chrom_sizes(path: Path) -> List[Tuple[str, int]]:
    sizes = []
    with path.open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            chrom, size_str = line.rstrip("\n").split("\t")[:2]
            sizes.append((chrom, int(size_str)))
    return sizes


def _score_window(
    seq: str, offset: int, log_matrix: List[List[float]]
) -> Optional[float]:
    width = len(log_matrix)
    score = 0.0
    for pos, base in enumerate(seq[offset : offset + width]):
        idx = BASE_TO_INDEX.get(base)
        if idx is None:
            return None
        score += log_matrix[pos][idx]
    return score


def _scan_log_odds(
    seq: str, log_matrix: List[List[float]], rev_matrix: List[List[float]]
) -> Tuple[List[float], float, int, str]:
    width = len(log_matrix)
    center_offset = width // 2
    seq_len = len(seq)
    scores = [0.0] * seq_len
    best_score = float("nan")
    best_offset = -1
    best_strand = "."
    scan_limit = seq_len - width + 1
    if scan_limit <= 0:
        return scores, best_score, best_offset, best_strand
    for offset in range(scan_limit):
        forward = _score_window(seq, offset, log_matrix)
        reverse = _score_window(seq, offset, rev_matrix)
        if forward is None and reverse is None:
            scores[offset] = 0.0
            continue
        if reverse is None or (forward is not None and forward >= reverse):
            score = float(forward)
            strand = "+"
        else:
            score = float(reverse)
            strand = "-"
        if score < 0:
            score = 0.0
        center_idx = offset + center_offset
        if 0 <= center_idx < seq_len:
            scores[center_idx] = score
        if best_offset < 0 or score > best_score:
            best_score = score
            best_offset = center_idx
            best_strand = strand
    return scores, best_score, best_offset, best_strand


def _collect_entries(
    regions: List[Tuple[str, int, int, str]],
    per_region_scores: List[List[float]],
    chrom_sizes: List[Tuple[str, int]],
) -> List[Tuple[str, int, int, float]]:
    size_map = {chrom: size for chrom, size in chrom_sizes}
    aggregate = {}
    for (chrom, start, end, _), scores in zip(regions, per_region_scores):
        chrom_size = size_map.get(chrom)
        if chrom_size is None:
            continue
        for offset, score in enumerate(scores):
            pos = start + offset
            if pos < 0 or pos >= chrom_size:
                continue
            key = (chrom, pos)
            if key in aggregate:
                total, count = aggregate[key]
                aggregate[key] = (total + score, count + 1)
            else:
                aggregate[key] = (score, 1)
    entries = [
        (chrom, pos, pos + 1, total / count)
        for (chrom, pos), (total, count) in aggregate.items()
    ]
    return entries


def _write_bigwig(
    output_path: Path, entries: List[Tuple[str, int, int, float]], chrom_sizes
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bw = pyBigWig.open(str(output_path), "w")
    bw.addHeader(chrom_sizes)
    if entries:
        chrom_order = {chrom: idx for idx, (chrom, _) in enumerate(chrom_sizes)}
        entries.sort(key=lambda row: (chrom_order.get(row[0], 1_000_000), row[1]))
        chroms, starts, ends, values = zip(*entries)
        bw.addEntries(list(chroms), list(starts), ends=list(ends), values=list(values))
    bw.close()


def _scan_hits_for_region(
    seq: str,
    log_matrix: List[List[float]],
    rev_matrix: List[List[float]],
    min_log_odds: Optional[float],
) -> Iterable[Tuple[int, float, str]]:
    width = len(log_matrix)
    seq_len = len(seq)
    scan_limit = seq_len - width + 1
    if scan_limit <= 0:
        return []
    hits: List[Tuple[int, float, str]] = []
    threshold = 0.0 if min_log_odds is None else min_log_odds
    for offset in range(scan_limit):
        forward = _score_window(seq, offset, log_matrix)
        reverse = _score_window(seq, offset, rev_matrix)
        if forward is None and reverse is None:
            continue
        if reverse is None or (forward is not None and forward >= reverse):
            score = float(forward)
            strand = "+"
        else:
            score = float(reverse)
            strand = "-"
        if score < threshold:
            continue
        hits.append((offset, score, strand))
    return hits


def write_hits_bed(
    regions_bed: Path,
    genome_fasta: Path,
    motif_meme: Path,
    motif_id: str,
    output_bed: Path,
    label: str = "motif_hit",
    min_log_odds: Optional[float] = None,
) -> None:
    regions = list(_read_bed(regions_bed))
    motif_matrix, background = load_motif_matrix(motif_meme, motif_id)
    log_matrix, _, _ = build_log_matrix(motif_matrix, background)
    rev_matrix = reverse_complement_matrix(log_matrix)
    width = len(log_matrix)
    fasta = pyfaidx.Fasta(str(genome_fasta))
    output_bed.parent.mkdir(parents=True, exist_ok=True)
    with output_bed.open("w") as handle:
        for chrom, start, end, _ in regions:
            seq = fasta[chrom][start:end].seq.upper()
            hits = _scan_hits_for_region(seq, log_matrix, rev_matrix, min_log_odds)
            for offset, score, strand in hits:
                hit_start = start + offset
                hit_end = hit_start + width
                handle.write(
                    f"{chrom}\t{hit_start}\t{hit_end}\t{label}\t{score:.6f}\t{strand}\n"
                )
    fasta.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan PWM motifs and report scores or hits."
    )
    parser.add_argument("--regions-bed", required=True)
    parser.add_argument("--genome-fasta", required=True)
    parser.add_argument("--motif-meme", required=True)
    parser.add_argument("--motif-id", required=True)
    parser.add_argument("--chrom-sizes")
    parser.add_argument("--output-tsv")
    parser.add_argument("--output-bw")
    parser.add_argument("--output-bed")
    parser.add_argument("--label", default="motif_hit")
    parser.add_argument("--min-log-odds", type=float, default=None)
    args = parser.parse_args()

    if args.output_bed:
        write_hits_bed(
            Path(args.regions_bed),
            Path(args.genome_fasta),
            Path(args.motif_meme),
            args.motif_id,
            Path(args.output_bed),
            label=args.label,
            min_log_odds=args.min_log_odds,
        )
        return

    missing = [
        name
        for name in ("chrom_sizes", "output_tsv", "output_bw")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Missing required arguments for score output: " + ", ".join(missing)
        )

    regions = list(_read_bed(Path(args.regions_bed)))
    chrom_sizes = _read_chrom_sizes(Path(args.chrom_sizes))

    motif_matrix, background = load_motif_matrix(Path(args.motif_meme), args.motif_id)
    log_matrix, _, _ = build_log_matrix(motif_matrix, background)
    rev_matrix = reverse_complement_matrix(log_matrix)

    fasta = pyfaidx.Fasta(args.genome_fasta)
    per_region_scores = []
    best_scores = []
    best_offsets = []
    best_strands = []
    for chrom, start, end, _ in regions:
        seq = fasta[chrom][start:end].seq.upper()
        scores, best_score, best_offset, best_strand = _scan_log_odds(
            seq, log_matrix, rev_matrix
        )
        per_region_scores.append(scores)
        best_scores.append(best_score)
        best_offsets.append(best_offset)
        best_strands.append(best_strand)
    fasta.close()

    output_tsv = Path(args.output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w") as handle:
        handle.write(
            "region_name\tchrom\tstart\tend\tmax_log_odds\tbest_center_offset\tbest_strand\n"
        )
        for (chrom, start, end, name), score, offset, strand in zip(
            regions, best_scores, best_offsets, best_strands
        ):
            handle.write(
                f"{name}\t{chrom}\t{start}\t{end}\t{score:.6f}\t{offset}\t{strand}\n"
            )

    entries = _collect_entries(regions, per_region_scores, chrom_sizes)
    _write_bigwig(Path(args.output_bw), entries, chrom_sizes)


if __name__ == "__main__":
    main()
