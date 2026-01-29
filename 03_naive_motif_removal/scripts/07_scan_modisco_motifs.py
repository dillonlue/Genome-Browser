#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

import scan_motif_hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrapper for scanning MoDISco motifs in top windows."
    )
    parser.add_argument("--regions-bed", required=True)
    parser.add_argument("--genome-fasta", required=True)
    parser.add_argument("--motif-meme", required=True)
    parser.add_argument("--motif-id", required=True)
    parser.add_argument("--output-bed", required=True)
    parser.add_argument("--label", default="motif_hit")
    parser.add_argument("--min-log-odds", type=float, default=None)
    args = parser.parse_args()

    scan_motif_hits.write_hits_bed(
        Path(args.regions_bed),
        Path(args.genome_fasta),
        Path(args.motif_meme),
        args.motif_id,
        Path(args.output_bed),
        label=args.label,
        min_log_odds=args.min_log_odds,
    )


if __name__ == "__main__":
    main()
