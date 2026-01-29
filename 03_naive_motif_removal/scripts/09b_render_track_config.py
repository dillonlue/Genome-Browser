#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict


def _build_mapping(motif_hits: str, names, bws) -> Dict[str, str]:
    mapping = {"__MOTIF_HITS__": motif_hits}
    for name, path in zip(names, bws):
        mapping[f"__MOTIF_ONLY_{name}__"] = path
    return mapping


def _replace_placeholders(payload, mapping: Dict[str, str]) -> None:
    for track in payload.get("tracks", []):
        if "file_location" in track:
            value = track["file_location"]
            if value in mapping:
                track["file_location"] = mapping[value]
        if "bed_path" in track:
            value = track["bed_path"]
            if value in mapping:
                track["bed_path"] = mapping[value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-experiment genome browser track config."
    )
    parser.add_argument("--template-json", required=True)
    parser.add_argument("--motif-hits-bed", required=True)
    parser.add_argument("--motif-only-names", nargs="+", required=True)
    parser.add_argument("--motif-only-bws", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if len(args.motif_only_names) != len(args.motif_only_bws):
        raise ValueError("motif-only names and bigWig list sizes differ")

    template_path = Path(args.template_json)
    payload = json.loads(template_path.read_text())

    mapping = _build_mapping(args.motif_hits_bed, args.motif_only_names, args.motif_only_bws)
    _replace_placeholders(payload, mapping)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
