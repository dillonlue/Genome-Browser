#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="/usr/local/bin/python"
TARGET_SCRIPT="${REPO_ROOT}/07_simplified_tracker/simplified_tracker.py"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" -m pdb "${TARGET_SCRIPT}" \
  --region-config "05_genome_browser_BPNet/shared/intermediate/03_top_regions.json" \
  --track-config "05_genome_browser_BPNet/configs/01_bpnet_paper_tracks.json" \
  --output-dir "07_simplified_tracker/output" \
  --window-span "1000" \
  --output-span "1000" \
  --title "Simplified genome tracker (top regions)"
