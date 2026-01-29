#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="/usr/local/bin/python"
TARGET_SCRIPT="${REPO_ROOT}/03_naive_motif_removal/scripts/09_ctcf_genome_browser.py"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" -m pdb "${TARGET_SCRIPT}" \
  --region-config "05_genome_browser_BPNet/shared/intermediate/30_paper_regions.json" \
  --track-config "05_genome_browser_BPNet/configs/30_bpnet_paper_bw_tracks.json" \
  --output-dir "05_genome_browser_BPNet/output/30_genome_browser" \
  --num-workers "1" \
  --title "BPNet paper genome browser (paper regions)" \
  --window-span "1000" \
  --output-span "1000" \
  --main-html "05_genome_browser_BPNet/output/30_main.html" \
  --indv-plots-dir "05_genome_browser_BPNet/output/30_indv_plots" \
  --summary-tsv "05_genome_browser_BPNet/output/30_summary_stats.tsv"
