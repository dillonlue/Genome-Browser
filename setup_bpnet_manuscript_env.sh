#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-bpnet-manuscript}"

echo "Creating conda environment: ${ENV_NAME}"
mamba create -n "${ENV_NAME}" python=3.7 -y

echo "Activating environment: ${ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Installing bioinformatics system dependencies"
mamba install -y -c conda-forge -c bioconda bedtools samtools htslib

echo "Installing TensorFlow 1 / Keras 2 stack"
pip install "tensorflow==1.15.5" "keras==2.2.4"

echo "Installing notebook and plotting dependencies"
pip install plotnine seaborn jupyter ipywidgets papermill

echo "Installing manuscript Python dependencies"
pip install pybedtools kipoi genomelake concise

echo "Installing bpnet-manuscript in editable mode"
pip install -e /repo/bpnet-manuscript

echo "Validating core imports"
python - <<'PY'
from basepair.imports import *
from basepair.cli.schemas import DataSpec, TaskSpec
from basepair.plot.evaluate import regression_eval
print("bpnet-manuscript imports OK")
PY

echo
echo "Environment setup completed."
echo "Remaining likely blockers are external manuscript data paths,"
echo "especially replicate bigWigs and calibrated model outputs."
