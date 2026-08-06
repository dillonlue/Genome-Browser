# Mask-Aware Shapley Attribution for Transcription Factor Binding

Research project on Mask-Aware Shapley attribution for BPNet, a deep learning model for transcription factor (TF) binding prediction. Developed under the mentorship of Dillon Lue, PhD researcher in Princeton University's Department of Computer Science.

BPNet is a convolutional neural network (CNN) that learns the relationship between DNA sequence and TF binding by predicting base-resolution chromatin immunoprecipitation (ChIP)-nexus binding profiles directly from genomic DNA. Current interpretability methods such as DeepLIFT and DeepSHAP use marginal and conditional distributions through baselines or background samples to substitute for absent features. These methods can however create often unrealistic DNA sequences and change bases that destroy the nature of motifs, making it impossible to separate between the importance of the motif's identity and its appearance in the DNA sequence.

This project proposes Mask-Aware Shapley attribution: a variant of BPNet trained with an explicit binary mask channel, so the model learns to predict conditioned on which positions are observed. The theoretical background — the Shapley game formulation, the masked-predictor construction, and the mask-aware training scheme — is written up in [`project_journal/main.tex`](project_journal/main.tex) (compiled PDF at [`project_journal/main.pdf`](project_journal/main.pdf)).

Before that model can be built and trusted, this repo first does the groundwork: reproducing BPNet from the original paper, reproducing the faster `bpnet-lite` reimplementation, checking that the two agree with each other and with the published results, and building tooling to visually and quantitatively inspect model predictions and contribution scores across genomic regions.

## Pipeline overview

The repo is organized as a sequence of numbered stages, each with its own `Snakefile` so every stage's inputs/outputs and dependencies are explicit and reproducible. The root [`Snakefile`](Snakefile) currently wires in stage `07`; each subdirectory's `Snakefile` can also be run independently.

| Stage | Purpose |
|---|---|
| [`01_download_data/`](01_download_data/) | Downloads and stages the source datasets (Zenodo ChIP-nexus data, ChromBPNet motif databases) used to train and evaluate the models, with checksum verification. |
| [`02_train/`](02_train/) | Trains the two models being compared: the original-paper BPNet architecture (`01_bpnet_paper_pipeline.json`) and `bpnet-lite` (`02_bpnet_lite_pipeline.json`), from matched hyperparameter configs. Includes scripts to replicate the paper's test-set metrics and to compare the two trained models (`compare_models_on_testset.py`, `replicate_test_metrics.py`). |
| [`03_naive_motif_removal/`](03_naive_motif_removal/) | An interpretability baseline: extracts TF-MoDISco motifs from contribution scores, scans for motif hits, and evaluates the effect of naively masking/shuffling motif instances (mono- and di-nucleotide shuffles, at several contribution/log-odds thresholds — see `configs/experiment_defs.tsv`) on model predictions and loss. |
| [`05_genome_browser_BPNet/`](05_genome_browser_BPNet/) | Applies the genome browser tool (below) to BPNet, building region/track configs that reproduce the figures from the original BPNet paper and rendering the paper's regions as well as held-out high-signal and randomly sampled regions. |
| [`07_simplified_tracker/`](07_simplified_tracker/) | A ~200-line, easy-to-audit reimplementation of the genome browser's core plotting logic (`simplest_tracker.py`), built to sanity-check the more complex tool in `genome_browser/` against a version simple enough to read end-to-end. |
| [`08_bpnet_vs_bpnet_lite/`](08_bpnet_vs_bpnet_lite/) | Head-to-head comparison of observed vs. predicted total read counts for both models across held-out chromosomes (chr1, chr8, chr9), reported via Pearson/Spearman correlation. |

## Genome browser visualization tool

[`genome_browser/main.py`](genome_browser/main.py) is a JSON-configuration-driven tool for visualizing and comparing signal tracks across BPNet-style models at specific genomic regions. Given a region config (chromosome/start/name windows) and a track config, it renders per-region tracks for:

- `bw` — bigWig observed/predicted signal tracks
- `keras_contribution` / `bpnet_lite_contribution` — per-base contribution (attribution) scores from either model implementation
- `motif_pwm` — on-the-fly motif scanning and logo rendering
- `bar_correlation` — pairwise Pearson or Spearman correlation between two tracks, quantifying model agreement and highlighting divergent regions

Output is a set of per-region plots plus an HTML index for browsing. Supporting scripts in `genome_browser/` (`select_high_signal_regions.py`, `select_top_ctcf_overlap_regions.py`, `bed_to_region_config.py`, etc.) build region configs from BED files, peak overlaps, or signal thresholds.

## Setup

The environment can be built via Docker (`Dockerfile` / `docker-compose.yml`), which installs system dependencies (`apt-packages.txt`), the MEME suite (for `tomtom`/motif reports), and the Python stack pinned in [`requirements/base.txt`](requirements/base.txt) (PyTorch, `bpnet-lite`, `captum`, `shap`, `snakemake`, etc.).

The original BPNet paper's codebase (`bpnet-manuscript`, TensorFlow 1 / Keras 2) is set up separately via [`setup_bpnet_manuscript_env.sh`](setup_bpnet_manuscript_env.sh), since it requires an older, incompatible dependency stack from the modern `bpnet-lite` training pipeline.
