# FRAP: Faithful and Realistic Text-to-Image Generation with Adaptive Prompt Weighting [TMLR 2025]

[![Paper](https://img.shields.io/badge/Paper-TMLR%202025-blue)](https://openreview.net/forum?id=MKCwO34oIq)
[![arXiv](https://img.shields.io/badge/arXiv-2408.11706-b31b1b.svg)](https://arxiv.org/abs/2408.11706)

Official implementation of **FRAP**, a method for improving text-to-image generation faithfulness and realism via **adaptive prompt weighting**.

## Overview

FRAP adaptively adjusts token weights during diffusion sampling to improve prompt alignment while preserving image quality.

This repository includes:

- FRAP inference pipeline (`pipeline_frap.py`)
- Command-line runner (`run_frap.py`)
- Interactive notebook (`run_frap.ipynb`)
- Prompt datasets used in the paper (`datasets/`)

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment/environment.yaml
conda activate frap
```

## Quick Start

### Run in Jupyter Notebook

Use `run_frap.ipynb` for interactive generation. The notebook uses default settings from `config_frap.py`.

### Run from Command Line

Generate from a manually provided prompt list:

```bash
python run_frap.py --prompt='["A black cat sitting on top of a green bench in a field"]' --seeds [10,48] --output_path outputs/images/
```

Generate from a prompt file (`.txt`, one prompt per line):

```bash
python run_frap.py --prompt_file="datasets/divide_and_bind/coco_attribute.txt" --seeds [0,1] --output_path outputs/coco_attribute/
```

## Configuration

Default run settings are defined in `config_frap.py` (`RunConfig`).

## Datasets

Prompt sets used in the paper are provided in `datasets/`.

Each file contains prompts, one per line.

## Outputs

Generated images are saved under `--output_path` with:

- per-prompt subfolders containing generated images 
- merged image grids
- `latency_metrics.json` with runtime statistics

## Bibtex Citation
If you find our method and paper useful, we kindly ask that you cite our paper.
You can also find the preprint on arXiv: https://arxiv.org/abs/2408.11706

```bibtex
@article{
  jiang2025frap,
  title={{FRAP}: Faithful and Realistic Text-to-Image Generation with Adaptive Prompt Weighting},
  author={Liyao Jiang and Negar Hassanpour and Mohammad Salameh and Mohan Sai Singamsetti and Fengyu Sun and Wei Lu and Di Niu},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=MKCwO34oIq},
  note={}
}
```
