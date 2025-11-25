# Fluorescence Knee Video Analysis

## Overview
This is a research tool for automated analysis of fluorescence-labeled synovial fluid flow in knee joints during flexion-extension cycles. It processes video data to segment knee regions and quantify synovial fluid dynamics, enabling comparison between normal and aging knees.

## Key Features
- Radial segmentation with configurable N sectors (e.g., 64 sectors for high resolution)
- Center of mass (COM) trajectory analysis across movement cycles
- Generation of spatiotemporal heatmaps showing intensity distribution
- Support for both normal and aging knee video datasets
- Batch processing capabilities for multiple videos

## Installation
1. Set up a conda environment:
```bash
conda env create -f environment-1.2.yml
conda activate knee-segmentation
```

2. Install the package:
```bash
pip install -e .
```

## Usage
1. **Process data**: Use segmentation scripts in `scripts/segmentation/` to segment video data and extract region intensities
2. **Generate figures**: Use analysis and visualization scripts in `scripts/visualization/` to create COM plots, heatmaps, and figures

Outputs are saved to `figures/` and exported data to `../data/`

## Requirements
- Python 3.11
- OpenCV, NumPy, pandas, matplotlib

## Status
This tool is primarily developed for academic research and comparative studies of joint physiology.
