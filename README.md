# Fluorescence Knee Video Analysis

A comprehensive research tool for automated analysis of fluorescence-labeled synovial fluid flow in knee joints during flexion-extension cycles. This project processes video data to segment knee regions, quantify synovial fluid dynamics, and enable comparative studies between normal and aging knees.

## Overview

This project addresses the need for automated analysis of fluorescence-labeled synovial fluid flow in knee joints during movement. Researchers studying osteoarthritis and joint health require quantitative measurements of fluid dynamics within the joint space, but manual analysis of video data is time-consuming and subjective.

The system provides:
- **Automated Segmentation**: Both three-part (L/M/R) and radial (N-sector) segmentation algorithms
- **Dynamic Analysis**: Center of mass (COM) trajectory tracking across movement cycles
- **Visualization**: Spatiotemporal heatmaps and comparative plots
- **Batch Processing**: Automated analysis of multiple video datasets

## Key Features

- **Flexible Segmentation**:
  - Radial segmentation with configurable N sectors (16 or 64 for high resolution)
  - Legacy three-part segmentation (left/middle/right division)
  - Anatomical boundary detection and circular geometry estimation

- **Advanced Analysis**:
  - Center of mass (COM) calculation from segmented intensity data
  - Flexion-extension cycle identification and temporal synchronization
  - Peak intensity tracking (in development)
  - Statistical comparisons between normal and aging knee groups

- **Comprehensive Processing Pipeline**:
  - Video stabilization and preprocessing
  - Otsu thresholding with adaptive scaling
  - Histogram matching for consistent frame processing
  - Memory-efficient processing for large video datasets

- **Rich Visualization**:
  - Spatiotemporal heatmaps showing intensity distribution over time
  - COM trajectory plots across movement cycles
  - Multi-video comparative analysis
  - Publication-quality figure generation

- **Extensible Architecture**:
  - Modular Python package structure
  - Command-line scripts for specific workflows
  - Jupyter notebooks for exploratory analysis

## Project Structure

```
├── src/                  # Main package
│   ├── core/              # Core algorithms
│   ├── utils/             # Utility functions
│   └── config.py          # Configuration
├── scripts/              # Executable analysis scripts
│   ├── segmentation/      # Video segmentation scripts
│   ├── analysis/          # Data analysis scripts
│   ├── visualization/     # Plotting and figure generation
│   └── utils/             # Utility scripts
├── notebooks/            # Jupyter notebooks for exploration
├── data/                 # Input data (not in repo)
├── figures/              # Generated plots and outputs
└── docs/                 # Additional documentation
```

## Installation

### Environment Setup
Choose the appropriate conda environment file based on your needs:

```bash
# Latest stable environment (recommended)
conda env create -f environment-1.3.yml
conda activate knee-segmentation

# Alternative versions
conda env create -f environment-1.1.yml  # Previous stable
conda env create -f environment-1.0.yml  # Initial version
```

### Package Installation
Install the package in editable mode:

```bash
pip install -e .
```

### Output Locations
- **Figures**: Generated plots saved to `figures/` directory
- **Data Exports**: Intensity data and measurements exported to `../data/` (parent directory)
- **Processed Videos**: NumPy arrays saved to `../data/segmented/`

## Requirements

### Core Dependencies
- Python 3.11
- OpenCV (cv2) - Video processing and image operations
- NumPy - Array operations and mathematical computations
- pandas - Data manipulation and coordinate handling
- matplotlib - Plotting and visualization
- scikit-image - Advanced image processing
- OpenPyXL - Excel file handling

### System Requirements
- Windows OS (developed and tested on Windows 11)
- Sufficient RAM for video processing (4GB+ recommended)
- Storage space for processed video arrays and generated figures

## Contributing

This is an academic research project developed for studies in joint physiology and osteoarthritis. The codebase follows research software development practices with comprehensive documentation and modular design.

## License

Academic research use. Please cite appropriately if used in publications.
