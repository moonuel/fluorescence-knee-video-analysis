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
  - Memory bank documentation system

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
├── memory-bank/          # Project documentation and context
├── data/                 # Input data (not in repo)
├── figures/              # Generated plots and outputs
└── docs/                 # Additional documentation
```

## Installation

### Environment Setup
Choose the appropriate conda environment file based on your needs:

```bash
# Latest stable environment (recommended)
conda env create -f environment-1.2.yml
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

## Usage

### Quick Start
1. **Prepare your data**: Place video files in the `data/` directory
2. **Run segmentation**: Use appropriate scripts from `scripts/segmentation/`
3. **Analyze results**: Generate heatmaps and plots with `scripts/analysis/` and `scripts/visualization/`

### Example Workflows

#### Process a Normal Knee Video (ID: 308)
```bash
python scripts/segmentation/normal_308_radial.py
python scripts/analysis/generate_spatiotemporal_heatmaps.py --video_id 308
python scripts/visualization/plot_com_cycles_from_heatmaps.py --video_ids 308
```

#### Process an Aging Knee Video (ID: 1339)
```bash
python scripts/segmentation/aging_1339_radial.py
python scripts/analysis/prepare_intensity_data.py --video_id 1339
```

#### Generate Comparative Analysis
```bash
# Plot COM cycles for all videos
python scripts/visualization/plot_all_com_cycles.py

# Generate spatiotemporal heatmaps for multiple videos
python scripts/analysis/generate_spatiotemporal_heatmaps.py --video_ids 308,1190,1193,1207,1339,1342,1358
```

### Available Video Datasets
- **Normal Knees**: 308, 1190, 1193, 1207
- **Aging Knees**: 1339, 1342, 1358

All videos are processed with N=64 radial segmentation by default.

### Output Locations
- **Figures**: Generated plots saved to `figures/` directory
- **Data Exports**: Intensity data and measurements exported to `../data/` (parent directory)
- **Processed Videos**: NumPy arrays saved to `../data/processed/`

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

## Documentation

This project uses a comprehensive Memory Bank documentation system:
- `memory-bank/projectbrief.md` - Core requirements and goals
- `memory-bank/productContext.md` - User experience and value proposition
- `memory-bank/techContext.md` - Technical implementation details
- `memory-bank/systemPatterns.md` - Architecture and design patterns
- `memory-bank/activeContext.md` - Current development focus
- `memory-bank/progress.md` - Project status and completion tracking

## Development Status

### ✅ Completed Features
- Complete spatiotemporal analysis pipeline for fluorescence knee video analysis
- Radial segmentation (N=64) with automated COM calculation
- Heatmap generation and temporal synchronization
- Multi-video comparative analysis (7 videos processed: 4 normal, 3 aging)
- Modular package structure with comprehensive test coverage

### 🔄 In Progress
- Peak intensity contour tracking implementation
- Automated cycle boundary detection
- Statistical analysis comparing normal vs aging groups

### 🎯 Future Goals
- Cross-validation of segmentation accuracy
- Extended temporal analysis capabilities
- Automated quality assessment metrics

## Contributing

This is an academic research project developed for studies in joint physiology and osteoarthritis. The codebase follows research software development practices with comprehensive documentation and modular design.

## License

Academic research use. Please cite appropriately if used in publications.

## Contact

For questions about usage or contributions, please refer to the project documentation in the `memory-bank/` directory.
