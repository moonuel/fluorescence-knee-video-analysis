# Fluorescence Knee Video Analysis

A comprehensive research tool for automated analysis of fluorescence-labeled synovial fluid flow in knee joints during flexion-extension cycles. This project processes video data to segment knee regions, quantify synovial fluid dynamics, and enable comparative studies between normal and aging knees.

## Overview

This project addresses the need for automated analysis of fluorescence-labeled synovial fluid flow in knee joints during movement. Researchers studying osteoarthritis and joint health require quantitative measurements of fluid dynamics within the joint space, but manual analysis of video data is time-consuming and subjective.

The system provides:
- **Automated Segmentation**: 
- **Analysis Tools**: 
- **Visualization**: 

## Workflows (segmentation → Excel → heatmaps → COM)

This repository is organized as a linear pipeline. Later steps assume the outputs (file formats, naming conventions, and metadata) produced by earlier steps.

### 1) Generate segmentations (radial masks)

1. Convert raw video to centered grayscale `.npy`
   - GUI entrypoints:
     - [`scripts/utils/gui_process_avi.py`](scripts/utils/gui_process_avi.py)
     - [`scripts/utils/gui_process_tif.py`](scripts/utils/gui_process_tif.py)
   - Processing backend (AVI): [`scripts/utils/process_avi_to_npy.py`](scripts/utils/process_avi_to_npy.py)
     - Produces grayscale `uint8` video arrays.
     - Centers frames using [`core.radial_segmentation.centre_video()`](src/core/radial_segmentation.py:1).

2. Run segmentation pipeline
   - Base workflow: [`KneeSegmentationPipeline`](src/pipelines/base.py:15)
   - Pattern: create a new file under `scripts/segmentation/` that subclasses `KneeSegmentationPipeline` and overrides preprocessing / mask-generation methods as needed (see example [`scripts/segmentation/aging_1339_radial.py`](scripts/segmentation/aging_1339_radial.py:1)).
   - Output location: `data/segmented/` (default; see [`default_output_dir()`](src/pipelines/base.py:128)).
   - Output files written by [`save_results()`](src/pipelines/base.py:534):
     - `{condition}_{video_id}_video_N{N}.npy`
     - `{condition}_{video_id}_radial_N{N}.npy`
     - `{condition}_{video_id}_femur_N{N}.npy`

**Dependency notes**
- The pipeline’s preview overlays anatomical region boundaries using metadata from [`get_knee_meta()`](src/config/knee_metadata.py:384). Missing metadata does not prevent segmentation, but will remove those overlays.
- Downstream analysis assumes radial masks are label images with background `0` and segments labeled `1..N`.

### 2) Generate comprehensive Excel (per-frame intensities)

- Script: [`scripts/analysis/prepare_intensity_data.py`](scripts/analysis/prepare_intensity_data.py)
- Inputs:
  - `data/segmented/{condition}_{video_id}_radial_N{N}.npy`
  - `data/segmented/{condition}_{video_id}_video_N{N}.npy`
  - Metadata from [`src/config/knee_metadata.py`](src/config/knee_metadata.py) via [`get_knee_meta()`](src/config/knee_metadata.py:384) (cycles + anatomical region segment ranges).
- Output:
  - `data/intensities_total/{video_id}N{N}intensities.xlsx` (written by [`save_to_excel()`](scripts/analysis/prepare_intensity_data.py:216))

**Dependency pitfalls**
- [`prepare_intensity_data.py`](scripts/analysis/prepare_intensity_data.py:131) discovers/loads files using **zero-padded** IDs (e.g. `0308`) when constructing filenames (see [`main()`](scripts/analysis/prepare_intensity_data.py:319)). Ensure segmentation outputs follow the same naming convention for IDs.

### 3) Generate spatiotemporal heatmaps (+ Excel)

- Script: [`scripts/analysis/generate_spatiotemporal_heatmaps.py`](scripts/analysis/generate_spatiotemporal_heatmaps.py)
- Depends on Step 2 output Excel:
  - `data/intensities_total/{video_id}N{N}intensities.xlsx` (validated by [`validate_input_file()`](scripts/analysis/generate_spatiotemporal_heatmaps.py:103)).
- Outputs:
  - Heatmap Excel workbooks in `figures/spatiotemporal_maps/` containing `avg_flexion` and `avg_extension` sheets (written by [`save_results_to_excel()`](scripts/analysis/generate_spatiotemporal_heatmaps.py:384)).
  - Matching PDFs of the heatmaps.

### 4) Multi-file COM plots + statistics (from heatmap workbooks)

- Script: [`scripts/visualization/plot_com_cycles_from_heatmaps.py`](scripts/visualization/plot_com_cycles_from_heatmaps.py)
- Depends on Step 3 heatmap Excel outputs in `figures/spatiotemporal_maps/`.
- Produces:
  - COM PDF plots
  - CSV summary tables (mean/sd/range and oscillation metrics)

### 5) Single-cycle / DMM analysis (region-wise COM, totals, flux)

- Script: [`scripts/visualization/dmm_analysis.py`](scripts/visualization/dmm_analysis.py)
- Inputs:
  - `data/segmented/{condition}_{id}_video_N{N}.npy` and `data/segmented/{condition}_{id}_radial_N{N}.npy` (loaded by [`load_video_data()`](scripts/visualization/dmm_analysis.py:131)).
  - Metadata from [`get_knee_meta()`](src/config/knee_metadata.py:384) for anatomical region ranges and cycle definitions.

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
Install the required dependencies from the provided environment files, then install the package in editable mode:

```bash
pip install -e .
```

### Output Locations
- **Figures**: Generated plots saved to `figures/` directory
- **Data**: Raw and intermediate data stored in `data/`

## Requirements

### Core Dependencies
- Python 3.11
- OpenCV (cv2) - Video processing and image operations
- NumPy - Array operations and mathematical computations
- pandas - Data manipulation and coordinate handling
- matplotlib - Plotting and visualization
- scikit-image - Advanced image processing
- OpenPyXL - Excel file handling
- See environment-*.yml for full dependency list

### System Requirements
- Windows OS (developed and tested on Windows 11)
- Sufficient RAM for video processing (4GB+ recommended)
- Storage space for processed video arrays and generated figures

## Contributing

This is an academic research project developed for studies in joint physiology and osteoarthritis. The codebase follows research software development practices with comprehensive documentation and modular design.

## License

Academic research use. Please cite appropriately if used in publications.
