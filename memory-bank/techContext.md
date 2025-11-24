# Tech Context: Fluorescence Knee Video Analysis

## Technologies Used

### Core Language & Runtime
- **Python 3.11**: Primary development language
- **Conda/Mamba**: Environment and package management
- **Jupyter Notebooks**: Exploratory analysis and validation

### Computer Vision & Image Processing
- **OpenCV (cv2)**: Video processing, morphological operations, thresholding
- **NumPy**: Array operations and mathematical computations
- **scikit-image**: Advanced image processing (histogram matching, exposure adjustment)

### Data Analysis & Visualization
- **pandas**: Data manipulation and coordinate handling
- **matplotlib**: Plotting and visualization
- **OpenPyXL**: Excel file handling for data export

### System Integration
- **multiprocessing**: Parallel video processing
- **tempfile**: Memory-mapped arrays for large datasets
- **functools**: Function composition utilities

## Development Setup

### Environment Management
Multiple conda environments maintained for different versions:
- `environment-1.0.yml` through `environment-1.2.yml`
- Incremental updates with new dependencies
- Version-specific package pinning

### Project Structure
```
├── src/                    # Main package
├── scripts/               # Executable analysis scripts
├── notebooks/            # Jupyter notebooks
├── data/                 # Input data (not in repo)
├── figures/              # Generated plots
├── memory-bank/          # Project documentation
└── docs/                 # Additional documentation
```

### Package Installation
- Local editable install via `pip install -e .`
- Version controlled through `pyproject.toml`
- Conda environment integration

## Technical Constraints

### Performance Limitations
- Memory-intensive video processing (large numpy arrays)
- CPU-bound operations (morphological operations, segmentation)
- Multiprocessing used for video stabilization to improve throughput

### Data Formats
- Input: Video files (.mp4, .avi) or numpy arrays (.npy)
- Output: Segmented masks, coordinate data, statistical summaries
- Intermediate: Memory-mapped arrays for large radial segmentations

### Platform Dependencies
- Windows development environment
- OpenCV compiled for Windows
- Path handling with backslashes (`\`)

## Dependencies & Tool Usage Patterns

### Import Patterns
```python
import numpy as np
import cv2
from core import knee_segmentation as ks
from utils import utils, io, views
```

### Array Conventions
- Videos: `(n_frames, height, width)` shape
- Masks: `(n_frames, height, width)` boolean/uint8 arrays
- Coordinates: pandas DataFrames with frame indices

### Error Handling
- Global `VERBOSE` flag for logging
- Input validation with shape checking
- Graceful handling of empty/missing frames

### File I/O
- NumPy `.npy` files for processed video data
- CSV/Excel export for coordinate and measurement data
- Relative path handling for data directories

## Development Workflow
- Script-based execution for specific analyses
- Jupyter notebooks for experimentation and validation
- Modular functions for reusability across different knee types (normal vs aging)
