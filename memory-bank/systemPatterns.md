# System Patterns: Fluorescence Knee Video Analysis

## System Architecture

### Core Components
```
src/
├── core/
│   ├── knee_segmentation.py     # Three-part segmentation algorithms
│   ├── radial_segmentation.py   # Radial segmentation algorithms
│   └── data_processing.py       # Video loading and preprocessing
├── utils/
│   ├── io.py                    # File I/O utilities
│   ├── utils.py                 # Image processing helpers
│   └── views.py                 # Visualization functions
└── config.py                    # Global configuration
```

### Processing Pipeline
1. **Data Ingestion** → `data_processing.py`
2. **Video Stabilization** → `utils.centroid_stabilization()`
3. **Thresholding** → `knee_segmentation.get_otsu_masks()`
4. **Segmentation** → Three-part or radial algorithms
5. **Analysis** → Region-based quantification
6. **Output** → Data export and visualization

## Key Technical Decisions

### Thresholding Strategy
- **Otsu Method**: Automatic threshold selection for fluorescence videos
- **Adaptive Scaling**: `thresh_scale` parameter for sensitivity adjustment
- **Histogram Matching**: Frame-to-frame intensity normalization

### Segmentation Approaches
- **Three-Part**: Left/Middle/Right division using coordinate-based bisection (old approach)
- **Radial**: N-sector division around circular knee geometry
- **Boundary Detection**: Edge-point sampling for anatomical landmarks

### Coordinate Systems
- **Original**: Raw video coordinates
- **Stabilized**: Centroid-corrected coordinates
- **Anatomical**: Knee-oriented coordinate system

## Design Patterns

### Function Composition
- Modular functions that transform video data through pipelines
- Consistent input/output signatures for composability
- Parameter passing for algorithm customization

### Array Processing
- NumPy-based vectorized operations for performance
- Frame-wise processing with consistent (n_frames, height, width) shapes
- Memory-efficient processing for large video files

### Error Handling
- Verbose logging with `VERBOSE` flag
- Graceful degradation for edge cases
- Input validation and shape checking

## Component Relationships

### Core Dependencies
- `knee_segmentation` → `utils` (image processing)
- `radial_segmentation` → `knee_segmentation` (thresholding)
- `radial_segmentation` → `utils` (stabilization)

### Data Flow
```
Raw Video → Stabilization → Thresholding → Segmentation → Quantification → Export
```

## Critical Implementation Paths

### Three-Part Segmentation
1. Load coordinates for knee boundaries
2. Apply centroid stabilization
3. Generate Otsu threshold masks
4. Create bisecting masks for L/M/R division
5. Extract region intensities

### Radial Segmentation
1. Sample femur boundary points
2. Estimate circular geometry (center + reference point)
3. Generate N equally-spaced radial sectors
4. Apply sector-based masking
5. Quantify fluorescence per sector

### Performance Considerations
- Multiprocessing support for video stabilization
- Memory mapping for large radial segmentation arrays
