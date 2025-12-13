# Active Context: Fluorescence Knee Video Analysis

## Current Work Focus
The project has implemented a complete spatiotemporal analysis pipeline for fluorescence knee video analysis. The core functionality includes radial segmentation (N=64), automated COM calculation, spatiotemporal heatmap generation, and multi-video comparative analysis. Current focus is on validating results and preparing for statistical analysis between normal and aging knee groups.

## Recent Changes
- **Complete Heatmap Pipeline**: Implemented `generate_spatiotemporal_heatmaps.py` with full COM calculation and temporal synchronization
- **Data Preparation**: Created `prepare_intensity_data.py` for automated Excel export of intensity data with cycle metadata
- **Visualization Tools**: Developed `plot_com_cycles_from_heatmaps.py` for comparative COM plotting across multiple videos
- **Segmentation Previews**: Exported MP4 and TIF previews for all 7 videos at N=64 via `export_saved_segmentation_previews.py`
- **Anatomical Boundaries**: Extended `KneeSegmentationPipeline._show_radial_preview()` to display JC/OT and OT/SB region boundaries from `knee_metadata.py`
- **Temporal Alignment**: Added 50:50 rescaling for equal-duration flexion/extension phase comparison
- **Multi-Video Analysis**: Enabled batch processing and comparison of all 7 videos (4 normal, 3 aging)

## Next Steps
- **Peak Intensity Implementation**: Complete the TODO in `generate_spatiotemporal_heatmaps.py` for peak intensity contour tracking
- **Statistical Analysis**: Perform quantitative group comparisons between normal vs aging knees
- **Cross-Validation**: Validate segmentation accuracy and COM reliability across videos
- **Results Documentation**: Generate comprehensive analysis reports and visualizations
- **Method Refinement**: Evaluate and potentially refine COM definition based on peak intensity analysis

## Active Decisions and Considerations

### Segmentation Resolution
- **Current**: Using N=64 radial segments for detailed analysis
- **Consideration**: N=16 provides faster processing but may miss subtle flow patterns
- **Decision**: Prioritize N=64 for research accuracy, optimize performance as needed

### COM Definition
- **Current**: Weighted average of segment intensities (standard center of mass)
- **Investigation**: Exploring peak intensity tracking as alternative/complement to COM
- **Rationale**: Peak intensity may better represent dominant flow regions

### Cycle Synchronization
- **Current**: Rescaling cycles to equal duration for averaging
- **Consideration**: 50:50 flexion/extension split vs. natural cycle timing
- **Decision**: Using equal-duration rescaling to enable meaningful averaging across cycles

## Important Patterns and Preferences

### Data Organization
- Video data stored as `.npy` arrays in `../data/processed/`
- Intensity data exported to Excel files in `../data/intensities_total/`
- Analysis outputs saved to `../figures/` directory

### Processing Workflow
1. Load pre-processed video arrays
2. Apply radial segmentation masks
3. Extract intensity per segment over time
4. Identify flexion/extension cycles
5. Compute COM trajectories
6. Generate heatmaps and statistical summaries

### Video Types
- **Normal knees**: IDs 308, 1190, 1193, 1207
- **Aging knees**: IDs 1339, 1342, 1358
- **Segmentation**: Both N=16 and N=64 available for comparison

## Learnings and Project Insights

### Technical Insights
- Radial segmentation provides better anatomical representation than three-part division
- Histogram matching crucial for consistent thresholding across video frames
- Multiprocessing significantly improves video stabilization performance
- Memory mapping required for large radial segmentation arrays

### Analysis Insights
- COM trajectories show distinct patterns between normal and aging knees
- Flexion and extension phases exhibit different flow characteristics
- Individual cycles vary in duration but show consistent intensity patterns when normalized
- Peak intensity regions may provide additional insights beyond COM

### Challenges Identified
- Manual identification of flexion/extension cycle boundaries
- Variable video quality affecting segmentation consistency
- Memory constraints with high-resolution (N=64) segmentation
- Need for automated cycle detection algorithms
