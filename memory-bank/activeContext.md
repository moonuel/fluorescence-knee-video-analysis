# Active Context: Fluorescence Knee Video Analysis

## Current Work Focus
The project is currently analyzing center of mass (COM) trajectories of synovial fluid flow in knee joints during flexion-extension cycles. The primary analysis compares fluorescence intensity distribution patterns between normal and aging knee joints using radial segmentation.

## Recent Changes
- **Radial Segmentation**: Transitioned from three-part (L/M/R) to radial segmentation with N=64 sectors for higher resolution analysis
- **COM Analysis**: Implemented automated COM calculation from segmented intensity data across flexion-extension cycles
- **Heatmap Generation**: Created spatiotemporal heatmaps showing intensity distribution over joint movement cycles
- **Multi-Video Processing**: Extended analysis to compare multiple videos within normal and aging groups

## Next Steps
- **Peak Intensity Analysis**: TODO - Implement peak intensity contour tracking to compare with COM curves and refine COM definition
- **Statistical Comparisons**: Perform quantitative comparisons between normal vs aging knee groups
- **Validation**: Cross-validate segmentation accuracy and COM calculations
- **Batch Processing**: Automate analysis pipeline for all available video datasets

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
- Intensity data exported to Excel files in `../data/video_intensities/`
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
