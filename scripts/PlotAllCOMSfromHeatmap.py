"""
Center of Mass (COM) Analysis from Pre-computed Heatmaps

This module loads pre-generated spatiotemporal heatmaps from Excel files
and plots center of mass values for flexion-extension cycles.

Usage:
    python PlotAllCOMSfromHeatmap.py <video_ids> <segments> <option> [--no-normalize] [--rescaled]

    video_ids: Single video number (e.g., 1339) or comma-separated list (e.g., 1339,308,1190)
    segments: Number of segments (e.g., 16, 64)
    option: "total" (normalized total intensities) or "unit" (normalized average per pixel)
    --no-normalize: Use non-normalized heatmaps (default: normalized)
    --rescaled: Use 50:50 rescaled heatmaps (default: original)

Example:
    python PlotAllCOMSfromHeatmap.py 1339 64 total
    python PlotAllCOMSfromHeatmap.py 1339,308,1190 64 total --rescaled
    python PlotAllCOMSfromHeatmap.py 1339 64 unit --no-normalize
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os.path
import argparse
from config import TYPES


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Options matching PlotAllCOMs.py for consistency
OPTIONS = {
    "total": "Normalized total intensities, per segment",
    "unit": "Normalized average intensity per pixel, per segment"
}


# =============================================================================
# HELPER FUNCTIONS - DATA LOADING
# =============================================================================

def load_heatmap_excel(video_number: int, segment_count: int, opt: str,
                       normalize: bool = True, rescaled: bool = False) -> tuple:
    """
    Load averaged flexion and extension intensity data from pre-computed heatmap Excel.

    Args:
        video_number: Video identifier
        segment_count: Number of segments
        opt: Processing option ("total" or "unit")
        normalize: Whether to load normalized heatmaps (default: True)
        rescaled: Whether to load rescaled (50:50) heatmaps (default: False)

    Returns:
        tuple: (avg_flex, avg_ext) - averaged intensity arrays

    Raises:
        FileNotFoundError: If the expected Excel file doesn't exist
    """
    # Construct filename suffixes
    norm_suffix = "_nonorm" if not normalize else ""
    rescale_suffix = "_rescaled" if rescaled else ""

    # Construct input file path
    input_xlsx = fr"../figures/spatiotemporal_maps/heatmap_{opt}{norm_suffix}{rescale_suffix}_{video_number}N{segment_count}.xlsx"

    if not os.path.isfile(input_xlsx):
        raise FileNotFoundError(
            f"Heatmap Excel file not found: {input_xlsx}\n"
            f"You may need to run AverageNormalizeHeatMap.py first to generate it."
        )

    # Load the averaged data from Excel
    xls = pd.ExcelFile(input_xlsx)
    avg_flex = pd.read_excel(xls, sheet_name="avg_flexion", header=None).values
    avg_ext = pd.read_excel(xls, sheet_name="avg_extension", header=None).values
    return avg_flex, avg_ext


# =============================================================================
# HELPER FUNCTIONS - COM COMPUTATIONS
# =============================================================================

def compute_com_from_intensity_array(intensity_array: np.ndarray) -> np.ndarray:
    """
    Compute center of mass from an intensity array.

    Args:
        intensity_array: Array with shape (n_frames, n_segments) containing intensity values

    Returns:
        Array of COM values, one per frame
    """
    n_frames, n_segments = intensity_array.shape
    segments = np.arange(1, n_segments + 1)  # 1-based segment indices
    com_values = []

    for i in range(n_frames):
        intensities = intensity_array[i, :]
        total = np.sum(intensities)
        if total > 0:
            com = np.sum(segments * intensities) / total
        else:
            com = np.nan
        com_values.append(com)

    return np.array(com_values)


def create_combined_com_series(com_flex: np.ndarray, com_ext: np.ndarray) -> pd.Series:
    """
    Create a combined COM series from separate flexion and extension COM curves.
    Flexion indices are negative (ending at -1), extension starts at 0.

    Args:
        com_flex: COM values for flexion phase
        com_ext: COM values for extension phase

    Returns:
        Combined pandas Series with centered indices
    """
    combined_com = np.concatenate([com_flex, com_ext])
    flex_len = len(com_flex)
    ext_len = len(com_ext)
    combined_idx = np.concatenate([
        np.arange(flex_len) - flex_len,  # flexion: -flex_len to -1
        np.arange(ext_len)                # extension: 0 to ext_len-1
    ])
    return pd.Series(combined_com, index=combined_idx, name="COM")


def resample_com_series_for_alignment(com_series: pd.Series, max_flex_len: int, max_ext_len: int) -> pd.Series:
    """
    Resample a COM series to match the maximum lengths for flexion and extension phases.

    Args:
        com_series: pandas Series with centered indices
        max_flex_len: Target length for flexion phase
        max_ext_len: Target length for extension phase

    Returns:
        Resampled pandas Series with aligned lengths
    """
    # Split into flexion and extension parts
    flex_series = com_series[com_series.index < 0].sort_index()
    ext_series = com_series[com_series.index >= 0].sort_index()

    # Resample flexion if needed
    if len(flex_series) < max_flex_len and max_flex_len > 0:
        old_x_flex = np.linspace(0, 1, len(flex_series)) if len(flex_series) > 1 else np.array([0])
        new_x_flex = np.linspace(0, 1, max_flex_len)
        new_flex_values = np.interp(new_x_flex, old_x_flex, flex_series.values)
    else:
        new_flex_values = flex_series.values

    # Resample extension if needed
    if len(ext_series) < max_ext_len and max_ext_len > 0:
        old_x_ext = np.linspace(0, 1, len(ext_series)) if len(ext_series) > 1 else np.array([0])
        new_x_ext = np.linspace(0, 1, max_ext_len)
        new_ext_values = np.interp(new_x_ext, old_x_ext, ext_series.values)
    else:
        new_ext_values = ext_series.values

    # Create new combined series
    combined_idx = np.concatenate([
        np.arange(max_flex_len) - max_flex_len,  # -max_flex_len to -1
        np.arange(max_ext_len)                    # 0 to max_ext_len-1
    ])
    combined_values = np.concatenate([new_flex_values, new_ext_values])
    return pd.Series(combined_values, index=combined_idx, name="COM")


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_single_video(video_number: int, segment_count: int, opt: str,
                         normalize: bool = True, rescaled: bool = False) -> tuple:
    """
    Process a single video's heatmap data to compute COM series.

    Args:
        video_number: Video identifier
        segment_count: Number of segments
        opt: Option for intensity calculation
        normalize: Whether to use normalized heatmaps
        rescaled: Whether to use rescaled heatmaps

    Returns:
        tuple: (com_series, video_label)
    """
    # Load averaged intensity data from heatmap Excel
    avg_flex, avg_ext = load_heatmap_excel(video_number, segment_count, opt, normalize, rescaled)

    # Compute COM from averaged flexion and extension arrays
    com_flex = compute_com_from_intensity_array(avg_flex)
    com_ext = compute_com_from_intensity_array(avg_ext)

    # Create combined COM series
    com_series = create_combined_com_series(com_flex, com_ext)

    # Create video identifier using TYPES config
    video_type = TYPES.get(video_number, "unknown")
    video_label = f"{video_number} {video_type}"

    return com_series, video_label


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_single_video_com(com_series: pd.Series, video_id: str, pdf_path: str = None) -> None:
    """
    Plot average COM cycle for a single video.

    Args:
        com_series: Combined COM series
        video_id: Video identifier string
        pdf_path: Optional path to save PDF
    """
    plt.figure(figsize=(19, 7))

    # Plot COM series (sort by index to ensure proper order)
    plt.plot(com_series.sort_index().index, com_series.sort_index().values, color='blue', linewidth=2, label="Average COM from Heatmap")

    # Vertical line at midpoint (flexion/extension transition)
    plt.axvline(0, linestyle="--", color='k', linewidth=1)

    # Formatting
    plt.title(f"Average position of fluorescence intensity from Heatmap [{video_id}]")
    plt.xlabel("Frames from midpoint (Left: flexion; Right: extension)")
    plt.ylabel("Segment number (JC to SB)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save to PDF if path provided
    if pdf_path is not None:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(plt.gcf())
        print(f"Exported heatmap COM plot to: {pdf_path}")

    # Show interactively
    plt.show()


def plot_multiple_videos_com(video_ids: list, segment_count: int, opt: str,
                            normalize: bool, rescaled: bool, pdf_path: str = None) -> None:
    """
    Plot average COM cycles for multiple videos on the same graph.
    Temporally aligns COM series by resampling to match the longest phase durations.

    Args:
        video_ids: List of video numbers to plot
        segment_count: Number of segments
        opt: Option for intensity calculation
        normalize: Whether using normalized heatmaps
        rescaled: Whether using rescaled heatmaps
        pdf_path: Optional path to save PDF
    """
    plt.figure(figsize=(19, 7))

    # Color map for multiple videos
    cmap = plt.get_cmap('tab10', len(video_ids))

    # First pass: process all videos and collect COM series
    all_com_series = []
    valid_video_labels = []
    valid_indices = []

    for idx, video_number in enumerate(video_ids):
        try:
            com_series, video_label = process_single_video(video_number, segment_count, opt, normalize, rescaled)
            all_com_series.append(com_series)
            valid_video_labels.append(video_label)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error processing video {video_number}N{segment_count} ({opt}, norm={normalize}, rescaled={rescaled}): {e}")
            continue

    # Find maximum lengths across all videos
    if not all_com_series:
        print("No valid videos to plot.")
        return

    flex_lengths = [len(series[series.index < 0]) for series in all_com_series]
    ext_lengths = [len(series[series.index >= 0]) for series in all_com_series]
    max_flex_len = max(flex_lengths)
    max_ext_len = max(ext_lengths)

    print(f"Aligning videos to max flexion length: {max_flex_len}, max extension length: {max_ext_len}")

    # Second pass: resample and plot
    for i, (com_series, video_label, idx) in enumerate(zip(all_com_series, valid_video_labels, valid_indices)):
        # Resample for temporal alignment
        aligned_com_series = resample_com_series_for_alignment(com_series, max_flex_len, max_ext_len)

        plt.plot(aligned_com_series.sort_index().index, aligned_com_series.sort_index().values,
                label=video_label,
                color=cmap(valid_indices[i]),
                linewidth=2)

    # Vertical line at midpoint (flexion/extension transition)
    plt.axvline(0, linestyle="--", color='k', linewidth=1)

    # Formatting
    normalize_str = "Normalized" if normalize else "Raw"
    rescaled_str = " (Rescaled 50:50)" if rescaled else ""
    plt.title(f"Average position of fluorescence intensity from Heatmaps ({normalize_str}{rescaled_str}) - Multiple Videos (Temporally Aligned)")
    plt.xlabel("Frames from midpoint (Left: flexion; Right: extension)")
    plt.ylabel("Segment number (JC to SB)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save to PDF if path provided
    if pdf_path is not None:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(plt.gcf())
        print(f"Exported multi-video heatmap COM plot to: {pdf_path}")

    # Show interactively
    plt.show()


# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def main(video_ids: list, segment_count: int, opt: str, normalize: bool, rescaled: bool) -> None:
    """
    Main orchestration function for COM heatmap analysis.

    Args:
        video_ids: List of video numbers to process
        segment_count: Number of segments
        opt: Option for intensity calculation
        normalize: Whether to use normalized heatmaps
        rescaled: Whether to use rescaled heatmaps
    """
    multiple_videos = len(video_ids) > 1

    if multiple_videos:
        # Plot multiple videos on same figure
        video_ids_str = '_'.join(map(str, video_ids))
        norm_str = "nonorm" if not normalize else ""
        rescale_str = "rescaled" if rescaled else ""
        suffix = f"{norm_str}{'_' if norm_str and rescale_str else ''}{rescale_str}".strip("_")
        if suffix:
            suffix = f"_{suffix}"
        pdf_path = fr"com_from_heatmap_{opt}{suffix}_{video_ids_str}_N{segment_count}.pdf"
        plot_multiple_videos_com(video_ids, segment_count, opt, normalize, rescaled, pdf_path=pdf_path)
    else:
        # Single video mode
        video_number = video_ids[0]

        # Process the video
        com_series, video_label = process_single_video(video_number, segment_count, opt, normalize, rescaled)

        # Create extended video identifier
        video_type = TYPES.get(video_number, "unknown")
        video_id_extended = f"{video_number} {video_type}"

        # Plot COM cycle for current video
        norm_str = "nonorm" if not normalize else ""
        rescale_str = "rescaled" if rescaled else ""
        suffix = f"{norm_str}{'_' if norm_str and rescale_str else ''}{rescale_str}".strip("_")
        pdf_name = f"com_from_heatmap_{opt}{f'_{suffix}' if suffix else ''}_{video_number}N{segment_count}.pdf"
        pdf_path = pdf_name
        plot_single_video_com(com_series, video_id_extended, pdf_path=pdf_path)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot center of mass from pre-computed spatiotemporal heatmaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python {sys.argv[0]} 1339 64 total                             # Default: normalized, original
  python {sys.argv[0]} 1339 64 total --no-normalize               # Non-normalized
  python {sys.argv[0]} 1339,308,1190 64 total --rescaled          # Multiple videos, 50:50 rescaled

Valid video types are: {list(TYPES)}
Options for the third argument are:
""" + "\n".join(f"  '{k}': {v}" for k, v in OPTIONS.items())
    )

    parser.add_argument("video_ids", type=str, help="Video ID(s): single number or comma-separated list")
    parser.add_argument("segment_count", type=int, help="Number of segments")
    parser.add_argument("opt", choices=OPTIONS.keys(), help="Processing option")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Use non-normalized heatmaps (default: normalized)")
    parser.add_argument("--rescaled", action="store_true",
                       help="Use 50:50 rescaled heatmaps (default: original)")

    args = parser.parse_args()

    # Parse video IDs (comma-separated or single)
    video_ids_str = args.video_ids
    if ',' in video_ids_str:
        video_ids = [int(vid.strip()) for vid in video_ids_str.split(',')]
    else:
        video_ids = [int(video_ids_str)]

    # Set flags (default True for normalize, False for rescaled)
    normalize = not args.no_normalize
    rescaled = args.rescaled

    # Validate video IDs (warn but don't fail)
    for vid_id in video_ids:
        if vid_id not in TYPES:
            print(f"Warning: Video {vid_id} not found in TYPES config, but will attempt to process anyway.")

    try:
        main(video_ids, args.segment_count, args.opt, normalize, rescaled)
    except Exception as e:
        print(f"Error: {e}")
