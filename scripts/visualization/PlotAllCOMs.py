"""
Center of Mass (COM) Analysis for Knee Video Data

This module analyzes fluorescence intensity data from knee videos to compute
and plot center of mass values across segments during flexion-extension cycles.

Usage:
    python PlotAllCOMs.py <video_ids> <segments> <option>
    
    video_ids: Single video number (e.g., 1339) or comma-separated list (e.g., 1339,308,1190)
    segments: Number of segments (e.g., 16, 64)
    option: "total" (normalized total intensities) or "unit" (normalized average per pixel)

Example:
    python PlotAllCOMs.py 1339 64 total
    python PlotAllCOMs.py 1339,308,1190 64 total
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os.path
from config import TYPES


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

OPTIONS = {
    "total": "Normalized total intensities, per segment", 
    "unit": "Normalized average intensity per pixel, per segment"
}


# =============================================================================
# HELPER FUNCTIONS - DATA LOADING AND PROCESSING
# =============================================================================

def validate_cli_arguments(argv):
    """
    Validate and parse command-line arguments for CLI usage.
    
    Args:
        argv: Command line arguments (sys.argv)

    Returns:
        tuple: (video_ids, segment_count, option, multiple_videos)

    Raises:
        SyntaxError: If arguments are invalid
    """

    if len(argv) != 4 or argv[3] not in OPTIONS.keys():
        options_str = "\n\t" + "\n\t".join(f"'{k}': {v}" for k, v in OPTIONS.items())
        raise SyntaxError(
            f"\n\tExample usage: {argv[0]} 1339 64 total"
            f"\n\t              {argv[0]} 1339,308,1190 64 total  (plots multiple videos)"
            f"\n\tValid types are: {list(TYPES)}"
            f"\n\tOptions for the third argument are:{options_str}"
        )
    
    # Parse video IDs - can be single number or comma-separated list
    video_ids_str = argv[1]
    segment_count = int(argv[2])
    opt = argv[3]

    # Check if multiple videos (comma-separated)
    if ',' in video_ids_str:
        video_ids = [int(vid.strip()) for vid in video_ids_str.split(',')]
        multiple_videos = True
    else:
        video_ids = [int(video_ids_str)]
        multiple_videos = False
    
    # Validate video IDs are in TYPES (warn but don't fail)
    for vid_id in video_ids:
        if vid_id not in TYPES:
            print(f"Warning: Video {vid_id} not found in TYPES config, but will attempt to process anyway.")
    
    return video_ids, segment_count, opt, multiple_videos


def load_excel_data(video_number, segment_count):
    """
    Load and parse Excel data for a single video.
    
    Args:
        video_number: Video identifier
        segment_count: Number of segments
        
    Returns:
        tuple: (df_intensity_raw, df_num_pixels_raw, df_flex, df_ext)
        
    Raises:
        ValueError: If expected Excel file doesn't exist
    """
    input_xlsx = fr"../data/video_intensities/{video_number}N{segment_count}intensities.xlsx"
    
    if not os.path.isfile(input_xlsx):
        raise ValueError(
            f"File '{video_number}N{segment_count}intensities.xlsx' doesn't exist. "
            f"Is {video_number=} and {segment_count=} correct?"
        )
    
    # Load sheets
    xls = pd.ExcelFile(input_xlsx)
    df_intensity_raw = pd.read_excel(xls, sheet_name="Segment Intensities", header=None)
    df_num_pixels_raw = pd.read_excel(xls, sheet_name="Number of Mask Pixels", header=None)
    df_flex = pd.read_excel(xls, sheet_name="Flexion Frames", header=None)
    df_ext = pd.read_excel(xls, sheet_name="Extension Frames", header=None)
    
    return df_intensity_raw, df_num_pixels_raw, df_flex, df_ext


def clean_intensity_data(df_intensity_raw, df_num_pixels_raw, option):
    """
    Clean and optionally normalize intensity data.
    
    Args:
        df_intensity_raw: Raw intensity DataFrame
        df_num_pixels_raw: Raw pixel count DataFrame
        option: "total" or "unit" for normalization
        
    Returns:
        tuple: (df_intensity, df_num_pixels)
    """
    # Skip the first row (header: "Frame", "Segment 1", ..., "Segment 16")
    df_intensity = df_intensity_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    df_num_pixels = df_num_pixels_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    
    # Optional: take average intensity per pixel, per segment
    if option == "unit":
        df_intensity = df_intensity / df_num_pixels
        df_intensity.fillna(0, inplace=True)
    
    return df_intensity, df_num_pixels


def clean_interval_data(df):
    """
    Clean interval data from flexion/extension sheets.
    
    Args:
        df: Raw interval DataFrame
        
    Returns:
        tuple: (starts, ends) - arrays of interval boundaries
    """
    df = df.dropna(how="all")
    first_row = df.iloc[0, :].astype(str).str.lower()
    if ("start" in first_row.values) and ("end" in first_row.values):
        df = df.iloc[1:, :]
    df = df.iloc[:, 0:3]
    starts = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().astype(int).to_numpy()
    ends = pd.to_numeric(df.iloc[:, 2], errors="coerce").dropna().astype(int).to_numpy()
    return starts, ends


def normalize_intensity_per_frame(df_intensity):
    """
    Normalize intensity per frame across segments.
    
    Args:
        df_intensity: DataFrame with intensity values
        
    Returns:
        DataFrame: Normalized intensity values
    """
    norm_intensity = df_intensity.copy().astype(float)
    for i in range(norm_intensity.shape[0]):
        row = norm_intensity.iloc[i, :]
        min_val, max_val = row.min(), row.max()
        if max_val > min_val:
            norm_intensity.iloc[i, :] = 100 * (row - min_val) / (max_val - min_val)
        else:
            norm_intensity.iloc[i, :] = 0
    return norm_intensity


def compute_center_of_mass_per_frame(norm_intensity):
    """
    Compute center of mass values for each frame.
    
    Args:
        norm_intensity: Normalized intensity DataFrame
        
    Returns:
        pd.Series: COM values per frame
    """
    segments = np.arange(1, norm_intensity.shape[1] + 1)
    com_values = []
    for i in range(norm_intensity.shape[0]):
        intensities = norm_intensity.iloc[i, :].to_numpy()
        total = np.sum(intensities)
        if total > 0:
            com = np.sum(segments * intensities) / total
        else:
            com = np.nan
        com_values.append(com)
    return pd.Series(com_values, name="COM")


# =============================================================================
# HELPER FUNCTIONS - COM CYCLE COMPUTATIONS
# =============================================================================

def compute_cycle_coms_from_excel(com_series: pd.Series, 
                                   starts_flex: np.ndarray, 
                                   ends_flex: np.ndarray,
                                   starts_ext: np.ndarray, 
                                   ends_ext: np.ndarray) -> pd.DataFrame:
    """
    Extracts and centers COM values for each flexion-extension cycle pair.
    
    Args:
        com_series: pd.Series with COM values per frame
        starts_flex, ends_flex: arrays of flexion cycle boundaries (0-indexed, inclusive)
        starts_ext, ends_ext: arrays of extension cycle boundaries (0-indexed, inclusive)
        
    Returns:
        pd.DataFrame: cycle_coms with shape (nframes_rel, ncycles)
    """
    n_cycles = len(starts_flex)
    if len(starts_ext) != n_cycles:
        raise ValueError(f"Number of flexion cycles ({n_cycles}) != extension cycles ({len(starts_ext)})")
    
    cycle_coms = []
    for i in range(n_cycles):
        # Extract flexion and extension values
        flx_start, flx_end = starts_flex[i], ends_flex[i] + 1  # +1 for inclusive end
        ext_start, ext_end = starts_ext[i], ends_ext[i] + 1  # +1 for inclusive end
        
        # Ensure indices are within bounds
        flx_end = min(flx_end, len(com_series))
        ext_end = min(ext_end, len(com_series))
        
        flx_vals = com_series.iloc[flx_start:flx_end].values
        ext_vals = com_series.iloc[ext_start:ext_end].values
        
        # Center indices: flexion ends at -1, extension starts at 0
        flx_idx = np.arange(flx_start, flx_end) - flx_end  # shift endpoint to origin (negative)
        ext_idx = np.arange(ext_start, ext_end) - ext_start  # shift startpoint to origin (positive)
        
        flx_vals_series = pd.Series(flx_vals, index=flx_idx, name=i)
        ext_vals_series = pd.Series(ext_vals, index=ext_idx, name=i)
        
        # Concatenate flexion and extension
        cycle_coms.append(pd.concat([flx_vals_series, ext_vals_series], axis=0))
    
    # Combine all cycles into DataFrame
    cycle_coms_df = pd.concat(cycle_coms, axis=1)  # shape (nframes_rel, ncycles)
    
    return cycle_coms_df


def compute_average_cycle(cycle_coms: pd.DataFrame) -> pd.Series:
    """
    Computes the average cycle across all cycles in the DataFrame.
    
    Args:
        cycle_coms: pd.DataFrame with shape (nframes_rel, ncycles)
        
    Returns:
        pd.Series: average COM values, indexed by relative frame numbers
    """
    return cycle_coms.mean(axis=1, skipna=False)


# =============================================================================
# HELPER FUNCTIONS - PLOTTING
# =============================================================================

def plot_average_cycle(average_cycle: pd.Series, 
                       video_id: str = None, 
                       pdf_path: str = None) -> None:
    """
    Plots the average center of mass cycle.
    
    Args:
        average_cycle: pd.Series with average COM values
        video_id: Optional string identifier for the video
        pdf_path: Optional path to save PDF
    """
    plt.figure(figsize=(19, 7))
    
    # Plot average cycle
    plt.plot(average_cycle.sort_index(), color='blue', linewidth=2, label="Average of cycles")
    
    # Vertical line at midpoint (x=0)
    plt.axvline(0, linestyle="--", color='k', linewidth=1)
    
    # Formatting
    if video_id is not None:
        title_suffix = f" [{video_id}]"
    else:
        title_suffix = ""
    plt.title("Average position of fluorescence intensity" + title_suffix)
    plt.xlabel("Frames from midpoint (Left: flexion; Right: extension)")
    plt.ylabel("Segment number (JC to SB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to PDF if path provided
    if pdf_path is not None:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(plt.gcf())
        print(f"Exported COM plot to: {pdf_path}")
    
    # Show interactively
    plt.show()


def plot_multiple_videos_com(video_ids: list, segment_count: int, opt: str, pdf_path: str = None) -> None:
    """
    Plots average COM cycles for multiple videos on the same graph.
    
    Args:
        video_ids: List of video numbers to plot
        segment_count: Number of segments
        opt: Option for intensity calculation ("total" or "unit")
        pdf_path: Optional path to save PDF
    """
    plt.figure(figsize=(19, 7))
    
    # Process each video and plot
    cmap = plt.get_cmap('tab10', len(video_ids))
    
    for idx, video_number in enumerate(video_ids):
        try:
            average_cycle, video_label = process_single_video(video_number, segment_count, opt)
            plt.plot(average_cycle.sort_index(), 
                    label=video_label, 
                    color=cmap(idx), 
                    linewidth=2)
        except Exception as e:
            print(f"Error processing video {video_number}N{segment_count} ({opt}): {e}")
            continue
    
    # Vertical line at midpoint
    plt.axvline(0, linestyle="--", color='k', linewidth=1)
    
    # Formatting
    plt.title("Average position of fluorescence intensity (Multiple Videos)")
    plt.xlabel("Frames from midpoint (Left: flexion; Right: extension)")
    plt.ylabel("Segment number (JC to SB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to PDF if path provided
    if pdf_path is not None:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(plt.gcf())
        print(f"Exported multi-video COM plot to: {pdf_path}")
    
    # Show interactively
    plt.show()


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_single_video(video_number: int, segment_count: int, opt: str) -> tuple:
    """
    Process a single video file to compute average COM cycle.
    
    Args:
        video_number: Video identifier
        segment_count: Number of segments
        opt: Option for intensity calculation ("total" or "unit")
        
    Returns:
        tuple: (average_cycle, video_id)
    """
    # Load data
    df_intensity_raw, df_num_pixels_raw, df_flex, df_ext = load_excel_data(video_number, segment_count)
    
    # Clean intensity data
    df_intensity, df_num_pixels = clean_intensity_data(df_intensity_raw, df_num_pixels_raw, opt)
    
    # Extract intervals
    starts_flex, ends_flex = clean_interval_data(df_flex)
    starts_ext, ends_ext = clean_interval_data(df_ext)
    
    # Normalize intensity per frame
    norm_intensity = normalize_intensity_per_frame(df_intensity)
    
    # Compute COM per frame
    com_series = compute_center_of_mass_per_frame(norm_intensity)
    
    # Compute cycle COMs and average
    cycle_coms = compute_cycle_coms_from_excel(com_series, starts_flex, ends_flex, starts_ext, ends_ext)
    average_cycle = compute_average_cycle(cycle_coms)
    
    # Create video identifier using TYPES
    video_type = TYPES.get(video_number, "unknown")
    video_id = f"{video_number} {video_type}"
    
    return average_cycle, video_id


# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def main(video_ids, segment_count, opt, multiple_videos):
    """
    Main orchestration function for COM analysis.
    
    Args:
        video_ids: List of video numbers to process
        segment_count: Number of segments
        opt: Option for intensity calculation
        multiple_videos: Whether processing multiple videos
    """
    if multiple_videos:
        # Plot multiple videos on same figure
        video_ids_str = '_'.join(map(str, video_ids))
        pdf_path = fr"com_cycle_{opt}_{video_ids_str}_N{segment_count}.pdf"
        plot_multiple_videos_com(video_ids, segment_count, opt, pdf_path=pdf_path)
    else:
        # Single video mode
        video_number = video_ids[0]
        
        # Process the video
        average_cycle, video_label = process_single_video(video_number, segment_count, opt)
        
        # Create extended video identifier
        video_type = TYPES.get(video_number, "unknown")
        video_id = f"{video_number} {video_type} ({segment_count} segs, {opt})"
        
        # Plot average cycle for current video
        pdf_path = fr"com_cycle_{opt}_{video_number}N{segment_count}.pdf"
        plot_average_cycle(average_cycle, video_id=video_id, pdf_path=pdf_path)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        # Validate and parse CLI arguments
        video_ids, segment_count, opt, multiple_videos = validate_cli_arguments(sys.argv)
        
        # Run main analysis
        main(video_ids, segment_count, opt, multiple_videos)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
