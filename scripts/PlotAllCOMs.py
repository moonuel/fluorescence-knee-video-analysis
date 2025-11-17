import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os.path
from typing import Optional
from config import TYPES

OPTIONS = {"total": "Normalized total intensities, per segment", 
           "unit": "Normalized average intensity per pixel, per segment"}

# --- Input validation  ---
if len(sys.argv[1:]) != 3 or sys.argv[3] not in OPTIONS.keys(): 
    options_str = "\n\t" + "\n\t".join(f"     '{k}': {v}" for k, v in OPTIONS.items())
    raise SyntaxError(
        f"\n\tExample usage: {sys.argv[0]} 1339 64 total"
        f"\n\t              {sys.argv[0]} 1339,308,1190 64 total  (plots multiple videos)"
        f"\n\tValid types are: {list(TYPES)}"
        f"\n\tOptions for the third argument are:{options_str}"
    )

# Parse video IDs - can be single number or comma-separated list
video_ids_str = sys.argv[1]
segment_count = int(sys.argv[2])
opt = sys.argv[3]

# Check if multiple videos (comma-separated)
if ',' in video_ids_str:
    video_ids = [int(vid.strip()) for vid in video_ids_str.split(',')]
    multiple_videos = True
else:
    video_ids = [int(video_ids_str)]
    multiple_videos = False

# --- COM Cycle Functions ---

def resample_cycle_half(values: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resamples a 1-D array to the specified target length using linear interpolation.
    """
    values = np.asarray(values, dtype=float)

    if target_len <= 0:
        raise ValueError(f"target_len must be positive. Received {target_len}")

    if values.size == 0:
        return np.full(target_len, np.nan)

    if np.all(np.isnan(values)):
        return np.full(target_len, np.nan)

    old_x = np.linspace(0, 1, values.size)
    new_x = np.linspace(0, 1, target_len)

    if np.isnan(values).any():
        valid_mask = ~np.isnan(values)
        # If only one valid value, repeat it
        if valid_mask.sum() == 1:
            return np.full(target_len, values[valid_mask][0])
        values = np.interp(old_x, old_x[valid_mask], values[valid_mask])

    return np.interp(new_x, old_x, values)


def compute_target_half_length(starts_flex: np.ndarray,
                               ends_flex: np.ndarray,
                               starts_ext: np.ndarray,
                               ends_ext: np.ndarray) -> int:
    """
    Computes the average duration across all half-cycles (flexion + extension)
    and returns it as an integer length for resampling.
    """
    flex_lengths = (ends_flex - starts_flex + 1)
    ext_lengths = (ends_ext - starts_ext + 1)

    all_lengths = np.concatenate([flex_lengths, ext_lengths])

    if all_lengths.size == 0:
        raise ValueError("No half-cycle lengths provided for rescaling.")

    target_len = int(np.round(np.mean(all_lengths)))
    target_len = max(target_len, 1)

    return target_len


def compute_cycle_coms_from_excel(com_series: pd.Series, 
                                   starts_flex: np.ndarray, 
                                   ends_flex: np.ndarray,
                                   starts_ext: np.ndarray, 
                                   ends_ext: np.ndarray,
                                   target_half_len: Optional[int] = None) -> pd.DataFrame:
    """
    Extracts and centers COM values for each flexion-extension cycle pair.
    
    Adapts compute_cycle_coms from plot_centre_of_mass.py to work with Excel-based data structure.
    Cycles are centered: flexion frames get negative indices (ending at -1), 
    extension frames get positive indices (starting at 0).
    
    Inputs:
        com_series: pd.Series with COM values per frame
        starts_flex, ends_flex: arrays of flexion cycle boundaries (0-indexed, inclusive)
        starts_ext, ends_ext: arrays of extension cycle boundaries (0-indexed, inclusive)
        Cycles are paired by index (i-th flexion pairs with i-th extension)
    
    Outputs:
        cycle_coms: pd.DataFrame with shape (nframes_rel, ncycles), each column is a centered cycle
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

        if target_half_len is not None:
            flx_vals = resample_cycle_half(flx_vals, target_half_len)
            ext_vals = resample_cycle_half(ext_vals, target_half_len)
            flx_idx = np.arange(-target_half_len, 0)
            ext_idx = np.arange(0, target_half_len)
        else:
            # Center indices: flexion ends at -1, extension starts at 0
            flx_idx = np.arange(flx_start, flx_end) - flx_end
            ext_idx = np.arange(ext_start, ext_end) - ext_start

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
    
    Inputs:
        cycle_coms: pd.DataFrame with shape (nframes_rel, ncycles)
    
    Outputs:
        average_cycle: pd.Series with average COM values, indexed by relative frame numbers
    """
    average_cycle = cycle_coms.mean(axis=1, skipna=False)
    return average_cycle


def plot_average_cycle(average_cycle: pd.Series, 
                       video_id: str = None, 
                       pdf_path: str = None) -> None:
    """
    Plots the average center of mass cycle.
    
    Adapts plot_cycle_coms from plot_centre_of_mass.py but plots only the average.
    Saves to PDF and shows interactively.
    
    Inputs:
        average_cycle: pd.Series with average COM values, indexed by relative frame numbers
        video_id: Optional string identifier for the video (e.g., "308 Normal")
        pdf_path: Optional path to save PDF. If None, saves to default location.
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


# Validate video IDs are in TYPES
for vid_id in video_ids:
    if vid_id not in TYPES:
        print(f"Warning: Video {vid_id} not found in TYPES config, but will attempt to process anyway.")

target_half_len = None  # will be determined per single-video mode
video_number = None
com_series = None
starts_flex = ends_flex = starts_ext = ends_ext = None

# For single video, use original behavior
if not multiple_videos:
    video_number = video_ids[0]
    INPUT_XLSX = fr"../data/video_intensities/{video_number}N{segment_count}intensities.xlsx"  
    
    if not os.path.isfile(INPUT_XLSX): 
        raise ValueError(f"File '{video_number}N{segment_count}intensities.xlsx' doesn't exist. \n\t    Is {video_number=} and {segment_count=} correct?")

    # --- Load sheets ---
    xls = pd.ExcelFile(INPUT_XLSX)
    df_intensity_raw = pd.read_excel(xls, sheet_name="Segment Intensities", header=None)
    df_num_pixels_raw = pd.read_excel(xls, sheet_name="Number of Mask Pixels", header=None)
    df_flex = pd.read_excel(xls, sheet_name="Flexion Frames", header=None)
    df_ext = pd.read_excel(xls, sheet_name="Extension Frames", header=None)

    # --- Clean intensity data ---
    # Skip the first row (header: "Frame", "Segment 1", ..., "Segment 16")
    df_intensity = df_intensity_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    df_num_pixels = df_num_pixels_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    # Optional: take average intensity per pixel, per segment?
    if opt == "total": 
        pass
    if opt == "unit":
        df_intensity = df_intensity / df_num_pixels
        df_intensity.fillna(0, inplace=True)

    # --- Function to clean interval sheets ---
    def clean_intervals(df):
        df = df.dropna(how="all")
        first_row = df.iloc[0, :].astype(str).str.lower()
        if ("start" in first_row.values) and ("end" in first_row.values):
            df = df.iloc[1:, :]
        df = df.iloc[:, 0:3]
        starts = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().astype(int).to_numpy()
        ends = pd.to_numeric(df.iloc[:, 2], errors="coerce").dropna().astype(int).to_numpy()
        return starts, ends

    # --- Extract intervals ---
    starts_flex, ends_flex = clean_intervals(df_flex)
    starts_ext, ends_ext = clean_intervals(df_ext)

    target_half_len = compute_target_half_length(starts_flex, ends_flex, starts_ext, ends_ext)

    # --- Step (1): normalize intensity per frame ---
    norm_intensity = df_intensity.copy().astype(float)
    for i in range(norm_intensity.shape[0]):
        row = norm_intensity.iloc[i, :]
        min_val, max_val = row.min(), row.max()
        if max_val > min_val:
            norm_intensity.iloc[i, :] = 100 * (row - min_val) / (max_val - min_val)
        else:
            norm_intensity.iloc[i, :] = 0

    # --- Step (2): Flexion average ---
    max_len_flex = max(ends_flex - starts_flex + 1)
    rescaled_flex_all = []
    for s, e in zip(starts_flex, ends_flex):
        data = norm_intensity.iloc[s:e+1, :].to_numpy()
        old_x = np.linspace(0, 1, data.shape[0])
        new_x = np.linspace(0, 1, max_len_flex)
        rescaled = np.array([np.interp(new_x, old_x, data[:, j]) for j in range(data.shape[1])]).T
        rescaled_flex_all.append(rescaled)
    avg_flex = np.mean(rescaled_flex_all, axis=0)

    # --- Step (3): Extension average ---
    max_len_ext = max(ends_ext - starts_ext + 1)
    rescaled_ext_all = []
    for s, e in zip(starts_ext, ends_ext):
        data = norm_intensity.iloc[s:e+1, :].to_numpy()
        old_x = np.linspace(0, 1, data.shape[0])
        new_x = np.linspace(0, 1, max_len_ext)
        rescaled = np.array([np.interp(new_x, old_x, data[:, j]) for j in range(data.shape[1])]).T
        rescaled_ext_all.append(rescaled)
    avg_ext = np.mean(rescaled_ext_all, axis=0)

    # --- Step (4): Rescale flextion:extension = 50%:50% and convert the horizontal axis to be angles---

    # Compute individual durations
    len_flex = ends_flex - starts_flex + 1
    len_ext = ends_ext - starts_ext + 1

    # Find the *average* duration across all cycles (flex + ext)
    avg_duration = int(np.round(np.mean(np.concatenate([len_flex, len_ext]))))

    rescaled_flex_all_50 = []
    for s, e in zip(starts_flex, ends_flex):
        data = norm_intensity.iloc[s:e+1, :].to_numpy()
        old_x = np.linspace(0, 1, data.shape[0])
        new_x = np.linspace(0, 1, avg_duration)
        rescaled = np.array([
            np.interp(new_x, old_x, data[:, j]) for j in range(data.shape[1])
        ]).T
        rescaled_flex_all_50.append(rescaled)

    rescaled_ext_all_50 = []
    for s, e in zip(starts_ext, ends_ext):
        data = norm_intensity.iloc[s:e+1, :].to_numpy()
        old_x = np.linspace(0, 1, data.shape[0])
        new_x = np.linspace(0, 1, avg_duration)
        rescaled = np.array([
            np.interp(new_x, old_x, data[:, j]) for j in range(data.shape[1])
        ]).T
        rescaled_ext_all_50.append(rescaled)

    # Compute the average cycle for each phase
    avg_flex_50 = np.mean(rescaled_flex_all_50, axis=0)
    avg_ext_50  = np.mean(rescaled_ext_all_50, axis=0)


    # --- Step (2): Compute COM per frame ---
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
    com_series = pd.Series(com_values, name="COM")


def process_video_com_cycles(video_number: int, segment_count: int, opt: str) -> tuple:
    """
    Processes a single video file to compute average COM cycle.
    
    Inputs:
        video_number: Video identifier (e.g., 1339, 308)
        segment_count: Number of segments (e.g., 16, 64)
        opt: Option for intensity calculation ("total" or "unit")
    
    Outputs:
        average_cycle: pd.Series with average COM cycle
        video_id: str identifier for the video
    """
    input_xlsx = fr"../data/video_intensities/{video_number}N{segment_count}intensities.xlsx"
    
    if not os.path.isfile(input_xlsx):
        raise ValueError(f"File '{video_number}N{segment_count}intensities.xlsx' doesn't exist.")
    
    # Load sheets
    xls = pd.ExcelFile(input_xlsx)
    df_intensity_raw = pd.read_excel(xls, sheet_name="Segment Intensities", header=None)
    df_num_pixels_raw = pd.read_excel(xls, sheet_name="Number of Mask Pixels", header=None)
    df_flex = pd.read_excel(xls, sheet_name="Flexion Frames", header=None)
    df_ext = pd.read_excel(xls, sheet_name="Extension Frames", header=None)
    
    # Clean intensity data
    df_intensity = df_intensity_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    df_num_pixels = df_num_pixels_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    
    # Optional: take average intensity per pixel, per segment?
    if opt == "unit":
        df_intensity = df_intensity / df_num_pixels
        df_intensity.fillna(0, inplace=True)
    
    # Extract intervals
    def clean_intervals_local(df):
        df = df.dropna(how="all")
        first_row = df.iloc[0, :].astype(str).str.lower()
        if ("start" in first_row.values) and ("end" in first_row.values):
            df = df.iloc[1:, :]
        df = df.iloc[:, 0:3]
        starts = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().astype(int).to_numpy()
        ends = pd.to_numeric(df.iloc[:, 2], errors="coerce").dropna().astype(int).to_numpy()
        return starts, ends
    
    starts_flex, ends_flex = clean_intervals_local(df_flex)
    starts_ext, ends_ext = clean_intervals_local(df_ext)

    target_half_len = compute_target_half_length(starts_flex, ends_flex, starts_ext, ends_ext)
    
    # Normalize intensity per frame
    norm_intensity = df_intensity.copy().astype(float)
    for i in range(norm_intensity.shape[0]):
        row = norm_intensity.iloc[i, :]
        min_val, max_val = row.min(), row.max()
        if max_val > min_val:
            norm_intensity.iloc[i, :] = 100 * (row - min_val) / (max_val - min_val)
        else:
            norm_intensity.iloc[i, :] = 0
    
    # Compute COM per frame
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
    com_series = pd.Series(com_values, name="COM")
    
    # Compute cycle COMs and average
    cycle_coms = compute_cycle_coms_from_excel(
        com_series,
        starts_flex,
        ends_flex,
        starts_ext,
        ends_ext,
        target_half_len=target_half_len,
    )
    average_cycle = compute_average_cycle(cycle_coms)
    
    # Create video identifier using TYPES
    video_type = TYPES.get(video_number, "unknown")
    video_id = f"{video_number} {video_type}"
    
    return average_cycle, video_id


def plot_multiple_videos_com(video_ids: list, segment_count: int, opt: str, pdf_path: str = None) -> None:
    """
    Plots average COM cycles for multiple videos on the same graph.
    
    Inputs:
        video_ids: List of video numbers to plot
        segment_count: Number of segments (e.g., 16, 64)
        opt: Option for intensity calculation ("total" or "unit")
        pdf_path: Optional path to save PDF. If None, uses default naming.
    """
    plt.figure(figsize=(19, 7))
    
    # Process each video and plot
    cmap = plt.get_cmap('tab10', len(video_ids))
    
    for idx, video_number in enumerate(video_ids):
        try:
            average_cycle, video_label = process_video_com_cycles(video_number, segment_count, opt)
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


# --- Step (5): Compute and plot COM cycles ---
if multiple_videos:
    # Plot multiple videos on same figure
    video_ids_str = '_'.join(map(str, video_ids))
    com_pdf_path = fr"com_cycle_{opt}_{video_ids_str}_N{segment_count}.pdf"
    plot_multiple_videos_com(video_ids, segment_count, opt, pdf_path=com_pdf_path)
else:
    # Single video mode - use already loaded data
    cycle_coms = compute_cycle_coms_from_excel(
        com_series,
        starts_flex,
        ends_flex,
        starts_ext,
        ends_ext,
        target_half_len=target_half_len,
    )
    average_cycle = compute_average_cycle(cycle_coms)
    
    # Create video identifier using TYPES
    video_type = TYPES.get(video_number, "unknown")
    video_id = f"{video_number} {video_type} ({segment_count} segs, {opt})"
    
    # Plot average cycle for current video
    com_pdf_path = fr"com_cycle_{opt}_{video_number}N{segment_count}.pdf"
    plot_average_cycle(average_cycle, video_id=video_id, pdf_path=com_pdf_path)

