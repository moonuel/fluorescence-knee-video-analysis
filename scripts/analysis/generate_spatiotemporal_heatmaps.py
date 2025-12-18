"""
Data analysis and visualization script for computing normalized averaged heatmaps from generated Excel files. 
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
import os.path
import argparse
from config import TYPES
from config.knee_metadata import KNEE_VIDEOS
import scipy as sp
from pathlib import Path

# Get project root directory for robust path handling
PROJECT_ROOT = Path(__file__).parent.parent.parent

OPTIONS = {
    "total": "Normalized total intensities, per segment",
    "unit": "Normalized average intensity per pixel, per segment"
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_arguments() -> tuple[argparse.Namespace, bool, list[int] | None, bool]:
    """
    Parse command line arguments and return args namespace, normalize flag, cycle indices, and rescale flag.

    Returns:
        Tuple of (parsed_args, normalize_flag, cycle_indices, rescale_50_50_flag) where:
        - normalize_flag is True for normalization enabled
        - cycle_indices is list of 1-based cycle numbers or None for all cycles
        - rescale_flag is True to enable 50:50 rescaled heatmaps (default)
    """
    parser = argparse.ArgumentParser(
        description="Generate spatiotemporal heatmaps from knee video intensity data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python {sys.argv[0]} normal 1339 64 total                    # Default: with normalization, all cycles
  python {sys.argv[0]} aging 1339 64 total --no-normalize      # Without normalization
  python {sys.argv[0]} normal 0308 64 total --cycles 1,3       # Only cycles 1 and 3

Valid video types are: {list(TYPES)}
Options for the fourth argument are:
""" + "\n".join(f"  '{k}': {v}" for k, v in OPTIONS.items())
    )

    # Extract unique conditions from KNEE_VIDEOS metadata
    available_conditions = sorted(set(condition for condition, _, _ in KNEE_VIDEOS.keys()))
    parser.add_argument("condition", choices=available_conditions, help="Condition (e.g. normal, aging, dmm-0w)")
    parser.add_argument("video_number", type=int, help="Video identifier number")
    parser.add_argument("segment_count", type=int, help="Number of segments")
    parser.add_argument("opt", choices=OPTIONS.keys(),
                       help="Processing option")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Disable intensity normalization (default: normalization enabled)")
    parser.add_argument(
        "--cycles",
        type=str,
        default=None,
        help=(
            "Comma-separated 1-based cycle numbers to include, e.g. '1,3'. "
            "If omitted, all cycles in the Excel file are used."
        ),
    )

    # 50:50 rescale option (default: enabled)
    rescale_group = parser.add_mutually_exclusive_group()
    rescale_group.add_argument(
        "--rescale", dest="rescale", action="store_true",
        help="Enable 50:50 rescaled heatmaps (default)"
    )
    rescale_group.add_argument(
        "--no-rescale", dest="rescale", action="store_false",
        help="Disable 50:50 rescaled heatmaps"
    )
    parser.set_defaults(rescale_50_50=True)

    args = parser.parse_args()

    # Set normalize flag (default True, inverted by --no-normalize)
    normalize = not args.no_normalize

    # Parse cycle indices (1-based)
    cycle_indices = None
    if args.cycles is not None:
        try:
            cycle_indices = [int(x.strip()) for x in args.cycles.split(",") if x.strip()]
        except ValueError:
            parser.error("--cycles must be a comma-separated list of integers, e.g. '1,3'")

    return args, normalize, cycle_indices, args.rescale_50_50


def validate_input_file(video_number: int, segment_count: int) -> str:
    """
    Validate that the input Excel file exists and return its path.

    Args:
        video_number: Video identifier number
        segment_count: Number of segments

    Returns:
        Path to the validated input Excel file

    Exits the program if the file doesn't exist.
    """
    input_xlsx = PROJECT_ROOT / "data" / "intensities_total" / f"{video_number}N{segment_count}intensities.xlsx"
    if not input_xlsx.exists():
        print(f"Error: File '{video_number}N{segment_count}intensities.xlsx' doesn't exist.")
        print(f"       Is video_number={video_number} and segment_count={segment_count} correct?")
        sys.exit(1)
    return str(input_xlsx)


def clean_intervals(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean and extract start/end intervals from an Excel sheet.
    
    Args:
        df: Raw DataFrame from Excel sheet containing interval data
        
    Returns:
        Tuple of (starts, ends) as numpy arrays
    """
    df = df.dropna(how="all")
    first_row = df.iloc[0, :].astype(str).str.lower()
    if ("start" in first_row.values) and ("end" in first_row.values):
        df = df.iloc[1:, :]
    df = df.iloc[:, 0:3]
    starts = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().astype(int).to_numpy()
    ends = pd.to_numeric(df.iloc[:, 2], errors="coerce").dropna().astype(int).to_numpy()
    return starts, ends


def load_and_clean_data(input_xlsx: str, opt: str) -> tuple:
    """
    Load Excel file and clean data sheets.
    
    Args:
        input_xlsx: Path to Excel file
        opt: Option for processing ("total" or "unit")
        
    Returns:
        Tuple of (df_intensity, starts_flex, ends_flex, starts_ext, ends_ext)
    """
    # Load sheets
    xls = pd.ExcelFile(input_xlsx)
    df_intensity_raw = pd.read_excel(xls, sheet_name="Segment Intensities", header=None)
    df_num_pixels_raw = pd.read_excel(xls, sheet_name="Number of Mask Pixels", header=None)
    df_flex = pd.read_excel(xls, sheet_name="Flexion Frames", header=None)
    df_ext = pd.read_excel(xls, sheet_name="Extension Frames", header=None)

    # Clean intensity data
    df_intensity = df_intensity_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    df_num_pixels = df_num_pixels_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    # Optional: take average intensity per pixel, per segment
    if opt == "total": 
        pass
    elif opt == "unit":
        df_intensity = df_intensity / df_num_pixels
        df_intensity.fillna(0, inplace=True)

    # Extract intervals
    starts_flex, ends_flex = clean_intervals(df_flex)
    starts_ext, ends_ext = clean_intervals(df_ext)
    
    return df_intensity, starts_flex, ends_flex, starts_ext, ends_ext


def normalize_intensity_per_frame(df_intensity: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize intensity values per frame to 0-100 scale.
    
    Args:
        df_intensity: DataFrame with raw intensity values
        
    Returns:
        Normalized intensity DataFrame
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


def rescale_phase_data(norm_intensity: pd.DataFrame, starts: np.ndarray, ends: np.ndarray) -> tuple:
    """
    Rescale all cycles in a phase (flexion or extension) to the same length.
    
    Args:
        norm_intensity: Normalized intensity DataFrame
        starts: Array of start frame indices
        ends: Array of end frame indices
        
    Returns:
        Tuple of (rescaled_all, average_rescaled)
    """
    max_len = max(ends - starts + 1)
    rescaled_all = []
    
    for s, e in zip(starts, ends):
        data = norm_intensity.iloc[s:e+1, :].to_numpy()
        old_x = np.linspace(0, 1, data.shape[0])
        new_x = np.linspace(0, 1, max_len)
        rescaled = np.array([np.interp(new_x, old_x, data[:, j]) for j in range(data.shape[1])]).T
        rescaled_all.append(rescaled)
    
    avg_rescaled = np.mean(rescaled_all, axis=0)
    return rescaled_all, avg_rescaled


def rescale_to_equal_duration(avg_flex: np.ndarray, avg_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rescale flexion and extension phases to equal duration (50:50 split).
    
    This function takes the averaged flexion and extension data and rescales both
    to have the same length, equal to the maximum of the two original lengths.
    This creates a true 50:50 temporal split between the two phases.
    
    Args:
        avg_flex: Average flexion data array with shape (n_flex_frames, n_segments)
        avg_ext: Average extension data array with shape (n_ext_frames, n_segments)
        
    Returns:
        Tuple of (avg_flex_50, avg_ext_50) both rescaled to equal duration
    """
    n_flex, n_segments = avg_flex.shape
    n_ext, _ = avg_ext.shape
    
    # Find the maximum duration between flexion and extension
    max_duration = max(n_flex, n_ext)
    
    # Rescale flexion to max_duration
    old_x_flex = np.linspace(0, 1, n_flex)
    new_x = np.linspace(0, 1, max_duration)
    avg_flex_50 = np.array([np.interp(new_x, old_x_flex, avg_flex[:, j]) 
                             for j in range(n_segments)]).T
    
    # Rescale extension to max_duration
    old_x_ext = np.linspace(0, 1, n_ext)
    avg_ext_50 = np.array([np.interp(new_x, old_x_ext, avg_ext[:, j]) 
                            for j in range(n_segments)]).T
    
    return avg_flex_50, avg_ext_50


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

    Args:
        com_flex: COM values for flexion phase
        com_ext: COM values for extension phase

    Returns:
        Combined pandas Series spanning both phases
    """
    combined_com = np.concatenate([com_flex, com_ext])
    return pd.Series(combined_com, index=np.arange(len(combined_com)), name="COM")





def plot_heatmap(avg_flex: np.ndarray, avg_ext: np.ndarray, avg_com_cycles: pd.Series,
                 pdf_path: str, title_prefix: str = "", title_suffix: str = "", normalize: bool = True, selection_label: str = "") -> None:
    """
    Plot and save heatmap with COM overlay to PDF.

    Args:
        avg_flex: Average flexion data
        avg_ext: Average extension data
        avg_com_cycles: Average COM cycle data
        pdf_path: Path to save PDF
        title_suffix: Optional suffix for plot title
        normalize: Whether data is normalized (affects title)
        selection_label: Optional label describing cycle selection for COM legend
    """
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        combined = np.concatenate([avg_flex, avg_ext], axis=0)
        im = ax.imshow(combined.T, aspect="auto", cmap="viridis", origin="lower")

        # Plot avg_com_cycles as a solid red line over the heatmap
        ax.plot(np.arange(len(avg_com_cycles)), avg_com_cycles,
                color='red', linewidth=2, label="Average COM")
        ax.legend(loc='upper right')

        # Add cycle information as text annotation in bottom-left corner
        if selection_label:
            ax.text(
                0.01, 0.01,
                selection_label,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=3),
            )
        
        # Flexion/Extension split
        split_index = avg_flex.shape[0]
        ax.axvline(x=split_index, color="white", linestyle="--", linewidth=1.5)
        
        # Define angle mappings
        flex_labels = np.linspace(30, 130, avg_flex.shape[0])
        ext_labels = np.linspace(135, 30, avg_ext.shape[0])
        
        # Define desired ticks for each phase
        flex_tick_labels = np.arange(30, 131, 15)   # 30 → 130
        ext_tick_labels = np.arange(135, 29, -15)   # 135 → 30

        # Find corresponding indices within each phase
        flex_tick_positions = [np.abs(flex_labels - deg).argmin() for deg in flex_tick_labels]
        ext_tick_positions = split_index + np.array([np.abs(ext_labels - deg).argmin() 
                                                       for deg in ext_tick_labels])

        # Combine both
        tick_positions = np.concatenate([flex_tick_positions, ext_tick_positions])
        tick_labels = [f"{d}" for d in np.concatenate([flex_tick_labels, ext_tick_labels])]

        # Apply ticks and labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Joint Angle (degrees)")

        # Titles and colorbar
        intensity_label = "Avg Normalized Intensity (%)" if normalize else "Avg Raw Intensity"
        title = f"{title_prefix} – Averaged {'Normalized ' if normalize else 'Raw '}Intensity Heatmap"
        if title_suffix:
            title += f" {title_suffix}"
        ax.set_title(title)
        ax.set_ylabel("Segment Index")
        plt.colorbar(im, ax=ax, label=intensity_label)
        
        pdf.savefig(fig)
        plt.close(fig)


def save_results_to_excel(excel_path: str, intensity_data: pd.DataFrame,
                          avg_flex: np.ndarray, avg_ext: np.ndarray, normalize: bool = True) -> None:
    """
    Save results to Excel file.

    Args:
        excel_path: Path to save Excel file
        intensity_data: Intensity DataFrame (normalized or raw)
        avg_flex: Average flexion data
        avg_ext: Average extension data
        normalize: Whether data is normalized (affects sheet name)
    """
    sheet_name = "normalized_frames" if normalize else "raw_frames"
    with pd.ExcelWriter(excel_path) as writer:
        intensity_data.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        pd.DataFrame(avg_flex).to_excel(writer, sheet_name="avg_flexion", index=False, header=False)
        pd.DataFrame(avg_ext).to_excel(writer, sheet_name="avg_extension", index=False, header=False)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(condition: str, video_number: int, segment_count: int, opt: str, input_xlsx: str, normalize: bool = True, cycle_indices: list[int] | None = None, rescale_50_50: bool = True) -> None:
    """
    Main processing function for generating spatiotemporal heatmaps.

    Args:
        video_number: Video identifier number
        segment_count: Number of segments
        opt: Processing option ("total" or "unit")
        input_xlsx: Path to the input Excel file
        normalize: Whether to normalize intensity values (default: True)
        cycle_indices: List of 1-based cycle numbers to include, or None for all cycles
        rescale: Whether to generate 50:50 rescaled heatmaps (default: True)
    """

    # Create filename suffix for non-normalized data
    norm_suffix = "_nonorm" if not normalize else ""

    # Initialize cycle selection tracking
    selected_cycle_indices_1based: list[int] | None = None
    selected_flex_ranges: list[tuple[int, int]] | None = None
    selected_ext_ranges: list[tuple[int, int]] | None = None

    # Load and clean data
    df_intensity, starts_flex, ends_flex, starts_ext, ends_ext = load_and_clean_data(input_xlsx, opt)

    # Apply cycle selection if requested
    num_cycles_flex = len(starts_flex)
    num_cycles_ext = len(starts_ext)
    if num_cycles_flex != num_cycles_ext:
        raise ValueError(
            f"Mismatch between flexion cycles ({num_cycles_flex}) and extension cycles ({num_cycles_ext})"
        )
    num_cycles = num_cycles_flex

    if cycle_indices is not None:
        # 1-based → 0-based
        idx_zero_based = []
        for c in cycle_indices:
            if c < 1 or c > num_cycles:
                raise ValueError(
                    f"Requested cycle {c} is out of range; file has {num_cycles} cycles (1-based)"
                )
            idx_zero_based.append(c - 1)

        idx = np.array(idx_zero_based, dtype=int)

        starts_flex = starts_flex[idx]
        ends_flex   = ends_flex[idx]
        starts_ext  = starts_ext[idx]
        ends_ext    = ends_ext[idx]

        selected_cycle_indices_1based = cycle_indices
        selected_flex_ranges = list(zip(starts_flex, ends_flex))
        selected_ext_ranges  = list(zip(starts_ext, ends_ext))

        print(f"Using cycles (1-based): {cycle_indices}")
        print(f"Total cycles available in Excel: {num_cycles}")
        print(f"Flexion frame ranges (1-based): {selected_flex_ranges}")
        print(f"Extension frame ranges (1-based): {selected_ext_ranges}")
    else:
        # When using all cycles, still build the tracking structures for legend
        selected_cycle_indices_1based = list(range(1, num_cycles + 1))
        selected_flex_ranges = list(zip(starts_flex, ends_flex))
        selected_ext_ranges  = list(zip(starts_ext, ends_ext))

        print(f"Using all {num_cycles} cycles found in Excel.")
        print(f"Flexion frame ranges (1-based): {selected_flex_ranges}")
        print(f"Extension frame ranges (1-based): {selected_ext_ranges}")

    # Create cycle suffix for filenames when subset selected
    cycles_suffix = ""
    if selected_cycle_indices_1based is not None:
        cycles_suffix = "_cycles_" + "_".join(str(c) for c in selected_cycle_indices_1based)

    # Normalize intensity per frame (conditionally)
    if normalize:
        norm_intensity = normalize_intensity_per_frame(df_intensity)
    else:
        norm_intensity = df_intensity.copy()

    # Rescale flexion and extension phases (each to their own max length within the phase)
    rescaled_flex_all, avg_flex = rescale_phase_data(norm_intensity, starts_flex, ends_flex)
    rescaled_ext_all, avg_ext = rescale_phase_data(norm_intensity, starts_ext, ends_ext)

    # Rescale with 50:50 duration (both phases rescaled to equal length = max of the two)
    avg_flex_50, avg_ext_50 = rescale_to_equal_duration(avg_flex, avg_ext)

    # Compute COM directly from averaged intensity arrays (improved temporal alignment)
    com_flex = compute_com_from_intensity_array(avg_flex)
    com_ext = compute_com_from_intensity_array(avg_ext)
    com_flex_50 = compute_com_from_intensity_array(avg_flex_50)
    com_ext_50 = compute_com_from_intensity_array(avg_ext_50)

    # Create combined COM series for plotting
    avg_com_cycles = create_combined_com_series(com_flex, com_ext)
    avg_com_cycles_50_50 = create_combined_com_series(com_flex_50, com_ext_50)

    # TODO: Step (5): Find the peak value for each frame
    # Oliver please finish this step. The peak intensity value will form a contour line to indicate how the peak intensity moves.
    # We will compare it with COM curve, then modify COM definition

    # Build selection label for plots
    selection_label = ""
    if selected_cycle_indices_1based is not None:
        cycle_summaries = []
        for c, (fs, fe), (es, ee) in zip(selected_cycle_indices_1based, selected_flex_ranges, selected_ext_ranges):
            cycle_summaries.append(f"Cycle {c}: {fs}-{fe}, {es}-{ee}")
        selection_label = "\n".join(cycle_summaries)

    # Create title prefix for plots
    title_prefix = f"{condition} {video_number} (N={segment_count})"

    # Ensure output directory exists
    output_dir = PROJECT_ROOT / "figures" / "spatiotemporal_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    if rescale:
        # Save results - 50/50 rescaled heatmap
        excel_path_50 = str(output_dir / f"heatmap_{opt}{norm_suffix}{cycles_suffix}_rescaled_{video_number}N{segment_count}.xlsx")
        pdf_path_50 = str(output_dir / f"heatmap_{opt}{norm_suffix}{cycles_suffix}_rescaled_{video_number}N{segment_count}.pdf")

        save_results_to_excel(excel_path_50, norm_intensity, avg_flex_50, avg_ext_50, normalize=normalize)
        plot_heatmap(avg_flex_50, avg_ext_50, avg_com_cycles_50_50, pdf_path_50, title_prefix=title_prefix, title_suffix="(Temporally Rescaled)", normalize=normalize, selection_label=selection_label)

        print("Exported:", excel_path_50, pdf_path_50)
    else:
        # Save results - original heatmap
        print("Skipping 50:50 rescaled heatmap outputs (per --no-rescale)")
        excel_path = str(output_dir / f"heatmap_{opt}{norm_suffix}{cycles_suffix}_{video_number}N{segment_count}.xlsx")
        pdf_path = str(output_dir / f"heatmap_{opt}{norm_suffix}{cycles_suffix}_{video_number}N{segment_count}.pdf")

        save_results_to_excel(excel_path, norm_intensity, avg_flex, avg_ext, normalize=normalize)
        plot_heatmap(avg_flex, avg_ext, avg_com_cycles, pdf_path, title_prefix=title_prefix, normalize=normalize, selection_label=selection_label)

        print("Exported:", excel_path, pdf_path)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    args, normalize, cycle_indices, rescale = parse_arguments()

    # Validate input file exists
    input_xlsx = validate_input_file(args.video_number, args.segment_count)

    # Run main processing
    main(args.condition, args.video_number, args.segment_count, args.opt, input_xlsx, normalize, cycle_indices, rescale)


# ============================================================================
# COMMENTED OUT CODE - Alternative COM calculation approach
# ============================================================================

# The following part is used to calculate averaged COM curves based on normalized intensity
# 
# # --- Step (1): Normalize intensity per frame ---
# norm_intensity = df_intensity.copy()
# for i in range(norm_intensity.shape[0]):
#       row = norm_intensity.iloc[i, :]
#       min_val, max_val = row.min(), row.max()
#       if max_val > min_val:
#           norm_intensity.iloc[i, :] = 100 * (row - min_val) / (max_val - min_val)
#       else:
#           norm_intensity.iloc[i, :] = 0


# # --- Step (2): Compute COM per frame ---
# segments = np.arange(1, norm_intensity.shape[1] + 1)
# com_values = []
# for i in range(norm_intensity.shape[0]):
#     intensities = norm_intensity.iloc[i, :].to_numpy()
#     total = np.sum(intensities)
#     if total > 0:
#         com = np.sum(segments * intensities) / total
#     else:
#         com = np.nan
#     com_values.append(com)
# com_series = pd.Series(com_values, name="COM")

# # --- Step (3): Rescale COMs within each cycle ---
# def rescale_cycles(starts, ends, com_series):
#     n_frames = len(com_series)
#     max_len = max(ends - starts + 1)
#     rescaled_cycles = []
#     for s, e in zip(starts, ends):
#         s = int(max(0, min(s, n_frames - 1)))
#         e = int(max(0, min(e, n_frames - 1)))
#         if e <= s:
#             continue
#         segment = com_series.iloc[s:e+1].to_numpy()
#         if segment.size == 0 or np.all(np.isnan(segment)):
#             continue
#         old_x = np.linspace(0, 1, segment.size)
#         new_x = np.linspace(0, 1, max_len)
#         rescaled = np.interp(new_x, old_x, segment)
#         rescaled_cycles.append(rescaled)
#     if rescaled_cycles:
#         return np.mean(rescaled_cycles, axis=0), max_len
#     else:
#         return np.array([]), 0

# avg_flex, len_flex = rescale_cycles(starts_flex, ends_flex, com_series)
# avg_ext, len_ext = rescale_cycles(starts_ext, ends_ext, com_series)
