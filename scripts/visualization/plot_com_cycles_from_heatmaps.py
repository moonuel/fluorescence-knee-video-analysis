"""scripts/visualization/plot_com_cycles_from_heatmaps.py

Center of Mass (COM) Analysis from Pre-computed Heatmaps

This script loads pre-generated spatiotemporal heatmaps from Excel workbooks in
`figures/spatiotemporal_maps/` and plots center-of-mass (COM) curves for
flexion-extension cycles.

New CLI (manual, filename-driven)
--------------------------------
The input filenames encode the processing context (e.g. total/unit, cycles,
rescaled, etc.). This script therefore accepts Excel filenames directly.

Positional arguments:
  One or more Excel *basenames* located in `figures/spatiotemporal_maps/`.

Flags:
  --list    Print the available heatmap Excel basenames found in
            `figures/spatiotemporal_maps/`.

Examples:
  # List available inputs
  python scripts/visualization/plot_com_cycles_from_heatmaps.py --list

  # Single file
  python scripts/visualization/plot_com_cycles_from_heatmaps.py \
    heatmap_total_cycles_1_2_rescaled_1339N64.xlsx

  # Multiple files (can mix segment counts; plots/CSVs are produced per N group)
  python scripts/visualization/plot_com_cycles_from_heatmaps.py \
    heatmap_total_cycles_1_2_rescaled_1339N64.xlsx \
    heatmap_total_cycles_1_2_rescaled_1342N64.xlsx \
    heatmap_total_cycles_1_2_3_4_5_rescaled_308N64.xlsx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
import os.path
import argparse
import re
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
# INPUT FILE DISCOVERY / PARSING
# =============================================================================

HEATMAP_DIR = os.path.join("figures", "spatiotemporal_maps")


def list_heatmap_excels() -> list[str]:
    """List candidate heatmap Excel basenames in `figures/spatiotemporal_maps/`.

    Notes:
        - Filters out LibreOffice lockfiles like `.~lock.*`.
        - Matches basenames of the form `heatmap_*.xlsx`.

    Returns:
        Sorted list of basenames.
    """
    if not os.path.isdir(HEATMAP_DIR):
        return []

    basenames = []
    for name in os.listdir(HEATMAP_DIR):
        if name.startswith(".~lock."):
            continue
        if not (name.startswith("heatmap_") and name.lower().endswith(".xlsx")):
            continue
        basenames.append(name)

    return sorted(basenames)


def resolve_excel_path(excel_basename: str) -> str:
    """Resolve an Excel basename to an existing path under `figures/spatiotemporal_maps/`.

    The basename must be provided exactly (case-sensitive on Linux/macOS).

    Raises:
        ValueError: If a path is supplied instead of a basename.
        FileNotFoundError: If the resolved file does not exist.
    """
    # Enforce basenames-only (per spec)
    if os.path.basename(excel_basename) != excel_basename or ("/" in excel_basename) or ("\\" in excel_basename):
        raise ValueError(
            "Excel inputs must be basenames located under figures/spatiotemporal_maps/ "
            f"(got: {excel_basename!r})."
        )

    xlsx_path = os.path.join(HEATMAP_DIR, excel_basename)
    if not os.path.isfile(xlsx_path):
        raise FileNotFoundError(
            f"Heatmap Excel file not found: {xlsx_path}\n"
            "Use --list to see available heatmap workbooks."
        )
    return xlsx_path


_HEATMAP_BASENAME_RE = re.compile(
    r"^heatmap_(?P<descriptor>.+)_(?P<video_number>\d+)N(?P<segment_count>\d+)\.xlsx$",
    flags=re.IGNORECASE,
)


def parse_heatmap_basename(excel_basename: str) -> dict:
    """Parse metadata encoded in a heatmap Excel basename.

    Expected pattern (minimum):
        heatmap_<descriptor>_<video_number>N<segment_count>.xlsx

    Examples:
        heatmap_total_1339N64.xlsx
        heatmap_total_cycles_1_2_rescaled_1339N64.xlsx

    Returns:
        dict with keys: video_number (int), segment_count (int), opt (str), descriptor (str)
    """
    m = _HEATMAP_BASENAME_RE.match(excel_basename)
    if not m:
        raise ValueError(
            f"Invalid heatmap filename: {excel_basename!r}. Expected something like 'heatmap_..._1339N64.xlsx'."
        )

    descriptor = m.group("descriptor")
    video_number = int(m.group("video_number"))
    segment_count = int(m.group("segment_count"))

    # Attempt to infer opt from the descriptor's first token.
    first_token = descriptor.split("_")[0]
    opt = first_token if first_token in OPTIONS else "unknown"

    return {
        "video_number": video_number,
        "segment_count": segment_count,
        "opt": opt,
        "descriptor": descriptor,
    }


# =============================================================================
# HELPER FUNCTIONS - DATA LOADING
# =============================================================================

def load_heatmap_excel_path(input_xlsx: str) -> tuple:
    """Load averaged flexion and extension intensity data from a heatmap Excel workbook.

    The workbook is expected to contain sheets:
      - `avg_flexion`
      - `avg_extension`

    Args:
        input_xlsx: Path to an Excel workbook.

    Returns:
        tuple: (avg_flex, avg_ext) - averaged intensity arrays
    """
    if not os.path.isfile(input_xlsx):
        raise FileNotFoundError(f"Heatmap Excel file not found: {input_xlsx}")

    xls = pd.ExcelFile(input_xlsx)
    avg_flex = pd.read_excel(xls, sheet_name="avg_flexion", header=0, index_col=0).values
    avg_ext = pd.read_excel(xls, sheet_name="avg_extension", header=0, index_col=0).values
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


def compute_com_stats(com_series: pd.Series) -> dict:
    """
    Compute mean, standard deviation, and excursion range of COM values.

    NaN values are ignored when computing the statistics.

    Args:
        com_series: pandas Series containing COM values

    Returns:
        Dictionary with keys "mean", "sd", and "range"
    """
    values = com_series.values.astype(float)
    values = values[~np.isnan(values)]  # drop NaNs

    if len(values) == 0:
        return {"mean": np.nan, "sd": np.nan, "range": np.nan}

    mean_val = float(np.mean(values))
    sd_val = float(np.std(values, ddof=1))  # sample SD
    range_val = float(np.max(values) - np.min(values))

    return {"mean": mean_val, "sd": sd_val, "range": range_val}


def compute_oscillation_indices(com_series: pd.Series) -> dict:
    """
    Compute mean absolute frame-to-frame COM change for full cycle, flexion, and extension.

    NaN values are ignored when computing the differences.

    Args:
        com_series: pandas Series containing COM values (sorted by index)

    Returns:
        Dictionary with keys "osc_full", "osc_flex", "osc_ext"
    """
    # Ensure sorted by index (flex < 0, ext >= 0)
    com_series = com_series.sort_index()

    # Split into flexion and extension
    flex_series = com_series[com_series.index < 0]
    ext_series = com_series[com_series.index >= 0]

    def _osc(values: np.ndarray) -> float:
        values = values.astype(float)
        values = values[~np.isnan(values)]
        if values.size < 2:
            return np.nan
        diffs = np.diff(values)
        diffs = diffs[~np.isnan(diffs)]
        if diffs.size == 0:
            return np.nan
        return float(np.mean(np.abs(diffs)))

    # Full-cycle oscillation index
    full_vals = com_series.values
    osc_full = _osc(full_vals)

    # Flexion and extension oscillation indices
    osc_flex = _osc(flex_series.values)
    osc_ext = _osc(ext_series.values)

    return {
        "osc_full": osc_full,
        "osc_flex": osc_flex,
        "osc_ext": osc_ext,
    }


def set_angle_xticks(flex_len: int, ext_len: int) -> None:
    """
    Set custom x-ticks with joint angles for COM plots.

    Flexion phase (indices -flex_len to -1): mapped to 30° to 135°
    Extension phase (indices 0 to ext_len-1): mapped to 135° to 30°

    Args:
        flex_len: Length of flexion phase
        ext_len: Length of extension phase
    """
    flex_tick_angles = [30, 45, 60, 75, 90, 105, 120, 135]
    ext_tick_angles = [135, 120, 105, 90, 75, 60, 45, 30]

    tick_positions = []
    tick_labels = []

    # Flexion ticks: 30° to 135°
    for ang in flex_tick_angles:
        relative_pos = (ang - 30) / 105.0
        frame_idx = -flex_len + relative_pos * flex_len
        tick_positions.append(frame_idx)
        tick_labels.append(str(ang))

    # Extension ticks: 135° to 30°
    for ang in ext_tick_angles:
        relative_pos = (135 - ang) / 105.0
        frame_idx = relative_pos * (ext_len - 1) if ext_len > 1 else 0
        tick_positions.append(frame_idx)
        tick_labels.append(str(ang))

    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Joint Angle (degrees)")


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_single_heatmap_workbook(input_xlsx: str, video_number: int) -> tuple:
    """Process a single heatmap workbook to compute the COM series.

    Args:
        input_xlsx: Path to a heatmap Excel workbook.
        video_number: Video identifier (parsed from the filename).

    Returns:
        tuple: (com_series, video_label)
    """
    # Load averaged intensity data from heatmap Excel
    avg_flex, avg_ext = load_heatmap_excel_path(input_xlsx)

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

    # Set angle-based x-ticks
    flex_len = len(com_series[com_series.index < 0])
    ext_len = len(com_series[com_series.index >= 0])
    set_angle_xticks(flex_len, ext_len)

    # Formatting
    plt.title(f"Average position of fluorescence intensity from Heatmap [{video_id}]")
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


def plot_multiple_heatmaps_com(file_metas: list[dict], segment_count: int, descriptor: str, pdf_path: str = None) -> None:
    """Plot average COM cycles for multiple heatmap workbooks on the same graph.

    The series are temporally aligned by resampling to match the maximum flexion
    and extension lengths across the provided files.

    Args:
        file_metas: List of dicts; each must include keys:
            - xlsx_path
            - basename
            - video_number
        segment_count: Number of segments (N) for this group.
        descriptor: Human-readable descriptor derived from the filename(s).
        pdf_path: Optional path to save PDF.
    """
    plt.figure(figsize=(19, 7))

    # Color map for multiple videos
    cmap = plt.get_cmap('tab10', len(file_metas))

    # First pass: process all videos and collect COM series
    all_com_series = []
    valid_video_labels = []
    valid_file_metas = []

    for meta in file_metas:
        video_number = meta["video_number"]
        try:
            com_series, video_label = process_single_heatmap_workbook(meta["xlsx_path"], video_number)
            all_com_series.append(com_series)
            valid_video_labels.append(video_label)
            valid_file_metas.append(meta)
        except Exception as e:
            print(f"Error processing workbook {meta.get('basename', meta['xlsx_path'])}: {e}")
            continue

    # Find maximum lengths across all videos
    if not all_com_series:
        print("No valid workbooks to plot.")
        return

    flex_lengths = [len(series[series.index < 0]) for series in all_com_series]
    ext_lengths = [len(series[series.index >= 0]) for series in all_com_series]
    max_flex_len = max(flex_lengths)
    max_ext_len = max(ext_lengths)

    print(f"Aligning videos to max flexion length: {max_flex_len}, max extension length: {max_ext_len}")

    # Initialize lists to collect COM statistics and oscillation indices for each video
    stats_rows = []
    osc_rows = []

    # Second pass: resample and plot
    stats_rows = []
    osc_rows = []

    for i, (com_series, video_label, meta) in enumerate(zip(all_com_series, valid_video_labels, valid_file_metas)):
        video_number = meta["video_number"]
        # Resample for temporal alignment
        aligned_com_series = resample_com_series_for_alignment(com_series, max_flex_len, max_ext_len)

        # Use dashed line for aging videos
        linestyle = '--' if TYPES.get(video_number, '') == "aging" else '-'

        plt.plot(
            aligned_com_series.sort_index().index,
            aligned_com_series.sort_index().values,
            label=video_label,
            color=cmap(i),
            linewidth=2,
            linestyle=linestyle,
        )

        # Compute COM statistics on the aligned series
        stats = compute_com_stats(aligned_com_series)

        # Compute oscillation indices on the aligned series
        osc = compute_oscillation_indices(aligned_com_series)

        condition = TYPES.get(video_number, "unknown")
        stats_rows.append({
            "video_id": f"{condition} {video_number}",
            "mean COM": stats["mean"],
            "sd COM": stats["sd"],
            "range COM": stats["range"],
        })

        osc_rows.append({
            "video_id": f"{condition} {video_number}",
            "osc COM full": osc["osc_full"],
            "osc COM flex": osc["osc_flex"],
            "osc COM ext": osc["osc_ext"],
        })

    # Vertical line at midpoint (flexion/extension transition)
    plt.axvline(0, linestyle="--", color='k', linewidth=1)

    # Set angle-based x-ticks (using aligned lengths)
    set_angle_xticks(max_flex_len, max_ext_len)

    # Formatting
    plt.title(
        f"Average position of fluorescence intensity from Heatmaps ({descriptor}) - Multiple Videos (Temporally Aligned)"
    )
    plt.ylabel("Segment number (JC to SB)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save to PDF if path provided
    if pdf_path is not None:
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(plt.gcf())
        print(f"Exported multi-video heatmap COM plot to: {pdf_path}")

    # Build and output COM statistics table
    stats_df = pd.DataFrame(stats_rows, columns=["video_id", "mean COM", "sd COM", "range COM"])

    print("\nCOM statistics (aligned series, flexion + extension):")
    print(stats_df.to_string(index=False))

    # Save statistics to CSV with consistent naming
    video_ids_str = "_".join(str(m["video_number"]) for m in valid_file_metas)
    stats_csv_path = fr"com_stats_from_heatmap_{descriptor}_{video_ids_str}_N{segment_count}.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved COM statistics table to: {stats_csv_path}")

    # Build and output COM oscillation indices table
    osc_df = pd.DataFrame(
        osc_rows,
        columns=["video_id", "osc COM full", "osc COM flex", "osc COM ext"],
    )

    print("\nCOM oscillation indices (aligned series, flexion + extension):")
    print(osc_df.to_string(index=False))

    # Save oscillation indices to CSV with consistent naming
    osc_csv_path = fr"com_osc_from_heatmap_{descriptor}_{video_ids_str}_N{segment_count}.csv"
    osc_df.to_csv(osc_csv_path, index=False)
    print(f"Saved COM oscillation index table to: {osc_csv_path}")

    # Show interactively
    plt.show()


# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def _common_descriptor(file_metas: list[dict]) -> str:
    """Return a descriptor string for a group of file metas.

    If all descriptors match, that descriptor is returned; otherwise returns
    `mixed`.
    """
    descriptors = [m.get("descriptor", "") for m in file_metas if m.get("descriptor")]
    if not descriptors:
        return "unknown"
    if len(set(descriptors)) == 1:
        return descriptors[0]
    return "mixed"


def main(excel_basenames: list[str]) -> None:
    """Main orchestration function for COM heatmap analysis.

    Args:
        excel_basenames: List of Excel basenames (must exist under
            `figures/spatiotemporal_maps/`).
    """
    # Build file metas and group by segment_count (N)
    file_metas = []
    for basename in excel_basenames:
        xlsx_path = resolve_excel_path(basename)
        meta = parse_heatmap_basename(basename)
        meta["basename"] = basename
        meta["xlsx_path"] = xlsx_path
        file_metas.append(meta)

    groups: dict[int, list[dict]] = {}
    for meta in file_metas:
        groups.setdefault(meta["segment_count"], []).append(meta)

    # Process each N group independently
    for segment_count, group_metas in sorted(groups.items(), key=lambda kv: kv[0]):
        descriptor = _common_descriptor(group_metas)

        if len(group_metas) == 1:
            meta = group_metas[0]
            video_number = meta["video_number"]
            com_series, _ = process_single_heatmap_workbook(meta["xlsx_path"], video_number)

            video_type = TYPES.get(video_number, "unknown")
            video_id_extended = f"{video_number} {video_type}"

            pdf_path = f"com_from_heatmap_{descriptor}_{video_number}N{segment_count}.pdf"
            plot_single_video_com(com_series, video_id_extended, pdf_path=pdf_path)
        else:
            video_ids_str = "_".join(str(m["video_number"]) for m in group_metas)
            pdf_path = fr"com_from_heatmap_{descriptor}_{video_ids_str}_N{segment_count}.pdf"
            plot_multiple_heatmaps_com(group_metas, segment_count, descriptor, pdf_path=pdf_path)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot center of mass from pre-computed spatiotemporal heatmaps (Excel workbooks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # List available heatmap Excel inputs
  python {sys.argv[0]} --list

  # Single file
  python {sys.argv[0]} heatmap_total_1339N64.xlsx

  # Multiple files (plots/CSVs are produced per N group)
  python {sys.argv[0]} heatmap_total_cycles_1_2_rescaled_1339N64.xlsx heatmap_total_cycles_1_2_rescaled_1342N64.xlsx

Directory scanned by --list / used for inputs:
  {HEATMAP_DIR}
""",
    )

    parser.add_argument(
        "excel_files",
        nargs="*",
        help=(
            "One or more heatmap Excel basenames located in figures/spatiotemporal_maps/ "
            "(e.g., heatmap_total_1339N64.xlsx)."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available heatmap Excel basenames under figures/spatiotemporal_maps/ and exit.",
    )

    args = parser.parse_args()

    if args.list:
        for name in list_heatmap_excels():
            print(name)
        sys.exit(0)

    if not args.excel_files:
        parser.error("At least one Excel filename must be provided, or use --list.")

    main(args.excel_files)

    # try:
    #     main(args.excel_files)
    # except Exception as e:
    #     print(f"Error: {e}")
