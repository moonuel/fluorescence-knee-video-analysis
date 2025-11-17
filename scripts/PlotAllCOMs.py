import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os.path
from config import TYPES

OPTIONS = {"total": "Normalized total intensities, per segment", 
           "unit": "Normalized average intensity per pixel, per segment"}

# --- Input validation  ---
if len(sys.argv[1:]) != 3 or sys.argv[3] not in OPTIONS.keys(): 
    options_str = "\n\t" + "\n\t".join(f"     '{k}': {v}" for k, v in OPTIONS.items())
    raise SyntaxError(
        f"\n\tExample usage: {sys.argv[0]} 1339 64 total"
        f"\n\tValid types are: {list(TYPES)}"
        f"\n\tOptions for the third argument are:{options_str}"
    )

# if not sys.argv[3] in OPTIONS: raise SyntaxError(f"")

video_number = int(sys.argv[1])
segment_count = int(sys.argv[2])
opt = sys.argv[3]
# video_number = 1339
# segment_count = 64
# num_cycles = 4

INPUT_XLSX = fr"../data/video_intensities/{video_number}N{segment_count}intensities.xlsx"  

if not os.path.isfile(INPUT_XLSX): raise ValueError(f"File '{video_number}N{segment_count}intensities.xlsx' doesn't exist. \n\t    Is {video_number=} and {segment_count=} correct?")

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


# --- COM Cycle Functions ---

def compute_cycle_coms_from_excel(com_series: pd.Series, 
                                   starts_flex: np.ndarray, 
                                   ends_flex: np.ndarray,
                                   starts_ext: np.ndarray, 
                                   ends_ext: np.ndarray) -> pd.DataFrame:
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


breakpoint()

# --- Step (5): Compute and plot COM cycles ---
# Compute cycle COMs from current video data
cycle_coms = compute_cycle_coms_from_excel(com_series, starts_flex, ends_flex, starts_ext, ends_ext)
average_cycle = compute_average_cycle(cycle_coms)

# Create video identifier using TYPES
video_type = TYPES.get(video_number, "unknown")
video_id = f"{video_number} {video_type} ({segment_count} segs, {opt})"

# Plot average cycle for current video
com_pdf_path = fr"com_cycle_{opt}_{video_number}N{segment_count}.pdf"
plot_average_cycle(average_cycle, video_id=video_id, pdf_path=com_pdf_path)

## --- Step (5): Find the peak value for each frame---
#Oliver please finish this step. The peak intensity value will form a contuour line to indicates how the peak intensity move.
# We will compare it with COM curve, then modify COM definition


# --- Step (6): Save to Excel ---
#Oliver, you need to modify the following to save more information such as angles (horizontal axis of heatmap)
##  and the peak "intensity" value of each frame
excel_path = fr"test_coms_{opt}_{video_number}N{segment_count}.xlsx"
pdf_path = fr"test_coms_{opt}_{video_number}N{segment_count}.pdf"

with pd.ExcelWriter(excel_path) as writer:
    norm_intensity.to_excel(writer, sheet_name="normalized_frames", index=False, header=False)
    pd.DataFrame(avg_flex).to_excel(writer, sheet_name="avg_flexion", index=False, header=False)
    pd.DataFrame(avg_ext).to_excel(writer, sheet_name="avg_extension", index=False, header=False)

# --- Step (7): Plot heatmap ---
with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    combined = np.concatenate([avg_flex, avg_ext], axis=0)
    im = ax.imshow(combined.T, aspect="auto", cmap="viridis", origin="lower")
    
    # --- Flexion/Extension split ---
    split_index = avg_flex.shape[0]
    ax.axvline(x=split_index - 0.5, color="white", linestyle="--", linewidth=1.5)
    
    # --- Define angle mappings ---
    flex_labels = np.linspace(30, 130, avg_flex.shape[0])
    ext_labels = np.linspace(135, 30, avg_ext.shape[0])
    joint_angles = np.concatenate([flex_labels, ext_labels])
    
    # --- Define desired ticks for each phase ---
    flex_tick_labels = np.arange(30, 131, 15)   # 30 → 130
    ext_tick_labels  = np.arange(135, 29, -15)  # 135 → 30

    # Find corresponding indices within each phase
    flex_tick_positions = [np.abs(flex_labels - deg).argmin() for deg in flex_tick_labels]
    ext_tick_positions  = split_index + np.array([np.abs(ext_labels - deg).argmin() for deg in ext_tick_labels])

    # Combine both
    tick_positions = np.concatenate([flex_tick_positions, ext_tick_positions])
    tick_labels = [f"{d}" for d in np.concatenate([flex_tick_labels, ext_tick_labels])]

    # --- Apply ticks and labels ---
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Joint Angle (degrees)")

    # --- Titles and colorbar ---
    ax.set_title("Averaged Normalized Intensity: Flexion (Left) | Extension (Right)")
    ax.set_ylabel("Segment Index")
    plt.colorbar(im, ax=ax, label="Avg Normalized Intensity (%)")
    
    pdf.savefig(fig)
    plt.close(fig)

print("Exported:", excel_path, pdf_path)

# --- Step (7.1): Plot 50/50 heatmap ---

excel_path_50 = fr"../figures/spatiotemporal_maps/heatmap_{opt}_rescaled_{video_number}N{segment_count}.xlsx"
pdf_path_50 = fr"../figures/spatiotemporal_maps/heatmap_{opt}_rescaled_{video_number}N{segment_count}.pdf"

with pd.ExcelWriter(excel_path_50) as writer:
    norm_intensity.to_excel(writer, sheet_name="normalized_frames", index=False, header=False)
    pd.DataFrame(avg_flex_50).to_excel(writer, sheet_name="avg_flexion", index=False, header=False)
    pd.DataFrame(avg_ext_50).to_excel(writer, sheet_name="avg_extension", index=False, header=False)

with PdfPages(pdf_path_50) as pdf:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    combined = np.concatenate([avg_flex_50, avg_ext_50], axis=0)
    im = ax.imshow(combined.T, aspect="auto", cmap="viridis", origin="lower")
    
    # --- Flexion/Extension split ---
    split_index = avg_flex_50.shape[0]
    ax.axvline(x=split_index - 0.5, color="white", linestyle="--", linewidth=1.5)

    # --- Define angle mapping ---
    flex_labels = np.linspace(30, 130, avg_flex_50.shape[0])
    ext_labels = np.linspace(135, 30, avg_ext_50.shape[0])
    joint_angles = np.concatenate([flex_labels, ext_labels])

    # --- Define tick labels for both halves ---
    # flexion: 30 → 130; extension: 135 → 30
    flex_ticks = np.arange(30, 131, 15)
    ext_ticks = np.arange(135, 29, -15)

    # Find corresponding indices within each phase
    flex_tick_positions = [np.abs(flex_labels - deg).argmin() for deg in flex_tick_labels]
    ext_tick_positions  = split_index + np.array([np.abs(ext_labels - deg).argmin() for deg in ext_tick_labels])

    # Combine both
    tick_positions = np.concatenate([flex_tick_positions, ext_tick_positions])
    tick_labels = [f"{d}" for d in np.concatenate([flex_tick_labels, ext_tick_labels])]

    # --- Apply ticks and labels ---
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Joint Angle (degrees)")

    # --- Titles and colorbar ---
    ax.set_title("Averaged Normalized Intensity: Flexion (Left) | Extension (Right)")
    ax.set_ylabel("Segment Index")
    plt.colorbar(im, ax=ax, label="Avg Normalized Intensity (%)")

    pdf.savefig(fig)
    plt.close(fig)

print("Exported:", excel_path_50, pdf_path_50)

#~~~~~~~~~~ The following part is used to calculate averaged COM curves based on normalized intensity
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

