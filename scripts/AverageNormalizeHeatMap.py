import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os.path
from config import TYPES
import scipy as sp

OPTIONS = {"total": "Normalized total intensities, per segment", 
           "unit": "Normalized average intensity per pixel, per segment"}


def rescale_cycle_coms(cycle_coms:pd.DataFrame) -> pd.DataFrame:
    """
    Resamples the flexion and extension parts of a centered cycle_com DataFrame 
    to ensure equal duration. 
    
    Inputs:
    -------
        cycle_coms : pd.DataFrame
            A centered df with rows "Frame Numbers (Relative)" and columns "cycle numbers". 
            Contains NaN for cycles (columns) that are shorter than the others. 
            Flexion parts should end at frame index -1, and extension parts should start at frame index 0,
            so that all cycles are aligned. 

    Outputs:
    --------
        cycle_coms_rescaled : pd.DataFrame
            A centered df with same structure as the input df, but with rescaled columns to resolve all NaN.
        
    Example usage:
    --------------
        
        # Compute COM over video and prepare cycles 
        total_sums, total_counts = dp.compute_sums_nonzeros(masks, video)
        centre_of_mass = compute_centre_of_mass(total_sums)
        cycles =   "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553	554-576	579-609" # 1339 aging
        cycles = parse_cycles(cycles) # Validate and convert to List[list]

        ...

        # Select cycle COMS and rescale
        cycle_coms = compute_cycle_coms(centre_of_mass, cycles)
        cycle_coms_rescaled = rescale_cycle_coms(cycle_coms)          

    """
    breakpoint()
    cycle_coms.sort_index(inplace=True)

    idx_flx = cycle_coms.loc[:-1].index.to_numpy()
    idx_ext = cycle_coms.loc[0:].index.to_numpy()

    flx = cycle_coms.loc[:-1, :]
    ext = cycle_coms.loc[0:, :]

    nfs, ncols = cycle_coms.shape

    # Stretch flexion frames
    flx_stretch = []
    for col in range(ncols):

        # breakpoint()

        # No need to stretch the longest cycle
        if len(flx[col].dropna()) == len(flx[col]): 
            flx_stretch.append(flx[col])
            continue 

        # Remap domain to frames to [0,1]
        x_old = np.linspace(0, 1, len(flx[col].dropna()))
        y_old = flx[col].dropna()

        # Define interpolation func
        f = sp.interpolate.interp1d(x_old, y_old, kind="linear")

        # Resample new values in [0,1]
        x_new = np.linspace(0, 1, len(flx))
        y_new = pd.Series(f(x_new), idx_flx, name=col)

        # Map back to longest frame range 
        flx_stretch.append(pd.Series(y_new, idx_flx))

    flx_stretch = pd.DataFrame(flx_stretch).T

    # Stretch extension frames
    ext_stretch = []
    for col in range(ncols):

        # breakpoint()

        # No need to stretch the longest cycle
        if len(ext[col].dropna()) == len(ext[col]): 
            ext_stretch.append(ext[col])
            continue 

        # Normalize ext frames to [0,1]
        x_old = np.linspace(0, 1, len(ext[col].dropna()))
        y_old = ext[col].dropna()

        # Define interpolation func
        f = sp.interpolate.interp1d(x_old, y_old, kind="linear")

        # Resample new values in [0,1]
        x_new = np.linspace(0, 1, len(ext))
        y_new = pd.Series(f(x_new), idx_ext, name=col)

        # Map back to longest frame range 
        ext_stretch.append(pd.Series(y_new, idx_ext))

    ext_stretch = pd.DataFrame(ext_stretch).T

    # Stitch together stretched cycles
    cycle_coms_stretch = pd.concat([flx_stretch, ext_stretch], axis=0)

    return cycle_coms_stretch


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
# avg_duration = int(np.round(np.mean(np.concatenate([len_flex, len_ext])))) 

# TODO: verify we rescale the cycles using Dr. Na's code correctly
# TODO: rescale the average com cycles to 50:50 and plot it on the 50:50 heatmap

max_len_flex = max(ends_flex - starts_flex + 1)
max_len_ext = max(ends_ext - starts_ext + 1)

rescaled_flex_all_50 = []
for s, e in zip(starts_flex, ends_flex):
    data = norm_intensity.iloc[s:e+1, :].to_numpy()
    old_x = np.linspace(0, 1, data.shape[0])
    new_x = np.linspace(0, 1, max_len_flex)
    rescaled = np.array([
        np.interp(new_x, old_x, data[:, j]) for j in range(data.shape[1])
    ]).T
    rescaled_flex_all_50.append(rescaled)

rescaled_ext_all_50 = []
for s, e in zip(starts_ext, ends_ext):
    data = norm_intensity.iloc[s:e+1, :].to_numpy()
    old_x = np.linspace(0, 1, data.shape[0])
    new_x = np.linspace(0, 1, max_len_ext)
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
com_series.index += 1 # 1-index shift

flex_slices = []
ext_slices = []

# Collect all flexion slices, rescale index so last frame of flexion is -1
for s, e in zip(starts_flex, ends_flex):
    sl = com_series.iloc[s:e+1].reset_index(drop=True)
    sl.index = sl.index - (len(sl))
    flex_slices.append(sl)

# Collect all extension slices, rescale index so first frame is 0
for s, e in zip(starts_ext, ends_ext):
    sl = com_series.iloc[s:e+1].reset_index(drop=True)
    ext_slices.append(sl)

# Concat all cycles: flex and ext
com_cycles_df = pd.concat([pd.DataFrame(flex_slices), pd.DataFrame(ext_slices)], axis=1).T
com_cycles_df.columns = [i for i in range(com_cycles_df.shape[1])] 



cycle_coms_rescaled = rescale_cycle_coms(com_cycles_df)

avg_com_cycles = cycle_coms_rescaled.mean(axis=1)

breakpoint()

## --- Step (5): Find the peak value for each frame---
#Oliver please finish this step. The peak intensity value will form a contuour line to indicates how the peak intensity move.
# We will compare it with COM curve, then modify COM definition


# --- Step (6): Save to Excel ---
#Oliver, you need to modify the following to save more information such as angles (horizontal axis of heatmap)
##  and the peak "intensity" value of each frame
excel_path = fr"../figures/spatiotemporal_maps/heatmap_{opt}_{video_number}N{segment_count}.xlsx"
pdf_path = fr"../figures/spatiotemporal_maps/heatmap_{opt}_{video_number}N{segment_count}.pdf"

with pd.ExcelWriter(excel_path) as writer:
    norm_intensity.to_excel(writer, sheet_name="normalized_frames", index=False, header=False)
    pd.DataFrame(avg_flex).to_excel(writer, sheet_name="avg_flexion", index=False, header=False)
    pd.DataFrame(avg_ext).to_excel(writer, sheet_name="avg_extension", index=False, header=False)

# --- Step (7): Plot heatmap ---
with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    combined = np.concatenate([avg_flex, avg_ext], axis=0)
    im = ax.imshow(combined.T, aspect="auto", cmap="viridis", origin="lower")

    # Plot avg_com_cycles as a solid red line over the heatmap
    # avg_com_cycles' index should correspond to the x-axis of the combined heatmap (avg_flex + avg_ext)
    # It must be mapped so that its values (segment indices) align with the heatmap row axis (vertical)
    ax.plot(np.arange(len(avg_com_cycles)), avg_com_cycles, color='red', linewidth=2, label='Average COM')
    ax.legend(loc='upper right')
    
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

