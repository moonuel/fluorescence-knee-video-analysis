import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Input file ---
video_number = 308
segment_count = 64
num_cycles = 4

INPUT_XLSX = fr"C:\research\kneevideo\me\Heatmap\video{video_number}N{segment_count}.xlsx"  
#Oliver, you shall change the directory path

# --- Load sheets ---
xls = pd.ExcelFile(INPUT_XLSX)
df_intensity_raw = pd.read_excel(xls, sheet_name=0, header=None)
df_flex = pd.read_excel(xls, sheet_name=1, header=None)
df_ext = pd.read_excel(xls, sheet_name=2, header=None)

# --- Clean intensity data ---
# Skip the first row (header: "Frame", "Segment 1", ..., "Segment 16")
df_intensity = df_intensity_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

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
norm_intensity = df_intensity.copy()
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
#Oliver please finish this step


## --- Step (5): Find the peak value for each frame---
#Oliver please finish this step. The peak intensity value will form a contuour line to indicates how the peak intensity move.
# We will compare it with COM curve, then modify COM definition


# --- Step (6): Save to Excel ---
#Oliver, you need to modify the following to save more information such as angles (horizontal axis of heatmap)
##  and the peak "intensity" value of each frame
excel_path = fr"C:\research\kneevideo\me\Heatmap\video{video_number}N{segment_count}_results.xlsx"
with pd.ExcelWriter(excel_path) as writer:
    norm_intensity.to_excel(writer, sheet_name="normalized_frames", index=False, header=False)
    pd.DataFrame(avg_flex).to_excel(writer, sheet_name="avg_flexion", index=False, header=False)
    pd.DataFrame(avg_ext).to_excel(writer, sheet_name="avg_extension", index=False, header=False)

# --- Step (7): Plot heatmap ---
pdf_path = fr"C:\research\kneevideo\me\Heatmap\video{video_number}N{segment_count}_heatmap.pdf"
with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(10, 5))
    combined = np.concatenate([avg_flex, avg_ext], axis=0)
    im = ax.imshow(combined.T, aspect="auto", cmap="viridis", origin="lower")
    ax.axvline(x=avg_flex.shape[0]-0.5, color="white", linestyle="--", linewidth=1.5)
    ax.set_title("Averaged Normalized Intensity: Flexion (Left) | Extension (Right)")
    ax.set_xlabel("Rescaled Frame Index")
    ax.set_ylabel("Segment Index")
    plt.colorbar(im, ax=ax, label="Avg Normalized Intensity (%)")
    pdf.savefig(fig)
    plt.close(fig)

print("Exported:", excel_path, pdf_path)

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

