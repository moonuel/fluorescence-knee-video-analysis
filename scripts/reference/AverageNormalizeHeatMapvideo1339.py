import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Input file ---
INPUT_XLSX = "video1339N64.xlsx"
OUTPUT_BASE = INPUT_XLSX.replace(".xlsx", "")

# --- Load sheets ---
xls = pd.ExcelFile(INPUT_XLSX)
df_intensity_raw = pd.read_excel(xls, sheet_name=0, header=None)
df_flex = pd.read_excel(xls, sheet_name=1, header=None)
df_ext = pd.read_excel(xls, sheet_name=2, header=None)

# --- Clean intensity data ---
# Skip header row ("Frame", "Segment 1", ...)
df_intensity = df_intensity_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

# --- Function to clean flexion/extension interval sheets ---
def clean_intervals(df):
    df = df.dropna(how="all")
    first_row = df.iloc[0, :].astype(str).str.lower()
    if ("start" in first_row.values) and ("end" in first_row.values):
        df = df.iloc[1:, :]  # skip header row
    starts = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().astype(int).to_numpy()
    ends = pd.to_numeric(df.iloc[:, 2], errors="coerce").dropna().astype(int).to_numpy()
    return starts, ends

# --- Extract intervals ---
starts_flex, ends_flex = clean_intervals(df_flex)
starts_ext, ends_ext = clean_intervals(df_ext)

# --- Step (1): Normalize intensity per frame ---
norm_intensity = df_intensity.copy()
for i in range(norm_intensity.shape[0]):
    row = norm_intensity.iloc[i, :]
    min_val, max_val = row.min(), row.max()
    if max_val > min_val:
        norm_intensity.iloc[i, :] = 100 * (row - min_val) / (max_val - min_val)
    else:
        norm_intensity.iloc[i, :] = 0

# --- Step (2): Flexion average ---
n_frames = norm_intensity.shape[0]
max_len_flex = max(ends_flex - starts_flex + 1)
rescaled_flex_all = []
for s, e in zip(starts_flex, ends_flex):
    # Clip to valid range
    s = int(max(0, min(s, n_frames - 1)))
    e = int(max(0, min(e, n_frames - 1)))
    if e <= s:
        continue
    data = norm_intensity.iloc[s:e+1, :].to_numpy()
    if data.size == 0:
        continue
    old_x = np.linspace(0, 1, data.shape[0])
    new_x = np.linspace(0, 1, max_len_flex)
    rescaled = np.array([np.interp(new_x, old_x, data[:, j]) for j in range(data.shape[1])]).T
    rescaled_flex_all.append(rescaled)

avg_flex = np.mean(rescaled_flex_all, axis=0) if rescaled_flex_all else np.zeros((max_len_flex, norm_intensity.shape[1]))

# --- Step (3): Extension average ---
max_len_ext = max(ends_ext - starts_ext + 1)
rescaled_ext_all = []
for s, e in zip(starts_ext, ends_ext):
    s = int(max(0, min(s, n_frames - 1)))
    e = int(max(0, min(e, n_frames - 1)))
    if e <= s:
        continue
    data = norm_intensity.iloc[s:e+1, :].to_numpy()
    if data.size == 0:
        continue
    old_x = np.linspace(0, 1, data.shape[0])
    new_x = np.linspace(0, 1, max_len_ext)
    rescaled = np.array([np.interp(new_x, old_x, data[:, j]) for j in range(data.shape[1])]).T
    rescaled_ext_all.append(rescaled)

avg_ext = np.mean(rescaled_ext_all, axis=0) if rescaled_ext_all else np.zeros((max_len_ext, norm_intensity.shape[1]))


# --- Step (4): Rescale flextion:extension = 50%:50% and convert the horizontal axis to be angles---
#Oliver please finish this step
 

## --- Step (5): Find the peak value for each frame---
#Oliver please finish this step. The peak intensity value will form a contuour line to indicates how the peak intensity move.
# We will compare it with COM curve, then modify COM definition


# --- Step (6): Save output ---
#Oliver, you need to modify the following to save more information such as angles (horizontal axis of heatmap)
##  and the peak "intensity" value of each frame
excel_path = f"{OUTPUT_BASE}_results.xlsx"
pdf_path = f"{OUTPUT_BASE}_heatmap.pdf"

# Excel output
with pd.ExcelWriter(excel_path) as writer:
    norm_intensity.to_excel(writer, sheet_name="normalized_frames", index=False, header=False)
    pd.DataFrame(avg_flex).to_excel(writer, sheet_name="avg_flexion", index=False, header=False)
    pd.DataFrame(avg_ext).to_excel(writer, sheet_name="avg_extension", index=False, header=False)

# PDF heatmap output
with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(10, 5))
    combined = np.concatenate([avg_flex, avg_ext], axis=0)
    im = ax.imshow(combined.T, aspect="auto", cmap="viridis", origin="lower")
    ax.axvline(x=avg_flex.shape[0]-0.5, color="white", linestyle="--", linewidth=1.5)
    ax.set_title(f"Averaged Normalized Intensity: {OUTPUT_BASE} (Flexion | Extension)")
    ax.set_xlabel("Rescaled Frame Index")
    ax.set_ylabel("Segment Index")
    plt.colorbar(im, ax=ax, label="Avg Normalized Intensity (%)")
    pdf.savefig(fig)
    plt.close(fig)

print("Exported:", excel_path, pdf_path)
