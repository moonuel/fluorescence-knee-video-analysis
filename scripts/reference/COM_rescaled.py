import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Parameters ---
#This code is for both videos: 308 and 1339
#video_number = 308
#segment_count = 16
#num_cycles = 4

video_number = 1339
segment_count = 64
num_cycles = 6

# --- Load data ---
file_path = f"{video_number}N{segment_count}COM.xlsx"
df = pd.read_excel(file_path)

# --- Clean up and rename columns ---
#df.columns = ["Frames (relative)", "Cycle1", "Cycle2", "Cycle3", "Cycle4"]  #video308
df.columns = ["Frames (relative)", "Cycle1", "Cycle2", "Cycle3", "Cycle4", "Cycle5", "Cycle6"] #video 1339

# --- Identify frame boundaries ---
left_frames_count = int(df["Frames (relative)"].min())
right_frames_count = int(df["Frames (relative)"].max())

print(f"Left frames count: {left_frames_count}")
print(f"Right frames count: {right_frames_count}")

# --- Determine longest left and right segments ---
left_lengths = []
right_lengths = []

for i in range(1, num_cycles + 1):
    cycle_data = df[["Frames (relative)", f"Cycle{i}"]].dropna()
    left_part = cycle_data[cycle_data["Frames (relative)"] < 0]
    right_part = cycle_data[cycle_data["Frames (relative)"] > 0]
    left_lengths.append(len(left_part))
    right_lengths.append(len(right_part))

max_left_len = max(left_lengths)
max_right_len = max(right_lengths)

# --- Define new x-values for interpolation based on the longest left and right segments---
x_left_new = np.linspace(left_frames_count, -1, max_left_len)
x_right_new = np.linspace(1, right_frames_count, max_right_len)

# --- Container for rescaled data ---
rescaled_data = {}

# --- Rescale each cycle ---
for i in range(1, num_cycles + 1):
    cycle_data = df[["Frames (relative)", f"Cycle{i}"]].dropna()
    x = cycle_data["Frames (relative)"].values
    y = cycle_data[f"Cycle{i}"].values

    # Separate portions
    mask_left = x < 0
    mask_right = x > 0
    y0 = y[x == 0][0] if np.any(x == 0) else np.nan

    # Interpolate left and right sides
    f_left = interp1d(x[mask_left], y[mask_left], kind="linear", fill_value="extrapolate")
    f_right = interp1d(x[mask_right], y[mask_right], kind="linear", fill_value="extrapolate")

    y_left_rescaled = f_left(x_left_new)
    y_right_rescaled = f_right(x_right_new)

    # Combine left, zero, right
    x_combined = np.concatenate([x_left_new, [0], x_right_new])
    y_combined = np.concatenate([y_left_rescaled, [y0], y_right_rescaled])

    rescaled_data[f"Cycle{i}"] = y_combined

# --- Average across all cycles ---
y_matrix = np.array(list(rescaled_data.values()))
y_avg = np.nanmean(y_matrix, axis=0)

#----Rescale flextion:extension = 50%:50% and convert the horizontal axis to be angles
# Oliver you needs to do this step

# --- Create final DataFrame for export ---
x_combined_full = np.concatenate([x_left_new, [0], x_right_new])
rescaled_df = pd.DataFrame({"Frames (relative)": x_combined_full})
for i in range(1, num_cycles + 1):
    rescaled_df[f"Cycle{i}"] = rescaled_data[f"Cycle{i}"]
rescaled_df["Average"] = y_avg

# --- Plot results ---
plt.figure(figsize=(10, 6))
for i in range(1, num_cycles + 1):
    plt.plot(x_combined_full, rescaled_data[f"Cycle{i}"], label=f"Cycle {i}", alpha=0.6)
plt.plot(x_combined_full, y_avg, label="Average", color="black", linewidth=2)
plt.xlabel("Frames (relative)")
plt.ylabel("Intensity")
plt.title(f"Video {video_number} | Segment count: {segment_count} | {num_cycles} cycles")
plt.legend()
plt.grid(True)

# --- Export files ---
pdf_path = f"Video{video_number}N{segment_count}_Rescaled_COMPlot.pdf"
csv_path = f"Video{video_number}N{segment_count}_Rescaled_COM.csv"

plt.savefig(pdf_path, bbox_inches="tight")
rescaled_df.to_csv(csv_path, index=False)

print(f"\nExported files:\n  PDF: {pdf_path}\n  CSV: {csv_path}")
