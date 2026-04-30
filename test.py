from pathlib import Path
from src.utils.io import load_knee_intensity_workbook

path = Path("data/intensities_total/0000N64intensities.xlsx")

# Test 1: raw only
wb = load_knee_intensity_workbook(path, source="raw")
assert wb.intensities.shape[0] > 0, "no frames"
assert len(wb.flexion_cycles) == len(wb.extension_cycles), "cycle count mismatch"
print(f"raw: {wb.intensities.shape}, {len(wb.flexion_cycles)} cycles")

# Test 2: bgsub
wb2 = load_knee_intensity_workbook(path, source="bgsub")
print(f"bgsub: {wb2.intensities.shape}")

# Test 3: with pixels
wb3 = load_knee_intensity_workbook(path, source="raw", include_pixels=True)
assert wb3.num_pixels is not None
print(f"pixels: {wb3.num_pixels.shape}")

# Test 4: with regions
wb4 = load_knee_intensity_workbook(path, source="raw", include_regions=True)
assert wb4.anatomical_regions is not None
print(f"regions: {wb4.anatomical_regions.columns.tolist()}")

print("All smoke tests passed")