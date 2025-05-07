from src.utils import utils, io, views
from src.core import knee_segmentation as ks
from src.core import data_processing as dp
from src.config import VERBOSE

def main():
    if VERBOSE: print("main() called!")

    # Load example video and coords
    video = io.load_nparray("../data/processed/aging_knee_processed.npy")
    translation_mxs = io.load_nparray("../data/processed/aging_translation_mxs.npy")
    coords, metadata = io.load_aging_knee_coords("../data/198_218 updated xy coordinates for knee-aging 250426.xlsx", 1)
    coords = ks.translate_coords(translation_mxs, coords) # Center the coords

    # Smooth coords
    coords_smthd = ks.smooth_coords(coords, 5)

    # Validate smoothing procedure
    views.plot_coords(video, coords, title="unsmoothed coordinates")
    views.plot_coords(video, coords_smthd, title="smoothed coordinates")

if __name__ == "__main__":
    main()