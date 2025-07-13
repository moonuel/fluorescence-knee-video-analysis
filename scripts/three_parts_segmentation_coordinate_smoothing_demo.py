from utils import utils, io, views
from core import knee_segmentation as ks
from core import data_processing as dp
from config import VERBOSE
import numpy as np

def main():
    if VERBOSE: print("main() called!")

    # Load example video and coords
    video = io.load_nparray("../data/processed/normal_knee_processed.npy")
    translation_mxs = io.load_nparray("../data/processed/normal_translation_mxs.npy")
    coords, metadata = io.load_normal_knee_coords("../data/xy coordinates for knee imaging 0913.xlsx", 0) # 8.6 normal knee
    coords = ks.translate_coords(translation_mxs, coords) # Center the coords

    # Smooth coords
    coords_smthd = ks.smooth_coords(coords, 5)

    # Validate smoothing procedure
    video_unsmthd = views.plot_coords(video, coords, show_video=False)
    video_unsmthd = views.rescale_video(video_unsmthd, 0.5, show_video=False)

    video_smthd = views.plot_coords(video, coords_smthd, show_video=False)
    video_smthd = views.rescale_video(video_smthd, 0.5, show_video=False)

    f0 = metadata['f0']
    fN = metadata['fN']
    # video_unsmthd = utils.crop_video_square(video_unsmthd, 350)
    # video_smthd = utils.crop_video_square(video_smthd, 350)
    views.show_frames(np.concatenate([video_unsmthd[f0:fN+1], video_smthd[f0:fN+1]], axis=2), show_num=False, title="Left: Unsmoothed coords, right: Smoothed coords")

if __name__ == "__main__":
    main()