"""
Loads, centers, and saves a raw grayscale knee video.
"""

import core.radial_segmentation as rdl
import core.knee_segmentation as ks
import utils.utils as utils
import utils.io as io
import utils.views as views
import numpy as np
from typing import Tuple





def main(file_path:str, frames:Tuple[int, int], save_path:str):

    video = io.load_hdf5_video_chunk(file_path, frames, verbose=True)

    video, _ = rdl.centre_video_mp(video)

    views.show_frames(views.rescale_video(video, scale_factor=0.5, show_video=False, show_num=False))

    io.save_nparray(video, save_path)

    return

# Example usage:
    # main(
        # "1 con-0 min-fluid movement_00001190.h5", 
        # frames, 
        # "1190_knee_frames_ctrd.npy")

if __name__ == "__main__":
    h5_path = "../data/raw/right 10 min-regional movement_00001342.h5"
    npy_path = "../data/processed/1342_knee_frames_ctrd.npy"

    frames = None # None defaults to all

    main(h5_path, frames, npy_path)