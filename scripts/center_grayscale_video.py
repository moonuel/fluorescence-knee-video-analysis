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
import sys
import os





def main(file_path:str, frames:Tuple[int, int], save_path:str):

    video = io.load_hdf5_video_chunk(file_path, frames, verbose=True)

    video, _ = rdl.centre_video_mp(video)

    views.show_frames(views.rescale_video(video, scale_factor=0.5, show_video=False, show_num=False))

    io.save_nparray(video, save_path)

    return


if __name__ == "__main__":
    # h5_path = "../data/raw/dmm-0 min-fluid movement_00001207.h5"
    # npy_path = "../data/processed/1207_knee_frames_ctrd.npy"

    if len(sys.argv) != 3: raise SyntaxError(f"{sys.argv[0]} expects two args: [file_in] [file_out]"
                                            f"\n\tExample usage: {sys.argv[0]} aging1339.h5 aging1339.npy")

    h5_path = sys.argv[1]
    npy_path = sys.argv[2]

    if not os.path.isfile(h5_path): raise FileNotFoundError("Input file not found.")
    if not npy_path[-4:] == ".npy": raise SyntaxError("Output file isn't a .npy file."
                                                    f"\n\tExample usage: {sys.argv[0]} aging1339.h5 aging1339.npy")

    frames = None # None defaults to all

    main(h5_path, frames, npy_path)