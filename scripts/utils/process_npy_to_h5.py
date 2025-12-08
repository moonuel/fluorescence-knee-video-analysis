"""
Loads, centers, and saves a raw grayscale knee video from .npy to .h5.
"""

import core.radial_segmentation as rdl
import utils.utils as utils
import utils.io as io
import utils.views as views
import numpy as np
import h5py
import sys
import os



def main(file_path:str, save_path:str):

    video = io.load_nparray(file_path)

    video, _ = rdl.centre_video_mp(video)

    views.show_frames(views.rescale_video(video, scale_factor=0.5, show_video=False, show_num=False))

    # Save to HDF5
    with h5py.File(save_path, "w") as h5f:
        dset = h5f.create_dataset("video", shape=video.shape, dtype=np.uint8, chunks=(200, video.shape[1], video.shape[2]), compression="gzip")
        dset[:] = video

    return


if __name__ == "__main__":
    if len(sys.argv) != 3: raise SyntaxError(f"{sys.argv[0]} expects two args: [file_in] [file_out]"
                                            f"\n\tExample usage: {sys.argv[0]} aging1339.npy aging1339.h5")

    npy_path = sys.argv[1]
    h5_path = sys.argv[2]

    if not os.path.isfile(npy_path): raise FileNotFoundError("Input file not found.")
    if not h5_path[-3:] == ".h5": raise SyntaxError("Output file isn't a .h5 file."
                                                    f"\n\tExample usage: {sys.argv[0]} aging1339.npy aging1339.h5")

    main(npy_path, h5_path)
