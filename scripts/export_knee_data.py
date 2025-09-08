"""Template script for saving comprehensive data for knee video analysis in a consistent format into an Excel spreadsheet file.

Data format:
    Sheet 1: Total Pixel Intensity per radial segment, for every frame
    Sheet 2: Total number of non-zero pixels per radial segment, for every frame
    Sheet 3: File number, total number of frames, frame numbers for each cycle, and segment numbers assigned to the left/middle/right parts of the knee. 
"""

from utils import utils, io, views
import numpy as np
import pandas as pd


def load_masks(filepath:str) -> np.ndarray:
    """Loads the mask at the specified location. 
    Handles both old uint8 radial masks with shape (nmasks, nframes, h, w) and new uint8 radial masks with shape (nframes, h, w).
    
    Old mask arrays are very space-inefficient and have one dimension for each segment. New mask arrays use a unique numerical label from {1...N} instead."""

    masks = io.load_nparray(filepath)

    if not masks.dtype == np.uint8:
        raise ValueError(f"File is not of type uint8. Is it a radial mask? Given: {masks.dtype=}")
    
    # Convert inefficient mask array to efficient array
    if len(masks.shape) == 4:
        N = masks.shape[0] # Expected shape: (nmasks, nframes, h, w)
        masks_bool = np.zeros(shape=masks.shape[1:], dtype=np.uint8) # Expected shape: (nframes, h, w)
        for n in range(N):
            masks_bool[masks[n] > 0] = n+1 # Convert each slice of inefficient array to a numerical label from {1...N}
        masks = masks_bool

    assert len(masks.shape) == 3 # Soft check that output is shape (nfs, h, w)

    return masks


def load_video(filepath:str) -> np.ndarray:
    """Loads the video at the specified location."""
    video = io.load_nparray(filepath)

    if not video.dtype == np.uint8: 
        raise ValueError(f"File is not of type uint8. Is it a video? Given: {video.dtype=}")
    
    if not len(video.shape) == 3:
        raise ValueError(f"File is not compatible with shape (nfs, h, w). Is it a grayscale video? Given: {video.shape=}")
    
    return video


def main():

    # Load data 
    shared_dir = "../data/processed/"
    mask_path =  shared_dir + "1339_knee_radial_masks_N16.npy" # Manually specify filenames 
    video_path = shared_dir + "1339_knee_radial_video_N16.npy"

    masks = load_masks(mask_path)
    video = load_video(video_path)
    assert masks.shape == video.shape # Sanity check that we're using new mask format

    mask_lbls = np.unique(masks[masks > 0])

    # Validate data
    print(f"{mask_lbls=}")
    views.show_frames(masks * (255 // mask_lbls.max())) # Rescale label intensities for better viewing.
    views.show_frames(video)

    return

if __name__ == "__main__":
    main()