"""Template script for saving knee video data with segments for verification of results.
"""

from utils import utils, io, views
import numpy as np
import pandas as pd
import pandas as pd
from pathlib import Path
import cv2


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


def draw_region_numbers(video: np.ndarray, mask_video: np.ndarray, font_scale=0.5, color=255, thickness=1) -> np.ndarray:
    """
    Overlay region numbers at the centroid of each labeled region in a grayscale video.

    Parameters
    ----------
    video : np.ndarray
        Grayscale video array of shape (frames, height, width), dtype should be uint8 or compatible.
    mask_video : np.ndarray
        Integer-labeled mask video of the same shape, where 0 = background and 1..N = regions.
    font_scale : float
        Scale factor for the text font.
    color : int
        Grayscale color for the text (0-255).
    thickness : int
        Thickness of the text.

    Returns
    -------
    modified_video : np.ndarray
        Copy of the video with region numbers drawn at centroids.
    """

    modified_video = video.copy()

    assert modified_video.shape == mask_video.shape

    # Loop over frames
    for t in range(video.shape[0]):
        frame_mask = mask_video[t]
        unique_labels = np.unique(frame_mask)
        # Skip background
        region_labels = unique_labels[unique_labels != 0]

        for label in region_labels:
            # Find coordinates of pixels belonging to this region
            ys, xs = np.where(frame_mask == label)

            # Skip if region is empty
            if len(xs) == 0:
                continue

            # Compute centroid
            centroid_x = int(xs.mean())
            centroid_y = int(ys.mean())

            # Overlay the label at the centroid
            cv2.putText(modified_video[t],
                        text=str(label),
                        org=(centroid_x, centroid_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=int(color),
                        thickness=thickness)

    return modified_video


def main(mask_path, video_path):

    # Load data 
    shared_dir = "../data/processed/"
    mask_path =  shared_dir + mask_path # Manually specify filenames 
    video_path = shared_dir + video_path

    masks = load_masks(mask_path)
    video = load_video(video_path)
    
    assert masks.shape == video.shape # Sanity check that we're using new mask format
    nfs, h, w = video.shape
    print(f"{video.shape=}")

    mask_lbls = np.unique(masks[masks > 0]).astype(int) # Returns sorted list of unique non-zero labels
    N = len(mask_lbls)

    
    # Validate data
    print(f"{mask_lbls=}")
    views.show_frames(masks * (255 // mask_lbls.max())) # Rescale label intensities for better viewing
    views.show_frames(video)

    # Draw mask labels
    video_out = views.draw_mask_boundaries(video, masks)
    video_out = draw_region_numbers(video_out, masks)

    return video_out


if __name__ == "__main__":
    mask_name = "1339_knee_radial_masks_N16.npy" # Path will be pre-pended
    video_name = "1339_knee_radial_video_N16.npy"
    
    video_out = main(mask_name, video_name)
    video_out = views.show_frames(video_out, frame_offset=289) # Overwrite frame nums with offset
    
    
    filename = "1339_aging_validation_video"
    io.save_avi(f"{filename}.avi", video_out)
    io.save_mp4(f"{filename}.mp4", video_out)
