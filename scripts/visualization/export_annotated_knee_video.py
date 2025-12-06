"""Template script for saving knee video data with segments for verification of results.
"""

from utils import utils, io, views
import numpy as np
import pandas as pd
import pandas as pd
from pathlib import Path
import cv2
from multiprocessing import Pool 
from typing import Tuple
import sys
import os
from config import TYPES

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


def draw_region_numbers(video: np.ndarray, mask_video: np.ndarray, step:int = 1, font_scale=0.5, color=255, thickness=1) -> np.ndarray:
    """
    Overlay region numbers at the centroid of each labeled region in a grayscale video.

    Parameters
    ----------
    video : np.ndarray
        Grayscale video array of shape (frames, height, width), dtype should be uint8 or compatible.
    mask_video : np.ndarray
        Integer-labeled mask video of the same shape, where 0 = background and 1..N = regions.
    step : int
        Sets the step for segment labeling. Defaults to 1 (every segment will be labeled).
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

    from scipy.stats import trim_mean

    modified_video = video.copy()

    assert modified_video.shape == mask_video.shape

    # Loop over frames
    for t in range(video.shape[0]):
        frame_mask = mask_video[t]
        unique_labels = np.unique(frame_mask)
        # Skip background
        region_labels = unique_labels[unique_labels != 0][::step] # Choose only every "step" labels

        for label in region_labels:
            # Find coordinates of pixels belonging to this region
            ys, xs = np.where(frame_mask == label)

            # Skip if region is empty
            if len(xs) == 0:
                continue

            # Compute position of label
            # centroid_x, centroid_y = int(xs.mean()), int(ys.mean())
            centroid_x, centroid_y = int(np.median(xs)), int(np.median(ys))
            # centroid_x, centroid_y = int(trim_mean(xs, 0.25)), int(trim_mean(ys, 0.25))
            

            # Overlay the label at the centroid
            cv2.putText(modified_video[t],
                        text=str(label),
                        org=(centroid_x, centroid_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=int(color),
                        thickness=thickness)

    return modified_video


def process_batch(video_batch, masks_batch):

        out = views.draw_mask_boundaries(video_batch, masks_batch, intensity=159)
        out = draw_region_numbers(out, masks_batch, step=STEP, color=191)

        return out


def main(masks:np.ndarray, video:np.ndarray):

    assert masks.shape == video.shape # Sanity check that we're using new mask format
    
    nfs, h, w = video.shape
    mask_lbls = np.unique(masks[masks > 0]).astype(int) # Returns sorted list of unique non-zero labels
    N = len(mask_lbls)
    assert N == np.max(mask_lbls)
    
    # Validate data
    print(f"{video.shape=}")
    print(f"{mask_lbls=}")
    # views.show_frames([video, masks * (255 // mask_lbls.max())]) # Rescale label intensities for better viewing

    # Draw mask labels
    N_batches = 10
    video_batches = np.array_split(video, N_batches, 0)
    mask_batches = np.array_split(masks, N_batches, 0)

    # Distribute batches to subprocesses
    with Pool(N_batches) as pool:
        results = pool.starmap(process_batch, zip(video_batches, mask_batches))

    # Collate batches 
    video_out = np.concatenate(results, axis=0)


    # video_out = views.draw_mask_boundaries(video, masks)
    # video_out = draw_region_numbers(video_out, masks)

    # video_out = views.show_frames(video_out)

    # breakpoint()

    return video_out


def load_1339_N16() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/aging_1339_radial_masks_N16.npy")
    video = io.load_video("../data/segmented/aging_1339_radial_video_N16.npy")
    cycles =   "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553	554-576	579-609" # 1339 aging

    return masks, video #, cycles


def load_1339_N64() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/aging_1339_radial_masks_N64.npy")
    video = io.load_video("../data/segmented/aging_1339_radial_video_N64.npy")
    
    return masks, video 


def load_308_N16() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/normal_0308_radial_masks_N16.npy")
    video = io.load_video("../data/segmented/normal_0308_radial_video_N16.npy")

    masks, video = np.flip(masks, axis=2), np.flip(video, axis=2) # Flip along horizontal dim
    masks[masks > 0] = (masks[masks > 0] - 2) % 16 + 1 # Shift segment labels by one for 308 N16 video

    return masks, video 


def load_308_N64() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/normal_0308_radial_masks_N64.npy")
    video = io.load_video("../data/segmented/normal_0308_radial_video_N64.npy")

    return masks, video


def load_1207_N64() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/normal_1207_radial_masks_N64.npy")
    video = io.load_video("../data/segmented/normal_1207_radial_video_N64.npy")

    return masks, video


def load_1190_N64() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/normal_1190_radial_masks_N64.npy")
    video = io.load_video("../data/segmented/normal_1190_radial_video_N64.npy")

    return masks, video


def load_1193_N64() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/normal_1193_radial_masks_N64.npy")
    video = io.load_video("../data/segmented/normal_1193_radial_video_N64.npy")

    return masks, video


# Global variable for setting step in segment labeling... lazy i know
STEP = 4 

if __name__ == "__main__":

    # Input validation
    if len(sys.argv) != 3: 
        raise SyntaxError(f"{sys.argv[0]} requires 2 args.\n\t"
                          f"Example usage: {sys.argv[0]} 1339 64\n\t"
                          f"Valid types are: {list(TYPES.keys())}")    
    
    if not int(sys.argv[1]) in TYPES.keys(): raise SyntaxError(f"Knee type not found. Valid types are: {list(TYPES.keys())}")
    
    TYPE = TYPES[int(sys.argv[1])]
    mask_path = f"../data/segmented/{sys.argv[1]}_{TYPE}_radial_masks_N{sys.argv[2]}.npy"
    video_path = f"../data/segmented/{sys.argv[1]}_{TYPE}_radial_video_N{sys.argv[2]}.npy"

    if not os.path.isfile(mask_path): raise FileNotFoundError(f"{mask_path} not found.")
    if not os.path.isfile(video_path): raise FileNotFoundError(f"{video_path} not found.")
    
    # Load data
    masks = io.load_masks(f"../data/segmented/{sys.argv[1]}_{TYPE}_radial_masks_N{sys.argv[2]}.npy")
    video = io.load_video(f"../data/segmented/{sys.argv[1]}_{TYPE}_radial_video_N{sys.argv[2]}.npy")

    # Process video
    video_out = main(masks, video)
    video_out = views.show_frames(video_out) # add frame nums
    
    # breakpoint()

    # Save video
    save = None
    out_file = f"{sys.argv[1]}N{sys.argv[2]}.mp4"
    while save not in ['y', 'n']:
        save = input("\nEnter 'y' to save video, or 'n' to exit.\n\t"
                    f"Output file: {out_file}\n > ").lower()
    if save == 'y': 
        io.save_mp4(f"{out_file}", video_out)
        print(f"Video saved to: {out_file}")
    if save == 'n':
        print("Exiting without saving video.")

    # video_out = views.show_frames(video_out, frame_offset=289) # Overwrite frame nums with offset
    
    


    # filename = "1339_aging_validation_video"
    # io.save_avi(f"{filename}.avi", video_out)
    # io.save_mp4(f"{filename}.mp4", video_out)
