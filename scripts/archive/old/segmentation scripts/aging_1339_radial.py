"""
Radial segmentation-based analysis of the 1339 aging knee video data.
Segments the knee into N equally-spaced radial slices, and then groups them into chunks associated with the left/middle/right portions of the knee.
"""


import utils.io as io
import utils.views as views
import utils.utils as utils
import core.data_processing as dp
import core.radial_segmentation as rdl
import core.knee_segmentation as ks
import numpy as np
import cv2
import numpy as np
from skimage.exposure import match_histograms
from config import VERBOSE
from typing import Tuple
import matplotlib.pyplot as plt

def save_1339_data():
    """Imports, centers, crops, and saves frames 0-650 of the 1339 aging video data. Only needs to be used once"""
    video = io.load_hdf5_video_chunk("../data/raw/right-0 min-regional movement_00001339_grayscale.h5", (0,650), verbose=True)
    video = np.rot90(video, k=1, axes=(1,2))
    video = np.flip(video, axis=2)

    video, _ = ks.centre_video(video)
    video = utils.center_crop(video, 500)

    io.save_nparray(video, "../data/segmented/1339_knee_frames_0-649_ctrd.npy")
    return

def load_1339_data() -> np.ndarray:
    """Loads the saved data. See save_1339_data()"""
    return io.load_nparray("../data/segmented/1339_knee_frames_0-649_ctrd.npy")

def match_histograms_video(video, reference_frame=None):
    """
    Apply histogram matching to each frame in the video.
    
    Args:
        video: np.ndarray of shape (n_frames, height, width)
        reference_frame: optional np.ndarray (height, width), default is video[0]

    Returns:
        matched_video: np.ndarray of same shape as input
    """
    if reference_frame is None:
        reference_frame = video[0]

    if isinstance(reference_frame, int):
        print(f"reference_frame is type(int). Taking reference_frame = video[{reference_frame}] instead.")
        reference_frame = video[reference_frame]

    if not isinstance(reference_frame, np.ndarray):
        raise TypeError(f"{reference_frame=} should be of type: np.ndarray. Given: {type(reference_frame)=}")
        

    matched_video = np.empty_like(video)
    for i in range(video.shape[0]):
        matched_video[i] = match_histograms(video[i], reference_frame)
        
    return matched_video


def forward_fill_jagged(arr):
    """
    Forward fills empty frames in a jagged NumPy array (dtype=object).
    
    Parameters:
        arr (np.ndarray): jagged array of shape (nframes, npts*, 2)
        
    Returns:
        np.ndarray: forward-filled jagged array (same shape/dtype)
    """
    filled = arr.copy()
    last_valid = None
    
    for i, frame in enumerate(filled):
        frame = np.asarray(frame)
        if frame.size == 0:
            if last_valid is not None:
                filled[i] = last_valid
        else:
            last_valid = frame
    
    return filled


def get_mask_around_femur(video:np.ndarray) -> np.ndarray:
    """Takes the centered grayscale 1339 video and returns a binary mask appropriate for estimating the position of the femur."""

    video_blrd = utils.blur_video(video)
    video_blrd_hist = match_histograms_video(video_blrd, 155) # For consistency of Otsu segmentation

    # Get outer mask
    otsu_mask = ks.get_otsu_masks(video_blrd_hist, 0.6)
    # otsu_mask = utils.morph_erode(otsu_mask, (41,41))
    
    # views.show_frames(otsu_mask, "debugging otsu mask params")
    # views.draw_mask_boundary(video_blrd_hist, otsu_mask)

    # Get inner mask
    femur_mask = ks.mask_adaptive(video_blrd, 151, 8)

    # Get mask for femur estimation
    femur_mask = rdl.interior_mask(otsu_mask, femur_mask)

    # Refinements
    femur_mask = utils.blur_video(femur_mask, (5,5)) 
    femur_mask = (femur_mask > 0).astype(np.uint8) * 255 # clip blurred mask to binary mask

    femur_mask[:, 352:, :] = 0
    femur_mask[:, :157, :] = 0
    femur_mask[:, :, :192] = 0

    return femur_mask



def main():
    print("main() called!")

    video = load_1339_data()
    video = np.flip(video, axis=2)
    nfs, h, w = video.shape
    print(f"{video.shape=}")
    
    agl = -26
    video = utils.rotate_video(video, agl)
    video[video==0] = 17 # Fill empty pixels with background noise 

    video = utils.blur_video(video, kernel_dims=(11,11), sigma=3)

    # Estimate femur position
    mask = get_mask_around_femur(video)
    femur_bndry = rdl.sample_femur_interior_pts(mask, 128)
    # views.draw_points(video, femur_bndry)

    # Estimate femur tip
    femur_tip_bndry = rdl.estimate_femur_tip_boundary(femur_bndry)
    femur_tip_bndry = rdl.filter_outlier_points_centroid(femur_tip_bndry, eps=60)
    # femur_tip_bndry = forward_fill_jagged(femur_tip_bndry)
    # v1 = views.draw_points(video, femur_tip_bndry)

    femur_tip = rdl.get_centroid_pts(femur_tip_bndry)
    femur_tip = rdl.smooth_points(femur_tip, 10)
    # views.draw_points(v1, femur_tip)

    # Estimate femur midpoint
    femur_mid_bndry = rdl.estimate_femur_midpoint_boundary(femur_bndry, 0.1, 0.4)
    # femur_mid_bndry = forward_fill_jagged(femur_mid_bndry)
    v2 = views.draw_points(video, femur_mid_bndry, False)
    
    femur_mid = rdl.get_centroid_pts(femur_mid_bndry)
    femur_mid = rdl.smooth_points(femur_mid, 10)    
    
    # views.draw_points(v2, femur_mid)

    # Get Otsu mask
    video_hist = rdl.match_histograms_video(video, video[244])
    otsu_masks = ks.get_otsu_masks(video_hist, thresh_scale=0.6)

    # Radially segment video
    radial_masks = rdl.label_radial_masks(otsu_masks, femur_tip, femur_mid, 16)

    breakpoint()

    return

    # Save segmentation data
    io.save_nparray(video, "../data/segmented/1339_knee_radial_video_N16.npy")
    io.save_nparray(radial_masks, "../data/segmented/1339_knee_radial_masks_N16.npy")
    io.save_nparray(radial_regions, "../data/segmented/1339_knee_radial_regions_N16.npy")

    return


if __name__ == "__main__":
    main()