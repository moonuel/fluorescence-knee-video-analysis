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

def save_1339_data():
    """Imports, centers, crops, and saves frames 0-650 of the 1339 aging video data. Only needs to be used once"""
    video = io.load_hdf5_video_chunk("../data/raw/right-0 min-regional movement_00001339_grayscale.h5", (0,650), verbose=True)
    video = np.rot90(video, k=1, axes=(1,2))
    video = np.flip(video, axis=2)

    video, _ = ks.centre_video(video)
    video = utils.crop_video_square(video, 500)

    io.save_nparray(video, "../data/processed/1339_knee_frames_0-649_ctrd.npy")
    return

def load_1339_data() -> np.ndarray:
    """Loads the saved data. See save_1339_data()"""
    return io.load_nparray("../data/processed/1339_knee_frames_0-649_ctrd.npy")

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
        
    matched_video = np.empty_like(video)
    for i in range(video.shape[0]):
        matched_video[i] = match_histograms(video[i], reference_frame)
        
    return matched_video


def get_mask_around_femur(video:np.ndarray) -> np.ndarray:
    """Takes the centered grayscale 1339 video and returns a binary mask appropriate for estimating the position of the femur."""

    video_blrd = utils.blur_video(video)
    video_blrd_hist = match_histograms_video(video_blrd) # For consistency of Otsu segmentation

    # Get outer mask
    otsu_mask = ks.get_otsu_masks(video_blrd_hist, 0.6)
    otsu_mask = utils.morph_erode(otsu_mask, (41,41))
    
    # views.show_frames(otsu_mask, "debugging otsu mask params")
    # views.draw_mask_boundary(video_blrd_hist, otsu_mask)

    # Get inner mask
    femur_mask = ks.mask_adaptive(video_blrd, 151, 8)

    # Get mask for femur estimation
    femur_mask = rdl.interior_mask(otsu_mask, femur_mask)

    # Refinements
    femur_mask = utils.blur_video(femur_mask, (5,5)) 
    femur_mask = (femur_mask > 0).astype(np.uint8) * 255 # clip blurred mask to binary mask

    return femur_mask

def analyze_video(video, radial_masks, radial_regions, 
                  lft: Tuple[int, int], mdl: Tuple[int, int], rgt: Tuple[int, int], 
                  show_figs: bool = True, save_dir: str = None, 
                  fig_size: Tuple[int, int] = (17, 9)) -> None:
    """Analyzes all frames in a radially-segmented knee fluorescence video.

    Parameters
    ----------
    video : np.ndarray
        Input video of shape (n_frames, H, W)
    radial_masks : np.ndarray
        Binary mask array of shape (n_slices, n_frames, H, W)
    radial_regions : np.ndarray
        Binary region array of same shape as radial_masks
    lft, mdl, rgt : Tuple[int, int]
        Circular slice ranges for left/middle/right knees
    show_figs : bool, optional
        Whether to display the figure
    save_dir : str, optional
        Directory to save output figure. If None, the figure is not saved.
    fig_size : Tuple[int, int], optional
        Size of the matplotlib figure

    Returns
    -------
    total_sums : np.ndarray
        Measured intensities
    fig : matplotlib.figure.Figure
        The generated figure
    axes : np.ndarray
        The figure's axes
    """
    if VERBOSE: print("analyze_video() called!")

    video = video.copy()
    nfs, h, w = video.shape

    assert nfs == radial_masks.shape[1] 
    assert nfs == radial_regions.shape[1]

    masks = {
        'l': rdl.combine_masks(rdl.circular_slice(radial_masks, lft)), 
        'm': rdl.combine_masks(rdl.circular_slice(radial_masks, mdl)),
        'r': rdl.combine_masks(rdl.circular_slice(radial_masks, rgt))
    }
    
    regions = {
        'l': rdl.combine_masks(rdl.circular_slice(radial_regions, lft)), 
        'm': rdl.combine_masks(rdl.circular_slice(radial_regions, mdl)),
        'r': rdl.combine_masks(rdl.circular_slice(radial_regions, rgt))
    }
    
    keys = ['l','m','r']
    
    total_sums = dp.measure_radial_intensities(np.asarray([
        regions["l"], regions["m"], regions["r"]
    ]))

    fig, axes = views.plot_radial_segment_intensities_2(total_sums, vert_layout=True, save_dir=save_dir, figsize=(17,9))

    if show_figs:
        import matplotlib.pyplot as plt
        plt.show()

    return total_sums, fig, axes
def main():
    print("main() called!")

    video = load_1339_data()[289:608] # aka 210 - 609, when written in 1-based indexing
    video = utils.blur_video(video, kernel_dims=(11,11), sigma=3)
    nfs, h, w = video.shape
    
    agl = 22
    video = utils.rotate_video(video, agl)

    # Estimate femur position
    mask = get_mask_around_femur(video)
    femur_bndry = rdl.sample_femur_interior_pts(mask, 128)
    # views.draw_points(video, femur_bndry)

    # Estimate femur tip
    femur_tip_bndry = rdl.estimate_femur_tip_boundary(femur_bndry)
    femur_tip_bndry = rdl.filter_outlier_points_centroid(femur_tip_bndry, eps=60)
    # v1 = views.draw_points(video, femur_tip_bndry)

    femur_tip = rdl.get_centroid_pts(femur_tip_bndry)
    femur_tip = rdl.smooth_points(femur_tip, 10)
    # views.draw_points(v1, femur_tip)

    # Estimate femur midpoint
    femur_mid_bndry = rdl.estimate_femur_midpoint_boundary(femur_bndry, 0.1, 0.4)
    # v2 = views.draw_points(video, femur_mid_bndry, False)
    
    femur_mid = rdl.get_centroid_pts(femur_mid_bndry)
    femur_mid = rdl.smooth_points(femur_mid, 10)    
    # views.draw_points(v2, femur_mid)

    # Estimate femur position
    circle_pts = rdl.get_N_points_on_circle(femur_tip, femur_mid, N=16, radius_scale=2)
    # views.draw_points(v2, circle_pts)

    radial_regions, radial_masks = rdl.get_radial_segments(video, femur_tip, circle_pts, thresh_scale=0.6)
    v1 = views.draw_radial_masks(video, radial_masks, False)
    views.draw_radial_slice_numbers(v1, circle_pts)

    # Knee analysis
    lft = (11,1)
    mdl = (7,11)
    rgt = (1,7)
    total_sums, fig, axes = analyze_video(video, radial_masks, radial_regions, lft, mdl, rgt)




    return


if __name__ == "__main__":
    main()