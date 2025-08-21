from utils import utils, io, views
from core import knee_segmentation as ks
from core import radial_segmentation as rdl
import numpy as np
from functools import partial
import cv2

"""
Steps:
    >>> Get femur mask
    >>> Estimate femur tip
    >>> Estimate femur midpoint
    >>> Get radial segmentation
"""

def get_femur_mask(video:np.ndarray) -> np.ndarray:

    video_blr = utils.parallel_process_video(video, utils.blur_video, num_workers=4, batch_size=150)
    # views.show_frames(video_blr)

    # Rescale intensities for more consistent otsu segmentation
    ref_fm = video[118]
    match_histograms_func = partial(rdl.match_histograms_video, reference_frame=ref_fm) # Establish pure function
    video_blr_hist = utils.parallel_process_video(video_blr, match_histograms_func, verbose=True, num_workers=10, batch_size=200) # Parallelize 
    views.show_frames(video_blr_hist)

    # Get outer mask 
    otsu_func = partial(ks.get_otsu_masks, thresh_scale=0.6)
    outer_mask = utils.parallel_process_video(video_blr_hist, otsu_func, verbose=True, num_workers=10, batch_size=200)
    
    morph_erode_func = partial(utils.morph_erode, kernel_size=(41,41))
    outer_mask = utils.parallel_process_video(outer_mask, morph_erode_func) #, batch_size=100, num_workers=4) # Parallel version
    # outer_mask = utils.morph_erode(outer_mask, (36,36)) # Non-parallel version

    # views.draw_mask_boundary(video, outer_mask)

    # Get inner mask
    inner_mask = ks.mask_adaptive(video_blr, 141, 8) # TODO: fill gaps in the segmentation

    # Exclude noise from inner mask
    femur_mask = rdl.interior_mask(outer_mask, inner_mask)

    # Refinements
    # femur_mask = utils.morph_open(femur_mask, (11,11))

    femur_mask = utils.blur_video(femur_mask, (15,15)) # Smooth/expand edges
    femur_mask = (femur_mask > 0).astype(np.uint8) * 255 
    
    femur_mask = utils.morph_close(femur_mask, (15,15))

    # Manual refinements 
    # From 613-783:
    femur_mask[610:790, 321:, :] = 0 # Cut y below 321
    femur_mask[610:790, :148, :] = 0 # Cut y above 148
    femur_mask[610:790, :, 292:] = 0 # Cut x above 292

    # From 1700-
    femur_mask[1700:, 333:, :] = 0 # Cut y below 333
    femur_mask[1700:, :127, :] = 0 # Cut y above 127
    femur_mask[1700:, :, 315:] = 0 # Cut x above 315

    views.show_frames(femur_mask)

    return femur_mask

def save_1193_mask():
    video = io.load_nparray("../data/processed/1193_knee_frames_ctrd.npy")#[600:1000]
    video = utils.crop_video_square(video, 500)
    video = np.rot90(video, k=1, axes=(1,2))
    video = np.flip(video, axis=2)
    video[video == 0] = 19 # Fill empty borders for histogram matching stability

    mask = get_femur_mask(video)

    io.save_nparray(mask, "../data/processed/1193_normal_mask.npy")

    return


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


def main():

    video = io.load_nparray("../data/processed/1193_knee_frames_ctrd.npy")#[600:1000]
    video = utils.crop_video_square(video, 500)
    video = np.rot90(video, k=1, axes=(1,2))
    video = np.flip(video, axis=2)
    video[video == 0] = 19 # Fill empty borders for histogram matching stability

    mask = io.load_nparray("../data/processed/1193_normal_mask.npy")

    print(video.shape, mask.shape)

    # views.show_frames(video)
    views.draw_mask_boundary(video, mask) # Validate results

    # Get interior boundary points
    femur_boundary = rdl.sample_femur_interior_pts(mask, 128)
    femur_boundary = forward_fill_jagged(femur_boundary) # Forward fill frames with empty point sets for safety

    views.draw_points(video, femur_boundary) # Validate results

    # Estimate femur tip 
    femur_tip_ = rdl.estimate_femur_tip_boundary(femur_boundary, 0.6)
    # femur_tip_ = rdl.filter_outlier_points_centroid(femur_tip_, 100) # Not necessary?   
    views.draw_points(video, femur_tip_) # Validate results

    femur_tip = rdl.get_centroid_pts(femur_tip_)
    femur_tip = rdl.smooth_points(femur_tip, window_size=7)
    views.draw_points(video, femur_tip) # Validate results

    # Estimate femur midpoint
    femur_midpt_ = rdl.estimate_femur_midpoint_boundary(femur_boundary, 0.1, 0.4)
    # femur_midpt_ = rdl.filter_outlier_points_centroid(femur_midpt_, 100)
    views.draw_points(video, femur_midpt_) # Validate results

    femur_midpt = rdl.get_centroid_pts(femur_midpt_)
    femur_midpt = rdl.smooth_points(femur_midpt, window_size=7)
    views.draw_points(video, femur_midpt) # Validate results


    # Radially segment 
    circle_pts = rdl.get_N_points_on_circle(femur_tip, femur_midpt, N=16)
    radial_regions, radial_masks = rdl.get_radial_segments(video, femur_tip, circle_pts, thresh_scale=0.8)

    v1 = views.draw_radial_masks(video, radial_masks, False)
    views.draw_radial_slice_numbers(v1, circle_pts)

if __name__ == "__main__":
    # save_1193_mask() # Doesn't need to be run again unless the mask needs adjustments 

    main()