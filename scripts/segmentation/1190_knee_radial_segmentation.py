from utils import utils, io, views
from core import knee_segmentation as ks
from core import radial_segmentation as rdl
import numpy as np
from functools import partial
import cv2
import pandas as pd

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
    ref_fm = video[492]
    match_histograms_func = partial(rdl.match_histograms_video, reference_frame=ref_fm) # Establish pure function
    video_blr_hist = utils.parallel_process_video(video_blr, match_histograms_func, verbose=True, num_workers=4, batch_size=150) # Parallelize 
    # views.show_frames(video_blr_hist)

    # Get outer mask 
    otsu_func = partial(ks.get_otsu_masks, thresh_scale=0.6)
    outer_mask = utils.parallel_process_video(video_blr_hist, otsu_func, verbose=True, num_workers=4, batch_size=150)
    
    morph_erode_func = partial(utils.morph_erode, kernel_size=(41,41))
    outer_mask = utils.parallel_process_video(outer_mask, morph_erode_func) #, batch_size=100, num_workers=4) # Parallel version
    # outer_mask = utils.morph_erode(outer_mask, (36,36)) # Non-parallel version

    # views.draw_mask_boundary(video, outer_mask)

    # Get inner mask
    inner_mask = ks.mask_adaptive(video_blr, 141, 6.5)

    # Exclude noise from inner mask
    femur_mask = rdl.interior_mask(outer_mask, inner_mask)

    # Refinements
    femur_mask = utils.morph_open(femur_mask, (11,11))

    femur_mask = utils.blur_video(femur_mask, (15,15)) # Smooth/expand edges
    femur_mask = (femur_mask > 0).astype(np.uint8) * 255 
    
    femur_mask = utils.morph_close(femur_mask, (27,23))

    # Manual refinements 
    femur_mask[:, 341: , :] = 0 # Cut y below 341
    femur_mask[:, 0:132, :] = 0 # Cut y above 132
    femur_mask[:, :, 291:] = 0 # Cut x above 291 
    femur_mask[445:475, :, 278: ] = 0 # Between 445 and 474, cut x above 278
    femur_mask[521:540, :, 282: ] = 0 # Between 521 and 539, cut x above 282

    # views.show_frames(femur_mask)

    return femur_mask

def main():

    video = io.load_nparray("../data/processed/1190_knee_frames_ctrd.npy")
    video = utils.crop_video_square(video, 500)
    video = np.rot90(video, k=1, axes=(1,2))

    # Fill empty regions with L=18
    video[video == 0] = 18

    # views.show_frames(video)

    # Get good femur mask
    mask = get_femur_mask(video)

    # Get interior boundary points
    femur_boundary = rdl.sample_femur_interior_pts(mask, 128)

    # Estimate femur tip 
    femur_tip_ = rdl.estimate_femur_tip_boundary(femur_boundary, 0.6)
    femur_tip_ = rdl.forward_fill_jagged(femur_tip_)
    # femur_tip_ = rdl.filter_outlier_points_centroid(femur_tip_, 100) # Not necessary?   
    
    femur_tip = rdl.get_centroid_pts(femur_tip_)
    femur_tip = rdl.smooth_points(femur_tip, window_size=15)

    # Estimate femur midpoint
    femur_midpt_ = rdl.estimate_femur_midpoint_boundary(femur_boundary, 0.1, 0.4)
    femur_midpt_ = rdl.forward_fill_jagged(femur_midpt_)
    # femur_midpt_ = rdl.filter_outlier_points_centroid(femur_midpt_, 100)
    # views.draw_points(video, femur_midpt_)

    femur_midpt = rdl.get_centroid_pts(femur_midpt_)
    femur_midpt = rdl.smooth_points(femur_midpt, window_size=15)

    # views.draw_points(video, femur_tip)
    # views.draw_points(video, femur_midpt)

    # Radially segment 
    # circle_pts = rdl.get_N_points_on_circle(femur_tip, femur_midpt, N=64)
    # radial_regions, radial_masks = rdl.get_radial_segments(video, femur_tip, circle_pts, thresh_scale=0.8)

    video_for_mask = utils.blur_video(video)
    video_for_mask = rdl.match_histograms_video(video_for_mask, video_for_mask[562])
    otsu_mask = ks.get_otsu_masks(video_for_mask, 0.7)

    breakpoint()

    radial_masks = rdl.label_radial_masks(otsu_mask, femur_tip, femur_midpt, 64)
    views.draw_mask_boundary
    v1 = views.show_frames(radial_masks * (255 // 64))


    # Save segmentation data
    # io.save_nparray(video, "../data/processed/1190_normal_radial_video_N16.npy")
    # io.save_nparray(radial_masks, "../data/processed/1190_normal_radial_masks_N16.npy")



if __name__ == "__main__":
    main()