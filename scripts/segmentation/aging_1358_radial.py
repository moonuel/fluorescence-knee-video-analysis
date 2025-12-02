import numpy as np
import cv2
from src.core import knee_segmentation as ks
from src.core import radial_segmentation as rdl
from utils import utils, io, views
from functools import partial
import pandas as pd
import os

def get_mask_femur_outer(video:np.ndarray):

    video = utils.blur_video(video)

    video_hist = rdl.match_histograms_video(video, video[1425])

    # Get outer mask
    outer_mask = ks.get_otsu_masks(video_hist, thresh_scale=0.7)
    # outer_mask = utils.morph_erode(outer_mask, (25,25)) # To shrink it/remove noise a bit 

    # Get inner mask
    inner_mask = ks.mask_adaptive(video_hist, 141, 10) 

    # Get femur mask
    femur_mask = rdl.interior_mask(outer_mask, inner_mask)
    femur_mask = utils.morph_close(femur_mask, (15,15), 2)

    # femur_mask[:, 159:322, 205:]
    femur_mask[:, :159, :] = 0
    femur_mask[:, 332:, :] = 0
    femur_mask[:, :, :205] = 0

    return femur_mask, outer_mask


def get_otsu_mask(video:np.ndarray) -> np.ndarray:

    video_blur = utils.blur_video(video)

    video_hist = rdl.match_histograms_video(video, video[1425])

    otsu_mask = ks.get_otsu_masks(video_hist, 0.7)
    otsu_mask = utils.morph_open(otsu_mask, (3,3))
    # otsu_mask = utils.morph_dilate(otsu_mask, (9,9))
    otsu_mask = utils.morph_close(otsu_mask, (39, 39))
    # otsu_mask = (utils.blur_video(otsu_mask) > 0)*255
    return otsu_mask


def get_boundary_points(mask, N_lns):

    mask = mask.copy()

    # Manual refinements
    mask[:, :182, :] = 0
    mask[:, 320:, :] = 0
    mask[:, :, 308:] = 0

    boundary_points = rdl.sample_femur_interior_pts(mask, N_lns=N_lns)
    boundary_points = rdl.forward_fill_jagged(boundary_points)

    # v0 = views.draw_points((mask*63).astype(np.uint8), boundary_points)
    # views.show_frames(v0)

    return boundary_points


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


def estimate_femur_tip(boundary_points, cutoff):

    femur_tip_boundary = rdl.estimate_femur_tip_boundary(boundary_points, cutoff)
    femur_tip_boundary = rdl.forward_fill_jagged(femur_tip_boundary)

    femur_tip = rdl.get_centroid_pts(femur_tip_boundary)
    femur_tip = rdl.smooth_points(femur_tip, window_size=9)

    return femur_tip


def estimate_femur_midpoint(boundary_points, start, end):

    femur_midpt_boundary = rdl.estimate_femur_midpoint_boundary(boundary_points, start, end)
    femur_midpt_boundary = rdl.forward_fill_jagged(femur_midpt_boundary)

    femur_midpt = rdl.get_centroid_pts(femur_midpt_boundary)
    femur_midpt = rdl.smooth_points(femur_midpt, window_size=9)

    return femur_midpt


def load_1358_video():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    input_path = os.path.join(project_root, "data", "processed", "aging1358video.npy")

    video = io.load_nparray(input_path)
    video = utils.center_crop(video, 500)
    video = np.rot90(video, k=1, axes=(1,2))
    video = (video*1.95).astype(np.uint8) # Increase brightness (without overflow)
    video[video==0] = 32

    return video


def main(video, femur_mask, outer_mask):

    # Get radial masks
    boundary_points = get_boundary_points(femur_mask, N_lns=128)
    femur_tip = estimate_femur_midpoint(boundary_points, start=0.05, end=0.5)
    femur_midpt = estimate_femur_midpoint(boundary_points, start=0.6, end=0.95)
    radial_masks = rdl.label_radial_masks(outer_mask, femur_tip, femur_midpt, N=64)

    v0 = views.draw_points((femur_mask*31).astype(np.uint8), femur_tip)
    v0 = views.draw_points(v0, femur_midpt)
    v0 = views.draw_points(v0, boundary_points)
    v0 = views.draw_line(v0, femur_midpt, femur_tip)
    views.show_frames(v0)

    # v1 = views.draw_mask_boundaries( (outer_mask*63).astype(np.uint8), radial_masks)
    # views.show_frames(v1)
    
    views.show_frames([radial_masks * (255 // radial_masks.max()), video], "Validate data before saving")

    breakpoint()

    # io.save_nparray(video, "../../data/processed/1358_aging_radial_video_N64.npy")
    # io.save_nparray(radial_masks, "../../data/processed/1358_aging_radial_masks_N64.npy")


if __name__ == "__main__":
    
    
    video = load_1358_video()
    femur_mask, _ = get_mask_femur_outer(video)

    otsu_mask = get_otsu_mask(video)

    main(video, femur_mask, otsu_mask)
