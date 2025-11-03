import numpy as np
import cv2
from core import knee_segmentation as ks
from core import radial_segmentation as rdl
from utils import utils, io, views
from functools import partial

def get_femur_mask():
    
    video = io.load_nparray("../data/processed/1342_knee_frames_ctrd.npy")[:500]
    video = utils.crop_video_square(video, 500)
    video = np.rot90(video, k=1, axes=(1,2))
    video = np.flip(video, axis=2)
    nfs, h, w = video.shape
    print(nfs, h, w)
    video = utils.rotate_video(video, 8)
    video[video == 0] = 16 # Fill empty borders with I=16

    # Blur
    video = utils.blur_video(video)

    # Rescale intensities for more consistent segmentation
    ref_fm = video[48]
    video_hist = rdl.match_histograms_video(video, ref_fm)

    # Get outer mask
    outer_mask = ks.get_otsu_masks(video_hist, thresh_scale=0.8)
    outer_mask = utils.morph_erode(outer_mask, (25,25)) # To shrink it/remove noise a bit 

    # Get inner mask
    inner_mask = ks.mask_adaptive(video_hist, 141, 8)

    # Get femur mask
    femur_mask = rdl.interior_mask(outer_mask, inner_mask)
    femur_mask = utils.morph_open(femur_mask, (15,15))
    femur_mask = utils.morph_close(femur_mask, (25,25))
    femur_mask = utils.blur_video(femur_mask)
    femur_mask = femur_mask > 127

    # views.show_frames(video, "video")
    views.show_frames(video_hist, "video hist")
    # views.show_frames(outer_mask, "outer mask")
    # views.show_frames(inner_mask, "inner mask")
    # views.show_frames(femur_mask, "femur mask")
    v0 = views.draw_mask_boundary(video, femur_mask)
    views.show_frames(v0, "mask boundary")

    return femur_mask


def get_boundary_points(mask, N_lns):

    mask = mask.copy()

    # Manual refinements
    mask[:, :182, :] = 0
    mask[:, 320:, :] = 0
    mask[:, :, 308:] = 0

    boundary_points = rdl.sample_femur_interior_pts(mask, N_lns=N_lns)

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

    femur_tip = rdl.get_centroid_pts(femur_tip_boundary)
    femur_tip = rdl.smooth_points(femur_tip, window_size=9)

    return femur_tip


def estimate_femur_midpoint(boundary_points, start, end):

    femur_midpt_boundary = rdl.estimate_femur_midpoint_boundary(boundary_points, start, end)

    femur_midpt = rdl.get_centroid_pts(femur_midpt_boundary)
    femur_midpt = rdl.smooth_points(femur_midpt, window_size=9)

    return femur_midpt


def load_video():

    video = io.load_nparray("../data/processed/1342_knee_frames_ctrd.npy")[:497]
    video = utils.crop_video_square(video, 500)
    video = np.rot90(video, k=1, axes=(1,2))
    # video = np.flip(video, axis=2)
    nfs, h, w = video.shape
    print(nfs, h, w)
    video = utils.rotate_video(video, 8)
    video[video == 0] = 16 # Fill empty borders with I=16

    return video



if __name__ == "__main__":

    video = load_video()

    # mask = get_femur_mask()
    # io.save_nparray(mask, "../data/processed/1342_aging_mask_0-499.npy")

    mask = io.load_nparray("../data/processed/1342_aging_mask_0-499.npy")[:497]

    boundary_points = get_boundary_points(mask, N_lns=128)
    
    femur_tip = estimate_femur_tip(boundary_points, cutoff=0.6)

    femur_midpt = estimate_femur_midpoint(boundary_points, start=0.1, end=0.5)

    radial_masks = rdl.label_radial_masks(mask, femur_tip, femur_midpt, N=64)

    v0 = views.draw_points((mask*31).astype(np.uint8), femur_tip)
    v0 = views.draw_points(v0, femur_midpt)
    v0 = views.draw_points(v0, boundary_points)
    v0 = views.draw_line(v0, femur_midpt, femur_tip)
    views.show_frames(v0)

    v1 = views.draw_mask_boundaries( (mask*63).astype(np.uint8), radial_masks)
    views.show_frames(v1)

    breakpoint()

    # io.save_nparray(video, "../data/processed/1342_aging_radial_video_N16.npy")
    # io.save_nparray(radial_masks, "../data/processed/1342_aging_radial_masks_N16.npy")