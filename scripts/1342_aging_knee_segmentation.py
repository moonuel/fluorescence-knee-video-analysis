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

    return


if __name__ == "__main__":
    get_femur_mask()