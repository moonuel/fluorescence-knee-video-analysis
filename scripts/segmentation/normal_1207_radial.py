import numpy as np
import cv2
from core import knee_segmentation as ks
from core import radial_segmentation as rdl
from utils import utils, io, views
from functools import partial

def get_femur_mask(video):
    
    # Blur
    video = utils.blur_video(video)
    # video = np.array([soft_knee_compression(frame, 190, 60, 10) for frame in video]) # idk if this is all that 
    video[video > 165] = 165 # Hard clipping

    # Rescale intensities for more consistent segmentation
    ref_fm = video[142]
    video_hist = rdl.match_histograms_video(video, ref_fm)

    # Get outer mask
    outer_mask = ks.get_otsu_masks(video_hist, thresh_scale=0.7)
    outer_mask = utils.morph_close(outer_mask, (45,45)) # Fill any holes
    outer_mask = utils.morph_erode(outer_mask, (25,25)) # To shrink it/remove noise a bit 

    # Get inner mask
    inner_mask = ks.mask_adaptive(video_hist, 161, 11)

    # Get femur mask
    femur_mask = rdl.interior_mask(outer_mask, inner_mask)

    breakpoint()

    # femur_mask = utils.morph_close(femur_mask, (19,19))

    # Manual refinements
    femur_mask = utils.morph_open(femur_mask, (11,11))
    femur_mask[:, 333:, :] = 0

    # views.show_frames(video, "video")
    views.show_frames(video_hist, "video hist")
    views.show_frames(outer_mask, "outer mask")
    views.show_frames(inner_mask, "inner mask")
    # views.show_frames(femur_mask, "femur mask")
    v0 = views.draw_mask_boundary(video, femur_mask)
    views.show_frames(v0, "mask boundary")

    return femur_mask


def get_boundary_points(mask, N_lns):

    mask = mask.copy()

    # Manually truncate mask for better boundary point estimation
    mask[:, 329:, :] = 0 # Cut y below 329 
    mask[:, :180, :] = 0 # Cut y above 180
    mask[:, :, 338:] = 0 # Cut x past 338

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


def estimate_femur_tip(boundary_points, cutoff, weight=0.5):

    femur_tip_boundary = rdl.estimate_femur_tip_boundary(boundary_points, cutoff)

    def jagged_slice(j_arr, start=None, stop=None, step=None):
        return np.array([pts[start:stop:step] for pts in j_arr], dtype=object)

    top_points = jagged_slice(femur_tip_boundary, step=2) # Every other point (even indices)
    btm_points = jagged_slice(femur_tip_boundary, start=1, step=2) # Every other point (odd indices)

    top_mean = rdl.get_centroid_pts(top_points)
    btm_mean = rdl.get_centroid_pts(btm_points)

    print(top_mean.shape)
    print(btm_mean.shape)

    femur_tip = 0.5*(weight*top_mean + (1-weight)*btm_mean) # Weighted average
    femur_tip = femur_tip.astype(int)

    print(femur_tip.shape)

    # femur_tip = rdl.get_centroid_pts(femur_tip_boundary)
    femur_tip = rdl.smooth_points(femur_tip, window_size=9)

    return femur_tip


def estimate_femur_midpoint(boundary_points, start, end):

    femur_midpt_boundary = rdl.estimate_femur_midpoint_boundary(boundary_points, start, end)
    femur_midpt_boundary = forward_fill_jagged(femur_midpt_boundary)

    femur_midpt = rdl.get_centroid_pts(femur_midpt_boundary)
    femur_midpt = rdl.smooth_points(femur_midpt, window_size=9)

    return femur_midpt


def load_1207_normal_video():

    video = io.load_nparray("../data/processed/1207_knee_frames_ctrd.npy")
    video = utils.center_crop(video, 500)
    video = np.rot90(video, k=1, axes=(1,2))
    # video = np.flip(video, axis=2)
    nfs, h, w = video.shape
    print(nfs, h, w)

    breakpoint()

    video = utils.rotate_video(video, -27)
    video[video == 0] = 17 # Fill empty borders with I=16

    return video


def soft_knee_compression(img: np.ndarray, 
                          knee: int = 235, 
                          width: int = 20, 
                          ratio: float = 4.0) -> np.ndarray:
    """
    Apply a soft-knee highlight compressor to a uint8 grayscale image.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (dtype=uint8).
    knee : int
        Knee center intensity (0–255). Around here, compression begins.
    width : int
        Width of the knee band (in intensity units).
    ratio : float
        Compression ratio > 1. Higher = stronger highlight compression.

    Returns
    -------
    np.ndarray
        Output image (dtype=uint8) with compressed highlights.
    """
    img_f = img.astype(np.float32)

    # Knee boundaries
    knee_low = knee - width / 2
    knee_high = knee + width / 2

    # Compressed mapping function (hard compression above knee)
    def comp(I):
        return knee + (I - knee) / ratio

    # Output array
    out = np.empty_like(img_f)

    # Case 1: Below knee_low → unchanged
    mask_low = img_f <= knee_low
    out[mask_low] = img_f[mask_low]

    # Case 2: Above knee_high → fully compressed
    mask_high = img_f >= knee_high
    out[mask_high] = comp(img_f[mask_high])

    # Case 3: Inside knee band → smooth blend
    mask_band = ~(mask_low | mask_high)
    I_band = img_f[mask_band]
    t = (I_band - knee_low) / width  # normalized [0,1]
    s = t**2 * (3 - 2*t)  # smoothstep
    out[mask_band] = (1 - s) * I_band + s * comp(I_band)

    # Clamp to [0,255] and return uint8
    return np.clip(out, 0, 255).astype(np.uint8)


def get_1207_binary_mask():
    video = load_1207_normal_video()
    views.show_frames(video)


    mask = get_femur_mask(video)
    views.show_frames(mask)

    return mask


def get_femur_points(mask):
    
    boundary_points = get_boundary_points(mask, N_lns=128)
    boundary_points = forward_fill_jagged(boundary_points)
    print(boundary_points.shape)
    print(boundary_points[0])
    
    # Get femur tip
    femur_tip = rdl.estimate_femur_tip_boundary(boundary_points, midpoint=0.6)
    femur_tip = forward_fill_jagged(femur_tip)
    femur_tip = rdl.get_centroid_pts(femur_tip)
    femur_tip = rdl.smooth_points(femur_tip, 9) # Moving average filter

    # Get femur midpoint
    femur_midpt = rdl.estimate_femur_midpoint_boundary(boundary_points, start=0.1, end=0.5)
    femur_midpt = forward_fill_jagged(femur_midpt)
    femur_midpt = rdl.get_centroid_pts(femur_midpt)
    femur_midpt = rdl.smooth_points(femur_midpt, 9) # Moving average filter

    return femur_tip, femur_midpt


def main():

    # Get video
    video = load_1207_normal_video()

    # Get Otsu mask
    views.show_frames(video)
    video_hist = utils.blur_video(video)
    video_hist = rdl.match_histograms_video(video_hist, video_hist[175])
    
    otsu_mask = ks.get_otsu_masks(video_hist, 0.65)
    otsu_mask = utils.morph_close(otsu_mask, (55,55))
    views.show_frames(otsu_mask, "otsu_mask")

    # Perform radial segmentation
    mask = io.load_nparray("../data/processed/1207_normal_mask.npy")

    femur_tip, femur_midpt = get_femur_points(mask)

    radial_masks = rdl.label_radial_masks(otsu_mask, femur_tip, femur_midpt, N=64)

    v0 = views.draw_points((otsu_mask*31).astype(np.uint8), femur_tip)
    v0 = views.draw_points(v0, femur_midpt)
    v0 = views.draw_line(v0, femur_midpt, femur_tip)
    views.show_frames(v0)

    v1 = views.draw_mask_boundaries( (otsu_mask*63).astype(np.uint8), radial_masks)
    views.show_frames(v1)

    print("save results if happy")
    breakpoint()
    # Save final results
    # io.save_nparray(video, "../data/processed/1207_normal_radial_video_N16.npy")
    # io.save_nparray(radial_masks, "../data/processed/1207_normal_radial_masks_N16.npy")

if __name__ == "__main__":
    main()

    # Get and save the binary mask
    # mask = get_1207_binary_mask()
    # views.show_frames(mask)
    # io.save_nparray(mask, "../data/processed/1207_normal_mask.npy")    


