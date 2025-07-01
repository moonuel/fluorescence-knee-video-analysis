import os
import sys
import numpy as np
import pandas as pd
import cv2
import math
import src.core.knee_segmentation as ks
from typing import Tuple, List
from src.utils import io, views, utils
from src.config import VERBOSE
from src.core import data_processing as dp
import src.core.radial_segmentation as rdl


def estimate_femur_position(mask:np.ndarray, init_guess:Tuple[int,int]) -> Tuple[np.ndarray, np.ndarray]:
    """Estimates the position of the femur based on an adaptive mean mask. Assumes femur is pointing to the left of the screen.
    init_guess = (endpt, midpt)
    
    Returns (femur_endpts, femur_midpts), 
        where femur_endpts is the position of the femur inside the knee, 
        and femur_midpts is a set of points somewhere along the femur 
    """
    if VERBOSE: print("estimate_femur_position() called!")

    mask = mask.copy()
    nframes, h, w = mask.shape

    # IDEA: for every frame, estimate the femur position. 
    # Then, align the video by centering it based on the femur position.
    # Then, estimate the next femur position
    # Greedy-type approach to reliably estimating the femur position?

    endpt, midpt = init_guess # (x1,y1), (x2,y2)
    total_angle = 0

    for cf, frame in enumerate(mask):

        # Get angle between femur estumation and left horizontal axis
        (x1,y1), (x2,y2) = endpt, midpt
        agl = np.arctan2(y2-y1, x2-x1)
        agl = np.degrees(agl) + 180 # angle with left horizontal axis
        total_angle += agl

        views.draw_text(frame, str(agl))

        # Rotate frame to flatten horizontal line
        ctr = (w//2, h//2)
        rot_mx = cv2.getRotationMatrix2D(ctr, total_angle, 1.0) 
        frame = cv2.warpAffine(frame, rot_mx, (w,h)) 
        
        # cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), 1)

        """Estimate the midpoint"""

        # Split into top/bottom slices
        slc_wdh = 70 # pixels. to be manually set 
        top_slc = frame[h//2 - slc_wdh : h//2, :]
        btm_slc = frame[h//2 : h//2 + slc_wdh , :]

        # Get leftmost pts
        topl = rdl.get_closest_pt_along_direction(top_slc, "b", 45)
        btml = rdl.get_closest_pt_along_direction(btm_slc, "t", -45)

        # topl = rdl.get_closest_pt_to_edge(top_slc, "l")
        # btml = rdl.get_closest_pt_to_edge(btm_slc, "l")
        # # cv2.circle(frame, topl, 3, (255,255,255), -1) # Validate leftmost pts
        # # cv2.circle(btm_slc, btml, 3, (255,255,255), -1)

        # Translate back into coordinates of original frame
        topl = list(topl)
        topl[1] = topl[1] + h//2 - slc_wdh
        topl = tuple(topl)

        btml = list(btml)
        btml[1] = btml[1] + h//2
        btml = tuple(btml)

        cv2.circle(frame, topl, 3, (255,255,255), -1) # Validate translated points
        cv2.circle(frame, btml, 3, (255,255,255), -1)

        # Get left midpoint
        midl = (np.array(btml) + np.array(topl)) // 2
        midl = tuple(midl)
        cv2.circle(frame, midl, 3, (255,255,255), -1) # Validate middle point

        # Update femur estimation
        endpt = endpt
        midpt = midl
        cv2.circle(frame, endpt, 3, (255,255,255), -1)

        # TODO: Set femur endpoint in the middle of the frame 

        # TODO: unrotate coords

        # TODO: plot points on frame

        cv2.imshow("postrotation", frame)
        cv2.imshow("topslc", top_slc)
        cv2.imshow("btmslc", btm_slc)
        if cv2.waitKey(0) == ord('q'): break

    cv2.destroyAllWindows()
    return

    # return femur_endpts, femur_midpts

def sample_femur_interior_pts(video: np.ndarray, N_lns: int) -> np.ndarray:
    """
    For each frame in a binary fluorescence mask video, place N_lns
    equally spaced vertical scan lines and keep the (x,y) coordinates
    where each line exhibits exactly four 0↔255 transitions (“zero-
    crossings”), corresponding to the femur's interior boundary.

    Parameters
    ----------
    video   : np.ndarray  # shape (n_frames, H, W), dtype binary (0/255)
    N_lns   : int         # number of vertical scan lines to test

    Returns
    -------
    np.ndarray (ragged, dtype=object)
        One-dimensional object array of length n_frames.  Each element
        is a Python list of (x,y) integer tuples giving the retained
        crossing coordinates for that frame.
    """
    video = video.copy()
    nframes, h, w = video.shape

    # Step 1 – choose scan columns (x‑indices)
    scan_cols = np.linspace(0, w - 1, N_lns)
    scan_cols = np.rint(scan_cols).astype(int)          # shape (N_lns,)

    femur_pts_per_frame: list[list[tuple[int, int]]] = []

    # Step 2 – per‑frame scan and zero‑crossing detection
    for cf in range(nframes):
        frame = video[cf]                               # (H, W)
        valid_pts: list[tuple[int, int]] = []

        for x in scan_cols:
            col = frame[:, x]                           # 1‑D column
            changes = np.where(col[:-1] != col[1:])[0] + 1  # crossing rows

            if changes.size == 4:                       # keep only 4‑crossing lines
                valid_pts.extend((int(x), int(y)) for y in changes)

        femur_pts_per_frame.append(valid_pts)

    # Step 3 – return as ragged NumPy object array
    return np.array(femur_pts_per_frame, dtype=object)

def main():

    # Import normal knee data
    video = io.load_nparray("../data/processed/normal_knee_processed.npy")
    video = np.rot90(video, k=-1, axes=(1,2))
    video = utils.crop_video_square(video, int(500*np.sqrt(2)))

    # Slight rotation
    angle = -15
    video = utils.rotate_video(video, angle)
    video = utils.crop_video_square(video, 500, 450)

    # Remove first 44 frames
    srt_fm = 45
    video = video[srt_fm:]
    # views.show_frames(video)

    # Get adaptive mean mask
    mask_src = utils.log_transform_video(video)
    mask_src = utils.blur_video(video, (41,41), sigma=0) # sigma is variance
    mask = utils.mask_adaptive(mask_src, 141, 14) # increase thresholding to get better femur boundary
    # mask = utils.morph_open(mask, (31,31)) # clean small artifacts
    # views.show_frames(mask)

    # Get otsu mask
    otsu_mask = ks.get_otsu_masks(mask_src, thresh_scale=0.5)
    otsu_mask = utils.morph_close(otsu_mask, (31,31)) # Close gaps in otsu mask
    # views.show_frames(np.concatenate([mask, otsu_mask], axis=2), "mask vs boundary mask") # type: ignore
    
    # Exclude intr_mask region outside of otsu mask
    intr_mask = rdl.interior_mask(otsu_mask, mask)
    intr_mask = utils.morph_open(intr_mask, (31,31)) # clean small artifacts
    intr_mask = utils.morph_close(intr_mask, (15,15)) # try to remove the dip

    views.show_frames(np.concatenate([mask, intr_mask], axis=2), "mask vs interior mask")
    views.show_frames(np.concatenate([video, intr_mask, otsu_mask], axis=2), "video vs interior mask vs otsu boundary")

    views.draw_mask_boundary(video, intr_mask)

    estimate_femur_position


    return


    # Estimate femur position
    init_guess = ((450//2, 500//2), (20, 500//2) )
    femur_endpts, femur_midpts = estimate_femur_position(intr_mask, init_guess)


    return


if __name__ == "__main__":
    main()
