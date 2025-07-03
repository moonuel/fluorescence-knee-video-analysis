import os
import sys
import numpy as np
import pandas as pd
import cv2
import sklearn as skl
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
    """Sample interior-boundary points of a binary femur-mask video.

    Parameters
    ----------
    video   : np.ndarray
        Binary fluorescence video mask, shape (n_frames, H, W) with pixel
        values 0 or 255.
    N_lns   : int
        Number of equally-spaced vertical scan lines (columns) to use.

    Returns
    -------
    np.ndarray (dtype=object)
        Length-n_frames object array.  Each element is a Python list of
        (x, y) tuples giving the coordinates of zero-crossing points along
        scan lines *that exhibit exactly four crossings* in that frame.
    """
    if VERBOSE: print("sample_femur_interior_pts() called!")

    video = video.copy()
    nframes, h, w = video.shape

    # Step 1 – choose equally-spaced column indices (x-coords)
    scan_cols = np.linspace(0, w, N_lns+2)
    scan_cols = scan_cols[1:-1] # interior lines only
    scan_cols = np.rint(scan_cols).astype(int) # Round to int column indices

    femur_pts_per_frame = []

    # Step 2 – per-frame scan
    for cf in range(nframes):
        frame = video[cf] # (H, W) binary mask
        valid_pts = []

        for x in scan_cols:
            col = frame[:, x] # 1-D array length H
            # Indices where pixel value changes (0 <-> 255)
            crossings = np.where(col[:-1] != col[1:])[0] + 1  # +1 -> row of change
            if crossings.size == 4:                  # accept columns with 4 crossings
                for y in crossings[1:-1]:
                    valid_pts.append([int(x), int(y)])

        femur_pts_per_frame.append(valid_pts)

    # Step 3 – return as object array (ragged structure)
    femur_pts_per_frame = np.array(femur_pts_per_frame, dtype=object)
    return femur_pts_per_frame

def estimate_femur_tip_boundary(sample_pts:np.ndarray, midpoint:float=0.5) -> np.ndarray:
    """Filters for only the points corresponding to the interior boundary of the femur"""

    print(sample_pts.shape)
    # print(sample_pts)

    sample_pts = sample_pts.copy()
    nfs = sample_pts.shape[0]

    # Select only right half of points 
    femur_pts = []
    for cf in range(nfs):
        pts = np.asarray(sample_pts[cf])
        npts, _ = pts.shape

        # points are stored in pairs
        # divide by 2, then divide by 2, and round 
        # take midpoint to be twice the previous number
        midpt = int(npts/2*midpoint)*2 # TODO: parameterize to use something like right 1/3 of points?

        femur_pt = pts[midpt:, :]
        femur_pts.append(femur_pt)

    return np.array(femur_pts, dtype=object)

def filter_outlier_points(points:np.ndarray, method:str) -> np.ndarray:

    # TODO: filter outlier points using some different methods?

    return NotImplemented

def get_centroid_pts(femur_pts: np.ndarray) -> np.ndarray:
    "Calculates centroid of all points, per frame."

    if VERBOSE: print("get_centroid_pts() called!")

    femur_pts = femur_pts.copy()
    nfs = femur_pts.shape[0] # Ragged array. intended dimensions (nfs, npts, 2_

    centroids = []
    for cf in range(nfs):
        
        pts = femur_pts[cf]
        n = pts.shape[0]

        # Compute centroid of points
        cntrd = np.mean(pts, axis=0, dtype=int)
        centroids.append([cntrd])

    return np.array(centroids, dtype=object)

def get_mask_convex_hull(mask: np.ndarray) -> np.ndarray:
    """Compute the convex‑hull mask for each frame in a binary‑mask video.

    Parameters
    ----------
    mask : np.ndarray
        Binary array of shape (n_frames, H, W).

    Returns
    -------
    np.ndarray
        Array of the same shape, where each frame is replaced by the filled
        convex hull of its foreground pixels.
    """
    mask = mask.copy()
    nfs, h, w = mask.shape
    dtype = mask.dtype

    hull_frames = []

    for cf in range(nfs):
        frame = mask[cf]

        # 1. Collect foreground pixel coordinates as (x, y) points
        ys, xs = np.where(frame > 0)
        if xs.size == 0:                       # empty mask → empty hull
            hull_frames.append(np.zeros((h, w), dtype=dtype))
            continue

        pts = np.stack((xs, ys), axis=1).astype(np.int32)  # shape (N, 2)
        pts = pts.reshape(-1, 1, 2)                       # required by cv2

        # 2. Compute convex hull
        hull = cv2.convexHull(pts)

        # 3. Draw filled convex hull into a blank mask
        hull_mask = np.zeros((h, w), dtype=dtype)
        cv2.fillConvexPoly(hull_mask, hull, (255,255,255))

        hull_frames.append(hull_mask)

    return np.stack(hull_frames, axis=0).astype(dtype)

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
    otsu_mask = utils.morph_erode(otsu_mask, (21,21))
    # otsu_mask = get_mask_convex_hull(otsu_mask) # works as intended but results are not good
    # views.show_frames(np.concatenate([mask, otsu_mask], axis=2), "mask vs boundary mask") # type: ignore
    
    # Exclude intr_mask region outside of otsu mask
    intr_mask = rdl.interior_mask(otsu_mask, mask)
    intr_mask = utils.morph_open(intr_mask, (31,31)) # clean small artifacts
    intr_mask = utils.morph_close(intr_mask, (15,15)) # try to remove the dip

    views.show_frames(np.concatenate([mask, intr_mask], axis=2), "mask vs interior mask")
    # views.show_frames(np.concatenate([video, intr_mask, otsu_mask], axis=2), "video vs interior mask vs otsu boundary")
    # views.draw_mask_boundary(video, intr_mask)

    # Sample points along the interior of the mask 
    sample_pts = sample_femur_interior_pts(intr_mask, 128)

    # Estimate the tip of the femur
    femur_pts = estimate_femur_tip_boundary(sample_pts)
    femur_tip_pts = estimate_femur_tip(femur_pts)

    views.draw_points(video, sample_pts); # All sampling points
    views.draw_points(video, femur_pts); # Filter for only right half of points to estimate the femur tip

    views.draw_points(video, femur_pts)

    return


    # Estimate femur position
    init_guess = ((450//2, 500//2), (20, 500//2) )
    femur_endpts, femur_midpts = estimate_femur_position(intr_mask, init_guess)


    return


if __name__ == "__main__":
    main()
