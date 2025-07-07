import os
import sys
import numpy as np
import pandas as pd
import cv2
import sklearn.cluster as sklc 
import hdbscan
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

def estimate_femur_midpoint_boundary(sample_pts:np.ndarray, midpoint:float=0.5) -> np.ndarray:
    """Estimates a point along the length of the femur, for every frame"""
    if VERBOSE: print("estimate_femur_midpoint_boundary() called!")

    sample_pts = sample_pts.copy()
    nfs = sample_pts.shape[0]

    femur_midpts = []
    for cf in range(nfs):
        pts = np.asarray(sample_pts[cf])
        npts = pts.shape[0]

        # Points are stored in pairs
        midpt = int(npts/2*midpoint)*2

        # Get the two adjacent points stored at the midpoint
        femur_midpt = ((pts[midpt] + pts[midpt+1]) / 2).astype(int)

        femur_midpts.append(femur_midpt)

    return np.array(femur_midpts)

def filter_outlier_points_dbscan(points: np.ndarray,
                          eps: float = 15.0,
                          min_samples: int = 5) -> np.ndarray:
    """Remove outlier 2-D points in each frame via DBSCAN.

    Parameters
    ----------
    points : np.ndarray (jagged, dtype=object or list-like)
        Shape (n_frames,), each element a (n_pts_i, 2) float/-int array.
    eps : float
        DBSCAN `eps` radius (default 15 px).
    min_samples : int
        DBSCAN `min_samples` (default 5).

    Returns
    -------
    np.ndarray (dtype=object)
        Filtered per-frame arrays; still jagged: (n_frames,) where
        each entry is (n_kept_i, 2).
    """
    print("filter_outlier_points_dbscan() called!")

    filtered_frames = []

    for cpts in points:
        # Guard: empty frame
        if cpts.size == 0 or cpts.shape[0] < min_samples:
            filtered_frames.append(np.empty((0, 2), dtype=cpts.dtype))
            continue

        # Run DBSCAN
        labels = sklc.DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cpts)

        # Keep only core/edge points (label != -1)
        keep_mask = labels != -1
        filtered_frames.append(cpts[keep_mask])

    # Return as an object-dtype array to keep jagged structure
    return np.array(filtered_frames, dtype=object)

def filter_outlier_points_hdbscan(points: np.ndarray,
                          min_cluster_size:int = 5,
                          allow_single_cluster:bool = False) -> np.ndarray:
    """Remove outlier 2-D points in each frame via HDBSCAN.

    Parameters
    ----------
    points : np.ndarray (jagged, dtype=object or list-like)
        Shape (n_frames,), each element a (n_pts_i, 2) float/-int array.
    eps : float
        DBSCAN `eps` radius (default 15 px).
    min_samples : int
        DBSCAN `min_samples` (default 5).

    Returns
    -------
    np.ndarray (dtype=object)
        Filtered per-frame arrays; still jagged: (n_frames,) where
        each entry is (n_kept_i, 2).
    """
    print("filter_outlier_points_hdbscan() called!")

    filtered_frames = []

    for cpts in points:
        # Guard: empty frame
        if cpts.size == 0:
            filtered_frames.append(np.empty((0, 2), dtype=cpts.dtype))
            continue

        # Run HDBSCAN
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=allow_single_cluster).fit_predict(cpts)

        # Keep only core/edge points (label != -1)
        keep_mask = labels != -1
        filtered_frames.append(cpts[keep_mask])

    # Return as an object-dtype array to keep jagged structure
    return np.array(filtered_frames, dtype=object)

def filter_outlier_points_centroid(points: np.ndarray, eps: float) -> np.ndarray:
    """Exclude points farther than `eps` from the centroid in each frame.

    Parameters
    ----------
    points : np.ndarray (dtype=object)
        Jagged array of shape (n_frames,), with each element an (n_pts_i, 2) array.
    eps : float
        Radius threshold; points with distance > eps are dropped.

    Returns
    -------
    np.ndarray (dtype=object)
        Jagged array of filtered point sets, shape (n_frames,).
    """
    if VERBOSE:
        print("filter_outlier_points_centroid() called!")

    points = points.copy()                     # keep original intact
    nfs = points.shape[0]

    centroids = get_centroid_pts(points)       # shape (n_frames, 2)
    filtered = []

    for cf in range(nfs):
        pts = np.asarray(points[cf], dtype=float)                       # (n_pts_i, 2)
        ctr = np.asarray(centroids[cf], dtype=float)                    # (2,)

        if pts.size == 0:
            filtered.append(pts)               # keep empty frame as‑is
            continue

        # Euclidean distance to centroid
        dists = pts - ctr
        dists = np.linalg.norm(dists, axis=1)
        keep_mask = dists <= eps

        print(cf, np.max(dists))

        filtered.append(pts[keep_mask].astype(int))

    return np.array(filtered, dtype=object)

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

        # Filter out points outside 
        centroids.append([cntrd])

    return np.array(centroids, dtype=object)

def get_mask_convex_hull(mask: np.ndarray) -> np.ndarray:
    """Compute the convex-hull mask for each frame in a binary-mask video.

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

def get_pts_convex_hull(points: np.ndarray, video:np.ndarray) -> np.ndarray:
    """Compute the convex-hull based on a set of points given for every frame in a video.

    Parameters
    ----------
    points : np.ndarray
        Jagged array of shape (n_frames,), each entry an (npts, 2) array of [x, y] coords.

    frame_shape : tuple
        Height and width of the binary output mask (default is 512x512).

    Returns
    -------
    np.ndarray
        Binary array of shape (n_frames, H, W), with the filled convex hull for each frame.
    """
    nfs = points.shape[0]
    nfs, h, w = video.shape
    conv_hull = []

    for cf in range(nfs):
        pts = points[cf].reshape(-1,1,2).astype(np.uint8)
        mask = np.zeros((h,w), dtype=np.uint8)

        # print(pts)

        # if len(pts) >= 3:
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        conv_hull.append(mask)

    print(np.array(conv_hull).shape)

    return np.array(conv_hull)

def generate_convex_hull_mask(points_per_frame, video):
    """
    Draw convex hulls on a binary mask for each frame.

    Parameters
    ----------
    points_per_frame : list of (n_i, 2) np.ndarrays
        List of per-frame point arrays with shape (n_i, 2). Each array contains
        float (x, y) coordinates. NaNs and infs will be ignored.
    video : np.ndarray, shape (n_frames, height, width)
        The reference video array, only used for shape.

    Returns
    -------
    mask : np.ndarray, shape (n_frames, height, width)
        Binary mask with convex hulls drawn for each frame.
    """
    n_frames, height, width = video.shape
    mask = np.zeros((n_frames, height, width), dtype=np.uint8)

    for i, pts in enumerate(points_per_frame):
        if pts is None or len(pts) < 3:
            continue
        pts = np.asarray(pts, dtype=np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 3:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask[i], hull, 1)

    return mask

def main():
    """Performs the radial segmentation analysis on the normal knee data. 
    
    Steps:
    - Use adaptive thresholding mask to get a good boundary around the femur
    - Use otsu thresholding mask to clean up artifacts along the borders of the frame
    - Discretize the boundary around the femur tip by sampling points along the interior of the resulting mask 
    - Remove outliers and calculate the centroid of the points around the femur tip 

    x Estimate a point along the length of the femur 
    x Estimate a line representing the position of the femur 
    x Perform the radial segmentation analysis

    """

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

    # views.show_frames(np.concatenate([mask, intr_mask], axis=2), "mask vs interior mask")
    # views.show_frames(np.concatenate([video, intr_mask, otsu_mask], axis=2), "video vs interior mask vs otsu boundary")
    # views.draw_mask_boundary(video, intr_mask)

    # Sample points along the interior of the mask 
    sample_pts = sample_femur_interior_pts(intr_mask, N_lns=128)
    # views.draw_points(video, sample_pts)

    # Estimate the tip of the femur
    femur_bndry = estimate_femur_tip_boundary(sample_pts, 0.45)
    # femur_bndry_filtered = filter_outlier_points_dbscan(femur_bndry, eps=20, min_samples=2) # not great 
    # femur_bndry_filtered = filter_outlier_points_hdbscan(femur_bndry, min_cluster_size=5, allow_single_cluster=True) # better but not perfect
    femur_bndry_filtered = filter_outlier_points_centroid(femur_bndry, 75) # Best results so far but rigid due to fixed threshold value?
    femur_tip = get_centroid_pts(femur_bndry_filtered)

    # Smooth femur tip points
    femur_tip = np.reshape(femur_tip, newshape=(-1, 2)) # Resize for rdl.smooth_points() 
    femur_tip = rdl.smooth_points(femur_tip, window_size=7) # Smooth points
    femur_tip = np.array(femur_tip) # Cast back to array
    femur_tip = np.reshape(femur_tip, (-1, 1, 2)) # Reshape for views.draw_points()

    pvw1 = views.draw_points(video, femur_bndry); # All sampling points
    pvw = views.draw_points(video, femur_bndry_filtered, show_video=False); # Femur tip points only (already filtered)
    pvw = views.draw_points(pvw, femur_tip, show_video=False)
    # views.show_frames(np.concatenate([pvw1, pvw], axis=2))

    # Estimate midpoint of femur
    femur_midpoint = estimate_femur_midpoint_boundary(sample_pts, 0.3)
    femur_midpoint = np.reshape(femur_midpoint, (-1, 2)) # Reshape for coordinate smoothing
    femur_midpoint = rdl.smooth_points(femur_midpoint, window_size=7)
    femur_midpoint = np.array(femur_midpoint)
    femur_midpoint = np.reshape(femur_midpoint, (-1, 1, 2)) # Reshape back to expected format for views.draw_points()
    
    # print(femur_midpoint)
    pvw2 = views.draw_points(pvw, femur_midpoint)
    views.show_frames(np.concatenate([pvw, pvw2], axis=2))

    # Draw convex hull. TODO since refinement is less important
    # femur_convex_hull = get_pts_convex_hull(femur_bndry_filtered, video) # Not working
    # femur_convex_hull = generate_convex_hull_mask(femur_bndry_filtered, video) # Not working
    # views.show_frames(femur_convex_hull)


    # return


    # Estimate femur position
    # init_guess = ((450//2, 500//2), (20, 500//2) )
    # femur_endpts, femur_midpts = estimate_femur_position(intr_mask, init_guess)


    return


    # Estimate femur position
    init_guess = ((450//2, 500//2), (20, 500//2) )
    femur_endpts, femur_midpts = estimate_femur_position(intr_mask, init_guess)


    return


if __name__ == "__main__":
    main()
