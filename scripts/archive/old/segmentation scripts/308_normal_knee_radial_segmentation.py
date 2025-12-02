"""
Legacy script for segmenting the normal 308 knee. Succeeded by pipeline-based script
"""


import os
import sys
import numpy as np
import pandas as pd
import cv2
import sklearn.cluster as sklc 
# import hdbscan # Outdated on env version 1.1
import math
import core.knee_segmentation as ks
from typing import Tuple, List
from utils import io, views, utils
from config import VERBOSE
from core import data_processing as dp
import core.radial_segmentation as rdl
from pathlib import Path
import pdb


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

    # print(sample_pts.shape)
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

def estimate_femur_midpoint_boundary(sample_pts:np.ndarray, start:float = 0.0, end:float=0.5) -> np.ndarray:
    """Gets the points on the boundary around a point along the length of the femur, for every frame"""
    if VERBOSE: print("estimate_femur_midpoint_boundary() called!")

    sample_pts = sample_pts.copy()
    nfs = sample_pts.shape[0] # shape (nfs, npts*, 2), where * indicates the jagged dimension

    midpoint_boundary = []
    for cf in range(nfs):
        pts = np.asarray(sample_pts[cf])
        npts = pts.shape[0] # shape (npts, 2)

        # Top and bottom boundary points are stored in pairs
        strt_idx = int(npts/2*start)*2
        end_idx = int(npts/2*end)*2

        # Get the boundary points between the start and end indices
        midpt_bndry = pts[strt_idx:end_idx]

        midpoint_boundary.append(midpt_bndry)

    return np.array(midpoint_boundary, dtype=object)

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

# def filter_outlier_points_hdbscan(points: np.ndarray,
#                           min_cluster_size:int = 5,
#                           allow_single_cluster:bool = False) -> np.ndarray:
#     """Remove outlier 2-D points in each frame via HDBSCAN.

#     Parameters
#     ----------
#     points : np.ndarray (jagged, dtype=object or list-like)
#         Shape (n_frames,), each element a (n_pts_i, 2) float/-int array.
#     eps : float
#         DBSCAN `eps` radius (default 15 px).
#     min_samples : int
#         DBSCAN `min_samples` (default 5).

#     Returns
#     -------
#     np.ndarray (dtype=object)
#         Filtered per-frame arrays; still jagged: (n_frames,) where
#         each entry is (n_kept_i, 2).
#     """
#     print("filter_outlier_points_hdbscan() called!")

#     filtered_frames = []

#     for cpts in points:
#         # Guard: empty frame
#         if cpts.size == 0:
#             filtered_frames.append(np.empty((0, 2), dtype=cpts.dtype))
#             continue

#         # Run HDBSCAN
#         labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=allow_single_cluster).fit_predict(cpts)

#         # Keep only core/edge points (label != -1)
#         keep_mask = labels != -1
#         filtered_frames.append(cpts[keep_mask])

#     # Return as an object-dtype array to keep jagged structure
#     return np.array(filtered_frames, dtype=object)

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

def analyze_all_aging_knees(video, radial_masks, radial_regions, show_figs=True, save_figs=False, figsize=(9,17)):

    sheet_nums = [0,1,2]

    for num in sheet_nums:
        print(f"\n=== Processing normal-{num} ===")

        # Manually assign left/middle/right knee
        lft = (14,1)
        mdl = (9,14)
        rgt = (1,9)

        l_mask = rdl.combine_masks(rdl.circular_slice(radial_masks, lft)) # 0 and 14-15
        m_mask = rdl.combine_masks(rdl.circular_slice(radial_masks, mdl))
        r_mask = rdl.combine_masks(rdl.circular_slice(radial_masks, rgt))

        l_region = rdl.combine_masks(rdl.circular_slice(radial_regions, lft)) # 0 and 14-15
        m_region = rdl.combine_masks(rdl.circular_slice(radial_regions, mdl))
        r_region = rdl.combine_masks(rdl.circular_slice(radial_regions, rgt))
        
        # Get metadata for comparison with normal knee manual segmentation
        _, metadata = io.load_normal_knee_coords("../data/xy coordinates for knee imaging 0913.xlsx", sheet_num=num)
        masks = {'l': l_mask, 'm': m_mask, 'r': r_mask} 
        regions = {'l': l_region, 'm': m_region, 'r': r_region}
        keys = ['l','m','r']

        # Get intensity data
        raw_intensities = dp.measure_region_intensities(regions, masks, keys)
        normalized_intensities = dp.measure_region_intensities(regions, masks, keys, normalized=True)
        radial_intensities = dp.measure_radial_intensities(np.array([l_region, m_region, r_region]))
        # print(raw_intensities)
        # print(metadata)

        # Validate intensity data
        views.plot_three_intensities(raw_intensities, metadata, show_figs, save_figs, vert_layout=True, figsize=figsize)
        views.plot_three_intensities(normalized_intensities, metadata, show_figs, save_figs, vert_layout=True, figsize=figsize, normalized=True)
        # views.plot_radial_segment_intensities(radial_intensities, f0=1, fN=None)

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


def get_femur_mask(video:np.ndarray) -> np.ndarray:
    """Intersects the adaptive mean mask with the Otsu mask to get a good mask around the femur"""

    # Get outer mask
    out_mask = ks.get_otsu_masks(video, 0.5)
    out_mask = utils.morph_erode(out_mask, (25,25))

    # Get inner mask
    in_mask = ks.mask_adaptive(video, 141, 14)

    # Exclude noise from inner mask
    femur_mask = rdl.interior_mask(out_mask, in_mask)
    femur_mask = utils.morph_open(femur_mask, (11,11))

    femur_mask = utils.morph_close(femur_mask, (27,27))

    # Manual refinements
    femur_mask[:, 330:, :] = 0 
    femur_mask[:, :121, :] = 0
    femur_mask[:, :, :143] = 0

    return femur_mask


def get_mask(video):
    video_hist = rdl.match_histograms_video(video)
    mask = ks.get_otsu_masks(video_hist, 0.5)
    mask = utils.morph_open(mask, (5, 5))
    mask = utils.morph_close(mask, (15,15))
    mask = utils.blur_video(mask, (11,11))
    mask = mask > 0

    return mask

def main():
    """Performs the radial segmentation analysis on the normal knee data. 
    
    Steps:
    - Use adaptive thresholding mask to get a good boundary around the femur
    - Use otsu thresholding mask to clean up artifacts along the borders of the frame
    - Discretize the boundary around the femur tip by sampling points along the interior of the resulting mask 
    - Remove outliers and calculate the centroid of the points around the femur tip 

    > Estimate a point along the length of the femur 
    > Estimate a line representing the position of the femur 
    > Perform the radial segmentation analysis

    """

    # Import normal knee data
    # Use absolute path to ensure it works regardless of CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    input_path = os.path.join(project_root, "data", "processed", "normal_knee_processed.npy")
    
    video = io.load_nparray(input_path)
    video = np.rot90(video, k=-1, axes=(1,2))
    video = utils.center_crop(video, int(500*np.sqrt(2)))
    
    # Slight rotation
    angle = -15
    video = utils.rotate_video(video, angle)
    video[video == 0] = 22 # fill empty pixels
    video = utils.center_crop(video, 500, 450)
    video = np.flip(video, axis=2)

    # Get femur mask
    femur_mask = get_femur_mask(video)

    # Get boundary points
    bndry = rdl.sample_femur_interior_pts(femur_mask, N_lns=128)

    # Estimate tip and midpoint
    tip_bndry = rdl.estimate_femur_tip_boundary(bndry, 0.4)
    tip = rdl.get_centroid_pts(tip_bndry)
    tip = rdl.smooth_points(tip, 9)

    midpt_bndry = rdl.estimate_femur_midpoint_boundary(bndry, 0.1, 0.5)
    midpt = rdl.get_centroid_pts(midpt_bndry)
    midpt = rdl.smooth_points(midpt, 9)

    # views.draw_points(video, np.concat([tip, midpt], axis=1), True)

    # Get radial segmentation
    mask = get_mask(video)

    radial_mask = rdl.label_radial_masks(mask, tip, midpt, 64)


    v1 = views.show_frames(radial_mask * (255 // 64))

    views.draw_points(v1, np.concat([tip, midpt], axis=1), True)

    fn = os.path.join(project_root, "data", "processed", "308_normal_radial_masks_N64.npy")
    
    if input(f"Save to file {fn}? Press 'y' to confirm.\n") == 'y': 
        io.save_nparray(radial_mask, fn)
        fn_vid = fn.replace("masks", "video")
        io.save_nparray(video, fn_vid)
        print(f"Files saved:\n\t{fn}\n\t{fn_vid}")
    else: print("File not saved.")

    return




if __name__ == "__main__":
    main()
