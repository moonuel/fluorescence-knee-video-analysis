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

def get_closest_pt_to_edge(mask:np.ndarray, edge:str) -> Tuple[int,int]:
    """
    Finds the closest point in a binary mask to one edge of the frame.
    Inputs:
        mask (np.ndarray): A binary image mask.
        edge (str): The edge {t,r,b,l} of the frame we want the point closest to.
    Outputs:
        (int, int): A tuple (x, y) returning the coordinates of the desired point.
    """

    y, x = np.nonzero(mask) # Validate binary mask
    if y.size == 0:
        return None  # No points found

    # Get all closest points to the edge
    edge_funcs = {
        "t": np.argmin(y),  # Top (min y)
        "r": np.argmax(x),   # Right (max x)
        "b": np.argmax(y),  # Bottom (max y)
        "l": np.argmin(x),  # Left (min x)
    }

    i = edge_funcs[edge]
    pt = x[i], y[i]

    return pt

def get_closest_pts_to_edge(video:np.ndarray, edge:str) -> List[Tuple[int,int]]:
    "Gets the closest points to an edge for an entire video. Edge = {t,r,b,l}"
    if VERBOSE: print("get_closest_pts_to_edge() called!")

    pts = []
    for cf, frame in enumerate(video):
        pt = get_closest_pt_to_edge(frame, edge)
        pts.append(pt)
    
    return pts

def get_closest_pt_along_direction(mask: np.ndarray, edge: str, angle_d: int) -> Tuple[int, int]:
    """Rotates the mask by the specified angle and finds the point closest to the specified edge.
    Returns the corresponding point in the original coordinate system.
    """

    mask = mask.copy()
    h, w = mask.shape

    # 1. Rotate the mask
    ctr = (w // 2, h // 2)
    rot_mx = cv2.getRotationMatrix2D(ctr, angle_d, 1.0)
    rotated = cv2.warpAffine(mask, rot_mx, (w, h))

    # 2. Get point in rotated frame
    pt_rot = get_closest_pt_to_edge(rotated, edge)

    # 3. Unrotate point (convert 2x3 rot matrix to 3x3, then invert)
    rot_affine = np.vstack([rot_mx, [0, 0, 1]])  # Make 3x3
    inv_rot_affine = np.linalg.inv(rot_affine)

    pt_hom = np.array([pt_rot[0], pt_rot[1], 1])
    pt_orig = inv_rot_affine @ pt_hom
    pt_orig = tuple(np.round(pt_orig[:2]).astype(int))

    return pt_orig


def _get_N_points_on_circle(circle_ctr:Tuple[int,int], ref_pt:Tuple[int,int], N:int, radius_scale:int=1) -> np.ndarray:
    """Returns N equally spaced points on a circle as a NumPy array.
    
    Args:
        circle_ctr: (x, y) center of the circle.
        ref_pt: (x, y) reference point on the circle.
        N: Number of points to generate.

    Returns:
        NumPy array of shape (N, 2) containing the (x, y) points.
    """
    cx, cy = circle_ctr
    rx, ry = ref_pt
    radius = math.hypot(rx - cx, ry - cy)*radius_scale
    start_angle = math.atan2(ry - cy, rx - cx)
    
    circle_pts = np.zeros((N, 2), dtype=np.int32)
    for i in range(N):
        angle = start_angle - 2 * math.pi * i / N
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        circle_pts[i] = [round(x), round(y)]
    
    return circle_pts

def get_N_points_on_circle(circle_ctrs:List[Tuple[int,int]], ref_pts:List[Tuple[int,int]], N:int, radius_scale:int=1) -> np.ndarray:
    """Gets N points on a circle for an entire video.
    
    Returns:
        NumPy array of shape (frames, N, 2) containing all points,
        or (frames, N, 2) zeros for None inputs.
    """
    if VERBOSE: 
        print("get_N_points_on_circle() called!")

    if len(circle_ctrs) != len(ref_pts):
        raise ValueError("Input lists must have the same length")

    circle_points = []
    for i in range(len(circle_ctrs)):
        if circle_ctrs[i] is None or ref_pts[i] is None:
            # Use zeros for missing frames to maintain array structure
            circle_points.append(np.zeros((N, 2), dtype=np.int32))
            continue
        
        circ_pts = _get_N_points_on_circle(circle_ctrs[i], ref_pts[i], N, radius_scale)
        circle_points.append(circ_pts)
    circle_points = np.array(circle_points)

    return circle_points

def smooth_points(points:List[Tuple[int,int]], window_size:int) -> List[Tuple[int,int]]:
    """Smooths a set of points using a moving average filter"""
    if VERBOSE: print("smooth_points() called!")

    # Compute rolling mean using pandas
    points = pd.DataFrame(points)
    points[0] = points[0].rolling(window_size, min_periods=1, center=True).mean().astype(int)
    points[1] = points[1].rolling(window_size, min_periods=1, center=True).mean().astype(int)

    # Cast back to list of tuples    
    points = list(points.itertuples(index=False, name=None))

    return points

def estimate_femur_position(mask:np.ndarray) -> Tuple[ np.ndarray, np.ndarray]:
    """Estimates the position of the femur based on an adaptive mean mask. Assumes femur is pointing to the left of the screen.
    
    Returns (femur_endpts, femur_midpts), 
        where femur_endpts is the position of the femur inside the knee, 
        and femur_midpts is a set of points somewhere along the femur 
    """
    if VERBOSE: print("estimate_femur_position() called!")

    raise NotImplementedError("Not meant to be called. This function should be designed for each data set.")

    return None, None

def intersect_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Performs frame-wise binary AND over all frames in two 3D binary masks.

    Both mask1 and mask2 must have shape (nframes, height, width).
    """
    assert mask1.shape == mask2.shape, "Masks must have the same shape"
    
    nfs,h,w = mask1.shape

    AND_frs = []
    for cf in range(nfs):
        AND_frs.append(mask1[cf] & mask2[cf])
    AND_frs = np.array(AND_frs, dtype=np.uint8)
    
    return AND_frs

def union_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:

    assert mask1.shape == mask2.shape

    nfs, h, w = mask1.shape

    OR_frs = []
    for cf in range(nfs):
        OR_frs.append(mask1[cf] | mask2[cf])
    OR_frs = np.array(OR_frs, dtype=np.uint8)

    return OR_frs

def combine_masks(masks:np.ndarray) -> np.ndarray:
    """Takes the frame-wise union of all input masks"""

    # TODO: input validation

    masks = masks.copy()
    nmsks, nfrms, h, w = masks.shape
    
    combined_masks = []
    for cf in range (nfrms):
        
        frame = np.zeros((h,w), dtype=np.uint8)

        for mn in range(nmsks):
            frame = frame | masks[mn, cf]

        combined_masks.append(frame)

    combined_masks = np.array(combined_masks)
    return combined_masks

def interior_mask(bndry_mask: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns the portion of the mask in the interior of the boundary mask, for every frame"""
    if VERBOSE: print("interior_mask() called!")

    assert bndry_mask.shape == mask.shape

    intr_mask = mask * (bndry_mask > 0) # Zero-out all elements outside the boundary! 

    return intr_mask

def combine_masks(masks:np.ndarray) -> np.ndarray:
    """Takes the frame-wise union of all input masks"""

    # TODO: input validation

    masks = masks.copy()
    nmsks, nfrms, h, w = masks.shape
    
    combined_masks = []
    for cf in range (nfrms):
        
        frame = np.zeros((h,w), dtype=np.uint8)

        for mn in range(nmsks):
            frame = frame | masks[mn, cf]

        combined_masks.append(frame)

    combined_masks = np.array(combined_masks)
    return combined_masks

def _get_N_points_on_circle(circle_ctr:Tuple[int,int], ref_pt:Tuple[int,int], N:int, radius_scale:int=1) -> np.ndarray:
    """Returns N equally spaced points on a circle as a NumPy array.
    
    Args:
        circle_ctr: (x, y) center of the circle.
        ref_pt: (x, y) reference point on the circle.
        N: Number of points to generate.

    Returns:
        NumPy array of shape (N, 2) containing the (x, y) points.
    """
    cx, cy = circle_ctr
    rx, ry = ref_pt
    radius = math.hypot(rx - cx, ry - cy)*radius_scale
    start_angle = math.atan2(ry - cy, rx - cx)
    
    circle_pts = np.zeros((N, 2), dtype=np.int32)
    for i in range(N):
        angle = start_angle - 2 * math.pi * i / N
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        circle_pts[i] = [round(x), round(y)]
    
    return circle_pts

def get_N_points_on_circle(circle_ctrs:List[Tuple[int,int]], ref_pts:List[Tuple[int,int]], N:int, radius_scale:int=1) -> np.ndarray:
    """Gets N points on a circle for an entire video.
    
    Returns:
        NumPy array of shape (frames, N, 2) containing all points,
        or (frames, N, 2) zeros for None inputs.
    """
    if VERBOSE: 
        print("get_N_points_on_circle() called!")

    if len(circle_ctrs) != len(ref_pts):
        raise ValueError("Input lists must have the same length")

    circle_points = []
    for i in range(len(circle_ctrs)):
        if circle_ctrs[i] is None or ref_pts[i] is None:
            # Use zeros for missing frames to maintain array structure
            circle_points.append(np.zeros((N, 2), dtype=np.int32))
            continue
        
        circ_pts = _get_N_points_on_circle(circle_ctrs[i], ref_pts[i], N, radius_scale)
        circle_points.append(circ_pts)
    circle_points = np.array(circle_points)

    return circle_points

def sample_femur_interior_pts(mask: np.ndarray, N_lns: int) -> np.ndarray:
    """Sample interior-boundary points of a binary femur-mask video.

    Parameters
    ----------
    mask   : np.ndarray
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

    mask = mask.copy()
    nframes, h, w = mask.shape

    # Step 1 – choose equally-spaced column indices (x-coords)
    scan_cols = np.linspace(0, w, N_lns+2)
    scan_cols = scan_cols[1:-1] # interior lines only
    scan_cols = np.rint(scan_cols).astype(int) # Round to int column indices

    femur_pts_per_frame = []

    # Step 2 – per-frame scan
    for cf in range(nframes):
        frame = mask[cf] # (H, W) binary mask
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

        if pts.size == 0: 
            femur_pts.append([]) # append empty
            continue

        npts, _ = pts.shape

        # points are stored in pairs
        # divide by 2, then divide by 2, and round 
        # take midpoint to be twice the previous number
        midpt = int(npts/2*midpoint)*2 # TODO: parameterize to use something like right 1/3 of points?

        femur_pt = pts[midpt:, :]
        femur_pts.append(femur_pt)

    return np.array(femur_pts, dtype=object)

def get_centroid_pts(femur_pts: np.ndarray) -> np.ndarray:
    "Calculates centroid of all points, per frame."

    if VERBOSE: print("get_centroid_pts() called!")

    femur_pts = femur_pts.copy()
    nfs = femur_pts.shape[0] # Ragged array. intended dimensions (nfs, npts, 2_

    centroids = []
    for cf in range(nfs):
        
        pts = np.asarray(femur_pts[cf])
        # n = pts.shape[0]

        # Compute centroid of points
        cntrd = np.mean(pts, axis=0, dtype=int)

        # Filter out points outside 
        centroids.append([cntrd])

    return np.array(centroids, dtype=object)

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

        # print(points[cf]) # Debug
        # print(centroids[cf])
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

def get_radial_segments(video:np.ndarray, circle_ctrs:np.ndarray, circle_pts:np.ndarray, thresh_scale:int=0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Gets the radial segments for the video. """
    if VERBOSE: print("get_radial_segments() called!")

    video = video.copy()
    circle_ctrs = np.array(circle_ctrs) # type cast for downstream operations for safety
    circle_pts = np.array(circle_pts)

    # TODO: input validation

    # Get Otsu masks
    otsu_masks = ks.get_otsu_masks(video, thresh_scale=thresh_scale)
    # views.show_frames(otsu_masks) # Validate otsu masks

    # Get bisection mask for every point on the circle
    nfs, h, w = video.shape
    _, N, _ = circle_pts.shape
    bsct_masks = np.empty((N, nfs, h,w), dtype=np.uint8) # dimensions (N_masks, nframes, h, w)
    for n in range(N):
        bsct_masks[n] = ks.get_bisecting_masks(video, circle_pts[:,n], circle_ctrs) 
        # views.show_frames(bsct_masks[n]) # Validate bisecting masks

    # Get radial slices
    radial_slices = np.empty((N, nfs, h,w), dtype=np.uint8) # dimensions (N_masks, nframes, h, w)
    for n in range(N):
        radial_slices[n] = intersect_masks(bsct_masks[n], ~bsct_masks[n-1])
        # views.show_frames(radial_slices[n]) # Validate radial slices

    # Get radial masks
    radial_masks = np.empty((N, nfs, h,w), dtype=np.uint8) # dimensions (N_masks, nframes, h, w)
    for n in range(N):
        radial_masks[n] = intersect_masks(radial_slices[n], otsu_masks)
        # views.show_frames(radial_masks[n]) # Validate radial masks

    # Get radial regions
    otsu_region = intersect_masks(otsu_masks, video)
    radial_regions = np.empty((N, nfs, h,w), dtype=np.uint8) # dimensions (N_masks, nframes, h, w)
    for n in range(N):
        radial_regions[n] = intersect_masks(radial_slices[n], otsu_region)
        # views.show_frames(radial_masks[n]) # Validate radial regions

    return radial_regions, radial_masks