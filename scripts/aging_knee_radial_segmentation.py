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

def get_N_points_on_circle(circle_ctr:Tuple[int,int], first_pt:Tuple[int,int], N: int) -> List[Tuple[int, int]]:
    """Returns a list of N equally spaced points on a circle, arranged clockwise.
    
    The circle is defined by:
    - Center point `circle_ctr`
    - Radius = distance between `circle_ctr` and `first_pt`
    - First point is `first_pt`, followed by the remaining points in clockwise order.

    Args:
        circle_ctr: (x, y) center of the circle.
        first_pt: (x, y) reference point on the circle (first point in the output).
        N: Number of points to generate. If N=1, returns [first_pt].

    Returns:
        List of (x, y) tuples representing the N points on the circle.
    """

    cx, cy = circle_ctr
    rx, ry = first_pt
    radius = math.hypot(rx - cx, ry - cy)
    start_angle = math.atan2(ry - cy, rx - cx)  # Angle of first_pt
    
    points = []
    for i in range(N):
        angle = start_angle - 2 * math.pi * i / N  # Clockwise
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((round(x), round(y)))
    
    return points

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

def estimate_femur_position(mask:np.ndarray) -> Tuple[ List[Tuple[int,int]], List[Tuple[int,int]] ]:
    """Estimates the position of the femur based on an adaptive mean mask. Assumes femur is pointing to the left of the screen.
    
    Returns (femur_endpts, femur_midpts), 
        where femur_endpts is the position of the femur inside the knee, 
        and femur_midpts is a set of points somewhere along the femur 
    """
    if VERBOSE: print("estimate_femur_position() called!")

    # Split frame along the middle. TODO: Parameterize the split line?
    split = 0.5
    nframes,h,w = mask.shape
    mask_top = mask[:,0:int(h*split),:]
    mask_btm = mask[:,int(h*split):,:]

    # Get left-most points on top/bottom halves
    topl_pts = get_closest_pts_to_edge(mask_top, "l")
    btml_pts_ = get_closest_pts_to_edge(mask_btm, "l")

    # views.draw_point(mask_top, topl_pts, True) # Validate left-most points
    # views.draw_point(mask_btm, btml_pts_, True)

    # Convert bottom-left coords to the whole mask
    btml_pts = [None] # to maintain 1-indexing 
    for pt in btml_pts_[1:]:
        pt = list(pt) # for mutability
        pt[1] = pt[1] + int(h*split) 
        btml_pts.append(tuple(pt))

    # views.draw_line(mask, topl_pts, btml_pts) # Validate drawn line

    # Get midpoint of left line 
    midl_pts = [None]
    for cf in range(1, len(mask)):
        topl_pt = np.array(topl_pts[cf])
        btml_pt = np.array(btml_pts[cf])
        midl_pt = (topl_pt + btml_pt)//2
        midl_pts.append(tuple(midl_pt))

    # Smooth midpoints
    midl_pts[1:] = smooth_points(midl_pts[1:], 5)

    views.draw_point(mask, midl_pts) # Validate midpoint

    frame_ctr = [(w//2,h//2)]*nframes
    views.draw_line(mask, midl_pts, frame_ctr) # Validate basic femur estimation

    

    femur_endpts = frame_ctr
    femur_midpts = midl_pts

    return femur_endpts, femur_midpts

def main():
    if VERBOSE: print("main() called!")

    # Load pre-processed video
    # video, _ = knee.pre_process_video(video) 
    video = io.load_nparray("../data/processed/aging_knee_processed.npy") # result of above function call

    # Pre-process video
    video = np.rot90(video, k=-1, axes=(1,2))
    video = utils.crop_video_square(video, int(350*np.sqrt(2))) 

    # Slight rotation
    angle = -29
    video = utils.rotate_video(video, angle)
    video = utils.crop_video_square(video, 350) # crop out black borders
    # views.draw_middle_lines(mask, show_video=True) # Validate rotation

    # Get adaptive mean mask
    video_blr = utils.blur_video(video, (31,31), 0)
    mask = utils.mask_adaptive(video_blr, 71, -2)
    mask = utils.morph_open(mask, (15,15)) # clean small 
    views.view_frames(mask) # Validate mask

    # > TODO: Get the leftmost points
    # > TODO: Get the basic femur estimation
    # x TODO: Brainstorm femur endpoint estimation improvements
    # x TODO: Get points on the interior of the mask region
    # x TODO: Fit least-squares line through all points 

    

    femur_endpts, femur_midpts = estimate_femur_position(mask)
    
    views.draw_line(video, femur_endpts, femur_midpts)
    exit(420)


    views.view_frames(mask_top)
    views.view_frames(mask_btm)


    # views.draw_middle_lines(video, show_video=True)
    # views.view_frames(video)

if __name__ == "__main__":
    main()