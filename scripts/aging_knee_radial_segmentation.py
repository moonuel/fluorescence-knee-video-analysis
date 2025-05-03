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

def main():
    if VERBOSE: print("main() called!")

    # Load pre-processed video
    # video, _ = knee.pre_process_video(video) 
    video = io.load_nparray("../data/processed/aging_knee_processed.npy") # result of above function call

    # Pre-process video
    video = np.rot90(video, k=-1, axes=(1,2))
    video = utils.crop_video_square(video, int(350*np.sqrt(2))) 

    # Slight rotation
    angle = -26
    video = utils.rotate_video(video, angle)
    video = utils.crop_video_square(video, 350) # crop out black borders

    # Get adaptive mean mask
    video_blr = utils.blur_video(video, (31,31), 0)
    mask = utils.mask_adaptive(video_blr, 71, -2)
    mask = utils.morph_open(mask, (15,15)) # clean small artifacts

    # > TODO: Get the leftmost points
    # x TODO: Get the basic femur estimation
    # x TODO: Brainstorm femur endpoint estimation improvements
    # x TODO: Get points on the interior of the mask region
    # x TODO: Fit least-squares line through all points 

    # Split frame along the middle
    h,w = mask.shape[1:]
    mask_top = mask[:,0:h//2,:]
    mask_btm = mask[:,h//2:,:]

    # Get left-most points
    tl_pts = get_closest_pts_to_edge(mask_top, "l")
    bl_pts = get_closest_pts_to_edge(mask_btm, "l")

    views.draw_point(mask_top, tl_pts, True)
    views.draw_point(mask_btm, bl_pts, True)

    views.view_frames(mask_top)
    views.view_frames(mask_btm)


    # views.draw_middle_lines(mask, show_video=True)
    # views.draw_middle_lines(video, show_video=True)
    # views.view_frames(video)

if __name__ == "__main__":
    main()