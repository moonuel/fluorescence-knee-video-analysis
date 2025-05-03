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

def get_closest_point_to_edge(mask:np.ndarray, edge:int=0):
    """
    Finds the closest point in a binary mask to one edge of the frame.
    Inputs:
        mask (np.ndarray): A binary image mask.
        edge (int): The edge of the frame we want the point closest to.
            edge = 0   -> closest to bottom (Default)
            edge = 1   -> closest to left
            edge = 2   -> closest to top
            edge = 3   -> closest to right
    Outputs:
        (int, int): A tuple (x, y) returning the coordinates of the desired point.
    """

    y, x = np.nonzero(mask) # Validate binary mask
    if y.size == 0:
        return None  # No points found

    edge %= 4 # cast to mod 4 

    # Get all closest points to the edge
    edge_funcs = {
        0: np.argmax(y),  # Bottom (max y)
        1: np.argmin(x),  # Left (min x)
        2: np.argmin(y),  # Top (min y)
        3: np.argmax(x)   # Right (max x)
    }

    i = edge_funcs[edge]
    pt = x[i], y[i]

    return pt

def get_N_points_on_circle(ctr_pt: Tuple[int, int], ref_pt: Tuple[int, int], N: int) -> List[Tuple[int, int]]:
    """Returns a list of N equally spaced points on a circle, arranged clockwise.
    
    The circle is defined by:
    - Center point `ctr_pt`
    - Radius = distance between `ctr_pt` and `ref_pt`
    - First point is `ref_pt`, followed by the remaining points in clockwise order.

    Args:
        ctr_pt: (x, y) center of the circle.
        ref_pt: (x, y) reference point on the circle (first point in the output).
        N: Number of points to generate. If N=1, returns [ref_pt].

    Returns:
        List of (x, y) tuples representing the N points on the circle.
    """

    cx, cy = ctr_pt
    rx, ry = ref_pt
    radius = math.hypot(rx - cx, ry - cy)
    start_angle = math.atan2(ry - cy, rx - cx)  # Angle of ref_pt
    
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
    angle = -36
    video = utils.rotate_video(video, angle)
    video = utils.crop_video_square(video, 350) # crop out black

    # Get adaptive mean mask
    video_blr = utils.blur_video(video, (31,31), 0)
    mask = utils.mask_adaptive(video_blr, 71, -2)
    mask = utils.morph_open(mask, (15,15)) # clean small artifacts

    # TODO: Get the leftmost points
    # TODO: Get the basic femur estimation
    # TODO: Brainstorm femur endpoint estimation improvements
    # TODO: Get points on the interior of the mask region
    # TODO: Fit least-squares line through all points 

    views.draw_middle_lines(mask, show_video=True)
    views.draw_middle_lines(video, show_video=True)

    views.view_frames(video)

if __name__ == "__main__":
    main()