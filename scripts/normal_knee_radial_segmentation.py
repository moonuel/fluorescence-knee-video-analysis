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
import aging_knee_radial_segmentation as rdl


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
        topl = rdl.get_closest_pt_to_edge(top_slc, "l")
        btml = rdl.get_closest_pt_to_edge(btm_slc, "l")
        # cv2.circle(frame, topl, 3, (255,255,255), -1) # Validate leftmost pts
        # cv2.circle(btm_slc, btml, 3, (255,255,255), -1)

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

    # Get left-most points on top/bottom halves
    topl_pts = rdl.get_closest_pts_to_edge(mask_top, "l")
    btml_pts_ = rdl.get_closest_pts_to_edge(mask_btm, "l")

    # views.draw_point(mask_top, topl_pts, True) # Validate left-most points
    # views.draw_point(mask_btm, btml_pts_, True)

    # Convert bottom-left coords to the whole mask
    btml_pts = [None] # to maintain 1-indexing. this gets skipped in the next block 
    for pt in btml_pts_[1:]:
        pt = list(pt) # for mutability
        pt[1] = pt[1] + int(h*spl) 
        btml_pts.append(tuple(pt)) # tuple for opencv compatibility

    # views.draw_line(mask, topl_pts, btml_pts) # Validate drawn line

    # Get midpoint of left line 
    midl_pts = [(0,0)]
    for cf in range(1, len(mask)):
        topl_pt = np.array(topl_pts[cf])
        btml_pt = np.array(btml_pts[cf])
        midl_pt = (topl_pt + btml_pt)//2
        midl_pts.append(tuple(midl_pt))

    # Smooth midpoints
    midl_pts[1:] = rdl.smooth_points(midl_pts[1:], 5)

    # views.draw_point(mask, midl_pts) # Validate midpoint

    frame_ctr = [(w//2,h//2)]*nframes
    # views.draw_line(mask, midl_pts, frame_ctr) # Validate basic femur estimation

    

    femur_endpts = frame_ctr
    femur_midpts = midl_pts

    # return femur_endpts, femur_midpts

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
    views.show_frames(video)

    # Get adaptive mean mask
    mask_src = utils.log_transform_video(video)
    mask_src = utils.blur_video(video, (61,61), 0)
    mask = utils.mask_adaptive(mask_src, 101, -2)
    # mask = utils.morph_open(mask, (31,31)) # clean small artifacts
    views.show_frames(mask)

    # Estimate femur position
    init_guess = ((450//2, 500//2), (20, 500//2) )
    femur_endpts, femur_midpts = estimate_femur_position(mask, init_guess)


    return


if __name__ == "__main__":
    main()
