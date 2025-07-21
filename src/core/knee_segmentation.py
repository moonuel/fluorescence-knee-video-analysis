"""
Functions for segmenting frames/videos should go in here, as well as functions related to combining or manipulating those segmented frames/video
"""



import os
import sys
import numpy as np
import pandas as pd
import cv2
from utils import utils
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

VERBOSE = True
DEBUG = True

def centre_video(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
        video (np.ndarray) - video to be pre-processed. Dimensions (nframes, height, width)

    Outputs:
        video (np.ndarray) - processed video. Dimensions (nframes, height, width)
        translation_mxs (np.ndarray) - translation matrices used to centre each frame
    """
    if VERBOSE: print("centre_video() called!")

    video_ctrd = []
    translation_mxs = []
    for idx, frame in enumerate(video):

        # Process frame
        frame, tr_mx = utils.centroid_stabilization(frame)

        # Store data
        video_ctrd.append(frame)
        translation_mxs.append(tr_mx)

    video_ctrd = np.array(video_ctrd)
    translation_mxs = np.array(translation_mxs)
    return video_ctrd, translation_mxs

def translate_coords(translation_mxs: np.ndarray, coords: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs: 
        translation_mxs (np.ndarray) - 2x3 translation matrices
        coords (pd.DataFrame) - df of coordinates to be transformed. Expected format: coords.index := frame number; four points per frame
    Outputs: 
        coords_ctrd (pd.DataFrame) - df of transformed coordinates
    """
    if VERBOSE: print("translate_coords() called!")

    coords_ctrd = pd.DataFrame(np.nan, index=coords.index, columns=coords.columns) # empty dataframe
    uqf = coords.index.unique()
    for cf in uqf:
        
        # Apply translations to coords
        tr_mx = translation_mxs[cf]
        xp = np.row_stack([coords.loc[cf].to_numpy().T, np.ones(4)])
        coord_ctrd = tr_mx @ xp

        # Store result
        coords_ctrd.loc[cf] = coord_ctrd.T

    return coords_ctrd

def smooth_coords(coords:pd.DataFrame, window_size:int) -> pd.DataFrame:
    """Implements a moving average filter over the three-part segmentation coordinate data."""
    if VERBOSE: print("smooth_coords() called!")

    assert coords.shape[0]%4 == 0
    nrows=coords.shape[0]//4

    p1 = coords.iloc[0::4, :].copy()
    p2 = coords.iloc[1::4, :].copy()
    p3 = coords.iloc[2::4, :].copy()
    p4 = coords.iloc[3::4, :].copy()

    ps = [p1, p2, p3, p4]
    for p in ps:
        p["X"] = p["X"].rolling(window_size, min_periods=1, center=True).mean()
        p["Y"] = p["Y"].rolling(window_size, min_periods=1, center=True).mean()
    
    coords_smtd = []
    for r in range(nrows):
        for p in ps:
            coords_smtd.append(p.iloc[r])
    coords_smtd = pd.DataFrame(coords_smtd)

    return coords_smtd

def get_three_segments(video: np.ndarray, coords: np.ndarray, thresh_scale:int=0.8) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Manually segments a knee video using a set of coordinates and Otsu thresholded masks. 
    Inputs: 
        video (np.ndarray) - video to be manually segmented
        coords (np.ndarray) - coordinates to use for three part segmentation
        thresh_scale(int) - to rescale the Otsu threshold mask 
    Outputs:
        regions (Dict[str, np.ndarray]) - segmented regions obtained 
        masks (Dict[str, np.ndarray]) - segmented masks obtained
    """
    if VERBOSE: print("get_three_segments() called!")

    video = video.copy()

    otsu_masks = []
    for cf, frame in enumerate(video):
        
        # Get otsu mask
        thresh_val, _ = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = int(thresh_val*thresh_scale)
        _, otsu_mask = cv2.threshold(frame, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Store otsu mask
        otsu_masks.append(otsu_mask)
        
    otsu_masks = np.array(otsu_masks)

    # otsu_masks = get_otsu_masks(video) # TODO: Validate that this replaces the above

    l_masks = []
    m_masks = []
    r_masks = []
    l_region = []
    m_region = []
    r_region = []
    otsu_region = []
    for cf in coords.index.unique():

        frame = video[cf]        
        cf_coords = coords.loc[cf].to_numpy().astype(int)

        # Get rough bisection mask
        mp0 = (cf_coords[0]+cf_coords[2])//2 # top 
        mp1 = (cf_coords[1]+cf_coords[3])//2 # top 
        lr_mask = utils.pixels_left_of_line(frame, mp1, mp0)

        # Get rough middle mask
        _m_mask_l = utils.pixels_left_of_line(frame, cf_coords[0], cf_coords[1])
        _m_mask_r = utils.pixels_left_of_line(frame, cf_coords[3], cf_coords[2])
        m_mask = _m_mask_l & _m_mask_r

        # Get rough left and right masks
        l_mask = lr_mask & ~m_mask
        r_mask = ~lr_mask & ~m_mask

        # Get final masks
        otsu_mask = otsu_masks[cf]
        l_mask = l_mask & otsu_mask
        m_mask = m_mask & otsu_mask
        r_mask = r_mask & otsu_mask

        # Get l/m/r/Otsu regions
        l_reg = l_mask & frame
        m_reg = m_mask & frame
        r_reg = r_mask & frame
        otsu_reg = otsu_mask & frame

        # Store vals
        l_masks.append(l_mask)
        m_masks.append(m_mask)
        r_masks.append(r_mask)

        l_region.append(l_reg)
        m_region.append(m_reg)
        r_region.append(r_reg)
        otsu_region.append(otsu_reg)

    # Cast to numpy arrays
    l_masks = np.array(l_masks)
    m_masks = np.array(m_masks)
    r_masks = np.array(r_masks)
    l_region = np.array(l_region)
    m_region = np.array(m_region)
    r_region = np.array(r_region)
    otsu_region = np.array(otsu_region)

    # Store in dict
    masks = {"l": l_masks, "m": m_masks, "r": r_masks}
    regions = {"l": l_region, "m": m_region, "r": r_region}
    
    return regions, masks

def get_otsu_masks(video:np.ndarray, thresh_scale:int=0.8) -> np.ndarray:
    """Gets the Otsu masks for the video. Optionally rescale the threshold value"""
    if VERBOSE: print("get_otsu_masks() called!")

    video = video.copy()
    otsu_masks = []
    for cf, frame in enumerate(video):
        
        # Get otsu mask
        thresh_val, _ = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = int(thresh_val*thresh_scale)
        _, otsu_mask = cv2.threshold(frame, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Store otsu mask
        otsu_masks.append(otsu_mask)
        
    otsu_masks = np.array(otsu_masks)

    return otsu_masks

def get_bisecting_mask(frame:np.ndarray, p1:Tuple[int,int], p2:Tuple[int,int]) -> np.ndarray:
    """Gets a binary mask bisecting the plane by a line (p2 - p1)."""
    # if VERBOSE: print("get_bisecting_mask() called!")

    h,w = frame.shape
    
    # Create a coordinate grid
    yi, xi = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Compute the signed area (cross product)
    cross = (xi - p1[0]) * (p2[1] - p1[1]) - (yi - p1[1]) * (p2[0] - p1[0])
    
    # Classify left side to positive, right side to negative
    bsct_mask = ((cross > 0) * 255).astype(np.uint8) 
    
    return bsct_mask

def get_bisecting_masks(video:np.ndarray, p1s:np.ndarray, p2s:np.ndarray) -> np.ndarray:
    """Gets a binary mask that bisects every frame in a video by the set of lines {(p2 - p1)}, for p2 in p2s and p1 in p1s"""
    if VERBOSE: print("get_bisecting_masks() called!")

    if not video.shape[0] == p1s.shape[0] == p2s.shape[0]:
        raise ValueError(f"get_bisecting_masks(): frame count mismatch. got {video.shape[0]}, {p1s.shape[0]}, {p2s.shape[0]}")

    video = video.copy()
    p1s = p1s.copy()
    p2s = p2s.copy()

    bsct_masks = []
    for cf, frame in enumerate(video):
        bsct_mask = get_bisecting_mask(frame, p1s[cf], p2s[cf])
        bsct_masks.append(bsct_mask)
    bsct_masks = np.array(bsct_masks)

    return bsct_masks

def mask_adaptive(video:np.ndarray, block_size:int, adj_value:int) -> np.ndarray:
    "Implements an adaptive thresholding mask over a grayscale video with dimensions (nframes,hgt,wth)"
    if VERBOSE: print("mask_adaptive() called!")

    masks = []
    for _, frame in enumerate(video):
        mask = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,block_size, adj_value)
        masks.append(mask)
    masks = np.array(masks)
    
    return masks