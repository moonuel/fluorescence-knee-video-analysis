import os
import numpy as np
import pandas as pd
from typing import Dict, List
from config import VERBOSE


def _measure_region_intensity(region: np.ndarray) -> np.ndarray:
    """
        Helper function. Returns sum of pixel intensities for every frame
    """
    # if VERBOSE: print("_measure_region_intensity() called!")

    intensities = []
    for cf, frame in enumerate(region):
        intsty = np.sum(frame)
        intensities.append(intsty)

    intensities = np.array(intensities)

    return intensities

# TODO: refine return type hint. TypedDict? 
def measure_region_intensities(regions: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], keys: List[str], normalized=False) -> Dict: 
    """
    Inputs: 
        regions (Dict[str, np.ndarray]) - segmented regions for which brightness is to be measured
        masks (Dict[str, np.ndarray]) - corresponding segmented masks used for normalization
        keys (List[str]) - specifies which regions are to be measured
        normalized (bool) - flag to normalize data
    Outputs:
        region_intensities (Dict[str, np.ndarray, bool]) - Dict of region intensity readings (np.ndarray); and also a normalized (bool) flag
    """
    if VERBOSE: print("measure_region_intensities() called!")

    if normalized:
        if VERBOSE: print(" > normalized!")
        mask_intensities = {}
        for k in keys:
            mask_intensities[k] = _measure_region_intensity(masks[k])

    region_intensities = {}
    for k in keys:
        region_intensities[k] = _measure_region_intensity(regions[k])
        
        if normalized:
            region_intensities[k] = region_intensities[k] / mask_intensities[k]
    
    region_intensities["normalized"] = normalized # Store some metadata

    return region_intensities

def _measure_region_mean_intensity(region: np.ndarray) -> np.ndarray:
    """
        Helper function. Returns mean of pixel intensities for every frame.
    """
    # if VERBOSE: print("_measure_region_intensity() called!")

    intensities = []
    for cf, frame in enumerate(region):
        nonzero = frame[frame!=0]
        intsty = np.mean(nonzero) # np.mean instead of np.sum
        intensities.append(intsty)

    intensities = np.array(intensities)

    return intensities

# TODO: refine return type hint. TypedDict? 
def measure_region_mean_intensities(regions: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], keys: List[str]) -> Dict: 
    """
    For testing/comparison with Juan's plots.
    Inputs: 
        regions (Dict[str, np.ndarray]) - segmented regions for which brightness is to be measured
        masks (Dict[str, np.ndarray]) - corresponding segmented masks used for normalization
        keys (List[str]) - specifies which regions are to be measured
        normalized (bool) - flag to normalize data
    Outputs:
        region_intensities (Dict[str, np.ndarray, bool]) - Dict of region intensity readings (np.ndarray); and also a normalized (bool) flag
    """
    if VERBOSE: print("measure_region_mean_intensities() called!")

    region_intensities = {}
    for k in keys:
        region_intensities[k] = _measure_region_mean_intensity(regions[k])
        
    region_intensities["normalized"] = False # Store some metadata TODO This needs to be retired...

    return 

def measure_radial_intensities(regions: np.ndarray) -> np.ndarray:
    """
    Calculates frame-size pixel intensity sum for arbitrarily many radial slices.
    Expects `regions` to have shape (nslices, nframes, h, w).
    Returns array of shape (nslices, nframes) with frame-wise sums.
    """
    if VERBOSE:
        print("measure_radial_intensities() called!")

    regions = np.asarray(regions)
    return regions.sum(axis=(2, 3))




def get_intensity_diffs(intensities: Dict) -> Dict[str, np.ndarray]:
    """
    Returns the difference in intensity values between every frame (backward difference). 
    Inputs:
        intensities (Dict[str, np.ndarray, bool]) - region intensities for which derivatives are to be obtained
    Outputs:
        derivs (Dict[np.ndarray]) - derivatives obtained
    """
    if VERBOSE: print("get_intensity_diffs() called!")

    keys = ['l','m','r']
    derivs = {}
    for k in keys:
        deriv = pd.Series(intensities[k].astype(np.int64)).diff() # Cast from uint32 
        derivs[k] = deriv
    return derivs

def get_intensity_derivs(intensities:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    """
    Implements a second-order accuracy finite difference approximation for the first derivative of the intensity data
    Inputs:
        intensities (Dict[str, np.ndarray, bool]) - region intensities for which derivatives are to be obtained
    Outputs:
        derivs (Dict[np.ndarray]) - derivatives obtained
    """
    if VERBOSE: print("get_intensity_derivs() called!")

    keys = ['l','m','r']
    derivs = {}
    for k in keys:
        derivs[k] = np.gradient(intensities[k], edge_order=2)
    return derivs


def compute_sums_nonzeros(masks, video):
    """
    Calculates the total pixel intensity sums and number of non-zero pixels 
        in each segment of a knee video and radially segmented mask.
        
    Inputs: 
        masks (np.ndarray): radially segmented mask with shape (nfs, h, w)
        video (np.ndarray): associated video with shape (nfs, h, w)
            
    Outputs: 
        total_sums (np.ndarray): total pixel intensity sum with shape (Nsegs, nfs)

    """

    assert masks.shape == video.shape # Sanity check

    nfs, h, w = masks.shape
    lbls = np.unique(masks[masks > 0])
    N = len(lbls)

    # Calculate total pixel intensities within each segment of the video
    total_sums = np.zeros(shape=(N, nfs), dtype=int)
    for n, lbl in enumerate(lbls):
        for f in range(nfs):
            frame = video[f]
            mask_f = masks[f]
            total_sums[n, f] = frame[mask_f == lbl].sum()

    # Calculate number of non-zero pixels within each segment of the video (for normalization purposes)
    total_nonzero = np.zeros((N, nfs), dtype=int)
    for n, lbl in enumerate(lbls):
        for f in range(nfs):
            frame = video[f]
            mask_f = masks[f]
            total_nonzero[n, f] = np.count_nonzero(frame[mask_f == lbl])

    assert total_sums.shape == total_nonzero.shape # Sanity check

    print(f"{total_sums[:, 0]=}")
    print(f"{total_nonzero[:, 0]=}")

    return total_sums, total_nonzero