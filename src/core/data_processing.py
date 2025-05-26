import os
import numpy as np
import pandas as pd
from typing import Dict, List
from src.config import VERBOSE


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
    Returns the frame-wise sum of pixel intensities, for every slice.
    Expects `regions` to be a NumPy array with shape (nslices, nframes, h, w).
    """
    if VERBOSE: print("measure_radial_intensities() called!")

    regions = regions.copy()
    nslcs, nfrms, h, w = regions.shape

    intensities = np.zeros(shape=(nslcs, nfrms))
    for slc in range(nslcs):
        # Sum over spatial dimensions (h, w) for each frame
        intensities[slc] = np.sum(regions[slc], axis=(1, 2))

    return intensities



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