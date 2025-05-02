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

def smooth_coords(coords:pd.DataFrame, window_size:int) -> pd.DataFrame:
    """Implements a moving average filter over the coordinate data."""
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

def get_intensity_derivs(intensities:Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    """
    Implements a second-order finite difference approximation for the first derivative of the intensity data
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