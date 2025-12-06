"""
Script for calculating the center of mass of grayscale intensity video across radial segments. 
"""

from utils import io, utils, views
import core.data_processing as dp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from copy import deepcopy
import pdb
import scipy as sp


def compute_centre_of_mass(total_sums: np.ndarray) -> np.ndarray:
    """
    Calculates the centre of mass of the pixel intensity data over the radial trajectory. 
    Assumes segments {1, ..., N} are labeled in ascending order from the joint cavity (JC) 
    up to the suprapatellar bursa (SB).

    Inputs: 
        total_sums (np.ndarray): pixel intensity sums with shape (N, nfs). 

    Outputs:
        centre_of_mass (np.ndarray): position array of length (nfs) giving the center of mass 
                                     for every frame (between 1 and N). Frames with zero total 
                                     intensity return np.nan.
    """
    N, nfs = total_sums.shape
    positions = np.arange(1, N + 1).reshape(-1, 1)  # shape (N, 1), 1-based indices

    weighted_sums = (positions * total_sums).sum(axis=0)
    totals = total_sums.sum(axis=0)

    # Avoid division by zero: return np.nan where totals == 0
    centre_of_mass = np.divide(
        weighted_sums,
        totals,
        out=np.full_like(weighted_sums, np.nan, dtype=float),
        where=totals != 0
    )

    return centre_of_mass


def parse_cycles(cycles:str) -> List[tuple]:
    """Converts the 1-indexed frame number ranges from the Excel file into **0-indexed, endpoint-inclusive** ranges for downstream code.
    
    Example:
        parse_cycles("71-116 117-155 " \
                    "253-298 299-335 " \
                    "585-618 630-669 " \
                    "156-199 210-250")

        Returns [(70, 116), (116, 156), 
                    (252, 298), (298, 336), ... etc]

        """

    if not isinstance(cycles, str): raise TypeError(f"Argument is not a string. Given: {type(cycles)}")
    cycles = cycles.split()

    if not len(cycles) % 2 == 0: raise ValueError(f"Argument needs to have even length. Given: {len(cycles)=}") # check that we have flexion/extension pairs 

    # Parse into list of frame ranges 
    for i, rng in enumerate(cycles):
        cycles[i] = rng.split("-")
        cycles[i] = list(map(int, cycles[i]))

    # Convert from 1-index to 0-index
    cycles = [[item - 1 for item in sublist] # 2. And within each sublist, subtract 1 from each item 
              for sublist in cycles] # 1. Parse each sublist 
    
    # But we want to include the last frame
    for rng in cycles:
        rng[1] += 1

    return cycles


def plot_cycle_coms(cycle_coms:pd.DataFrame, video_id:str = None) -> None:
    """Accepts a centre_of_mass data array and plots the passed cycles. 
    "cycles" should have structure [[flx1, flx2], [ext1, ext2], 
                                    [flx3, flx4], [ext3, ext4], ...]
        i.e. "cycles" is a list of frame range pairs 
    
    Inputs:
        cycle_coms : (pd.DataFrame)
            DataFrame with shape (nframes_rel, ncycles) containing COMs per cycle, aligned and centered with each other.
        video_id : (str) 
            For example "308 normal". Just for plot formatting

    Example usage:

        # Plot individual cycles 

        plot_cycles(centre_of_mass, cycles, contiguous=False, video_id="308 normal")

    """

    nfrs, ncycs = cycle_coms.shape

    plt.figure(figsize=(19, 7))
    cmap = plt.get_cmap('cool', ncycs)

    # Get average position
    avg_com = cycle_coms.mean(axis=1, skipna=False)

    # Plot all cycles, centered around the midpoint 
    for i in range(ncycs):
        com = cycle_coms.iloc[:, i]
        plt.plot(com.sort_index(), label=f"Cycle {i + 1}", color=cmap(i))
    plt.plot(avg_com.sort_index(), color='gray', linestyle='--', label="Average of cycles")

    # Final formatting
    if video_id is not None: video_id = f" [{video_id}]"
    else: video_id = ""
    plt.title("Average position of fluorescence intensity" + video_id)
    plt.xlabel("Frames from midpoint (Left: flexion; Right: extension)")
    plt.ylabel("Segment number (JC to SB)")

    plt.axvline(0, linestyle="--", color='k')
    
    plt.legend()
    plt.show()

    print(cycle_coms)
    print(cycle_coms.shape)
    print(cycle_coms.info())


def pad_empty_frames(array:np.ndarray, padding:Tuple[int,int]) -> np.ndarray:
    """
    Pads empty frames to the start and/or end of the passed numpy array.

    Inputs: 
        array (np.ndarray): the array with shape (nfs, h, w)
        padding (Tuple[int,int]): for input (x,y), pads x frames to the front and y frames to the back

    Outputs:
        array_padded (np.ndarray): the padded array

    """

    array = np.asarray(array)
    padding = tuple(padding)

    if not len(padding) == 2: raise ValueError(f"padding must be 2-tuple. Given: {padding=}")

    return np.pad(array, pad_width=(padding, (0,0), (0,0)), mode="constant")


def compute_cycle_coms(centre_of_mass:np.ndarray, cycle_fs:List[list]) -> pd.DataFrame:

    if not isinstance(cycle_fs, list): raise TypeError(f"passed cycles is not a list. Given: {type(cycle_fs)}")

    cycle_fs = deepcopy(cycle_fs) # protect internal list modification

    # Cast cycles to Series for easier indexing 
    cycle_coms = [] 
    for i in np.arange(0, len(cycle_fs), 2): 

        # Unpack ranges 
        flx = cycle_fs[i]
        ext = cycle_fs[i+1]

        flx_vals = centre_of_mass[flx[0]:flx[1]]
        ext_vals = centre_of_mass[ext[0]:ext[1]]

        end = min(ext[1], len(centre_of_mass)) # catch indexing errors for slices ending on last video frame

        flx_idx = np.arange(flx[0], flx[1]) - flx[1] # shift endpoint to origin
        ext_idx = np.arange(ext[0], end) - ext[0] # shift startpoint to origin

        flx_vals = pd.Series(flx_vals, index=flx_idx)
        ext_vals = pd.Series(ext_vals, index=ext_idx)

        cycle_coms.append(pd.concat([flx_vals, ext_vals], axis=0))

    cycle_coms = pd.concat(cycle_coms, axis=1) # shape (nfs, ncycs)

    return cycle_coms


def rescale_cycle_coms(cycle_coms:pd.DataFrame) -> pd.DataFrame:
    """
    Resamples the flexion and extension parts of a centered cycle_com DataFrame 
    to ensure equal duration. 
    
    Inputs:
    -------
        cycle_coms : pd.DataFrame
            A centered df with rows "Frame Numbers (Relative)" and columns "cycle numbers". 
            Contains NaN for cycles (columns) that are shorter than the others. 
            Flexion parts should end at frame index -1, and extension parts should start at frame index 0,
            so that all cycles are aligned. 

    Outputs:
    --------
        cycle_coms_rescaled : pd.DataFrame
            A centered df with same structure as the input df, but with rescaled columns to resolve all NaN.
        
    Example usage:
    --------------
        
        # Compute COM over video and prepare cycles 
        total_sums, total_counts = dp.compute_sums_nonzeros(masks, video)
        centre_of_mass = compute_centre_of_mass(total_sums)
        cycles =   "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553	554-576	579-609" # 1339 aging
        cycles = parse_cycles(cycles) # Validate and convert to List[list]

        ...

        # Select cycle COMS and rescale
        cycle_coms = compute_cycle_coms(centre_of_mass, cycles)
        cycle_coms_rescaled = rescale_cycle_coms(cycle_coms)          

    """

    cycle_coms.sort_index(inplace=True)

    idx_flx = cycle_coms.loc[:-1].index.to_numpy()
    idx_ext = cycle_coms.loc[0:].index.to_numpy()

    flx = cycle_coms.loc[:-1, :]
    ext = cycle_coms.loc[0:, :]

    nfs, ncols = cycle_coms.shape

    # Stretch flexion frames
    flx_stretch = []
    for col in range(ncols):

        # breakpoint()

        # No need to stretch the longest cycle
        if len(flx[col].dropna()) == len(flx[col]): 
            flx_stretch.append(flx[col])
            continue 

        # Remap domain to frames to [0,1]
        x_old = np.linspace(0, 1, len(flx[col].dropna()))
        y_old = flx[col].dropna()

        # Define interpolation func
        f = sp.interpolate.interp1d(x_old, y_old, kind="linear")

        # Resample new values in [0,1]
        x_new = np.linspace(0, 1, len(flx))
        y_new = pd.Series(f(x_new), idx_flx, name=col)

        # Map back to longest frame range 
        flx_stretch.append(pd.Series(y_new, idx_flx))

    flx_stretch = pd.DataFrame(flx_stretch).T

    # Stretch extension frames
    ext_stretch = []
    for col in range(ncols):

        # breakpoint()

        # No need to stretch the longest cycle
        if len(ext[col].dropna()) == len(ext[col]): 
            ext_stretch.append(ext[col])
            continue 

        # Normalize ext frames to [0,1]
        x_old = np.linspace(0, 1, len(ext[col].dropna()))
        y_old = ext[col].dropna()

        # Define interpolation func
        f = sp.interpolate.interp1d(x_old, y_old, kind="linear")

        # Resample new values in [0,1]
        x_new = np.linspace(0, 1, len(ext))
        y_new = pd.Series(f(x_new), idx_ext, name=col)

        # Map back to longest frame range 
        ext_stretch.append(pd.Series(y_new, idx_ext))

    ext_stretch = pd.DataFrame(ext_stretch).T

    # Stitch together stretched cycles
    cycle_coms_stretch = pd.concat([flx_stretch, ext_stretch], axis=0)

    return cycle_coms_stretch


def main(masks:np.ndarray, video:np.ndarray, cycles:str, num_type:str):
    
    # Validate data 
    if masks.shape != video.shape: raise ValueError(f"{masks.shape=} != {video.shape=}. Is the data correct?")

    nfs, h, w = masks.shape
    lbls = np.unique(masks[masks > 0])
    N = len(lbls)

    assert N == np.max(lbls)

    print(f"{nfs = }, {h = }, {w = } \n{lbls = } \n{N = }") 
    views.show_frames([video, masks * (255 // masks.max())], "Validate data") # Sanity check

    # Calculate sums
    total_sums, total_counts = dp.compute_sums_nonzeros(masks, video)
    print(f"{total_sums.shape=}, {total_counts.shape=}")

    # Calculate center of mass over entire video
    centre_of_mass = compute_centre_of_mass(total_sums)

    plt.plot(centre_of_mass)
    plt.title("Centre of mass over non-zero frames"); plt.xlabel("Frame number"); plt.ylabel("Segment number (JC to SB)")
    plt.show()

    # Get COMs per cycle
    cycle_coms = compute_cycle_coms(centre_of_mass, cycles)
    cycle_coms_rescaled = rescale_cycle_coms(cycle_coms) # Rescale COMs per cycle

    breakpoint()
   
    # Plot individual cycles 
    print(f"{cycles=}")
    print(cycle_coms.info())
    print(cycle_coms)
    plot_cycle_coms(cycle_coms, video_id = num_type) 
    plot_cycle_coms(cycle_coms_rescaled, video_id = num_type)

    return


def load_1339_N16() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/aging_1339_radial_masks_N16.npy")
    video = io.load_video("../data/segmented/aging_1339_radial_video_N16.npy")
    cycles =   "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553	554-576	579-609" # 1339 aging
    cycles = parse_cycles(cycles) # Validate and convert to List[list]

    return masks, video, cycles


def load_1339_N64_rem_C7() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/aging_1339_radial_masks_N64.npy")
    video = io.load_video("../data/segmented/aging_1339_radial_video_N64.npy")
    cycles =   "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553" # 1339 aging
    cycles = parse_cycles(cycles) # Validate and convert to List[list]

    return masks, video, cycles

def load_1339_N16_rem_C7() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/aging_1339_radial_masks_N16.npy")
    video = io.load_video("../data/segmented/aging_1339_radial_video_N16.npy")
    cycles =   "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553" # 1339 aging
    cycles = parse_cycles(cycles) # Validate and convert to List[list]

    return masks, video, cycles


def load_1339_N64() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/aging_1339_radial_masks_N64.npy")
    video = io.load_video("../data/segmented/aging_1339_radial_video_N64.npy")
    cycles =   "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553	554-576	579-609" # 1339 aging
    cycles = parse_cycles(cycles) # Validate and convert to List[list]

    return masks, video, cycles


def load_308_N16() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/normal_0308_radial_masks_N16.npy")
    video = io.load_video("../data/segmented/normal_0308_radial_video_N16.npy")
    cycles = "71-116 117-155 253-298 299-335 585-618 630-669 156-199 210-250" 
    cycles = parse_cycles(cycles) # Validate and convert to List[list]

    masks, video = np.flip(masks, axis=2), np.flip(video, axis=2) # Flip along horizontal dim
    masks[masks > 0] = (masks[masks > 0] - 2) % 16 + 1 # Shift segment labels by one for 308 N16 video

    return masks, video, cycles


def load_308_N64() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/segmented/normal_0308_radial_masks_N64.npy")
    video = io.load_video("../data/segmented/normal_0308_radial_video_N64.npy")
    cycles = "71-116 117-155 253-298 299-335 585-618 630-669 156-199 210-250" 
    cycles = parse_cycles(cycles) # Validate and convert to List[list]

    return masks, video, cycles


if __name__ == "__main__":
    
    masks, video, cycles = load_308_N16(); main(masks, video, cycles, "308 Normal (16 segs)")
    # masks, video, cycles = load_308_N64(); main(masks, video, cycles, "308 Normal (64 segs)")
    masks, video, cycles = load_1339_N16_rem_C7(); main(masks, video, cycles, "1339 Aging (16 segs)")
    # masks, video, cycles = load_1339_N64_rem_C7(); main(masks, video, cycles, "1339 Aging (64 segs)")