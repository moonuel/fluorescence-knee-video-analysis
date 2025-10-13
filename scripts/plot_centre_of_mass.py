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


def plot_com_cycles(centre_of_mass:np.ndarray, cycle_fs:List[list], contiguous:bool = False, video_id:str = None) -> None:
    """Accepts a centre_of_mass data array and plots the passed cycles. 
    "cycles" should have structure [[flx1, flx2], [ext1, ext2], 
                                    [flx3, flx4], [ext3, ext4], ...]
        i.e. "cycles" is a list of frame range pairs 
    
    Inputs:
        centre_of_mass (np.ndarray): array of length (nfs) giving the position between 1-N of the centre of mass, for each frame
        cycles (List[list]): list containing frame ranges (0-indexed) of flexion and extension cycles to be plotted. 
        contiguous (bool): Set to True if flexion/extension cycles should be contiguous. False by default
        video_id (str): For example "308 normal". Just for plot formatting

    Example usage:

        # Plot individual cycles 
        cycles = "71-116 117-155 " \\
                "253-298 299-335 " \\
                "585-618 630-669 " \\
                "156-199 210-250"
        cycles = parse_cycles(cycles)

        plot_cycles(centre_of_mass, cycles, contiguous=False, video_id="308 normal")

    """

    if not isinstance(cycle_fs, list): raise TypeError(f"passed cycles is not a list. Given: {type(cycle_fs)}")
    assert len(cycle_fs) % 2 == 0 # sanity check that we have a complete set of pairs 

    cycle_fs = deepcopy(cycle_fs) # protect internal list modification

    plt.figure(figsize=(19, 7))
    cmap = plt.get_cmap('cool', len(cycle_fs)//2)

    if contiguous:
        # We want contiguous frame ranges for flexion/extension frame range pairs
        for i in np.arange(0, len(cycle_fs), 2):

            flx = cycle_fs[i] # mutable
            ext = cycle_fs[i+1]

            mp = (flx[1] + ext[0]) // 2
            # print(f"{flx[1]=}, {ext[0]=}, {mp=}") # sanity check

            flx[1] = mp 
            ext[0] = mp 

    # Cast cycles to Series for easier indexing 
    cycle_coms = [] 
    for i in np.arange(0, len(cycle_fs), 2): 

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

    # Get average position
    avg_com = cycle_coms.mean(axis=1, skipna=False)

    # Plot all cycles, centered around the midpoint 
    for i in range(cycle_coms.shape[1]):
        com = cycle_coms.iloc[:, i]
        plt.plot(com.sort_index(), label=f"Cycle {i + 1}", color=cmap(i))
    plt.plot(avg_com.sort_index(), color='gray', linestyle='--', label="Average of cycles")

    breakpoint()

    # Final formatting
    if video_id is not None: video_id = f" ({video_id})"
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

    # Calculate center of mass
    centre_of_mass = compute_centre_of_mass(total_sums)

    plt.plot(centre_of_mass)
    plt.title("Centre of mass over non-zero frames"); plt.xlabel("Frame number"); plt.ylabel("Segment number (JC to SB)")
    plt.show()

    # Plot individual cycles 
    cycles = parse_cycles(cycles)
    print(f"{cycles=}")

    plot_com_cycles(centre_of_mass, cycles, video_id = num_type) # plotting function with optional contiguous frame ranges

    return


def load_1339_N16() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/processed/1339_aging_radial_masks_N16.npy")
    video = io.load_video("../data/processed/1339_aging_radial_video_N16.npy")
    cycles =   "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553	554-576	579-609" # 1339 aging

    return masks, video, cycles


def load_1339_N64() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/processed/1339_aging_radial_masks_N64.npy")
    video = io.load_video("../data/processed/1339_aging_radial_video_N64.npy")
    cycles =   "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553	554-576	579-609" # 1339 aging
    
    return masks, video, cycles


def load_308_N16() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/processed/normal_knee_radial_masks_N16.npy")
    video = io.load_video("../data/processed/normal_knee_radial_video_N16.npy")
    cycles = "71-116 117-155 253-298 299-335 585-618 630-669 156-199 210-250" # 308 normal

    masks, video = np.flip(masks, axis=2), np.flip(video, axis=2) # Flip along horizontal dim
    masks[masks > 0] = (masks[masks > 0] - 2) % 16 + 1 # Shift segment labels by one for 308 N16 video

    return masks, video, cycles


def load_308_N64() -> Tuple[np.ndarray, np.ndarray]:

    masks = io.load_masks("../data/processed/308_normal_radial_masks_N64.npy")
    video = io.load_video("../data/processed/308_normal_radial_video_N64.npy")
    cycles = "71-116 117-155 253-298 299-335 585-618 630-669 156-199 210-250" # 308 normal

    return masks, video, cycles


if __name__ == "__main__":
    
    masks, video, cycles = load_308_N16(); main(masks, video, cycles, "308 Normal (16 segs)")
    masks, video, cycles = load_308_N64(); main(masks, video, cycles, "308 Normal (64 segs)")
    masks, video, cycles = load_1339_N16(); main(masks, video, cycles, "1339 Aging (16 segs)")
    masks, video, cycles = load_1339_N64(); main(masks, video, cycles, "1339 Aging (64 segs)")