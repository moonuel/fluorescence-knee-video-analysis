"""
Script for calculating the center of mass of grayscale intensity video across radial segments. 
"""

from utils import io, utils, views
import core.data_processing as dp
import numpy as np
import matplotlib.pyplot as plt
from typing import List
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
    """Converts the frame number ranges from the Excel file into usable frame ranges for downstream code.
        Shifts all indices from 1-index to 0-index, and then adds 1 to all range endpoints to include it in the interval. 
    
    Example:
        parse_cycles("71-116 117-155 
                    "253-298 299-335 " \
                    "585-618 630-669 " \
                    "156-199 210-250")

        Returns [(70, 117), (116, 156), 
                    (252, 299), (298, 336), ... etc]

        """

    if not isinstance(cycles, str): raise TypeError(f"Argument is not a string. Given: {type(cycles)}")
    assert len(cycles) % 2 == 0 # check that we have flexion/extension pairs 

    cycles = cycles.split(" ")

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


def plot_cycles(centre_of_mass:np.ndarray, cycles:List[list]) -> None:
    """Accepts a centre_of_mass data array and plots the passed cycles. 
    "cycles" should have structure [[flx1, flx2], [ext1, ext2], 
                                    [flx3, flx4], [ext3, ext4], ...]
        i.e. "cycles" is a list of frame range pairs 
    
    Inputs:
        centre_of_mass (np.ndarray): array of length (nfs) giving the position between 1-N of the centre of mass, for each frame
        cycles (List[list]): list containing frame ranges (0-indexed) of flexion and extension cycles to be plotted. 
    """

    if not isinstance(cycles, list): raise TypeError(f"passed cycles is not a list. Given: {type(cycles)}")
    assert len(cycles) % 2 == 0 # sanity check that we have a complete set of pairs 

    cycles = deepcopy(cycles) # protect internal list modification

    plt.figure(figsize=(19, 7))

    # We want contiguous frame ranges for flexion/extension frame range pairs
    for i in np.arange(0, len(cycles), 2):

        flx = [cycles[i][0], cycles[i][1]] # unpack the list of lists
        ext = [cycles[i+1][0], cycles[i+1][1]]

        mp = (flx[1] + ext[0]) // 2
        # print(f"{flx[1]=}, {ext[0]=}, {mp=}") # sanity check

        # Update values directly
        cycles[i][1] = mp # flx[1]
        cycles[i+1][0] = mp # ext[0]

    # Plot all cycles, centered around the midpoint 
    for i in np.arange(0, len(cycles), 2):
        print(f"{i=}:", cycles[i], cycles[i+1])

        flx = cycles[i][0], cycles[i][1] # unpack the list of lists
        ext = cycles[i+1][0], cycles[i+1][1]

        plt.plot(np.arange(flx[0] - flx[1] + 1, 1), # shifted to the left, ending at 0
                 centre_of_mass[flx[0]:flx[1]], color='r') 

        plt.plot(centre_of_mass[ext[0]:ext[1]], color='b') # plotted normally, from 0

        # pdb.set_trace() # python debugger! pretty cool

    pdb.set_trace()
    plt.axvline(0, linestyle="--", color='k')
    plt.show()


def main():
    
    # Import data and video
    masks = io.load_masks("../data/processed/normal_knee_radial_masks_N16.npy")
    video = io.load_video("../data/processed/normal_knee_radial_video_N16.npy")

    masks, video = np.flip(masks, axis=2), np.flip(video, axis=2) # Flip along horizontal dim
    
    if masks.shape != video.shape: raise ValueError(f"{masks.shape=} != {video.shape=}. Is the data correct?")

    nfs, h, w = masks.shape
    lbls = np.unique(masks[masks > 0])
    N = len(lbls)

    print(f"{nfs = }, {h = }, {w = } \n{lbls = } \n{N = }") 
    # views.show_frames([video, masks * (255 // masks.max())], "Validate data") # Sanity check

    # Shift segment labels by one
    shift = 1
    masks[masks > 0] = (masks[masks > 0] - 1 - shift) % N + 1
    
    # views.show_frames([video, masks * (255 // masks.max())], "Validate segment shift") # Sanity check

    # Calculate sums
    total_sums, total_counts = dp.compute_sums_nonzeros(masks, video)
    
    print(f"{total_sums.shape=}, {total_counts.shape=}")

    # Calculate center of mass
    centre_of_mass = compute_centre_of_mass(total_sums)

    plt.plot(centre_of_mass)
    plt.show()

    # Plot individual cycles 
    cycles = "71-116 117-155 " \
            "253-298 299-335 " \
            "585-618 630-669 " \
            "156-199 210-250"
    
    cycles = parse_cycles(cycles)

    print(f"{cycles=}")

    plot_cycles(centre_of_mass, cycles)

    return


if __name__ == "__main__":
    main()