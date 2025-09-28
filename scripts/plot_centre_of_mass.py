"""
Script for calculating the center of mass of grayscale intensity video across radial segments. 
"""

from utils import io, utils, views
import core.data_processing as dp
import numpy as np
import matplotlib.pyplot as plt
from typing import List


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
    
    Example:
        parse_cycles("71-116 117-155 
                    "253-298 299-335 " \
                    "585-618 630-669 " \
                    "156-199 210-250")

        Returns [(71, 116), (117, 155), 
                    (253, 298), (299, 335), ... etc]

        """

    if not isinstance(cycles, str): raise TypeError(f"Argument is not a string. Given: {type(cycles)}")

    cycles = cycles.split(" ")

    # Parse into list of frame ranges 
    for i, rng in enumerate(cycles):
        cycles[i] = rng.split("-")
        cycles[i] = list(map(int, cycles[i]))

    # Convert from 1-index to 0-index
    cycles = [[item - 1 for item in sublist] # 2. And within each sublist, subtract 1 from each item 
              for sublist in cycles] # 1. Parse each sublist 

    return cycles


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

    print(cycles)

    # plot_cycles(centre_of_mass, cycles)

    return


if __name__ == "__main__":
    main()