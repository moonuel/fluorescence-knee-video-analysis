"""
DMM knee video analysis: Compute intra-region center of mass (COM) within anatomical segments.

This script loads DMM video and radial mask data, computes total pixel intensities per radial segment,
partitions the data into three anatomical regions (JC, OT, SB), and plots the intra-region COM
over a specified frame range.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from utils import io, views, utils
from core import data_processing as dp
from config.knee_metadata import get_knee_meta
import sys

@dataclass
class RegionRange:
    """Defines a contiguous anatomical region by segment indices."""
    name: str
    start_idx: int  # 1-based inclusive start segment in total_sums
    end_idx: int    # 1-based inclusive end segment in total_sums


def load_video_data(condition, id, nsegs) -> Tuple[np.ndarray, np.ndarray]:
    """Load centered DMM video and its radial masks.

    Returns
    -------
    video : np.ndarray
        Shape (n_frames, H, W), dtype uint8.
    masks : np.ndarray
        Shape (n_frames, H, W), dtype uint8, labels 1..64.
    """
    video = io.load_nparray(f"data/segmented/{condition}_{id}_video_N{nsegs}.npy")
    radial_masks = io.load_nparray(f"data/segmented/{condition}_{id}_radial_N{nsegs}.npy")
    return video, radial_masks


def load_intensity_data(video: np.ndarray,
                        masks: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-segment intensities using core.data_processing.compute_sums_nonzeros.

    Parameters
    ----------
    video : np.ndarray
        Shape (n_frames, H, W), dtype uint8.
    masks : np.ndarray
        Shape (n_frames, H, W), dtype uint8, labels 1..64.

    Returns
    -------
    total_sums : np.ndarray
        Shape (64, n_frames), total intensity per segment per frame.
    total_nonzero : np.ndarray
        Shape (64, n_frames), number of active pixels per segment per frame.
    segment_labels : np.ndarray
        Shape (64,), sorted unique labels (1..64).
    """
    total_sums, total_nonzero = dp.compute_sums_nonzeros(masks, video)
    segment_labels = np.unique(masks[masks > 0]).astype(int)
    return total_sums, total_nonzero, segment_labels


def split_three_parts_indexwise(total_sums: np.ndarray,
                                region_ranges: List[RegionRange]
                                ) -> Dict[str, np.ndarray]:
    """Split total_sums into three anatomical regions using index-based ranges.

    Parameters
    ----------
    total_sums : np.ndarray
        Shape (64, n_frames).
    region_ranges : List[RegionRange]
        List of region definitions.

    Returns
    -------
    region_arrays : Dict[str, np.ndarray]
        {region_name: region_sums}, each with shape (N_region, n_frames).
    """
    region_arrays = {}
    for region in region_ranges:
        region_arrays[region.name] = total_sums[region.start_idx - 1:region.end_idx, :]
    return region_arrays


def compute_centre_of_mass_region(region_sums: np.ndarray) -> np.ndarray:
    """Compute intra-region COM per frame in local 1..N_region coordinates.

    Parameters
    ----------
    region_sums : np.ndarray
        Shape (N_region, n_frames).

    Returns
    -------
    com : np.ndarray
        Shape (n_frames,), COM values in range [1, N_region].
    """
    N_region, nfs = region_sums.shape
    positions = np.arange(1, N_region + 1).reshape(-1, 1)

    weighted = (positions * region_sums).sum(axis=0)
    totals = region_sums.sum(axis=0)

    com = np.divide(
        weighted,
        totals,
        out=np.full_like(weighted, np.nan, dtype=float),
        where=totals != 0,
    )
    return com


def extract_frame_window(com_series: np.ndarray,
                          frame_start: int,
                          frame_end: int
                          ) -> np.ndarray:
    """Extract COM values for the specified frame window.

    Parameters
    ----------
    com_series : np.ndarray
        Shape (n_frames,), COM per frame.
    frame_start : int
        0-based inclusive start frame.
    frame_end : int
        0-based inclusive end frame.

    Returns
    -------
    com_window : np.ndarray
        Shape (frame_end - frame_start + 1,).
    """
    return com_series[frame_start:frame_end + 1]


def plot_intra_region_coms(all_region_coms: List[Tuple[Dict[str, np.ndarray], str, 'FrameRange']]
                           ) -> None:
    """Create 3 stacked subplots (SB, OT, JC) of intra-region COM vs frame index for multiple cycles.

    Parameters
    ----------
    all_region_coms : List[Tuple[Dict[str, np.ndarray], str, FrameRange]]
        List of (region_coms, cycle_info, frame_range) for each cycle to plot.
    """
    fig, axes = plt.subplots(3, 1, sharex=False, figsize=(12, 10))
    region_order = ["SB", "OT", "JC"]  # top to bottom

    for ax, name in zip(axes, region_order):
        for region_coms, cycle_info, fr in all_region_coms:
            com = region_coms[name]
            t = np.arange(fr.start, fr.end + 1)
            ax.plot(t, com, label=cycle_info)
        ax.set_ylabel(f"{name} COM")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name} Intra-region COM")

    axes[-1].set_xlabel("Frame index")
    fig.suptitle(f"Intra-region COM for selected cycles")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(condition, id, nsegs, cycle_indices=None, phase="both"):
    """Main analysis pipeline."""
    if cycle_indices is None:
        cycle_indices = [0]

    # Get metadata
    meta = get_knee_meta(condition, int(id), int(nsegs))

    # Load video and masks
    video, masks = load_video_data(condition, id, nsegs)
    views.show_frames([video, (masks*(255//64))], "Validate data")

    # Normalize video intensity
    # video = utils.normalize_video_intensity(video)
    # views.show_frames(video, "Validate intensity normalization")

    # Compute intensity data
    total_sums, total_nonzero, segment_labels = load_intensity_data(video, masks)

    # Get anatomical regions from metadata
    region_ranges = [
        RegionRange(name, reg.start, reg.end)
        for name, reg in meta.regions.items()
    ]

    # Split into three anatomical parts
    region_arrays = split_three_parts_indexwise(total_sums, region_ranges)

    # Compute intra-region COM over full time series
    region_coms_full = {
        region.name: compute_centre_of_mass_region(region_arrays[region.name])
        for region in region_ranges
    }

    # Collect data for all selected cycles
    all_region_coms = []
    for cycle_idx in cycle_indices:
        cycle = meta.get_cycle(cycle_idx)
        if phase == "flexion":
            fr = cycle.flex
        elif phase == "extension":
            fr = cycle.ext
        else:  # "both"
            fr = cycle.full_cycle()

        region_coms_window = {
            name: extract_frame_window(com, fr.start, fr.end)
            for name, com in region_coms_full.items()
        }

        cycle_info = f"cycle {cycle_idx}, {phase}"
        all_region_coms.append((region_coms_window, cycle_info, fr))

    # Plot the results
    plot_intra_region_coms(all_region_coms)


if __name__ == "__main__":
    args = sys.argv[1:]
    condition = args[0]
    id = args[1]
    nsegs = args[2]
    cycle_indices_str = args[3] if len(args) > 3 else "0"
    cycle_indices = [int(x.strip()) for x in cycle_indices_str.split(',')]
    phase = args[4] if len(args) > 4 else "both"
    main(condition, id, nsegs, cycle_indices, phase)
