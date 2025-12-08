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
from utils import io, views
from core import data_processing as dp


@dataclass
class RegionRange:
    """Defines a contiguous anatomical region by segment indices."""
    name: str
    start_idx: int  # 0-based inclusive start index in total_sums
    end_idx: int    # 0-based exclusive end index in total_sums


def load_video_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load centered DMM video and its radial masks.

    Returns
    -------
    video : np.ndarray
        Shape (n_frames, H, W), dtype uint8.
    masks : np.ndarray
        Shape (n_frames, H, W), dtype uint8, labels 1..64.
    """
    # TODO: Update paths to actual DMM data files
    video = io.load_nparray("data/segmented/normal_1207_video_N64.npy")
    radial_masks = io.load_nparray("data/segmented/normal_1207_radial_N64.npy")
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
        region_arrays[region.name] = total_sums[region.start_idx:region.end_idx, :]
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


def plot_intra_region_coms(region_coms: Dict[str, np.ndarray],
                           frame_start: int,
                           frame_end: int
                           ) -> None:
    """Create 3 stacked subplots (SB, OT, JC) of intra-region COM vs frame index.

    Parameters
    ----------
    region_coms : Dict[str, np.ndarray]
        {region_name: com_values}, each with shape (n_window_frames,).
    frame_start : int
        0-based inclusive start frame (for plot labeling).
    frame_end : int
        0-based inclusive end frame (for plot labeling).
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
    region_order = ["SB", "OT", "JC"]  # top to bottom

    t = np.arange(frame_start, frame_end + 1)

    for ax, name in zip(axes, region_order):
        com = region_coms[name]
        ax.plot(t, com)
        ax.set_ylabel(f"{name} COM")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name} Intra-region COM")

    axes[-1].set_xlabel("Frame index")
    fig.suptitle(f"Intra-region COM, frames {frame_start + 1}-{frame_end + 1}")
    plt.tight_layout()
    plt.show()


def main():
    """Main analysis pipeline."""
    # Load video and masks
    video, masks = load_video_data()

    # Compute intensity data
    total_sums, total_nonzero, segment_labels = load_intensity_data(video, masks)

    # Define anatomical regions (0-based indices in total_sums)
    # JC: segments 1-29 -> indices 0-28 (29 elements)
    # OT: segments 30-42 -> indices 29-41 (13 elements)
    # SB: segments 43-64 -> indices 42-63 (22 elements)
    region_ranges = [
        RegionRange("JC", 0, 29),
        RegionRange("OT", 29, 42),
        RegionRange("SB", 42, 64),
    ]

    # Split into three anatomical parts
    region_arrays = split_three_parts_indexwise(total_sums, region_ranges)

    # Compute intra-region COM over full time series
    region_coms_full = {
        region.name: compute_centre_of_mass_region(region_arrays[region.name])
        for region in region_ranges
    }

    # Extract frame window: 242-352 (1-based) -> 241-352 (0-based inclusive)
    frame_start = 241
    frame_end = 352

    region_coms_window = {
        name: extract_frame_window(com, frame_start, frame_end)
        for name, com in region_coms_full.items()
    }

    # Plot the results
    plot_intra_region_coms(region_coms_window, frame_start, frame_end)


if __name__ == "__main__":
    main()
