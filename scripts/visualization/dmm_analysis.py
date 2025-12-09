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
from config.knee_metadata import get_knee_meta, Cycle
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


def plot_intra_region_coms(all_region_coms: List[Tuple[Dict[str, np.ndarray], str, 'FrameRange', Cycle]]
                           ) -> None:
    """Create 3 stacked subplots (SB, OT, JC) of intra-region COM vs frame index for multiple cycles.

    Parameters
    ----------
    all_region_coms : List[Tuple[Dict[str, np.ndarray], str, FrameRange, Cycle]]
        List of (region_coms, cycle_info, frame_range, cycle) for each cycle to plot.
    """
    fig, axes = plt.subplots(3, 1, sharex=False, figsize=(12, 10))
    region_order = ["SB", "OT", "JC"]  # top to bottom

    # Create color map for consistent cycle coloring
    unique_cycles = set()
    for _, cycle_info, _, _ in all_region_coms:
        # Extract cycle number from "Cycle N, ..." format
        if cycle_info.startswith("Cycle "):
            cycle_num = cycle_info.split()[1]  # Get "N" from "Cycle N, ..."
            unique_cycles.add(f"Cycle {cycle_num}")
        else:
            # Fallback for old format
            cycle_idx = cycle_info.split(',')[0].strip()
            unique_cycles.add(cycle_idx)
    cycle_colors = plt.cm.tab10(range(len(unique_cycles)))
    color_map = dict(zip(sorted(unique_cycles), cycle_colors))

    for ax, name in zip(axes, region_order):
        # Track which cycles have been labeled to avoid duplicate legend entries
        labeled_cycles = set()

        for region_coms, cycle_info, fr, cycle in all_region_coms:
            com = region_coms[name]
            t = np.arange(fr.start, fr.end + 1)
            # Extract cycle identifier for coloring
            if cycle_info.startswith("Cycle "):
                cycle_key = f"Cycle {cycle_info.split()[1]}"
            else:
                cycle_key = cycle_info.split(',')[0].strip()
            color = color_map[cycle_key]

            # Only add legend label once per cycle
            label = cycle_info if cycle_key not in labeled_cycles else ""
            labeled_cycles.add(cycle_key)

            ax.plot(t, com, label=label, color=color)

        # Add vertical reference lines for all cycles (collect unique cycles to avoid duplicates)
        plotted_cycles = set()
        for _, cycle_info, fr, cycle in all_region_coms:
            # Extract cycle identifier for reference lines
            if cycle_info.startswith("Cycle "):
                cycle_key = f"Cycle {cycle_info.split()[1]}"
            else:
                cycle_key = cycle_info.split(',')[0].strip()
            if cycle_key not in plotted_cycles:
                plotted_cycles.add(cycle_key)
                # Black lines at start of flexion and end of extension
                ax.axvline(cycle.flex.start, color='black', linestyle='-', linewidth=1)
                ax.axvline(cycle.ext.end, color='black', linestyle='-', linewidth=1)
                # Dashed gray lines at phase transitions (end of flex, start of ext)
                ax.axvline(cycle.flex.end, color='gray', linestyle='--', linewidth=1)
                ax.axvline(cycle.ext.start, color='gray', linestyle='--', linewidth=1)

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
            frame_ranges = [cycle.flex]
            phase_labels = ["flexion"]
        elif phase == "extension":
            frame_ranges = [cycle.ext]
            phase_labels = ["extension"]
        else:  # "both" - plot flexion and extension phases separately, omitting gap
            frame_ranges = [cycle.flex, cycle.ext]
            phase_labels = ["flexion", "extension"]

        # Create legend label showing frame ranges
        if phase == "both":
            legend_label = f"Cycle {cycle_idx}, frames {cycle.flex.start}-{cycle.flex.end} and {cycle.ext.start}-{cycle.ext.end}"
        else:
            legend_label = f"Cycle {cycle_idx}, {phase}"

        for fr, phase_label in zip(frame_ranges, phase_labels):
            region_coms_window = {
                name: extract_frame_window(com, fr.start, fr.end)
                for name, com in region_coms_full.items()
            }

            # Use the same legend label for both phases of the same cycle
            cycle_info = legend_label
            all_region_coms.append((region_coms_window, cycle_info, fr, cycle))

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
