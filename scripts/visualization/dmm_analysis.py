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


def compute_centre_of_mass_region(region_sums: np.ndarray,
                                  positions: np.ndarray | None = None) -> np.ndarray:
    """Compute intra-region COM per frame using specified position coordinates.

    Parameters
    ----------
    region_sums : np.ndarray
        Shape (N_region, n_frames).
    positions : np.ndarray, optional
        Shape (N_region,), position coordinates for each segment.
        If None, uses 1..N_region (local coordinates).

    Returns
    -------
    com : np.ndarray
        Shape (n_frames,), COM values in the coordinate system specified by positions.
    """
    N_region, nfs = region_sums.shape
    if positions is None:
        positions = np.arange(1, N_region + 1)
    positions = positions.reshape(-1, 1)

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


def interpolate_com_to_angle(com_series: np.ndarray,
                             n_samples: int,
                             angle_start: float,
                             angle_end: float
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate COM series to fixed number of samples over an angle range.

    Parameters
    ----------
    com_series : np.ndarray
        Shape (n_frames,), COM values for this phase.
    n_samples : int
        Number of samples to interpolate to.
    angle_start : float
        Starting angle in degrees.
    angle_end : float
        Ending angle in degrees.

    Returns
    -------
    interpolated_com : np.ndarray
        Shape (n_samples,), interpolated COM values.
    angles : np.ndarray
        Shape (n_samples,), corresponding angle values.
    """
    # Create angle array
    angles = np.linspace(angle_start, angle_end, n_samples)

    # Create frame indices for interpolation
    n_frames = len(com_series)
    frame_indices = np.linspace(0, n_frames - 1, n_samples)

    # Interpolate COM values
    interpolated_com = np.interp(frame_indices, np.arange(n_frames), com_series)

    return interpolated_com, angles


def build_angle_axis_for_cycles(cycle_indices: List[int],
                               meta: 'KneeVideoMeta',
                               phase: str,
                               n_interp_samples: int
                               ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Build contiguous angle axis for multiple cycles.

    Parameters
    ----------
    cycle_indices : List[int]
        Cycle indices to include.
    meta : KneeVideoMeta
        Metadata for this knee.
    phase : str
        "flexion", "extension", or "both".
    n_interp_samples : int
        Samples per phase.

    Returns
    -------
    cycle_x_offsets : List[np.ndarray]
        X-axis positions for each cycle (global coordinates).
    cycle_angles : List[np.ndarray]
        Angle arrays for each cycle.
    cycle_lengths : List[int]
        Length of each cycle in samples.
    """
    cycle_x_offsets = []
    cycle_angles = []
    cycle_lengths = []
    current_offset = 0

    for cycle_idx in cycle_indices:
        cycle = meta.get_cycle(cycle_idx) # Does this have a purpose?

        if phase == "flexion":
            # Only flexion phase: 30° → 135°
            _, angles = interpolate_com_to_angle(
                np.array([0]), n_interp_samples, 30, 135  # dummy data, we'll replace later
            )
            cycle_angles.append(angles)
            cycle_lengths.append(n_interp_samples)
            cycle_x_offsets.append(np.arange(current_offset, current_offset + n_interp_samples))
            current_offset += n_interp_samples

        elif phase == "extension":
            # Only extension phase: 135° → 30°
            _, angles = interpolate_com_to_angle(
                np.array([0]), n_interp_samples, 135, 30  # dummy data, we'll replace later
            )
            cycle_angles.append(angles)
            cycle_lengths.append(n_interp_samples)
            cycle_x_offsets.append(np.arange(current_offset, current_offset + n_interp_samples))
            current_offset += n_interp_samples

        else:  # "both" - flexion + extension concatenated
            # Flexion: 30° → 135°
            _, flex_angles = interpolate_com_to_angle(
                np.array([0]), n_interp_samples, 30, 135  # dummy data, we'll replace later
            )
            # Extension: 135° → 30°
            _, ext_angles = interpolate_com_to_angle(
                np.array([0]), n_interp_samples, 135, 30  # dummy data, we'll replace later
            )

            # Concatenate angles for this cycle
            cycle_angle_block = np.concatenate([flex_angles, ext_angles])
            cycle_angles.append(cycle_angle_block)
            cycle_lengths.append(2 * n_interp_samples)

            # X positions for this cycle block
            cycle_x_offsets.append(np.arange(current_offset, current_offset + 2 * n_interp_samples))
            current_offset += 2 * n_interp_samples

    return cycle_x_offsets, cycle_angles, cycle_lengths


def plot_intra_region_coms_frame_mode(all_region_coms: List[Tuple[Dict[str, np.ndarray], str, 'FrameRange', Cycle]]
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

    # Place legend on the top subplot (SB)
    axes[0].legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_intra_region_coms_angle_mode(all_cycle_data: List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]],
                                      phase: str,
                                      n_interp_samples: int
                                      ) -> None:
    """Create 3 stacked subplots (SB, OT, JC) of intra-region COM vs angle for multiple cycles.

    Parameters
    ----------
    all_cycle_data : List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]]
        List of (cycle_com_data, x_positions, angles, legend_label, cycle) for each cycle.
    phase : str
        "flexion", "extension", or "both".
    n_interp_samples : int
        Number of samples per phase used for interpolation.
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
    region_order = ["SB", "OT", "JC"]  # top to bottom

    # Create color map for consistent cycle coloring
    unique_cycles = set()
    for _, _, _, legend_label, _ in all_cycle_data:
        cycle_num = legend_label.split()[1]  # Extract "N" from "Cycle N, ..."
        unique_cycles.add(f"Cycle {cycle_num}")
    cycle_colors = plt.cm.tab10(range(len(unique_cycles)))
    color_map = dict(zip(sorted(unique_cycles), cycle_colors))

    # Build x-axis ticks for each cycle based on angle positions
    all_tick_positions = []
    all_tick_labels = []

    for cycle_com_data, x_positions, angles, legend_label, cycle in all_cycle_data:
        # Define angle mappings for this cycle's block
        if phase == "flexion":
            # Only flexion phase: 30° → 135°
            flex_labels = np.linspace(30, 135, n_interp_samples)
            target_angles = np.arange(30, 136, 15)  # 30, 45, 60, 75, 90, 105, 120, 135
            # Find indices in flex_labels closest to target_angles
            for target_angle in target_angles:
                idx = np.argmin(np.abs(flex_labels - target_angle))
                all_tick_positions.append(x_positions[idx])
                all_tick_labels.append(f'{int(target_angle)}°')

        elif phase == "extension":
            # Only extension phase: 135° → 30°
            ext_labels = np.linspace(135, 30, n_interp_samples)
            target_angles = np.arange(135, 29, -15)  # 135, 120, 105, 90, 75, 60, 45, 30
            # Find indices in ext_labels closest to target_angles
            for target_angle in target_angles:
                idx = np.argmin(np.abs(ext_labels - target_angle))
                all_tick_positions.append(x_positions[idx])
                all_tick_labels.append(f'{int(target_angle)}°')

        else:  # "both" - flexion + extension concatenated
            # Flexion part: 30° → 135°
            flex_labels = np.linspace(30, 135, n_interp_samples)
            flex_target_angles = np.arange(30, 136, 15)  # 30, 45, 60, 75, 90, 105, 120, 135
            for target_angle in flex_target_angles:
                idx = np.argmin(np.abs(flex_labels - target_angle))
                all_tick_positions.append(x_positions[idx])
                all_tick_labels.append(f'{int(target_angle)}°')

            # Extension part: 135° → 30°
            ext_labels = np.linspace(135, 30, n_interp_samples)
            ext_target_angles = np.arange(120, 44, -15)  # 120, 105, 90, 75, 60, 45 (skip 135 and 30)
            for target_angle in ext_target_angles:
                idx = n_interp_samples + np.argmin(np.abs(ext_labels - target_angle))
                all_tick_positions.append(x_positions[idx])
                all_tick_labels.append(f'{int(target_angle)}°')

    for ax, name in zip(axes, region_order):
        # Track which cycles have been labeled to avoid duplicate legend entries
        labeled_cycles = set()

        for cycle_com_data, x_positions, angles, legend_label, cycle in all_cycle_data:
            com = cycle_com_data[name]
            cycle_num = legend_label.split()[1]
            cycle_key = f"Cycle {cycle_num}"
            color = color_map[cycle_key]

            # Only add legend label once per cycle
            label = legend_label if cycle_key not in labeled_cycles else ""
            labeled_cycles.add(cycle_key)

            ax.plot(x_positions, com, label=label, color=color)

        # Add cycle demarcation lines
        # Black lines: boundaries between cycles
        # Gray lines: flex/ext transitions within each cycle
        for i, (cycle_com_data, x_positions, angles, legend_label, cycle) in enumerate(all_cycle_data):
            # Black line at start of each cycle (except if it's the first cycle's start)
            if i > 0:  # Not the first cycle
                cycle_start = x_positions[0]
                ax.axvline(cycle_start, color='black', linestyle='-', linewidth=1)

        # Black line at end of last cycle
        last_cycle_end = all_cycle_data[-1][1][-1]
        ax.axvline(last_cycle_end, color='black', linestyle='-', linewidth=1)

        # Gray lines at flex/ext transitions (only for "both" phase)
        if phase == "both":
            for cycle_com_data, x_positions, angles, legend_label, cycle in all_cycle_data:
                transition_line = x_positions[n_interp_samples-1]
                ax.axvline(transition_line, color='gray', linestyle='--', linewidth=1)

        ax.set_ylabel(f"{name} COM")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name} Intra-region COM")

    # Set x-axis limits to eliminate empty space
    if all_cycle_data:
        min_x = min(x_positions[0] for _, x_positions, _, _, _ in all_cycle_data)
        max_x = max(x_positions[-1] for _, x_positions, _, _, _ in all_cycle_data)
        for ax in axes:
            ax.set_xlim(min_x, max_x)

    # Set x-axis ticks and labels for all subplots
    if all_tick_positions:
        for ax in axes:
            ax.set_xticks(all_tick_positions)
            ax.set_xticklabels(all_tick_labels)

    axes[-1].set_xlabel("Knee Angle (°)")
    fig.suptitle(f"Intra-region COM for selected cycles (angle-based)")

    # Place legend on the top subplot (SB)
    axes[0].legend(loc="best")

    plt.tight_layout()
    plt.show()


def main(condition, id, nsegs, cycle_indices=None, phase="both", mode="angle", n_interp_samples=105):
    """Main analysis pipeline."""
    if cycle_indices is None:
        cycle_indices = [0]

    # Get metadata
    meta = get_knee_meta(condition, int(id), int(nsegs))

    if mode == "frame":
        # Original frame-based mode
        # Load video and masks
        video, masks = load_video_data(condition, id, nsegs)
        views.show_frames([video, (masks*(255//64))], "Validate data")

        # Compute intensity data
        total_sums, total_nonzero, segment_labels = load_intensity_data(video, masks)

        # Get anatomical regions from metadata
        region_ranges = [
            RegionRange(name, reg.start, reg.end)
            for name, reg in meta.regions.items()
        ]

        # Split into three anatomical parts
        region_arrays = split_three_parts_indexwise(total_sums, region_ranges)

        # Compute intra-region COM over full time series using true segment indices
        region_coms_full = {
            region.name: compute_centre_of_mass_region(
                region_arrays[region.name],
                positions=np.arange(region.start_idx, region.end_idx + 1)
            )
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

            # Create legend label showing frame ranges (convert from 0-based to 1-based for transparency)
            if phase == "both":
                legend_label = f"Cycle {cycle_idx+1}, frames {cycle.flex.start+1}-{cycle.flex.end+1} and {cycle.ext.start+1}-{cycle.ext.end+1}"
            else:
                legend_label = f"Cycle {cycle_idx+1}, {phase}"

            for fr, phase_label in zip(frame_ranges, phase_labels):
                region_coms_window = {
                    name: extract_frame_window(com, fr.start, fr.end)
                    for name, com in region_coms_full.items()
                }

                # Use the same legend label for both phases of the same cycle
                cycle_info = legend_label
                all_region_coms.append((region_coms_window, cycle_info, fr, cycle))

        # Plot the results
        plot_intra_region_coms_frame_mode(all_region_coms)

    else:  # mode == "angle"
        # New angle-based mode with interpolation and contiguous plotting
        # Load video and masks
        video, masks = load_video_data(condition, id, nsegs)
        views.show_frames([video, (masks*(255//64))], "Validate data")

        # Compute intensity data
        total_sums, total_nonzero, segment_labels = load_intensity_data(video, masks)

        # Get anatomical regions from metadata
        region_ranges = [
            RegionRange(name, reg.start, reg.end)
            for name, reg in meta.regions.items()
        ]

        # Split into three anatomical parts
        region_arrays = split_three_parts_indexwise(total_sums, region_ranges)

        # Compute intra-region COM over full time series using true segment indices
        region_coms_full = {
            region.name: compute_centre_of_mass_region(
                region_arrays[region.name],
                positions=np.arange(region.start_idx, region.end_idx + 1)
            )
            for region in region_ranges
        }

        # Build angle axis for all cycles
        cycle_x_offsets, cycle_angles, cycle_lengths = build_angle_axis_for_cycles(
            cycle_indices, meta, phase, n_interp_samples
        )

        # Collect interpolated data for all cycles
        all_cycle_data = []
        for i, cycle_idx in enumerate(cycle_indices):
            cycle = meta.get_cycle(cycle_idx)

            # Interpolate COM data for each region
            cycle_com_data = {}
            for region_name in region_coms_full.keys():
                if phase == "flexion":
                    # Only flexion phase
                    flex_com = extract_frame_window(region_coms_full[region_name],
                                                   cycle.flex.start, cycle.flex.end)
                    interp_com, _ = interpolate_com_to_angle(flex_com, n_interp_samples, 30, 135)
                    cycle_com_data[region_name] = interp_com

                elif phase == "extension":
                    # Only extension phase
                    ext_com = extract_frame_window(region_coms_full[region_name],
                                                  cycle.ext.start, cycle.ext.end)
                    interp_com, _ = interpolate_com_to_angle(ext_com, n_interp_samples, 135, 30)
                    cycle_com_data[region_name] = interp_com

                else:  # "both" - concatenate flexion + extension
                    flex_com = extract_frame_window(region_coms_full[region_name],
                                                   cycle.flex.start, cycle.flex.end)
                    ext_com = extract_frame_window(region_coms_full[region_name],
                                                  cycle.ext.start, cycle.ext.end)

                    flex_interp, _ = interpolate_com_to_angle(flex_com, n_interp_samples, 30, 135)
                    ext_interp, _ = interpolate_com_to_angle(ext_com, n_interp_samples, 135, 30)
                    
                    cycle_com_data[region_name] = np.concatenate([flex_interp, ext_interp])

            # Create legend label
            if phase == "both":
                legend_label = f"Cycle {cycle_idx+1}, frames {cycle.flex.start+1}-{cycle.flex.end+1} and {cycle.ext.start+1}-{cycle.ext.end+1}"
            else:
                legend_label = f"Cycle {cycle_idx+1}, {phase}"

            all_cycle_data.append((
                cycle_com_data,           # interpolated COM data per region
                cycle_x_offsets[i],       # x positions for this cycle
                cycle_angles[i],          # angle values for this cycle
                legend_label,             # legend label
                cycle                     # cycle metadata (for reference lines if needed)
            ))

        # Plot the results
        plot_intra_region_coms_angle_mode(all_cycle_data, phase, n_interp_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DMM knee video analysis")
    parser.add_argument("condition", help="Condition (e.g., normal, aging, dmm-0w)")
    parser.add_argument("id", help="Video ID")
    parser.add_argument("nsegs", help="Number of segments")
    parser.add_argument("cycle_indices", nargs='?', default="0",
                       help="Comma-separated cycle indices (default: 0)")
    parser.add_argument("phase", nargs='?', default="both",
                       choices=["flexion", "extension", "both"],
                       help="Phase to plot (default: both)")
    parser.add_argument("--mode", choices=["angle", "frame"], default="angle",
                       help="Plotting mode: angle (rescaled, contiguous) or frame (default: angle)")
    parser.add_argument("--n-interp-samples", type=int, default=105,
                       help="Number of interpolation samples per phase in angle mode (default: 105)")

    args = parser.parse_args()

    cycle_indices = [int(x.strip()) for x in args.cycle_indices.split(',')]
    main(args.condition, args.id, args.nsegs, cycle_indices, args.phase, args.mode, args.n_interp_samples)
