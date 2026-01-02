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
import argparse

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


def draw_segment_boundaries(video: np.ndarray, radial_regions: np.ndarray, meta) -> np.ndarray:
    """
    Draws the segment boundaries on the video, for verification before showing final results.

    Shows only JC-OT and OT-SB boundaries as white overlay lines on grayscale video.

    Args:
        video: Video array of shape (nframes, h, w)
        radial_regions: Mask array of shape (nframes, h, w) with segment labels
        meta: KneeVideoMeta object with region information

    Returns:
        Video with boundary overlay
    """
    # Input validation
    assert video.shape == radial_regions.shape, f"Shape mismatch: video {video.shape} vs radial_regions {radial_regions.shape}"
    assert video.dtype == np.uint8, f"Video must be uint8, got {video.dtype}"
    assert radial_regions.dtype in [np.uint8, np.int32, np.int64], f"Radial regions must be integer type, got {radial_regions.dtype}"

    # Get segment ranges from metadata
    jc_start, jc_end = meta.regions["JC"].start, meta.regions["JC"].end
    ot_start, ot_end = meta.regions["OT"].start, meta.regions["OT"].end
    sb_start, sb_end = meta.regions["SB"].start, meta.regions["SB"].end

    # For efficiency, only consider boundaries between these specific segment pairs
    # JC-OT: between jc_end and ot_start
    # OT-SB: between ot_end and sb_start
    boundary_pairs = {
        (jc_end, ot_start),  # JC-OT
        (ot_start, jc_end),
        (ot_end, sb_start),  # OT-SB
        (sb_start, ot_end),
    }

    # Create boundary mask
    boundaries = np.zeros_like(video, dtype=bool)
    nfs, h, w = video.shape

    for f in range(nfs):
        seg = radial_regions[f]

        # Horizontal neighbors (left-right)
        if w > 1:
            left  = seg[:, :-1]
            right = seg[:, 1:]

            # Both pixels must be valid segments (not background)
            valid = (left > 0) & (right > 0)

            # Check if this neighbor pair is a boundary we care about
            is_boundary = np.zeros_like(valid, dtype=bool)
            for pair in boundary_pairs:
                is_boundary |= ((left == pair[0]) & (right == pair[1]))

            h_mask = valid & is_boundary

            # Mark boundary pixels (both sides of the edge)
            boundaries[f, :, :-1] |= h_mask
            boundaries[f, :, 1:]  |= h_mask

        # Vertical neighbors (up-down)
        if h > 1:
            up    = seg[:-1, :]
            down  = seg[1:, :]

            # Both pixels must be valid segments (not background)
            valid = (up > 0) & (down > 0)

            # Check if this neighbor pair is a boundary we care about
            is_boundary = np.zeros_like(valid, dtype=bool)
            for pair in boundary_pairs:
                is_boundary |= ((up == pair[0]) & (down == pair[1]))

            v_mask = valid & is_boundary

            # Mark boundary pixels (both sides of the edge)
            boundaries[f, :-1, :] |= v_mask
            boundaries[f, 1:, :]  |= v_mask

    # Create grayscale overlay: white lines on original video
    overlay = video.copy()
    overlay[boundaries] = 255

    return overlay


def normalize_intensity_per_frame_2d(total_sums: np.ndarray) -> np.ndarray:
    """
    Normalize intensity values per frame (column-wise) to 0–100 scale.

    This mirrors the normalization logic from generate_spatiotemporal_heatmaps.py,
    applied to the (n_segments, n_frames) array where each column is a frame.

    Args:
        total_sums: Array with shape (n_segments, n_frames) containing raw intensities

    Returns:
        Normalized array with same shape, values scaled 0–100 per frame
    """
    norm_intensity = total_sums.astype(float).copy()
    n_segments, n_frames = norm_intensity.shape

    for frame_idx in range(n_frames):
        frame_data = norm_intensity[:, frame_idx]
        min_val, max_val = frame_data.min(), frame_data.max()
        if max_val > min_val:
            norm_intensity[:, frame_idx] = 100 * (frame_data - min_val) / (max_val - min_val)
        else:
            norm_intensity[:, frame_idx] = 0

    return norm_intensity


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


def compute_region_metrics(region_arrays: Dict[str, np.ndarray],
                           metrics: List[str],
                           region_ranges: List[RegionRange] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Return per-region time series for requested metrics.

    Parameters
    ----------
    region_arrays : Dict[str, np.ndarray]
        {region_name: region_sums}, each with shape (N_region, n_frames).
    metrics : List[str]
        List of metrics to compute: "com", "total", and/or "flux".
    region_ranges : List[RegionRange], optional
        List of region definitions. If provided and "com" is in metrics,
        COM values will be shifted from local to global segment indices.

    Returns
    -------
    region_metrics : Dict[str, Dict[str, np.ndarray]]
        {region_name: {metric_name: series}}, where each series has shape (n_frames,).
    """
    region_metrics = {}
    for region_name, region_sums in region_arrays.items():
        region_dict = {}
        if "com" in metrics:
            region_dict["com"] = compute_centre_of_mass_region(region_sums)
        if "total" in metrics or "flux" in metrics:
            # flux requires total intensities, so compute if either is requested
            region_dict["total"] = region_sums.sum(axis=0)
        region_metrics[region_name] = region_dict

    # Fix y-axis: shift COM from local (1..N_region) to global segment indices
    if region_ranges is not None and "com" in metrics:
        region_start_indices = {r.name: r.start_idx for r in region_ranges}
        for region_name, metrics_dict in region_metrics.items():
            if "com" in metrics_dict:
                start_1_based = region_start_indices[region_name]
                offset = start_1_based - 1  # convert to 0-based offset
                metrics_dict["com"] = metrics_dict["com"] + offset

    return region_metrics


def compute_boundary_flux(I_SB: np.ndarray, I_JC: np.ndarray) -> Dict:
    """Compute boundary flux between SB-OT and OT-JC regions.

    Parameters
    ----------
    I_SB : np.ndarray
        Shape (n_frames,), total intensity in SB region per frame.
    I_JC : np.ndarray
        Shape (n_frames,), total intensity in JC region per frame.

    Returns
    -------
    flux_data : Dict
        Dictionary containing:
        - 'SB→OT': {'series': flux array of shape (n_frames-1,)}
        - 'OT→JC': {'series': flux array of shape (n_frames-1,)}
        - 'total_flux': scalar, sum of absolute fluxes
        - 'net_flux': scalar, sum of signed fluxes
    """
    # Flux definitions:
    # F_SB_to_OT(t) = I_SB(t) - I_SB(t+1)  [positive when SB loses intensity]
    # F_OT_to_JC(t) = I_JC(t+1) - I_JC(t)  [positive when JC gains intensity]
    F_SB_to_OT = I_SB[:-1] - I_SB[1:]
    F_OT_to_JC = I_JC[1:] - I_JC[:-1]
    
    total_flux = np.sum(np.abs(F_SB_to_OT) + np.abs(F_OT_to_JC))
    net_flux = np.sum(F_SB_to_OT + F_OT_to_JC)
    
    return {
        'SB→OT': {
            'series': F_SB_to_OT,
        },
        'OT→JC': {
            'series': F_OT_to_JC,
        },
        'total_flux': total_flux,
        'net_flux': net_flux,
    }


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


def plot_intra_region_coms_angle_mode(all_cycle_data: List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]],
                                      phase: str,
                                      n_interp_samples: int,
                                      video_title: str,
                                      norm_label: str
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
    fig.suptitle(f"{video_title}: Intra-region COM for selected cycles (angle-based, based on {norm_label})")

    # Place legend on the top subplot (SB)
    axes[0].legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_boundary_flux_angle_mode(all_flux_data: List[Tuple[Dict, np.ndarray, np.ndarray, str, Cycle]],
                                   phase: str,
                                   n_interp_samples: int,
                                   video_title: str,
                                   norm_label: str
                                   ) -> None:
    """Create single subplot of boundary flux vs angle for multiple cycles.

    Parameters
    ----------
    all_flux_data : List[Tuple[Dict, np.ndarray, np.ndarray, str, Cycle]]
        List of (flux_data, x_positions, angles, legend_label, cycle) for each cycle.
        flux_data contains 'SB→OT' and 'OT→JC' keys with 'series', 'total_flux', 'net_flux'.
    phase : str
        "flexion", "extension", or "both".
    n_interp_samples : int
        Number of samples per phase used for interpolation.
    video_title : str
        Title for the plot.
    norm_label : str
        Label indicating normalization status.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    # Fixed colors for boundaries
    boundary_colors = {'SB→OT': 'blue', 'OT→JC': 'red'}
    
    # Create color map for consistent cycle styling
    unique_cycles = set()
    for _, _, _, legend_label, _ in all_flux_data:
        cycle_num = legend_label.split()[1]
        unique_cycles.add(f"Cycle {cycle_num}")
    
    # Use line styles to distinguish cycles
    line_styles = ['-']#, '--', '-.', ':']
    cycle_line_styles = {}
    for i, cycle_key in enumerate(sorted(unique_cycles)):
        cycle_line_styles[cycle_key] = line_styles[i % len(line_styles)]

    # Build x-axis ticks based on angle positions
    all_tick_positions = []
    all_tick_labels = []

    for flux_data, x_positions, angles, legend_label, cycle in all_flux_data:
        if phase == "flexion":
            flex_labels = np.linspace(30, 135, n_interp_samples)
            target_angles = np.arange(30, 136, 15)
            for target_angle in target_angles:
                idx = np.argmin(np.abs(flex_labels - target_angle))
                if idx < len(x_positions) - 1:  # Ensure within flux array bounds
                    all_tick_positions.append(x_positions[idx])
                    all_tick_labels.append(f'{int(target_angle)}°')
        elif phase == "extension":
            ext_labels = np.linspace(135, 30, n_interp_samples)
            target_angles = np.arange(135, 29, -15)
            for target_angle in target_angles:
                idx = np.argmin(np.abs(ext_labels - target_angle))
                if idx < len(x_positions) - 1:  # Ensure within flux array bounds
                    all_tick_positions.append(x_positions[idx])
                    all_tick_labels.append(f'{int(target_angle)}°')
        else:  # "both"
            flex_labels = np.linspace(30, 135, n_interp_samples)
            flex_target_angles = np.arange(30, 136, 15)
            for target_angle in flex_target_angles:
                idx = np.argmin(np.abs(flex_labels - target_angle))
                if idx < len(x_positions) - 1:
                    all_tick_positions.append(x_positions[idx])
                    all_tick_labels.append(f'{int(target_angle)}°')
            
            ext_labels = np.linspace(135, 30, n_interp_samples)
            ext_target_angles = np.arange(120, 44, -15)
            for target_angle in ext_target_angles:
                idx = n_interp_samples + np.argmin(np.abs(ext_labels - target_angle))
                if idx < len(x_positions) - 1:
                    all_tick_positions.append(x_positions[idx])
                    all_tick_labels.append(f'{int(target_angle)}°')

    # Track which (cycle, boundary) combinations have been labeled
    labeled_entries = set()

    for flux_data, x_positions, angles, legend_label, cycle in all_flux_data:
        cycle_num = legend_label.split()[1]
        cycle_key = f"Cycle {cycle_num}"
        line_style = cycle_line_styles[cycle_key]
        
        # x positions for flux (one less than interpolated data due to differencing)
        x_flux = x_positions[:-1]
        
        # Plot both boundaries
        for boundary_name in ['SB→OT', 'OT→JC']:
            flux_series = flux_data[boundary_name]['series']
            color = boundary_colors[boundary_name]
            
            # Create label with scalar metrics
            label_key = (cycle_key, boundary_name)
            if label_key not in labeled_entries:
                total_f = flux_data['total_flux']
                net_f = flux_data['net_flux']
                label = f"{legend_label} {boundary_name} (total={total_f:.1f}, net={net_f:.1f})"
                labeled_entries.add(label_key)
            else:
                label = ""
            
            ax.plot(x_flux, flux_series, label=label, color=color, linestyle=line_style, alpha=0.8)

    # Add cycle demarcation lines
    for i, (flux_data, x_positions, angles, legend_label, cycle) in enumerate(all_flux_data):
        if i > 0:
            cycle_start = x_positions[0]
            ax.axvline(cycle_start, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Black line at end of last cycle
    last_cycle_end = all_flux_data[-1][1][-1]
    ax.axvline(last_cycle_end, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Gray lines at flex/ext transitions (only for "both" phase)
    if phase == "both":
        for flux_data, x_positions, angles, legend_label, cycle in all_flux_data:
            transition_line = x_positions[n_interp_samples-1]
            ax.axvline(transition_line, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Set x-axis limits
    if all_flux_data:
        min_x = min(x_positions[0] for _, x_positions, _, _, _ in all_flux_data)
        max_x = max(x_positions[-1] for _, x_positions, _, _, _ in all_flux_data)
        ax.set_xlim(min_x, max_x)

    # Set x-axis ticks and labels
    if all_tick_positions:
        ax.set_xticks(all_tick_positions)
        ax.set_xticklabels(all_tick_labels)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Knee Angle (°)")
    ax.set_ylabel("Boundary Flux (signed)")
    ax.set_title(f"{video_title}: Boundary Flux for selected cycles (angle-based, based on {norm_label})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    
    plt.tight_layout()
    plt.show()


def plot_intra_region_totals_angle_mode(all_cycle_data: List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]],
                                        phase: str,
                                        n_interp_samples: int,
                                        video_title: str,
                                        norm_label: str
                                        ) -> None:
    """Create 3 stacked subplots (SB, OT, JC) of intra-region total intensity vs angle for multiple cycles.

    Parameters
    ----------
    all_cycle_data : List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]]
        List of (cycle_total_data, x_positions, angles, legend_label, cycle) for each cycle.
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

    for cycle_total_data, x_positions, angles, legend_label, cycle in all_cycle_data:
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

        for cycle_total_data, x_positions, angles, legend_label, cycle in all_cycle_data:
            total = cycle_total_data[name]
            cycle_num = legend_label.split()[1]
            cycle_key = f"Cycle {cycle_num}"
            color = color_map[cycle_key]

            # Only add legend label once per cycle
            label = legend_label if cycle_key not in labeled_cycles else ""
            labeled_cycles.add(cycle_key)

            ax.plot(x_positions, total, label=label, color=color)

        # Add cycle demarcation lines
        # Black lines: boundaries between cycles
        # Gray lines: flex/ext transitions within each cycle
        for i, (cycle_total_data, x_positions, angles, legend_label, cycle) in enumerate(all_cycle_data):
            # Black line at start of each cycle (except if it's the first cycle's start)
            if i > 0:  # Not the first cycle
                cycle_start = x_positions[0]
                ax.axvline(cycle_start, color='black', linestyle='-', linewidth=1)

        # Black line at end of last cycle
        last_cycle_end = all_cycle_data[-1][1][-1]
        ax.axvline(last_cycle_end, color='black', linestyle='-', linewidth=1)

        # Gray lines at flex/ext transitions (only for "both" phase)
        if phase == "both":
            for cycle_total_data, x_positions, angles, legend_label, cycle in all_cycle_data:
                transition_line = x_positions[n_interp_samples-1]
                ax.axvline(transition_line, color='gray', linestyle='--', linewidth=1)

        ax.set_ylabel(f"{name} Total Intensity")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name} Intra-region Total Intensity")

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
    fig.suptitle(f"{video_title}: Intra-region Total Intensity for selected cycles (angle-based, based on {norm_label})")

    # Place legend on the top subplot (SB)
    axes[0].legend(loc="best")

    plt.tight_layout()
    plt.show()


def main(condition, id, nsegs, cycle_indices=None, phase="both", mode="angle", n_interp_samples=105, metrics=None, normalize=True):
    """Main analysis pipeline."""
    if cycle_indices is None:
        cycle_indices = [0]
    if metrics is None:
        metrics = ["com"]

    # Create normalization label for plots and console output
    norm_label = "normalized intensities" if normalize else "raw intensities"
    print(f"Loading video: {condition} {id} (N{nsegs})")
    print(f"Processing cycles: {cycle_indices} in {mode} mode")
    print(f"Computing metrics: {metrics}")
    print(f"Normalization: {norm_label}")

    # Get metadata
    meta = get_knee_meta(condition, int(id), int(nsegs))

    if mode == "angle":
        # New angle-based mode with interpolation and contiguous plotting
        # Load video and masks
        video, masks = load_video_data(condition, id, nsegs)

        # Draw segment boundaries on video
        video_with_boundaries = draw_segment_boundaries(video, masks, meta)
        video_with_boundaries = views.draw_outer_radial_mask_boundary( # Add outer knee boundary
            video_with_boundaries, masks, intensity=255, thickness=1
        )
        views.show_frames([video_with_boundaries, (masks*(255//64))], "Validate data with segment boundaries")

        # Compute intensity data
        total_sums, total_nonzero, segment_labels = load_intensity_data(video, masks)
        
        # Apply normalization if requested
        if normalize:
            total_sums = normalize_intensity_per_frame_2d(total_sums)

        # Split into three anatomical parts
        region_ranges = [ # Get anatomical regions from metadata
            RegionRange(name, reg.start, reg.end)
            for name, reg in meta.regions.items()]
        region_arrays = split_three_parts_indexwise(total_sums, region_ranges)

        # Compute requested metrics over full time series
        region_metrics_full = compute_region_metrics(region_arrays, metrics, region_ranges)

        # Build angle axis for all cycles
        cycle_x_offsets, cycle_angles, cycle_lengths = build_angle_axis_for_cycles(
            cycle_indices, meta, phase, n_interp_samples
        )

        # Plot each requested metric
        video_title = f"{condition} {id} (N{nsegs})"
        for metric in metrics:
            # Collect interpolated data for all cycles
            all_cycle_data = []
            for i, cycle_idx in enumerate(cycle_indices):
                cycle = meta.get_cycle(cycle_idx)

                # Interpolate metric data for each region
                # Note: flux is not stored per-region; we interpolate totals and compute flux later
                cycle_metric_data = {}
                for region_name in region_metrics_full.keys():
                    # Flux uses total intensities, not a per-region flux series
                    source_metric = "total" if metric == "flux" else metric
                    
                    # Concatenate flexion + extension
                    flex_data = extract_frame_window(region_metrics_full[region_name][source_metric],
                                                    cycle.flex.start, cycle.flex.end)
                    ext_data = extract_frame_window(region_metrics_full[region_name][source_metric],
                                                    cycle.ext.start, cycle.ext.end)

                    flex_interp, _ = interpolate_com_to_angle(flex_data, n_interp_samples, 30, 135)
                    ext_interp, _ = interpolate_com_to_angle(ext_data, n_interp_samples, 135, 30)

                    cycle_metric_data[region_name] = np.concatenate([flex_interp, ext_interp])

                # Create legend label
                if phase == "both":
                    legend_label = f"Cycle {cycle_idx+1}, frames {cycle.flex.start+1}-{cycle.flex.end+1} and {cycle.ext.start+1}-{cycle.ext.end+1}"
                else:
                    legend_label = f"Cycle {cycle_idx+1}, {phase}"

                all_cycle_data.append((
                    cycle_metric_data,      # interpolated metric data per region
                    cycle_x_offsets[i],     # x positions for this cycle
                    cycle_angles[i],        # angle values for this cycle
                    legend_label,           # legend label
                    cycle                   # cycle metadata (for reference lines if needed)
                ))

            # Plot the results for this metric
            if metric == "com":
                plot_intra_region_coms_angle_mode(all_cycle_data, phase, n_interp_samples, video_title, norm_label)
            elif metric == "total":
                plot_intra_region_totals_angle_mode(all_cycle_data, phase, n_interp_samples, video_title, norm_label)
            elif metric == "flux":
                # For flux, compute from interpolated SB and JC total intensities
                all_flux_data = []
                for cycle_metric_data, x_positions, angles, legend_label, cycle in all_cycle_data:
                    # Extract interpolated SB and JC intensities
                    I_SB_interp = cycle_metric_data['SB']
                    I_JC_interp = cycle_metric_data['JC']
                    
                    # Compute flux in angle space
                    flux_data = compute_boundary_flux(I_SB_interp, I_JC_interp)
                    
                    all_flux_data.append((flux_data, x_positions, angles, legend_label, cycle))
                
                plot_boundary_flux_angle_mode(all_flux_data, phase, n_interp_samples, video_title, norm_label)


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
    parser.add_argument("--mode", choices=["angle"], default="angle", # TODO: retire and use angle mode always
                       help="Plotting mode: angle (rescaled, contiguous)")
    parser.add_argument("--metric", default="com",
                       help="Comma-separated metrics to plot: com,total,flux (default: com)")
    parser.add_argument("--n-interp-samples", type=int, default=105,
                       help="Number of interpolation samples per phase in angle mode (default: 105)")

    # Normalization toggle (default True)
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--normalize", dest="normalize", action="store_true",
        help="Enable per-frame intensity normalization (default)"
    )
    norm_group.add_argument(
        "--no-normalize", dest="normalize", action="store_false",
        help="Disable per-frame intensity normalization"
    )
    parser.set_defaults(normalize=True)

    args = parser.parse_args()

    cycle_indices = [int(x.strip()) for x in args.cycle_indices.split(',')]
    metrics = [x.strip() for x in args.metric.split(',')]
    main(args.condition, args.id, args.nsegs, cycle_indices, args.phase, args.mode, args.n_interp_samples, metrics, args.normalize)
