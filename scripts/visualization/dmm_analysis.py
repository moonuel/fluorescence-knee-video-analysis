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
import pandas as pd
from pathlib import Path
from utils import io, views, utils
from utils.io import FigureFilename
from core import data_processing as dp
from config.knee_metadata import get_knee_meta, Cycle
import sys
import argparse
from collections.abc import Iterable
import scipy

# Angles to show on x-axis (others will have blank labels)
# IMPORTANT_ANGLE_LABELS = {30, 60, 105, 135}
IMPORTANT_ANGLE_LABELS = {30, 45, 60, 75, 90, 105, 120, 135}


#################################################
# HELPER FUNCTIONS
#################################################

def parse_cycles_arg(cycles_arg: str) -> Tuple[List[int | None], str]:
    """Parse `--cycles` CLI argument.

    Accepts comma-separated 1-based indices, plus placeholders (`_`, `x`, `X`) that
    reserve a blank cycle slot in the angle axis.

    Returns
    -------
    tokens : List[int | None]
        Ordered tokens, where ints are 0-based cycle indices and None is a blank slot.
    cycles_str : str
        Normalized string used for filenames, e.g. "cycles_1,_,3".
    """
    tokens: List[int | None] = []
    rendered: List[str] = []

    if cycles_arg is None:
        return [0], "cycles_1"

    parts = [p.strip() for p in str(cycles_arg).split(',') if p.strip() != ""]
    if not parts:
        return [0], "cycles_1"

    for part in parts:
        if part in {"_", "x", "X"}:
            tokens.append(None)
            rendered.append("_")
            continue

        try:
            idx_1_based = int(part)
        except ValueError as e:
            raise ValueError(
                f"Invalid token in --cycles: {part!r}. Use 1-based integers or one of: _, x, X"
            ) from e

        if idx_1_based < 1:
            raise ValueError(f"Cycle indices in --cycles must be >= 1, got {idx_1_based}")

        tokens.append(idx_1_based - 1)
        rendered.append(str(idx_1_based))

    return tokens, "cycles_" + ",".join(rendered)


def construct_filename(analysis_type: str, meta: 'KneeVideoMeta', normalization: str, cycles: str, modifiers: List[str], extension: str = "png") -> str:
    """Construct a standardized filename for saved figures.

    Parameters
    ----------
    analysis_type : str
        Type of analysis (e.g., "boundary_flux", "intra_region_totals").
    meta : KneeVideoMeta
        Video metadata containing condition, video_id, n_segments.
    normalization : str
        Normalization label (e.g., "normalized", "raw").
    cycles : str
        Cycle specification (e.g., "cycles1-3", "all_cycles").
    modifiers : List[str]
        Additional modifiers (e.g., ["angle_based"]).
    extension : str, optional
        File extension (default "png").

    Returns
    -------
    str
        Constructed filename.
    """
    fig_filename = FigureFilename(
        analysis_type=analysis_type,
        condition=meta.condition,
        video_id=meta.video_id,
        n_segments=meta.n_segments,
        normalization=normalization,
        cycles=cycles,
        modifiers=modifiers,
        extension=extension
    )
    return str(fig_filename)


def get_labeled_angles_for_phase(phase: str, n_interp_samples: int) -> Tuple[List[int], List[str]]:
    """Return the labeled angles (integers) and their display labels for a given phase.
    
    Parameters
    ----------
    phase : str
        "flexion", "extension", or "both".
    n_interp_samples : int
        Number of interpolation samples per phase.
    
    Returns
    -------
    labeled_angles : List[int]
        Integer angles that should be labeled (e.g., [30, 45, 60, ...]).
    angle_labels : List[str]
        Display labels (e.g., ["30°", "45°", "60°", ...]).
    """
    if phase == "flexion":
        labeled_angles = list(range(30, 136, 15))  # 30, 45, ..., 135
        angle_labels = [f"{a}°" for a in labeled_angles]
    elif phase == "extension":
        labeled_angles = list(range(135, 29, -15))  # 135, 120, ..., 30
        angle_labels = [f"{a}°" for a in labeled_angles]
    else:  # "both"
        # Flex half: 30..135
        flex_angles = list(range(30, 136, 15))
        # Ext half: 120..45 (skip 135 and 30)
        ext_angles = list(range(120, 44, -15))
        labeled_angles = flex_angles + ext_angles
        angle_labels = [f"{a}°" for a in flex_angles] + [f"{a}°" for a in ext_angles]
    return labeled_angles, angle_labels


def find_closest_sample_indices(angles_array: np.ndarray, target_angles: List[int]) -> np.ndarray:
    """Find indices in angles_array closest to each target angle.
    
    Parameters
    ----------
    angles_array : np.ndarray
        Shape (n_samples,) of continuous angle values.
    target_angles : List[int]
        Integer angles to locate.
    
    Returns
    -------
    indices : np.ndarray
        Shape (len(target_angles),) of indices into angles_array.
    """
    indices = []
    for target in target_angles:
        idx = np.argmin(np.abs(angles_array - target))
        indices.append(idx)
    return np.array(indices)


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
    jc_start, jc_end = meta.regions["JC"].s, meta.regions["JC"].e
    ot_start, ot_end = meta.regions["OT"].s, meta.regions["OT"].e
    sb_start, sb_end = meta.regions["SB"].s, meta.regions["SB"].e

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


def compute_intensity_data(video: np.ndarray,
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


#################################################
# HELPER FUNCTIONS
#################################################


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
    """Return time series for requested metrics, grouped by metric type.

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
    metric_data : Dict[str, Dict[str, np.ndarray]]
        {metric_name: {series_name: series}}, where each series has shape (n_frames,).
        - For "com" and "total": series_name is region name (e.g., "SB", "OT", "JC")
        - For "flux": series_name is transition (e.g., "SB->OT", "OT->JC")
          plus "total_flux" and "net_flux" scalars.
    """
    metric_data = {m: {} for m in metrics}

    # Compute com and total for each region
    for region_name, region_sums in region_arrays.items():
        if "com" in metrics:
            metric_data["com"][region_name] = compute_centre_of_mass_region(region_sums)
        if "total" in metrics:
            metric_data["total"][region_name] = region_sums.sum(axis=0)
        if "flux" in metrics:
            metric_data["flux"][region_name] = region_sums.sum(axis=0) # Temporarily store total sums for flux computations

    # Fix COM y-axis: shift indices to from local (1..N_region) to global segment indices
    if region_ranges is not None and "com" in metrics:
        region_start_indices = {r.name: r.start_idx for r in region_ranges}
        for region_name in metric_data["com"]:
            start_1_based = region_start_indices[region_name]
            offset = start_1_based - 1  # convert to 0-based offset
            metric_data["com"][region_name] = metric_data["com"][region_name] + offset

    # Compute boundary flux if requested (on raw frame data)
    if "flux" in metrics and "SB" in metric_data["flux"] and "JC" in metric_data["flux"]:
        I_SB = metric_data["flux"]["SB"] # Temporarily stored total sums
        I_JC = metric_data["flux"]["JC"]

        # Smooth the intensity data
        b, a = scipy.signal.butter(1, 0.25, btype='low', analog=False) 
        I_SB = scipy.signal.filtfilt(b, a, I_SB)
        I_JC = scipy.signal.filtfilt(b, a, I_JC)

        # Version with moving average
        # I_SB = pd.Series(I_SB).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
        # I_JC = pd.Series(I_JC).rolling(window=5, center=True, min_periods=1).mean().to_numpy()

        flux_data = compute_boundary_flux(I_SB, I_JC)
        metric_data["flux"] = flux_data # Overwrite with actual flux data

    return metric_data


def compute_boundary_flux(I_SB: np.ndarray, I_JC: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute boundary flux between SB-OT and OT-JC regions.

    Parameters
    ----------
    I_SB : np.ndarray
        Shape (n_frames,), total intensity in SB region per frame.
    I_JC : np.ndarray
        Shape (n_frames,), total intensity in JC region per frame.

    Returns
    -------
    flux_data : Dict[str, np.ndarray]
        Dictionary containing:
        - 'SB->OT': flux array of shape (n_frames-1,)
        - 'OT->JC': flux array of shape (n_frames-1,)
    """
    # Flux definitions:
    # F_SB_to_OT(t) = I_SB(t) - I_SB(t+1)  [positive when SB loses intensity]
    # F_OT_to_JC(t) = I_JC(t+1) - I_JC(t)  [positive when JC gains intensity]
    F_SB_to_OT = I_SB[:-1] - I_SB[1:]
    F_OT_to_JC = I_JC[1:] - I_JC[:-1]

    total_flux = np.sum(np.abs(F_SB_to_OT) + np.abs(F_OT_to_JC))
    net_flux = np.sum(F_SB_to_OT + F_OT_to_JC)    
    return {
        'SB->OT': F_SB_to_OT,
        'OT->JC': F_OT_to_JC,
        "total_flux": total_flux,
        "net_flux": net_flux
    }


def compute_flux_scalars(I_SB: np.ndarray, I_JC: np.ndarray) -> Tuple[float, float]:
    """Compute total and net flux scalars for a time series.

    Parameters
    ----------
    I_SB : np.ndarray
        Shape (n_frames,), total intensity in SB region per frame.
    I_JC : np.ndarray
        Shape (n_frames,), total intensity in JC region per frame.

    Returns
    -------
    total_flux : float
        Sum of absolute fluxes.
    net_flux : float
        Sum of signed fluxes.
    """
    F_SB_to_OT = I_SB[:-1] - I_SB[1:]
    F_OT_to_JC = I_JC[1:] - I_JC[:-1]
    
    total_flux = float(np.sum(np.abs(F_SB_to_OT) + np.abs(F_OT_to_JC)))
    net_flux = float(np.sum(F_SB_to_OT + F_OT_to_JC))
    
    return total_flux, net_flux


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


def interpolate_series_to_angle(series,
                                n_samples: int,
                                angle_start: float,
                                angle_end: float
                                ) -> Tuple:
    """Interpolate a time series to fixed number of samples over an angle range.

    If series is not iterable (e.g., a scalar), returns it as-is with None for angles.

    Parameters
    ----------
    series : array-like or scalar
        Data values for this phase.
    n_samples : int
        Number of samples to interpolate to.
    angle_start : float
        Starting angle in degrees.
    angle_end : float
        Ending angle in degrees.

    Returns
    -------
    interpolated : array-like or scalar
        Interpolated values or original scalar.
    angles : np.ndarray or None
        Corresponding angle values, or None for scalars.
    """
    if not isinstance(series, Iterable):
        return series, None

    # Create angle array
    angles = np.linspace(angle_start, angle_end, n_samples)

    # Create frame indices for interpolation
    n_frames = len(series)
    frame_indices = np.linspace(0, n_frames - 1, n_samples)

    # Interpolate values
    interpolated = np.interp(frame_indices, np.arange(n_frames), series)

    return interpolated, angles


def build_angle_axis_for_cycles(cycle_indices: List[int | None],
                               meta: 'KneeVideoMeta',
                               phase: str,
                               n_interp_samples: int
                               ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Build contiguous angle axis for multiple cycles.

    Parameters
    ----------
    cycle_indices : List[int | None]
        Cycle indices to include (0-based). Use None to insert a blank cycle slot.
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

    for _cycle_idx in cycle_indices:
        if phase == "flexion":
            # Only flexion phase: 30° -> 135°
            _, angles = interpolate_series_to_angle(
                np.array([0]), n_interp_samples, 30, 135  # dummy data, we'll replace later
            )
            cycle_angles.append(angles)
            cycle_lengths.append(n_interp_samples)
            cycle_x_offsets.append(np.arange(current_offset, current_offset + n_interp_samples))
            current_offset += n_interp_samples

        elif phase == "extension":
            # Only extension phase: 135° -> 30°
            _, angles = interpolate_series_to_angle(
                np.array([0]), n_interp_samples, 135, 30  # dummy data, we'll replace later
            )
            cycle_angles.append(angles)
            cycle_lengths.append(n_interp_samples)
            cycle_x_offsets.append(np.arange(current_offset, current_offset + n_interp_samples))
            current_offset += n_interp_samples

        else:  # "both" - flexion + extension concatenated
            # Flexion: 30° -> 135°
            _, flex_angles = interpolate_series_to_angle(
                np.array([0]), n_interp_samples, 30, 135  # dummy data, we'll replace later
            )
            # Extension: 135° -> 30°
            _, ext_angles = interpolate_series_to_angle(
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


def build_angle_ticks(phase: str, n_interp_samples: int, x_positions_list: List[np.ndarray]) -> Tuple[List[float], List[str]]:
    """Build tick positions and labels for angle-based plots.

    Parameters
    ----------
    phase : str
        "flexion", "extension", or "both".
    n_interp_samples : int
        Number of samples per phase.
    x_positions_list : List[np.ndarray]
        List of x_positions arrays for each cycle.

    Returns
    -------
    Tuple[List[float], List[str]]
        Tick positions and corresponding labels.
    """
    all_tick_positions = []
    all_tick_labels = []

    for x_positions in x_positions_list:
        if phase == "flexion":
            flex_labels = np.linspace(30, 135, n_interp_samples)
            target_angles = np.arange(30, 136, 15)
            for target_angle in target_angles:
                idx = np.argmin(np.abs(flex_labels - target_angle))
                if idx < len(x_positions):
                    all_tick_positions.append(x_positions[idx])
                    all_tick_labels.append(f'{int(target_angle)}°')
        elif phase == "extension":
            ext_labels = np.linspace(135, 30, n_interp_samples)
            target_angles = np.arange(135, 29, -15)
            for target_angle in target_angles:
                idx = np.argmin(np.abs(ext_labels - target_angle))
                if idx < len(x_positions):
                    all_tick_positions.append(x_positions[idx])
                    all_tick_labels.append(f'{int(target_angle)}°')
        else:  # "both"
            flex_labels = np.linspace(30, 135, n_interp_samples)
            flex_target_angles = np.arange(30, 136, 15)
            for target_angle in flex_target_angles:
                idx = np.argmin(np.abs(flex_labels - target_angle))
                if idx < len(x_positions):
                    all_tick_positions.append(x_positions[idx])
                    all_tick_labels.append(f'{int(target_angle)}°')

            ext_labels = np.linspace(135, 30, n_interp_samples)
            ext_target_angles = np.arange(120, 44, -15)  # Adjusted to avoid overlap with flex
            for target_angle in ext_target_angles:
                idx = n_interp_samples + np.argmin(np.abs(ext_labels - target_angle))
                if idx < len(x_positions):
                    all_tick_positions.append(x_positions[idx])
                    all_tick_labels.append(f'{int(target_angle)}°')

    return all_tick_positions, all_tick_labels


def plot_intra_region_coms_angle_mode(all_cycle_data: List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]],
                                      phase: str,
                                      n_interp_samples: int,
                                      video_title: str,
                                      norm_label: str,
                                      meta: 'KneeVideoMeta',
                                      blocks: List[Dict],
                                      cycles_str: str) -> None:
    """Create 3 stacked subplots (SB, OT, JC) of intra-region COM vs angle for multiple cycles.

    Parameters
    ----------
    all_cycle_data : List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]]
        List of (cycle_com_data, x_positions, angles, legend_label, cycle) for each cycle.
    phase : str
        "flexion", "extension", or "both".
    n_interp_samples : int
        Number of samples per phase used for interpolation.
    video_title : str
        Title for the plot.
    norm_label : str
        Label indicating normalization status.
    meta : KneeVideoMeta
        Video metadata.
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

    # Build x-axis ticks (use full layout, including placeholder blocks)
    x_positions_list = [b["x_positions"] for b in blocks]
    all_tick_positions, all_tick_labels = build_angle_ticks(phase, n_interp_samples, x_positions_list)

    for ax, name in zip(axes, region_order):
        # Shade placeholder blocks (behind plot elements)
        for b in blocks:
            if b.get("kind") == "blank":
                xs = b["x_positions"]
                ax.axvspan(xs[0], xs[-1], facecolor="0.9", edgecolor="none", alpha=0.6, zorder=-10)

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

        # Add block demarcation lines
        # Black lines: boundaries between blocks (cycles and blanks)
        # Gray lines: flex/ext transitions within each block
        for i, b in enumerate(blocks):
            if i > 0:
                ax.axvline(b["x_positions"][0], color='black', linestyle='-', linewidth=1)

        if blocks:
            ax.axvline(blocks[-1]["x_positions"][-1], color='black', linestyle='-', linewidth=1)

        if phase == "both":
            for b in blocks:
                transition_line = b["x_positions"][n_interp_samples - 1]
                ax.axvline(transition_line, color='gray', linestyle='--', linewidth=1)

        ax.set_ylabel(f"{name} COM")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name} Intra-region COM")

    # Set x-axis limits to eliminate empty space (include placeholder blocks)
    if blocks:
        min_x = min(b["x_positions"][0] for b in blocks)
        max_x = max(b["x_positions"][-1] for b in blocks)
        for ax in axes:
            ax.set_xlim(min_x, max_x)

    # Filter x-axis labels to reduce visual clutter: keep only specific angles
    filtered_tick_labels = []
    if all_tick_labels:
        for label in all_tick_labels:
            angle = None
            if label.endswith("°"):
                try:
                    angle = int(label[:-1])
                except ValueError:
                    angle = None
            if angle in IMPORTANT_ANGLE_LABELS:
                filtered_tick_labels.append(label)
            else:
                # Use a whitespace label so the tick is present but unlabeled
                filtered_tick_labels.append(" ")
    else:
        filtered_tick_labels = all_tick_labels

    # Set x-axis ticks and labels for all subplots
    if all_tick_positions:
        for ax in axes:
            ax.set_xticks(all_tick_positions)
            ax.set_xticklabels(filtered_tick_labels)

    axes[-1].set_xlabel("Knee Angle (°)")
    fig.suptitle(f"{video_title}: Intra-region COM for selected cycles (angle-based, based on {norm_label} intensities)")

    # Place legend on the top subplot (SB)
    axes[0].legend(loc="best")

    # Save figure
    filename = construct_filename("intra_region_coms", meta, norm_label, cycles_str, ["angle_based"])
    save_path = Path("figures") / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path}")

    plt.tight_layout()
    plt.show()

    


def plot_boundary_flux_angle_mode(all_flux_data: List[Tuple[Dict, np.ndarray, np.ndarray, str, Cycle]],
                                   phase: str,
                                   n_interp_samples: int,
                                   video_title: str,
                                   norm_label: str,
                                   meta: 'KneeVideoMeta',
                                   blocks: List[Dict],
                                   cycles_str: str) -> None:
    """Create single subplot of boundary flux vs angle for multiple cycles.

    Parameters
    ----------
    all_flux_data : List[Tuple[Dict, np.ndarray, np.ndarray, str, Cycle]]
        List of (flux_data, x_positions, angles, legend_label, cycle) for each cycle.
        flux_data contains 'SB->OT' and 'OT->JC' keys with 'series', 'total_flux', 'net_flux'.
    phase : str
        "flexion", "extension", or "both".
    n_interp_samples : int
        Number of samples per phase used for interpolation.
    video_title : str
        Title for the plot.
    norm_label : str
        Label indicating normalization status.
    meta : KneeVideoMeta
        Video metadata.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    # Fixed colors for boundaries
    boundary_colors = {'SB->OT': 'blue', 'OT->JC': 'red'}
    
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

    # Shade placeholder blocks (behind plot elements)
    for b in blocks:
        if b.get("kind") == "blank":
            xs = b["x_positions"]
            ax.axvspan(xs[0], xs[-1], facecolor="0.9", edgecolor="none", alpha=0.6, zorder=-10)

    # Build x-axis ticks (use full layout, including placeholder blocks)
    x_positions_list = [b["x_positions"] for b in blocks]
    all_tick_positions, all_tick_labels = build_angle_ticks(phase, n_interp_samples, x_positions_list)

    # Block demarcation lines
    for i, b in enumerate(blocks):
        if i > 0:
            ax.axvline(b["x_positions"][0], color='black', linestyle='-', linewidth=1)
    if blocks:
        ax.axvline(blocks[-1]["x_positions"][-1], color='black', linestyle='-', linewidth=1)
    if phase == "both":
        for b in blocks:
            transition_line = b["x_positions"][n_interp_samples - 1]
            ax.axvline(transition_line, color='gray', linestyle='--', linewidth=1)

    # Track which (cycle, boundary) combinations have been labeled
    labeled_entries = set()

    for flux_data, x_positions, angles, legend_label, cycle in all_flux_data:
        cycle_num = legend_label.split()[1]
        cycle_key = f"Cycle {cycle_num}"
        line_style = cycle_line_styles[cycle_key]
        
        # Get scalar metrics from the flux data
        total_f = flux_data.get('total_flux')
        net_f = flux_data.get('net_flux')
        
        # Plot both boundaries
        for boundary_name in ['SB->OT', 'OT->JC']:
            flux_series = flux_data[boundary_name]
            color = boundary_colors[boundary_name]
            
            # Create label with scalar metrics
            label_key = (cycle_key, boundary_name)
            if label_key not in labeled_entries:
                if total_f is not None and net_f is not None:
                    label = f"{legend_label} {boundary_name} (total={total_f:.1f}, net={net_f:.1f})"
                else:
                    label = f"{legend_label} {boundary_name}"
                labeled_entries.add(label_key)
            else:
                label = ""
            
            ax.plot(x_positions, flux_series, label=label, color=color, linestyle=line_style, alpha=0.8)

    # Set x-axis ticks and labels
    if all_tick_positions:
        ax.set_xticks(all_tick_positions)
        ax.set_xticklabels(all_tick_labels)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Knee Angle (°)")
    ax.set_ylabel("Boundary Flux (signed)")
    ax.set_title(f"{video_title}: Boundary Flux for selected cycles (angle-based, based on {norm_label} intensities)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Set x-axis limits to eliminate empty space (include placeholder blocks)
    if blocks:
        min_x = min(b["x_positions"][0] for b in blocks)
        max_x = max(b["x_positions"][-1] for b in blocks)
        ax.set_xlim(min_x, max_x)

    # Save figure
    filename = construct_filename("boundary_flux", meta, norm_label, cycles_str, ["angle_based"])
    save_path = Path("figures") / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path}")

    plt.tight_layout()
    plt.show()


def plot_intra_region_totals_angle_mode(all_cycle_data: List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]],
                                        phase: str,
                                        n_interp_samples: int,
                                        video_title: str,
                                        norm_label: str,
                                        meta: 'KneeVideoMeta',
                                        blocks: List[Dict],
                                        cycles_str: str) -> None:
    """Create 3 stacked subplots (SB, OT, JC) of intra-region total intensity vs angle for multiple cycles.

    Parameters
    ----------
    all_cycle_data : List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]]
        List of (cycle_total_data, x_positions, angles, legend_label, cycle) for each cycle.
    phase : str
        "flexion", "extension", or "both".
    n_interp_samples : int
        Number of samples per phase used for interpolation.
    video_title : str
        Title for the plot.
    norm_label : str
        Label indicating normalization status.
    meta : KneeVideoMeta
        Video metadata.
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

    # Build x-axis ticks (use full layout, including placeholder blocks)
    x_positions_list = [b["x_positions"] for b in blocks]
    all_tick_positions, all_tick_labels = build_angle_ticks(phase, n_interp_samples, x_positions_list)

    for ax, name in zip(axes, region_order):
        # Shade placeholder blocks (behind plot elements)
        for b in blocks:
            if b.get("kind") == "blank":
                xs = b["x_positions"]
                ax.axvspan(xs[0], xs[-1], facecolor="0.9", edgecolor="none", alpha=0.6, zorder=-10)

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

        # Add block demarcation lines
        # Black lines: boundaries between blocks (cycles and blanks)
        # Gray lines: flex/ext transitions within each block
        for i, b in enumerate(blocks):
            if i > 0:
                ax.axvline(b["x_positions"][0], color='black', linestyle='-', linewidth=1)

        if blocks:
            ax.axvline(blocks[-1]["x_positions"][-1], color='black', linestyle='-', linewidth=1)

        if phase == "both":
            for b in blocks:
                transition_line = b["x_positions"][n_interp_samples - 1]
                ax.axvline(transition_line, color='gray', linestyle='--', linewidth=1)

        ax.set_ylabel(f"{name} Total Intensity")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{name} Intra-region Total Intensity")

    # Set x-axis limits to eliminate empty space (include placeholder blocks)
    if blocks:
        min_x = min(b["x_positions"][0] for b in blocks)
        max_x = max(b["x_positions"][-1] for b in blocks)
        for ax in axes:
            ax.set_xlim(min_x, max_x)

    # Filter x-axis labels to reduce visual clutter: keep only specific angles
    filtered_tick_labels = []
    if all_tick_labels:
        for label in all_tick_labels:
            angle = None
            if label.endswith("°"):
                try:
                    angle = int(label[:-1])
                except ValueError:
                    angle = None
            if angle in IMPORTANT_ANGLE_LABELS:
                filtered_tick_labels.append(label)
            else:
                # Use a whitespace label so the tick is present but unlabeled
                filtered_tick_labels.append(" ")
    else:
        filtered_tick_labels = all_tick_labels

    # Set x-axis ticks and labels for all subplots
    if all_tick_positions:
        for ax in axes:
            ax.set_xticks(all_tick_positions)
            ax.set_xticklabels(filtered_tick_labels)

    axes[-1].set_xlabel("Knee Angle (°)")
    fig.suptitle(f"{video_title}: Intra-region Total Intensity for selected cycles (angle-based, based on {norm_label} intensities)")

    # Place legend on the top subplot (SB)
    axes[0].legend(loc="best")

    # Save figure
    filename = construct_filename("intra_region_totals", meta, norm_label, cycles_str, ["angle_based"])
    save_path = Path("figures") / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path}")

    plt.tight_layout()
    plt.show()


def export_to_excel(all_cycle_data: List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]],
                    meta: 'KneeVideoMeta',
                    phase: str,
                    metric: str,
                    normalize: bool,
                    n_interp_samples: int,
                    output_path: Path) -> None:
    """Export the plotted data to an Excel workbook with multiple sheets.
    
    Parameters
    ----------
    all_cycle_data : List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, str, Cycle]]
        The same data structure used for plotting.
    meta : KneeVideoMeta
        Video metadata.
    phase : str
        "flexion", "extension", or "both".
    metric : str
        "com" or "total".
    normalize : bool
        Whether intensities were normalized.
    n_interp_samples : int
        Number of interpolation samples per phase.
    output_path : Path
        Path to save the Excel file.
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Sheet: run_info
    run_info = pd.DataFrame([{
        "condition": meta.condition,
        "video_id": meta.video_id,
        "n_segments": meta.n_segments,
        "mode": "angle",
        "phase": phase,
        "metric": metric,
        "normalize": normalize,
        "n_interp_samples": n_interp_samples,
        "important_angle_labels": ",".join(str(a) for a in sorted(IMPORTANT_ANGLE_LABELS)),
        "cycles_requested": ",".join(str(cycle_num) for cycle_num in range(1, len(all_cycle_data) + 1))
    }])
    
    # 2. Sheet: cycles
    cycles_data = []
    for i, (_, _, _, legend_label, cycle) in enumerate(all_cycle_data):
        cycles_data.append({
            "cycle_number": i + 1,
            "cycle_index_0based": i,
            "flex_start_frame": cycle.flex.s + 1,  # 1-based
            "flex_end_frame": cycle.flex.e + 1,
            "ext_start_frame": cycle.ext.s + 1,
            "ext_end_frame": cycle.ext.e + 1,
            "legend_label": legend_label
        })
    cycles_df = pd.DataFrame(cycles_data)
    
    # 3. Sheet: timeseries_long (full plotted curves)
    timeseries_rows = []
    for i, (cycle_metric_data, x_positions, angles, legend_label, _) in enumerate(all_cycle_data):
        cycle_num = i + 1
        for sample_idx in range(len(x_positions)):
            timeseries_rows.append({
                "cycle_number": cycle_num,
                "sample_in_cycle": sample_idx,
                "x": x_positions[sample_idx],
                "angle_deg": angles[sample_idx],
                "angle_label": "",  # will fill in labeled_angles sheet
                "SB": cycle_metric_data["SB"][sample_idx],
                "OT": cycle_metric_data["OT"][sample_idx],
                "JC": cycle_metric_data["JC"][sample_idx]
            })
    timeseries_df = pd.DataFrame(timeseries_rows)
    
    # 4. Sheet: labeled_angles (exactly what's shown on plot)
    labeled_rows = []
    labeled_angles, angle_labels = get_labeled_angles_for_phase(phase, n_interp_samples)
    
    # Special rule: for phase="both", the last data element should be labeled 30°
    if phase == "both":
        # Add 30° at the end of extension half for each cycle
        labeled_angles.append(30)
        angle_labels.append("30°")
    
    for i, (cycle_metric_data, x_positions, angles, legend_label, _) in enumerate(all_cycle_data):
        cycle_num = i + 1
        # Find indices for each labeled angle
        indices = find_closest_sample_indices(angles, labeled_angles)
        
        for label_idx, (target_angle, label_str) in enumerate(zip(labeled_angles, angle_labels)):
            sample_idx = indices[label_idx]
            # Determine phase block
            if phase == "flexion":
                phase_block = "flex"
            elif phase == "extension":
                phase_block = "ext"
            else:  # "both"
                # Check if sample is in flexion or extension half
                if sample_idx < n_interp_samples:
                    phase_block = "flex"
                else:
                    phase_block = "ext"
            
            labeled_rows.append({
                "cycle_number": cycle_num,
                "phase_block": phase_block,
                "angle_label_deg": target_angle,
                "x": x_positions[sample_idx],
                "sample_in_cycle": sample_idx,
                "angle_deg": angles[sample_idx],
                "SB": cycle_metric_data["SB"][sample_idx],
                "OT": cycle_metric_data["OT"][sample_idx],
                "JC": cycle_metric_data["JC"][sample_idx]
            })
    
    labeled_df = pd.DataFrame(labeled_rows)
    
    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        run_info.to_excel(writer, sheet_name='run_info', index=False)
        cycles_df.to_excel(writer, sheet_name='cycles', index=False)
        timeseries_df.to_excel(writer, sheet_name='timeseries_long', index=False)
        labeled_df.to_excel(writer, sheet_name='labeled_angles', index=False)
    
    print(f"✅ Excel export saved to: {output_path}")

    
def main(knee_cond, id, nsegs, 
         cycle_idxs: List[int | None] | None = None, phase="both", 
         processing_mode="angle", n_interp_samples=525, 
         metrics=None, is_norm=True, preview=False,
         export_xlsx: bool = False, export_path: str | None = None,
         cycles_str: str | None = None):
    
    # Set default processing
    if cycle_idxs is None:
        cycle_idxs = [0]
    if metrics is None:
        metrics = ["com"]

    if cycles_str is None:
        rendered = ["_" if c is None else str(int(c) + 1) for c in cycle_idxs]
        cycles_str = "cycles_" + ",".join(rendered)

    # Create normalization label for plots and console output
    norm_label = "norm" if is_norm else "raw"
    print(f"Loading video: {knee_cond} {id} (N{nsegs})")
    print(f"Processing cycles: {cycle_idxs} in {processing_mode} mode")
    print(f"Computing metrics: {metrics}")
    print(f"Normalization: {norm_label}")

    # Get metadata
    knee_meta = get_knee_meta(knee_cond, int(id), int(nsegs))

    # New angle-based mode with interpolation and contiguous plotting
    # Load video and masks
    video, masks = load_video_data(knee_cond, id, nsegs)

    # Draw segment boundaries on video
    if preview: 
        video_with_boundaries = draw_segment_boundaries(video, masks, knee_meta)
        video_with_boundaries = views.draw_outer_radial_mask_boundary( # Add outer knee boundary
            video_with_boundaries, masks, intensity=255, thickness=1
        )
        views.show_frames([video_with_boundaries, (masks*(255//64))], "Validate data with segment boundaries")

    # Compute intensity data
    total_sums, total_nonzero, segment_labels = compute_intensity_data(video, masks)
    
    # Apply normalization if requested
    if is_norm:
        total_sums = normalize_intensity_per_frame_2d(total_sums)

    # Split into three anatomical parts
    region_ranges = [ # Get anatomical regions from metadata
        RegionRange(name, reg.s, reg.e)
        for name, reg in knee_meta.regions.items()]

    # Build angle axis for all blocks (cycles and placeholders)
    cycle_x_offsets, cycle_angles, cycle_lengths = build_angle_axis_for_cycles(
        cycle_idxs, knee_meta, phase, n_interp_samples
    )

    # Build layout blocks in token order; metrics are computed only for real cycles.
    blocks: List[Dict] = []

    # Interpolate all metrics into angle domain for all cycles
    # Structure: [{cycle_idx, cycle, legend_label, x_positions, angles, metrics: {metric_name: {series_name: array}}}, ...]
    all_cycle_data = [] 
    for i, cycle_idx in enumerate(cycle_idxs):
        if cycle_idx is None:
            blocks.append({
                "kind": "blank",
                "x_positions": cycle_x_offsets[i],
                "angles": cycle_angles[i],
            })
            continue

        blocks.append({
            "kind": "cycle",
            "cycle_idx": cycle_idx,
            "x_positions": cycle_x_offsets[i],
            "angles": cycle_angles[i],
        })

        cycle = knee_meta.get_cycle(cycle_idx)
        
        # Extract cycle
        flex_data = total_sums[:, cycle.flex.s:cycle.flex.e + 1]
        ext_data = total_sums[:, cycle.ext.s:cycle.ext.e + 1]

        # Split cycle into anatomical regions
        flex_regions = split_three_parts_indexwise(flex_data, region_ranges)
        ext_regions = split_three_parts_indexwise(ext_data, region_ranges)

        # Compute metrics over cycle 
        flex_metrics_data = compute_region_metrics(flex_regions, metrics, region_ranges)
        ext_metrics_data = compute_region_metrics(ext_regions, metrics, region_ranges)
        # Returns dict of format {"metric": {region_name: array}}
        
        # Interpolate metrics into angle domain
        flex_metrics_interp = {}
        ext_metrics_interp = {}
        for metric in metrics:
            # Interpolate flex metrics
            flex_interpolated = {}
            for series_name, data in flex_metrics_data[metric].items():
                interp, _ = interpolate_series_to_angle(data, n_interp_samples, 30, 135)
                flex_interpolated[series_name] = interp
            flex_metrics_interp[metric] = flex_interpolated

            # Interpolate ext metrics
            ext_interpolated = {}
            for series_name, data in ext_metrics_data[metric].items():
                interp, _ = interpolate_series_to_angle(data, n_interp_samples, 135, 30)
                ext_interpolated[series_name] = interp
            ext_metrics_interp[metric] = ext_interpolated

        # Concatenate flex and ext metrics
        cycle_metrics = {}
        for metric in metrics:
            cycle_metrics[metric] = {}
            for series_name in flex_metrics_interp[metric]:
                flex_data = flex_metrics_interp[metric][series_name]
                ext_data = ext_metrics_interp[metric][series_name]
                if isinstance(flex_data, Iterable) and isinstance(ext_data, Iterable):
                    cycle_metrics[metric][series_name] = np.concatenate([flex_data, ext_data])
                else:
                    # For scalars, keep as is (e.g., total_flux, net_flux for flux metric)
                    cycle_metrics[metric][series_name] = flex_data  # Assuming flex and ext are the same or handle separately if needed

        # Create legend label
        if phase == "both":
            legend_label = f"Cycle {cycle_idx+1}, frames {cycle.flex.s+1}-{cycle.flex.e+1} and {cycle.ext.s+1}-{cycle.ext.e+1}"
        else:
            legend_label = f"Cycle {cycle_idx+1}, {phase}"
        
        all_cycle_data.append({
            "cycle_idx": cycle_idx,
            "cycle": cycle,
            "legend_label": legend_label,
            "x_positions": cycle_x_offsets[i],
            "angles": cycle_angles[i],
            "metrics": cycle_metrics, # NOTE: cycle_metrics was a dict of form {"metric": iterable}
        })

    # Plot each requested metric
    video_title = f"{knee_cond} {id} (N{nsegs})"
    for metric in metrics:
        # Convert to tuple format expected by plotting functions
        all_metric_data = [
            (
                cycle_data["metrics"][metric],
                cycle_data["x_positions"],
                cycle_data["angles"],
                cycle_data["legend_label"],
                cycle_data["cycle"]
            )
            for cycle_data in all_cycle_data
        ]
        
        if metric == "com":
            # if export_xlsx:
            #     out_path = (
            #         Path(export_path)
            #         if export_path
            #         else Path("figures") / "dmm_analysis_exports" / f"dmm_analysis_{meta.condition}_{meta.video_id}_N{meta.n_segments}_{metric}_{phase}.xlsx"
            #     )
            #     export_to_excel(all_metric_data, meta, phase, metric, normalize, n_interp_samples, out_path)
            plot_intra_region_coms_angle_mode(
                all_metric_data, phase, n_interp_samples, video_title, norm_label, knee_meta,
                blocks=blocks, cycles_str=cycles_str,
            )
        elif metric == "total":
            # if export_xlsx:
            #     out_path = (
            #         Path(export_path)
            #         if export_path
            #         else Path("figures") / "dmm_analysis_exports" / f"dmm_analysis_{meta.condition}_{meta.video_id}_N{meta.n_segments}_{metric}_{phase}.xlsx"
            #     )
            #     export_to_excel(all_metric_data, meta, phase, metric, normalize, n_interp_samples, out_path)
            plot_intra_region_totals_angle_mode(
                all_metric_data, phase, n_interp_samples, video_title, norm_label, knee_meta,
                blocks=blocks, cycles_str=cycles_str,
            )
        elif metric == "flux":
            plot_boundary_flux_angle_mode(
                all_metric_data, phase, n_interp_samples, video_title, norm_label, knee_meta,
                blocks=blocks, cycles_str=cycles_str,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DMM knee video analysis")
    parser.add_argument("condition", help="Condition (e.g., normal, aging, dmm-0w)")
    parser.add_argument("id", help="Video ID")
    parser.add_argument("nsegs", help="Number of segments")
    parser.add_argument(
        "--cycles",
        default="1",
        help=(
            "Comma-separated 1-based cycle indices. Supports placeholders '_'/'x'/'X' to insert blank cycle slots "
            "for alignment. Example: --cycles 1,_,3"
        ),
    )
    parser.add_argument("phase", nargs='?', default="both",
                       choices=["flexion", "extension", "both"],
                       help="Phase to plot (default: both)")
    parser.add_argument("--mode", choices=["angle"], default="angle", # TODO: retire and use angle mode always
                       help="Plotting mode: angle (rescaled, contiguous)")
    parser.add_argument("--metric", default="com",
                       help="Comma-separated metrics to plot: com,total,flux (default: com)")
    parser.add_argument("--n-interp-samples", type=int, default=525,
                       help="Number of interpolation samples per phase in angle mode (default: 525)")

    # Excel export
    parser.add_argument(
        "--export-xlsx", action="store_true",
        help="Export plotted data (angle mode) to an Excel workbook"
    )
    parser.add_argument(
        "--export-path", default=None,
        help="Optional output .xlsx path (default: figures/dmm_analysis_exports/...)"
    )

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

    # Preview toggle (default True)
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--preview", dest="preview", action="store_true",
        help="Enable video preview before analysis (default)"
    )
    preview_group.add_argument(
        "--no-preview", dest="preview", action="store_false",
        help="Disable video preview before analysis"
    )
    parser.set_defaults(preview=True)

    args = parser.parse_args()

    cycle_tokens, cycles_str = parse_cycles_arg(args.cycles)
    metrics = [x.strip() for x in args.metric.split(',')]
    main(
        args.condition,
        args.id,
        args.nsegs,
        cycle_tokens,
        args.phase,
        args.mode,
        args.n_interp_samples,
        metrics,
        args.normalize,
        args.preview,
        export_xlsx=args.export_xlsx,
        export_path=args.export_path,
        cycles_str=cycles_str,
    )
