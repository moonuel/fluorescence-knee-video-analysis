"""Template script for saving comprehensive data for knee video analysis in a consistent format into an Excel spreadsheet file.

Data format:
    Sheet 1: Total Pixel Intensity per radial segment, for every frame
    Sheet 2: Total number of non-zero pixels per radial segment, for every frame
    Sheet 3: File number, total number of frames, frame numbers for each cycle, and segment numbers assigned to the left/middle/right parts of the knee. 
"""

from utils import utils, io, views
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Tuple, Dict
import sys
import argparse
from config.knee_metadata import get_knee_meta

# Get project root directory for robust path handling
PROJECT_ROOT = Path(__file__).parent.parent.parent

#==============================================================================
#                                HELPER FUNCTIONS
#==============================================================================

def load_masks(filepath:str) -> np.ndarray:
    """Loads the mask at the specified location. 
    Handles both old uint8 radial masks with shape (nmasks, nframes, h, w) and new uint8 radial masks with shape (nframes, h, w).
    
    Old mask arrays are very space-inefficient and have one dimension for each segment. New mask arrays use a unique numerical label from {1...N} instead."""

    masks = io.load_nparray(filepath)

    if not masks.dtype == np.uint8:
        raise ValueError(f"File is not of type uint8. Is it a radial mask? Given: {masks.dtype=}")
    
    # Convert inefficient mask array to efficient array
    if len(masks.shape) == 4:
        N = masks.shape[0] # Expected shape: (nmasks, nframes, h, w)
        masks_bool = np.zeros(shape=masks.shape[1:], dtype=np.uint8) # Expected shape: (nframes, h, w)
        for n in range(N):
            masks_bool[masks[n] > 0] = n+1 # Convert each slice of inefficient array to a numerical label from {1...N}
        masks = masks_bool

    assert len(masks.shape) == 3 # Soft check that output is shape (nfs, h, w)

    return masks


def load_video(filepath:str) -> np.ndarray:
    """Loads the video at the specified location."""
    video = io.load_nparray(filepath)

    if not video.dtype == np.uint8: 
        raise ValueError(f"File is not of type uint8. Is it a video? Given: {video.dtype=}")
    
    if not len(video.shape) == 3:
        raise ValueError(f"File is not compatible with shape (nfs, h, w). Is it a grayscale video? Given: {video.shape=}")
    
    return video


def compute_sums_nonzeros(masks: np.ndarray, video: np.ndarray):
    """
    Compute total pixel intensities and non-zero pixel counts per segment for each frame.

    Args:
        masks: Array of shape (nframes, h, w) with segment labels
        video: Array of shape (nframes, h, w) with pixel intensities

    Returns:
        Tuple of (total_sums, total_nonzero) where each is shape (N_segments, nframes)
    """
    assert masks.shape == video.shape  # Sanity check that masks and video match
    nfs, h, w = video.shape

    mask_lbls = np.unique(masks[masks > 0]).astype(int)  # Returns sorted list of unique non-zero labels
    N = len(mask_lbls)

    print(f"Computing sums for {N} segments across {nfs} frames")

    # Calculate total pixel intensities within each segment of the video
    total_sums = np.zeros(shape=(N, nfs), dtype=int)
    for n, lbl in enumerate(mask_lbls):
        for f in range(nfs):
            frame = video[f]
            mask_f = masks[f]
            total_sums[n, f] = frame[mask_f == lbl].sum()

    # Calculate number of non-zero pixels within each segment of the video (for normalization purposes)
    total_nonzero = np.zeros((N, nfs), dtype=int)
    for n, lbl in enumerate(mask_lbls):
        for f in range(nfs):
            frame = video[f]
            mask_f = masks[f]
            total_nonzero[n, f] = np.count_nonzero(frame[mask_f == lbl])

    assert total_sums.shape == total_nonzero.shape  # Sanity check

    print(f"{total_sums[:, 0]=}")
    print(f"{total_nonzero[:, 0]=}")

    return total_sums, total_nonzero


def discover_available_videos(segmented_dir: Path) -> Dict[int, Dict]:
    """
    Scan directory for segmented video files and extract metadata.
    Returns: {video_id: {'type': str, 'N_values': set}}
    """
    videos = {}
    pattern = "*_video_N*.npy"

    for file in segmented_dir.glob(pattern):
        # Parse: "aging_1339_video_N64.npy" or "normal_0308_video_N16.npy"
        parts = file.stem.split("_")
        knee_type = parts[0]  # "aging" or "normal"
        video_id_str = parts[1]  # "1339" or "0308"
        N_part = parts[3]  # "N64" or "N16"

        video_id = int(video_id_str)
        N = int(N_part[1:])  # Extract number after 'N'

        if video_id not in videos:
            videos[video_id] = {'type': knee_type, 'N_values': set()}
        videos[video_id]['N_values'].add(N)

    return videos


def parse_arguments():
    """
    Parse command line arguments with dynamic validation based on available files.
    """
    parser = argparse.ArgumentParser(
        description="Prepare intensity data for knee video analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python prepare_intensity_data.py --list\n"
               "  python prepare_intensity_data.py 1339 64\n"
               "  python prepare_intensity_data.py --all"
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help="List all available video IDs and segment counts"
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help="Process all available video IDs and segment combinations"
    )

    parser.add_argument(
        'video_id',
        type=int,
        nargs='?',
        help="Video ID number"
    )

    parser.add_argument(
        'N',
        type=int,
        nargs='?',
        help="Number of radial segments"
    )

    args = parser.parse_args()

    # Discover available videos
    segmented_dir = PROJECT_ROOT / "data" / "segmented"
    available = discover_available_videos(segmented_dir)

    # Handle --list mode
    if args.list:
        print("\nAvailable videos in data/segmented:\n")
        for vid, info in sorted(available.items()):
            n_vals = ', '.join(f"N={n}" for n in sorted(info['N_values']))
            print(f"  {vid:4d} ({info['type']:6s}) - {n_vals}")
        sys.exit(0)

    # Handle --all mode
    if args.all:
        tasks = []
        for vid, info in available.items():
            for n in info['N_values']:
                tasks.append((vid, n, info['type']))
        return tasks

    # Handle single video mode
    if args.video_id is None or args.N is None:
        parser.error("video_id and N are required (or use --list or --all)")

    # Validate video_id exists
    if args.video_id not in available:
        valid_ids = ', '.join(str(v) for v in sorted(available.keys()))
        parser.error(
            f"Video ID {args.video_id} not found.\n"
            f"Available IDs: {valid_ids}\n"
            f"Use --list to see details."
        )

    # Validate N value exists for this video
    if args.N not in available[args.video_id]['N_values']:
        valid_n = ', '.join(f"N={n}" for n in sorted(available[args.video_id]['N_values']))
        parser.error(
            f"N={args.N} not available for video {args.video_id}.\n"
            f"Available: {valid_n}"
        )

    return [(args.video_id, args.N, available[args.video_id]['type'])]


def save_to_excel(total_sums, total_nonzero, flex, ext, meta):
    intensities_dir = PROJECT_ROOT / "data" / "intensities_total"
    intensities_dir.mkdir(exist_ok=True)  # Ensure directory exists
    output_file = intensities_dir / f"{meta.video_id}N{meta.n_segments}intensities.xlsx"

    # Create region ranges DataFrame from metadata
    region_data = [{"Region": name, "Start": reg.start, "End": reg.end} for name, reg in meta.regions.items()]
    df_regions = pd.DataFrame(region_data)

    with pd.ExcelWriter(output_file) as writer:
        total_sums.to_excel(writer, sheet_name="Segment Intensities", index=True)
        total_nonzero.to_excel(writer, sheet_name="Number of Mask Pixels", index=True)
        flex.to_excel(writer, sheet_name="Flexion Frames", index=True)
        ext.to_excel(writer, sheet_name="Extension Frames", index=True)
        df_regions.to_excel(writer, sheet_name="Anatomical Regions", index=False)

    print(f"✅ Analysis results saved to {output_file}")


def draw_segment_boundaries(video, radial_regions, meta):
    """
    Draws the segment boundaries on the video, for verification before showing final results.

    Shows only JC-OT and OT-SB boundaries as white overlay lines on grayscale video.
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

    # Display the result
    views.show_frames(overlay, f"Region boundaries {meta.video_id}N{meta.n_segments}")

    return overlay

#==============================================================================
#                                MAIN FUNCTION
#==============================================================================

def main(tasks: List[Tuple[int, int, str]]):
    """
    Main function to process knee video intensity data.

    Args:
        tasks: List of (video_id, N, condition) tuples to process
    """
    batch_mode = len(tasks) > 1

    for video_id, N, condition in tasks:
        print(f"---------- Processing {video_id=}, {condition=}, {N=} ----------")

        # Get metadata
        meta = get_knee_meta(condition, video_id, N)

        # Select data
        segmented_dir = PROJECT_ROOT / "data" / "segmented"
        masks = load_masks(segmented_dir / f"{condition}_{video_id:04d}_radial_N{N}.npy")
        video = load_video(segmented_dir / f"{condition}_{video_id:04d}_video_N{N}.npy")

        print(f"Loaded data: masks {masks.shape}, video {video.shape}")
        views.show_frames([masks * (255 // np.max(masks)), video], f"Validate data {video_id}N{N}")
        # breakpoint()

        # Verify segment ranges
        draw_segment_boundaries(video, masks, meta)

        # Compute within-segment total intensities and number of pixels in each segment
        total_sums, total_nonzero = compute_sums_nonzeros(masks, video)

        # Cast to dataframe for better storage
        total_sums = pd.DataFrame(total_sums.T)
        total_nonzero = pd.DataFrame(total_nonzero.T)

        # Build flexion/extension DataFrames from metadata (1-based for Excel)
        flex_rows = [(c.flex.s + 1, c.flex.e + 1) for c in meta.cycles]
        ext_rows = [(c.ext.s + 1, c.ext.e + 1) for c in meta.cycles]
        flex = pd.DataFrame(flex_rows, columns=["Start", "End"])
        ext = pd.DataFrame(ext_rows, columns=["Start", "End"])

        total_sums.index = total_sums.index + 1; total_nonzero.index = total_nonzero.index + 1  # 1-indexing
        flex.index = flex.index + 1; ext.index = ext.index + 1

        total_sums.columns = total_sums.columns + 1; total_nonzero.columns = total_nonzero.columns + 1  # Formatting
        # flex.columns and ext.columns already set

        print("-----------------------------------------------------------")
        print(total_sums.head())
        print(flex)
        print(ext)

        if not batch_mode:
            if input("Save to file? (y/n)\n".lower()) == 'y':
                save_to_excel(total_sums, total_nonzero, flex, ext, meta)
            else:
                print("File not saved.")
        else:
            save_to_excel(total_sums, total_nonzero, flex, ext, meta)

    return

#==============================================================================
#                                ENTRY POINT
#==============================================================================

if __name__ == "__main__":
    tasks = parse_arguments()
    main(tasks)
