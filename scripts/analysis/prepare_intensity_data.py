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


def save_analysis_to_excel(total_sums: np.ndarray,
                           total_nonzero: np.ndarray,
                           metadata: "Metadata",
                           output_file: str | Path):
    """
    Save knee analysis results to an Excel file with three sheets:
      - Total sums (N x nframes)
      - Total nonzero (N x nframes)
      - Metadata (one row)
    """
    output_file = Path(output_file)

    # --- Convert Metadata dataclass into a dict row ---
    meta_dict = asdict(metadata)

    df_meta = pd.DataFrame([meta_dict])

    # --- Convert arrays to DataFrames ---
    N, nframes = total_sums.shape[0], total_sums.shape[1]

    # Use the true frame numbers from metadata
    frame_index = range(metadata.frame_start, metadata.frame_end + 1)

    df_sums = pd.DataFrame(
        total_sums.T,
        index=frame_index,
        columns=[f"Segment {i}" for i in range(1, N + 1)]
    )
    df_sums.index.name = "Frame"

    df_nonzero = pd.DataFrame(
        total_nonzero.T,
        index=frame_index,
        columns=[f"Segment {i}" for i in range(1, N + 1)]
    )
    df_nonzero.index.name = "Frame"

    # --- Write all three sheets ---
    with pd.ExcelWriter(output_file) as writer:
        df_sums.to_excel(writer, sheet_name="Sum of Pixel Intensities (0-255)")
        df_nonzero.to_excel(writer, sheet_name="Number of Non-zero Pixels (Size of Mask)")
        df_meta.to_excel(writer, sheet_name="Analysis Metadata", index=False)

    print(f"✅ Analysis results saved to {output_file.resolve()}")


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
               "  python prepare_intensity_data.py 1339 64"
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help="List all available video IDs and segment counts"
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

    # Validate required arguments
    if args.video_id is None or args.N is None:
        parser.error("video_id and N are required (or use --list)")

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

    return args.video_id, args.N, available[args.video_id]['type']

def save_to_excel(total_sums, total_nonzero, flex, ext, video_id, N):
    intensities_dir = PROJECT_ROOT / "data" / "intensities_total"
    intensities_dir.mkdir(exist_ok=True)  # Ensure directory exists
    output_file = intensities_dir / f"{video_id}N{N}intensities.xlsx"

    # Get region ranges for this video
    ranges = REGION_RANGES.get((video_id, N))
    if ranges is not None:
        # Create region ranges DataFrame
        region_data = []
        for region, (start, end) in ranges.items():
            region_data.append({"Region": region, "Start": start, "End": end})
        df_regions = pd.DataFrame(region_data)
    else:
        # If no ranges defined, create empty DataFrame with correct columns
        df_regions = pd.DataFrame(columns=["Region", "Start", "End"])

    with pd.ExcelWriter(output_file) as writer:
        total_sums.to_excel(writer, sheet_name="Segment Intensities", index=True)
        total_nonzero.to_excel(writer, sheet_name="Number of Mask Pixels", index=True)
        flex.to_excel(writer, sheet_name="Flexion Frames", index=True)
        ext.to_excel(writer, sheet_name="Extension Frames", index=True)
        df_regions.to_excel(writer, sheet_name="Anatomical Regions", index=False)

    print(f"✅ Analysis results saved to {output_file}")


def draw_segment_boundaries(video, radial_regions, video_id, nsegs):
    """
    Draws the segment boundaries on the video, for verification before showing final results.

    Shows only JC-OT and OT-SB boundaries as white overlay lines on grayscale video.
    """
    # Input validation
    assert video.shape == radial_regions.shape, f"Shape mismatch: video {video.shape} vs radial_regions {radial_regions.shape}"
    assert video.dtype == np.uint8, f"Video must be uint8, got {video.dtype}"
    assert radial_regions.dtype in [np.uint8, np.int32, np.int64], f"Radial regions must be integer type, got {radial_regions.dtype}"

    # Retrieve segment ranges from REGION_RANGES
    ranges = REGION_RANGES.get((video_id, nsegs))
    if ranges is None:
        raise KeyError(f"No REGION_RANGES entry for (video_id={video_id}, N={nsegs})")

    jc_start, jc_end = ranges["JC"]
    ot_start, ot_end = ranges["OT"]
    sb_start, sb_end = ranges["SB"]

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
    views.show_frames(overlay, f"Region boundaries {video_id}N{nsegs}")

    return overlay

#==============================================================================
#                                MAIN FUNCTION
#==============================================================================

def main(video_id:int, N:int, condition:str):
    """
    Main function to process knee video intensity data.

    Args:
        video_id: Video ID number (e.g., 1339)
        N: Number of radial segments (e.g., 64)
        knee_type: Type of knee ("normal" or "aging")
    """
    # Select data
    segmented_dir = PROJECT_ROOT / "data" / "segmented"
    masks = load_masks(segmented_dir / f"{condition}_{video_id:04d}_radial_N{N}.npy")
    video = load_video(segmented_dir / f"{condition}_{video_id:04d}_video_N{N}.npy")
    cycles = [c.split("-") for c in CYCLES[video_id].split()]

    print(f"---------- {video_id=}, {condition=}, {N=} ----------")
    views.show_frames([masks * (255 // np.max(masks)), video], "Validate data")
    # breakpoint()

    # Verify segment ranges
    draw_segment_boundaries(video, masks, video_id, N)

    # Compute within-segment total intensities and number of pixels in each segment
    total_sums, total_nonzero = compute_sums_nonzeros(masks, video)

    # Cast to dataframe for better storage
    total_sums = pd.DataFrame(total_sums.T)
    total_nonzero = pd.DataFrame(total_nonzero.T)
    flex = pd.DataFrame(cycles[::2])
    ext = pd.DataFrame(cycles[1::2])

    total_sums.index = total_sums.index + 1; total_nonzero.index = total_nonzero.index + 1 # 1-indexing
    flex.index = flex.index + 1; ext.index = ext.index + 1

    total_sums.columns = total_sums.columns + 1; total_nonzero.columns = total_nonzero.columns + 1 # Formatting
    flex.columns = ["Start", "End"]; ext.columns = ["Start", "End"]

    print("-----------------------------------------------------------")
    print(total_sums)
    print(flex)
    print(ext)

    if input("Save to file? (y/n)\n".lower()) == 'y':
        save_to_excel(total_sums, total_nonzero, flex, ext, video_id, N)
    else:
        print("File not saved.")

    return

#==============================================================================
#                                DATA STORAGE
#==============================================================================

# Store cycle ranges here
CYCLES = {
    1207: "242-254	264-280	281-293	299-312	318-335	337-352	353-372	373-389	391-411	412-431	434-451	453-467	472-486	488-505	614-632	633-651	652-671	672-690	693-708	709-727	731-748	751-767	768-786	787-804	807-822	824-841	844-862	863-877",
    1190: "66-89	92-109 421-452	470-492	503-532	533-569 737-767	770-793	794-822	823-860",
    1193: "1792-1801 1802-1812 1813-1822 1823-1833 1834-1843 1844-1852 1853-1863 1864-1872 1873-1881 1881-1889",
    308: "71-116 117-155 253-298 299-335 585-618 630-669 156-199 210-250",

    1339: "290-309	312-329	331-352	355-374	375-394	398-421	422-439	441-463	464-488	490-512	513-530	532-553	554-576	579-609",
    1342: "62-81	82-100	102-119	123-151	152-171	178-199",
    1357: "218-240	241-272	278-305	306-330 420-447	449-467	469-492	493-517 639-660	662-682	683-709	710-732	744-775	777-779	801-828	837-858	859-890	893-917	1067-1091	1092-1118	1136-1171	1173-1198	1199-1230	1232-1260	1261-1285	1286-1311	1313-1340	1342-1365	1368-1394	1395-1419",
    1358: "1360-1384	1385-1406	1407-1433	1434-1454	1461-1483	1484-1508	1509-1540	1541-1559	1618-1648	1649-1669	1672-1696	1697-1720	"#1721-1748"
}

# Store knee types here
TYPES = {
    1207: "normal",
    1193: "normal",
    1190: "normal",
    308: "normal",

    1339: "aging",
    1342: "aging",
    1357: "aging",
    1358: "aging",
}

assert CYCLES.keys() == TYPES.keys()

REGION_RANGES = {
    (1207, 64): {
        "JC": (1, 29),
        "OT": (30, 42),
        "SB": (43, 64),
    },
    # (308, 64): {
    #     "JC": (1, 29),
    #     "OT": (30, 42),
    #     "SB": (43, 64),
    # },
}

#==============================================================================
#                                ENTRY POINT
#==============================================================================

if __name__ == "__main__":
    video_id, N, condition = parse_arguments()
    main(video_id, N, condition)
