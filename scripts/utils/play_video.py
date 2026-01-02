from utils import io, views
from typing import Dict
from pathlib import Path
import numpy as np
import argparse
import sys

# Get project root directory for robust path handling
PROJECT_ROOT = Path(__file__).parent.parent.parent


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
        description="Play segmented knee videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python play_video.py --list\n"
               "  python play_video.py 1339 64"
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
        print("\n📋 Available videos in data/segmented:\n")
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


def main(video_id: int, N: int, knee_type: str):
    """
    Display the radial segmentation mask alongside the processed video.
    """
    segmented_dir = PROJECT_ROOT / "data" / "segmented"

    # Load radial mask and video
    radial_path = segmented_dir / f"{knee_type}_{video_id:04d}_radial_N{N}.npy"
    video_path = segmented_dir / f"{knee_type}_{video_id:04d}_video_N{N}.npy"

    print(f"Loading radial mask: {radial_path}")
    radial_mask = io.load_nparray(radial_path)

    print(f"Loading video: {video_path}")
    video = io.load_nparray(video_path)

    # Scale radial mask for visualization
    max_label = np.max(radial_mask)
    if max_label > 0:
        scale = 255 // max_label
        radial_display = radial_mask * scale
    else:
        radial_display = radial_mask

    print(f"Displaying {knee_type} video {video_id} with {N} segments...")
    print("Radial mask (left) and processed video (right)")
    views.show_frames([radial_display, video])


if __name__ == "__main__":
    video_id, N, knee_type = parse_arguments()
    breakpoint()
    main(video_id, N, knee_type)
