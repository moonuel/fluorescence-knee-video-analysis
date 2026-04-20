from utils import io, views
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import argparse
import sys
import re

from config.knee_metadata import get_knee_meta_by_condition

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


def parse_video_selector(text: str) -> Tuple[int, Optional[int]]:
    """Parse selector into (video_id, N).

    Accepted:
      - "1339" -> (1339, None) raw/unsegmented
      - "1339N64" / "1339n64" -> (1339, 64) segmented
    """
    s = text.strip()
    m = re.fullmatch(r"(\d+)", s)
    if m:
        return int(m.group(1)), None

    m = re.fullmatch(r"(\d+)[Nn](\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    raise ValueError(
        "Invalid selector. Use <video_id> for raw or <video_id>N<N> for segmented (e.g. 1339N64)."
    )


def discover_raw_videos(raw_dir: Path) -> Dict[int, Dict]:
    """Discover raw/unsegmented previewable files in data/raw.

    Returns {video_id: {'files': list[Path]}}.

    Only supports .npy and .avi (non-recursive scan).
    Video id is extracted from trailing "_<digits>" in the stem.
    """
    videos: Dict[int, Dict] = {}
    if not raw_dir.exists():
        return videos

    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".npy", ".avi"}:
            continue

        m = re.search(r"_(\d+)$", p.stem)
        if not m:
            continue
        video_id = int(m.group(1))
        videos.setdefault(video_id, {"files": []})["files"].append(p)

    for info in videos.values():
        info["files"].sort(key=lambda x: (0 if x.suffix.lower() == ".npy" else 1, x.name.lower()))
    return videos


def choose_preferred_raw_file(files: list[Path]) -> Path:
    if not files:
        raise FileNotFoundError("No raw files available")
    files_sorted = sorted(files, key=lambda x: (0 if x.suffix.lower() == ".npy" else 1, x.name.lower()))
    return files_sorted[0]


def parse_arguments():
    """
    Parse command line arguments with dynamic validation based on available files.
    """
    parser = argparse.ArgumentParser(
        description="Play knee videos (segmented or raw)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python play_video.py --list\n"
               "  python play_video.py --list-raw\n"
               "  python play_video.py 1339N64\n"
               "  python play_video.py 1339\n"
               "  python play_video.py 1339 64  # legacy"
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help="List all available video IDs and segment counts"
    )

    parser.add_argument(
        '--list-raw',
        action='store_true',
        help="List raw/unsegmented previewable videos in data/raw"
    )

    parser.add_argument(
        'selector',
        nargs='?',
        help="Video selector: <id> (raw) or <id>N<N> (segmented), e.g. 1339 or 1339N64"
    )

    parser.add_argument(
        'N',
        type=int,
        nargs='?',
        help="Legacy segmented mode: play_video.py <id> <N>"
    )

    args = parser.parse_args()

    # Discover available videos
    segmented_dir = PROJECT_ROOT / "data" / "segmented"
    available = discover_available_videos(segmented_dir)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_available = discover_raw_videos(raw_dir)

    # Handle --list mode
    if args.list:
        print("\nAvailable videos in data/segmented:\n")
        for vid, info in sorted(available.items()):
            n_vals = ', '.join(f"N={n}" for n in sorted(info['N_values']))
            print(f"  {vid:4d} ({info['type']:6s}) - {n_vals}")
        sys.exit(0)

    if args.list_raw:
        print("\nAvailable videos in data/raw (previewable: .npy, .avi):\n")
        for vid, info in sorted(raw_available.items()):
            files = ", ".join(p.name for p in info["files"])
            print(f"  {vid:4d} - {files}")
        sys.exit(0)

    # Validate required arguments
    if args.selector is None:
        parser.error("selector is required (or use --list/--list-raw)")

    # Backward-compat: play_video.py <id> <N>
    selector_text = args.selector
    if args.N is not None and re.fullmatch(r"\d+", selector_text.strip()):
        selector_text = f"{selector_text}N{args.N}"

    try:
        video_id, N = parse_video_selector(selector_text)
    except ValueError as e:
        parser.error(str(e))

    if N is not None:
        # Segmented validation
        if video_id not in available:
            valid_ids = ', '.join(str(v) for v in sorted(available.keys()))
            parser.error(
                f"Video ID {video_id} not found in data/segmented.\n"
                f"Available IDs: {valid_ids}\n"
                f"Use --list to see details."
            )
        if N not in available[video_id]['N_values']:
            valid_n = ', '.join(f"N={n}" for n in sorted(available[video_id]['N_values']))
            parser.error(
                f"N={N} not available for video {video_id}.\n"
                f"Available: {valid_n}"
            )
        return video_id, N, available[video_id]['type'], None

    # Raw validation
    if video_id not in raw_available:
        valid_ids = ', '.join(str(v) for v in sorted(raw_available.keys()))
        parser.error(
            f"Video ID {video_id} not found in data/raw.\n"
            f"Available IDs: {valid_ids}\n"
            f"Use --list-raw to see details."
        )
    raw_path = choose_preferred_raw_file(raw_available[video_id]["files"])
    return video_id, None, None, raw_path


def main(video_id: int, N: Optional[int], knee_type: Optional[str], raw_path: Optional[Path]):
    """
    Display the radial segmentation mask alongside the processed video.
    """
    if N is not None:
        segmented_dir = PROJECT_ROOT / "data" / "segmented"
        assert knee_type is not None

        radial_path = segmented_dir / f"{knee_type}_{video_id:04d}_radial_N{N}.npy"
        video_path = segmented_dir / f"{knee_type}_{video_id:04d}_video_N{N}.npy"
        femur_path = segmented_dir / f"{knee_type}_{video_id:04d}_femur_N{N}.npy"
        if not radial_path.exists():
            raise FileNotFoundError(f"Missing radial mask: {radial_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video: {video_path}")

        print(f"Loading radial mask: {radial_path}")
        radial_mask = io.load_nparray(radial_path)

        print(f"Loading video: {video_path}")
        video = io.load_nparray(video_path)

        max_label = np.max(radial_mask)
        if max_label > 0:
            scale = 255 // max_label
            radial_display = radial_mask * scale
        else:
            radial_display = radial_mask

        print(f"Displaying {knee_type} video {video_id} with {N} segments...")
        print("Radial mask (left) and processed video (right)")

        # Default: fall back to current behavior if metadata is missing/incomplete.
        boundary_overlay = video.copy()

        # Always draw the outer segmentation boundary (if possible)
        try:
            boundary_overlay = views.draw_outer_radial_mask_boundary(
                boundary_overlay,
                radial_mask,
                intensity=255,
                thickness=1,
                show_video=False,
            )
        except Exception as e:
            print(f"Warning: failed to draw outer segmentation boundary ({e})")

        # Femur line in this project = reference radial boundary between seg 1 and seg N.
        # Draw it across the whole video (not cycle-limited) to match existing previews.
        try:
            views.draw_boundary_line(
                boundary_overlay,
                radial_mask,
                seg_num=1,
                n_segments=N,
                intensity=255,
                thickness=1,
                show_video=False,
                inplace=True,
                dashed=False,
            )
        except Exception as e:
            print(f"Warning: failed to draw femur line (seg 1/N boundary) ({e})")

        try:
            meta = get_knee_meta_by_condition(knee_type, int(video_id), int(N))
            regions = meta.regions
            ot = regions.get("OT")
            sb = regions.get("SB")
            if ot is None or sb is None or not meta.cycles:
                raise KeyError("Incomplete region/cycle metadata")
            for cycle in meta.cycles:
                fr = cycle.full_cycle()
                sl = fr.to_slice()  # 0-based inclusive -> python slice

                # OT-JC boundary (between OT.s-1 and OT.s)
                views.draw_boundary_line(
                    boundary_overlay[sl],
                    radial_mask[sl],
                    seg_num=ot.s,
                    n_segments=N,
                    intensity=200,
                    thickness=1,
                    show_video=False,
                    inplace=True,
                    dashed=False,
                )

                # SB-OT boundary (between SB.s-1 and SB.s)
                views.draw_boundary_line(
                    boundary_overlay[sl],
                    radial_mask[sl],
                    seg_num=sb.s,
                    n_segments=N,
                    intensity=150,
                    thickness=1,
                    show_video=False,
                    inplace=True,
                    dashed=False,
                )
        except Exception as e:
            print(
                f"Warning: no usable knee metadata for {knee_type}_{video_id:04d}_N{N}; "
                f"skipping anatomical boundaries ({e})"
            )

        views.show_frames([radial_display, boundary_overlay])
        return

    assert raw_path is not None
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file: {raw_path}")

    print(f"Loading raw video: {raw_path}")
    if raw_path.suffix.lower() == ".npy":
        video = io.load_nparray(raw_path)
    elif raw_path.suffix.lower() == ".avi":
        video = io.load_avi(str(raw_path))
    else:
        raise ValueError(f"Unsupported raw extension: {raw_path.suffix}")

    print(f"Displaying raw video {video_id} from {raw_path.name}...")
    views.show_frames(video, title=f"raw_{video_id:04d}")


if __name__ == "__main__":
    video_id, N, knee_type, raw_path = parse_arguments()
    main(video_id, N, knee_type, raw_path)
